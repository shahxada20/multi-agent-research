import sys
import io

# Force UTF-8 stdout so print() can handle non-ASCII chars from Tavily/Groq
# (e.g.   narrow no-break space) on Windows cp1252 consoles.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from agents import build_reader_agent , build_search_agent , writer_chain , critic_chain

def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n"+"="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    topic = "Iran Strait of Hormuz 2026 updates"

    search_result = search_agent.invoke({
    "messages": [{
            "role": "user",
            "content": f"Find recent, reliable and detailed information about: {topic}"
        }]
    })

    state["search_results"] = search_result['messages'][-1].content

    print("\n search result ",state['search_results'])

    #step 2 - reader agent 
    print("\n"+"="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Based on the following search results about '{topic}', "
                    f"pick the single most relevant and authoritative URL to scrape for deeper content.\n\n"
                    f"CRITICAL FILTERING RULES:\n"
                    f"- DO NOT pick premium paywalled domains if possible (e.g., wsj.com, bloomberg.com, nytimes.com).\n"
                    f"- DO NOT pick social hubs or media engines (e.g., reddit.com, youtube.com).\n"
                    f"- Prioritize open global journalism portals (e.g., reuters.com, aljazeera.com, apnews.com).\n\n"
                    f"Search Results Context:\n{state['search_results'][:5000]}"
                )
            }
        ]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nscraped content: \n", state['scraped_content'])

    #step 3 - writer chain 

    print("\n"+"="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    # Trim inputs to fit within the Groq TPM window (6K for llama-3.1-8b-instant).
    # The writer only needs the gist — keep things tight so we stay under the limit.
    search_trimmed = state['search_results'][:1800]
    scraped_trimmed = state['scraped_content'][:1800]

    research_combined = (
        f"SEARCH RESULTS:\n{search_trimmed}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{scraped_trimmed}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    #critic report 

    print("\n"+"="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state



if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
