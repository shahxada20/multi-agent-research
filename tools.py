import os
import requests
from rich import print
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.tools import tool

load_dotenv()

tavily_key = os.getenv("TAVILY_API_KEY", "")

# Only initialize if the key is present to prevent hard startup crashes
if tavily_key:
    tavily = TavilyClient(api_key=tavily_key)
else:
    tavily = None

@tool
def web_search(query: str) -> str:
    """Find recent, reliable and detailed information about a topic."""
    if not tavily:
        return "Error: TAVILY_API_KEY is missing from the server environment variables. Please check your Space secrets."
    try: 
        response = tavily.search(query=query, num_results=5)
        result = response.get("results", [])

        list_of_results = []    
        for r in result:
            title = r.get("title", "No Title")
            url = r.get("url", "")
            snippet = r.get("content", "") or "No description available."

            if "reddit.com" in url or "youtube.com" in url or "wikipedia.org" in url:
                continue

            list_of_results.append(f"Title: {title}\nURL: {url}\nExtracted Content: {snippet}\n---\n")

        return "\n".join(list_of_results) if list_of_results else "No results found."

    except Exception as e:
        return f"Error performing web search: {str(e)}"


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)[:3000]
    except Exception as e:
        return f"Could not scrape URL: {str(e)}"

# test_url = "https://www.reuters.com/world/iran"
# test_url = "https://www.cbsnews.com/live-updates/iran-us-war-talks-suspended-trump-mou-israel-lebanon-hezbollah-fighting"
# test_url = "https://www.bbc.com/news/topics/cx2jyv8j8gwt"
# test_url = "https://www.csis.org/programs/latest-analysis-war-iran"
# test_url = "https://www.wsj.com/topics/place/iran"


# print(f"Scraping content from: {test_url} ...\n")
# print(scrape_url.invoke({"url": test_url}))
