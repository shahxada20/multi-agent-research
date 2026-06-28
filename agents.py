from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# ── Model setup ──────────────────────────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.5,
    timeout=10,
    max_retries=3,
)


# ── 1st agent — Search ───────────────────────────────────────────────────────
# System prompt explicitly forbids inventing tool names. Without this, smaller
# models will hallucinate tools like `brave_search` that weren't registered
# with the agent, which surfaces as a 400 "tool_use_failed" error from Groq.
SEARCH_SYSTEM_PROMPT = """You are the Search Agent in a multi-agent research pipeline.

Your ONLY job is to call the `web_search` tool to gather recent, reliable, and
detailed information about the user's topic, then return a concise summary that
lists every useful URL you found.

Strict rules:
- You have exactly one tool available: `web_search`. Never invent or call any
  other tool name (e.g. `brave_search`, `google_search`, `serp`, etc.). If you
  think another tool would be needed, do NOT call it — just report the
  limitation in your final answer instead.
- Call `web_search` at most a few times. Stop as soon as you have a diverse
  set of high-quality sources.
- Prefer reputable news and analysis outlets. Skip reddit.com, youtube.com,
  wikipedia.org, and obvious navigation/boilerplate pages.
- End your final answer with a 'Sources:' section listing every URL you found,
  one per line.
"""


def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=SEARCH_SYSTEM_PROMPT,
    )


# ── 2nd agent — Reader ───────────────────────────────────────────────────────
READER_SYSTEM_PROMPT = """You are the Reader Agent in a multi-agent research pipeline.

Your ONLY job is to pick the single most relevant URL from the search results
you are given and call the `scrape_url` tool on it to extract deeper content.
Then return a clean, focused excerpt (key facts, names, dates, quotes) that the
writer can use.

Strict rules:
- You have exactly one tool available: `scrape_url`. Never invent or call any
  other tool name (e.g. `brave_search`, `web_search`, `google_search`, etc.).
  If you are tempted to search again, do NOT — just scrape the best URL from
  the search results provided.
- Pick at most ONE URL to scrape. Do not call `scrape_url` more than twice
  (e.g. a fallback if the first URL fails).
- Strip navigation, cookie banners, "skip to main content", and other
  boilerplate from the scraped text in your summary.
- Return plain prose — no JSON, no tool-call syntax.
"""


def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=READER_SYSTEM_PROMPT,
    )


# ── Writer chain ─────────────────────────────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# ── Critic chain ─────────────────────────────────────────────────────────────
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
