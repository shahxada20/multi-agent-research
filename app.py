import re
import time
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from groq import RateLimitError

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · Multi-Agent Research System",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Load external stylesheet ─────────────────────────────────────────────────
_CSS_PATH = Path(__file__).parent / "styles.css"
_CSS = _CSS_PATH.read_text(encoding="utf-8")


# ── Session state defaults ───────────────────────────────────────────────────
_DEFAULTS = {
    "results": {},
    "running": False,
    "done": False,
    "topic_input": "",
    "error": None,
    "theme": "light",
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Helpers: theme + wrapper ─────────────────────────────────────────────────
def _toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


# ── Inject stylesheet + theme wrapper ───────────────────────────────────────
st.html(f"<style>{_CSS}</style>")

st.html(
    f'<div class="rm-root" data-theme="{st.session_state.theme}">'
    '<script>'
    '(function(){try{var s=localStorage.getItem("rm-theme");'
    'var r=document.querySelector(".rm-root");'
    'if(s&&r)r.setAttribute("data-theme",s);}catch(e){}})();'
    '</script>'
)


# ── Pipeline step contracts ──────────────────────────────────────────────────
STEPS = ["search", "reader", "writer", "critic"]


# ── Step runners (unchanged) ─────────────────────────────────────────────────
def _invoke_with_retry(fn, *, max_attempts: int = 4):
    last_err = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except RateLimitError as e:
            last_err = e
            wait_s = _parse_retry_after(str(e))
            if attempt == max_attempts - 1:
                break
            time.sleep(wait_s)
    raise last_err  # type: ignore[misc]


def _parse_retry_after(error_text: str) -> float:
    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)\b", error_text, re.IGNORECASE)
    if m:
        value, unit = float(m.group(1)), m.group(2).lower()
        seconds = (value / 1000.0) * 1.5 if unit == "ms" else value * 1.5
        return max(seconds, 0.5)
    return min(2 ** _parse_retry_after._attempt, 8)
_parse_retry_after._attempt = 0  # type: ignore[attr-defined]


def _bump_fallback_counter():
    _parse_retry_after._attempt += 1  # type: ignore[attr-defined]
def _reset_fallback_counter():
    _parse_retry_after._attempt = 0  # type: ignore[attr-defined]


def _run_search(topic: str) -> str:
    agent = build_search_agent()
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": f"Find recent, reliable and detailed information about: {topic}",
        }],
    })
    return result["messages"][-1].content


def _run_reader(topic: str, search_output: str) -> str:
    agent = build_reader_agent()
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": (
                f"Based on the following search results about '{topic}', "
                f"pick the single most relevant and authoritative URL to scrape for deeper content.\n\n"
                f"CRITICAL FILTERING RULES:\n"
                f"- DO NOT pick premium paywalled domains if possible (e.g., wsj.com, bloomberg.com, nytimes.com).\n"
                f"- DO NOT pick social hubs or media engines (e.g., reddit.com, youtube.com).\n"
                f"- Prioritize open global journalism portals (e.g., reuters.com, aljazeera.com, apnews.com).\n\n"
                f"Search Results Context:\n{search_output[:5000]}"
            ),
        }],
    })
    return result["messages"][-1].content


def _run_writer(topic: str, search_output: str, reader_output: str) -> str:
    research = (
        f"SEARCH RESULTS:\n{search_output[:1800]}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{reader_output[:1800]}"
    )
    return writer_chain.invoke({"topic": topic, "research": research})


def _run_critic(report: str) -> str:
    return critic_chain.invoke({"report": report})


STEP_RUNNERS = {
    "search": lambda topic, prev: _invoke_with_retry(lambda: _run_search(topic)),
    "reader": lambda topic, prev: _invoke_with_retry(lambda: _run_reader(topic, prev["search"])),
    "writer": lambda topic, prev: _invoke_with_retry(lambda: _run_writer(topic, prev["search"], prev["reader"])),
    "critic": lambda topic, prev: _invoke_with_retry(lambda: _run_critic(prev["writer"])),
}

STEP_META = {
    "search": ("01", "Search Agent",  "Gathers recent web information via Tavily."),
    "reader": ("02", "Reader Agent",  "Scrapes & extracts deep content from the top URL."),
    "writer": ("03", "Writer Chain",  "Drafts the full research report."),
    "critic": ("04", "Critic Chain",  "Reviews the report and assigns a quality score."),
}


# ── URL extraction for the search-log panel ─────────────────────────────────
_URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.IGNORECASE)


def extract_urls(text: str, limit: int = 6) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for url in _URL_RE.findall(text or ""):
        url = url.rstrip(".,;:!?")
        if url in seen:
            continue
        seen.add(url)
        try:
            p = urlparse(url)
            host = p.netloc.replace("www.", "")
        except Exception:
            host, path = url, ""
        else:
            path = p.path if p.path not in ("", "/") else ""
        out.append({"host": host, "path": path, "full": url})
        if len(out) >= limit:
            break
    return out


# ── Critic score parser ─────────────────────────────────────────────────────
_SCORE_RE = re.compile(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)


def parse_score(text: str) -> str | None:
    if not text:
        return None
    m = _SCORE_RE.search(text)
    return m.group(1) if m else None


def _friendly_rate_limit_message(err: Exception) -> str:
    text = str(err)
    hint = ""
    m = re.search(r"try again in\s+([\d.]+)\s*(ms|s)", text, re.IGNORECASE)
    if m:
        value, unit = float(m.group(1)), m.group(2).lower()
        secs = value / 1000.0 if unit == "ms" else value
        hint = f" Try again in about {secs:.0f} seconds."
    return (
        "The Groq API hit its per-minute token rate limit just before the "
        "request could complete. Retries didn't clear it in time." + hint +
        " Either wait a minute and run again, or upgrade to the Groq Dev Tier "
        "for a higher TPM window."
    )


# ── Step state ───────────────────────────────────────────────────────────────
def _step_state(step: str) -> str:
    """Return 'waiting' | 'running' | 'done' for a given step."""
    results = st.session_state.results
    if step in results:
        return "done"
    if st.session_state.running and step not in results and all(
        s in results for s in STEPS[: STEPS.index(step)]
    ):
        return "running"
    return "waiting"


# ── Renderers ────────────────────────────────────────────────────────────────
def _status_label(state: str) -> str:
    return {
        "waiting": "Waiting",
        "running": "● Running",
        "done":    "Done",
    }[state]


def _render_stepper() -> None:
    """Vertical 4-agent stepper. Each node = circle + body (num, title, desc, status).
    Connectors are short vertical lines between adjacent circles, with the same
    waiting / running / done states."""
    rows = []
    for i, step in enumerate(STEPS):
        num, title, desc = STEP_META[step]
        state = _step_state(step)
        # Inner circle content: number while waiting/running, checkmark when done.
        # We use two child spans so CSS can swap which is visible per state.
        rows.append(f"""
        <div class="stepper-node" data-state="{state}">
            <div class="stepper-node-circle">
                <span class="stepper-node-circle-text">{num}</span>
                <span class="stepper-node-circle-check">✓</span>
            </div>
            <div class="stepper-node-body">
                <div class="stepper-node-title">
                    <span class="stepper-node-num">{num}</span>
                    <span>{title}</span>
                </div>
                <div class="stepper-node-desc">{desc}</div>
                <div class="stepper-node-status">{_status_label(state)}</div>
            </div>
        </div>""")
        if i < len(STEPS) - 1:
            if state == "done":
                connector_state = "done"
            elif state == "running":
                connector_state = "running"
            else:
                connector_state = "waiting"
            rows.append(f'<div class="stepper-connector" data-state="{connector_state}"></div>')

    st.html(
        '<div class="stepper">'
        '<div class="stepper-track">' + "".join(rows) + '</div>'
        '</div>'
    )


def _render_search_log() -> None:
    """Left-column Search Agent Log. Always rendered so layout doesn't jump."""
    results = st.session_state.results
    has_search = "search" in results

    if st.session_state.running and not has_search:
        state = "running"
        body = (
            '<div class="search-log-spinner">'
            '<span class="search-log-spinner-dot"></span>'
            'Querying the web for sources…'
            '</div>'
        )
        state_label = "Querying"
    elif has_search:
        state = "ready"
        urls = extract_urls(results["search"])
        if not urls:
            body = (
                '<div class="search-log-empty">'
                'Search complete, but no source URLs surfaced.'
                '</div>'
            )
        else:
            items = "".join(
                f'<li class="search-log-item" style="--i:{i}">'
                f'<span class="search-log-arrow">↗</span>'
                f'<span class="search-log-host">{u["host"]}</span>'
                f'<span class="search-log-path">/{u["path"].lstrip("/")}</span>'
                f'</li>'
                for i, u in enumerate(urls)
            )
            body = f'<ul class="search-log-list">{items}</ul>'
        state_label = "Ready"
    else:
        state = "empty"
        body = (
            '<div class="search-log-empty">'
            'Run the pipeline to see search queries and sources surface here.'
            '</div>'
        )
        state_label = "Idle"

    st.html(
        f'<div class="search-log" data-state="{state}">'
        f'<div class="search-log-header">'
        f'<span class="search-log-label">Search Agent Log</span>'
        f'<span class="search-log-state">{state_label}</span>'
        f'</div>{body}</div>'
    )


# ── Topbar: brand chip (left) + theme toggle (extreme right) ────────────────
# A wide left column plus a narrow right column pushes the toggle button to
# the right edge of the page. The button itself is a regular st.button so the
# click handler triggers st.rerun() and the wrapper data-theme flips.
spacer, toggle_col = st.columns([10, 1])
with spacer:
    st.html('<div class="topbar"><div class="topbar-brand">RM · ResearchMind</div></div>')
with toggle_col:
    toggle_label = "☾ Dark" if st.session_state.theme == "light" else "☀ Light"
    if st.button(toggle_label, key="theme_toggle_btn", help="Switch theme"):
        _toggle_theme()
        st.rerun()


# ── Hero ─────────────────────────────────────────────────────────────────────
st.html(
    '<div class="hero">'
    '<div class="hero-badge">'
    '<span class="hero-badge-dot"></span>'
    '<span>Multi-Agent Research System</span>'
    '</div>'
    '<h1 class="hero-title">Research<span class="hero-title-accent">Mind</span></h1>'
    '<p class="hero-description">'
    'Four specialized AI agents collaborating in sequence from live web search and scraping the resources to'
    'iterative drafting and critique, to synthesize any topic into a highly polished research brief.'
    '</p>'
    '</div>'
)

 

# ── Two-column layout: input + log left, stepper right ──────────────────────
col_input, col_pipeline = st.columns([1, 1.05], gap="medium")

with col_input:
    st.html('<div class="input-card">')
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. The future of Thermal energy",
        key="topic_input",
        label_visibility="visible",
    )
    run_clicked = st.button(
        "▶  Run Pipeline",
        use_container_width=True,
        disabled=st.session_state.running,
    )
    st.html('</div>')
    _render_search_log()

with col_pipeline:
    st.html('<div class="section-heading">Live Pipeline</div>')
    _render_stepper()


# ── Pipeline runner (one rerun per step) ────────────────────────────────────
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a research topic...")
    else:
        st.session_state.results = {}
        st.session_state.running = True
        st.session_state.done = False
        st.session_state.error = None
        st.rerun()

if st.session_state.running and not st.session_state.done:
    results = dict(st.session_state.results)
    topic_val = st.session_state.topic_input

    try:
        for step in STEPS:
            if step in results:
                continue
            with st.spinner(f"{STEP_META[step][1]} is working…"):
                results[step] = STEP_RUNNERS[step](topic_val, results)
            st.session_state.results = results
            st.rerun()
    except RateLimitError as e:
        st.session_state.results = results
        st.session_state.running = False
        st.session_state.done = True
        st.session_state.error = _friendly_rate_limit_message(e)
        st.rerun()
    except Exception as e:
        st.session_state.results = results
        st.session_state.running = False
        st.session_state.done = True
        st.session_state.error = f"{type(e).__name__}: {e}"
        st.rerun()

    st.session_state.error = None
    st.session_state.running = False
    st.session_state.done = True
    st.rerun()


# ── Results: writer panel + critic panel ────────────────────────────────────
results = st.session_state.results
if results:
    st.html('<hr class="divider">')
    st.html('<div class="section-heading">Results</div>')

    topic_text = st.session_state.topic_input.strip() or "Research Report"
    score = parse_score(results.get("critic", ""))

    # Report hero (topic title + score meta)
    if "writer" in results:
        score_html = (
            f'<div class="score-badge"><span class="num">{score}</span>'
            f'<span class="label">/ 10</span></div>'
            if score
            else '<span class="awaiting-pill">Awaiting review</span>'
        )
        st.html(
            f'<div class="report-hero">'
            f'<div class="report-eyebrow">Final Research Report</div>'
            f'<h2 class="report-topic">{topic_text}</h2>'
            f'<div class="report-meta">{score_html}</div>'
            f'</div>'
        )

    # Split writer / critic panels
    if "writer" in results or "critic" in results:
        cols = st.columns(2, gap="medium")

        with cols[0]:
            if "writer" in results:
                st.html(
                    '<div class="panel panel-writer">'
                    '<div class="panel-rail"></div>'
                    '<div class="panel-header">'
                    '<span class="panel-eyebrow">Draft</span>'
                    '<h3 class="panel-title">Writer’s Report</h3>'
                    '<div class="panel-meta">'
                )
                slug = re.sub(r"[^a-z0-9]+", "_", topic_text.lower()).strip("_") or "report"
                st.download_button(
                    label="⬇  Download .md",
                    data=results["writer"],
                    file_name=f"research_{slug}_{int(time.time())}.md",
                    mime="text/markdown",
                    key="download_writer",
                )
                st.html('</div></div><div class="panel-body">')
                st.markdown(results["writer"])
                st.html('</div></div>')

        with cols[1]:
            if "critic" in results:
                st.html(
                    '<div class="panel panel-critic">'
                    '<div class="panel-rail"></div>'
                    '<div class="panel-header">'
                    '<span class="panel-eyebrow critic-eyebrow">Critique</span>'
                    '<h3 class="panel-title">Reviewer Feedback</h3>'
                    '<div class="panel-meta">'
                    '<span class="critic-stamp">Reviewed</span>'
                    '</div>'
                    '</div>'
                    '<div class="panel-body">'
                )
                st.markdown(results["critic"])
                st.html('</div></div>')

    if st.session_state.error:
        st.html(
            f'<div class="awaiting-pill" '
            f'style="margin-top:1rem; color: var(--danger); border-color: var(--danger);">'
            f'⚠ {st.session_state.error}</div>'
        )


# ── Footer + close wrapper ──────────────────────────────────────────────────
st.html(
    '<div class="notice">ResearchMind · multi-agent pipeline · Powered by Groq &amp; Tavily</div>'
)
st.html('</div>')
