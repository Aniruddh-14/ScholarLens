"""
LangGraph node functions for the Agentic AI Research Assistant.

Each function receives the current ResearchState, performs its work,
and returns a dict of state keys to update. LangGraph merges these
updates automatically.

Node execution order (defined in graph.py):
    search_node → retrieve_node → summarize_node
    → validate_node → report_node → followup_node
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import ResearchState

# ── Constants ──────────────────────────────────────────────────────────
_MAX_RESULTS = 5          # DuckDuckGo results to fetch
_FETCH_TIMEOUT = 8        # seconds per HTTP request
_MAX_CONTENT_CHARS = 4000 # truncate fetched pages to this length
_MAX_RETRIES = 1          # LLM retry attempts on failure


# ── LLM factory ───────────────────────────────────────────────────────

def _get_llm(api_key: str, temperature: float = 0.3) -> ChatGroq:
    """Return a Groq ChatLLM instance."""
    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=temperature,
        max_tokens=2048,
    )


def _call_llm(llm: ChatGroq, system: str, human: str) -> str:
    """
    Call the LLM with retry logic.
    Returns the response text, or an empty string on failure.
    """
    messages = [SystemMessage(content=system), HumanMessage(content=human)]
    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            return response.content.strip()
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                return f"[LLM error: {exc}]"
            time.sleep(2 ** attempt)
    return ""


# ── Node 1: search_node ────────────────────────────────────────────────

def search_node(state: ResearchState) -> dict:
    """
    Search DuckDuckGo for the user's query.

    Returns up to _MAX_RESULTS results as [{title, url, snippet}].
    Sets state["error"] if no results are found.
    """
    query = state["query"]
    logs = list(state.get("step_logs", []))
    logs.append(f"🔍 Searching the web for: *{query}*")

    results: list[dict] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=_MAX_RESULTS):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as exc:
        logs.append(f"⚠️ Search error: {exc}")
        return {
            "search_results": [],
            "error": f"Web search failed: {exc}",
            "step_logs": logs,
        }

    if not results:
        logs.append("⚠️ No search results found for this query.")
        return {
            "search_results": [],
            "error": "No search results found. Try a different query.",
            "step_logs": logs,
        }

    logs.append(f"✅ Found {len(results)} sources.")
    return {
        "search_results": results,
        "error": None,
        "step_logs": logs,
    }


# ── Node 2: retrieve_node ──────────────────────────────────────────────

def _fetch_url(url: str) -> str:
    """Fetch a URL and return clean body text (truncated)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ScholarLensBot/1.0)"}
    resp = requests.get(url, timeout=_FETCH_TIMEOUT, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove script / style noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text[:_MAX_CONTENT_CHARS]


def retrieve_node(state: ResearchState) -> dict:
    """
    Fetch full page content for each search result URL.

    Failed URLs are skipped with a warning log; we continue with
    whatever we successfully retrieved.
    """
    logs = list(state.get("step_logs", []))
    logs.append("📥 Fetching full content from sources…")

    fetched: list[dict] = []
    for item in state["search_results"]:
        url = item["url"]
        try:
            content = _fetch_url(url)
            fetched.append({"url": url, "content": content})
        except Exception as exc:
            # Fall back to the snippet so we don't lose the source entirely
            logs.append(f"⚠️ Could not fetch {url}: {exc} — using snippet.")
            fetched.append({"url": url, "content": item.get("snippet", "")})

    logs.append(f"✅ Retrieved content from {len(fetched)} sources.")
    return {"fetched_content": fetched, "step_logs": logs}


# ── Node 3: summarize_node ─────────────────────────────────────────────

def summarize_node(state: ResearchState) -> dict:
    """
    Ask the LLM to summarize each fetched source in 3-5 sentences,
    focused on information relevant to the original query.

    Falls back to raw snippet on LLM failure.
    """
    logs = list(state.get("step_logs", []))
    logs.append("🧠 Summarizing each source with AI…")

    api_key = state.get("_api_key", "")
    llm = _get_llm(api_key)

    system = (
        "You are a precise research assistant. "
        "Summarize the given webpage content in 3-5 sentences, "
        "focusing only on information relevant to the research query. "
        "Be factual and concise. Do not include opinions or filler text."
    )

    summaries: list[dict] = []
    for item in state["fetched_content"]:
        human = (
            f"Research query: {state['query']}\n\n"
            f"Webpage content:\n{item['content']}"
        )
        summary = _call_llm(llm, system, human)
        if not summary or summary.startswith("[LLM error"):
            # Fallback to snippet from search results
            sr = next((r for r in state["search_results"] if r["url"] == item["url"]), {})
            summary = sr.get("snippet", "No summary available.")

        summaries.append({"url": item["url"], "summary": summary})

    logs.append(f"✅ Summarized {len(summaries)} sources.")
    return {"source_summaries": summaries, "step_logs": logs}


# ── Node 4: validate_node ──────────────────────────────────────────────

def validate_node(state: ResearchState) -> dict:
    """
    Cross-check all source summaries to extract:
      - consistent, well-supported facts
      - contradictions between sources

    Skips cross-checking if fewer than 2 sources are available.
    """
    logs = list(state.get("step_logs", []))
    logs.append("🔬 Validating and cross-checking sources…")

    summaries = state["source_summaries"]
    if len(summaries) < 2:
        logs.append("ℹ️ Only one source available — skipping cross-check.")
        single = summaries[0]["summary"] if summaries else ""
        return {
            "validated_facts": [single] if single else [],
            "contradictions": [],
            "step_logs": logs,
        }

    api_key = state.get("_api_key", "")
    llm = _get_llm(api_key, temperature=0.1)

    combined = "\n\n".join(
        f"Source {i+1} ({s['url']}):\n{s['summary']}"
        for i, s in enumerate(summaries)
    )
    system = (
        "You are a critical fact-checker. "
        "Given multiple source summaries about the same research topic, "
        "extract: (1) facts that are consistent across sources, "
        "and (2) any contradictions or conflicting claims. "
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"facts": ["fact 1", "fact 2", ...], "contradictions": ["contradiction 1", ...]}'
    )
    human = f"Research query: {state['query']}\n\nSource summaries:\n{combined}"

    raw = _call_llm(llm, system, human)

    # Parse JSON safely
    facts: list[str] = []
    contradictions: list[str] = []
    try:
        # Extract JSON block even if wrapped in markdown
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            facts = data.get("facts", [])
            contradictions = data.get("contradictions", [])
    except Exception:
        # If parsing fails, treat the whole response as a single fact
        facts = [raw] if raw and not raw.startswith("[LLM") else []

    logs.append(f"✅ Extracted {len(facts)} facts, {len(contradictions)} contradictions.")
    return {
        "validated_facts": facts,
        "contradictions": contradictions,
        "step_logs": logs,
    }


# ── Node 5: report_node ────────────────────────────────────────────────

def report_node(state: ResearchState) -> dict:
    """
    Generate the final structured research report from validated facts.

    Report structure:
        title       : concise title for the report
        abstract    : 2-3 sentence overview
        findings    : list of key findings
        sources     : list of source URLs used
        conclusion  : 2-3 sentence synthesis
    """
    logs = list(state.get("step_logs", []))
    logs.append("📝 Generating structured research report…")

    api_key = state.get("_api_key", "")
    llm = _get_llm(api_key, temperature=0.4)

    facts_text = "\n".join(f"- {f}" for f in state["validated_facts"])
    contradictions_text = (
        "\n".join(f"- {c}" for c in state["contradictions"])
        if state["contradictions"]
        else "None identified."
    )
    sources = [s["url"] for s in state["source_summaries"]]

    system = (
        "You are an expert research report writer. "
        "Write a structured academic-style research report based on the provided facts. "
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"title": "...", "abstract": "...", '
        '"findings": ["finding 1", "finding 2", ...], '
        '"conclusion": "..."}'
        "\nBe precise, factual, and avoid hallucination. "
        "Use only the provided facts."
    )
    human = (
        f"Research query: {state['query']}\n\n"
        f"Validated facts:\n{facts_text}\n\n"
        f"Contradictions noted:\n{contradictions_text}"
    )

    raw = _call_llm(llm, system, human)

    report: dict = {}
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            report = json.loads(match.group())
    except Exception:
        pass

    # Ensure all required keys exist (fallback values)
    report.setdefault("title", f"Research Report: {state['query']}")
    report.setdefault("abstract", " ".join(state["validated_facts"][:2]))
    report.setdefault("findings", state["validated_facts"])
    report.setdefault("conclusion", "Insufficient data to draw a firm conclusion.")
    report["sources"] = sources

    logs.append("✅ Report generated successfully.")
    return {"report": report, "step_logs": logs}


# ── Node 6: followup_node ──────────────────────────────────────────────

def followup_node(state: ResearchState) -> dict:
    """
    Generate 3-5 intelligent follow-up research questions based on the
    report findings. Returns an empty list on failure so the report
    is never blocked.
    """
    logs = list(state.get("step_logs", []))
    logs.append("💡 Generating follow-up research questions…")

    api_key = state.get("_api_key", "")
    llm = _get_llm(api_key, temperature=0.6)

    findings_text = "\n".join(
        f"- {f}" for f in state["report"].get("findings", [])[:5]
    )
    system = (
        "You are a curious research assistant. "
        "Based on the research findings below, generate exactly 4 insightful "
        "follow-up questions a researcher might want to explore next. "
        "Respond ONLY with a JSON array of strings: "
        '["question 1", "question 2", "question 3", "question 4"]'
    )
    human = (
        f"Original query: {state['query']}\n\n"
        f"Key findings:\n{findings_text}"
    )

    raw = _call_llm(llm, system, human)
    questions: list[str] = []
    try:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            questions = json.loads(match.group())
    except Exception:
        pass

    logs.append(f"✅ Generated {len(questions)} follow-up questions.")
    return {"follow_up_questions": questions, "step_logs": logs}
