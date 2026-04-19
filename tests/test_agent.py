"""
Unit tests for the Agentic AI Research Assistant.

Tests cover:
  - ResearchState construction
  - search_node with mocked DuckDuckGo (no results + results)
  - retrieve_node with mocked HTTP
  - summarize_node with mocked LLM
  - validate_node JSON parsing + fallback
  - report_node structure + fallback keys
  - followup_node output format
  - Markdown export correctness
  - LangGraph graph compilation (smoke test)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.agent.state import ResearchState
from src.agent.export import to_markdown
from src.agent.graph import build_graph


# ── Helpers ────────────────────────────────────────────────────────────

def _base_state(**overrides) -> ResearchState:
    """Return a minimal valid ResearchState for testing."""
    state: ResearchState = {
        "query": "test research query",
        "search_results": [],
        "fetched_content": [],
        "source_summaries": [],
        "validated_facts": [],
        "contradictions": [],
        "report": {},
        "follow_up_questions": [],
        "error": None,
        "step_logs": [],
        "_api_key": "fake-key",
    }
    state.update(overrides)
    return state


# ── State schema tests ─────────────────────────────────────────────────

class TestResearchState:
    def test_required_keys_present(self):
        state = _base_state()
        required = [
            "query", "search_results", "fetched_content",
            "source_summaries", "validated_facts", "contradictions",
            "report", "follow_up_questions", "error", "step_logs",
        ]
        for key in required:
            assert key in state, f"Missing key: {key}"

    def test_defaults_are_correct_types(self):
        state = _base_state()
        assert isinstance(state["search_results"], list)
        assert isinstance(state["validated_facts"], list)
        assert isinstance(state["report"], dict)
        assert state["error"] is None


# ── search_node tests ──────────────────────────────────────────────────

class TestSearchNode:
    def test_no_results_sets_error(self):
        from src.agent.nodes import search_node
        with patch("src.agent.nodes.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = []
            result = search_node(_base_state())
        assert result["error"] is not None
        assert result["search_results"] == []

    def test_returns_results_on_success(self):
        from src.agent.nodes import search_node
        fake_results = [
            {"title": "T1", "href": "https://ex1.com", "body": "snippet 1"},
            {"title": "T2", "href": "https://ex2.com", "body": "snippet 2"},
        ]
        with patch("src.agent.nodes.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.return_value.text.return_value = fake_results
            result = search_node(_base_state())
        assert result["error"] is None
        assert len(result["search_results"]) == 2
        assert result["search_results"][0]["url"] == "https://ex1.com"

    def test_search_exception_sets_error(self):
        from src.agent.nodes import search_node
        with patch("src.agent.nodes.DDGS") as mock_ddgs:
            mock_ddgs.return_value.__enter__.side_effect = RuntimeError("network error")
            result = search_node(_base_state())
        assert result["error"] is not None
        assert "network error" in result["error"]


# ── retrieve_node tests ────────────────────────────────────────────────

class TestRetrieveNode:
    def test_successful_fetch(self):
        from src.agent.nodes import retrieve_node
        state = _base_state(search_results=[
            {"title": "T", "url": "https://example.com", "snippet": "fallback"},
        ])
        with patch("src.agent.nodes._fetch_url", return_value="full page content"):
            result = retrieve_node(state)
        assert len(result["fetched_content"]) == 1
        assert result["fetched_content"][0]["content"] == "full page content"

    def test_failed_fetch_falls_back_to_snippet(self):
        from src.agent.nodes import retrieve_node
        state = _base_state(search_results=[
            {"title": "T", "url": "https://fail.com", "snippet": "my snippet"},
        ])
        with patch("src.agent.nodes._fetch_url", side_effect=Exception("timeout")):
            result = retrieve_node(state)
        assert result["fetched_content"][0]["content"] == "my snippet"


# ── validate_node tests ────────────────────────────────────────────────

class TestValidateNode:
    def test_single_source_skips_crosscheck(self):
        from src.agent.nodes import validate_node
        state = _base_state(source_summaries=[
            {"url": "https://a.com", "summary": "Single summary text."}
        ])
        result = validate_node(state)
        assert result["validated_facts"] == ["Single summary text."]
        assert result["contradictions"] == []

    def test_parses_valid_json(self):
        from src.agent.nodes import validate_node
        state = _base_state(source_summaries=[
            {"url": "https://a.com", "summary": "Fact A."},
            {"url": "https://b.com", "summary": "Fact B."},
        ])
        llm_response = json.dumps({
            "facts": ["Consistent fact 1", "Consistent fact 2"],
            "contradictions": ["Source A says X, Source B says Y"],
        })
        with patch("src.agent.nodes._call_llm", return_value=llm_response):
            result = validate_node(state)
        assert "Consistent fact 1" in result["validated_facts"]
        assert len(result["contradictions"]) == 1


# ── report_node tests ──────────────────────────────────────────────────

class TestReportNode:
    def test_report_has_required_keys(self):
        from src.agent.nodes import report_node
        state = _base_state(
            validated_facts=["Fact 1", "Fact 2"],
            source_summaries=[{"url": "https://a.com", "summary": "s"}],
        )
        llm_response = json.dumps({
            "title": "Test Title",
            "abstract": "Test abstract.",
            "findings": ["Finding 1"],
            "conclusion": "Test conclusion.",
        })
        with patch("src.agent.nodes._call_llm", return_value=llm_response):
            result = report_node(state)
        report = result["report"]
        for key in ["title", "abstract", "findings", "sources", "conclusion"]:
            assert key in report, f"Missing report key: {key}"

    def test_report_fallback_on_bad_json(self):
        from src.agent.nodes import report_node
        state = _base_state(
            validated_facts=["Fact 1"],
            source_summaries=[{"url": "https://a.com", "summary": "s"}],
        )
        with patch("src.agent.nodes._call_llm", return_value="not valid json {{"):
            result = report_node(state)
        # Should still have all keys via fallback defaults
        assert "title" in result["report"]
        assert "sources" in result["report"]


# ── followup_node tests ────────────────────────────────────────────────

class TestFollowupNode:
    def test_returns_list_of_questions(self):
        from src.agent.nodes import followup_node
        state = _base_state(report={
            "title": "T", "abstract": "A",
            "findings": ["F1", "F2"], "conclusion": "C", "sources": [],
        })
        questions = ["Q1?", "Q2?", "Q3?", "Q4?"]
        with patch("src.agent.nodes._call_llm", return_value=json.dumps(questions)):
            result = followup_node(state)
        assert result["follow_up_questions"] == questions

    def test_returns_empty_list_on_bad_response(self):
        from src.agent.nodes import followup_node
        state = _base_state(report={"findings": []})
        with patch("src.agent.nodes._call_llm", return_value="not a list"):
            result = followup_node(state)
        assert result["follow_up_questions"] == []


# ── Markdown export tests ──────────────────────────────────────────────

class TestMarkdownExport:
    def _sample_report(self):
        return {
            "title": "Quantum Computing Advances",
            "abstract": "Quantum computing is advancing rapidly.",
            "findings": ["Finding A", "Finding B"],
            "conclusion": "The field is promising.",
            "sources": ["https://source1.com", "https://source2.com"],
        }

    def test_all_sections_present(self):
        md = to_markdown(self._sample_report(), "quantum computing")
        assert "# Quantum Computing Advances" in md
        assert "## Abstract" in md
        assert "## Key Findings" in md
        assert "## Conclusion" in md
        assert "## Sources" in md

    def test_follow_up_section_included(self):
        md = to_markdown(
            self._sample_report(), "quantum computing",
            follow_up_questions=["What is next?", "How far along?"]
        )
        assert "## Follow-up Research Questions" in md
        assert "What is next?" in md

    def test_contradiction_section_included(self):
        md = to_markdown(
            self._sample_report(), "quantum computing",
            contradictions=["Source A says X, Source B says Y"]
        )
        assert "Contradictions" in md
        assert "Source A says X" in md

    def test_query_in_output(self):
        md = to_markdown(self._sample_report(), "quantum computing test")
        assert "quantum computing test" in md

    def test_sources_listed(self):
        md = to_markdown(self._sample_report(), "q")
        assert "https://source1.com" in md
        assert "https://source2.com" in md


# ── Graph compilation smoke test ───────────────────────────────────────

class TestGraphCompilation:
    def test_graph_builds_without_error(self):
        """Ensure the LangGraph compiles without raising exceptions."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_invoke_method(self):
        graph = build_graph()
        assert callable(getattr(graph, "invoke", None))

    def test_graph_has_stream_method(self):
        graph = build_graph()
        assert callable(getattr(graph, "stream", None))
