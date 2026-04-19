"""
LangGraph graph builder for the Agentic AI Research Assistant.

Assembles the six node functions into a compiled StateGraph with:
  - A conditional edge after search_node to bail gracefully on failure
  - Linear edges for the remaining pipeline steps
  - Explicit START and END markers
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from src.agent.state import ResearchState
from src.agent.nodes import (
    search_node,
    retrieve_node,
    summarize_node,
    validate_node,
    report_node,
    followup_node,
)


def _should_continue(state: ResearchState) -> str:
    """
    Conditional edge after search_node.

    If an error occurred (e.g. no results), route directly to END
    so the UI can surface the error without running downstream nodes.
    """
    return "end" if state.get("error") else "retrieve"


def build_graph():
    """
    Build and compile the research agent StateGraph.

    Graph topology
    --------------
    START
      └─► search_node
             ├─(error)─► END
             └─(ok)────► retrieve_node
                             └─► summarize_node
                                     └─► validate_node
                                             └─► report_node
                                                     └─► followup_node
                                                             └─► END

    Returns
    -------
    Compiled LangGraph runnable (supports .invoke() and .stream()).
    """
    graph = StateGraph(ResearchState)

    # ── Register nodes ──────────────────────────────────────────────
    graph.add_node("search",    search_node)
    graph.add_node("retrieve",  retrieve_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("validate",  validate_node)
    graph.add_node("report",    report_node)
    graph.add_node("followup",  followup_node)

    # ── Entry point ─────────────────────────────────────────────────
    graph.add_edge(START, "search")

    # ── Conditional edge: bail on search failure ─────────────────────
    graph.add_conditional_edges(
        "search",
        _should_continue,
        {"retrieve": "retrieve", "end": END},
    )

    # ── Linear pipeline ──────────────────────────────────────────────
    graph.add_edge("retrieve",  "summarize")
    graph.add_edge("summarize", "validate")
    graph.add_edge("validate",  "report")
    graph.add_edge("report",    "followup")
    graph.add_edge("followup",  END)

    return graph.compile()
