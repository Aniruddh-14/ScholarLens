"""
Agent package for the Agentic AI Research Assistant.

Exposes the compiled LangGraph graph and the ResearchState type.
"""

from src.agent.graph import build_graph
from src.agent.state import ResearchState

__all__ = ["build_graph", "ResearchState"]
