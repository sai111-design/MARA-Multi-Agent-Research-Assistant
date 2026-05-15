"""Shared state schema for the MARA multi-agent pipeline.

Defines the TypedDict / dataclass that all LangGraph nodes read from and write to,
ensuring a single source of truth for the data flowing through the graph.
"""

from typing import List, Optional
from typing_extensions import TypedDict


class ResearchState(TypedDict):
    query: str
    sub_questions: List[str]
    search_results: List[dict]
    critique: Optional[str]
    needs_more_research: bool
    final_report: Optional[str]
    research_loop_count: int
    current_step: str


def create_initial_state(query: str) -> ResearchState:
    return ResearchState(
        query=query,
        sub_questions=[],
        search_results=[],
        critique=None,
        needs_more_research=False,
        final_report=None,
        research_loop_count=0,
        current_step="idle",
    )
