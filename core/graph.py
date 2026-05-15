import logging
import time
from typing import Generator
from langgraph.graph import StateGraph, END
from core.state import ResearchState, create_initial_state
from core.config import MAX_RESEARCH_LOOPS
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.critic import critic_agent
from agents.synthesizer import synthesizer_agent
from memory.vector_store import get_research_context, store_research_session

logger = logging.getLogger(__name__)


# ── CONDITIONAL ROUTING FUNCTION ──────────────────────────────────────────────

def route_after_critic(state: ResearchState) -> str:
    needs_more = state.get("needs_more_research", False)
    loop_count = state.get("research_loop_count", 0)

    if needs_more and loop_count < MAX_RESEARCH_LOOPS:
        logger.info(
            "[Router] Critic flagged gaps — looping back to Researcher "
            "(loop %d of %d)", loop_count + 1, MAX_RESEARCH_LOOPS,
        )
        return "researcher"

    logger.info("[Router] Research sufficient — proceeding to Synthesizer")
    return "synthesizer"


# ── GRAPH BUILDER ─────────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    graph.add_node("planner",     planner_agent)
    graph.add_node("researcher",  researcher_agent)
    graph.add_node("critic",      critic_agent)
    graph.add_node("synthesizer", synthesizer_agent)

    graph.set_entry_point("planner")

    graph.add_edge("planner",    "researcher")
    graph.add_edge("researcher", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "researcher":  "researcher",
            "synthesizer": "synthesizer",
        },
    )

    graph.add_edge("synthesizer", END)

    return graph.compile()


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def run_research(query: str) -> ResearchState:
    start = time.time()
    logger.info("[Graph] Starting research pipeline for: %r", query)

    prior_context = get_research_context(query)
    initial_state = create_initial_state(query)
    if prior_context:
        initial_state["prior_context"] = prior_context
        logger.info("[Graph] Prior research context injected into state")

    graph = build_research_graph()
    final_state = graph.invoke(initial_state)

    try:
        store_research_session(final_state)
        logger.info("[Graph] Session stored to vector memory")
    except Exception as e:
        logger.warning("[Graph] Memory storage failed (non-fatal): %s", e)

    elapsed = time.time() - start
    logger.info("[Graph] Pipeline complete in %.1fs", elapsed)
    return final_state


def run_research_stream(query: str) -> Generator:
    logger.info("[Graph] Starting streaming pipeline for: %r", query)

    prior_context = get_research_context(query)
    initial_state = create_initial_state(query)
    if prior_context:
        initial_state["prior_context"] = prior_context

    graph = build_research_graph()
    final_state = None

    for event in graph.stream(initial_state):
        final_state = event
        yield event

    if final_state:
        try:
            merged: ResearchState = create_initial_state(query)
            merged.update(initial_state)
            for node_output in final_state.values():
                if isinstance(node_output, dict):
                    merged.update(node_output)
            store_research_session(merged)
        except Exception as e:
            logger.warning("[Graph] Memory storage failed (non-fatal): %s", e)
