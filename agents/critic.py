import logging

from langchain_core.prompts import ChatPromptTemplate
from core.config import call_llm_with_fallback, CRITIC_TEMPERATURE, MAX_RESEARCH_LOOPS
from core.state import ResearchState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the quality-gate critic for MARA, a Multi-Agent Research Assistant.

Your ONLY job is to evaluate whether the gathered search results adequately and comprehensively answer the original research query.

EVALUATION CRITERIA:
1. COVERAGE — Are all sub-questions answered by at least one search result?
2. DEPTH — Do the results contain substantive information, not just surface-level mentions?
3. RECENCY — For time-sensitive topics, are the results reasonably up-to-date?
4. DIVERSITY — Do the results come from multiple sources, not just one perspective?
5. GAPS — Are there obvious aspects of the query that NO result addresses?

ANALYSIS RULES:
- Compare each sub-question against the available search results
- Identify specific gaps: name what is missing, not just "more research needed"
- If gaps exist, suggest 1-3 NEW sub-questions that would fill them
- New sub-questions must NOT repeat the original ones — they must target the gaps
- Be pragmatic: perfection is impossible, focus on significant gaps only

OUTPUT FORMAT (STRICT):
Your response MUST end with EXACTLY one of these verdicts on its own line:

VERDICT: SUFFICIENT
VERDICT: INSUFFICIENT

Before the verdict, provide:
1. A brief coverage assessment (2-3 sentences)
2. If INSUFFICIENT: list the specific gaps and 1-3 new sub-questions to fill them

DECISION LOGIC:
- If all sub-questions are reasonably covered → SUFFICIENT
- If 1+ sub-questions have NO relevant results → INSUFFICIENT
- If results are shallow across the board → INSUFFICIENT
- When in doubt and the research loop count is already at the max → SUFFICIENT
  (prefer generating a report with partial info over infinite loops)"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",
     "Original query: {query}\n\n"
     "Sub-questions:\n{sub_questions}\n\n"
     "Search results:\n{results}\n\n"
     "Current research loop: {loop_count} / {max_loops}"),
])


def critic_agent(state: ResearchState) -> dict:
    results_text = "\n".join(
        f"- [{r['title']}] ({r['url']}): {r['content']}"
        for r in state["search_results"]
    ) or "(no results collected)"

    sub_q_text = "\n".join(
        f"{i}. {q}" for i, q in enumerate(state["sub_questions"], 1)
    ) or "(none)"

    loop_count = state["research_loop_count"]

    messages = PROMPT.format_messages(
        query=state["query"],
        sub_questions=sub_q_text,
        results=results_text,
        loop_count=loop_count,
        max_loops=MAX_RESEARCH_LOOPS,
    )

    response = call_llm_with_fallback(
        messages=messages,
        temperature=CRITIC_TEMPERATURE,
    )

    critique = response.content.strip()

    last_line = critique.splitlines()[-1].upper() if critique else ""
    needs_more = "INSUFFICIENT" in last_line and loop_count < MAX_RESEARCH_LOOPS

    new_loop_count = loop_count + 1

    logger.info(
        "Critic verdict: %s (loop %d/%d)",
        "INSUFFICIENT" if needs_more else "SUFFICIENT",
        new_loop_count,
        MAX_RESEARCH_LOOPS,
    )

    return {
        "critique": critique,
        "needs_more_research": needs_more,
        "current_step": "critiquing",
        "research_loop_count": new_loop_count,
    }
