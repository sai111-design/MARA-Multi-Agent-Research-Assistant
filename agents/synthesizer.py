import logging
import time
import re
from langchain_core.messages import SystemMessage, HumanMessage
from core.state import ResearchState
from core.config import call_llm_with_fallback, SYNTHESIZER_TEMPERATURE

logger = logging.getLogger(__name__)


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYNTHESIZER_SYSTEM_PROMPT = """You are a research report synthesizer for MARA,
a Multi-Agent Research Assistant.

Your job is to synthesize web search results into a comprehensive,
well-structured, cited research report that directly and completely
answers the original research question.

═══════════════════════════════════════════════════════════
REPORT STRUCTURE  —  follow this order exactly
═══════════════════════════════════════════════════════════

1. # [Report Title]
   Create a clear, descriptive title based on the research question.
   Do NOT use "Research Report on..." — make it specific and informative.

2. ## Overview
   2-3 sentence executive summary of the key findings.
   Answer the research question directly upfront.

3. ## [Section 1 Title]
   First major topic or aspect of the research question.
   Use findings from the search results.
   Every factual claim needs an inline citation [N].

4. ## [Section 2 Title]
   Second major topic. Adapt section titles to the query — not all
   reports need the same sections.

5. ## [Section 3–5 Titles]  (add as many sections as needed, max 6 total)

6. ## Key Takeaways
   3-5 bullet points summarising the most important findings.
   Each bullet point should be a complete, standalone insight.

7. ## Limitations   (ONLY include if the Critic identified research gaps)
   Briefly acknowledge what could not be covered with available sources.
   Keep to 2-3 sentences maximum.

8. ## Sources
   List every cited source in this EXACT format, one per line:
     [1] Title of Page — https://full-url-here.com
     [2] Title of Page — https://full-url-here.com
   Use an em dash (—) between title and URL.
   Only list sources that were actually cited in the report body.

═══════════════════════════════════════════════════════════
CITATION RULES  —  non-negotiable
═══════════════════════════════════════════════════════════

- Every factual claim, statistic, or specific assertion MUST have an
  inline citation.
- Citation format: [N] where N matches the source number in the
  Sources section.
- Place the citation IMMEDIATELY after the claim, before punctuation:
    "Quantum computers use qubits instead of classical bits [1], which..."
- You may cite the same source multiple times using the same number.
- NEVER invent citations — only use sources provided in the context.
- If two sources support the same claim, cite both: "...has grown [2][4]"
- Do NOT write "[Source N]" or "(N)" — only the "[N]" bracket format.

═══════════════════════════════════════════════════════════
WRITING STANDARDS
═══════════════════════════════════════════════════════════

- Write in clear, professional prose — no conversational filler.
- Target 600-900 words in the body (not counting the Sources section).
- Use ## for section headers, ### for sub-headers only when needed.
- Use **bold** for key terms on first use only.
- Prefer prose paragraphs over bullet points in the body sections.
- If sources contradict each other, present BOTH perspectives with
  citations from each side.
- Do NOT make up information — if the search results do not cover
  something, skip it or note the limitation.
- Do NOT begin sentences with "According to source [N]" — integrate
  citations naturally into the prose.

═══════════════════════════════════════════════════════════
INPUT FORMAT YOU WILL RECEIVE
═══════════════════════════════════════════════════════════

RESEARCH QUESTION: [the original user query]

SOURCE INDEX:
[1] Title | URL
[2] Title | URL
...

RESEARCH RESULTS:
--- Source [1] ---
Title: ...
URL: ...
Content: ...

--- Source [2] ---
Title: ...
URL: ...
Content: ...

CRITIC ASSESSMENT (if present):
[gap analysis from the Critic agent — use this to write the Limitations section]

Use the SOURCE INDEX to assign citation numbers. The numbers in the
SOURCE INDEX are the exact numbers you must use in your inline
citations and Sources section.
"""


# ── HUMAN MESSAGE BUILDER ─────────────────────────────────────────────────────

def build_human_message(state: ResearchState) -> str:
    query = state["query"]
    search_results = state.get("search_results", [])
    critique = state.get("critique")

    seen_urls: dict = {}
    source_index: list = []
    for result in search_results:
        url = result.get("url", "")
        if url and url not in seen_urls:
            seen_urls[url] = len(source_index) + 1
            source_index.append(result)

    index_lines = []
    for i, src in enumerate(source_index, 1):
        index_lines.append(
            f"[{i}] {src.get('title', 'Untitled')} | {src.get('url', '')}"
        )

    results_lines = []
    for i, src in enumerate(source_index, 1):
        content = src.get("content", "")
        results_lines.append(
            f"--- Source [{i}] ---\n"
            f"Title: {src.get('title', 'Untitled')}\n"
            f"URL: {src.get('url', '')}\n"
            f"Content: {content}"
        )

    parts = [
        f"RESEARCH QUESTION: {query}",
        "",
        "SOURCE INDEX:",
        "\n".join(index_lines) if index_lines else "(no sources available)",
        "",
        "RESEARCH RESULTS:",
        "\n\n".join(results_lines) if results_lines else "(no results available)",
    ]

    if critique and "failed" not in critique.lower():
        parts += ["", "CRITIC ASSESSMENT:", critique]

    return "\n".join(parts)


# ── FALLBACK REPORT GENERATOR ─────────────────────────────────────────────────

def generate_fallback_report(state: ResearchState) -> str:
    query = state["query"]
    results = state.get("search_results", [])

    lines = [
        f"# Research Results: {query}",
        "",
        "> Report synthesis failed. Showing raw research results below.",
        "",
        "## Sources Found",
        "",
    ]
    for i, r in enumerate(results, 1):
        title = r.get("title", "Untitled")
        url = r.get("url", "")
        content = r.get("content", "")[:200]
        lines.append(f"**[{i}] {title}**")
        lines.append(f"URL: {url}")
        lines.append(f"Summary: {content}...")
        lines.append("")

    if not results:
        lines.append("No search results were collected. Please try again.")

    return "\n".join(lines)


# ── MAIN AGENT FUNCTION ───────────────────────────────────────────────────────

def synthesizer_agent(state: ResearchState) -> dict:
    start = time.time()
    source_count = len(state.get("search_results", []))
    logger.info("[Synthesizer] Starting report synthesis (%d sources)", source_count)

    try:
        human_message = build_human_message(state)

        messages = [
            SystemMessage(content=SYNTHESIZER_SYSTEM_PROMPT),
            HumanMessage(content=human_message),
        ]

        response = call_llm_with_fallback(
            messages=messages,
            temperature=SYNTHESIZER_TEMPERATURE,
        )

        final_report = response.content.strip()

        has_title = "# " in final_report
        has_sources = "## Sources" in final_report

        if not has_title or not has_sources:
            logger.warning(
                "[Synthesizer] Validation failed (title=%s, sources=%s) — using fallback",
                has_title, has_sources,
            )
            final_report = generate_fallback_report(state)
        else:
            citation_count = len(re.findall(r"\[\d+\]", final_report))
            word_count = len(final_report.split())
            elapsed = time.time() - start
            logger.info(
                "[Synthesizer] Done in %.1fs | %d words | %d citations | %d sources",
                elapsed, word_count, citation_count, source_count,
            )

        return {
            "final_report": final_report,
            "current_step": "synthesizing",
        }

    except Exception as e:
        elapsed = time.time() - start
        logger.exception("[Synthesizer] Failed after %.1fs", elapsed)
        return {
            "final_report": generate_fallback_report(state),
            "current_step": "synthesizing",
        }
