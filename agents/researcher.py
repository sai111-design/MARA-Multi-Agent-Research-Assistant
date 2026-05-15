import time
import logging
from typing import List

from core.config import TAVILY_API_KEY, SEARCH_RESULTS_PER_QUERY, MAX_CONTENT_LENGTH
from core.state import ResearchState

logger = logging.getLogger(__name__)


def _truncate(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _search_tavily(query: str, max_results: int) -> List[dict]:
    from tavily import TavilyClient

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=query, search_depth="advanced", max_results=max_results
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
        }
        for r in response.get("results", [])
    ]


def duckduckgo_fallback(query: str, max_results: int = 3) -> List[dict]:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        raw = list(ddgs.text(query, max_results=max_results))
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("href", ""),
            "content": r.get("body", ""),
        }
        for r in raw
    ]


def _search_single(query: str, max_results: int, use_tavily: bool) -> List[dict]:
    search_fn = _search_tavily if use_tavily else duckduckgo_fallback

    for attempt in range(2):
        try:
            return search_fn(query, max_results)
        except Exception as e:
            if attempt == 0:
                logger.warning("Search failed for '%s' (attempt 1), retrying: %s", query, e)
            else:
                logger.error("Search failed for '%s' after retry: %s", query, e)
    return []


def researcher_agent(state: ResearchState) -> dict:
    start = time.time()

    use_tavily = bool(TAVILY_API_KEY)
    backend = "Tavily" if use_tavily else "DuckDuckGo"
    logger.info("Researcher using %s backend", backend)

    all_results: List[dict] = []

    for sub_q in state["sub_questions"]:
        raw = _search_single(sub_q, SEARCH_RESULTS_PER_QUERY, use_tavily)
        for r in raw:
            all_results.append({
                "sub_question": sub_q,
                "title": r["title"],
                "url": r["url"],
                "content": _truncate(r["content"]),
            })

    elapsed = time.time() - start
    logger.info(
        "Researcher collected %d results in %.1fs (%s)",
        len(all_results), elapsed, backend,
    )

    return {"search_results": all_results, "current_step": "researching"}
