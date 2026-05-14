import os
from tavily import TavilyClient
from graph.state import MARAState
from utils.logger import log_agent

def researcher_agent(state: MARAState) -> dict:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = []
    for question in state["sub_questions"]:
        try:
            response = client.search(query=question, search_depth="advanced", max_results=3)
            for r in response.get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", "")[:500],
                })
        except Exception as e:
            results.append({"title": "Error", "url": "", "content": str(e)})
    log_agent("Researcher", state["sub_questions"], results)
    return {
        "search_results": results,
        "research_loop_count": state.get("research_loop_count", 0) + 1,
        "current_step": "researcher",
    }
