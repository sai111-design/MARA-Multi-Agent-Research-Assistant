from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph.state import MARAState
from utils.logger import log_agent

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research synthesizer. Write a comprehensive, well-structured markdown report answering the query. "
               "Use numbered citations [1], [2], etc. referencing the provided sources. "
               "Include a ## Sources section at the end listing each URL."),
    ("human", "Query: {query}\n\nCritic feedback: {critique}\n\nSources:\n{sources}"),
])

def synthesizer_agent(state: MARAState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    chain = PROMPT | llm
    sources_text = "\n".join(
        f"[{i+1}] {r['title']} ({r['url']}): {r['content']}"
        for i, r in enumerate(state["search_results"])
    )
    try:
        response = chain.invoke({
            "query": state["query"],
            "critique": state.get("critique", ""),
            "sources": sources_text,
        })
        report = response.content.strip()
    except Exception:
        report = "\n".join(f"- [{r['title']}]({r['url']}): {r['content']}" for r in state["search_results"])
    log_agent("Synthesizer", state["query"], report[:200])
    return {"final_report": report, "current_step": "synthesizer"}
