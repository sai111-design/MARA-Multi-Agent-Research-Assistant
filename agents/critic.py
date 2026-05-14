from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph.state import MARAState
from utils.logger import log_agent

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research critic. Evaluate whether the search results sufficiently answer the original query. "
               "Identify gaps. End your response with either SUFFICIENT or INSUFFICIENT on its own line."),
    ("human", "Query: {query}\n\nSub-questions: {sub_questions}\n\nResults:\n{results}"),
])

def critic_agent(state: MARAState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    chain = PROMPT | llm
    results_text = "\n".join(
        f"- {r['title']}: {r['content']}" for r in state["search_results"]
    )
    response = chain.invoke({
        "query": state["query"],
        "sub_questions": "\n".join(state["sub_questions"]),
        "results": results_text,
    })
    critique = response.content.strip()
    needs_more = "INSUFFICIENT" in critique.upper().splitlines()[-1]
    log_agent("Critic", state["query"], {"critique": critique, "needs_more": needs_more})
    return {"critique": critique, "needs_more_research": needs_more, "current_step": "critic"}
