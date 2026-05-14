from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from graph.state import MARAState
from utils.logger import log_agent

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research planner. Decompose the user's query into 3-5 focused sub-questions. "
               "Output a numbered list only, one sub-question per line."),
    ("human", "{query}"),
])

def planner_agent(state: MARAState) -> dict:
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    chain = PROMPT | llm
    response = chain.invoke({"query": state["query"]})
    lines = [l.strip() for l in response.content.strip().splitlines() if l.strip()]
    sub_questions = [l.lstrip("0123456789). ") for l in lines if l]
    log_agent("Planner", state["query"], sub_questions)
    return {"sub_questions": sub_questions or [state["query"]], "current_step": "planner"}
