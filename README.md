# MARA — Multi-Agent Research Assistant

MARA is a LangGraph-powered multi-agent pipeline that plans, researches, critiques,
and synthesises answers to complex research queries, backed by a Qdrant vector store
for long-term memory and a Streamlit frontend.

## Project structure

```
mara/
├── agents/          # Planner, Researcher, Critic, Synthesizer agents
├── core/            # LangGraph graph, shared state schema, and config
├── memory/          # Qdrant vector store integration
├── ui/              # Streamlit frontend (app.py)
├── tests/           # Pipeline tests
├── .env.example     # API key template
├── requirements.txt
├── Dockerfile
└── pyproject.toml
```

## Quick start

```bash
cp .env.example .env          # fill in your API keys
pip install -r requirements.txt
streamlit run ui/app.py
```

## Environment variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq LLM API key |
| `TAVILY_API_KEY` | Tavily web-search API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
