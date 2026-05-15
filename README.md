---
title: MARA Multi-Agent Research Assistant
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.0"
app_file: ui/app.py
app_port: 7860
---

# MARA — Multi-Agent Research Assistant

MARA is a LangGraph-powered multi-agent pipeline that plans, researches, critiques,
and synthesises answers to complex research queries, backed by a Qdrant vector store
for long-term memory and a Gradio frontend.

## Project structure

```
mara/
├── agents/          # Planner, Researcher, Critic, Synthesizer agents
├── core/            # LangGraph graph, shared state schema, and config
├── memory/          # Qdrant vector store integration
├── ui/              # Gradio frontend (app.py)
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
python ui/app.py
# Opens at http://localhost:7860
```

## Tech stack

| Layer | Tool |
|---|---|
| Frontend | Gradio 4+ |

## Environment variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq LLM API key |
| `TAVILY_API_KEY` | Tavily web-search API key |
| `GOOGLE_API_KEY` | Google Gemini API key |
