---
title: MARA Multi-Agent Research Assistant
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: ui/app.py
pinned: false
---

# MARA — Multi-Agent Research Assistant

MARA is a LangGraph-powered multi-agent pipeline that transforms any research question into a comprehensive, cited report. Four specialised AI agents — **Planner**, **Researcher**, **Critic**, and **Synthesizer** — collaborate in a loop, backed by a Qdrant vector store for long-term memory and a dark-themed Gradio frontend with live pipeline streaming.

## Agent Pipeline

| Agent | Role |
|---|---|
| **Planner** | Decomposes the user query into targeted sub-questions |
| **Researcher** | Searches the web (Tavily) and gathers evidence for each sub-question |
| **Critic** | Evaluates whether the collected research is sufficient or flags gaps |
| **Synthesizer** | Writes the final report with structured sections and inline citations |

If the Critic finds gaps, the pipeline loops back to the Researcher for additional searches before synthesising.

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| Primary LLM | Google Gemini 2.0 Flash |
| Fallback LLM | Groq Llama 3.3 70B |
| Web Search | Tavily API |
| Vector Memory | Qdrant |
| Embeddings | Sentence Transformers |
| Frontend | Gradio 6 |
| Containerisation | Docker |

## Project Structure

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

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/your-username/mara.git
cd mara
cp .env.example .env          # fill in your API keys
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Launch the UI

```bash
python ui/app.py
# Opens at http://localhost:7860
```

Or with Gradio hot-reload for development:

```bash
gradio ui/app.py
```

### Docker

```bash
docker build -t mara .
docker run -p 7860:7860 --env-file .env mara
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq LLM API key (fallback model) |
| `TAVILY_API_KEY` | Yes | Tavily web-search API key |
| `GOOGLE_API_KEY` | Yes | Google Gemini API key (primary model) |
| `PRIMARY_MODEL` | No | Override primary model (default: `gemini-2.0-flash`) |
| `FALLBACK_MODEL` | No | Override fallback model (default: `llama-3.3-70b-versatile`) |

## Hugging Face Spaces

This repo is configured for one-click deployment on [Hugging Face Spaces](https://huggingface.co/spaces). The YAML front-matter at the top of this file sets `sdk: gradio`, so HF uses its native Gradio runtime — no Docker required, faster cold starts on the free tier.

Set your API keys as **Repository Secrets** in the Space settings.

## License

MIT
