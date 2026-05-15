"""Central configuration for MARA: API keys, model identifiers, and runtime constants.

Loads environment variables from .env via python-dotenv and exposes typed settings
consumed by agents and the vector store.
"""

import os
import time
import logging
from typing import List
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "gemini-2.0-flash"

MAX_RESEARCH_LOOPS = 2
SEARCH_RESULTS_PER_QUERY = 3
MAX_CONTENT_LENGTH = 500
AGENT_TIMEOUT = 30

PLANNER_TEMPERATURE = 0.0
CRITIC_TEMPERATURE = 0.1
SYNTHESIZER_TEMPERATURE = 0.2


def get_llm(temperature: float = 0.0):
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Add it to your .env file before running MARA."
        )
    from langchain_groq import ChatGroq
    return ChatGroq(model=PRIMARY_MODEL, temperature=temperature, api_key=GROQ_API_KEY)


def get_gemini_llm(temperature: float = 0.0):
    if not GOOGLE_API_KEY:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Cannot use Gemini fallback. "
            "Add GOOGLE_API_KEY to your .env file."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=FALLBACK_MODEL,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    err_str = str(exc).lower()
    return (
        "rate limit" in err_str
        or "429" in err_str
        or "ratelimit" in err_str
        or "too many requests" in err_str
    )


def call_llm_with_fallback(
    messages: List[BaseMessage],
    temperature: float = 0.0,
    max_attempts: int = 3,
):
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        use_gemini = attempt > 1
        provider_name = "Gemini Flash" if use_gemini else f"Groq {PRIMARY_MODEL}"

        try:
            if use_gemini:
                llm = get_gemini_llm(temperature=temperature)
            else:
                llm = get_llm(temperature=temperature)

            logger.info("[LLM] Attempt %d/%d via %s", attempt, max_attempts, provider_name)
            response = llm.invoke(messages)
            logger.info("[LLM] Success via %s", provider_name)
            return response

        except Exception as exc:
            last_exc = exc

            if _is_rate_limit_error(exc) and not use_gemini:
                logger.warning("[LLM] Groq rate-limited — switching to Gemini Flash")
                continue

            wait = 2 ** attempt
            logger.warning(
                "[LLM] %s failed on attempt %d: %s. Retrying in %ds…",
                provider_name, attempt, exc, wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"LLM call failed after {max_attempts} attempts. Last error: {last_exc}"
    )
