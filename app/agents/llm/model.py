from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.core.config import load_project_env, require_openai_api_key


@lru_cache(maxsize=1)
def get_response_model() -> ChatOpenAI:
    """Create and cache a deterministic chat model."""
    load_project_env()
    require_openai_api_key()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)
