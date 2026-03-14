import os
from pathlib import Path

from dotenv import load_dotenv

from app.core.constants import APP_NAME


def load_project_env() -> None:
    """Load environment variables from the repository root .env."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


def require_openai_api_key() -> None:
    """Ensure OpenAI API key exists for embedding/chat calls."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or environment.")


def setup_langsmith() -> bool:
    """Enable LangSmith tracing when a key is available."""
    load_project_env()
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", APP_NAME)
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    return True
