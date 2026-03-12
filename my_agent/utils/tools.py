import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from rag.chunking import split_documents_into_chunks
from rag.retrieval import create_retrieve_blog_posts_tool
from rag.web_sources import load_web_documents


def load_project_env() -> None:
    """Load environment variables from the repository root .env."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


def setup_langsmith() -> bool:
    """Enable LangSmith tracing when a key is available."""
    load_project_env()
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "agentic-custom-rag")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    return True


@lru_cache(maxsize=1)
def get_retriever_tool():
    """Build and cache the blog retrieval tool."""
    load_project_env()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or environment.")

    documents = load_web_documents()
    chunks = split_documents_into_chunks(documents)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings()
    )
    return create_retrieve_blog_posts_tool(vectorstore.as_retriever())
