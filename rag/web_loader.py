import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

try:
    from rag.chunking import split_documents_into_chunks
    from rag.retrieval import create_retrieve_blog_posts_tool
    from rag.web_sources import load_web_documents
except ModuleNotFoundError:
    # Support direct execution: python rag/web_loader.py
    from chunking import split_documents_into_chunks
    from retrieval import create_retrieve_blog_posts_tool
    from web_sources import load_web_documents


def _load_env() -> None:
    """Load API keys from project .env regardless of current working directory."""
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or environment.")

@lru_cache(maxsize=1)
def get_retriever_tool():
    """Build and cache the retrieval tool used by the graph node."""
    _load_env()
    documents = load_web_documents()
    chunks = split_documents_into_chunks(documents)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings()
    )
    return create_retrieve_blog_posts_tool(vectorstore.as_retriever())


def main() -> None:
    _load_env()
    documents = load_web_documents()
    chunks = split_documents_into_chunks(documents)

    retriever_tool = get_retriever_tool()
    result = retriever_tool.invoke({"query": "types of reward hacking"})
    print(result[:500])

    print(f"Loaded {len(documents)} documents")
    print(f"Split into {len(chunks)} chunks")


if __name__ == "__main__":
    main()