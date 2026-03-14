from functools import lru_cache

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.core.config import load_project_env, require_openai_api_key
from app.rag.loaders.web_loader import load_web_documents
from app.rag.processing.chunking import split_documents_into_chunks


def create_retrieve_blog_posts_tool(retriever: BaseRetriever):
    """Create a tool that queries the provided retriever."""

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about Lilian Weng blog posts."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve_blog_posts


@lru_cache(maxsize=1)
def get_retriever_tool():
    """Build and cache the blog retrieval tool."""
    load_project_env()
    require_openai_api_key()
    documents = load_web_documents()
    chunks = split_documents_into_chunks(documents)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=chunks, embedding=OpenAIEmbeddings()
    )
    return create_retrieve_blog_posts_tool(vectorstore.as_retriever())
