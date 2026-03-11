from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool


def create_retrieve_blog_posts_tool(retriever: BaseRetriever):
    """Create a tool that queries the provided retriever."""

    @tool
    def retrieve_blog_posts(query: str) -> str:
        """Search and return information about Lilian Weng blog posts."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content for doc in docs])

    return retrieve_blog_posts
