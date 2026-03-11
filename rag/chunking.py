from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents_into_chunks(
    documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 75
) -> List[Document]:
    """Split documents into smaller chunks for embedding/retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(documents)
