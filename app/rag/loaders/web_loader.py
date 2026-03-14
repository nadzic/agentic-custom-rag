from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.rag.sources.source_registry import DEFAULT_URLS


def load_web_documents(urls: Iterable[str] = DEFAULT_URLS) -> List[Document]:
    """Load and flatten webpage content into LangChain documents."""
    documents: List[Document] = []

    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator="\n", strip=True)
            title = soup.title.string.strip() if soup.title and soup.title.string else url

            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": url, "title": title},
                )
            )
        except Exception as exc:
            print(f"Failed to load {url}: {exc}")

    return documents
