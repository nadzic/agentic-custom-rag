## Agentic Custom RAG

Simple retrieval pipeline over web content (Lilian Weng posts) using:
- web loading (`requests` + `BeautifulSoup`)
- chunking (`RecursiveCharacterTextSplitter`)
- embeddings (`OpenAIEmbeddings`)
- in-memory vector search (`InMemoryVectorStore`)

## How It Works (Chart)

```mermaid
flowchart TD
    A["`rag/constants.py
    DEFAULT_URLS`"] --> B["`rag/web_sources.py
    load_web_documents()`"]
    B --> C["`List[Document]`"]
    C --> D["`rag/chunking.py
    split_documents_into_chunks()`"]
    D --> E["`chunked documents`"]
    E --> F["`OpenAIEmbeddings`"]
    F --> G["`InMemoryVectorStore.from_documents(...)`"]
    G --> H["`vectorstore.as_retriever()`"]
    H --> I["`rag/retrieval.py
    create_retrieve_blog_posts_tool(retriever)`"]
    I --> J["`retriever_tool.invoke({'query': ...})`"]
    J --> K["`top matching chunks as text output`"]
```

Flow owner: `rag/web_loader.py` orchestrates this sequence end-to-end.

## Project Structure

- `rag/constants.py` - source URLs
- `rag/web_sources.py` - load documents from web pages
- `rag/chunking.py` - split documents into chunks
- `rag/retrieval.py` - build a retrieval tool from a retriever
- `rag/web_loader.py` - pipeline entrypoint / example run

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create `.env` in project root:

```env
OPENAI_API_KEY=your_key_here
```

## Run

```bash
uv run python rag/web_loader.py
```

You should see:
- a short retrieved text snippet
- `Loaded X documents`
- `Split into Y chunks`

## Notes

- `.env` is ignored by git to avoid committing secrets.
- `rag/web_loader.py` supports both:
  - `uv run python rag/web_loader.py`
  - module-style imports when used from package context
