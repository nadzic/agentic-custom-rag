## Agentic Custom RAG

Simple retrieval pipeline over web content (Lilian Weng posts) using:
- web loading (`requests` + `BeautifulSoup`)
- chunking (`RecursiveCharacterTextSplitter`)
- embeddings (`OpenAIEmbeddings`)
- in-memory vector search (`InMemoryVectorStore`)

The LangGraph app is now organized in an app-centric production layout with
`app/` as the primary project root.

## How It Works (Chart)

```mermaid
flowchart TD
    A["`app/rag/sources/source_registry.py
    DEFAULT_URLS`"] --> B["`app/rag/loaders/web_loader.py
    load_web_documents()`"]
    B --> C["`List[Document]`"]
    C --> D["`app/rag/processing/chunking.py
    split_documents_into_chunks()`"]
    D --> E["`chunked documents`"]
    E --> F["`OpenAIEmbeddings`"]
    F --> G["`InMemoryVectorStore.from_documents(...)`"]
    G --> H["`vectorstore.as_retriever()`"]
    H --> I["`app/rag/retrieval/retriever.py
    create_retrieve_blog_posts_tool(retriever)`"]
    I --> J["`retriever_tool.invoke({'query': ...})`"]
    J --> K["`top matching chunks as text output`"]
```

Flow owner: `app/agents/graph.py` orchestrates this sequence end-to-end.

## Agent Workflow Graph

The LangGraph workflow (query -> retrieve -> grade -> rewrite/answer) is defined in
`app/agents/graph.py`.

Generated workflow image:

![Workflow graph](app/agents/workflow_graph.png)

To regenerate the image:

```bash
uv run python app/main.py
```

## Project Structure

```text
app/
├── main.py
├── core/
│   ├── config.py
│   ├── constants.py
│   └── logging.py
├── api/
│   ├── routes.py
│   └── schemas.py
├── services/
│   └── rag_service.py
├── rag/
│   ├── loaders/
│   │   └── web_loader.py
│   ├── sources/
│   │   └── source_registry.py
│   ├── processing/
│   │   └── chunking.py
│   └── retrieval/
│       └── retriever.py
├── llm/
│   └── model.py
└── agents/
    ├── graph.py
    ├── state.py
    └── nodes/
        ├── retrieve_node.py
        ├── answer_node.py
        └── route_node.py
```

Other project files:
- `langgraph.json` - LangGraph graph configuration
- `requirements.txt` - pip-compatible dependency list
- `app/` - main application package

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Create `.env` in project root:

```env
OPENAI_API_KEY=your_key_here
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=agentic-custom-rag
LANGSMITH_TRACING=true
```

## Run

```bash
uv run python app/main.py
```

You should see:
- a short retrieved text snippet
- `Loaded X documents`
- `Split into Y chunks`

Run the LangGraph workflow (with LangSmith tracing when key is set):

```bash
uv run python app/main.py
```

If `LANGSMITH_API_KEY` is present, runs are tracked in LangSmith under your project.

## Notes

- `.env` is ignored by git to avoid committing secrets.
