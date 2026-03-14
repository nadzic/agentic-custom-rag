from types import SimpleNamespace

from langchain_core.documents import Document

from app.core import config
from app.rag.retrieval import retriever as tools


def test_setup_langsmith_disabled_without_api_key(monkeypatch):
    monkeypatch.setattr(config, "load_project_env", lambda: None)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)

    assert config.setup_langsmith() is False
    assert "LANGSMITH_TRACING" not in config.os.environ


def test_setup_langsmith_enabled_with_api_key(monkeypatch):
    monkeypatch.setattr(config, "load_project_env", lambda: None)
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)

    assert config.setup_langsmith() is True
    assert config.os.environ["LANGSMITH_TRACING"] == "true"
    assert config.os.environ["LANGSMITH_PROJECT"] == "agentic-custom-rag"
    assert config.os.environ["LANGSMITH_ENDPOINT"] == "https://api.smith.langchain.com"


def test_get_retriever_tool_requires_openai_api_key(monkeypatch):
    tools.get_retriever_tool.cache_clear()
    monkeypatch.setattr(tools, "load_project_env", lambda: None)
    monkeypatch.setattr(
        tools,
        "require_openai_api_key",
        lambda: (_ for _ in ()).throw(RuntimeError("OPENAI_API_KEY is missing")),
    )

    try:
        tools.get_retriever_tool()
        assert False, "Expected RuntimeError when OPENAI_API_KEY is missing"
    except RuntimeError as exc:
        assert "OPENAI_API_KEY is missing" in str(exc)


def test_get_retriever_tool_builds_tool(monkeypatch):
    class DummyVectorStore:
        @classmethod
        def from_documents(cls, documents, embedding):
            assert len(documents) == 1
            assert embedding == "embedding"
            return cls()

        def as_retriever(self):
            return "retriever"

    tools.get_retriever_tool.cache_clear()
    monkeypatch.setattr(tools, "load_project_env", lambda: None)
    monkeypatch.setattr(tools, "require_openai_api_key", lambda: None)
    monkeypatch.setattr(
        tools, "load_web_documents", lambda: [Document(page_content="doc", metadata={})]
    )
    monkeypatch.setattr(tools, "split_documents_into_chunks", lambda docs: docs)
    monkeypatch.setattr(tools, "OpenAIEmbeddings", lambda: "embedding")
    monkeypatch.setattr(tools, "InMemoryVectorStore", DummyVectorStore)
    monkeypatch.setattr(
        tools,
        "create_retrieve_blog_posts_tool",
        lambda retriever: SimpleNamespace(name="retrieve_blog_posts", retriever=retriever),
    )

    result = tools.get_retriever_tool()

    assert result.name == "retrieve_blog_posts"
    assert result.retriever == "retriever"
