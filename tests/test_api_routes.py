from types import SimpleNamespace

from app.api.routes import answer_query
from app.api.schemas import QueryRequest
import app.api.routes as routes


def test_answer_query_uses_graph_and_returns_schema(monkeypatch):
    class DummyGraph:
        def invoke(self, payload):
            assert payload["messages"][0]["content"] == "hello"
            return {"messages": [SimpleNamespace(content="world")]}

    monkeypatch.setattr(routes, "graph", DummyGraph())
    response = answer_query(QueryRequest(query="hello"))

    assert response.answer == "world"
