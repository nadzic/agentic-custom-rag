from app.api.schemas import QueryRequest, QueryResponse
from app.agents.graph import graph


def answer_query(payload: QueryRequest) -> QueryResponse:
    """Simple callable route for answering a query via graph."""
    result = graph.invoke({"messages": [{"role": "user", "content": payload.query}]})
    return QueryResponse(answer=result["messages"][-1].content)
