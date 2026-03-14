from app.agents.state import AgentState
from app.llm.model import get_response_model
from app.services.rag_service import get_retriever_tool


def generate_query_or_respond(state: AgentState) -> AgentState:
    """Let the model decide whether to call retrieval tool or respond directly."""
    response_model = get_response_model()
    retriever_tool = get_retriever_tool()
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}
