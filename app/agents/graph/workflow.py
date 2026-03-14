from pathlib import Path

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agents.graph.edges import grade_documents
from app.agents.graph.nodes import (
    generate_answer,
    generate_query_or_respond,
    rewrite_question,
)
from app.agents.graph.state import AgentState
from app.agents.tools.search import get_retriever_tool, setup_langsmith


def build_graph():
    """Construct and compile the LangGraph workflow."""
    workflow = StateGraph(AgentState)

    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([get_retriever_tool()]))
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {"tools": "retrieve", END: END},
    )
    workflow.add_conditional_edges("retrieve", grade_documents)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


graph = build_graph()


def save_graph_image(path: Path) -> None:
    """Render and save workflow diagram as PNG."""
    path.write_bytes(graph.get_graph().draw_mermaid_png())


def run_demo_query() -> str:
    """Run one example query through the graph."""
    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What does Lilian Weng say about types of reward hacking?"
                )
            ]
        },
        config={
            "run_name": "rag_workflow",
            "tags": ["langgraph", "rag"],
            "metadata": {"app": "agentic-custom-rag"},
        },
    )
    return result["messages"][-1].content


def run() -> None:
    """CLI-like run helper."""
    tracing_enabled = setup_langsmith()
    print(
        "LangSmith tracing enabled."
        if tracing_enabled
        else "LangSmith disabled (LANGSMITH_API_KEY not set)."
    )

    output_path = Path(__file__).resolve().parents[1] / "workflow_graph.png"
    save_graph_image(output_path)
    print(f"Graph image saved to: {output_path}")
    print(run_demo_query())
