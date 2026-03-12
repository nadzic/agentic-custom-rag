from pathlib import Path
import sys

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# Support direct execution: uv run python my_agent/agent.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from my_agent.utils.nodes import (  # noqa: E402
    generate_answer,
    generate_query_or_respond,
    grade_documents,
    rewrite_question,
)
from my_agent.utils.state import AgentState  # noqa: E402
from my_agent.utils.tools import get_retriever_tool, setup_langsmith  # noqa: E402


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


def main() -> None:
    tracing_enabled = setup_langsmith()
    print(
        "LangSmith tracing enabled."
        if tracing_enabled
        else "LangSmith disabled (LANGSMITH_API_KEY not set)."
    )

    output_path = Path(__file__).with_name("workflow_graph.png")
    save_graph_image(output_path)
    print(f"Graph image saved to: {output_path}")

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
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
