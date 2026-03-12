from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pathlib import Path
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

# Support direct execution from rag/nodes:
# uv run python workflow_graph.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from rag.nodes.generate_query_or_respond import generate_query_or_respond
from rag.nodes.grade_documents import grade_documents
from rag.nodes.rewrite_question import rewrite_question
from rag.nodes.generate_answer import generate_answer
from rag.web_loader import get_retriever_tool


def setup_langsmith() -> bool:
    """Enable LangSmith tracing when API key is available."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")
    if not os.getenv("LANGSMITH_API_KEY"):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", "agentic-custom-rag")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    return True


workflow = StateGraph(MessagesState)

# define the nodes
workflow.add_node("generate_query_or_respond", generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([get_retriever_tool()]))
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("generate_answer", generate_answer)

# define the edges
workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after retrieve node is called.
workflow.add_conditional_edges(
  "retrieve",
  grade_documents
)

workflow.add_edge("rewrite_question", "generate_query_or_respond")
workflow.add_edge("generate_answer", END)

# compile
graph = workflow.compile()
graph_png = graph.get_graph().draw_mermaid_png()
output_path = Path(__file__).with_name("workflow_graph.png")
output_path.write_bytes(graph_png)

# Display graph only when IPython is available (e.g., notebooks).
try:
    import importlib

    ipy_display = importlib.import_module("IPython.display")
    image = ipy_display.Image(graph_png)
    ipy_display.display(image)
except ModuleNotFoundError:
    print(f"Graph image saved to: {output_path}")


if __name__ == "__main__":
    tracing_enabled = setup_langsmith()
    if tracing_enabled:
        print("LangSmith tracing enabled.")
    else:
        print("LangSmith disabled (LANGSMITH_API_KEY not set).")

    result = graph.invoke(
        {"messages": [HumanMessage(content="What does Lilian Weng say about reward hacking types?")]},
        config={
            "run_name": "rag_workflow",
            "tags": ["langgraph", "rag"],
            "metadata": {"app": "agentic-custom-rag"},
        },
    )
    print(result["messages"][-1].content)