from pathlib import Path
import os
import sys

# Support direct execution from rag/nodes:
# uv run python workflow_graph.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from my_agent.agent import graph, save_graph_image, setup_langsmith


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
                {
                    "role": "user",
                    "content": "What does Lilian Weng say about reward hacking types?",
                }
            ]
        }
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()