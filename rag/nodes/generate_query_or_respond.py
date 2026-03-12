from functools import lru_cache
from pathlib import Path
import sys
from rag.web_loader import get_retriever_tool
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Support direct execution from rag/nodes directory:
# uv run python generate_query_or_respond.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

def _load_env() -> None:
    """Load .env from project root for direct script execution."""
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env")


@lru_cache(maxsize=1)
def get_response_model() -> ChatOpenAI:
    """Create and cache the response model lazily."""
    _load_env()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def generate_query_or_respond(state: MessagesState) -> MessagesState:
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response_model = get_response_model()
    retriever_tool = get_retriever_tool()
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])

    return {"messages": [response]}


if __name__ == "__main__":
    state: MessagesState = {"messages": [HumanMessage(content="What does Lilian Weng say about types of reward hacking?")]}
    generate_query_or_respond(state)["messages"][-1].pretty_print()
