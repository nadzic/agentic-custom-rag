from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from langchain_core.messages import convert_to_messages

def _load_env() -> None:
  """Load .env from project root for direct script execution."""
  repo_root = Path(__file__).resolve().parents[2]
  load_dotenv(repo_root / ".env")


@lru_cache(maxsize=1)
def get_response_model() -> ChatOpenAI:
  """Create and cache the response model lazily."""
  _load_env()
  return ChatOpenAI(model="gpt-4o-mini", temperature=0)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

def rewrite_question(state: MessagesState):
  """Rewrite the original user question."""
  messages = state["messages"]
  question = messages[0].content
  prompt = REWRITE_PROMPT.format(question=question)
  response_model = get_response_model()
  response = response_model.invoke([{"role": "user", "content": prompt}])
  return {"messages": [HumanMessage(content=response.content)]}

# This is just for debugging purposes
if __name__ == "__main__":
  input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
  }

  response = rewrite_question(input)
  print(response["messages"][-1].content)