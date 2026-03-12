from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from functools import lru_cache
from pathlib import Path
from langchain_core.messages import HumanMessage
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


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)

def generate_answer(state: MessagesState):
  """Generate an answer"""
  question = state["messages"][0].content
  context = state["messages"][-1].content
  prompt = GENERATE_PROMPT.format(question=question, context=context)
  response_model = get_response_model()
  response = response_model.invoke([{"role": "user", "content": prompt}])
  return { "messages": [HumanMessage(content=response.content)]}

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
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
  }

  response = generate_answer(input)
  print(response["messages"][-1].content)