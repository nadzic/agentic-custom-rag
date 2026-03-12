from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import convert_to_messages

GRADE_PROMPT = (
  "You are a grader assessing relevance of a retrieved document to a user question. \n "
  "Here is the retrieved document: \n\n {context} \n\n"
  "Here is the user question: {question} \n"
  "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant, even for short snippets.\n"
  "Focus on topical relevance, not answer completeness.\n"
  "Return exactly one token in lowercase: 'yes' or 'no'."
)

class GradeDocuments(BaseModel):
  """Grade documents using a binary score for relevance check."""
  binary_score: Literal["yes", "no"] = Field(
      description="Relevance score: 'yes' if relevant, 'no' if not relevant."
  )


def _load_env() -> None:
  """Load .env from project root for direct script execution."""
  repo_root = Path(__file__).resolve().parents[2]
  load_dotenv(repo_root / ".env")


@lru_cache(maxsize=1)
def get_grader_model() -> ChatOpenAI:
  """Create and cache grader model lazily."""
  _load_env()
  return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
  """Determine whether the retrived document are relevant to the user question. If not, rewrite the question to be more specific."""
  question = state["messages"][0].content
  context = state["messages"][-1].content

  prompt = GRADE_PROMPT.format(context=context, question=question)


  
  grader_model = get_grader_model()
  response = grader_model.with_structured_output(GradeDocuments).invoke([{"role": "user", "content": prompt}])

  print(response)

  if response.binary_score == "yes":
    return "generate_answer"
  else:
    return "rewrite_question"

if __name__ == "__main__":
  irrelevant_input = {
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

  relevant_input = {
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
                  "content": (
                      "Lilian Weng explains that reward hacking has two main types: "
                      "environment/goal misspecification and reward tampering."
                  ),
                  "tool_call_id": "1",
              },
          ]
      )
  }
  print(grade_documents(irrelevant_input))
  print(grade_documents(relevant_input))