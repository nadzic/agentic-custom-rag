from typing import Literal

from pydantic import BaseModel, Field

from app.agents.graph.state import AgentState
from app.agents.llm.model import get_response_model

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question.\n"
    "Retrieved document:\n\n{context}\n\n"
    "User question: {question}\n"
    "If the document contains keyword matches or semantic meaning related to the user question, "
    "grade it as relevant.\n"
    "Focus on topical relevance, not answer completeness.\n"
    "Return exactly one token in lowercase: 'yes' or 'no'."
)


class GradeDocuments(BaseModel):
    """Binary relevance score for retrieved context."""

    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, 'no' if not relevant."
    )


def grade_documents(state: AgentState) -> Literal["generate_answer", "rewrite_question"]:
    """Route based on relevance of retrieved tool output."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GRADE_PROMPT.format(context=context, question=question)

    response_model = get_response_model()
    result = response_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    return "generate_answer" if result.binary_score == "yes" else "rewrite_question"
