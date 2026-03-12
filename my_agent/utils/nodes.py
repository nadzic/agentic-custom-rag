from functools import lru_cache
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from my_agent.utils.state import AgentState
from my_agent.utils.tools import get_retriever_tool, load_project_env

REWRITE_PROMPT = (
    "Look at the input and reason about the underlying semantic intent.\n"
    "Initial question:\n"
    "-------\n"
    "{question}\n"
    "-------\n"
    "Formulate an improved question."
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you do not know the answer, say you do not know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)

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


@lru_cache(maxsize=1)
def get_response_model() -> ChatOpenAI:
    """Create and cache a deterministic chat model."""
    load_project_env()
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


def generate_query_or_respond(state: AgentState) -> AgentState:
    """Let the model decide whether to call retrieval tool or respond directly."""
    response_model = get_response_model()
    retriever_tool = get_retriever_tool()
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


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


def rewrite_question(state: AgentState) -> AgentState:
    """Rewrite the original user question for a better retrieval query."""
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)

    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer from question and retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)

    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}
