from langchain_core.messages import HumanMessage

from app.agents.graph.state import AgentState
from app.agents.llm.model import get_response_model
from app.agents.tools.search import get_retriever_tool

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


def generate_query_or_respond(state: AgentState) -> AgentState:
    """Let the model decide whether to call retrieval tool or respond directly."""
    response_model = get_response_model()
    retriever_tool = get_retriever_tool()
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


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
