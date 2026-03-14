from langchain_core.messages import HumanMessage

from app.agents.state import AgentState
from app.llm.model import get_response_model

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you do not know the answer, say you do not know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question}\n"
    "Context: {context}"
)


def generate_answer(state: AgentState) -> AgentState:
    """Generate the final answer from question and retrieved context."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)

    response_model = get_response_model()
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}
