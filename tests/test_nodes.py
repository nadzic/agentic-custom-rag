from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from my_agent.utils import nodes


class DummyResponseModel:
    def __init__(self, invoke_result=None, structured_result=None):
        self._invoke_result = invoke_result
        self._structured_result = structured_result
        self.bound_tools = None
        self.invoked_with = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, payload):
        self.invoked_with = payload
        if self._structured_result is not None:
            return self._structured_result
        return self._invoke_result

    def with_structured_output(self, _schema):
        return self

    def set_structured_result(self, result):
        self._structured_result = result
        return self


def test_generate_query_or_respond_binds_and_invokes(monkeypatch):
    model = DummyResponseModel(invoke_result=SimpleNamespace(content="tool-call"))
    monkeypatch.setattr(nodes, "get_response_model", lambda: model)
    monkeypatch.setattr(nodes, "get_retriever_tool", lambda: "retriever_tool")

    state = {"messages": [HumanMessage(content="hello")]}
    result = nodes.generate_query_or_respond(state)

    assert model.bound_tools == ["retriever_tool"]
    assert result["messages"][0].content == "tool-call"


def test_grade_documents_routes_generate_answer(monkeypatch):
    model = DummyResponseModel()
    model.set_structured_result(SimpleNamespace(binary_score="yes"))
    monkeypatch.setattr(nodes, "get_response_model", lambda: model)

    state = {"messages": [HumanMessage(content="question"), HumanMessage(content="context")]}
    route = nodes.grade_documents(state)

    assert route == "generate_answer"


def test_grade_documents_routes_rewrite_question(monkeypatch):
    model = DummyResponseModel()
    model.set_structured_result(SimpleNamespace(binary_score="no"))
    monkeypatch.setattr(nodes, "get_response_model", lambda: model)

    state = {"messages": [HumanMessage(content="question"), HumanMessage(content="context")]}
    route = nodes.grade_documents(state)

    assert route == "rewrite_question"


def test_rewrite_question_returns_human_message(monkeypatch):
    model = DummyResponseModel(invoke_result=SimpleNamespace(content="better question"))
    monkeypatch.setattr(nodes, "get_response_model", lambda: model)

    state = {"messages": [HumanMessage(content="bad question"), HumanMessage(content="ctx")]}
    result = nodes.rewrite_question(state)

    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "better question"


def test_generate_answer_returns_human_message(monkeypatch):
    model = DummyResponseModel(invoke_result=SimpleNamespace(content="final answer"))
    monkeypatch.setattr(nodes, "get_response_model", lambda: model)

    state = {"messages": [HumanMessage(content="q"), HumanMessage(content="ctx")]}
    result = nodes.generate_answer(state)

    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "final answer"
