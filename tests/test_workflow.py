from pathlib import Path

import app.agents.graph.workflow as workflow


def test_save_graph_image_writes_png(tmp_path, monkeypatch):
    class DummyCompiledGraph:
        class _Graph:
            @staticmethod
            def draw_mermaid_png():
                return b"png-bytes"

        @staticmethod
        def get_graph():
            return DummyCompiledGraph._Graph()

    monkeypatch.setattr(workflow, "graph", DummyCompiledGraph())
    output = tmp_path / "workflow_graph.png"
    workflow.save_graph_image(output)

    assert output.exists()
    assert output.read_bytes() == b"png-bytes"


def test_run_prints_status_and_answer(monkeypatch, capsys):
    saved_paths = []
    monkeypatch.setattr(workflow, "setup_langsmith", lambda: True)
    monkeypatch.setattr(workflow, "save_graph_image", lambda path: saved_paths.append(path))
    monkeypatch.setattr(workflow, "run_demo_query", lambda: "final answer")

    workflow.run()
    out = capsys.readouterr().out

    assert "LangSmith tracing enabled." in out
    assert "final answer" in out
    assert saved_paths and saved_paths[0] == Path(workflow.__file__).resolve().parents[1] / "workflow_graph.png"
