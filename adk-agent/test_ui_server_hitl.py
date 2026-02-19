from pathlib import Path

import pytest

import ui_server
from workflow import create_task


@pytest.mark.asyncio
async def test_start_endpoint_uses_plan_version_id(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    async def fake_start(task_id: str, plan_version_id: str | None = None):
        calls.append((task_id, plan_version_id))
        return ui_server.RunRecord(
            run_id="run_start_1",
            kind="start_task",
            status="queued",
            task_id=task_id,
        )

    monkeypatch.setattr(ui_server.runtime, "ready", True)
    monkeypatch.setattr(ui_server.runtime, "start_task", fake_start)

    payload = ui_server.StartRequest(plan_version_id="plan_123")
    result = await ui_server.start_task("task_abc", payload)

    assert calls == [("task_abc", "plan_123")]
    assert result["run_id"] == "run_start_1"
    assert result["kind"] == "start_task"


@pytest.mark.asyncio
async def test_query_endpoint_forwards_conversation_and_parent(monkeypatch):
    calls: list[tuple[str, str | None, str | None]] = []

    async def fake_start(query: str, *, conversation_id: str | None = None, parent_task_id: str | None = None):
        calls.append((query, conversation_id, parent_task_id))
        return ui_server.RunRecord(
            run_id="run_query_1",
            kind="new_query",
            status="queued",
            query=query,
        )

    monkeypatch.setattr(ui_server.runtime, "ready", True)
    monkeypatch.setattr(ui_server.runtime, "start_new_query", fake_start)

    payload = ui_server.QueryRequest(
        query="follow-up",
        conversation_id="conv_task_1",
        parent_task_id="task_1",
    )
    result = await ui_server.start_query(payload)

    assert calls == [("follow-up", "conv_task_1", "task_1")]
    assert result["run_id"] == "run_query_1"


@pytest.mark.asyncio
async def test_continue_endpoint_is_start_alias_with_deprecation_log(monkeypatch):
    start_calls: list[tuple[str, str | None]] = []
    logs: list[tuple[str, str]] = []

    async def fake_start(task_id: str, plan_version_id: str | None = None):
        start_calls.append((task_id, plan_version_id))
        return ui_server.RunRecord(
            run_id="run_continue_1",
            kind="start_task",
            status="queued",
            task_id=task_id,
        )

    async def fake_log(run_id: str, message: str):
        logs.append((run_id, message))

    monkeypatch.setattr(ui_server.runtime, "ready", True)
    monkeypatch.setattr(ui_server.runtime, "start_task", fake_start)
    monkeypatch.setattr(ui_server.runtime, "_log", fake_log)

    result = await ui_server.continue_task("task_xyz")

    assert start_calls == [("task_xyz", None)]
    assert result["run_id"] == "run_continue_1"
    assert any("/continue" in message for _, message in logs)


@pytest.mark.asyncio
async def test_feedback_and_revise_endpoints_use_feedback_path(monkeypatch):
    feedback_calls: list[tuple[str, str]] = []
    logs: list[tuple[str, str]] = []

    async def fake_feedback(task_id: str, message: str):
        feedback_calls.append((task_id, message))
        return ui_server.RunRecord(
            run_id=f"run_feedback_{len(feedback_calls)}",
            kind="feedback_task",
            status="queued",
            task_id=task_id,
            query=message,
        )

    async def fake_log(run_id: str, message: str):
        logs.append((run_id, message))

    monkeypatch.setattr(ui_server.runtime, "ready", True)
    monkeypatch.setattr(ui_server.runtime, "feedback_task", fake_feedback)
    monkeypatch.setattr(ui_server.runtime, "_log", fake_log)

    feedback_result = await ui_server.feedback_task(
        "task_one",
        ui_server.FeedbackRequest(message="prioritize stronger evidence"),
    )
    revise_result = await ui_server.revise_task(
        "task_two",
        ui_server.ReviseRequest(scope="add affiliation metadata"),
    )

    assert feedback_result["run_id"] == "run_feedback_1"
    assert revise_result["run_id"] == "run_feedback_2"
    assert feedback_calls == [
        ("task_one", "prioritize stronger evidence"),
        ("task_two", "add affiliation metadata"),
    ]
    assert any("/revise" in message for _, message in logs)


@pytest.mark.asyncio
async def test_task_detail_includes_enriched_hitl_fields(monkeypatch):
    monkeypatch.setattr(ui_server.runtime, "get_task_detail", lambda _task_id: {
        "task": {"task_id": "task_1", "objective": "x"},
        "active_plan_version": {"version_id": "plan_1"},
        "latest_plan_delta": {"summary": "added 1 step(s)"},
        "pending_feedback_queue_count": 2,
        "checkpoint_reason": "feedback_replan",
        "revisions": [],
        "report_markdown_path": None,
        "report_markdown": None,
    })

    payload = await ui_server.task_detail("task_1")

    assert payload["active_plan_version"]["version_id"] == "plan_1"
    assert payload["latest_plan_delta"]["summary"] == "added 1 step(s)"
    assert payload["pending_feedback_queue_count"] == 2
    assert payload["checkpoint_reason"] == "feedback_replan"


@pytest.mark.asyncio
async def test_conversation_detail_endpoint_returns_payload(monkeypatch):
    monkeypatch.setattr(ui_server.runtime, "get_conversation_detail", lambda _conversation_id: {
        "conversation": {"conversation_id": "conv_task_1", "iteration_count": 2},
        "iterations": [],
    })

    payload = await ui_server.conversation_detail("conv_task_1")
    assert payload["conversation"]["conversation_id"] == "conv_task_1"
    assert payload["conversation"]["iteration_count"] == 2


@pytest.mark.asyncio
async def test_export_report_pdf_endpoint_returns_file_response(monkeypatch, tmp_path):
    pdf_path = tmp_path / "task_1.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    calls: list[str] = []

    def fake_export(task_id: str):
        calls.append(task_id)
        return pdf_path

    monkeypatch.setattr(ui_server.runtime, "export_report_pdf", fake_export)

    response = await ui_server.export_report_pdf("task_1")

    assert calls == ["task_1"]
    assert Path(response.path) == pdf_path
    assert response.media_type == "application/pdf"
    assert response.filename == "task_1.pdf"


@pytest.mark.asyncio
async def test_export_report_pdf_endpoint_maps_missing_report_to_404(monkeypatch):
    def fake_export(_task_id: str):
        raise FileNotFoundError("No final report found.")

    monkeypatch.setattr(ui_server.runtime, "export_report_pdf", fake_export)

    with pytest.raises(ui_server.HTTPException) as exc:
        await ui_server.export_report_pdf("missing_task")

    assert exc.value.status_code == 404
    assert "No final report found." in str(exc.value.detail)


@pytest.mark.asyncio
async def test_export_report_pdf_endpoint_maps_generation_error_to_503(monkeypatch):
    def fake_export(_task_id: str):
        raise RuntimeError("reportlab is missing")

    monkeypatch.setattr(ui_server.runtime, "export_report_pdf", fake_export)

    with pytest.raises(ui_server.HTTPException) as exc:
        await ui_server.export_report_pdf("task_2")

    assert exc.value.status_code == 503
    assert "PDF export failed" in str(exc.value.detail)


def test_get_task_detail_report_matches_final_step_output(monkeypatch, tmp_path):
    """The UI report must be the LLM's final step output — no rewriting."""
    task = create_task("Compare LRRK2 vs GBA1")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
    final_output = (
        "## Final Report: LRRK2 vs GBA1\n\n"
        "### Recommendation\n"
        "Prioritize GBA1 based on clinical de-risking.\n\n"
        "### Rationale\n"
        "GBA1 shows stronger clinical de-risking than LRRK2.\n\n"
        "### Limitations\n"
        "Data coverage is limited to public sources."
    )
    task.steps[-1].output = final_output
    task.status = "completed"

    class DummyStore:
        def get_task(self, task_id: str):
            return task if task_id == task.task_id else None

        def list_revisions(self, _task_id: str, limit: int = 24):
            del limit
            return []

    runtime = ui_server.UiRuntime(tmp_path / "state.json")
    runtime.state_store = DummyStore()

    markdown_path = tmp_path / f"{task.task_id}.md"
    markdown_path.write_text("stale content from old run", encoding="utf-8")
    pdf_path = tmp_path / f"{task.task_id}.pdf"
    monkeypatch.setattr(runtime, "_report_markdown_path", lambda _task_id: markdown_path)
    monkeypatch.setattr(runtime, "_report_pdf_path", lambda _task_id: pdf_path)

    detail = runtime.get_task_detail(task.task_id)

    assert detail is not None
    report_markdown = str(detail["report_markdown"] or "")
    assert "Prioritize GBA1" in report_markdown
    assert "### Recommendation" in report_markdown
    assert "### Rationale" in report_markdown
    assert "### Limitations" in report_markdown


def test_append_checkpoint_event_preserves_temporal_sequence(tmp_path):
    runtime = ui_server.UiRuntime(tmp_path / "state.json")
    task = create_task("Compare LRRK2 vs GBA1")

    runtime._append_checkpoint_event(task, "quality_gap_spike")
    runtime._append_checkpoint_event(task, "quality_gap_spike")
    task.hitl_history.append("continue")
    runtime._append_checkpoint_event(task, "quality_gap_spike")
    runtime._append_checkpoint_event(task, "")
    runtime._append_checkpoint_event(task, None)
    runtime._append_checkpoint_event(task, "pre_final_after_intent_change")

    checkpoint_events = [item for item in task.hitl_history if item.startswith("checkpoint:")]
    assert checkpoint_events == [
        "checkpoint:quality_gap_spike",
        "checkpoint:quality_gap_spike",
        "checkpoint:pre_final_after_intent_change",
    ]


def test_task_summary_includes_title():
    task = create_task("Find top Parkinson's disease target hypotheses")
    summary = ui_server._task_summary(task)

    assert summary["title"]
    assert summary["task_id"] == task.task_id


def test_first_person_progress_text_normalizes_third_person():
    assert ui_server._first_person_progress_text("The agent is gathering evidence.") == "I am gathering evidence."
    assert ui_server._first_person_progress_text("Agents are executing the approved plan.") == "I am executing the approved plan."
    assert (
        ui_server._first_person_progress_text("Task is not at checkpoint; feedback queued for next adaptive gate.")
        == "I am not at checkpoint; feedback queued for next adaptive gate."
    )
