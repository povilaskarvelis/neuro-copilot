from pathlib import Path

import ui_server
from task_state_store import TaskStateStore
from workflow import create_task, derive_follow_up_suggestions, split_report_and_next_actions


def _mark_task_completed(task, output: str) -> None:
    for step in task.steps:
        step.status = "completed"
        step.output = "step"
    task.steps[-1].output = output
    task.status = "completed"


def test_split_report_and_next_actions_extracts_suggestions() -> None:
    markdown = """
## Recommendation
Prioritize Target A.

## Limitations
- Gap 1

## Next Actions
1. Validate with an external dataset.
2. Run a safety-focused follow-up.
"""
    body, actions = split_report_and_next_actions(markdown)

    assert "Next Actions" not in body
    assert "Prioritize Target A" in body
    assert actions == [
        "Validate with an external dataset",
        "Run a safety-focused follow-up",
    ]


def test_follow_up_suggestions_fallback_from_quality_gaps() -> None:
    markdown = "## Recommendation\nPrioritize Target A."
    suggestions = derive_follow_up_suggestions(
        markdown,
        quality_report={"unresolved_gaps": ["Need stronger genetics evidence"]},
    )

    assert suggestions
    assert "Need stronger genetics evidence" in suggestions[0]


def test_conversation_detail_has_separate_research_logs(tmp_path: Path, monkeypatch) -> None:
    store = TaskStateStore(tmp_path / "workflow_tasks.json")

    root = create_task("Root question")
    root.conversation_id = f"conv_{root.task_id}"
    root.user_query = "Root question"
    root.progress_events = [
        {
            "type": "step.completed",
            "metrics": {"tool_calls": 2, "evidence_refs": 1},
            "at": "2026-02-19T10:00:00+00:00",
        }
    ]
    root.progress_summaries = [{"summary": "root summary"}]
    _mark_task_completed(root, "## Recommendation\nRoot recommendation")
    store.save_task(root, note="root")

    child = create_task("Follow-up")
    child.conversation_id = root.conversation_id
    child.parent_task_id = root.task_id
    child.user_query = "Follow-up"
    child.progress_events = [
        {
            "type": "step.completed",
            "metrics": {"tool_calls": 5, "evidence_refs": 3},
            "at": "2026-02-19T11:00:00+00:00",
        }
    ]
    child.progress_summaries = [{"summary": "child summary"}]
    _mark_task_completed(child, "## Recommendation\nChild recommendation")
    store.save_task(child, note="child")

    runtime = ui_server.UiRuntime(tmp_path / "state.json")
    runtime.state_store = store

    monkeypatch.setattr(runtime, "_report_markdown_path", lambda task_id: tmp_path / f"{task_id}.md")
    monkeypatch.setattr(runtime, "_report_pdf_path", lambda task_id: tmp_path / f"{task_id}.pdf")

    detail = runtime.get_conversation_detail(root.conversation_id)

    assert detail is not None
    iterations = detail["iterations"]
    assert len(iterations) == 2
    assert iterations[0]["task"]["task_id"] == root.task_id
    assert iterations[1]["task"]["task_id"] == child.task_id

    root_log = iterations[0]["research_log"]
    child_log = iterations[1]["research_log"]
    assert root_log["stats"]["tool_call_count"] == 2
    assert child_log["stats"]["tool_call_count"] == 5
    assert len(root_log["events"]) == 1
    assert len(child_log["events"]) == 1
    assert root_log["events"][0]["at"] != child_log["events"][0]["at"]
