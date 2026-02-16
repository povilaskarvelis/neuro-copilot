from pathlib import Path

from task_state_store import TaskStateStore
from workflow import WorkflowTask, create_task


def test_save_creates_revision_history(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "workflow_tasks.json")
    task = create_task("Find IPF experts")

    store.save_task(task, note="created")
    task.status = "in_progress"
    task.current_step_index = 0
    store.save_task(task, note="step_1_started")

    revisions = store.list_revisions(task.task_id, limit=10)
    assert len(revisions) == 2
    assert revisions[0]["note"] == "step_1_started"
    assert revisions[1]["note"] == "created"
    assert "active_plan_version_id" in revisions[0]
    assert "checkpoint_reason" in revisions[0]


def test_rollback_restores_prior_revision(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "workflow_tasks.json")
    task = create_task("Assess ATP13A2 target risk")

    store.save_task(task, note="created")
    task.status = "in_progress"
    task.current_step_index = 1
    store.save_task(task, note="step_2_completed")
    task.status = "completed"
    store.save_task(task, note="workflow_completed")

    revisions = store.list_revisions(task.task_id, limit=10)
    target_revision_id = revisions[1]["revision_id"]  # step_2_completed snapshot
    restored = store.rollback_task(task.task_id, target_revision_id)

    assert restored is not None
    assert restored.status == "in_progress"
    assert restored.current_step_index == 1
    assert store.latest_task() is not None
    assert store.latest_task().status == "in_progress"


def test_workflow_task_from_dict_sets_new_hitl_defaults() -> None:
    legacy_payload = {
        "task_id": "task_legacy",
        "objective": "legacy objective",
        "status": "pending",
        "steps": [],
        "current_step_index": -1,
        "awaiting_hitl": False,
        "hitl_history": [],
    }

    task = WorkflowTask.from_dict(legacy_payload)

    assert task.base_objective == "legacy objective"
    assert task.plan_versions == []
    assert task.active_plan_version_id is None
    assert task.pending_feedback_queue == []
    assert task.latest_plan_delta is None
    assert task.checkpoint_state == "closed"
    assert task.title
    assert task.progress_events == []
    assert task.progress_summaries == []
