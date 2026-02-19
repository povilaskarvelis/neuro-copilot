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
    assert task.conversation_id == ""
    assert task.parent_task_id is None
    assert task.user_query == "legacy objective"
    assert task.follow_up_suggestions == []
    assert task.context_source_task_ids == []
    assert task.branch_label == ""
    assert task.internal_context_brief == ""


def test_state_store_conversation_helpers_support_legacy_and_branching(tmp_path: Path) -> None:
    store = TaskStateStore(tmp_path / "workflow_tasks.json")
    root = create_task("Root request")
    root.user_query = "Root request"
    store.save_task(root, note="root")

    child = create_task("Follow-up request")
    child.user_query = "Follow-up request"
    child.conversation_id = f"conv_{root.task_id}"
    child.parent_task_id = root.task_id
    store.save_task(child, note="child")

    branch = create_task("Branch request")
    branch.user_query = "Branch request"
    branch.conversation_id = f"conv_{root.task_id}"
    branch.parent_task_id = root.task_id
    store.save_task(branch, note="branch")

    conversations = store.list_conversations()
    assert len(conversations) == 1
    conv = conversations[0]
    assert conv["conversation_id"] == f"conv_{root.task_id}"
    assert conv["iteration_count"] == 3

    task_ids = [task.task_id for task in store.list_tasks_in_conversation(f"conv_{root.task_id}")]
    assert root.task_id in task_ids and child.task_id in task_ids and branch.task_id in task_ids

    ancestry = store.get_task_ancestry(child.task_id)
    assert [task.task_id for task in ancestry] == [root.task_id, child.task_id]

    children = store.list_children(root.task_id)
    assert sorted(task.task_id for task in children) == sorted([child.task_id, branch.task_id])
