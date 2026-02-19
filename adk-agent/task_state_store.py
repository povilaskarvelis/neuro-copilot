"""
Lightweight JSON-backed persistence for workflow tasks.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

from workflow import WorkflowTask, generate_chat_title


MAX_TASK_HISTORY_REVISIONS = 120


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStateStore:
    """Persist workflow tasks to a local JSON file."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_payload(self) -> dict:
        if not self.file_path.exists():
            return {"tasks": [], "task_histories": {}}
        raw = self.file_path.read_text(encoding="utf-8").strip()
        if not raw:
            return {"tasks": [], "task_histories": {}}
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                return {"tasks": [], "task_histories": {}}
            if "tasks" not in payload or not isinstance(payload["tasks"], list):
                payload["tasks"] = []
            if "task_histories" not in payload or not isinstance(payload["task_histories"], dict):
                payload["task_histories"] = {}
            return payload
        except json.JSONDecodeError:
            # If the file is corrupted, preserve it but avoid crashing sessions.
            return {"tasks": [], "task_histories": {}}

    def _write_payload(self, payload: dict) -> None:
        self.file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _upsert_task_payload(self, payload: dict, task_payload: dict) -> None:
        tasks = payload["tasks"]
        existing_index = next(
            (idx for idx, item in enumerate(tasks) if item.get("task_id") == task_payload.get("task_id")),
            None,
        )
        if existing_index is None:
            tasks.append(task_payload)
        else:
            tasks[existing_index] = task_payload

    def _append_history_snapshot(self, payload: dict, task_payload: dict, note: str | None = None) -> None:
        task_id = str(task_payload.get("task_id", "")).strip()
        if not task_id:
            return
        histories = payload["task_histories"]
        task_history = histories.setdefault(task_id, [])
        normalized_note = (note or "").strip()

        if task_history:
            latest = task_history[-1]
            if latest.get("task") == task_payload and latest.get("note", "") == normalized_note:
                return

        task_history.append(
            {
                "revision_id": f"rev_{uuid.uuid4().hex[:10]}",
                "saved_at": _utc_now(),
                "note": normalized_note,
                "task": task_payload,
            }
        )
        if len(task_history) > MAX_TASK_HISTORY_REVISIONS:
            histories[task_id] = task_history[-MAX_TASK_HISTORY_REVISIONS:]

    def resolve_conversation_id(self, task: WorkflowTask) -> str:
        conversation_id = str(getattr(task, "conversation_id", "") or "").strip()
        if conversation_id:
            return conversation_id
        task_id = str(getattr(task, "task_id", "") or "").strip()
        if not task_id:
            return "conv_unknown"
        return f"conv_{task_id}"

    def save_task(self, task: WorkflowTask, note: str | None = None) -> None:
        task.conversation_id = self.resolve_conversation_id(task)
        if not str(task.user_query or "").strip():
            task.user_query = str(task.objective or "").strip()
        payload = self._read_payload()
        task_payload = task.to_dict()
        self._upsert_task_payload(payload, task_payload)
        self._append_history_snapshot(payload, task_payload, note=note)
        self._write_payload(payload)

    def get_task(self, task_id: str) -> WorkflowTask | None:
        payload = self._read_payload()
        for item in payload["tasks"]:
            if item.get("task_id") == task_id:
                task = WorkflowTask.from_dict(item)
                if not str(task.conversation_id or "").strip():
                    task.conversation_id = self.resolve_conversation_id(task)
                if not str(task.user_query or "").strip():
                    task.user_query = str(task.objective or "").strip()
                return task
        return None

    def list_tasks(self) -> list[WorkflowTask]:
        payload = self._read_payload()
        tasks: list[WorkflowTask] = []
        for item in payload["tasks"]:
            task = WorkflowTask.from_dict(item)
            if not str(task.conversation_id or "").strip():
                task.conversation_id = self.resolve_conversation_id(task)
            if not str(task.user_query or "").strip():
                task.user_query = str(task.objective or "").strip()
            tasks.append(task)
        return tasks

    def latest_task(self) -> WorkflowTask | None:
        tasks = self.list_tasks()
        if not tasks:
            return None
        tasks.sort(key=lambda task: task.updated_at, reverse=True)
        return tasks[0]

    def list_revisions(self, task_id: str, limit: int = 20) -> list[dict]:
        payload = self._read_payload()
        task_history = payload["task_histories"].get(task_id, [])
        revisions: list[dict] = []
        for entry in reversed(task_history):
            task_payload = entry.get("task", {})
            revisions.append(
                {
                    "revision_id": entry.get("revision_id", ""),
                    "saved_at": entry.get("saved_at", ""),
                    "note": entry.get("note", ""),
                    "status": task_payload.get("status", "unknown"),
                    "current_step_index": task_payload.get("current_step_index", -1),
                    "awaiting_hitl": bool(task_payload.get("awaiting_hitl", False)),
                    "active_plan_version_id": task_payload.get("active_plan_version_id"),
                    "checkpoint_reason": task_payload.get("checkpoint_reason", ""),
                }
            )
            if len(revisions) >= max(1, limit):
                break
        return revisions

    def rollback_task(self, task_id: str, revision_id: str) -> WorkflowTask | None:
        payload = self._read_payload()
        task_history = payload["task_histories"].get(task_id, [])
        revision = next((entry for entry in task_history if entry.get("revision_id") == revision_id), None)
        if not revision:
            return None

        restored_payload = revision.get("task")
        if not isinstance(restored_payload, dict):
            return None

        self._upsert_task_payload(payload, restored_payload)
        self._append_history_snapshot(
            payload,
            restored_payload,
            note=f"rollback_to:{revision_id}",
        )
        self._write_payload(payload)
        return WorkflowTask.from_dict(restored_payload)

    def list_tasks_in_conversation(self, conversation_id: str) -> list[WorkflowTask]:
        normalized = str(conversation_id or "").strip()
        if not normalized:
            return []
        tasks = [task for task in self.list_tasks() if self.resolve_conversation_id(task) == normalized]
        tasks.sort(key=lambda task: (str(task.created_at or ""), str(task.task_id or "")))
        return tasks

    def list_children(self, task_id: str) -> list[WorkflowTask]:
        normalized = str(task_id or "").strip()
        if not normalized:
            return []
        children = [task for task in self.list_tasks() if str(task.parent_task_id or "").strip() == normalized]
        children.sort(key=lambda task: (str(task.created_at or ""), str(task.task_id or "")))
        return children

    def get_task_ancestry(self, task_id: str) -> list[WorkflowTask]:
        normalized = str(task_id or "").strip()
        if not normalized:
            return []
        tasks_by_id = {task.task_id: task for task in self.list_tasks()}
        current = tasks_by_id.get(normalized)
        if current is None:
            return []
        ancestry_rev: list[WorkflowTask] = []
        seen: set[str] = set()
        conversation_id = self.resolve_conversation_id(current)
        while current and current.task_id not in seen:
            seen.add(current.task_id)
            ancestry_rev.append(current)
            parent_id = str(current.parent_task_id or "").strip()
            if not parent_id:
                break
            parent = tasks_by_id.get(parent_id)
            if parent is None:
                break
            if self.resolve_conversation_id(parent) != conversation_id:
                break
            current = parent
        return list(reversed(ancestry_rev))

    def list_conversations(self) -> list[dict]:
        tasks = self.list_tasks()
        grouped: dict[str, list[WorkflowTask]] = {}
        for task in tasks:
            conversation_id = self.resolve_conversation_id(task)
            grouped.setdefault(conversation_id, []).append(task)

        summaries: list[dict] = []
        for conversation_id, items in grouped.items():
            items.sort(key=lambda task: (str(task.updated_at or ""), str(task.task_id or "")), reverse=True)
            latest = items[0]
            root = min(items, key=lambda task: (str(task.created_at or ""), str(task.task_id or "")))
            summaries.append(
                {
                    "conversation_id": conversation_id,
                    "title": str(latest.title or root.title or "").strip() or generate_chat_title(root.objective),
                    "latest_task_id": latest.task_id,
                    "latest_status": latest.status,
                    "updated_at": latest.updated_at,
                    "iteration_count": len(items),
                    "root_task_id": root.task_id,
                }
            )
        summaries.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
        return summaries
