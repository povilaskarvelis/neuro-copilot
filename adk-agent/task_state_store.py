"""
Lightweight JSON-backed persistence for workflow tasks.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

from workflow import WorkflowTask


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

    def save_task(self, task: WorkflowTask, note: str | None = None) -> None:
        payload = self._read_payload()
        task_payload = task.to_dict()
        self._upsert_task_payload(payload, task_payload)
        self._append_history_snapshot(payload, task_payload, note=note)
        self._write_payload(payload)

    def get_task(self, task_id: str) -> WorkflowTask | None:
        payload = self._read_payload()
        for item in payload["tasks"]:
            if item.get("task_id") == task_id:
                return WorkflowTask.from_dict(item)
        return None

    def list_tasks(self) -> list[WorkflowTask]:
        payload = self._read_payload()
        return [WorkflowTask.from_dict(item) for item in payload["tasks"]]

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
