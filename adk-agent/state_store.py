from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Protocol
import uuid

logger = logging.getLogger(__name__)

WORKFLOW_SNAPSHOT_SCHEMA = "workflow_session_state.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SupportsWorkflowStateStore(Protocol):
    def save_task(self, task: dict[str, Any], *, owner_ip: str = "", flush: bool = True) -> None: ...
    def get_task(self, task_id: str) -> dict[str, Any] | None: ...
    def list_conversations(self, *, owner_ip: str = "") -> list[dict[str, Any]]: ...
    def conversation_owned_by(self, conversation_id: str, owner_ip: str) -> bool: ...
    def get_conversation_tasks(self, conversation_id: str) -> list[dict[str, Any]]: ...
    def save_run(self, run: dict[str, Any], *, flush: bool = False) -> None: ...
    def get_run(self, run_id: str) -> dict[str, Any] | None: ...
    def mark_incomplete_runs_failed(self, reason: str) -> int: ...
    def save_workflow_session(
        self,
        conversation_id: str,
        *,
        task_id: str = "",
        state: dict[str, Any] | None = None,
    ) -> None: ...
    def get_workflow_session(self, conversation_id: str) -> dict[str, Any] | None: ...


class JsonTaskStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {
            "conversations": {},
            "tasks": {},
            "runs": {},
            "workflow_sessions": {},
        }
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load JSON state store from %s", self.path, exc_info=True)
                return
            if isinstance(loaded, dict):
                self._data["conversations"] = dict(loaded.get("conversations") or {})
                self._data["tasks"] = dict(loaded.get("tasks") or {})
                self._data["runs"] = dict(loaded.get("runs") or {})
                self._data["workflow_sessions"] = dict(loaded.get("workflow_sessions") or {})

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    def has_any_data(self) -> bool:
        return bool(self._data.get("conversations") or self._data.get("tasks") or self._data.get("runs"))

    def save_task(self, task: dict[str, Any], *, owner_ip: str = "", flush: bool = True) -> None:
        stored_task = copy.deepcopy(task)
        stored_task["updated_at"] = _utc_now()
        self._data["tasks"][stored_task["task_id"]] = stored_task
        conv_id = str(stored_task.get("conversation_id", "") or "").strip()
        if conv_id:
            conv = self._data["conversations"].setdefault(
                conv_id,
                {
                    "conversation_id": conv_id,
                    "title": stored_task.get("title", ""),
                    "task_ids": [],
                    "owner_ip": owner_ip,
                    "created_at": stored_task.get("created_at", _utc_now()),
                    "updated_at": _utc_now(),
                },
            )
            if stored_task["task_id"] not in conv["task_ids"]:
                conv["task_ids"].append(stored_task["task_id"])
            conv["updated_at"] = _utc_now()
            conv["title"] = stored_task.get("title") or conv.get("title", "")
            if owner_ip:
                conv["owner_ip"] = owner_ip
        if flush:
            self._save()

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = self._data["tasks"].get(task_id)
        return copy.deepcopy(task) if isinstance(task, dict) else None

    def list_conversations(self, *, owner_ip: str = "") -> list[dict[str, Any]]:
        result = []
        for conv in self._data["conversations"].values():
            if owner_ip and conv.get("owner_ip", "") != owner_ip:
                continue
            task_ids = conv.get("task_ids", [])
            tasks = [self._data["tasks"].get(tid) for tid in task_ids]
            tasks = [t for t in tasks if isinstance(t, dict)]
            latest = max(tasks, key=lambda t: t.get("updated_at", "")) if tasks else None
            result.append(
                {
                    "conversation_id": conv["conversation_id"],
                    "title": conv.get("title", "Research"),
                    "latest_status": latest["status"] if latest else "unknown",
                    "updated_at": conv.get("updated_at", ""),
                    "iteration_count": len(tasks),
                }
            )
        result.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
        return result

    def conversation_owned_by(self, conversation_id: str, owner_ip: str) -> bool:
        conv = self._data["conversations"].get(conversation_id)
        if not isinstance(conv, dict):
            return False
        return conv.get("owner_ip", "") == owner_ip

    def get_conversation_tasks(self, conversation_id: str) -> list[dict[str, Any]]:
        conv = self._data["conversations"].get(conversation_id)
        if not isinstance(conv, dict):
            return []
        tasks = [self._data["tasks"].get(tid) for tid in conv.get("task_ids", [])]
        return [copy.deepcopy(task) for task in tasks if isinstance(task, dict)]

    def save_run(self, run: dict[str, Any], *, flush: bool = False) -> None:
        stored_run = copy.deepcopy(run)
        stored_run["updated_at"] = _utc_now()
        self._data["runs"][stored_run["run_id"]] = stored_run
        if flush:
            self._save()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self._data["runs"].get(run_id)
        return copy.deepcopy(run) if isinstance(run, dict) else None

    def mark_incomplete_runs_failed(self, reason: str) -> int:
        updated = 0
        for run_id, run in list(self._data["runs"].items()):
            if not isinstance(run, dict):
                continue
            if _interrupt_run_payload(run, reason):
                self._data["runs"][run_id] = run
                updated += 1
        if updated:
            self._save()
        return updated

    def save_workflow_session(
        self,
        conversation_id: str,
        *,
        task_id: str = "",
        state: dict[str, Any] | None = None,
    ) -> None:
        if not conversation_id:
            return
        if not state:
            self._data["workflow_sessions"].pop(conversation_id, None)
            self._save()
            return
        self._data["workflow_sessions"][conversation_id] = {
            "conversation_id": conversation_id,
            "task_id": task_id or "",
            "schema_version": WORKFLOW_SNAPSHOT_SCHEMA,
            "state": copy.deepcopy(state),
            "updated_at": _utc_now(),
        }
        self._save()

    def get_workflow_session(self, conversation_id: str) -> dict[str, Any] | None:
        payload = self._data["workflow_sessions"].get(conversation_id)
        return copy.deepcopy(payload) if isinstance(payload, dict) else None


def _require_psycopg() -> tuple[Any, Any, Any]:
    try:
        import psycopg  # type: ignore[import-not-found]
        from psycopg.rows import dict_row  # type: ignore[import-not-found]
        from psycopg.types.json import Jsonb  # type: ignore[import-not-found]
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "Postgres persistence requires `psycopg` to be installed. "
            "Add `psycopg[binary]` to the environment before setting NEURO_COPILOT_POSTGRES_DSN."
        ) from exc
    return psycopg, dict_row, Jsonb


class PostgresTaskStore:
    def __init__(self, dsn: str, *, legacy_state_path: Path | None = None) -> None:
        self.dsn = str(dsn or "").strip()
        if not self.dsn:
            raise RuntimeError("PostgresTaskStore requires a non-empty DSN.")
        self.legacy_state_path = legacy_state_path
        self._ensure_schema()
        self._maybe_import_legacy_json()

    def _connect(self):
        psycopg, dict_row, _ = _require_psycopg()
        return psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            owner_ip TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            parent_task_id TEXT NULL,
            title TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            task_json JSONB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_tasks_conversation_created
            ON tasks (conversation_id, created_at ASC, task_id ASC);

        CREATE INDEX IF NOT EXISTS idx_tasks_conversation_updated
            ON tasks (conversation_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS workflow_sessions (
            conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            task_id TEXT NULL,
            schema_version TEXT NOT NULL DEFAULT 'workflow_session_state.v1',
            session_state_json JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            task_id TEXT NULL,
            kind TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            run_json JSONB NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_runs_task_updated
            ON runs (task_id, updated_at DESC);
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(ddl)

    def _table_has_rows(self, table_name: str) -> bool:
        query = f"SELECT EXISTS (SELECT 1 FROM {table_name} LIMIT 1) AS present"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone() or {}
            return bool(row.get("present"))

    def _maybe_import_legacy_json(self) -> None:
        if self._table_has_rows("conversations") or self._table_has_rows("tasks"):
            return
        if self.legacy_state_path is None or not self.legacy_state_path.exists():
            return
        legacy = JsonTaskStore(self.legacy_state_path)
        if not legacy.has_any_data():
            return

        logger.info("Importing legacy JSON task state from %s into Postgres", self.legacy_state_path)
        for conversation in legacy.list_conversations():
            conversation_id = str(conversation.get("conversation_id", "") or "").strip()
            if not conversation_id:
                continue
            tasks = legacy.get_conversation_tasks(conversation_id)
            owner_ip = ""
            raw_conv = legacy._data.get("conversations", {}).get(conversation_id, {})
            if isinstance(raw_conv, dict):
                owner_ip = str(raw_conv.get("owner_ip", "") or "").strip()
            for task in tasks:
                self.save_task(task, owner_ip=owner_ip)
            workflow_session = legacy.get_workflow_session(conversation_id)
            if workflow_session:
                self.save_workflow_session(
                    conversation_id,
                    task_id=str(workflow_session.get("task_id", "") or "").strip(),
                    state=workflow_session.get("state") if isinstance(workflow_session.get("state"), dict) else {},
                )
        for run in (legacy._data.get("runs") or {}).values():
            if isinstance(run, dict):
                self.save_run(run)

    def _upsert_conversation(
        self,
        *,
        conversation_id: str,
        title: str,
        owner_ip: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        if not conversation_id:
            return
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (conversation_id, title, owner_ip, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (conversation_id) DO UPDATE SET
                    title = CASE
                        WHEN EXCLUDED.title <> '' THEN EXCLUDED.title
                        ELSE conversations.title
                    END,
                    owner_ip = CASE
                        WHEN EXCLUDED.owner_ip <> '' THEN EXCLUDED.owner_ip
                        ELSE conversations.owner_ip
                    END,
                    updated_at = GREATEST(conversations.updated_at, EXCLUDED.updated_at)
                """,
                (conversation_id, title, owner_ip, created_at, updated_at),
            )

    def save_task(self, task: dict[str, Any], *, owner_ip: str = "", flush: bool = True) -> None:
        _, _, Jsonb = _require_psycopg()
        stored_task = copy.deepcopy(task)
        stored_task["updated_at"] = _utc_now()
        conversation_id = str(stored_task.get("conversation_id", "") or "").strip()
        self._upsert_conversation(
            conversation_id=conversation_id,
            title=str(stored_task.get("title", "") or ""),
            owner_ip=owner_ip,
            created_at=str(stored_task.get("created_at", "") or _utc_now()),
            updated_at=str(stored_task.get("updated_at", "") or _utc_now()),
        )
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tasks (
                    task_id, conversation_id, parent_task_id, title, status, created_at, updated_at, task_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    conversation_id = EXCLUDED.conversation_id,
                    parent_task_id = EXCLUDED.parent_task_id,
                    title = EXCLUDED.title,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    task_json = EXCLUDED.task_json
                """,
                (
                    stored_task["task_id"],
                    conversation_id,
                    stored_task.get("parent_task_id"),
                    str(stored_task.get("title", "") or ""),
                    str(stored_task.get("status", "") or ""),
                    str(stored_task.get("created_at", "") or _utc_now()),
                    str(stored_task.get("updated_at", "") or _utc_now()),
                    Jsonb(stored_task),
                ),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT task_json FROM tasks WHERE task_id = %s", (task_id,))
            row = cur.fetchone()
        if not row:
            return None
        payload = row.get("task_json")
        return copy.deepcopy(payload) if isinstance(payload, dict) else None

    def list_conversations(self, *, owner_ip: str = "") -> list[dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    c.conversation_id,
                    c.title,
                    c.updated_at,
                    COALESCE(task_counts.iteration_count, 0) AS iteration_count,
                    COALESCE(latest_task.status, 'unknown') AS latest_status
                FROM conversations c
                LEFT JOIN LATERAL (
                    SELECT COUNT(*) AS iteration_count
                    FROM tasks t
                    WHERE t.conversation_id = c.conversation_id
                ) AS task_counts ON TRUE
                LEFT JOIN LATERAL (
                    SELECT status
                    FROM tasks t
                    WHERE t.conversation_id = c.conversation_id
                    ORDER BY t.updated_at DESC, t.task_id DESC
                    LIMIT 1
                ) AS latest_task ON TRUE
                WHERE (%s = '' OR c.owner_ip = %s)
                ORDER BY c.updated_at DESC, c.conversation_id DESC
                """,
                (owner_ip, owner_ip),
            )
            rows = cur.fetchall() or []
        return [
            {
                "conversation_id": row["conversation_id"],
                "title": row.get("title", "Research"),
                "latest_status": row.get("latest_status", "unknown"),
                "updated_at": row.get("updated_at").isoformat() if getattr(row.get("updated_at"), "isoformat", None) else str(row.get("updated_at", "")),
                "iteration_count": int(row.get("iteration_count", 0) or 0),
            }
            for row in rows
        ]

    def conversation_owned_by(self, conversation_id: str, owner_ip: str) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT owner_ip FROM conversations WHERE conversation_id = %s",
                (conversation_id,),
            )
            row = cur.fetchone()
        if not row:
            return False
        return str(row.get("owner_ip", "") or "") == owner_ip

    def get_conversation_tasks(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_json
                FROM tasks
                WHERE conversation_id = %s
                ORDER BY created_at ASC, task_id ASC
                """,
                (conversation_id,),
            )
            rows = cur.fetchall() or []
        tasks: list[dict[str, Any]] = []
        for row in rows:
            payload = row.get("task_json")
            if isinstance(payload, dict):
                tasks.append(copy.deepcopy(payload))
        return tasks

    def save_run(self, run: dict[str, Any], *, flush: bool = False) -> None:
        _, _, Jsonb = _require_psycopg()
        stored_run = copy.deepcopy(run)
        stored_run["updated_at"] = _utc_now()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    run_id, task_id, kind, status, created_at, updated_at, run_json
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE SET
                    task_id = EXCLUDED.task_id,
                    kind = EXCLUDED.kind,
                    status = EXCLUDED.status,
                    created_at = EXCLUDED.created_at,
                    updated_at = EXCLUDED.updated_at,
                    run_json = EXCLUDED.run_json
                """,
                (
                    stored_run["run_id"],
                    stored_run.get("task_id"),
                    str(stored_run.get("kind", "") or ""),
                    str(stored_run.get("status", "") or ""),
                    str(stored_run.get("created_at", "") or _utc_now()),
                    str(stored_run.get("updated_at", "") or _utc_now()),
                    Jsonb(stored_run),
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT run_json FROM runs WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
        if not row:
            return None
        payload = row.get("run_json")
        return copy.deepcopy(payload) if isinstance(payload, dict) else None

    def mark_incomplete_runs_failed(self, reason: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT run_json
                FROM runs
                WHERE status IN ('queued', 'running', 'in_progress')
                """
            )
            rows = cur.fetchall() or []
        updated = 0
        for row in rows:
            payload = row.get("run_json")
            if not isinstance(payload, dict):
                continue
            if _interrupt_run_payload(payload, reason):
                self.save_run(payload)
                updated += 1
        return updated

    def save_workflow_session(
        self,
        conversation_id: str,
        *,
        task_id: str = "",
        state: dict[str, Any] | None = None,
    ) -> None:
        _, _, Jsonb = _require_psycopg()
        if not conversation_id:
            return
        self._upsert_conversation(
            conversation_id=conversation_id,
            title="",
            owner_ip="",
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        with self._connect() as conn, conn.cursor() as cur:
            if not state:
                cur.execute(
                    "DELETE FROM workflow_sessions WHERE conversation_id = %s",
                    (conversation_id,),
                )
                return
            cur.execute(
                """
                INSERT INTO workflow_sessions (
                    conversation_id, task_id, schema_version, session_state_json, updated_at
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (conversation_id) DO UPDATE SET
                    task_id = EXCLUDED.task_id,
                    schema_version = EXCLUDED.schema_version,
                    session_state_json = EXCLUDED.session_state_json,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    conversation_id,
                    task_id or "",
                    WORKFLOW_SNAPSHOT_SCHEMA,
                    Jsonb(copy.deepcopy(state)),
                    _utc_now(),
                ),
            )

    def get_workflow_session(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT conversation_id, task_id, schema_version, session_state_json, updated_at
                FROM workflow_sessions
                WHERE conversation_id = %s
                """,
                (conversation_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        updated_at = row.get("updated_at")
        return {
            "conversation_id": row.get("conversation_id", conversation_id),
            "task_id": row.get("task_id", ""),
            "schema_version": row.get("schema_version", WORKFLOW_SNAPSHOT_SCHEMA),
            "state": copy.deepcopy(row.get("session_state_json", {})) if isinstance(row.get("session_state_json"), dict) else {},
            "updated_at": updated_at.isoformat() if getattr(updated_at, "isoformat", None) else str(updated_at or ""),
        }


def create_state_store(state_store_path: Path) -> SupportsWorkflowStateStore:
    postgres_dsn = str(
        os.getenv("NEURO_COPILOT_POSTGRES_DSN")
        or os.getenv("AI_CO_SCIENTIST_POSTGRES_DSN")
        or os.getenv("POSTGRES_DSN")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()
    if postgres_dsn:
        return PostgresTaskStore(postgres_dsn, legacy_state_path=state_store_path)
    return JsonTaskStore(state_store_path)


def _interrupt_run_payload(run: dict[str, Any], reason: str) -> bool:
    status = str(run.get("status", "") or "").strip()
    if status not in {"queued", "running", "in_progress"}:
        return False

    timestamp = _utc_now()
    human_line = str(reason or "Run interrupted before completion.").strip()
    event = {
        "event_id": f"evt_{uuid.uuid4().hex[:10]}",
        "at": timestamp,
        "phase": "finalize",
        "type": "run.interrupted",
        "status": "error",
        "human_line": human_line,
        "task_id": str(run.get("task_id", "") or ""),
        "step_index": None,
        "step_title": "",
        "tool": "",
        "metrics": {"reason": "server_restart"},
    }

    run["status"] = "failed"
    run["error"] = human_line
    run["updated_at"] = timestamp

    progress_events = list(run.get("progress_events") or [])
    progress_events.append(event)
    run["progress_events"] = progress_events[-600:]

    logs = list(run.get("logs") or [])
    logs.append({"at": timestamp, "message": human_line})
    run["logs"] = logs[-300:]
    return True
