"""
Local web UI for AI Co-Scientist task workflows.

Run:
    python ui_server.py
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import traceback
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from pydantic import BaseModel, Field

from co_scientist.app import service as app_service
from co_scientist.runtime import event_orchestrator as runtime_events
from report_pdf import write_markdown_pdf
from task_state_store import TaskStateStore
from workflow import (
    WorkflowTask,
    active_plan_version,
    derive_follow_up_suggestions,
    generate_chat_title,
    render_final_report,
    render_status,
    replan_remaining_steps,
    split_report_and_next_actions,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_task_sort_key(task: WorkflowTask) -> str:
    return str(getattr(task, "updated_at", "") or "")


def _task_summary(task: WorkflowTask) -> dict:
    title = str(getattr(task, "title", "") or "").strip() or generate_chat_title(task.objective)
    return {
        "task_id": task.task_id,
        "title": title,
        "conversation_id": str(getattr(task, "conversation_id", "") or ""),
        "parent_task_id": getattr(task, "parent_task_id", None),
        "objective": task.objective,
        "user_query": str(getattr(task, "user_query", "") or task.objective),
        "status": task.status,
        "awaiting_hitl": bool(task.awaiting_hitl),
        "current_step_index": task.current_step_index,
        "step_count": len(task.steps),
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _task_detail(task: WorkflowTask) -> dict:
    payload = task.to_dict()
    payload["status_text"] = render_status(task)
    quality = app_service.evaluate_quality_gates(task)
    payload["quality_snapshot"] = {
        "passed": bool(quality.get("passed", False)),
        "unresolved_gaps": list(quality.get("unresolved_gaps", []) or [])[:8],
        "tool_call_count": int(quality.get("tool_call_count", 0) or 0),
        "evidence_count": int(quality.get("evidence_count", 0) or 0),
        "quality_confidence": str(quality.get("quality_confidence", "") or ""),
        "quality_score": float(quality.get("quality_score", 0.0) or 0.0),
    }
    # Explicit API contract fields for agentic phase/state reporting.
    payload["planner_mode"] = str(getattr(task, "planner_mode", "") or "")
    payload["phase_state"] = dict(getattr(task, "phase_state", {}) or {})
    payload["checkpoint_payload"] = dict(getattr(task, "checkpoint_payload", {}) or {})
    payload["quality_confidence"] = str(getattr(task, "quality_confidence", "") or "")
    payload["researcher_candidates"] = list(getattr(task, "researcher_candidates", []) or [])
    payload["event_log"] = list(getattr(task, "event_log", []) or [])
    return payload


def _compact_text(value: str, *, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _first_person_progress_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""

    replacements: list[tuple[str, str]] = [
        (r"\b[Tt]he agents?\b", "I"),
        (r"\b[Aa]gents?\b", "I"),
        (r"\b[Tt]he workflow\b", "I"),
        (r"\b[Tt]his workflow\b", "I"),
        (r"\b[Tt]he system\b", "I"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    text = re.sub(r"^\s*Task is\b", "I am", text)
    text = re.sub(r"^\s*Task was\b", "I was", text)
    text = re.sub(r"^\s*Task has\b", "I have", text)
    text = re.sub(r"^\s*Task will\b", "I will", text)

    grammar_fixes: list[tuple[str, str]] = [
        (r"\bI are\b", "I am"),
        (r"\bI is\b", "I am"),
        (r"\bI has\b", "I have"),
        (r"\bI does\b", "I do"),
        (r"\bI needs\b", "I need"),
        (r"\bI waits\b", "I wait"),
        (r"\bI runs\b", "I run"),
        (r"\bI opens\b", "I open"),
        (r"\bI checks\b", "I check"),
        (r"\bI builds\b", "I build"),
        (r"\bI applies\b", "I apply"),
        (r"\bI completes\b", "I complete"),
    ]
    for pattern, repl in grammar_fixes:
        text = re.sub(pattern, repl, text)
    return text


def _extract_json_payload(raw: str) -> dict | None:
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


@dataclass
class RunRecord:
    run_id: str
    kind: str
    status: str = "queued"
    task_id: str | None = None
    query: str = ""
    title: str = ""
    logs: list[dict] = field(default_factory=list)
    progress_events: list[dict] = field(default_factory=list)
    progress_summaries: list[dict] = field(default_factory=list)
    final_report: str | None = None
    follow_up_suggestions: list[str] = field(default_factory=list)
    clarification: str | None = None
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "status": self.status,
            "task_id": self.task_id,
            "query": self.query,
            "title": self.title,
            "logs": list(self.logs),
            "progress_events": list(self.progress_events),
            "progress_summaries": list(self.progress_summaries),
            "final_report": self.final_report,
            "follow_up_suggestions": list(self.follow_up_suggestions),
            "clarification": self.clarification,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=5000)
    conversation_id: str | None = Field(default=None, max_length=128)
    parent_task_id: str | None = Field(default=None, max_length=128)


class ReviseRequest(BaseModel):
    scope: str = Field(..., min_length=1, max_length=5000)


class FeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class StartRequest(BaseModel):
    plan_version_id: str | None = Field(default=None, max_length=128)


class RollbackRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=256)


class UiRuntime:
    def __init__(self, state_store_path: Path) -> None:
        self.state_store = TaskStateStore(state_store_path)
        self.ready = False
        self.ready_error: str | None = None

        self.user_id = "researcher"
        self.session_service: InMemorySessionService | None = None
        self.runner: Runner | None = None
        self.clarifier_runner: Runner | None = None
        self.feedback_parser_runner: Runner | None = None
        self.planner_runner: Runner | None = None
        self.title_summarizer_runner: Runner | None = None
        self.progress_summarizer_runner: Runner | None = None
        self.session_id: str | None = None
        self.clarifier_session_id: str | None = None
        self.feedback_parser_session_id: str | None = None
        self.planner_session_id: str | None = None
        self.title_summarizer_session_id: str | None = None
        self.progress_summarizer_session_id: str | None = None
        self.mcp_tools = None

        self.execution_lock = asyncio.Lock()
        self.runs_lock = asyncio.Lock()
        self.runs: dict[str, RunRecord] = {}
        self.active_task_run: dict[str, str] = {}
        self.runtime_feedback_queue: dict[str, list[str]] = {}
        self.run_progress_state: dict[str, dict] = {}
        self.background_tasks: set[asyncio.Task] = set()

    async def startup(self) -> None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            self.ready_error = (
                "GOOGLE_API_KEY is not configured. Update adk-agent/.env before starting the UI."
            )
            return

        try:
            self.session_service = InMemorySessionService()
            components = await app_service.create_runtime_components(
                self.session_service,
                user_id=self.user_id,
            )
            self.runner = components.runner
            self.clarifier_runner = components.clarifier_runner
            self.feedback_parser_runner = components.feedback_parser_runner
            self.planner_runner = components.planner_runner
            self.title_summarizer_runner = components.title_summarizer_runner
            self.progress_summarizer_runner = components.progress_summarizer_runner
            self.session_id = components.session_id
            self.clarifier_session_id = components.clarifier_session_id
            self.feedback_parser_session_id = components.feedback_parser_session_id
            self.planner_session_id = components.planner_session_id
            self.title_summarizer_session_id = components.title_summarizer_session_id
            self.progress_summarizer_session_id = components.progress_summarizer_session_id
            self.mcp_tools = components.mcp_tools

            if self.mcp_tools:
                tools = await self.mcp_tools.get_tools()
                print(f"[ui] Connected to MCP server ({len(tools)} tools).")
            self.ready = True
            self.ready_error = None
        except Exception as exc:
            self.ready_error = f"UI startup failed: {exc}"
            traceback.print_exc()

    async def shutdown(self) -> None:
        pending = list(self.background_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if self.mcp_tools:
            await self.mcp_tools.close()

    def _conversation_id_for_task(self, task: WorkflowTask) -> str:
        return self.state_store.resolve_conversation_id(task)

    def _build_task_research_log(self, task: WorkflowTask) -> dict:
        events = [item for item in (task.progress_events or []) if isinstance(item, dict)]
        summaries = [item for item in (task.progress_summaries or []) if isinstance(item, dict)]

        step_events = [event for event in events if str(event.get("type", "")).strip() == "step.completed"]
        step_completed_count = len(step_events)
        tool_call_count = 0
        evidence_ref_count = 0
        for event in step_events:
            metrics = event.get("metrics") or {}
            try:
                tool_call_count += int(metrics.get("tool_calls", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                evidence_ref_count += int(metrics.get("evidence_refs", 0) or 0)
            except (TypeError, ValueError):
                pass

        started_at = (
            str(events[0].get("at", "")).strip()
            if events
            else str(task.created_at or "").strip()
        )
        ended_at = ""
        if str(task.status or "") in {"completed", "failed"}:
            ended_at = str(task.updated_at or "").strip()

        return {
            "task_id": task.task_id,
            "events": events,
            "summaries": summaries,
            "stats": {
                "step_completed_count": step_completed_count,
                "tool_call_count": tool_call_count,
                "evidence_ref_count": evidence_ref_count,
                "status": str(task.status or ""),
            },
            "started_at": started_at,
            "ended_at": ended_at or None,
        }

    def _report_payload_for_task(self, task: WorkflowTask) -> dict:
        report_path = self._report_markdown_path(task.task_id)
        report_pdf_path = self._report_pdf_path(task.task_id)
        report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else None
        if task.status == "completed":
            report_text = self._ensure_current_report(task, report_text)
            report_path = self._report_markdown_path(task.task_id)
        return {
            "report_markdown_path": str(report_path) if report_path.exists() else None,
            "report_markdown": report_text,
            "report_pdf_path": str(report_pdf_path) if report_pdf_path.exists() else None,
            "has_report": bool(report_text and str(report_text).strip()),
        }

    def list_conversations(self) -> list[dict]:
        return self.state_store.list_conversations()

    def get_conversation_detail(self, conversation_id: str) -> dict | None:
        normalized = str(conversation_id or "").strip()
        if not normalized:
            return None
        tasks = self.state_store.list_tasks_in_conversation(normalized)
        if not tasks:
            return None
        tasks.sort(key=lambda task: (str(task.created_at or ""), str(task.task_id or "")))

        iterations: list[dict] = []
        for idx, task in enumerate(tasks, start=1):
            report_payload = self._report_payload_for_task(task)
            active_plan = active_plan_version(task)
            iterations.append(
                {
                    "iteration_index": idx,
                    "task": _task_detail(task),
                    "task_summary": _task_summary(task),
                    "active_plan_version": active_plan.to_dict() if active_plan else None,
                    "latest_plan_delta": task.latest_plan_delta.to_dict() if task.latest_plan_delta else None,
                    "research_log": self._build_task_research_log(task),
                    "report": report_payload,
                    "follow_up_suggestions": list(task.follow_up_suggestions or []),
                    "branch_label": str(task.branch_label or "").strip(),
                    "parent_task_id": str(task.parent_task_id or "").strip() or None,
                }
            )

        latest = max(tasks, key=lambda task: str(task.updated_at or ""))
        root = min(tasks, key=lambda task: (str(task.created_at or ""), str(task.task_id or "")))
        latest_completed = next(
            (
                item
                for item in sorted(tasks, key=lambda task: str(task.updated_at or ""), reverse=True)
                if str(item.status or "") == "completed"
            ),
            None,
        )
        selected_report_task_id = latest_completed.task_id if latest_completed else latest.task_id
        return {
            "conversation": {
                "conversation_id": normalized,
                "title": str(latest.title or root.title or "").strip() or generate_chat_title(root.objective),
                "root_task_id": root.task_id,
                "latest_task_id": latest.task_id,
                "latest_status": latest.status,
                "updated_at": latest.updated_at,
                "iteration_count": len(tasks),
                "selected_report_task_id": selected_report_task_id,
            },
            "iterations": iterations,
        }

    def list_tasks(self) -> list[dict]:
        tasks = self.state_store.list_tasks()
        tasks.sort(key=_safe_task_sort_key, reverse=True)
        return [_task_summary(task) for task in tasks]

    def get_task_detail(self, task_id: str) -> dict | None:
        task = self.state_store.get_task(task_id)
        if not task:
            return None
        revisions = self.state_store.list_revisions(task_id, limit=24)
        report_payload = self._report_payload_for_task(task)
        runtime_pending = self.runtime_feedback_queue.get(task.task_id, [])
        active_plan = active_plan_version(task)
        return {
            "task": _task_detail(task),
            "active_plan_version": active_plan.to_dict() if active_plan else None,
            "latest_plan_delta": task.latest_plan_delta.to_dict() if task.latest_plan_delta else None,
            "pending_feedback_queue_count": len(task.pending_feedback_queue) + len(runtime_pending),
            "checkpoint_reason": task.checkpoint_reason,
            "checkpoint_payload": dict(task.checkpoint_payload or {}),
            "phase_state": dict(task.phase_state or {}),
            "planner_mode": str(task.planner_mode or ""),
            "quality_confidence": str(task.quality_confidence or ""),
            "researcher_candidates": list(task.researcher_candidates or []),
            "revisions": revisions,
            "report_markdown_path": report_payload["report_markdown_path"],
            "report_markdown": report_payload["report_markdown"],
            "report_pdf_path": report_payload["report_pdf_path"],
        }

    def _ensure_current_report(self, task: WorkflowTask, report_text: str | None) -> str | None:
        quality = app_service.evaluate_quality_gates(task)
        regenerated = render_final_report(task, quality_report=quality)
        report_body, _ = split_report_and_next_actions(regenerated)
        existing = (report_text or "").strip()
        refreshed = report_body.strip()
        if existing != refreshed:
            self.write_report_markdown(task.task_id, report_body)
        return report_body

    async def create_run(self, kind: str, *, query: str = "", task_id: str | None = None) -> RunRecord:
        run_title = generate_chat_title(query) if query.strip() else ""
        run = RunRecord(
            run_id=f"run_{uuid.uuid4().hex[:10]}",
            kind=kind,
            query=query,
            title=run_title,
            task_id=task_id,
        )
        async with self.runs_lock:
            self.runs[run.run_id] = run
        self.run_progress_state[run.run_id] = {
            "last_phase": "",
            "events_since_summary": 0,
            "last_summary_mono": 0.0,
            "last_summary_event_count": 0,
        }
        return run

    async def get_run(self, run_id: str) -> dict | None:
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return None
            return run.to_dict()

    async def _update_run(self, run_id: str, **updates) -> None:
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return
            for key, value in updates.items():
                setattr(run, key, value)
            run.updated_at = _utc_now()

    async def _log(self, run_id: str, message: str) -> None:
        entry = {"at": _utc_now(), "message": message}
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return
            run.logs.append(entry)
            if len(run.logs) > 300:
                run.logs = run.logs[-300:]
            run.updated_at = _utc_now()
        print(f"[ui:{run_id}] {message}")

    def _sanitize_title(self, title: str, *, fallback: str) -> str:
        value = _compact_text(title, max_chars=72).strip(" .")
        value = re.sub(r"\s+", " ", value)
        words = value.split()
        if len(words) > 8:
            value = " ".join(words[:8]).strip()
        return value or fallback

    async def _generate_chat_title(self, query: str) -> str:
        fallback = generate_chat_title(query)
        if not self.title_summarizer_runner or not self.title_summarizer_session_id:
            return fallback
        prompt = (
            "Create a concise title for this user research request.\n"
            "Return strict JSON with {\"title\": \"...\"}.\n"
            f"Query: {query}"
        )
        try:
            raw = await app_service.run_runner_turn(
                self.title_summarizer_runner,
                self.title_summarizer_session_id,
                self.user_id,
                prompt,
            )
        except Exception:
            return fallback
        payload = _extract_json_payload(raw)
        if not payload:
            return fallback
        candidate = str(payload.get("title", "")).strip()
        return self._sanitize_title(candidate, fallback=fallback)

    async def _save_task(self, task: WorkflowTask, *, note: str, run_id: str | None = None) -> None:
        if run_id:
            async with self.runs_lock:
                run = self.runs.get(run_id)
                if run:
                    task.title = run.title or task.title or generate_chat_title(task.objective)
                    task.progress_events = list(run.progress_events[-240:])
                    task.progress_summaries = list(run.progress_summaries[-40:])
        if not task.title:
            task.title = generate_chat_title(task.objective)
        self.state_store.save_task(task, note=note)

    def _phase_for_step(self, task: WorkflowTask, step_idx: int) -> str:
        if step_idx <= 0:
            return "plan"
        if step_idx >= max(0, len(task.steps) - 1):
            return "synthesize"
        title = str(task.steps[step_idx].title or "").lower()
        if any(token in title for token in ["evidence", "search", "collect", "gather"]):
            return "search"
        if any(token in title for token in ["compare", "safety", "druggability", "genetic", "analysis"]):
            return "analyze"
        return "execute"

    def _default_next_steps(self, phase: str, *, status: str = "progress") -> list[str]:
        if status == "error":
            return ["I will handle this failure or continue with the best available evidence."]
        if phase == "intake":
            return ["I will route intent and build an initial execution plan."]
        if phase == "plan":
            return ["I will open the first checkpoint so you can start execution."]
        if phase == "checkpoint":
            return ["I will wait for your input to continue or revise the plan."]
        if phase in {"search", "analyze", "execute"}:
            return ["I will continue evidence retrieval and update quality-state gaps."]
        if phase == "synthesize":
            return ["I will finalize the recommendation and compile diagnostics."]
        if phase == "finalize":
            return ["I will persist report artifacts and return the final output."]
        return ["I will continue workflow execution."]

    def _fallback_progress_summary(
        self,
        *,
        phase: str,
        status: str,
        events_window: list[dict],
        prior_summary: dict | None,
    ) -> dict:
        completed: list[str] = []
        for event in reversed(events_window):
            if str(event.get("status", "")) != "done":
                continue
            line = _compact_text(
                _first_person_progress_text(str(event.get("human_line", "")).strip()),
                max_chars=120,
            )
            if line and line not in completed:
                completed.append(line)
            if len(completed) >= 3:
                break
        completed.reverse()

        latest_line = ""
        if events_window:
            latest_line = _compact_text(
                _first_person_progress_text(str(events_window[-1].get("human_line", "")).strip()),
                max_chars=180,
            )
        if not latest_line:
            latest_line = "I updated progress."
        headline = latest_line
        if len(headline.split()) > 12:
            headline = " ".join(headline.split()[:12])

        confidence = "medium"
        if status == "error":
            confidence = "low"
        elif events_window and str(events_window[-1].get("type", "")) == "run.completed":
            confidence = "high"
        elif prior_summary:
            confidence = str(prior_summary.get("confidence", "medium") or "medium")
            if confidence not in {"low", "medium", "high"}:
                confidence = "medium"

        return {
            "headline": headline or "Progress updated",
            "summary": latest_line,
            "completed": completed[:3],
            "next": self._default_next_steps(phase, status=status)[:2],
            "confidence": confidence,
        }

    async def _model_progress_summary(
        self,
        *,
        phase: str,
        status: str,
        events_window: list[dict],
        prior_summary: dict | None,
    ) -> dict | None:
        if not self.progress_summarizer_runner or not self.progress_summarizer_session_id:
            return None
        rendered_events = []
        for event in events_window[-12:]:
            rendered_events.append(
                {
                    "phase": event.get("phase"),
                    "type": event.get("type"),
                    "status": event.get("status"),
                    "human_line": _compact_text(str(event.get("human_line", "")), max_chars=160),
                    "metrics": event.get("metrics") or {},
                }
            )
        prompt = (
            "Generate a high-level progress summary from observable workflow events only.\n"
            "Return strict JSON with keys: headline, summary, completed, next, confidence.\n"
            "Use first-person voice as if speaking directly to the user (e.g., 'I ...').\n"
            "Never refer to yourself as 'the agent', 'the workflow', or 'the system'.\n"
            f"Current phase: {phase}\n"
            f"Status: {status}\n"
            f"Prior summary: {json.dumps(prior_summary or {}, ensure_ascii=True)}\n"
            f"Events window: {json.dumps(rendered_events, ensure_ascii=True)}"
        )
        try:
            raw = await app_service.run_runner_turn(
                self.progress_summarizer_runner,
                self.progress_summarizer_session_id,
                self.user_id,
                prompt,
            )
        except Exception:
            return None
        payload = _extract_json_payload(raw)
        if not payload:
            return None
        headline = _compact_text(
            _first_person_progress_text(str(payload.get("headline", "")).strip()),
            max_chars=72,
        )
        summary = _compact_text(
            _first_person_progress_text(str(payload.get("summary", "")).strip()),
            max_chars=220,
        )
        completed = [
            _first_person_progress_text(str(item).strip())
            for item in payload.get("completed", [])
            if str(item).strip()
        ][:3]
        nxt = [
            _first_person_progress_text(str(item).strip())
            for item in payload.get("next", [])
            if str(item).strip()
        ][:2]
        confidence = str(payload.get("confidence", "medium")).strip().lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = "medium"
        if not headline or not summary:
            return None
        return {
            "headline": headline,
            "summary": summary,
            "completed": completed,
            "next": nxt,
            "confidence": confidence,
        }

    def _should_emit_progress_summary(self, run_id: str, *, phase: str, event_type: str, status: str) -> bool:
        now_mono = time.monotonic()
        state = self.run_progress_state.setdefault(
            run_id,
            {
                "last_phase": "",
                "events_since_summary": 0,
                "last_summary_mono": 0.0,
                "last_summary_event_count": 0,
            },
        )
        previous_phase = str(state.get("last_phase", ""))
        phase_changed = bool(previous_phase and phase != previous_phase)
        state["last_phase"] = phase
        milestone_event = event_type in {
            "task.created",
            "step.completed",
            "checkpoint.opened",
            "quality.evaluated",
            "run.completed",
            "run.failed",
        }
        terminal = status == "error" or event_type in {"run.completed", "run.failed"}
        debounce_seconds = 10.0
        if not terminal and now_mono - float(state["last_summary_mono"]) < debounce_seconds:
            return False
        if terminal:
            return True
        if phase_changed:
            return True
        if milestone_event:
            return True
        if int(state["events_since_summary"]) >= 5:
            return True
        if now_mono - float(state["last_summary_mono"]) >= 25.0 and int(state["events_since_summary"]) >= 2:
            return True
        return False

    async def _append_progress_summary(
        self,
        run_id: str,
        *,
        phase: str,
        status: str,
        trigger_type: str,
    ) -> None:
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return
            state = self.run_progress_state.setdefault(
                run_id,
                {
                    "last_phase": "",
                    "events_since_summary": 0,
                    "last_summary_mono": 0.0,
                    "last_summary_event_count": 0,
                },
            )
            start_idx = int(state.get("last_summary_event_count", 0) or 0)
            events_window = list(run.progress_events[start_idx:])
            if not events_window:
                return
            prior_summary = run.progress_summaries[-1] if run.progress_summaries else None

        fallback = self._fallback_progress_summary(
            phase=phase,
            status=status,
            events_window=events_window,
            prior_summary=prior_summary,
        )
        model_summary = await self._model_progress_summary(
            phase=phase,
            status=status,
            events_window=events_window,
            prior_summary=prior_summary,
        )
        payload = model_summary or fallback
        snapshot = {
            "snapshot_id": f"snap_{uuid.uuid4().hex[:10]}",
            "at": _utc_now(),
            "phase": phase,
            "trigger_type": trigger_type,
            "headline": payload.get("headline", ""),
            "summary": payload.get("summary", ""),
            "completed": list(payload.get("completed", [])),
            "next": list(payload.get("next", [])),
            "confidence": payload.get("confidence", "medium"),
        }

        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return
            run.progress_summaries.append(snapshot)
            if len(run.progress_summaries) > 80:
                run.progress_summaries = run.progress_summaries[-80:]
            run.updated_at = _utc_now()
            state = self.run_progress_state.setdefault(
                run_id,
                {
                    "last_phase": "",
                    "events_since_summary": 0,
                    "last_summary_mono": 0.0,
                    "last_summary_event_count": 0,
                },
            )
            state["events_since_summary"] = 0
            state["last_summary_mono"] = time.monotonic()
            state["last_summary_event_count"] = len(run.progress_events)

    async def _append_progress_event(
        self,
        run_id: str,
        *,
        phase: str,
        event_type: str,
        status: str,
        human_line: str,
        task_id: str | None = None,
        step_index: int | None = None,
        step_title: str | None = None,
        tool: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        event = {
            "event_id": f"evt_{uuid.uuid4().hex[:10]}",
            "at": _utc_now(),
            "phase": phase,
            "type": event_type,
            "status": status,
            "human_line": _compact_text(_first_person_progress_text(human_line), max_chars=220),
            "task_id": task_id or "",
            "step_index": step_index,
            "step_title": step_title or "",
            "tool": tool or "",
            "metrics": metrics or {},
        }
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return
            if task_id and not run.task_id:
                run.task_id = task_id
            run.progress_events.append(event)
            if len(run.progress_events) > 600:
                run.progress_events = run.progress_events[-600:]
            if event["human_line"]:
                run.logs.append({"at": event["at"], "message": event["human_line"]})
                if len(run.logs) > 300:
                    run.logs = run.logs[-300:]
            run.updated_at = _utc_now()
            state = self.run_progress_state.setdefault(
                run_id,
                {
                    "last_phase": "",
                    "events_since_summary": 0,
                    "last_summary_mono": 0.0,
                    "last_summary_event_count": 0,
                },
            )
            state["events_since_summary"] = int(state.get("events_since_summary", 0)) + 1
        print(f"[ui:{run_id}] {event['human_line']}")

        if self._should_emit_progress_summary(run_id, phase=phase, event_type=event_type, status=status):
            await self._append_progress_summary(
                run_id,
                phase=phase,
                status=status,
                trigger_type=event_type,
            )

    def _track_background_task(self, task: asyncio.Task) -> None:
        self.background_tasks.add(task)
        task.add_done_callback(lambda done: self.background_tasks.discard(done))

    async def start_new_query(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> RunRecord:
        run = await self.create_run("new_query", query=query)
        job = asyncio.create_task(
            self._run_new_query(
                run.run_id,
                query,
                conversation_id=conversation_id,
                parent_task_id=parent_task_id,
            )
        )
        self._track_background_task(job)
        return run

    async def start_task(self, task_id: str, plan_version_id: str | None = None) -> RunRecord:
        run = await self.create_run("start_task", task_id=task_id, query=plan_version_id or "")
        job = asyncio.create_task(self._run_start_task(run.run_id, task_id, plan_version_id))
        self._track_background_task(job)
        return run

    async def continue_task(self, task_id: str) -> RunRecord:
        return await self.start_task(task_id)

    async def feedback_task(self, task_id: str, message: str) -> RunRecord:
        run = await self.create_run("feedback_task", task_id=task_id, query=message)
        job = asyncio.create_task(self._run_feedback_task(run.run_id, task_id, message))
        self._track_background_task(job)
        return run

    async def revise_task(self, task_id: str, scope: str) -> RunRecord:
        return await self.feedback_task(task_id, scope)

    def rollback_task(self, task_id: str, token: str) -> dict:
        revision_id, error_msg = app_service.resolve_rollback_revision_id(self.state_store, task_id, token)
        if error_msg or not revision_id:
            raise HTTPException(status_code=400, detail=error_msg or "Could not resolve rollback revision.")
        rolled_back = self.state_store.rollback_task(task_id, revision_id)
        if not rolled_back:
            raise HTTPException(status_code=404, detail=f"Revision {revision_id} not found for task {task_id}.")
        return {
            "task": _task_detail(rolled_back),
            "revision_id": revision_id,
        }

    def _build_branch_ancestry(self, parent_task_id: str) -> list[WorkflowTask]:
        return self.state_store.get_task_ancestry(parent_task_id)

    def _summarize_iteration_for_context(self, task: WorkflowTask) -> dict:
        report_text = ""
        report_path = self._report_markdown_path(task.task_id)
        if report_path.exists():
            try:
                report_text = report_path.read_text(encoding="utf-8")
            except OSError:
                report_text = ""
        if not report_text and task.steps:
            report_text = render_final_report(task, quality_report=app_service.evaluate_quality_gates(task))
        report_body, _ = split_report_and_next_actions(report_text)
        answer_line = _compact_text(report_body.replace("\n", " "), max_chars=320) if report_body else ""
        evidence_refs = []
        for step in task.steps:
            for ref in (step.evidence_refs or []):
                value = str(ref).strip()
                if value and value not in evidence_refs:
                    evidence_refs.append(value)
                if len(evidence_refs) >= 8:
                    break
            if len(evidence_refs) >= 8:
                break
        quality = app_service.evaluate_quality_gates(task)
        gaps = [str(item).strip() for item in quality.get("unresolved_gaps", []) if str(item).strip()][:4]
        return {
            "task_id": task.task_id,
            "query": str(task.user_query or task.objective).strip(),
            "recommendation": answer_line,
            "evidence_refs": evidence_refs,
            "gaps": gaps,
        }

    def _build_context_brief(self, ancestry: list[WorkflowTask]) -> str:
        if not ancestry:
            return ""
        lines = [
            "Prior branch context (root to selected report):",
        ]
        for idx, task in enumerate(ancestry, start=1):
            summary = self._summarize_iteration_for_context(task)
            query = _compact_text(summary.get("query", ""), max_chars=200)
            recommendation = _compact_text(summary.get("recommendation", ""), max_chars=260)
            evidence_refs = summary.get("evidence_refs", [])[:4]
            gaps = summary.get("gaps", [])[:2]
            lines.append(f"{idx}. task_id={task.task_id}")
            if query:
                lines.append(f"   - query: {query}")
            if recommendation:
                lines.append(f"   - prior finding: {recommendation}")
            if evidence_refs:
                lines.append(f"   - evidence refs: {', '.join(evidence_refs)}")
            if gaps:
                lines.append(f"   - unresolved gaps: {'; '.join(gaps)}")
        text = "\n".join(lines).strip()
        if len(text) > 3500:
            text = text[:3497].rstrip() + "..."
        return text

    def _compose_objective_with_context(self, query: str, context_brief: str) -> str:
        if not context_brief.strip():
            return query
        return (
            f"{query.strip()}\n\n"
            "Branch ancestry context (for planning/execution continuity):\n"
            f"{context_brief.strip()}\n"
            "Use this context to build on prior findings; do not repeat prior report text verbatim."
        ).strip()

    async def _run_new_query(
        self,
        run_id: str,
        query: str,
        *,
        conversation_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> None:
        await self._update_run(run_id, status="running")
        if not self.ready or not self.runner or not self.session_id:
            await self._update_run(
                run_id,
                status="failed",
                error=self.ready_error or "Runtime is not ready.",
            )
            return

        try:
            async with self.execution_lock:
                title = await self._generate_chat_title(query)
                await self._update_run(run_id, title=title)
                normalized_parent_task_id = str(parent_task_id or "").strip() or None
                normalized_conversation_id = str(conversation_id or "").strip() or None
                parent_task = None
                ancestry: list[WorkflowTask] = []
                context_brief = ""
                effective_objective = query
                effective_conversation_id: str | None = None
                branch_label = ""
                if normalized_parent_task_id:
                    parent_task = self.state_store.get_task(normalized_parent_task_id)
                    if not parent_task:
                        raise ValueError(f"Parent task {normalized_parent_task_id} not found.")
                    if str(parent_task.status or "") != "completed":
                        raise ValueError(f"Parent task {normalized_parent_task_id} is not completed yet.")
                    parent_conversation_id = self._conversation_id_for_task(parent_task)
                    if normalized_conversation_id and normalized_conversation_id != parent_conversation_id:
                        raise ValueError(
                            "conversation_id does not match parent_task_id conversation."
                        )
                    ancestry = self._build_branch_ancestry(parent_task.task_id)
                    context_brief = self._build_context_brief(ancestry)
                    effective_objective = self._compose_objective_with_context(query, context_brief)
                    effective_conversation_id = parent_conversation_id
                    branch_label = f"Branched from report {parent_task.task_id}"

                await self._append_progress_event(
                    run_id,
                    phase="intake",
                    event_type="run.started",
                    status="start",
                    human_line=f"I started this research run: {title}",
                )
                await self._append_progress_event(
                    run_id,
                    phase="intake",
                    event_type="clarification.check",
                    status="progress",
                    human_line="I'm checking whether clarification is required.",
                )
                clarification_msg = await app_service.build_clarification_request(
                    effective_objective,
                    clarifier_runner=self.clarifier_runner,
                    clarifier_session_id=self.clarifier_session_id,
                    user_id=self.user_id,
                )
                if clarification_msg:
                    await self._append_progress_event(
                        run_id,
                        phase="intake",
                        event_type="clarification.required",
                        status="done",
                        human_line="I need clarification before I can execute this plan.",
                    )
                    await self._update_run(
                        run_id,
                        status="needs_clarification",
                        clarification=clarification_msg,
                    )
                    return

                await self._append_progress_event(
                    run_id,
                    phase="plan",
                    event_type="plan.initializing",
                    status="start",
                    human_line="I'm building the initial workflow plan.",
                )

                task = await app_service.start_new_workflow_task(
                    self.runner,
                    self.session_id,
                    self.user_id,
                    self.state_store,
                    effective_objective,
                    planner_runner=self.planner_runner,
                    planner_session_id=self.planner_session_id,
                )
                task.user_query = query
                task.parent_task_id = parent_task.task_id if parent_task else None
                task.conversation_id = (
                    effective_conversation_id
                    or f"conv_{task.task_id}"
                )
                task.context_source_task_ids = [item.task_id for item in ancestry]
                task.internal_context_brief = context_brief
                task.branch_label = branch_label
                if not task.title:
                    task.title = title
                task.touch()
                await self._save_task(task, note="conversation_context_initialized_ui", run_id=run_id)
                await self._update_run(run_id, task_id=task.task_id, title=task.title or title)
                await self._append_progress_event(
                    run_id,
                    phase="plan",
                    event_type="task.created",
                    status="done",
                    human_line=f"I created task {task.task_id} with {len(task.steps)} planned steps.",
                    task_id=task.task_id,
                    metrics={"steps_total": len(task.steps)},
                )

                first_step = task.steps[0] if task.steps else None
                if first_step:
                    tool_failures = sum(
                        1
                        for entry in (first_step.tool_trace or [])
                        if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
                    )
                    await self._append_progress_event(
                        run_id,
                        phase=self._phase_for_step(task, 0),
                        event_type="step.completed",
                        status="done",
                        human_line=f"I completed: {first_step.title}.",
                        task_id=task.task_id,
                        step_index=0,
                        step_title=first_step.title,
                        metrics={
                            "steps_total": len(task.steps),
                            "steps_completed": max(task.current_step_index + 1, 1),
                            "tool_calls": len(first_step.tool_trace or []),
                            "tool_failures": tool_failures,
                            "evidence_refs": len(first_step.evidence_refs or []),
                        },
                    )
                    scope_output = (first_step.output or "").strip()
                    if scope_output:
                        await self._log(run_id, scope_output[:7000])

                status = "awaiting_hitl" if task.awaiting_hitl else "completed"
                if task.awaiting_hitl:
                    self._append_checkpoint_event(task, task.checkpoint_reason)
                    task.touch()
                    await self._append_progress_event(
                        run_id,
                        phase="checkpoint",
                        event_type="checkpoint.opened",
                        status="done",
                        human_line=f"I opened a checkpoint: {task.checkpoint_reason or 'pre_evidence_execution'}.",
                        task_id=task.task_id,
                        metrics={"checkpoint_reason": task.checkpoint_reason or "pre_evidence_execution"},
                    )
                    await self._save_task(task, note="initial_checkpoint_opened_ui", run_id=run_id)
                await self._update_run(run_id, status=status)
                if task.awaiting_hitl:
                    await self._append_progress_event(
                        run_id,
                        phase="checkpoint",
                        event_type="checkpoint.waiting",
                        status="progress",
                        human_line="I'm waiting for your start/feedback at this checkpoint.",
                        task_id=task.task_id,
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            await self._append_progress_event(
                run_id,
                phase="finalize",
                event_type="run.failed",
                status="error",
                human_line=f"I hit an execution error: {error}",
            )
            await self._update_run(run_id, status="failed", error=error)
            traceback.print_exc()

    def _drain_runtime_feedback(self, task_id: str) -> list[str]:
        queued = self.runtime_feedback_queue.get(task_id, [])
        self.runtime_feedback_queue[task_id] = []
        return [str(item).strip() for item in queued if str(item).strip()]

    def _append_checkpoint_event(self, task: WorkflowTask, reason: str | None) -> None:
        code = str(reason or "").strip().lower()
        if not code:
            return
        token = f"checkpoint:{code}"
        if task.hitl_history and str(task.hitl_history[-1]).strip().lower() == token:
            return
        task.hitl_history.append(token)

    def _revision_opportunity_used(self, task: WorkflowTask) -> bool:
        return sum(1 for item in (task.hitl_history or []) if str(item).startswith("revise:")) >= 1

    async def _apply_feedback_replan(
        self,
        run_id: str,
        task: WorkflowTask,
        feedback_text: str,
        *,
        gate_reason: str,
    ) -> None:
        if self._revision_opportunity_used(task):
            task.hitl_history.append("revision_limit_reached")
            task.awaiting_hitl = True
            task.checkpoint_state = "open"
            task.checkpoint_reason = "feedback_replan_limit_reached"
            task.touch()
            await self._append_progress_event(
                run_id,
                phase="plan",
                event_type="feedback.rejected",
                status="done",
                human_line="I already used the one allowed plan revision for this task.",
                task_id=task.task_id,
            )
            await self._save_task(task, note="feedback_replan_limit_reached_ui", run_id=run_id)
            return

        if not task.base_objective:
            task.base_objective = task.objective
        revision_intent = await app_service.parse_revision_intent(
            feedback_text,
            feedback_parser_runner=self.feedback_parser_runner,
            feedback_parser_session_id=self.feedback_parser_session_id,
            user_id=self.user_id,
        )
        prior_version = active_plan_version(task)
        merged_intent = app_service.merge_revision_intents(
            prior_version.revision_intent if prior_version else None,
            revision_intent,
        )
        revised_objective = app_service.merge_objective_with_revision_intent(
            task.base_objective or task.objective,
            merged_intent,
        )
        model_plan_graph = await app_service.draft_model_plan_graph(
            revised_objective,
            planner_runner=self.planner_runner,
            planner_session_id=self.planner_session_id,
            user_id=self.user_id,
        )
        plan_version, delta = replan_remaining_steps(
            task,
            revised_objective=revised_objective,
            revision_intent=merged_intent,
            gate_reason=gate_reason,
            plan_graph_override=model_plan_graph,
        )
        task.hitl_history.append(f"revise:{feedback_text}")
        task.pending_feedback_queue = []
        task.awaiting_hitl = True
        task.checkpoint_state = "open"
        task.checkpoint_reason = gate_reason
        self._append_checkpoint_event(task, gate_reason)
        task.status = "in_progress"
        task.touch()
        await self._append_progress_event(
            run_id,
            phase="plan",
            event_type="plan.replanned",
            status="done",
            human_line=f"I updated the plan from your feedback: {delta.summary}.",
            task_id=task.task_id,
            metrics={
                "added_steps": len(delta.added_steps),
                "modified_steps": len(delta.modified_steps),
                "removed_steps": len(delta.removed_steps),
                "reordered_steps": len(delta.reordered_steps),
            },
        )
        await self._append_progress_event(
            run_id,
            phase="checkpoint",
            event_type="checkpoint.opened",
            status="done",
            human_line=f"I opened a checkpoint after applying your feedback: {gate_reason}.",
            task_id=task.task_id,
            metrics={"checkpoint_reason": gate_reason},
        )
        await self._save_task(task, note="feedback_replan_ui", run_id=run_id)

        await self._log(run_id, "Updated plan after feedback.")
        await self._log(run_id, f"Plan delta: {delta.summary}")
        if plan_version.steps:
            await self._log(
                run_id,
                "Remaining plan:\n" + "\n".join(
                    f"{idx + 1}. {step.title}" for idx, step in enumerate(plan_version.steps)
                ),
            )

    async def _execute_until_next_gate_or_completion(
        self,
        run_id: str,
        task: WorkflowTask,
        *,
        bypass_first_gate: bool,
    ) -> tuple[str, dict | None]:
        quality_state: dict = {}
        first_gate_check = True
        while task.current_step_index + 1 < len(task.steps):
            next_idx = task.current_step_index + 1
            next_step = task.steps[next_idx]
            queued_feedback = [*task.pending_feedback_queue, *self.runtime_feedback_queue.get(task.task_id, [])]
            open_gate, gate_reason = app_service.should_open_checkpoint(
                task,
                next_step,
                quality_state,
                queued_feedback,
            )
            if open_gate:
                if bypass_first_gate and first_gate_check and gate_reason == "pre_evidence_execution":
                    first_gate_check = False
                else:
                    if queued_feedback:
                        merged_feedback = "\n".join([str(item).strip() for item in queued_feedback if str(item).strip()])
                        task.pending_feedback_queue = []
                        self._drain_runtime_feedback(task.task_id)
                        await self._append_progress_event(
                            run_id,
                            phase="checkpoint",
                            event_type="feedback.queued_applied",
                            status="progress",
                            human_line="I'm applying your queued feedback at this adaptive checkpoint.",
                            task_id=task.task_id,
                            metrics={"queued_feedback_count": len(queued_feedback)},
                        )
                        await self._apply_feedback_replan(
                            run_id,
                            task,
                            merged_feedback,
                            gate_reason="queued_feedback_pending",
                        )
                        return "awaiting_hitl", None

                    task.awaiting_hitl = True
                    task.checkpoint_state = "open"
                    task.checkpoint_reason = gate_reason
                    self._append_checkpoint_event(task, gate_reason)
                    task.touch()
                    await self._append_progress_event(
                        run_id,
                        phase="checkpoint",
                        event_type="checkpoint.opened",
                        status="done",
                        human_line=f"I opened an adaptive checkpoint: {gate_reason}.",
                        task_id=task.task_id,
                        metrics={"checkpoint_reason": gate_reason},
                    )
                    await self._save_task(task, note="adaptive_hitl_checkpoint_opened_ui", run_id=run_id)
                    return "awaiting_hitl", None

            first_gate_check = False
            step_text = await app_service.execute_step(
                self.runner,
                self.session_id,
                self.user_id,
                task,
                next_idx,
            )
            await self._log(run_id, step_text[:4000])
            step = task.steps[next_idx]
            tool_failures = sum(
                1
                for entry in (step.tool_trace or [])
                if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
            )
            await self._append_progress_event(
                run_id,
                phase=self._phase_for_step(task, next_idx),
                event_type="step.completed",
                status="done",
                human_line=f"I completed: {step.title}.",
                task_id=task.task_id,
                step_index=next_idx,
                step_title=step.title,
                metrics={
                    "steps_total": len(task.steps),
                    "steps_completed": next_idx + 1,
                    "tool_calls": len(step.tool_trace or []),
                    "tool_failures": tool_failures,
                    "evidence_refs": len(step.evidence_refs or []),
                },
            )
            await self._save_task(task, note=f"step_{next_idx + 1}_completed_ui", run_id=run_id)

            base_quality = app_service.evaluate_quality_gates(task)
            quality_state = {
                "unresolved_gaps": base_quality.get("unresolved_gaps", []),
                "last_step_failures": tool_failures,
                "last_step_output": step.output,
            }

        final_quality = app_service.evaluate_quality_gates(task)
        await self._append_progress_event(
            run_id,
            phase="finalize",
            event_type="quality.evaluated",
            status="done",
            human_line=(
                "I evaluated quality: "
                f"{final_quality.get('tool_call_count', 0)} tool calls, "
                f"{final_quality.get('evidence_count', 0)} evidence IDs."
            ),
            task_id=task.task_id,
            metrics={
                "tool_call_count": final_quality.get("tool_call_count", 0),
                "evidence_count": final_quality.get("evidence_count", 0),
                "unresolved_gaps": len(final_quality.get("unresolved_gaps", []) or []),
            },
        )
        return "completed", final_quality

    async def _run_start_task(self, run_id: str, task_id: str, plan_version_id: str | None) -> None:
        await self._update_run(run_id, status="running")
        if not self.ready or not self.runner or not self.session_id:
            await self._update_run(
                run_id,
                status="failed",
                error=self.ready_error or "Runtime is not ready.",
            )
            return

        try:
            async with self.execution_lock:
                task = self.state_store.get_task(task_id)
                if not task:
                    await self._update_run(run_id, status="failed", error=f"Task {task_id} not found.")
                    return
                await self._update_run(run_id, task_id=task.task_id, title=task.title or generate_chat_title(task.objective))
                if task.status == "completed":
                    await self._update_run(run_id, status="failed", error=f"Task {task_id} is already completed.")
                    return
                if not task.awaiting_hitl:
                    await self._update_run(run_id, status="failed", error=f"Task {task_id} is not at a checkpoint.")
                    return
                if plan_version_id and task.active_plan_version_id and plan_version_id != task.active_plan_version_id:
                    await self._update_run(
                        run_id,
                        status="failed",
                        error=(
                            f"Plan version mismatch. active={task.active_plan_version_id}, "
                            f"requested={plan_version_id}."
                        ),
                    )
                    return

                self.active_task_run[task_id] = run_id
                await self._append_progress_event(
                    run_id,
                    phase="checkpoint",
                    event_type="checkpoint.resumed",
                    status="start",
                    human_line=f"I resumed execution for task {task_id}.",
                    task_id=task_id,
                )
                ack_token = app_service.gate_ack_token(task.checkpoint_reason, task.active_plan_version_id)
                if ack_token and ack_token not in task.hitl_history:
                    task.hitl_history.append(ack_token)
                task.hitl_history.append("continue")
                runtime_events.append_event(
                    task,
                    runtime_events.EVENT_CHECKPOINT_APPROVED,
                    reason=task.checkpoint_reason,
                    payload=dict(task.checkpoint_payload or {}),
                )
                task.awaiting_hitl = False
                task.checkpoint_state = "closed"
                task.checkpoint_reason = ""
                task.checkpoint_payload = {}
                task.touch()
                await self._save_task(task, note="hitl_start_ui", run_id=run_id)
                await self._append_progress_event(
                    run_id,
                    phase="execute",
                    event_type="execution.running",
                    status="progress",
                    human_line="I'm running from this checkpoint until completion or the next adaptive gate.",
                    task_id=task_id,
                )

                terminal_status, quality = await self._execute_until_next_gate_or_completion(
                    run_id,
                    task,
                    bypass_first_gate=True,
                )

                if terminal_status == "awaiting_hitl":
                    task.status = "in_progress"
                    task.touch()
                    await self._save_task(task, note="adaptive_checkpoint_waiting_ui", run_id=run_id)
                    await self._update_run(run_id, status="awaiting_hitl", task_id=task.task_id)
                    await self._append_progress_event(
                        run_id,
                        phase="checkpoint",
                        event_type="checkpoint.waiting",
                        status="done",
                        human_line=f"I opened a checkpoint: {task.checkpoint_reason or 'adaptive_gate'}.",
                        task_id=task.task_id,
                        metrics={"checkpoint_reason": task.checkpoint_reason or "adaptive_gate"},
                    )
                    return

                quality = quality or app_service.evaluate_quality_gates(task)
                task.status = "completed"
                task.awaiting_hitl = False
                task.checkpoint_state = "closed"
                task.checkpoint_reason = ""
                task.checkpoint_payload = {}

                await self._append_progress_event(
                    run_id,
                    phase="finalize",
                    event_type="quality.evaluated",
                    status="done",
                    human_line=(
                        "I completed quality gates: "
                        f"evidence={quality.get('evidence_count', 0)}, "
                        f"coverage={quality.get('coverage_ratio', 0):.2f}, "
                        f"tools={quality.get('tool_call_count', 0)}."
                    ),
                    task_id=task.task_id,
                    metrics={
                        "evidence_count": quality.get("evidence_count", 0),
                        "coverage_ratio": quality.get("coverage_ratio", 0.0),
                        "tool_call_count": quality.get("tool_call_count", 0),
                        "unresolved_gaps": len(quality.get("unresolved_gaps", []) or []),
                    },
                )

                final_report_raw = render_final_report(task, quality_report=quality)
                final_report_body, extracted_suggestions = split_report_and_next_actions(final_report_raw)
                follow_up_suggestions = extracted_suggestions or derive_follow_up_suggestions(
                    final_report_raw,
                    quality_report=quality,
                )
                task.follow_up_suggestions = [str(item).strip() for item in follow_up_suggestions if str(item).strip()][:5]
                task.touch()
                await self._save_task(task, note="workflow_completed_ui", run_id=run_id)
                self.write_report_markdown(task.task_id, final_report_body)
                await self._update_run(
                    run_id,
                    status="completed",
                    task_id=task.task_id,
                    final_report=final_report_body,
                    follow_up_suggestions=task.follow_up_suggestions,
                )
                await self._append_progress_event(
                    run_id,
                    phase="finalize",
                    event_type="run.completed",
                    status="done",
                    human_line="I completed the report.",
                    task_id=task.task_id,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            await self._append_progress_event(
                run_id,
                phase="finalize",
                event_type="run.failed",
                status="error",
                human_line=f"I hit an execution error: {error}",
                task_id=task_id,
            )
            await self._update_run(run_id, status="failed", error=error)
            traceback.print_exc()
        finally:
            self.active_task_run.pop(task_id, None)

    async def _run_feedback_task(self, run_id: str, task_id: str, message: str) -> None:
        await self._update_run(run_id, status="running")
        if not self.ready or not self.runner or not self.session_id:
            await self._update_run(
                run_id,
                status="failed",
                error=self.ready_error or "Runtime is not ready.",
            )
            return

        try:
            active_run = self.active_task_run.get(task_id)
            if active_run:
                queue = self.runtime_feedback_queue.setdefault(task_id, [])
                queue.append(message)
                await self._append_progress_event(
                    run_id,
                    phase="checkpoint",
                    event_type="feedback.queued",
                    status="done",
                    human_line="I queued your feedback and will apply it at the next checkpoint.",
                    task_id=task_id,
                    metrics={"queued_feedback_count": len(queue)},
                )
                await self._update_run(run_id, status="queued", task_id=task_id)
                return

            async with self.execution_lock:
                task = self.state_store.get_task(task_id)
                if not task:
                    await self._update_run(run_id, status="failed", error=f"Task {task_id} not found.")
                    return
                await self._update_run(run_id, task_id=task.task_id, title=task.title or generate_chat_title(task.objective))
                if task.status == "completed":
                    await self._update_run(run_id, status="failed", error=f"Task {task_id} is already completed.")
                    return
                if not task.awaiting_hitl:
                    if self._revision_opportunity_used(task):
                        await self._append_progress_event(
                            run_id,
                            phase="plan",
                            event_type="feedback.rejected",
                            status="done",
                            human_line="I already used the one allowed plan revision for this task.",
                            task_id=task.task_id,
                        )
                        await self._update_run(run_id, status="completed", task_id=task.task_id)
                        return
                    task.pending_feedback_queue.append(message)
                    task.touch()
                    await self._save_task(task, note="feedback_queued_pending_ui", run_id=run_id)
                    await self._append_progress_event(
                        run_id,
                        phase="checkpoint",
                        event_type="feedback.queued",
                        status="done",
                        human_line="I'm not at a checkpoint yet, so I queued your feedback for the next adaptive gate.",
                        task_id=task.task_id,
                        metrics={"queued_feedback_count": len(task.pending_feedback_queue)},
                    )
                    await self._update_run(run_id, status="queued", task_id=task_id)
                    return

                await self._append_progress_event(
                    run_id,
                    phase="plan",
                    event_type="feedback.applying",
                    status="start",
                    human_line=f"I'm applying your feedback to task {task_id}.",
                    task_id=task.task_id,
                )
                await self._apply_feedback_replan(
                    run_id,
                    task,
                    message,
                    gate_reason="feedback_replan",
                )
                await self._update_run(run_id, status="awaiting_hitl", task_id=task.task_id)
                await self._append_progress_event(
                    run_id,
                    phase="checkpoint",
                    event_type="feedback.applied",
                    status="done",
                    human_line="I incorporated your feedback. The updated plan is ready at checkpoint.",
                    task_id=task.task_id,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            await self._append_progress_event(
                run_id,
                phase="finalize",
                event_type="run.failed",
                status="error",
                human_line=f"I hit an execution error: {error}",
                task_id=task_id,
            )
            await self._update_run(run_id, status="failed", error=error)
            traceback.print_exc()

    async def _run_continue_task(self, run_id: str, task_id: str) -> None:
        await self._run_start_task(run_id, task_id, None)

    async def _run_revise_task(self, run_id: str, task_id: str, scope: str) -> None:
        await self._run_feedback_task(run_id, task_id, scope)

    def _safe_report_task_id(self, task_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)

    def _report_markdown_path(self, task_id: str) -> Path:
        safe_task_id = self._safe_report_task_id(task_id)
        return Path(__file__).resolve().parent / "reports" / f"{safe_task_id}.md"

    def _report_pdf_path(self, task_id: str) -> Path:
        safe_task_id = self._safe_report_task_id(task_id)
        return Path(__file__).resolve().parent / "reports" / f"{safe_task_id}.pdf"

    def write_report_markdown(self, task_id: str, report_markdown: str) -> Path:
        report_path = self._report_markdown_path(task_id)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        normalized = (report_markdown or "").rstrip() + "\n"
        report_path.write_text(normalized, encoding="utf-8")

        # Invalidate stale export so PDF is generated on explicit user request.
        report_pdf_path = self._report_pdf_path(task_id)
        if report_pdf_path.exists():
            report_pdf_path.unlink()
        return report_path

    def export_report_pdf(self, task_id: str) -> Path:
        task = self.state_store.get_task(task_id)
        if not task:
            raise FileNotFoundError(f"Task {task_id} not found.")

        report_path = self._report_markdown_path(task_id)
        if not report_path.exists():
            raise FileNotFoundError(
                f"No final report found for task {task_id}. Complete the workflow first."
            )

        report_pdf_path = self._report_pdf_path(task_id)
        markdown = report_path.read_text(encoding="utf-8")
        error = write_markdown_pdf(
            markdown,
            report_pdf_path,
            title=f"Workflow Report ({task.task_id})",
        )
        if error:
            raise RuntimeError(error)
        return report_pdf_path


ROOT_DIR = Path(__file__).resolve().parent
UI_DIR = ROOT_DIR / "ui"
STATE_PATH = ROOT_DIR / "state" / "workflow_tasks.json"

runtime = UiRuntime(STATE_PATH)
app = FastAPI(title="AI Co-Scientist UI", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


@app.on_event("startup")
async def _startup() -> None:
    await runtime.startup()
    if runtime.ready:
        print("[ui] Server ready at http://127.0.0.1:8080")
    else:
        print(f"[ui] Startup warning: {runtime.ready_error}")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await runtime.shutdown()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": runtime.ready,
        "busy": runtime.execution_lock.locked(),
        "error": runtime.ready_error,
    }


@app.get("/api/tasks")
async def list_tasks() -> dict:
    return {"tasks": runtime.list_tasks()}


@app.get("/api/conversations")
async def list_conversations() -> dict:
    return {"conversations": runtime.list_conversations()}


@app.get("/api/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str) -> dict:
    detail = runtime.get_conversation_detail(conversation_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found.")
    return detail


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str) -> dict:
    detail = runtime.get_task_detail(task_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return detail


@app.get("/api/tasks/{task_id}/report.pdf")
async def export_report_pdf(task_id: str) -> FileResponse:
    normalized_task_id = task_id.strip()
    try:
        pdf_path = await asyncio.to_thread(runtime.export_report_pdf, normalized_task_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"PDF export failed: {exc}")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{runtime._safe_report_task_id(normalized_task_id)}.pdf",
    )


@app.post("/api/query")
async def start_query(payload: QueryRequest) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    run = await runtime.start_new_query(
        payload.query.strip(),
        conversation_id=payload.conversation_id.strip() if payload.conversation_id else None,
        parent_task_id=payload.parent_task_id.strip() if payload.parent_task_id else None,
    )
    return run.to_dict()


@app.post("/api/tasks/{task_id}/continue")
async def continue_task(task_id: str) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    run = await runtime.start_task(task_id.strip())
    await runtime._log(run.run_id, "Deprecated endpoint used: /continue. Prefer /start.")
    return run.to_dict()


@app.post("/api/tasks/{task_id}/revise")
async def revise_task(task_id: str, payload: ReviseRequest) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    run = await runtime.feedback_task(task_id.strip(), payload.scope.strip())
    await runtime._log(run.run_id, "Deprecated endpoint used: /revise. Prefer /feedback.")
    return run.to_dict()


@app.post("/api/tasks/{task_id}/start")
async def start_task(task_id: str, payload: StartRequest | None = None) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    plan_version_id = payload.plan_version_id.strip() if payload and payload.plan_version_id else None
    run = await runtime.start_task(task_id.strip(), plan_version_id=plan_version_id)
    return run.to_dict()


@app.post("/api/tasks/{task_id}/feedback")
async def feedback_task(task_id: str, payload: FeedbackRequest) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    run = await runtime.feedback_task(task_id.strip(), payload.message.strip())
    return run.to_dict()


@app.post("/api/tasks/{task_id}/rollback")
async def rollback_task(task_id: str, payload: RollbackRequest) -> dict:
    return runtime.rollback_task(task_id.strip(), payload.token.strip())


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    payload = await runtime.get_run(run_id.strip())
    if not payload:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return payload


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("CO_SCI_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("CO_SCI_UI_PORT", "8080"))
    uvicorn.run("ui_server:app", host=host, port=port, reload=False)
