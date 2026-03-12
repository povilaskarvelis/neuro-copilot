"""
Web UI for AI Co-Scientist (adapted to ADK-native workflow).

Run:
    python ui_server.py
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import threading
import time
import traceback
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from pydantic import BaseModel, Field

from agent import validate_runtime_configuration
from state_store import SupportsWorkflowStateStore, create_state_store
from report_pdf import write_markdown_pdf
from co_scientist.tool_registry import TOOL_SOURCE_NAMES
from co_scientist.workflow import (
    STATE_EXECUTOR_ACTIVE_STEP_ID,
    STATE_PRIOR_RESEARCH,
    STATE_WORKFLOW_TASK,
    _describe_tool_call,
    STATE_PLAN_PENDING_APPROVAL,
    _is_continue_execution_command,
    create_workflow_agent,
    _resolve_source_label,
)

load_dotenv()
logger = logging.getLogger(__name__)

RATE_LIMIT_QUERIES = int(os.environ.get("RATE_LIMIT_QUERIES", "20"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW", "3600"))
GA4_MEASUREMENT_ID = os.environ.get("GA4_MEASUREMENT_ID", "").strip()
_GA4_ID_PATTERN = re.compile(r"^G-[A-Z0-9]+$")


class RateLimiter:
    """In-memory sliding-window rate limiter keyed by IP address."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> tuple[bool, int]:
        now = time.time()
        hits = self._hits[key]
        self._hits[key] = hits = [t for t in hits if now - t < self.window]
        if len(hits) >= self.max_requests:
            retry_after = int(self.window - (now - hits[0])) + 1
            return False, retry_after
        hits.append(now)
        return True, 0

    def remaining(self, key: str) -> int:
        now = time.time()
        hits = [t for t in self._hits.get(key, []) if now - t < self.window]
        return max(0, self.max_requests - len(hits))


query_limiter = RateLimiter(RATE_LIMIT_QUERIES, RATE_LIMIT_WINDOW)


def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    ip = _client_ip(request)
    allowed, retry_after = query_limiter.check(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. You can run {RATE_LIMIT_QUERIES} queries per hour. Try again in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_text(value: str, *, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


_TRANSIENT_WORKFLOW_RESPONSE_PATTERNS = (
    re.compile(r"^_?\s*rate limit hit\s+[—-]\s+waited\s+\d+s,\s+retrying[.…_ ]*$", re.IGNORECASE),
)

PERSISTED_SESSION_STATE_KEYS = (
    STATE_WORKFLOW_TASK,
    STATE_PRIOR_RESEARCH,
    STATE_PLAN_PENDING_APPROVAL,
)


def _is_transient_workflow_response(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    return any(pattern.match(normalized) for pattern in _TRANSIENT_WORKFLOW_RESPONSE_PATTERNS)


def _parse_step_event_text(text: str) -> dict:
    """Extract structured info from step_executor rendered markdown."""
    info: dict = {}
    normalized = str(text or "").strip()
    if not normalized:
        return info

    # If the payload contains multiple rendered steps, use the latest one.
    step_matches = list(re.finditer(r"^###?\s+S\d+\s*[·\-]\s*`?\w+`?", normalized, flags=re.MULTILINE))
    if step_matches:
        normalized = normalized[step_matches[-1].start():].strip()

    # Step ID and status: "### S1 · `completed`"
    m = re.search(r"###?\s+(S\d+)\s*[·\-]\s*`?(\w+)`?", normalized)
    if m:
        info["step_id"] = m.group(1)
        info["status"] = m.group(2)
    # Goal: "**Goal:** ..."
    m = re.search(r"\*\*Goal:\*\*\s*(.+?)(?:\n|$)", normalized)
    if m:
        info["goal"] = m.group(1).strip()
    # Findings (supports "Key Findings", "Detailed Findings", "Findings")
    m = re.search(r"\*\*(?:Key |Detailed )?Findings\*\*\s*\n+([\s\S]*?)(?=\n\*\*|\n_Progress|\n---|\Z)", normalized)
    if m:
        info["findings"] = re.sub(r"\s+", " ", m.group(1)).strip()[:300]
    # Tools used: "**Tools used:** `tool1`, `tool2`"
    m = re.search(r"\*\*Tools used:\*\*\s*(.+?)(?:\n|$)", normalized)
    if m:
        info["tools"] = re.findall(r"`([^`]+)`", m.group(1))
    # Evidence IDs
    evidence = re.findall(r"`((?:PMID|DOI|NCT|PMC|UniProt|PubChem|PDB|ChEMBL|Reactome)[:\s][^`]+|(?:GCST|CHEMBL|rs)\S[^`]*)`", normalized)
    if evidence:
        info["evidence"] = evidence[:10]
    # Progress: "_Progress: 2/5 steps complete..."
    m = re.search(r"_Progress:\s*(.+?)_", normalized)
    if m:
        info["progress"] = m.group(1).strip()
    # ReAct trace block
    m = re.search(r"\*\*(?:ReAct Trace|Tool Trace)\*\*\s*\n+([\s\S]*?)(?=\n\*\*|\n_Progress|\n---|\Z)", normalized)
    if m:
        block = m.group(1)
        cleaned_lines: list[str] = []
        phase_map: dict[str, str] = {}
        phase_pattern = re.compile(r"\*\*(Reason|Act|Observe|Conclude):\*\*\s*(.+)", flags=re.IGNORECASE)
        for raw_line in block.splitlines():
            line = re.sub(r"^\s*>\s?", "", raw_line).strip()
            if not line:
                continue
            cleaned_lines.append(line)
            phase_match = phase_pattern.match(line)
            if phase_match:
                phase_map[phase_match.group(1).lower()] = phase_match.group(2).strip()
        if cleaned_lines:
            info["react_trace"] = "\n".join(cleaned_lines)
        if phase_map:
            info["react_phases"] = phase_map
    return info


def _build_step_completed_event_metrics(text: str) -> dict | None:
    """Return structured metrics only for actual rendered step-completion blocks."""
    step_info = _parse_step_event_text(text)
    step_id = str(step_info.get("step_id", "") or "").strip()
    if not step_id:
        return None
    return {
        "step_id": step_id,
        "step_status": str(step_info.get("status", "completed") or "completed").strip(),
        "goal": str(step_info.get("goal", "") or "").strip(),
        "findings": str(step_info.get("findings", "") or "").strip(),
        "tools": list(step_info.get("tools", []) or []),
        "evidence": list(step_info.get("evidence", []) or []),
        "progress": str(step_info.get("progress", "") or "").strip(),
        "react_trace": str(step_info.get("react_trace", "") or "").strip(),
        "react_phases": step_info.get("react_phases", {}) if isinstance(step_info.get("react_phases", {}), dict) else {},
        "rendered_step_markdown": text,
    }


def _extract_tool_error_metrics(function_response) -> dict | None:
    """Extract a user-visible tool error payload from a function response."""
    if not function_response:
        return None
    tool_name = str(getattr(function_response, "name", "") or "").strip()
    response = getattr(function_response, "response", None) or {}
    if not isinstance(response, dict) or not bool(response.get("error")):
        return None
    message = str(response.get("message", "") or "").strip()
    suggestion = str(response.get("suggestion", "") or "").strip()
    error_type = str(response.get("error_type", "") or "").strip()
    if not tool_name and not message:
        return None
    return {
        "tool": tool_name,
        "error_type": error_type,
        "message": message,
        "suggestion": suggestion,
    }




def _generate_chat_title(query: str) -> str:
    words = re.sub(r"\s+", " ", str(query or "")).strip().split()
    if len(words) <= 8:
        return " ".join(words) or "Research"
    return " ".join(words[:8]).rstrip(".,;:!?")


def _extract_persistable_session_state(session_state: dict | None) -> dict:
    persisted: dict[str, object] = {}
    if not isinstance(session_state, dict):
        return persisted
    for key in PERSISTED_SESSION_STATE_KEYS:
        if key in session_state and session_state[key] is not None:
            persisted[key] = copy.deepcopy(session_state[key])
    return persisted


# ---------------------------------------------------------------------------
# Conversation session: ADK runner + session per conversation
# ---------------------------------------------------------------------------

@dataclass
class ConversationSession:
    runner: Runner
    session_id: str
    app_name: str
    mcp_tools: object | None


# ---------------------------------------------------------------------------
# RunRecord: tracks background execution progress
# ---------------------------------------------------------------------------

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


class FeedbackRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class ReviseRequest(BaseModel):
    scope: str = Field(..., min_length=1, max_length=5000)


class StartRequest(BaseModel):
    plan_version_id: str | None = Field(default=None, max_length=128)


class RollbackRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=256)


# ---------------------------------------------------------------------------
# Workflow task helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_id: str,
    objective: str,
    conversation_id: str,
    *,
    title: str = "",
    user_query: str = "",
    parent_task_id: str | None = None,
) -> dict:
    now = _utc_now()
    return {
        "task_id": task_id,
        "title": title or _generate_chat_title(objective),
        "conversation_id": conversation_id,
        "parent_task_id": parent_task_id,
        "objective": objective,
        "user_query": user_query or objective,
        "status": "in_progress",
        "awaiting_hitl": False,
        "current_step_index": 0,
        "steps": [],
        "hitl_history": [],
        "follow_up_suggestions": [],
        "branch_label": "",
        "created_at": now,
        "updated_at": now,
        "progress_events": [],
        "progress_summaries": [],
    }


def _steps_from_workflow_state(wf_state: dict | None) -> list[dict]:
    if not wf_state:
        return []
    def _source_label(tool_name: str) -> str:
        return _resolve_source_label(tool_name or "")

    return [
        {
            "title": step.get("goal", f"Step {step.get('id', '?')}"),
            "instruction": (
                f"Potential source: {_source_label(step.get('tool_hint', ''))}. "
                f"Done when: {str(step.get('completion_condition', '')).strip()}"
            ),
            "status": step.get("status", "pending"),
            "id": step.get("id", ""),
            "tool_hint": str(step.get("tool_hint", "")).strip(),
            "source": _source_label(step.get("tool_hint", "")),
            "completion_condition": str(step.get("completion_condition", "")).strip(),
            "result_summary": step.get("result_summary", ""),
            "executor_prose": step.get("executor_prose", ""),
            "tool_reasoning": step.get("tool_reasoning", ""),
            "evidence_refs": step.get("evidence_ids", []),
            "tool_trace": [
                {"tool": t} for t in step.get("tools_called", [])
            ],
            "output": step.get("executor_prose", "") or step.get("result_summary", ""),
        }
        for step in wf_state.get("steps", [])
    ]


def _normalize_steps_for_ui(steps: list[dict] | None) -> list[dict]:
    """Ensure plan steps are user-facing (database names, not tool ids)."""
    normalized: list[dict] = []
    for raw_step in (steps or []):
        step = dict(raw_step or {})
        tool_hint = str(step.get("tool_hint", "")).strip()
        source = str(step.get("source", "")).strip()
        completion = str(step.get("completion_condition", "")).strip()
        instruction = str(step.get("instruction", "")).strip()

        if not source and tool_hint:
            source = _resolve_source_label(tool_hint)

        if not source and instruction:
            tool_match = re.search(r"Tool:\s*([^.\n]+)", instruction)
            if tool_match:
                inferred_tool = tool_match.group(1).strip()
                source = _resolve_source_label(inferred_tool)
                tool_hint = tool_hint or inferred_tool

        if not completion and instruction:
            done_match = re.search(r"Done when:\s*(.+)$", instruction)
            if done_match:
                completion = done_match.group(1).strip()

        if source and completion:
            step["instruction"] = f"Potential source: {source}. Done when: {completion}"
        elif source:
            step["instruction"] = f"Potential source: {source}."
        elif completion:
            step["instruction"] = f"Done when: {completion}"

        if tool_hint:
            step["tool_hint"] = tool_hint
        if source:
            step["source"] = source
        if completion:
            step["completion_condition"] = completion
        normalized.append(step)
    return normalized


def _task_summary(task: dict) -> dict:
    return {
        "task_id": task["task_id"],
        "title": task.get("title", ""),
        "conversation_id": task.get("conversation_id", ""),
        "parent_task_id": task.get("parent_task_id"),
        "objective": task.get("objective", ""),
        "user_query": task.get("user_query", task.get("objective", "")),
        "status": task.get("status", ""),
        "awaiting_hitl": bool(task.get("awaiting_hitl")),
        "current_step_index": task.get("current_step_index", 0),
        "step_count": len(task.get("steps", [])),
        "created_at": task.get("created_at", ""),
        "updated_at": task.get("updated_at", ""),
    }


def _task_detail(task: dict) -> dict:
    return {
        **task,
        "status_text": f"Status: {task.get('status', 'unknown')}",
        "quality_snapshot": {
            "passed": task.get("status") == "completed",
            "unresolved_gaps": [],
            "tool_call_count": 0,
            "evidence_count": 0,
            "quality_confidence": "",
            "quality_score": 0.0,
        },
        "planner_mode": "",
        "phase_state": {},
        "checkpoint_payload": {},
        "quality_confidence": "",
        "researcher_candidates": [],
        "event_log": [],
    }


def _iteration_from_task(task: dict, idx: int = 1) -> dict:
    steps = _normalize_steps_for_ui(task.get("steps", []))
    active_plan = {
        "version_id": f"plan_{task['task_id']}",
        "steps": steps,
    } if steps else None

    is_direct = bool(task.get("is_direct_response"))
    report_md = task.get("report_markdown", "") if not is_direct else ""
    direct_response_text = task.get("direct_response_text", "") if is_direct else ""

    return {
        "iteration_index": idx,
        "task": _task_detail(task),
        "task_summary": _task_summary(task),
        "active_plan_version": active_plan,
        "latest_plan_delta": None,
        "is_direct_response": is_direct,
        "direct_response_text": direct_response_text,
        "research_log": {
            "task_id": task["task_id"],
            "events": task.get("progress_events", []),
            "summaries": task.get("progress_summaries", []),
            "stats": {},
            "started_at": task.get("created_at", ""),
            "ended_at": task.get("updated_at", "") if task.get("status") in ("completed", "failed") else None,
        },
        "report": {
            "report_markdown_path": None,
            "report_markdown": report_md,
            "report_pdf_path": None,
            "has_report": bool(report_md and str(report_md).strip()),
        },
        "follow_up_suggestions": task.get("follow_up_suggestions", []),
        "branch_label": task.get("branch_label", ""),
        "parent_task_id": task.get("parent_task_id"),
    }


# ---------------------------------------------------------------------------
# UiRuntime: main runtime managing sessions and execution
# ---------------------------------------------------------------------------

class UiRuntime:
    def __init__(self, state_store_path: Path) -> None:
        self.store: SupportsWorkflowStateStore = create_state_store(state_store_path)
        self.ready = False
        self.ready_error: str | None = None
        self.user_id = "researcher"
        self.session_service: InMemorySessionService | None = None
        self.conv_sessions: dict[str, ConversationSession] = {}
        self.conv_thread_locks: dict[str, threading.Lock] = {}
        self.runs_lock = asyncio.Lock()
        self.runs: dict[str, RunRecord] = {}
        self.background_tasks: set[asyncio.Task] = set()
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="wf")

    async def startup(self) -> None:
        is_valid, error_message = validate_runtime_configuration()
        if not is_valid:
            self.ready_error = error_message
            return
        self.session_service = InMemorySessionService()
        interrupted = self.store.mark_incomplete_runs_failed(
            "Run interrupted because the server restarted before completion."
        )
        if interrupted:
            logger.warning("Marked %d incomplete runs as failed during startup.", interrupted)
        self.ready = True
        self.ready_error = None

    async def shutdown(self) -> None:
        pending = list(self.background_tasks)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._thread_pool.shutdown(wait=False)
        for cs in self.conv_sessions.values():
            if cs.mcp_tools is not None:
                try:
                    await cs.mcp_tools.close()
                except Exception:
                    pass

    def _get_conv_thread_lock(self, conversation_id: str) -> threading.Lock:
        if conversation_id not in self.conv_thread_locks:
            self.conv_thread_locks[conversation_id] = threading.Lock()
        return self.conv_thread_locks[conversation_id]

    async def _get_or_create_session(self, conversation_id: str) -> ConversationSession:
        if conversation_id in self.conv_sessions:
            return self.conv_sessions[conversation_id]
        if not self.session_service:
            raise RuntimeError("Session service is not initialized.")
        workflow_agent, mcp_tools = create_workflow_agent(require_plan_approval=True)
        app_name = f"co_scientist_ui_{conversation_id}"
        runner = Runner(
            agent=workflow_agent,
            app_name=app_name,
            session_service=self.session_service,
        )
        restored_session = self.store.get_workflow_session(conversation_id)
        restored_state = None
        if isinstance(restored_session, dict):
            payload = restored_session.get("state")
            if isinstance(payload, dict):
                restored_state = _extract_persistable_session_state(payload) or None
        session = await self.session_service.create_session(
            app_name=app_name,
            user_id=self.user_id,
            state=restored_state,
        )
        cs = ConversationSession(
            runner=runner,
            session_id=session.id,
            app_name=app_name,
            mcp_tools=mcp_tools,
        )
        self.conv_sessions[conversation_id] = cs
        return cs

    async def _read_session_state(self, conversation_id: str) -> dict | None:
        cs = self.conv_sessions.get(conversation_id)
        if not cs or not self.session_service:
            return None
        session = await self.session_service.get_session(
            app_name=cs.app_name,
            user_id=self.user_id,
            session_id=cs.session_id,
        )
        if not session:
            return None
        return session.state

    async def _read_workflow_state(self, conversation_id: str) -> dict | None:
        session_state = await self._read_session_state(conversation_id)
        if not session_state:
            return None
        return session_state.get(STATE_WORKFLOW_TASK)

    async def _read_persistable_session_state(self, conversation_id: str) -> dict:
        return _extract_persistable_session_state(await self._read_session_state(conversation_id))

    async def _persist_conversation_state(self, conversation_id: str, *, task_id: str = "") -> None:
        if not conversation_id:
            return
        state = await self._read_persistable_session_state(conversation_id)
        self.store.save_workflow_session(conversation_id, task_id=task_id, state=state or None)

    async def _is_plan_pending_approval(self, conversation_id: str) -> bool:
        session_state = await self._read_session_state(conversation_id)
        if not session_state:
            return False
        return bool(session_state.get(STATE_PLAN_PENDING_APPROVAL, False))

    async def _run_workflow_turn(
        self,
        conversation_id: str,
        prompt: str,
        *,
        run_id: str,
    ) -> tuple[str, str]:
        """Run a workflow turn in a dedicated thread.

        Returns (response_text, responding_author).
        """
        cs = await self._get_or_create_session(conversation_id)
        main_loop = asyncio.get_running_loop()
        thread_lock = self._get_conv_thread_lock(conversation_id)

        def _thread_target() -> tuple[str, str]:
            thread_lock.acquire()
            try:
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(
                        self._workflow_turn_inner(cs, conversation_id, prompt, run_id=run_id, caller_loop=main_loop)
                    )
                finally:
                    loop.close()
            finally:
                thread_lock.release()

        return await main_loop.run_in_executor(self._thread_pool, _thread_target)

    async def _run_workflow_turn_filtered(
        self,
        conversation_id: str,
        prompt: str,
        *,
        run_id: str,
        max_transient_retries: int = 3,
    ) -> tuple[str, str]:
        """Retry turns when the only returned content is a transient status line."""
        last_response = ("", "")
        for attempt in range(max_transient_retries + 1):
            response_text, responding_author = await self._run_workflow_turn(
                conversation_id,
                prompt,
                run_id=run_id,
            )
            last_response = (response_text, responding_author)
            if not _is_transient_workflow_response(response_text):
                return response_text, responding_author
            await self._append_progress_event(
                run_id,
                phase="execute",
                event_type="model.retry",
                status="progress",
                human_line="Rate limit hit; waiting for the workflow to resume before surfacing output.",
            )
            if attempt < max_transient_retries:
                await asyncio.sleep(0.2)
        return last_response

    async def _workflow_turn_inner(
        self,
        cs: ConversationSession,
        conversation_id: str,
        prompt: str,
        *,
        run_id: str,
        caller_loop: asyncio.AbstractEventLoop,
    ) -> tuple[str, str]:
        """The actual event-processing loop — runs inside its own thread event loop.

        Returns (response_text, responding_author).
        """
        current_message = Content(role="user", parts=[Part(text=prompt)])
        partial_by_author: dict[str, str] = {}
        final_by_author: dict[str, str] = {}
        fallback_chunks: list[str] = []
        step_counter = 0
        planner_seen = False
        step_started_ids: set[str] = set()
        step_completed_ids: set[str] = set()
        tool_error_signatures: set[tuple[str, str, str]] = set()

        def _step_source_label(tn: str) -> str:
            return _resolve_source_label(tn or "")

        def _fire_progress(**kwargs) -> None:
            asyncio.run_coroutine_threadsafe(
                self._append_progress_event(run_id, **kwargs), caller_loop
            )

        async for event in cs.runner.run_async(
            session_id=cs.session_id,
            user_id=self.user_id,
            new_message=current_message,
        ):
            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue

            if not planner_seen:
                for part in parts:
                    fc = getattr(part, "function_call", None)
                    if fc and getattr(fc, "name", "") == "transfer_to_agent":
                        target = (getattr(fc, "args", None) or {}).get("agent_name", "")
                        if target == "research_workflow":
                            planner_seen = True
                            _fire_progress(
                                phase="plan",
                                event_type="plan.initializing",
                                status="progress",
                                human_line="Preparing a plan...",
                            )
                            break

            author = str(getattr(event, "author", "") or "").strip()
            for part in parts:
                fc = getattr(part, "function_call", None)
                if not fc:
                    continue
                name = str(getattr(fc, "name", "") or "").strip()
                if not name or name == "transfer_to_agent":
                    continue
                args = {}
                try:
                    raw = getattr(fc, "args", None) or {}
                    args = dict(raw) if not isinstance(raw, dict) else raw
                except Exception:  # noqa: BLE001
                    pass
                human = _describe_tool_call(name, args)
                _fire_progress(
                    phase="execute",
                    event_type="tool.called",
                    status="progress",
                    human_line=human,
                    metrics={"tool": name},
                )
                # Emit step.started + live step summary so frontend gets tool_log in real time
                if author == "step_executor":
                    wf_state = await self._read_workflow_state(conversation_id)
                    sid = ""
                    if wf_state:
                        for s in (wf_state.get("steps") or []):
                            st = str(s.get("status", "")).strip()
                            if st == "in_progress":
                                sid = str(s.get("id", "")).strip()
                                break
                        if not sid:
                            for s in (wf_state.get("steps") or []):
                                if str(s.get("status", "")).strip() != "completed":
                                    sid = str(s.get("id", "")).strip()
                                    break
                    if sid and sid not in step_started_ids:
                        step_started_ids.add(sid)
                        step_goal = ""
                        if wf_state:
                            for s in (wf_state.get("steps") or []):
                                if str(s.get("id", "")) == sid:
                                    step_goal = str(s.get("goal", "")).strip()
                                    break
                        human_step = f"{sid}: {step_goal}" if step_goal else f"Executing {sid}…"
                        _fire_progress(
                            phase="execute",
                            event_type="step.started",
                            status="progress",
                            human_line=human_step,
                            metrics={"step_id": sid},
                        )
                    # Emit a live summary snapshot so the frontend has step_details + tool_log
                    if wf_state:
                        try:
                            asyncio.run_coroutine_threadsafe(
                                self._emit_step_summary(run_id, wf_state, step_counter), caller_loop
                            ).result(timeout=5)
                        except Exception:  # noqa: BLE001
                            pass
            for part in parts:
                fr = getattr(part, "function_response", None)
                error_metrics = _extract_tool_error_metrics(fr)
                if not error_metrics:
                    continue
                signature = (
                    str(error_metrics.get("tool", "") or "").strip(),
                    str(error_metrics.get("error_type", "") or "").strip(),
                    str(error_metrics.get("message", "") or "").strip(),
                )
                if signature in tool_error_signatures:
                    continue
                tool_error_signatures.add(signature)
                tool_name = str(error_metrics.get("tool", "") or "").strip()
                source = _step_source_label(tool_name) if tool_name else ""
                error_message = str(error_metrics.get("message", "") or "").strip()
                if source:
                    human = f"{source} error: {error_message or 'tool call failed.'}"
                elif tool_name:
                    human = f"{tool_name} error: {error_message or 'tool call failed.'}"
                else:
                    human = error_message or "Tool call failed."
                _fire_progress(
                    phase="execute",
                    event_type="tool.failed",
                    status="error",
                    human_line=human,
                    metrics=error_metrics,
                )

            text = "".join(
                part.text
                for part in parts
                if isinstance(getattr(part, "text", None), str)
            ).strip()
            if not text:
                continue
            if _is_transient_workflow_response(text):
                continue
            if not author:
                continue

            fallback_chunks.append(text)

            if author == "step_executor":
                accumulated = f"{partial_by_author.get('step_executor', '')}{text}"
                matches = re.findall(r"###\s*(S\d+)", accumulated)
                if matches:
                    sid = matches[-1]
                    if sid not in step_started_ids:
                        step_started_ids.add(sid)
                        wf_state = await self._read_workflow_state(conversation_id)
                        step_goal = ""
                        if wf_state:
                            for s in (wf_state.get("steps") or []):
                                if str(s.get("id", "")) == sid:
                                    step_goal = str(s.get("goal", "")).strip()
                                    break
                        human = f"{sid}: {step_goal}" if step_goal else f"Executing {sid}…"
                        _fire_progress(
                            phase="execute",
                            event_type="step.started",
                            status="progress",
                            human_line=human,
                            metrics={"step_id": sid},
                        )

            if author == "step_executor" and not bool(getattr(event, "partial", False)):
                metrics = _build_step_completed_event_metrics(text)
                if not metrics:
                    logger.debug(
                        "[ui:%s] ignoring non-step step_executor text chunk: %s",
                        run_id,
                        _compact_text(text, max_chars=120),
                    )
                    continue
                step_id = str(metrics["step_id"]).strip()
                if step_id in step_completed_ids:
                    logger.debug("[ui:%s] suppressing duplicate completion event for %s", run_id, step_id)
                    continue
                step_completed_ids.add(step_id)
                step_counter += 1

                goal = str(metrics.get("goal", "") or "").strip()
                headline = f"{step_id} · {goal}" if goal else f"{step_id} complete"
                _fire_progress(
                    phase="execute",
                    event_type="step.completed",
                    status="done",
                    human_line=_compact_text(headline, max_chars=220),
                    metrics={**metrics, "step_number": step_counter},
                )

                # Emit intermediate step summary so frontend gets tool_log data mid-run
                wf_snap = await self._read_workflow_state(conversation_id)
                if wf_snap:
                    asyncio.run_coroutine_threadsafe(
                        self._emit_step_summary(run_id, wf_snap, step_counter), caller_loop
                    ).result(timeout=10)

            elif author == "planner" and not bool(getattr(event, "partial", False)):
                _fire_progress(
                    phase="plan",
                    event_type="plan.generated",
                    status="done",
                    human_line="Research plan generated.",
                )

            elif author == "report_synthesizer" and not bool(getattr(event, "partial", False)):
                _fire_progress(
                    phase="synthesize",
                    event_type="synthesis.completed",
                    status="done",
                    human_line="Final report synthesized.",
                )

            if bool(getattr(event, "partial", False)):
                partial_by_author[author] = f"{partial_by_author.get(author, '')}{text}"
                continue

            is_final = getattr(event, "is_final_response", None)
            if callable(is_final) and bool(is_final()):
                final_by_author[author] = (
                    f"{partial_by_author.pop(author, '')}{text}".strip() or text
                )
                continue
            partial_by_author[author] = f"{partial_by_author.get(author, '')}{text}"

        wf_state = await self._read_workflow_state(conversation_id)
        if wf_state and step_counter > 0:
            fut = asyncio.run_coroutine_threadsafe(
                self._emit_step_summary(run_id, wf_state, step_counter), caller_loop
            )
            fut.result(timeout=10)

        _preferred_authors = (
            "report_synthesizer",
            "general_qa",
            "clarifier",
            "report_assistant",
            "research_workflow",
            "co_scientist_router",
        )
        for preferred_author in _preferred_authors:
            candidate = final_by_author.get(preferred_author, "").strip()
            if candidate:
                return candidate, preferred_author
        if final_by_author:
            for author in sorted(final_by_author.keys(), reverse=True):
                candidate = final_by_author.get(author, "").strip()
                if candidate:
                    return candidate, author
        fallback = "\n".join(chunk for chunk in fallback_chunks if chunk).strip() or "(No response)"
        return fallback, ""

    async def _emit_step_summary(self, run_id: str, wf_state: dict, steps_executed: int) -> None:
        """Build a structured progress summary from workflow state after execution."""
        steps = wf_state.get("steps", [])
        completed = sum(1 for s in steps if s.get("status") == "completed")
        total = len(steps)
        plan_status = wf_state.get("plan_status", "")

        completed_lines = []
        for s in steps:
            if s.get("status") != "completed":
                continue
            sid = s.get("id", "?")
            goal = _compact_text(s.get("goal", ""), max_chars=80)
            completed_lines.append(f"{sid}: {goal}")

        next_lines = []
        for s in steps:
            if s.get("status") == "pending":
                sid = s.get("id", "?")
                goal = _compact_text(s.get("goal", ""), max_chars=80)
                next_lines.append(f"{sid}: {goal}")
                if len(next_lines) >= 2:
                    break

        if plan_status == "completed":
            headline = f"All {total} steps complete — generating report"
        else:
            headline = f"Completed {completed}/{total} steps"

        step_details = []
        for s in steps:
            tool_hint = str(s.get("tool_hint", "")).strip()
            source = _resolve_source_label(tool_hint) if tool_hint else ""
            step_details.append({
                "id": s.get("id", ""),
                "goal": s.get("goal", ""),
                "status": s.get("status", "pending"),
                "tool_hint": tool_hint,
                "source": source,
                "result_summary": s.get("result_summary", ""),
                "executor_prose": s.get("executor_prose", ""),
                "tool_reasoning": s.get("tool_reasoning", ""),
                "evidence_ids": s.get("evidence_ids", []),
                "tools_called": s.get("tools_called", []),
                "open_gaps": s.get("open_gaps", []),
                "reasoning_trace": s.get("reasoning_trace", ""),
                "tool_log": s.get("tool_log", []),
            })

        summary = {
            "snapshot_id": f"snap_{uuid.uuid4().hex[:10]}",
            "at": _utc_now(),
            "phase": "synthesize" if plan_status == "completed" else "execute",
            "trigger_type": "step.completed",
            "headline": headline,
            "summary": f"{completed}/{total} plan steps executed",
            "completed": completed_lines[:6],
            "next": next_lines[:3],
            "confidence": "high" if plan_status == "completed" else "medium",
            "step_details": step_details,
            "steps_completed": completed,
            "steps_total": total,
        }
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if run:
                run.progress_summaries.append(summary)
                run.updated_at = _utc_now()

    async def _save_task_with_progress(
        self, task: dict, run_id: str | None = None, *, owner_ip: str = "", merge_progress: bool = True
    ) -> None:
        """Save task to store, syncing progress data from the active run unless merge_progress=False."""
        active_run_id = ""
        if run_id and merge_progress:
            async with self.runs_lock:
                run = self.runs.get(run_id)
                if run:
                    task["progress_events"] = list(run.progress_events[-600:])
                    task["progress_summaries"] = list(run.progress_summaries[-80:])
                    if run.status in self._ACTIVE_RUN_STATUSES:
                        active_run_id = run.run_id
        elif run_id:
            async with self.runs_lock:
                run = self.runs.get(run_id)
                if run and run.status in self._ACTIVE_RUN_STATUSES:
                    active_run_id = run.run_id
        if active_run_id:
            task["active_run_id"] = active_run_id
        else:
            task.pop("active_run_id", None)
        self.store.save_task(task, owner_ip=owner_ip)
        conversation_id = str(task.get("conversation_id", "") or "").strip()
        if conversation_id:
            await self._persist_conversation_state(
                conversation_id,
                task_id=str(task.get("task_id", "") or "").strip(),
            )

    # -- Run management -------------------------------------------------------

    async def _create_run(self, kind: str, *, query: str = "", task_id: str | None = None) -> RunRecord:
        run = RunRecord(
            run_id=f"run_{uuid.uuid4().hex[:10]}",
            kind=kind,
            query=query,
            title=_generate_chat_title(query) if query.strip() else "",
            task_id=task_id,
        )
        async with self.runs_lock:
            self.runs[run.run_id] = run
        self.store.save_run(run.to_dict())
        return run

    async def _update_run(self, run_id: str, **updates) -> None:
        run_payload: dict | None = None
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return
            for key, value in updates.items():
                setattr(run, key, value)
            run.updated_at = _utc_now()
            run_payload = run.to_dict()
        if run_payload:
            self.store.save_run(run_payload)

    async def _append_progress_event(
        self,
        run_id: str,
        *,
        phase: str,
        event_type: str,
        status: str,
        human_line: str,
        task_id: str | None = None,
        metrics: dict | None = None,
    ) -> None:
        event = {
            "event_id": f"evt_{uuid.uuid4().hex[:10]}",
            "at": _utc_now(),
            "phase": phase,
            "type": event_type,
            "status": status,
            "human_line": _compact_text(human_line, max_chars=220),
            "task_id": task_id or "",
            "step_index": None,
            "step_title": "",
            "tool": "",
            "metrics": metrics or {},
        }
        run_payload: dict | None = None
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
            run_payload = run.to_dict()
        if run_payload:
            self.store.save_run(run_payload)
        logger.info("[ui:%s] %s", run_id, event["human_line"])

    def _track_background_task(self, task: asyncio.Task) -> None:
        self.background_tasks.add(task)
        task.add_done_callback(lambda done: self.background_tasks.discard(done))

    # -- Execution flows -------------------------------------------------------

    async def start_new_query(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
        parent_task_id: str | None = None,
        owner_ip: str = "",
    ) -> RunRecord:
        run = await self._create_run("new_query", query=query)
        job = asyncio.create_task(
            self._run_new_query(
                run.run_id,
                query,
                conversation_id=conversation_id,
                parent_task_id=parent_task_id,
                owner_ip=owner_ip,
            )
        )
        self._track_background_task(job)
        return run

    async def start_task(self, task_id: str) -> RunRecord:
        run = await self._create_run("start_task", task_id=task_id)
        job = asyncio.create_task(self._run_start_task(run.run_id, task_id))
        self._track_background_task(job)
        return run

    async def feedback_task(self, task_id: str, message: str) -> RunRecord:
        run = await self._create_run("feedback_task", task_id=task_id, query=message)
        job = asyncio.create_task(self._run_feedback_task(run.run_id, task_id, message))
        self._track_background_task(job)
        return run

    async def _run_new_query(
        self,
        run_id: str,
        query: str,
        *,
        conversation_id: str | None = None,
        parent_task_id: str | None = None,
        owner_ip: str = "",
    ) -> None:
        await self._update_run(run_id, status="running")
        if not self.ready:
            await self._update_run(run_id, status="failed", error=self.ready_error or "Not ready.")
            return

        try:
            task_id = f"task_{uuid.uuid4().hex[:10]}"
            conv_id = conversation_id or f"conv_{task_id}"
            await self._get_or_create_session(conv_id)

            title = _generate_chat_title(query)
            parent = parent_task_id.strip() if parent_task_id else None
            branch_label = f"Branched from report {parent}" if parent else ""

            task = _make_task(
                task_id,
                query,
                conv_id,
                title=title,
                user_query=query,
                parent_task_id=parent,
            )
            task["branch_label"] = branch_label
            await self._save_task_with_progress(task, run_id, owner_ip=owner_ip)
            await self._update_run(run_id, task_id=task_id, title=title)

            _DIRECT_RESPONSE_AGENTS = {"general_qa", "clarifier", "report_assistant"}

            max_plan_attempts = 2
            responding_author = ""
            planner_failed = False
            plan_pending = False

            response_text, responding_author = await self._run_workflow_turn_filtered(
                conv_id, query, run_id=run_id,
            )

            wf_state = await self._read_workflow_state(conv_id)
            plan_pending = await self._is_plan_pending_approval(conv_id)
            has_steps = bool(_steps_from_workflow_state(wf_state))

            direct_response_detected = (
                responding_author in _DIRECT_RESPONSE_AGENTS
                or (not wf_state and not plan_pending and not has_steps)
            )

            if not direct_response_detected:
                await self._append_progress_event(
                    run_id,
                    phase="intake",
                    event_type="run.started",
                    status="start",
                    human_line=f"Started: {title}",
                    task_id=task_id,
                )
                await self._append_progress_event(
                    run_id,
                    phase="plan",
                    event_type="plan.initializing",
                    status="progress",
                    human_line="Building research plan...",
                    task_id=task_id,
                )

                task["steps"] = _steps_from_workflow_state(wf_state)
                task["current_step_index"] = 0

                restated = (wf_state or {}).get("objective", "").strip()
                if restated:
                    task["objective"] = restated
                    task["title"] = _generate_chat_title(restated)

                planner_failed = not wf_state and not plan_pending
                if planner_failed:
                    for plan_attempt in range(2, max_plan_attempts + 1):
                        logger.warning(
                            "[new_task] Planner failed (attempt %d/%d), retrying...",
                            plan_attempt - 1, max_plan_attempts,
                        )
                        await self._append_progress_event(
                            run_id,
                            phase="plan",
                            event_type="plan.retry",
                            status="progress",
                            human_line=f"Plan generation failed (attempt {plan_attempt - 1}), retrying...",
                            task_id=task_id,
                        )
                        response_text, responding_author = await self._run_workflow_turn_filtered(
                            conv_id, query, run_id=run_id,
                        )
                        wf_state = await self._read_workflow_state(conv_id)
                        plan_pending = await self._is_plan_pending_approval(conv_id)
                        task["steps"] = _steps_from_workflow_state(wf_state)
                        task["current_step_index"] = 0
                        restated = (wf_state or {}).get("objective", "").strip()
                        if restated:
                            task["objective"] = restated
                            task["title"] = _generate_chat_title(restated)
                        planner_failed = not wf_state and not plan_pending
                        if not planner_failed:
                            break

            if direct_response_detected:
                task["status"] = "completed"
                task["is_direct_response"] = True
                task["direct_response_text"] = response_text
                task["progress_events"] = []
                task["progress_summaries"] = []
                await self._save_task_with_progress(task, run_id, merge_progress=False)
                async with self.runs_lock:
                    run = self.runs.get(run_id)
                    if run:
                        run.progress_events.clear()
                        run.progress_summaries.clear()
                await self._update_run(
                    run_id, status="completed", task_id=task_id,
                    final_report=response_text,
                )

            elif planner_failed:
                task["status"] = "failed"
                task["report_markdown"] = response_text
                await self._save_task_with_progress(task, run_id)
                await self._update_run(
                    run_id, status="failed", task_id=task_id,
                    error="Planner failed to generate a valid research plan.",
                )
                await self._append_progress_event(
                    run_id,
                    phase="plan",
                    event_type="plan.failed",
                    status="error",
                    human_line="Failed to generate research plan. Please try again.",
                    task_id=task_id,
                )
            elif plan_pending:
                task["awaiting_hitl"] = True
                task["status"] = "in_progress"
                await self._append_progress_event(
                    run_id,
                    phase="plan",
                    event_type="task.created",
                    status="done",
                    human_line=f"Plan ready with {len(task['steps'])} steps. Waiting for approval.",
                    task_id=task_id,
                    metrics={"steps_total": len(task["steps"])},
                )
                await self._save_task_with_progress(task, run_id)
                await self._update_run(run_id, status="awaiting_hitl", task_id=task_id)
            else:
                task["status"] = "completed"
                task["follow_up_suggestions"] = _extract_next_steps(response_text)
                task["report_markdown"] = _strip_next_steps_section(response_text)
                await self._save_task_with_progress(task, run_id)
                await self._update_run(
                    run_id, status="completed", task_id=task_id,
                    final_report=task["report_markdown"],
                )
                await self._append_progress_event(
                    run_id,
                    phase="finalize",
                    event_type="run.completed",
                    status="done",
                    human_line="Research complete.",
                    task_id=task_id,
                )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            await self._append_progress_event(
                run_id,
                phase="finalize",
                event_type="run.failed",
                status="error",
                human_line=f"Run failed: {error}",
            )
            await self._update_run(run_id, status="failed", error=error)
            traceback.print_exc()

    async def _run_start_task(self, run_id: str, task_id: str) -> None:
        await self._update_run(run_id, status="running")
        if not self.ready:
            await self._update_run(run_id, status="failed", error=self.ready_error or "Not ready.")
            return

        try:
            task = self.store.get_task(task_id)
            if not task:
                await self._update_run(run_id, status="failed", error=f"Task {task_id} not found.")
                return
            if not task.get("awaiting_hitl"):
                await self._update_run(run_id, status="failed", error="Task is not at checkpoint.")
                return

            conv_id = task["conversation_id"]
            await self._update_run(run_id, task_id=task_id, title=task.get("title", ""))
            task["hitl_history"].append("approve")
            task["awaiting_hitl"] = False
            await self._save_task_with_progress(task, run_id)

            await self._append_progress_event(
                run_id,
                phase="execute",
                event_type="execution.running",
                status="start",
                human_line="Executing plan...",
                task_id=task_id,
            )

            max_continue_loops = 100
            for loop_idx in range(max_continue_loops):
                response_text, _ = await self._run_workflow_turn_filtered(
                    conv_id, "approve", run_id=run_id,
                )

                wf_state = await self._read_workflow_state(conv_id)
                plan_pending = await self._is_plan_pending_approval(conv_id)

                task["steps"] = _steps_from_workflow_state(wf_state)
                completed_steps = sum(
                    1 for s in task["steps"] if s.get("status") == "completed"
                )
                task["current_step_index"] = completed_steps

                plan_status = wf_state.get("plan_status", "") if wf_state else ""
                has_synthesis = bool(
                    wf_state
                    and wf_state.get("latest_synthesis", {})
                    and wf_state["latest_synthesis"].get("markdown", "").strip()
                )

                if plan_status == "completed" and has_synthesis:
                    task["status"] = "completed"
                    final_md = wf_state["latest_synthesis"]["markdown"]
                    task["follow_up_suggestions"] = _extract_next_steps(final_md)
                    stripped_md = _strip_next_steps_section(final_md)
                    task["report_markdown"] = stripped_md
                    await self._append_progress_event(
                        run_id,
                        phase="finalize",
                        event_type="run.completed",
                        status="done",
                        human_line="Report completed.",
                        task_id=task_id,
                    )
                    await self._save_task_with_progress(task, run_id)
                    self._write_report(task_id, stripped_md)
                    await self._update_run(
                        run_id, status="completed", task_id=task_id,
                        final_report=stripped_md,
                        follow_up_suggestions=task["follow_up_suggestions"],
                    )
                    break

                if plan_pending:
                    task["awaiting_hitl"] = True
                    task["status"] = "in_progress"
                    await self._save_task_with_progress(task, run_id)
                    await self._update_run(run_id, status="awaiting_hitl", task_id=task_id)
                    await self._append_progress_event(
                        run_id,
                        phase="checkpoint",
                        event_type="checkpoint.opened",
                        status="done",
                        human_line="Revised plan ready. Waiting for approval.",
                        task_id=task_id,
                    )
                    break

                if plan_status != "completed":
                    await self._save_task_with_progress(task, run_id)
                    await self._append_progress_event(
                        run_id,
                        phase="execute",
                        event_type="execution.running",
                        status="progress",
                        human_line=f"Executing step {completed_steps + 1}/{len(task['steps'])}...",
                        task_id=task_id,
                    )
                    continue

                task["status"] = "completed"
                task["follow_up_suggestions"] = _extract_next_steps(response_text)
                stripped_md = _strip_next_steps_section(response_text)
                task["report_markdown"] = stripped_md
                await self._append_progress_event(
                    run_id,
                    phase="finalize",
                    event_type="run.completed",
                    status="done",
                    human_line="Report completed.",
                    task_id=task_id,
                )
                await self._save_task_with_progress(task, run_id)
                self._write_report(task_id, stripped_md)
                await self._update_run(
                    run_id, status="completed", task_id=task_id,
                    final_report=stripped_md,
                    follow_up_suggestions=task["follow_up_suggestions"],
                )
                break

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            await self._append_progress_event(
                run_id,
                phase="finalize",
                event_type="run.failed",
                status="error",
                human_line=f"Run failed: {error}",
                task_id=task_id,
            )
            await self._update_run(run_id, status="failed", error=error)
            traceback.print_exc()

    async def _run_feedback_task(self, run_id: str, task_id: str, message: str) -> None:
        await self._update_run(run_id, status="running")
        if not self.ready:
            await self._update_run(run_id, status="failed", error=self.ready_error or "Not ready.")
            return

        try:
            task = self.store.get_task(task_id)
            if not task:
                await self._update_run(run_id, status="failed", error=f"Task {task_id} not found.")
                return

            conv_id = task["conversation_id"]
            await self._update_run(run_id, task_id=task_id, title=task.get("title", ""))

            prompt = f"revise: {message}"
            await self._append_progress_event(
                run_id,
                phase="plan",
                event_type="feedback.applying",
                status="start",
                human_line="Applying feedback...",
                task_id=task_id,
            )

            response_text, _ = await self._run_workflow_turn_filtered(
                conv_id, prompt, run_id=run_id,
            )

            wf_state = await self._read_workflow_state(conv_id)
            plan_pending = await self._is_plan_pending_approval(conv_id)

            task["steps"] = _steps_from_workflow_state(wf_state)
            task["hitl_history"].append(f"revise:{message}")
            task["awaiting_hitl"] = plan_pending
            task["status"] = "in_progress"

            restated = (wf_state or {}).get("objective", "").strip()
            if restated:
                task["objective"] = restated
                task["title"] = _generate_chat_title(restated)
                await self._update_run(run_id, title=task["title"])

            await self._save_task_with_progress(task, run_id)

            if plan_pending:
                await self._update_run(run_id, status="awaiting_hitl", task_id=task_id)
                await self._append_progress_event(
                    run_id,
                    phase="checkpoint",
                    event_type="feedback.applied",
                    status="done",
                    human_line="Revised plan ready. Waiting for approval.",
                    task_id=task_id,
                )
            else:
                await self._update_run(run_id, status="completed", task_id=task_id)
                await self._append_progress_event(
                    run_id,
                    phase="plan",
                    event_type="feedback.applied",
                    status="done",
                    human_line="Feedback applied.",
                    task_id=task_id,
                )

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            await self._append_progress_event(
                run_id,
                phase="finalize",
                event_type="run.failed",
                status="error",
                human_line=f"Run failed: {error}",
                task_id=task_id,
            )
            await self._update_run(run_id, status="failed", error=error)
            traceback.print_exc()

    # -- Reports ---------------------------------------------------------------

    def _report_dir(self) -> Path:
        d = Path(__file__).resolve().parent / "reports"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _write_report(self, task_id: str, markdown: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", task_id)
        path = self._report_dir() / f"{safe_id}.md"
        path.write_text((markdown or "").rstrip() + "\n", encoding="utf-8")
        return path

    # -- Read APIs (conversations, tasks, runs) --------------------------------

    _ACTIVE_RUN_STATUSES = frozenset({"running", "queued", "awaiting_hitl", "in_progress"})

    async def _overlay_live_progress(self, task: dict) -> dict:
        """Merge live progress from a run into the task dict. Adds active_run_id when run is in progress."""
        task_id = task.get("task_id", "")
        if not task_id:
            return task
        async with self.runs_lock:
            best: RunRecord | None = None
            for run in self.runs.values():
                if run.task_id != task_id:
                    continue
                if best is None or run.updated_at > best.updated_at:
                    best = run
            if best is not None:
                task = dict(task)
                if best.status in self._ACTIVE_RUN_STATUSES:
                    task["active_run_id"] = best.run_id
                if best.progress_events:
                    task["progress_events"] = list(best.progress_events)
                if best.progress_summaries:
                    task["progress_summaries"] = list(best.progress_summaries)
        return task

    def list_conversations(self, *, owner_ip: str = "") -> list[dict]:
        return self.store.list_conversations(owner_ip=owner_ip)

    async def get_conversation_detail(self, conversation_id: str) -> dict | None:
        tasks = self.store.get_conversation_tasks(conversation_id)
        if not tasks:
            return None
        tasks.sort(key=lambda t: (t.get("created_at", ""), t.get("task_id", "")))

        tasks = [await self._overlay_live_progress(t) for t in tasks]

        iterations = []
        for idx, task in enumerate(tasks, start=1):
            iterations.append(_iteration_from_task(task, idx))

        latest = max(tasks, key=lambda t: t.get("updated_at", ""))
        root = tasks[0]
        latest_completed = next(
            (t for t in reversed(tasks) if t.get("status") == "completed"), None
        )
        selected_report_task_id = (
            latest_completed["task_id"] if latest_completed else latest["task_id"]
        )

        return {
            "conversation": {
                "conversation_id": conversation_id,
                "title": latest.get("title") or root.get("title") or "Research",
                "root_task_id": root["task_id"],
                "latest_task_id": latest["task_id"],
                "latest_status": latest.get("status", ""),
                "updated_at": latest.get("updated_at", ""),
                "iteration_count": len(tasks),
                "selected_report_task_id": selected_report_task_id,
            },
            "iterations": iterations,
        }

    async def get_task_detail(self, task_id: str) -> dict | None:
        task = self.store.get_task(task_id)
        if not task:
            return None
        task = await self._overlay_live_progress(task)
        return {
            "task": _task_detail(task),
            "active_plan_version": {
                "version_id": f"plan_{task_id}",
                "steps": _normalize_steps_for_ui(task.get("steps", [])),
            } if task.get("steps") else None,
            "latest_plan_delta": None,
            "pending_feedback_queue_count": 0,
            "checkpoint_reason": "",
            "checkpoint_payload": {},
            "phase_state": {},
            "planner_mode": "",
            "quality_confidence": "",
            "researcher_candidates": [],
            "revisions": [],
            "report_markdown_path": None,
            "report_markdown": task.get("report_markdown", ""),
            "report_pdf_path": None,
        }

    async def get_task_workflow_state_debug(self, task_id: str) -> dict | None:
        task = self.store.get_task(task_id)
        if not task:
            return None
        conversation_id = str(task.get("conversation_id", "") or "").strip()
        live_state = await self._read_persistable_session_state(conversation_id)
        persisted = self.store.get_workflow_session(conversation_id) if conversation_id else None
        source = "none"
        state: dict[str, object] = {}
        if live_state:
            source = "live"
            state = live_state
        elif isinstance(persisted, dict) and isinstance(persisted.get("state"), dict):
            source = "persisted"
            state = copy.deepcopy(persisted["state"])
        return {
            "task_id": task_id,
            "conversation_id": conversation_id,
            "source": source,
            "state": state,
            "persisted_updated_at": str((persisted or {}).get("updated_at", "") or ""),
            "persisted_task_id": str((persisted or {}).get("task_id", "") or ""),
        }

    async def get_run(self, run_id: str) -> dict | None:
        async with self.runs_lock:
            run = self.runs.get(run_id)
            if not run:
                return self.store.get_run(run_id)
            payload = run.to_dict()
        self.store.save_run(payload)
        return payload


def _extract_next_steps(markdown: str) -> list[str]:
    """Try to pull next steps / potential next steps from synthesized report."""
    suggestions: list[str] = []
    in_next_steps = False
    for line in str(markdown or "").split("\n"):
        stripped = line.strip()
        lowered = stripped.lower()
        if "next step" in lowered or "potential next" in lowered:
            in_next_steps = True
            continue
        if in_next_steps:
            if stripped.startswith("#"):
                break
            m = re.match(r"^[-*\d.]+\s+(.+)$", stripped)
            if m:
                suggestions.append(m.group(1).strip())
            if len(suggestions) >= 3:
                break
    return suggestions


def _strip_next_steps_section(markdown: str) -> str:
    """Remove the Next Steps / Potential Next Steps heading and its content."""
    lines = str(markdown or "").split("\n")
    out: list[str] = []
    skipping = False
    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped.startswith("#") and ("next step" in lowered or "potential next" in lowered):
            skipping = True
            continue
        if skipping:
            if stripped.startswith("#"):
                skipping = False
            else:
                continue
        out.append(line)
    return "\n".join(out).rstrip()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
UI_DIR = ROOT_DIR / "ui"
STATE_PATH = ROOT_DIR / "state" / "workflow_tasks.json"

runtime = UiRuntime(STATE_PATH)
app = FastAPI(title="AI Co-Scientist UI", version="0.2.0")
app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


def _ga4_head_snippet() -> str:
    if not GA4_MEASUREMENT_ID:
        return ""
    if not _GA4_ID_PATTERN.fullmatch(GA4_MEASUREMENT_ID):
        logger.warning("Ignoring invalid GA4_MEASUREMENT_ID format.")
        return ""
    ga_id = GA4_MEASUREMENT_ID
    return (
        f'<script async src="https://www.googletagmanager.com/gtag/js?id={ga_id}"></script>\n'
        "<script>\n"
        "  window.dataLayer = window.dataLayer || [];\n"
        "  function gtag(){dataLayer.push(arguments);}\n"
        "  gtag('js', new Date());\n"
        f"  gtag('config', '{ga_id}', {{ anonymize_ip: true }});\n"
        "</script>"
    )


def _ui_asset_version() -> str:
    parts: list[str] = []
    for name in ("app.js", "styles.css"):
        try:
            parts.append(str(int((UI_DIR / name).stat().st_mtime)))
        except OSError:
            continue
    return "-".join(parts) or str(int(time.time()))


def _render_ui_page(filename: str) -> HTMLResponse:
    html_path = UI_DIR / filename
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("<!-- GA4_SNIPPET -->", _ga4_head_snippet(), 1)
    html = html.replace("__UI_VERSION__", _ui_asset_version())
    return HTMLResponse(
        content=html,
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.on_event("startup")
async def _startup() -> None:
    await runtime.startup()
    if runtime.ready:
        _port = int(os.environ.get("CO_SCI_UI_PORT", "8080"))
        print(f"[ui] Server ready at http://127.0.0.1:{_port}")
    else:
        print(f"[ui] Startup warning: {runtime.ready_error}")


@app.on_event("shutdown")
async def _shutdown() -> None:
    await runtime.shutdown()


@app.get("/")
async def index() -> HTMLResponse:
    return _render_ui_page("index.html")


@app.get("/about")
async def about() -> HTMLResponse:
    return _render_ui_page("about.html")


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": runtime.ready,
        "busy": any(lock.locked() for lock in runtime.conv_thread_locks.values()),
        "error": runtime.ready_error,
    }


@app.get("/api/rate-limit")
async def rate_limit_status(request: Request) -> dict:
    ip = _client_ip(request)
    remaining = query_limiter.remaining(ip)
    return {
        "limit": RATE_LIMIT_QUERIES,
        "remaining": remaining,
        "window_seconds": RATE_LIMIT_WINDOW,
    }


@app.get("/api/tasks")
async def list_tasks(request: Request) -> dict:
    ip = _client_ip(request)
    convs = runtime.list_conversations(owner_ip=ip)
    all_tasks = []
    for c in convs:
        tasks = runtime.store.get_conversation_tasks(c["conversation_id"])
        all_tasks.extend([_task_summary(t) for t in tasks])
    all_tasks.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
    return {"tasks": all_tasks}


@app.get("/api/conversations")
async def list_conversations(request: Request) -> dict:
    ip = _client_ip(request)
    return {"conversations": runtime.list_conversations(owner_ip=ip)}


@app.get("/api/conversations/{conversation_id}")
async def conversation_detail(conversation_id: str, request: Request) -> dict:
    ip = _client_ip(request)
    if not runtime.store.conversation_owned_by(conversation_id, ip):
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found.")
    detail = await runtime.get_conversation_detail(conversation_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found.")
    return detail


@app.get("/api/tasks/{task_id}")
async def task_detail(task_id: str, request: Request) -> dict:
    task = runtime.store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    ip = _client_ip(request)
    conv_id = task.get("conversation_id", "")
    if conv_id and not runtime.store.conversation_owned_by(conv_id, ip):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    detail = await runtime.get_task_detail(task_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return detail


@app.get("/api/tasks/{task_id}/debug/workflow-state")
async def task_workflow_state_debug(task_id: str, request: Request) -> dict:
    _check_task_ownership(task_id.strip(), request)
    detail = await runtime.get_task_workflow_state_debug(task_id.strip())
    if not detail:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return detail


@app.post("/api/query")
async def start_query(payload: QueryRequest, request: Request) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    _enforce_rate_limit(request)
    ip = _client_ip(request)
    run = await runtime.start_new_query(
        payload.query.strip(),
        conversation_id=payload.conversation_id.strip() if payload.conversation_id else None,
        parent_task_id=payload.parent_task_id.strip() if payload.parent_task_id else None,
        owner_ip=ip,
    )
    return run.to_dict()


def _check_task_ownership(task_id: str, request: Request) -> dict:
    """Return the task dict if it exists and belongs to the caller, else 404."""
    task = runtime.store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    conv_id = task.get("conversation_id", "")
    ip = _client_ip(request)
    if conv_id and not runtime.store.conversation_owned_by(conv_id, ip):
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    return task


@app.post("/api/tasks/{task_id}/start")
async def start_task(task_id: str, request: Request, payload: StartRequest | None = None) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    _enforce_rate_limit(request)
    _check_task_ownership(task_id.strip(), request)
    run = await runtime.start_task(task_id.strip())
    return run.to_dict()


@app.post("/api/tasks/{task_id}/continue")
async def continue_task(task_id: str, request: Request) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    _enforce_rate_limit(request)
    _check_task_ownership(task_id.strip(), request)
    run = await runtime.start_task(task_id.strip())
    return run.to_dict()


@app.post("/api/tasks/{task_id}/feedback")
async def feedback_task(task_id: str, request: Request, payload: FeedbackRequest) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    _enforce_rate_limit(request)
    task = _check_task_ownership(task_id.strip(), request)
    msg = payload.message.strip()
    # When user types "approve"/"continue" etc. while plan is pending, treat as approval (start_task)
    # instead of revision feedback — otherwise "revise: approve" confuses the planner.
    if task.get("awaiting_hitl") and _is_continue_execution_command(msg):
        run = await runtime.start_task(task_id.strip())
        return run.to_dict()
    run = await runtime.feedback_task(task_id.strip(), msg)
    return run.to_dict()


@app.post("/api/tasks/{task_id}/revise")
async def revise_task(task_id: str, request: Request, payload: ReviseRequest) -> dict:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail=runtime.ready_error or "Runtime not ready.")
    _enforce_rate_limit(request)
    _check_task_ownership(task_id.strip(), request)
    run = await runtime.feedback_task(task_id.strip(), payload.scope.strip())
    return run.to_dict()


@app.post("/api/tasks/{task_id}/rollback")
async def rollback_task(task_id: str, payload: RollbackRequest) -> dict:
    raise HTTPException(status_code=501, detail="Rollback not supported in this version.")


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    payload = await runtime.get_run(run_id.strip())
    if not payload:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")
    return payload


@app.get("/api/tasks/{task_id}/report.pdf")
async def export_report_pdf(task_id: str, request: Request) -> FileResponse:
    _check_task_ownership(task_id.strip(), request)
    task = runtime.store.get_task(task_id.strip())
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found.")
    markdown = task.get("report_markdown", "").strip()
    if not markdown:
        raise HTTPException(status_code=404, detail="No report available for this task.")
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", task_id.strip())
    pdf_path = runtime._report_dir() / f"{safe_id}.pdf"
    error = write_markdown_pdf(markdown, pdf_path, title=task.get("title", "Report"))
    if error:
        raise HTTPException(status_code=503, detail=f"PDF export failed: {error}")
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{safe_id}.pdf",
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("CO_SCI_UI_HOST", "127.0.0.1")
    port = int(os.environ.get("CO_SCI_UI_PORT", "8080"))
    uvicorn.run("ui_server:app", host=host, port=port, reload=False)
