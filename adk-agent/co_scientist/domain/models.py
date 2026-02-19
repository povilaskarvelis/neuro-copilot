"""
Core workflow domain models shared across planner/runtime layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_chat_title(objective: str, *, max_words: int = 8) -> str:
    text = str(objective or "").strip()
    if not text:
        return "Untitled Research"

    markers = [
        "\nUser revision to scope/decomposition:",
        "\nRevision directives to apply:",
        "\nRequired revision constraints:",
        "\nUser clarification:",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    compact = re.sub(r"\s+", " ", first_line).strip(" .")
    compact = re.sub(
        r"^(?:please\s+|can you\s+|could you\s+|i need\s+|find me\s+|show me\s+|tell me\s+)",
        "",
        compact,
        flags=re.IGNORECASE,
    )
    compact = compact.strip(" .")
    if not compact:
        return "Untitled Research"

    words = compact.split()
    title = " ".join(words[:max(3, max_words)])
    if len(words) > max_words:
        title = f"{title}..."
    if len(title) > 72:
        title = f"{title[:69].rstrip()}..."
    return title or "Untitled Research"


@dataclass
class WorkflowStep:
    """A single executable step in a multi-step workflow."""

    step_id: str
    title: str
    instruction: str
    status: str = "pending"
    output: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    recommended_tools: list[str] = field(default_factory=list)
    fallback_tools: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    tool_trace: list[dict] = field(default_factory=list)
    rationale: str = ""
    reasoning_summary: str = ""
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    subgoal_id: str = ""
    evidence_requirements: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    max_tool_calls: int = 0
    done_criteria: list[str] = field(default_factory=list)
    critic_verdict: str = ""
    confidence_label: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "instruction": self.instruction,
            "status": self.status,
            "output": self.output,
            "evidence_refs": self.evidence_refs,
            "recommended_tools": self.recommended_tools,
            "fallback_tools": self.fallback_tools,
            "allowed_tools": self.allowed_tools,
            "tool_trace": self.tool_trace,
            "rationale": self.rationale,
            "reasoning_summary": self.reasoning_summary,
            "actions": self.actions,
            "observations": self.observations,
            "subgoal_id": self.subgoal_id,
            "evidence_requirements": self.evidence_requirements,
            "dependencies": self.dependencies,
            "max_tool_calls": self.max_tool_calls,
            "done_criteria": self.done_criteria,
            "critic_verdict": self.critic_verdict,
            "confidence_label": self.confidence_label,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkflowStep":
        return cls(
            step_id=payload.get("step_id", ""),
            title=payload.get("title", ""),
            instruction=payload.get("instruction", ""),
            status=payload.get("status", "pending"),
            output=payload.get("output", ""),
            evidence_refs=list(payload.get("evidence_refs", [])),
            recommended_tools=list(payload.get("recommended_tools", [])),
            fallback_tools=list(payload.get("fallback_tools", [])),
            allowed_tools=list(payload.get("allowed_tools", [])),
            tool_trace=list(payload.get("tool_trace", [])),
            rationale=payload.get("rationale", ""),
            reasoning_summary=payload.get("reasoning_summary", ""),
            actions=list(payload.get("actions", [])),
            observations=list(payload.get("observations", [])),
            subgoal_id=str(payload.get("subgoal_id", "")).strip(),
            evidence_requirements=[str(x).strip() for x in payload.get("evidence_requirements", []) if str(x).strip()],
            dependencies=[str(x).strip() for x in payload.get("dependencies", []) if str(x).strip()],
            max_tool_calls=int(payload.get("max_tool_calls", 0) or 0),
            done_criteria=[str(x).strip() for x in payload.get("done_criteria", []) if str(x).strip()],
            critic_verdict=str(payload.get("critic_verdict", "")).strip(),
            confidence_label=str(payload.get("confidence_label", "")).strip(),
        )


@dataclass
class RevisionIntent:
    raw_feedback: str
    objective_adjustments: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    evidence_preferences: list[str] = field(default_factory=list)
    output_preferences: list[str] = field(default_factory=list)
    confidence: float = 0.0
    parser_source: str = "fallback"

    def to_dict(self) -> dict:
        return {
            "raw_feedback": self.raw_feedback,
            "objective_adjustments": list(self.objective_adjustments),
            "constraints": list(self.constraints),
            "priorities": list(self.priorities),
            "exclusions": list(self.exclusions),
            "evidence_preferences": list(self.evidence_preferences),
            "output_preferences": list(self.output_preferences),
            "confidence": float(self.confidence),
            "parser_source": self.parser_source,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "RevisionIntent | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            raw_feedback=str(payload.get("raw_feedback", "")).strip(),
            objective_adjustments=[str(x).strip() for x in payload.get("objective_adjustments", []) if str(x).strip()],
            constraints=[str(x).strip() for x in payload.get("constraints", []) if str(x).strip()],
            priorities=[str(x).strip() for x in payload.get("priorities", []) if str(x).strip()],
            exclusions=[str(x).strip() for x in payload.get("exclusions", []) if str(x).strip()],
            evidence_preferences=[str(x).strip() for x in payload.get("evidence_preferences", []) if str(x).strip()],
            output_preferences=[str(x).strip() for x in payload.get("output_preferences", []) if str(x).strip()],
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            parser_source=str(payload.get("parser_source", "fallback") or "fallback"),
        )


@dataclass
class PlanDelta:
    from_version_id: str | None
    to_version_id: str
    added_steps: list[str] = field(default_factory=list)
    removed_steps: list[str] = field(default_factory=list)
    modified_steps: list[str] = field(default_factory=list)
    reordered_steps: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "from_version_id": self.from_version_id,
            "to_version_id": self.to_version_id,
            "added_steps": list(self.added_steps),
            "removed_steps": list(self.removed_steps),
            "modified_steps": list(self.modified_steps),
            "reordered_steps": list(self.reordered_steps),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PlanDelta | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            from_version_id=str(payload.get("from_version_id", "")).strip() or None,
            to_version_id=str(payload.get("to_version_id", "")).strip(),
            added_steps=[str(x).strip() for x in payload.get("added_steps", []) if str(x).strip()],
            removed_steps=[str(x).strip() for x in payload.get("removed_steps", []) if str(x).strip()],
            modified_steps=[str(x).strip() for x in payload.get("modified_steps", []) if str(x).strip()],
            reordered_steps=[str(x).strip() for x in payload.get("reordered_steps", []) if str(x).strip()],
            summary=str(payload.get("summary", "")).strip(),
        )


@dataclass
class PlanVersion:
    version_id: str
    created_at: str
    base_from_step_index: int
    revision_intent: RevisionIntent | None = None
    steps: list[WorkflowStep] = field(default_factory=list)
    gate_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "base_from_step_index": self.base_from_step_index,
            "revision_intent": self.revision_intent.to_dict() if self.revision_intent else None,
            "steps": [step.to_dict() for step in self.steps],
            "gate_reason": self.gate_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PlanVersion | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            version_id=str(payload.get("version_id", "")).strip(),
            created_at=str(payload.get("created_at", "")).strip() or _utc_now(),
            base_from_step_index=int(payload.get("base_from_step_index", 0) or 0),
            revision_intent=RevisionIntent.from_dict(payload.get("revision_intent")),
            steps=[WorkflowStep.from_dict(step) for step in payload.get("steps", []) if isinstance(step, dict)],
            gate_reason=str(payload.get("gate_reason", "")).strip(),
        )


@dataclass
class WorkflowTask:
    """A task with explicit planner output and execution state."""

    task_id: str
    objective: str
    title: str = ""
    conversation_id: str = ""
    parent_task_id: str | None = None
    user_query: str = ""
    success_criteria: list[str] = field(default_factory=list)
    status: str = "pending"
    steps: list[WorkflowStep] = field(default_factory=list)
    current_step_index: int = -1
    awaiting_hitl: bool = False
    hitl_history: list[str] = field(default_factory=list)
    base_objective: str = ""
    plan_versions: list[PlanVersion] = field(default_factory=list)
    active_plan_version_id: str | None = None
    pending_feedback_queue: list[str] = field(default_factory=list)
    latest_plan_delta: PlanDelta | None = None
    checkpoint_state: str = "closed"
    checkpoint_reason: str = ""
    progress_events: list[dict] = field(default_factory=list)
    progress_summaries: list[dict] = field(default_factory=list)
    planner_graph: list[dict] = field(default_factory=list)
    planner_mode: str = "dynamic"
    quality_confidence: str = ""
    phase_state: dict[str, str] = field(default_factory=dict)
    event_log: list[dict] = field(default_factory=list)
    checkpoint_payload: dict = field(default_factory=dict)
    researcher_candidates: list[dict] = field(default_factory=list)
    follow_up_suggestions: list[str] = field(default_factory=list)
    context_source_task_ids: list[str] = field(default_factory=list)
    branch_label: str = ""
    internal_context_brief: str = ""
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict:
        title = (self.title or "").strip() or generate_chat_title(self.objective)
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "title": title,
            "conversation_id": str(self.conversation_id or "").strip(),
            "parent_task_id": str(self.parent_task_id or "").strip() or None,
            "user_query": str(self.user_query or "").strip(),
            "success_criteria": self.success_criteria,
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "current_step_index": self.current_step_index,
            "awaiting_hitl": self.awaiting_hitl,
            "hitl_history": self.hitl_history,
            "base_objective": self.base_objective or self.objective,
            "plan_versions": [version.to_dict() for version in self.plan_versions],
            "active_plan_version_id": self.active_plan_version_id,
            "pending_feedback_queue": list(self.pending_feedback_queue),
            "latest_plan_delta": self.latest_plan_delta.to_dict() if self.latest_plan_delta else None,
            "checkpoint_state": self.checkpoint_state,
            "checkpoint_reason": self.checkpoint_reason,
            "progress_events": list(self.progress_events),
            "progress_summaries": list(self.progress_summaries),
            "planner_graph": list(self.planner_graph),
            "planner_mode": self.planner_mode,
            "quality_confidence": self.quality_confidence,
            "phase_state": dict(self.phase_state),
            "event_log": list(self.event_log),
            "checkpoint_payload": dict(self.checkpoint_payload),
            "researcher_candidates": list(self.researcher_candidates),
            "follow_up_suggestions": [str(x).strip() for x in self.follow_up_suggestions if str(x).strip()],
            "context_source_task_ids": [str(x).strip() for x in self.context_source_task_ids if str(x).strip()],
            "branch_label": str(self.branch_label or "").strip(),
            "internal_context_brief": str(self.internal_context_brief or "").strip(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkflowTask":
        checkpoint_state_raw = str(
            payload.get("checkpoint_state", "open" if payload.get("awaiting_hitl") else "closed")
        ).strip().lower()
        checkpoint_state = "open" if checkpoint_state_raw == "open" else "closed"
        return cls(
            task_id=payload.get("task_id", ""),
            objective=payload.get("objective", ""),
            title=(str(payload.get("title", "")).strip() or generate_chat_title(payload.get("objective", ""))),
            conversation_id=str(payload.get("conversation_id", "")).strip(),
            parent_task_id=str(payload.get("parent_task_id", "")).strip() or None,
            user_query=str(payload.get("user_query", "")).strip() or str(payload.get("objective", "")).strip(),
            success_criteria=list(payload.get("success_criteria", [])),
            status=payload.get("status", "pending"),
            steps=[WorkflowStep.from_dict(step) for step in payload.get("steps", [])],
            current_step_index=payload.get("current_step_index", -1),
            awaiting_hitl=bool(payload.get("awaiting_hitl", False)),
            hitl_history=list(payload.get("hitl_history", [])),
            base_objective=payload.get("base_objective", "") or payload.get("objective", ""),
            plan_versions=[
                version
                for version in (PlanVersion.from_dict(item) for item in payload.get("plan_versions", []))
                if version is not None
            ],
            active_plan_version_id=payload.get("active_plan_version_id"),
            pending_feedback_queue=[str(x).strip() for x in payload.get("pending_feedback_queue", []) if str(x).strip()],
            latest_plan_delta=PlanDelta.from_dict(payload.get("latest_plan_delta")),
            checkpoint_state=checkpoint_state,
            checkpoint_reason=str(payload.get("checkpoint_reason", "")),
            progress_events=[item for item in payload.get("progress_events", []) if isinstance(item, dict)],
            progress_summaries=[item for item in payload.get("progress_summaries", []) if isinstance(item, dict)],
            planner_graph=[item for item in payload.get("planner_graph", []) if isinstance(item, dict)],
            planner_mode=str(payload.get("planner_mode", "dynamic") or "dynamic"),
            quality_confidence=str(payload.get("quality_confidence", "")).strip(),
            phase_state={
                str(k).strip(): str(v).strip()
                for k, v in (payload.get("phase_state", {}) or {}).items()
                if str(k).strip()
            },
            event_log=[item for item in payload.get("event_log", []) if isinstance(item, dict)],
            checkpoint_payload=dict(payload.get("checkpoint_payload", {}) or {}),
            researcher_candidates=[item for item in payload.get("researcher_candidates", []) if isinstance(item, dict)],
            follow_up_suggestions=[str(x).strip() for x in payload.get("follow_up_suggestions", []) if str(x).strip()],
            context_source_task_ids=[str(x).strip() for x in payload.get("context_source_task_ids", []) if str(x).strip()],
            branch_label=str(payload.get("branch_label", "")).strip(),
            internal_context_brief=str(payload.get("internal_context_brief", "")).strip(),
            created_at=payload.get("created_at", _utc_now()),
            updated_at=payload.get("updated_at", _utc_now()),
        )
