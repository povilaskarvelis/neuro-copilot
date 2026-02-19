"""
Event-driven phase orchestration helpers for the agentic workflow.
"""
from __future__ import annotations

from co_scientist.domain.models import WorkflowTask


PHASE_EVIDENCE = "evidence_discovery"
PHASE_RESEARCHERS = "researcher_scouting"
PHASE_SYNTHESIS = "synthesis_reporting"

EVENT_PHASE_STARTED = "phase_started"
EVENT_EVIDENCE_BATCH_READY = "evidence_batch_ready"
EVENT_PHASE_COMPLETED = "phase_completed"
EVENT_CHECKPOINT_OPENED = "checkpoint_opened"
EVENT_CHECKPOINT_APPROVED = "checkpoint_approved"
EVENT_CHECKPOINT_REVISED = "checkpoint_revised"


def infer_phase_for_step(step) -> str:
    subgoal = str(getattr(step, "subgoal_id", "") or "").strip().lower()
    observations = [str(item).strip().lower() for item in getattr(step, "observations", []) if str(item).strip()]
    for item in observations:
        if not item.startswith("phase="):
            continue
        _, _, value = item.partition("=")
        normalized = value.strip()
        if normalized in {PHASE_EVIDENCE, PHASE_RESEARCHERS, PHASE_SYNTHESIS}:
            return normalized
    if subgoal.startswith("sg_final"):
        return PHASE_SYNTHESIS
    return PHASE_EVIDENCE


def ensure_phase_state(task: WorkflowTask) -> None:
    if not isinstance(task.phase_state, dict):
        task.phase_state = {}
    for phase in (PHASE_EVIDENCE, PHASE_RESEARCHERS, PHASE_SYNTHESIS):
        task.phase_state.setdefault(phase, "pending")


def append_event(task: WorkflowTask, event_type: str, **payload) -> None:
    ensure_phase_state(task)
    task.event_log.append(
        {
            "event_type": str(event_type).strip(),
            "step_index": int(getattr(task, "current_step_index", -1) or -1),
            **payload,
        }
    )


def maybe_mark_phase_started(task: WorkflowTask, phase: str) -> None:
    ensure_phase_state(task)
    if task.phase_state.get(phase) in {"pending", ""}:
        task.phase_state[phase] = "in_progress"
        append_event(task, EVENT_PHASE_STARTED, phase=phase)


def maybe_mark_phase_completed(task: WorkflowTask, phase: str) -> None:
    ensure_phase_state(task)
    if task.phase_state.get(phase) != "completed":
        task.phase_state[phase] = "completed"
        append_event(task, EVENT_PHASE_COMPLETED, phase=phase)


def phase_completion_summary(task: WorkflowTask, phase: str) -> str:
    phase_steps = [step for step in task.steps if infer_phase_for_step(step) == phase]
    completed = [step for step in phase_steps if step.status == "completed"]
    return f"{len(completed)}/{len(phase_steps)} steps completed in {phase.replace('_', ' ')}"


def should_checkpoint_for_phase_boundary(task: WorkflowTask, next_step) -> tuple[bool, str]:
    if not next_step:
        return False, "none"
    if task.current_step_index < 0:
        return False, "none"
    if task.current_step_index >= len(task.steps):
        return False, "none"
    current = task.steps[task.current_step_index]
    current_phase = infer_phase_for_step(current)
    next_phase = infer_phase_for_step(next_step)
    if current_phase == next_phase:
        return False, "none"
    if current.status != "completed":
        return False, "none"
    return True, f"phase_boundary:{current_phase}->{next_phase}"


def checkpoint_payload_for_transition(task: WorkflowTask, from_phase: str, to_phase: str) -> dict:
    summary = phase_completion_summary(task, from_phase)
    question = (
        "I have completed "
        + from_phase.replace("_", " ")
        + f" ({summary}). Would you like me to continue to "
        + to_phase.replace("_", " ")
        + " next?"
    )
    return {
        "from_phase": from_phase,
        "to_phase": to_phase,
        "summary": summary,
        "question": question,
    }
