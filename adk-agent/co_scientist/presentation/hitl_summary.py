"""Rendering helpers for HITL checkpoint scope summaries."""
from __future__ import annotations

import re

from co_scientist.domain.models import WorkflowTask


def clean_model_text(value: str) -> str:
    text = (value or "").strip()
    text = re.sub(r"`+", "", text)
    text = text.replace("**", "")
    text = text.replace("*", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_labeled_value(text: str, labels: list[str]) -> str | None:
    lowered_labels = [label.lower() for label in labels]
    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", stripped)
        cleaned = clean_model_text(stripped)
        lower = cleaned.lower()
        for label in lowered_labels:
            if not lower.startswith(label):
                continue
            if ":" in cleaned:
                candidate = cleaned.split(":", 1)[1].strip()
                if candidate:
                    return candidate
    return None


def extract_decomposition_subtasks(text: str) -> list[str]:
    if not text:
        return []
    lines = [line.rstrip() for line in text.splitlines()]
    capture = False
    items: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if capture and items:
                break
            continue

        normalized = clean_model_text(stripped).lower().replace(" ", "_")
        if any(token in normalized for token in ["decomposition_subtasks", "decomposition", "subtasks", "sub_tasks"]):
            capture = True
            continue

        bullet_match = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", stripped)
        if not bullet_match:
            if capture and items:
                break
            continue
        if not capture:
            continue
        item = clean_model_text(bullet_match.group(1))
        if item:
            if ":" in item and len(item.split(":", 1)[0].split()) <= 5:
                item = item.split(":", 1)[1].strip() or item
            items.append(item)
    return items


def default_hitl_subtasks(task: WorkflowTask) -> list[str]:
    # Build fallback subtasks from the current dynamic plan state instead of intent hardcoding.
    steps = [step for step in task.steps if step.status != "completed" and step.subgoal_id != "sg_final_report"]
    if not steps:
        steps = [step for step in task.steps if step.subgoal_id != "sg_final_report"]

    subtasks: list[str] = []
    for step in steps:
        source = clean_model_text(step.instruction or step.title)
        if not source:
            continue
        first_sentence = re.split(r"(?<=[.!?])\s+", source, maxsplit=1)[0].strip()
        if len(first_sentence) < 20:
            first_sentence = clean_model_text(step.title)
        if first_sentence and first_sentence not in subtasks:
            subtasks.append(first_sentence.rstrip(".") + ".")
        if len(subtasks) >= 5:
            break

    if subtasks:
        return subtasks

    # Final fallback: derive simple action list from objective clauses.
    objective = clean_model_text(task.objective or "")
    clauses = [part.strip(" .") for part in re.split(r"[.;]\s+|\bthen\b|\band then\b", objective, flags=re.IGNORECASE)]
    for clause in clauses:
        if len(clause) < 12:
            continue
        subtasks.append(clause.rstrip(".") + ".")
        if len(subtasks) >= 3:
            break
    return subtasks or ["Review request scope.", "Gather supporting evidence.", "Prepare a recommendation."]


def extract_focus_from_step_output(step_output: str) -> str | None:
    disease = extract_labeled_value(step_output, ["Disease Area", "Disease", "Focus"])
    if not disease:
        return None
    focus = clean_model_text(disease)
    focus = re.sub(r"\s*\([^)]*[_:]\d+[^)]*\)\s*", "", focus).strip()
    focus = re.sub(r"\s+", " ", focus).strip(" .")
    return focus or None


def render_hitl_scope_summary(task: WorkflowTask, step_output: str) -> str:
    subtasks = extract_decomposition_subtasks(step_output)
    if len(subtasks) < 2:
        subtasks = default_hitl_subtasks(task)

    focus = extract_focus_from_step_output(step_output)
    if focus:
        lead = f"Planned next steps for {focus}:"
    else:
        lead = "Planned next steps:"

    lines = [lead]
    for idx, subtask in enumerate(subtasks[:5], start=1):
        lines.append(f"{idx}. {subtask}")
    if len(subtasks) > 5:
        lines.append(f"... plus {len(subtasks) - 5} additional sub-task(s).")
    return "\n".join(lines)


__all__ = [
    "clean_model_text",
    "default_hitl_subtasks",
    "extract_decomposition_subtasks",
    "extract_focus_from_step_output",
    "extract_labeled_value",
    "render_hitl_scope_summary",
]
