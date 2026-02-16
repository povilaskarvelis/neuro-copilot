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
    if "researcher_discovery" in task.intent_tags:
        return [
            "Query disease/topic context and confirm timeframe.",
            "Identify topic-matched publications.",
            "Extract authors and affiliations.",
            "Rank researchers by activity and impact.",
        ]
    return [
        "Confirm scope and concrete entities.",
        "Gather evidence from primary sources.",
        "Synthesize findings into a direct answer.",
    ]


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

    if "researcher_discovery" in task.intent_tags:
        focus = extract_focus_from_step_output(step_output)
        lead = (
            f"To find the top researchers in {focus}, I will:"
            if focus
            else "To find the top researchers for your query, I will:"
        )
    else:
        lead = "To answer your request, I will:"

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
