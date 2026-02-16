"""Revision-intent parsing and objective merge helpers."""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from co_scientist.domain.models import RevisionIntent

RunRunnerTurnFn = Callable[[object, str, str, str], Awaitable[str]]


def extract_revision_directives(revised_scope: str) -> list[str]:
    text = revised_scope.strip()
    if not text:
        return []

    normalized = re.sub(r"\r\n?", "\n", text)
    parts: list[str] = []
    for line in normalized.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^(?:[-*]|\d+[.)])\s*", "", cleaned).strip()
        if not cleaned:
            continue
        segments = [segment.strip() for segment in cleaned.split(";") if segment.strip()]
        parts.extend(segments if segments else [cleaned])

    directives: list[str] = []
    for part in parts:
        collapsed = re.sub(r"\s+", " ", part).strip(" .")
        if not collapsed:
            continue
        if collapsed.lower().startswith("and "):
            collapsed = collapsed[4:].strip()
        if len(collapsed) < 3:
            continue
        if collapsed not in directives:
            directives.append(collapsed)
        if len(directives) >= 6:
            break
    return directives


def merge_objective_with_revision(original_objective: str, revised_scope: str) -> str:
    base = original_objective.strip()
    revision = revised_scope.strip()
    if not base:
        return revision
    if not revision:
        return base
    directives = extract_revision_directives(revision)
    directives_block = ""
    if directives:
        rendered_directives = "\n".join(f"- {item}" for item in directives)
        directives_block = (
            "\nRevision directives to apply:\n"
            f"{rendered_directives}\n"
            "Treat these directives as mandatory when rebuilding and executing the workflow."
        )
    return (
        f"{base}\n"
        f"User revision to scope/decomposition: {revision}\n"
        "Treat this revision as the authoritative update when rebuilding the workflow."
        f"{directives_block}"
    )


def extract_revision_directive_from_objective(objective: str) -> str | None:
    match = re.search(r"User revision to scope/decomposition:\s*(.+)", objective or "", flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def extract_timeframe_hint(text: str) -> str | None:
    if not text:
        return None
    range_match = re.search(
        r"\b((?:19|20)\d{2})\s*(?:-|–|to|through)\s*((?:19|20)\d{2})\b",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))
        if start_year <= end_year:
            return f"{start_year}-{end_year}"
    current_year_match = re.search(r"\bcurrent year is\s+((?:19|20)\d{2})\b", text, flags=re.IGNORECASE)
    if current_year_match:
        return f"up to {current_year_match.group(1)}"
    relative_match = re.search(
        r"\b(?:last|past)\s+(\d{1,2})(?:\s*(?:-|to)\s*(\d{1,2}))?\s+years?\b",
        text,
        flags=re.IGNORECASE,
    )
    if relative_match:
        first = relative_match.group(1)
        second = relative_match.group(2)
        return f"last {first}-{second} years" if second else f"last {first} years"
    return None


def extract_primary_objective_text(objective: str) -> str:
    text = (objective or "").strip()
    markers = [
        "\nUser revision to scope/decomposition:",
        "\nUser clarification:",
        "\nUse this clarification as the intended meaning for ambiguous abbreviations.",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else text


def extract_json_payload(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        payload = json.loads(snippet)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def dedupe_compact(items: list[str], *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in items:
        value = re.sub(r"\s+", " ", str(item or "")).strip(" .")
        if not value:
            continue
        lowered = value.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(value)
        if len(cleaned) >= limit:
            break
    return cleaned


def coerce_str_list(value) -> list[str]:
    if isinstance(value, list):
        raw_items = [str(item) for item in value]
    elif isinstance(value, str):
        raw_items = re.split(r"[\n;]+", value)
    else:
        raw_items = []
    return dedupe_compact(raw_items)


def build_deterministic_revision_intent(feedback: str) -> RevisionIntent:
    parts = extract_revision_directives(feedback)
    constraints: list[str] = []
    priorities: list[str] = []
    exclusions: list[str] = []
    evidence_prefs: list[str] = []
    output_prefs: list[str] = []
    objective_adjustments: list[str] = []

    for item in parts:
        lowered = item.lower()
        if any(token in lowered for token in ("do not", "don't", "exclude", "avoid", "without")):
            exclusions.append(item)
            continue
        if any(token in lowered for token in ("must", "need to", "required", "should")):
            constraints.append(item)
            continue
        if any(token in lowered for token in ("prioritize", "focus", "first", "before")):
            priorities.append(item)
            continue
        if any(token in lowered for token in ("evidence", "citation", "pmid", "nct", "source")):
            evidence_prefs.append(item)
            continue
        if any(token in lowered for token in ("format", "table", "output", "report", "summary")):
            output_prefs.append(item)
            continue
        objective_adjustments.append(item)

    return RevisionIntent(
        raw_feedback=feedback.strip(),
        objective_adjustments=dedupe_compact(objective_adjustments),
        constraints=dedupe_compact(constraints),
        priorities=dedupe_compact(priorities),
        exclusions=dedupe_compact(exclusions),
        evidence_preferences=dedupe_compact(evidence_prefs),
        output_preferences=dedupe_compact(output_prefs),
        confidence=0.45,
        parser_source="fallback",
    )


async def parse_revision_intent(
    feedback: str,
    *,
    feedback_parser_runner=None,
    feedback_parser_session_id: str | None = None,
    user_id: str = "researcher",
    run_runner_turn_fn: RunRunnerTurnFn | None = None,
) -> RevisionIntent:
    deterministic = build_deterministic_revision_intent(feedback)
    if feedback_parser_runner is None or not feedback_parser_session_id or run_runner_turn_fn is None:
        return deterministic

    prompt = (
        "Parse this checkpoint feedback into structured intent updates.\n"
        f"Feedback: {feedback}\n"
        "Return strict JSON only."
    )
    try:
        raw = await run_runner_turn_fn(feedback_parser_runner, feedback_parser_session_id, user_id, prompt)
        payload = extract_json_payload(raw)
    except Exception:
        payload = None
    if not payload:
        return deterministic

    try:
        confidence = float(payload.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    parsed = RevisionIntent(
        raw_feedback=feedback.strip(),
        objective_adjustments=coerce_str_list(payload.get("objective_adjustments")),
        constraints=coerce_str_list(payload.get("constraints")),
        priorities=coerce_str_list(payload.get("priorities")),
        exclusions=coerce_str_list(payload.get("exclusions")),
        evidence_preferences=coerce_str_list(payload.get("evidence_preferences")),
        output_preferences=coerce_str_list(payload.get("output_preferences")),
        confidence=confidence,
        parser_source="model",
    )

    has_signal = any(
        (
            parsed.objective_adjustments,
            parsed.constraints,
            parsed.priorities,
            parsed.exclusions,
            parsed.evidence_preferences,
            parsed.output_preferences,
        )
    )
    if not has_signal:
        return deterministic
    return parsed


def render_revision_intent_as_text(intent: RevisionIntent) -> str:
    lines = []
    if intent.objective_adjustments:
        lines.append("Objective adjustments:")
        lines.extend([f"- {item}" for item in intent.objective_adjustments])
    if intent.constraints:
        lines.append("Constraints:")
        lines.extend([f"- {item}" for item in intent.constraints])
    if intent.priorities:
        lines.append("Priorities:")
        lines.extend([f"- {item}" for item in intent.priorities])
    if intent.exclusions:
        lines.append("Exclusions:")
        lines.extend([f"- {item}" for item in intent.exclusions])
    if intent.evidence_preferences:
        lines.append("Evidence preferences:")
        lines.extend([f"- {item}" for item in intent.evidence_preferences])
    if intent.output_preferences:
        lines.append("Output preferences:")
        lines.extend([f"- {item}" for item in intent.output_preferences])
    return "\n".join(lines).strip()


def merge_objective_with_revision_intent(original_objective: str, intent: RevisionIntent) -> str:
    base = (original_objective or "").strip()
    if not base:
        base = (intent.raw_feedback or "").strip()
    intent_text = render_revision_intent_as_text(intent)
    if not intent_text:
        return merge_objective_with_revision(base, intent.raw_feedback)
    return (
        f"{base}\n"
        f"User revision to scope/decomposition: {intent.raw_feedback.strip()}\n"
        "Revision directives to apply:\n"
        f"{intent_text}\n"
        "Treat these directives as mandatory when rebuilding and executing the workflow."
    )


def merge_revision_intents(previous: RevisionIntent | None, incoming: RevisionIntent) -> RevisionIntent:
    merged = RevisionIntent(
        raw_feedback=incoming.raw_feedback.strip() or (previous.raw_feedback if previous else ""),
        objective_adjustments=[],
        constraints=[],
        priorities=[],
        exclusions=[],
        evidence_preferences=[],
        output_preferences=[],
        confidence=max(float(previous.confidence if previous else 0.0), float(incoming.confidence)),
        parser_source=incoming.parser_source or (previous.parser_source if previous else "fallback"),
    )
    merged.objective_adjustments = dedupe_compact(
        [
            *(previous.objective_adjustments if previous else []),
            *incoming.objective_adjustments,
        ]
    )
    merged.constraints = dedupe_compact(
        [
            *(previous.constraints if previous else []),
            *incoming.constraints,
        ]
    )
    merged.priorities = dedupe_compact(
        [
            *(previous.priorities if previous else []),
            *incoming.priorities,
        ]
    )
    merged.exclusions = dedupe_compact(
        [
            *(previous.exclusions if previous else []),
            *incoming.exclusions,
        ]
    )
    merged.evidence_preferences = dedupe_compact(
        [
            *(previous.evidence_preferences if previous else []),
            *incoming.evidence_preferences,
        ]
    )
    merged.output_preferences = dedupe_compact(
        [
            *(previous.output_preferences if previous else []),
            *incoming.output_preferences,
        ]
    )
    return merged


__all__ = [
    "build_deterministic_revision_intent",
    "coerce_str_list",
    "dedupe_compact",
    "extract_json_payload",
    "extract_primary_objective_text",
    "extract_revision_directive_from_objective",
    "extract_revision_directives",
    "extract_timeframe_hint",
    "merge_objective_with_revision",
    "merge_objective_with_revision_intent",
    "merge_revision_intents",
    "parse_revision_intent",
    "render_revision_intent_as_text",
]
