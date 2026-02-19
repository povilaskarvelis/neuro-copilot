"""Clarification and query-normalization helpers for workflow planning."""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

QUERY_TYPO_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\breseachers\b", "researchers"),
    (r"\breseacher\b", "researcher"),
    (r"\bresearchs\b", "researchers"),
    (r"\bwich\b", "which"),
    (r"\btime\s*frame\b", "timeframe"),
)

AMBIGUOUS_ABBREVIATIONS: dict[str, dict[str, list[str]]] = {
    "ER": {
        "options": ["Estrogen receptor (ESR1/ESR2)", "Endoplasmic reticulum pathway"],
        "disambiguators": ["estrogen receptor", "esr1", "esr2", "endoplasmic reticulum", "er stress"],
    },
    "AD": {
        "options": ["Alzheimer disease", "Atopic dermatitis", "Autosomal dominant context"],
        "disambiguators": ["alzheimer", "atopic dermatitis", "autosomal dominant"],
    },
    "PD": {
        "options": ["Parkinson disease", "Pharmacodynamics", "PD-1/PD-L1 axis context"],
        "disambiguators": ["parkinson", "pharmacodynamic", "pd-1", "pd-l1", "programmed death"],
    },
    "MS": {
        "options": ["Multiple sclerosis", "Mass spectrometry"],
        "disambiguators": ["multiple sclerosis", "mass spectrometry", "proteomics"],
    },
    "RA": {
        "options": ["Rheumatoid arthritis", "Retinoic acid signaling"],
        "disambiguators": ["rheumatoid arthritis", "retinoic acid"],
    },
}

RunRunnerTurnFn = Callable[[object, str, str, str], Awaitable[str]]


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


def find_ambiguous_abbreviations(query: str) -> list[tuple[str, list[str]]]:
    lowered = query.lower()
    matches: list[tuple[str, list[str]]] = []
    for abbr, cfg in AMBIGUOUS_ABBREVIATIONS.items():
        if not re.search(rf"\b{abbr}\b", query):
            continue
        if any(hint in lowered for hint in cfg["disambiguators"]):
            continue
        matches.append((abbr, cfg["options"]))
    return matches


def merge_query_with_clarification(original_query: str, clarification: str) -> str:
    return (
        f"{original_query}\n"
        f"User clarification: {clarification.strip()}\n"
        "Use this clarification as the intended meaning for ambiguous abbreviations."
    )


def contains_malformed_biomedical_identifier(query: str) -> bool:
    tokens = re.findall(r"\b[A-Za-z0-9_]+\b", query)
    for token in tokens:
        upper = token.upper()
        if upper.startswith("ENSG") and not re.fullmatch(r"ENSG\d{11}", upper):
            return True
        if upper.startswith("MONDO_") and not re.fullmatch(r"MONDO_\d{7}", upper):
            return True
        if upper.startswith("EFO_") and not re.fullmatch(r"EFO_\d+", upper):
            return True
        if upper.startswith("NCT") and not re.fullmatch(r"NCT\d{8}", upper):
            return True
        if upper.startswith("PMID") and not re.fullmatch(r"PMID\d{5,9}", upper):
            return True
    return False


def looks_like_low_value_typo_clarification(query: str, questions: list[str], reason: str) -> bool:
    combined = " ".join(questions + [reason]).lower()
    typo_like = any(
        phrase in combined
        for phrase in ["did you mean", "did you intend", "possible typo", "spelling", "misspell"]
    )
    if not typo_like:
        return False
    if contains_malformed_biomedical_identifier(query):
        return False
    return True


async def build_model_clarification_request(
    clarifier_runner,
    clarifier_session_id: str,
    user_id: str,
    query: str,
    *,
    run_runner_turn_fn: RunRunnerTurnFn,
) -> str | None:
    prompt = (
        "Analyze whether this biomedical query requires clarification before tools run.\n"
        f"Query: {query}\n"
        "Return strict JSON only."
    )
    raw = await run_runner_turn_fn(clarifier_runner, clarifier_session_id, user_id, prompt)
    payload = extract_json_payload(raw)
    if not payload:
        return None

    needs_clarification = bool(payload.get("needs_clarification"))
    if not needs_clarification:
        return None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.8:
        return None

    questions = [str(q).strip() for q in payload.get("questions", []) if str(q).strip()]
    if not questions:
        return None
    reason = str(payload.get("reason", "")).strip()
    if looks_like_low_value_typo_clarification(query, questions, reason):
        return None

    lines = ["I need a quick clarification before I run tools:"]
    for question in questions[:2]:
        if question.startswith("-"):
            lines.append(question)
        else:
            lines.append(f"- {question}")
    lines.append("Reply with a short clarification and I will continue.")
    return "\n".join(lines)


async def build_clarification_request(
    query: str,
    *,
    clarifier_runner=None,
    clarifier_session_id: str | None = None,
    user_id: str = "researcher",
    run_runner_turn_fn: RunRunnerTurnFn | None = None,
) -> str | None:
    if clarifier_runner is None or not clarifier_session_id or run_runner_turn_fn is None:
        return None
    return await build_model_clarification_request(
        clarifier_runner,
        clarifier_session_id,
        user_id,
        query,
        run_runner_turn_fn=run_runner_turn_fn,
    )


def normalize_user_query(query: str) -> str:
    normalized = query.strip()
    for pattern, replacement in QUERY_TYPO_REPLACEMENTS:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", normalized).strip()


__all__ = [
    "AMBIGUOUS_ABBREVIATIONS",
    "QUERY_TYPO_REPLACEMENTS",
    "build_clarification_request",
    "build_model_clarification_request",
    "contains_malformed_biomedical_identifier",
    "extract_json_payload",
    "find_ambiguous_abbreviations",
    "looks_like_low_value_typo_clarification",
    "merge_query_with_clarification",
    "normalize_user_query",
]
