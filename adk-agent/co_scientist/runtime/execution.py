"""
Runtime execution, tool tracing, and retry/escalation behavior.
"""
from __future__ import annotations

import asyncio
import json
import re

from .tool_registry import ToolRegistry, infer_capabilities_from_text
from .event_orchestrator import (
    EVENT_EVIDENCE_BATCH_READY,
    append_event,
    infer_phase_for_step,
    maybe_mark_phase_completed,
    maybe_mark_phase_started,
)


STEP_SCOPE_TOOLS = {
    "search_diseases",
    "search_targets",
    "expand_disease_context",
}

VALID_MCP_RESULT_STATUSES = {"ok", "error", "not_found_or_empty", "degraded"}
SEQUENCE_FALLBACK_TRIGGER_OUTCOMES = {"error", "not_found_or_empty", "no_response", "degraded"}
GENERIC_PAYLOAD_REQUIRED_KEYS = [
    "schema",
    "result_status",
    "tool_name",
    "summary",
    "content_part_count",
    "text_excerpt",
]

TYPED_PAYLOAD_REQUIREMENTS = {
    "search_disease_targets": {
        "schema": "search_disease_targets.v1",
        "required_keys": [
            "result_status",
            "disease_id",
            "disease_name",
            "targets_returned",
            "targets",
        ],
    },
    "summarize_clinical_trials_landscape": {
        "schema": "summarize_clinical_trials_landscape.v1",
        "required_keys": [
            "result_status",
            "query",
            "studies_analyzed",
            "status_breakdown",
            "phase_breakdown",
        ],
    },
    "expand_disease_context": {
        "schema": "expand_disease_context.v1",
        "required_keys": [
            "result_status",
            "query",
            "ontology",
            "candidate_count",
            "candidates",
        ],
    },
    "search_chembl_compounds_for_target": {
        "schema": "search_chembl_compounds_for_target.v1",
        "required_keys": [
            "result_status",
            "query",
            "selected_target",
            "compounds_returned",
            "compounds",
        ],
    },
    "summarize_target_expression_context": {
        "schema": "summarize_target_expression_context.v1",
        "required_keys": [
            "result_status",
            "target_id",
            "rows_considered",
            "rows_returned",
            "expression_rows",
        ],
    },
    "summarize_target_competitive_landscape": {
        "schema": "summarize_target_competitive_landscape.v1",
        "required_keys": [
            "result_status",
            "target_id",
            "rows_analyzed",
            "phase_distribution",
            "lead_assets",
        ],
    },
    "summarize_target_safety_liabilities": {
        "schema": "summarize_target_safety_liabilities.v1",
        "required_keys": [
            "result_status",
            "target_id",
            "liabilities_analyzed",
            "unique_events",
            "events",
        ],
    },
    "rank_researchers_by_activity": {
        "schema": "rank_researchers_by_activity.v1",
        "required_keys": ["result_status", "query", "from_year", "researcher_count", "researchers"],
    },
    "infer_genetic_effect_direction": {
        "schema": "infer_genetic_effect_direction.v1",
        "required_keys": [
            "result_status",
            "gene_symbol",
            "disease_query",
            "direction_counts",
            "matched_associations",
        ],
    },
    "compare_targets_multi_axis": {
        "schema": "compare_targets_multi_axis.v1",
        "required_keys": [
            "result_status",
            "targets_requested",
            "targets_compared",
            "weights",
            "rankings",
        ],
    },
}


def _payload_requirement_for_tool(tool_name: str) -> dict | None:
    normalized = str(tool_name or "").strip()
    if not normalized:
        return None
    explicit = TYPED_PAYLOAD_REQUIREMENTS.get(normalized)
    if explicit:
        return explicit
    return {
        "schema": f"{normalized}.generic.v1",
        "required_keys": GENERIC_PAYLOAD_REQUIRED_KEYS,
    }


def extract_evidence_refs_from_text(text: str) -> list[str]:
    if not text:
        return []
    pmids = {match for match in re.findall(r"\bPMID[:\s]*([0-9]{5,9})\b", text, flags=re.IGNORECASE)}
    ncts = {match.upper() for match in re.findall(r"\b(NCT[0-9]{8})\b", text, flags=re.IGNORECASE)}
    dois = {match.lower() for match in re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)}
    openalex_urls = {match for match in re.findall(r"https?://openalex\.org/[AW]\d+", text, flags=re.IGNORECASE)}
    openalex_ids = {
        match.upper()
        for match in re.findall(r"\bhttps?://openalex\.org/([AW]\d+)\b", text, flags=re.IGNORECASE)
    }
    reactome_ids = {match for match in re.findall(r"\bR-HSA-\d+\b", text, flags=re.IGNORECASE)}
    string_ids = {match for match in re.findall(r"\b9606\.[A-Za-z0-9_.-]+\b", text, flags=re.IGNORECASE)}
    chembl_ids = {match.upper() for match in re.findall(r"\b(CHEMBL\d{3,})\b", text, flags=re.IGNORECASE)}
    mondo_ids = {match.upper().replace(":", "_") for match in re.findall(r"\bMONDO[:_]\d+\b", text, flags=re.IGNORECASE)}
    efo_ids = {match.upper().replace(":", "_") for match in re.findall(r"\bEFO[:_]\d+\b", text, flags=re.IGNORECASE)}
    rs_ids = {match.lower() for match in re.findall(r"\b(rs\d+)\b", text, flags=re.IGNORECASE)}

    refs = (
        [f"PMID:{pmid}" for pmid in sorted(pmids)]
        + sorted(ncts)
        + [f"DOI:{doi}" for doi in sorted(dois)]
        + sorted(openalex_urls)
        + [f"OpenAlex:{oid}" for oid in sorted(openalex_ids)]
        + [f"Reactome:{rid}" for rid in sorted(reactome_ids)]
        + [f"STRING:{sid}" for sid in sorted(string_ids)]
        + [f"ChEMBL:{cid}" for cid in sorted(chembl_ids)]
        + [f"MONDO:{mid}" for mid in sorted(mondo_ids)]
        + [f"EFO:{eid}" for eid in sorted(efo_ids)]
        + [f"GWAS:{rsid}" for rsid in sorted(rs_ids)]
    )
    return refs


def extract_evidence_refs_from_response_payload(response_payload) -> list[str]:
    if response_payload is None:
        return []
    snippets: list[str] = []
    pending = [response_payload]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            value = current.strip()
            if value:
                snippets.append(value)
            continue
        if isinstance(current, list):
            pending.extend(current)
            continue
        if isinstance(current, dict):
            for value in current.values():
                pending.append(value)
    if not snippets:
        return []
    return extract_evidence_refs_from_text("\n".join(snippets))


def extract_bibliography_entries_from_response_payload(response_payload) -> list[dict]:
    if response_payload is None:
        return []
    snippets: list[str] = []
    pending = [response_payload]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            value = current.strip()
            if value:
                snippets.append(value)
            continue
        if isinstance(current, list):
            pending.extend(current)
            continue
        if isinstance(current, dict):
            pending.extend(current.values())

    if not snippets:
        return []

    text = "\n".join(snippets)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    entries: list[dict] = []
    seen_keys: set[str] = set()
    doi_link_pattern = re.compile(r"\[DOI:(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\]\(https?://doi\.org/[^)]+\)", flags=re.IGNORECASE)
    bare_doi_pattern = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", flags=re.IGNORECASE)
    year_pattern = re.compile(r"\((?:19|20)\d{2}\)")
    noise_prefixes = (
        "summary:",
        "key fields:",
        "sources:",
        "limitations:",
        "retrieved ",
        "search_",
        "http://",
        "https://",
        "- http://",
        "- https://",
    )

    for raw in lines:
        line = re.sub(r"^\s*[-*]\s+", "", raw).strip()
        line = re.sub(r"^\d+[.)]\s+", "", line).strip()
        if not line:
            continue
        lower_line = line.lower()
        if any(lower_line.startswith(prefix) for prefix in noise_prefixes):
            continue
        # Drop trailing telemetry fragments like "| Cited by: 123".
        line = re.sub(r"\s*\|\s*Cited by:\s*\d+\s*$", "", line, flags=re.IGNORECASE).strip()

        doi_match = doi_link_pattern.search(line)
        doi = doi_match.group(1).lower() if doi_match else ""
        if not doi:
            bare_match = bare_doi_pattern.search(line)
            doi = bare_match.group(0).lower() if bare_match else ""

        # Keep only citation-like lines; avoid generic tool/source summaries.
        lower = line.lower()
        looks_citation = (
            bool(year_pattern.search(line))
            and (
                bool(doi)
                or "pmid:" in lower
                or "openalex" in lower
                or ". " in line
            )
        )
        if not looks_citation:
            continue

        key = doi or re.sub(r"\s+", " ", line).strip().lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        entries.append({"citation": line, "doi": doi})
        if len(entries) >= 12:
            break
    return entries


def _extract_researcher_candidates_from_text(text: str) -> list[dict]:
    content = str(text or "")
    lines = [line.strip(" -*") for line in content.splitlines() if line.strip()]
    candidates: list[dict] = []
    seen: set[str] = set()
    for line in lines:
        if ":" in line and any(token in line.lower() for token in ["activity", "affiliation", "contact", "author"]):
            name = line.split(":", 1)[0].strip()
        else:
            match = re.match(r"^([A-Z][A-Za-z'\\-]+(?:\\s+[A-Z][A-Za-z'\\-]+){1,3})\\b", line)
            if not match:
                continue
            name = match.group(1).strip()
        key = name.lower()
        if len(name) < 4 or key in seen:
            continue
        seen.add(key)
        candidates.append({"name": name, "signal": line[:220]})
        if len(candidates) >= 12:
            break
    return candidates


def validate_mcp_response_contract(response_payload) -> dict:
    issues: list[str] = []
    text_part_count = 0
    has_structured_content = False

    if not isinstance(response_payload, dict):
        issues.append("response payload is not an object.")
    else:
        content = response_payload.get("content")
        if not isinstance(content, list) or not content:
            issues.append("response.content must be a non-empty list.")
        else:
            for idx, part in enumerate(content):
                if not isinstance(part, dict):
                    issues.append(f"response.content[{idx}] is not an object.")
                    continue
                part_type = str(part.get("type", "")).strip().lower()
                text_value = part.get("text")
                if part_type and part_type != "text":
                    issues.append(f"response.content[{idx}].type should be text.")
                if isinstance(text_value, str) and text_value.strip():
                    text_part_count += 1
                else:
                    issues.append(f"response.content[{idx}].text is missing or empty.")
            if text_part_count == 0:
                issues.append("response.content contains no non-empty text parts.")
        structured_content = response_payload.get("structuredContent")
        if structured_content is None:
            issues.append("response.structuredContent is required.")
        elif not isinstance(structured_content, dict):
            issues.append("response.structuredContent must be an object.")
        else:
            has_structured_content = True
            envelope_version = str(structured_content.get("envelope_version", "")).strip()
            if not envelope_version:
                issues.append("response.structuredContent.envelope_version is required.")
            tool_name = str(structured_content.get("tool_name", "")).strip()
            if not tool_name:
                issues.append("response.structuredContent.tool_name is required.")
            status = str(structured_content.get("status", "")).strip().lower()
            if status not in VALID_MCP_RESULT_STATUSES:
                issues.append("response.structuredContent.status is invalid.")
            structured_text = structured_content.get("text")
            if not isinstance(structured_text, str) or not structured_text.strip():
                issues.append("response.structuredContent.text is required and must be non-empty.")

            payload_requirement = _payload_requirement_for_tool(tool_name)
            if payload_requirement:
                payload = structured_content.get("payload")
                if not isinstance(payload, dict):
                    issues.append(
                        "response.structuredContent.payload is required for tool "
                        f"{tool_name}."
                    )
                else:
                    expected_schema = str(payload_requirement.get("schema", "")).strip()
                    payload_schema = str(payload.get("schema", "")).strip()
                    if payload_schema != expected_schema:
                        issues.append(
                            "response.structuredContent.payload.schema must be "
                            f"{expected_schema} for tool {tool_name}."
                        )
                    for key in payload_requirement.get("required_keys", []):
                        if key not in payload:
                            issues.append(
                                "response.structuredContent.payload missing required key "
                                f"`{key}` for tool {tool_name}."
                            )
                    payload_status = str(payload.get("result_status", "")).strip().lower()
                    if payload_status and payload_status not in VALID_MCP_RESULT_STATUSES:
                        issues.append("response.structuredContent.payload.result_status is invalid.")

    return {
        "version": "mcp_response_v1",
        "valid": len(issues) == 0,
        "issues": issues[:8],
        "text_part_count": text_part_count,
        "has_structured_content": has_structured_content,
    }


def is_reasoning_only_step(task, step_idx: int) -> bool:
    return step_idx == 0 or step_idx == len(task.steps) - 1


def build_step_allowed_tools(task, step_idx: int, tool_registry: ToolRegistry | None = None) -> list[str]:
    step = task.steps[step_idx]

    # Keep scope and final synthesis steps tool-constrained.
    if is_reasoning_only_step(task, step_idx):
        return sorted(STEP_SCOPE_TOOLS) if step_idx == 0 else []

    if tool_registry is not None:
        hints = set(getattr(step, "evidence_requirements", []) or [])
        allowed_list = tool_registry.names()
        ranked = tool_registry.rank_tools(
            query=f"{step.title} {step.instruction}",
            capability_hints=hints,
            candidates=allowed_list,
            k=10,
        )
        if ranked:
            short_context = tool_registry.compact_descriptions(ranked[:6])
            step.observations = list(step.observations) + [f"tool_shortlist={', '.join(ranked[:6])}"] + short_context
            return ranked
    return sorted(set(step.recommended_tools + step.fallback_tools))


def should_escalate_allowlist(step, trace_entries: list[dict], output: str) -> bool:
    if not step.recommended_tools:
        return False
    if not trace_entries:
        return True
    outcomes = {str(entry.get("outcome", "unknown")) for entry in trace_entries}
    if outcomes and outcomes.issubset({"error", "not_found_or_empty", "no_response", "degraded"}):
        return True
    lower = (output or "").lower()
    if any(token in lower for token in ["cannot be completed", "insufficient data", "unable to identify"]):
        return True
    return False


def build_escalated_allowed_tools(
    task, step_idx: int, tool_registry: ToolRegistry | None = None
) -> list[str]:
    base = set(build_step_allowed_tools(task, step_idx, tool_registry=tool_registry))
    # Escalation broadens coverage while keeping synthesis steps tool-free.
    if is_reasoning_only_step(task, step_idx):
        return sorted(base)
    escalated = sorted(base)
    if tool_registry is not None:
        hints = set(getattr(task.steps[step_idx], "evidence_requirements", []) or [])
        if not escalated:
            escalated = tool_registry.names()
        ranked = tool_registry.rank_tools(
            query=f"{task.steps[step_idx].title} {task.steps[step_idx].instruction} escalation",
            capability_hints=hints,
            candidates=escalated,
            k=min(max(len(escalated), 1), 20),
        )
        if ranked:
            return ranked
    return escalated


def validate_step_tool_sequence(task, step, trace_entries: list[dict]) -> dict:
    entries = [entry for entry in (trace_entries or []) if isinstance(entry, dict)]
    if not entries:
        return {"valid": True, "issues": [], "priority_tools": []}

    issues: list[str] = []
    recommended_tools = [
        str(name).strip()
        for name in getattr(step, "recommended_tools", [])
        if str(name).strip()
    ]
    recommended_set = set(recommended_tools)
    called_tools = [str(entry.get("tool_name", "")).strip() for entry in entries]
    if recommended_set and not any(tool in recommended_set for tool in called_tools):
        issues.append("No recommended tool from this subgoal shortlist was attempted.")

    # Soft policy check: at least one called tool should align to evidence requirements.
    evidence_requirements = {str(item).strip() for item in getattr(step, "evidence_requirements", []) if str(item).strip()}
    if evidence_requirements:
        aligned = False
        for tool_name in called_tools:
            if not tool_name:
                continue
            inferred = infer_capabilities_from_text(tool_name)
            if inferred.intersection(evidence_requirements):
                aligned = True
                break
        if not aligned:
            issues.append(
                "Observed tool calls do not clearly align to required evidence classes "
                f"({', '.join(sorted(evidence_requirements))})."
            )

    return {
        "valid": len(issues) == 0,
        "issues": issues[:6],
        "priority_tools": [],
    }


def create_step_runner(base_runner, allowed_tools: list[str], create_agent_fn, runner_cls):
    step_agent, step_mcp_tools = create_agent_fn(tool_filter=allowed_tools)
    step_runner = runner_cls(
        agent=step_agent,
        app_name=base_runner.app_name,
        session_service=base_runner.session_service,
        artifact_service=getattr(base_runner, "artifact_service", None),
        memory_service=getattr(base_runner, "memory_service", None),
        credential_service=getattr(base_runner, "credential_service", None),
    )
    return step_runner, step_mcp_tools


async def run_runner_turn(runner, session_id: str, user_id: str, prompt: str, run_runner_turn_with_trace_fn=None) -> str:
    """Run one model turn and return text only."""
    trace_fn = run_runner_turn_with_trace_fn or run_runner_turn_with_trace
    response_text, _ = await trace_fn(runner, session_id, user_id, prompt)
    return response_text


async def run_runner_turn_with_timeout(
    runner,
    session_id: str,
    user_id: str,
    prompt: str,
    *,
    run_runner_turn_with_trace_fn,
    default_timeout_seconds: float,
    timeout_seconds: float | None = None,
) -> tuple[str, list[dict]]:
    timeout = float(default_timeout_seconds if timeout_seconds is None else timeout_seconds)
    if timeout <= 0:
        return await run_runner_turn_with_trace_fn(runner, session_id, user_id, prompt)
    try:
        return await asyncio.wait_for(
            run_runner_turn_with_trace_fn(runner, session_id, user_id, prompt),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Step model/tool turn timed out after {timeout:g}s.") from exc


def extract_missing_tool_name(error: Exception) -> str | None:
    match = re.search(r"Tool '([^']+)' not found", str(error))
    return match.group(1).strip() if match else None


def runtime_tool_constraint_suffix(tool_filter: list[str] | None) -> str:
    if tool_filter is None:
        return ""
    allowed = sorted(set(tool_filter))
    if not allowed:
        return (
            "\n\n## Runtime Tool Constraint\n"
            "No tools are available for this run. Do not emit any tool calls.\n"
            "Produce reasoning-only output."
        )
    return (
        "\n\n## Runtime Tool Constraint\n"
        "You may call ONLY these tools in this run:\n"
        f"- {', '.join(allowed)}\n"
        "Do not call any tool that is not in this list."
    )


def safe_model_dump(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=True)
        except TypeError:
            dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    return {"value": str(value)}


def normalize_trace_detail(text: str, *, max_chars: int = 260) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def compact_json(value, *, max_chars: int = 100) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(value)
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def summarize_for_report(text: str, *, max_chars: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    if len(sentence) <= max_chars:
        return sentence
    return f"{sentence[: max_chars - 3].rstrip()}..."


def format_step_execution_error(
    error: Exception,
    allowed_tools: list[str],
    *,
    normalize_trace_detail_fn=normalize_trace_detail,
) -> str:
    allowed_text = ", ".join(allowed_tools) if allowed_tools else "none"
    detail = normalize_trace_detail_fn(str(error), max_chars=420)
    return (
        "Step execution issue encountered.\n"
        f"Details: {detail}\n"
        f"Allowed tools for this step: {allowed_text}\n"
        "Continuing with best-effort output under current constraints."
    )


def populate_step_rao_fields(
    step,
    *,
    summarize_for_report_fn=summarize_for_report,
    compact_json_fn=compact_json,
) -> None:
    """Populate explicit Reasoning/Actions/Observations fields for report rendering."""
    reasoning = summarize_for_report_fn(step.output) or summarize_for_report_fn(step.instruction, max_chars=180)
    actions: list[str] = []
    observations: list[str] = []

    trace_entries = step.tool_trace if isinstance(step.tool_trace, list) else []
    if trace_entries:
        for entry in trace_entries[:4]:
            tool_name = str(entry.get("tool_name", "unknown_tool"))
            outcome = str(entry.get("outcome", "unknown"))
            args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
            args_text = f" args={compact_json_fn(args, max_chars=80)}" if args else ""
            actions.append(f"Called `{tool_name}` ({outcome}).{args_text}")
        omitted = len(trace_entries) - 4
        if omitted > 0:
            actions.append(f"{omitted} additional tool call(s) omitted for brevity.")
    else:
        actions.append("No tool calls executed; step completed as reasoning/synthesis.")

    output_summary = summarize_for_report_fn(step.output, max_chars=260)
    if output_summary:
        observations.append(output_summary)
    if step.evidence_refs:
        preview = ", ".join(step.evidence_refs[:5])
        suffix = f", +{len(step.evidence_refs) - 5} more" if len(step.evidence_refs) > 5 else ""
        observations.append(f"Citation IDs captured: {preview}{suffix}.")
    issue_count = sum(
        1
        for entry in trace_entries
        if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
    )
    if issue_count:
        observations.append(f"{issue_count} tool call(s) returned errors/empty responses.")
    if not observations:
        observations.append("No additional observations captured.")

    step.reasoning_summary = reasoning or "Reasoning summary unavailable."
    step.actions = actions
    step.observations = observations


def extract_response_excerpt(response_payload, *, normalize_trace_detail_fn=normalize_trace_detail) -> str:
    if not isinstance(response_payload, dict):
        return "No structured response payload captured."

    content = response_payload.get("content")
    snippets: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    snippets.append(text)

    if not snippets:
        output_payload = response_payload.get("output")
        if isinstance(output_payload, str) and output_payload.strip():
            snippets.append(output_payload)
        elif output_payload is not None:
            snippets.append(str(output_payload))

    if not snippets:
        error_payload = response_payload.get("error")
        if error_payload:
            snippets.append(str(error_payload))

    if not snippets:
        snippets.append(str(response_payload))
    return normalize_trace_detail_fn(" | ".join(snippets))


def classify_tool_response(
    response_payload,
    *,
    normalize_trace_detail_fn=normalize_trace_detail,
    extract_response_excerpt_fn=extract_response_excerpt,
) -> tuple[str, str]:
    if response_payload is None:
        return "no_response", "Tool call was issued but no response payload was returned."
    if not isinstance(response_payload, dict):
        return "unknown", normalize_trace_detail_fn(str(response_payload))

    excerpt = extract_response_excerpt_fn(response_payload)

    explicit_error = bool(response_payload.get("error")) or response_payload.get("isError") is True
    if explicit_error:
        return "error", excerpt

    lower = excerpt.lower()
    if lower.startswith("error in ") or lower.startswith("error:") or "request failed (" in lower:
        return "error", excerpt
    not_found_markers = (
        "not found",
        "no results",
        "no matching",
        "no records",
        "no data found",
        "no target data found",
        "no clinical trials found",
        "no expression context found",
        "couldn't find",
        "unable to find",
        "did not find",
        "no evidence found",
    )
    if any(marker in lower for marker in not_found_markers):
        return "not_found_or_empty", excerpt

    degraded_markers = (
        "critical gap",
        "service unavailable",
        "underlying gwas call error",
        "fallback uses open targets genetics evidence scores",
        "risk-increasing vs protective direction cannot be inferred",
        "could not infer genetic direction-of-effect",
    )
    if any(marker in lower for marker in degraded_markers):
        return "degraded", excerpt

    return "ok", excerpt


async def run_runner_turn_with_trace(
    runner,
    session_id: str,
    user_id: str,
    prompt: str,
    *,
    safe_model_dump_fn=safe_model_dump,
    classify_tool_response_fn=classify_tool_response,
) -> tuple[str, list[dict]]:
    """Run one model turn and collect both text output and exact tool trace."""
    from google.genai.types import Content, Part

    message = Content(role="user", parts=[Part(text=prompt)])
    response_text = ""
    trace_entries: list[dict] = []
    pending_by_call_id: dict[str, int] = {}
    pending_by_tool_name: dict[str, list[int]] = {}
    sequence = 0

    async for event in runner.run_async(
        session_id=session_id,
        user_id=user_id,
        new_message=message,
    ):
        if not hasattr(event, "content") or not event.content or not hasattr(event.content, "parts"):
            continue
        if not event.content.parts:
            continue
        for part in event.content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text

            function_call = getattr(part, "function_call", None)
            if function_call:
                payload = safe_model_dump_fn(function_call)
                sequence += 1
                call_id = str(payload.get("id") or f"call-{sequence}")
                tool_name = str(payload.get("name") or "unknown_tool")
                args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
                entry = {
                    "sequence": sequence,
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "args": args,
                    "outcome": "pending",
                    "detail": "",
                    "evidence_refs": [],
                    "bibliography": [],
                    "response_contract": None,
                    "phase": "main",
                }
                trace_entries.append(entry)
                pending_by_call_id[call_id] = len(trace_entries) - 1
                pending_by_tool_name.setdefault(tool_name, []).append(len(trace_entries) - 1)

            function_response = getattr(part, "function_response", None)
            if function_response:
                payload = safe_model_dump_fn(function_response)
                call_id = str(payload.get("id") or "")
                tool_name = str(payload.get("name") or "unknown_tool")
                response_payload = payload.get("response")
                outcome, detail = classify_tool_response_fn(response_payload)
                evidence_refs = extract_evidence_refs_from_response_payload(response_payload)
                bibliography = extract_bibliography_entries_from_response_payload(response_payload)
                response_contract = validate_mcp_response_contract(response_payload)

                target_index = pending_by_call_id.get(call_id) if call_id else None
                if target_index is None:
                    for candidate_index in pending_by_tool_name.get(tool_name, []):
                        if trace_entries[candidate_index].get("outcome") == "pending":
                            target_index = candidate_index
                            break

                if target_index is None:
                    sequence += 1
                    trace_entries.append(
                        {
                            "sequence": sequence,
                            "call_id": call_id or f"response-{sequence}",
                            "tool_name": tool_name,
                            "args": {},
                            "outcome": outcome,
                            "detail": detail,
                            "evidence_refs": evidence_refs,
                            "bibliography": bibliography,
                            "response_contract": response_contract,
                            "phase": "main",
                        }
                    )
                    continue

                trace_entries[target_index]["outcome"] = outcome
                trace_entries[target_index]["detail"] = detail
                trace_entries[target_index]["evidence_refs"] = evidence_refs
                trace_entries[target_index]["bibliography"] = bibliography
                trace_entries[target_index]["response_contract"] = response_contract

    for entry in trace_entries:
        if entry.get("outcome") == "pending":
            entry["outcome"] = "no_response"
            entry["detail"] = "Tool call was issued but no matching function_response event was captured."
            entry["response_contract"] = {
                "version": "mcp_response_v1",
                "valid": False,
                "issues": ["No function_response event was captured for this tool call."],
                "text_part_count": 0,
                "has_structured_content": False,
            }
            entry["evidence_refs"] = []
            entry["bibliography"] = []

    return response_text.strip(), trace_entries


async def execute_step(
    runner,
    session_id: str,
    user_id: str,
    task,
    step_idx: int,
    *,
    step_prompt_fn,
    extract_evidence_refs_fn,
    build_step_allowed_tools_fn,
    create_step_runner_fn,
    run_runner_turn_with_timeout_fn,
    extract_missing_tool_name_fn=extract_missing_tool_name,
    format_step_execution_error_fn=format_step_execution_error,
    should_escalate_allowlist_fn=should_escalate_allowlist,
    build_escalated_allowed_tools_fn=None,
    validate_step_tool_sequence_fn=validate_step_tool_sequence,
    populate_step_rao_fields_fn=populate_step_rao_fields,
) -> str:
    """Execute a single workflow step and update task status."""
    if build_escalated_allowed_tools_fn is None:
        raise ValueError("build_escalated_allowed_tools_fn is required")

    step = task.steps[step_idx]
    current_phase = infer_phase_for_step(step)
    maybe_mark_phase_started(task, current_phase)
    task.status = "in_progress"
    task.current_step_index = step_idx
    step.status = "in_progress"
    task.touch()

    step.allowed_tools = build_step_allowed_tools_fn(task, step_idx)
    prompt = step_prompt_fn(task, step)

    step_failed = False
    step_runner, step_mcp_tools = create_step_runner_fn(runner, step.allowed_tools)
    try:
        try:
            output, trace_entries = await run_runner_turn_with_timeout_fn(
                step_runner,
                session_id,
                user_id,
                prompt,
            )
        except Exception as exc:
            step_failed = True
            missing_tool = extract_missing_tool_name_fn(exc)
            if missing_tool:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: The prior attempt called unavailable tool `{missing_tool}`.\n"
                    "Do not call unavailable tools. Use only the allowed-tools list above. "
                    "If no allowed tool is relevant, provide reasoning-only output for this step."
                )
                try:
                    output, trace_entries = await run_runner_turn_with_timeout_fn(
                        step_runner,
                        session_id,
                        user_id,
                        retry_prompt,
                    )
                    for entry in trace_entries:
                        entry["phase"] = "retry_after_missing_tool"
                    step_failed = False
                except Exception as retry_exc:
                    output = format_step_execution_error_fn(retry_exc, step.allowed_tools)
                    trace_entries = []
            else:
                output = format_step_execution_error_fn(exc, step.allowed_tools)
                trace_entries = []
    finally:
        if step_mcp_tools:
            await step_mcp_tools.close()

    if should_escalate_allowlist_fn(step, trace_entries, output):
        escalated_tools = build_escalated_allowed_tools_fn(task, step_idx)
        if set(escalated_tools) != set(step.allowed_tools):
            step.allowed_tools = escalated_tools
            escalated_runner, escalated_mcp_tools = create_step_runner_fn(runner, step.allowed_tools)
            try:
                try:
                    escalated_output, escalated_trace = await run_runner_turn_with_timeout_fn(
                        escalated_runner,
                        session_id,
                        user_id,
                        prompt,
                    )
                except Exception as exc:
                    step_failed = True
                    escalated_output = format_step_execution_error_fn(exc, step.allowed_tools)
                    escalated_trace = []
            finally:
                if escalated_mcp_tools:
                    await escalated_mcp_tools.close()
            for entry in escalated_trace:
                entry["phase"] = "step_allowlist_escalation"
            if escalated_trace:
                trace_entries.extend(escalated_trace)
            if escalated_output:
                output = escalated_output

    sequence_report = validate_step_tool_sequence_fn(task, step, trace_entries)
    step.critic_verdict = "pass" if sequence_report.get("valid", True) else "needs_revision"
    if trace_entries and not sequence_report.get("valid", True):
        issue_preview = "; ".join(
            str(item).strip() for item in (sequence_report.get("issues") or [])[:2] if str(item).strip()
        ) or "tool policy constraints were not satisfied."
        if output:
            output = (
                f"{output}\n\n"
                "Tool policy warning: "
                f"{issue_preview}"
            )

    step.output = output if output else "(No response generated)"
    if int(getattr(step, "max_tool_calls", 0) or 0) > 0 and len(trace_entries) > int(step.max_tool_calls):
        step_failed = True
        step.output = (
            f"{step.output}\n\n"
            f"Budget notice: step used {len(trace_entries)} tool calls, exceeding max_tool_calls={int(step.max_tool_calls)}."
        )
    output_refs = extract_evidence_refs_fn(step.output)
    trace_refs = sorted(
        {
            str(ref).strip()
            for entry in trace_entries
            for ref in (entry.get("evidence_refs") or [])
            if str(ref).strip()
        }
    )
    step.evidence_refs = sorted(set(output_refs).union(trace_refs))
    step.tool_trace = trace_entries
    populate_step_rao_fields_fn(step)
    step.status = "blocked" if step_failed else ("completed" if output else "blocked")
    if step.status == "completed":
        append_event(
            task,
            EVENT_EVIDENCE_BATCH_READY,
            phase=current_phase,
            step_id=step.step_id,
            subgoal_id=step.subgoal_id,
            evidence_refs=list(step.evidence_refs),
        )
        # Mark phase complete when all steps in that phase are completed.
        if all(s.status == "completed" for s in task.steps if infer_phase_for_step(s) == current_phase):
            maybe_mark_phase_completed(task, current_phase)
    if current_phase == "researcher_scouting":
        extracted = _extract_researcher_candidates_from_text(step.output)
        merged = list(task.researcher_candidates or [])
        seen_names = {str(item.get("name", "")).strip().lower() for item in merged if isinstance(item, dict)}
        for item in extracted:
            key = str(item.get("name", "")).strip().lower()
            if not key or key in seen_names:
                continue
            merged.append(item)
            seen_names.add(key)
        task.researcher_candidates = merged[:20]
    task.progress_events.append(
        {
            "event_type": "step_completed",
            "step_id": step.step_id,
            "subgoal_id": step.subgoal_id,
            "status": step.status,
            "critic_verdict": step.critic_verdict,
            "tool_calls": len(trace_entries),
            "confidence_label": step.confidence_label or task.quality_confidence,
        }
    )
    task.touch()
    return step.output
