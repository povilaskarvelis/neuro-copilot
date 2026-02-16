"""
Workflow models and rendering helpers for the Co-Investigator mode.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import re

from co_scientist.domain.models import (
    PlanDelta,
    PlanVersion,
    RevisionIntent,
    WorkflowStep,
    WorkflowTask,
    _utc_now,
    generate_chat_title,
)
from co_scientist.planning import workflow_planning as _planning


def extract_evidence_refs(text: str) -> list[str]:
    pmids = {match for match in re.findall(r"\bPMID[:\s]*([0-9]{5,9})\b", text, flags=re.IGNORECASE)}
    ncts = {match.upper() for match in re.findall(r"\b(NCT[0-9]{8})\b", text)}
    dois = {match.lower() for match in re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)}
    openalex_urls = {match for match in re.findall(r"https?://openalex\.org/[AW]\d+", text)}
    openalex_ids = {match for match in re.findall(r"\bhttps?://openalex\.org/([AW]\d+)\b", text)}
    reactome_ids = {match for match in re.findall(r"\bR-HSA-\d+\b", text)}
    string_ids = {match for match in re.findall(r"\b9606\.[A-Za-z0-9_.-]+\b", text)}
    chembl_ids = {match.upper() for match in re.findall(r"\b(CHEMBL\d{3,})\b", text, flags=re.IGNORECASE)}
    mondo_ids = {match for match in re.findall(r"\bMONDO[:_]\d+\b", text, flags=re.IGNORECASE)}
    efo_ids = {match for match in re.findall(r"\bEFO[:_]\d+\b", text, flags=re.IGNORECASE)}
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
        + [f"MONDO:{mid.upper().replace(':', '_')}" for mid in sorted(mondo_ids)]
        + [f"EFO:{eid.upper().replace(':', '_')}" for eid in sorted(efo_ids)]
        + [f"GWAS:{rsid}" for rsid in sorted(rs_ids)]
    )
    return refs


def render_plan(task: WorkflowTask) -> str:
    lines = [f"Task ID: {task.task_id}", "Plan:"]
    for idx, step in enumerate(task.steps, start=1):
        lines.append(f"{idx}. {step.title}")
    return "\n".join(lines)


def render_status(task: WorkflowTask) -> str:
    lines = [
        f"Task {task.task_id}",
        f"Status: {task.status}",
        f"Awaiting HITL: {'yes' if task.awaiting_hitl else 'no'}",
        f"Current step index: {task.current_step_index}",
        "",
        "Step Status:",
    ]
    for idx, step in enumerate(task.steps, start=1):
        lines.append(f"- Step {idx}: {step.title} -> {step.status}")
    return "\n".join(lines)


def _summarize_step_output(output: str, max_chars: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", output or "").strip()
    if not normalized:
        return "No output captured."
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _compact_json(value, max_chars: int = 180) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _extract_primary_query(objective: str) -> str:
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


# Planning/revision compatibility exports now delegate to co_scientist.planning.workflow_planning.
VALID_REQUEST_TYPES = _planning.VALID_REQUEST_TYPES
VALID_INTENT_TAGS = _planning.VALID_INTENT_TAGS
RECOGNIZED_TOOL_NAMES = _planning.RECOGNIZED_TOOL_NAMES
TOOL_CAPABILITIES = _planning.TOOL_CAPABILITIES
CAPABILITY_PATTERNS = _planning.CAPABILITY_PATTERNS

classify_request_type = _planning.classify_request_type
sanitize_request_type = _planning.sanitize_request_type
sanitize_intent_tags = _planning.sanitize_intent_tags
infer_intent_tags = _planning.infer_intent_tags
tool_bundle_for_intent = _planning.tool_bundle_for_intent
build_success_criteria = _planning.build_success_criteria
build_plan_steps = _planning.build_plan_steps
create_task = _planning.create_task

_extract_revision_directive = _planning._extract_revision_directive
_extract_revision_directives = _planning._extract_revision_directives
_extract_revision_tool_hints = _planning._extract_revision_tool_hints
_apply_revision_plan_overrides = _planning._apply_revision_plan_overrides

clone_step = _planning.clone_step
build_plan_delta = _planning.build_plan_delta
active_plan_version = _planning.active_plan_version
register_plan_version = _planning.register_plan_version
initialize_plan_version = _planning.initialize_plan_version
replan_remaining_steps = _planning.replan_remaining_steps


def _extract_explicit_timeframe(text: str) -> str | None:
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
    relative_match = re.search(
        r"\b(?:last|past)\s+(\d{1,2})(?:\s*(?:-|to)\s*(\d{1,2}))?\s+years?\b",
        text,
        flags=re.IGNORECASE,
    )
    if relative_match:
        first = relative_match.group(1)
        second = relative_match.group(2)
        return f"last {first}-{second} years" if second else f"last {first} years"
    current_year_match = re.search(r"\bcurrent year is\s+((?:19|20)\d{2})\b", text, flags=re.IGNORECASE)
    if current_year_match:
        end_year = int(current_year_match.group(1))
        return f"up to {end_year}"
    return None


def _infer_timeframe(task: WorkflowTask) -> str:
    sources = [task.objective]
    if task.steps:
        sources.append(task.steps[0].output)
    for source in sources:
        candidate = _extract_explicit_timeframe(source or "")
        if candidate:
            return candidate

    rank_calls = [
        entry
        for step in task.steps
        for entry in step.tool_trace
        if str(entry.get("tool_name", "")) == "rank_researchers_by_activity"
    ]
    for call in rank_calls:
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        from_year = args.get("fromYear")
        if isinstance(from_year, (int, float)) and from_year > 0:
            return f"{int(from_year)}-{datetime.now(timezone.utc).year} (from ranking call)"
        if isinstance(from_year, str) and from_year.isdigit():
            return f"{int(from_year)}-{datetime.now(timezone.utc).year} (from ranking call)"

    return "Not explicitly specified."


def _detect_source_families(task: WorkflowTask) -> list[str]:
    families: set[str] = set()
    all_entries = [entry for step in task.steps for entry in step.tool_trace]
    all_entries.extend(task.fallback_tool_trace or [])
    for entry in all_entries:
            tool = str(entry.get("tool_name", ""))
            if "openalex" in tool or tool == "rank_researchers_by_activity":
                families.add("OpenAlex")
            elif "pubmed" in tool:
                families.add("PubMed")
            elif "chembl" in tool:
                families.add("ChEMBL")
            elif "gwas" in tool:
                families.add("GWAS Catalog")
            elif "target" in tool or "druggability" in tool or "safety" in tool or "expression" in tool:
                families.add("Open Targets")
            elif "clinical" in tool:
                families.add("ClinicalTrials.gov")
            elif "local" in tool:
                families.add("Local datasets")
    return sorted(families)


def _render_scope_snapshot(task: WorkflowTask) -> list[str]:
    query = _extract_primary_query(task.objective) or task.objective
    revision = _extract_revision_directive(task.objective)
    timeframe = _infer_timeframe(task)
    source_families = _detect_source_families(task)
    lines = ["## Query", f"> \"{query}\"", "", "## Scope"]
    if revision:
        lines.append(f"- Revision applied: {revision}")
    lines.append(f"- Timeframe considered: {timeframe}")
    if "researcher_discovery" in task.intent_tags:
        lines.append(
            "- Definition of \"top\" researchers: topic-specific activity score from publication volume, citations, recency, and authorship leadership."
        )
    if source_families:
        lines.append(f"- Primary evidence sources: {', '.join(source_families)}")
    return lines


def _extract_decomposition_from_text(text: str) -> list[str]:
    if not text:
        return []
    cleaned = re.sub(r"\r\n?", "\n", text)
    lines = [line.rstrip() for line in cleaned.splitlines()]
    capture = False
    items: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if capture and items:
                break
            continue

        normalized = stripped.lower().replace(" ", "_")
        if any(token in normalized for token in ["decomposition", "subtask", "sub_task", "work_breakdown"]):
            capture = True
            continue

        bullet_match = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", stripped)
        if not bullet_match:
            if capture and items:
                break
            continue
        entry = bullet_match.group(1).strip()
        entry = re.sub(r"^\*{1,2}|\*{1,2}$", "", entry).strip()
        entry = entry.replace("**", "").replace("`", "")
        entry = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", entry)
        entry = re.sub(r"\s*:\*+\s*$", ":", entry)
        entry = re.sub(r"\s+", " ", entry).strip()
        if entry:
            items.append(entry)

    if items:
        return items

    # Fallback: parse numbered lines from the full text when no explicit decomposition block is labeled.
    loose_items = []
    for line in lines:
        stripped = line.strip()
        bullet_match = re.match(r"^(?:\d+[.)])\s+(.+)$", stripped)
        if not bullet_match:
            continue
        entry = bullet_match.group(1).replace("**", "").replace("`", "")
        entry = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", entry)
        entry = re.sub(r"\s*:\*+\s*$", ":", entry)
        entry = re.sub(r"\s+", " ", entry.strip())
        if entry:
            loose_items.append(entry)
    return loose_items


def _default_decomposition(task: WorkflowTask) -> list[str]:
    if "researcher_discovery" in task.intent_tags:
        return [
            "Query disease/topic context and lock timeframe constraints.",
            "Identify topic-matched publications from OpenAlex/PubMed.",
            "Find candidate authors and affiliation signals from publication data.",
            "Assess author activity/prominence and produce a ranked shortlist.",
        ]
    target_tags = {
        "target_comparison",
        "clinical_landscape",
        "chemistry_evidence",
        "safety_assessment",
        "genetics_direction",
        "expression_context",
        "competitive_landscape",
        "variant_check",
        "pathway_context",
    }
    if any(tag in target_tags for tag in task.intent_tags):
        return [
            "Normalize target/disease entities and decision criteria.",
            "Collect cross-source evidence (genetics, druggability, clinical, literature).",
            "Resolve contradictions and produce recommendation with limitations.",
        ]
    return [
        "Extract concrete scope, entities, and success criteria.",
        "Gather evidence from relevant tools and document fallback pivots.",
        "Synthesize findings into a direct answer with caveats and next actions.",
    ]


def _decomposition_status(task: WorkflowTask, subtask: str) -> str:
    tool_names = {
        str(entry.get("tool_name", ""))
        for step in task.steps
        for entry in step.tool_trace
    }
    tool_names.update(str(entry.get("tool_name", "")) for entry in task.fallback_tool_trace)
    combined_output = "\n".join(step.output for step in task.steps if step.output).lower()
    lower = subtask.lower()

    if any(token in lower for token in ["disease", "topic", "scope", "timeframe"]):
        observed = bool(task.steps and task.steps[0].output.strip()) or bool(
            tool_names.intersection({"search_diseases", "search_targets", "expand_disease_context"})
        )
    elif any(token in lower for token in ["publication", "literature"]):
        observed = bool(
            tool_names.intersection(
                {
                    "search_openalex_works",
                    "search_pubmed",
                    "search_pubmed_advanced",
                    "get_pubmed_abstract",
                    "get_pubmed_paper_details",
                }
            )
        )
    elif any(token in lower for token in ["author", "researcher", "investigator", "affiliation"]):
        observed = bool(
            tool_names.intersection(
                {"search_openalex_authors", "get_pubmed_author_profile", "rank_researchers_by_activity"}
            )
        )
    elif any(token in lower for token in ["activity", "rank", "prominence", "shortlist"]):
        observed = "rank_researchers_by_activity" in tool_names or "rank" in combined_output
    else:
        observed = bool(combined_output.strip())

    if observed and task.status == "completed":
        return "completed"
    if observed:
        return "in progress"
    return "not observed"


def _render_decomposition(task: WorkflowTask) -> list[str]:
    step_one_output = task.steps[0].output if task.steps else ""
    subtasks = _extract_decomposition_from_text(step_one_output)
    if len(subtasks) < 2:
        subtasks = _default_decomposition(task)
    lines = ["## Decomposition"]
    for idx, subtask in enumerate(subtasks, start=1):
        status = _decomposition_status(task, subtask)
        lines.append(f"{idx}. {subtask} ({status})")
    return lines


def _detect_pivots(trace_entries: list[dict]) -> list[str]:
    pivots: list[str] = []
    for idx in range(len(trace_entries) - 1):
        current = trace_entries[idx]
        nxt = trace_entries[idx + 1]
        current_outcome = str(current.get("outcome", "unknown"))
        if current_outcome in {"error", "not_found_or_empty", "no_response", "degraded"}:
            current_tool = str(current.get("tool_name", "unknown_tool"))
            next_tool = str(nxt.get("tool_name", "unknown_tool"))
            if current_tool != next_tool:
                detail = str(current.get("detail", "")).strip()
                if detail:
                    pivots.append(
                        f"After {current_tool} returned {current_outcome} ({detail}), switched to {next_tool}."
                    )
                else:
                    pivots.append(
                        f"After {current_tool} returned {current_outcome}, switched to {next_tool}."
                    )
    return pivots


def _render_tool_trace(trace_entries: list[dict]) -> list[str]:
    if not trace_entries:
        return ["- Tool activity summary: no tool calls were recorded for this step."]

    outcome_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    notable_errors: list[str] = []
    for entry in trace_entries:
        outcome = str(entry.get("outcome", "unknown"))
        tool_name = str(entry.get("tool_name", "unknown_tool"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        if outcome in {"error", "not_found_or_empty", "no_response", "degraded"} and len(notable_errors) < 3:
            detail = _summarize_step_output(str(entry.get("detail", "")), max_chars=100)
            detail_text = f" ({detail})" if detail and detail != "No output captured." else ""
            notable_errors.append(f"{tool_name}{detail_text}")

    ok_count = outcome_counts.get("ok", 0)
    issue_count = (
        outcome_counts.get("error", 0)
        + outcome_counts.get("not_found_or_empty", 0)
        + outcome_counts.get("no_response", 0)
        + outcome_counts.get("degraded", 0)
    )
    lines = [
        (
            "- Tool activity summary: "
            f"{len(trace_entries)} calls across {len(tool_counts)} tools "
            f"({ok_count} successful, {issue_count} with issues)."
        )
    ]
    if notable_errors:
        lines.append(f"- Notable issues: {'; '.join(notable_errors)}")

    pivot_notes = _detect_pivots(trace_entries)
    lines.append(
        f"- Pivot behavior: {len(pivot_notes)} adaptation(s) after blocked/failed tool responses."
        if pivot_notes
        else "- Pivot behavior: none."
    )
    return lines


def _render_methodology(task: WorkflowTask) -> list[str]:
    lines = ["## Methodology"]
    if not task.steps:
        lines.append("- No workflow steps were created.")
        return lines

    reasoning_points = [step.reasoning_summary.strip() for step in task.steps if step.reasoning_summary.strip()]
    if not reasoning_points:
        reasoning_points = [_summarize_step_output(task.steps[0].instruction, max_chars=180)]
    lines.extend(["### Reasoning"])
    for point in reasoning_points[:3]:
        lines.append(f"- {point}")
    if len(reasoning_points) > 3:
        lines.append(f"- {len(reasoning_points) - 3} additional reasoning notes omitted for brevity.")

    all_trace_entries = [entry for step in task.steps for entry in step.tool_trace]
    all_tool_names = sorted({str(entry.get("tool_name", "unknown_tool")) for entry in all_trace_entries})
    lines.extend(["", "### Actions", f"- Tools involved: {', '.join(all_tool_names) if all_tool_names else 'none'}"])
    lines.extend(_render_tool_trace(all_trace_entries))
    if task.fallback_tool_trace:
        fallback_tools = sorted({str(entry.get("tool_name", "unknown_tool")) for entry in task.fallback_tool_trace})
        lines.append(f"- Fallback tools involved: {', '.join(fallback_tools) if fallback_tools else 'none'}")
    lines.append("")

    observation_points: list[str] = []
    for step in task.steps:
        if step.observations:
            observation_points.append(step.observations[0])
    lines.extend(["### Observations"])
    if observation_points:
        for point in observation_points[:4]:
            lines.append(f"- {point}")
        if len(observation_points) > 4:
            lines.append(f"- {len(observation_points) - 4} additional observations omitted for brevity.")
    else:
        lines.append("- No observations captured.")
    return lines


def _humanize_fallback_label(label: str) -> str:
    mapping = {
        "selected_tools": "Selected Tools",
        "why_chosen": "Rationale",
        "key_results": "Key Results",
        "remaining_gaps": "Remaining Gaps",
    }
    normalized = re.sub(r"\s+", "_", (label or "").strip().lower())
    return mapping.get(normalized, normalized.replace("_", " ").title())


def _split_answer_and_fallback(answer_text: str) -> tuple[str, str | None]:
    if not answer_text:
        return "No answer generated.", None
    match = re.search(r"\nFallback recovery notes:\s*", answer_text, flags=re.IGNORECASE)
    if not match:
        return answer_text.strip(), None
    main = answer_text[: match.start()].strip()
    fallback = answer_text[match.end() :].strip()
    return main or "No answer generated.", (fallback or None)


def _format_fallback_notes(fallback_text: str) -> list[str]:
    if not fallback_text:
        return ["- No fallback notes were captured."]

    lines: list[str] = []
    seen_sections: set[str] = set()
    for raw in fallback_text.splitlines():
        stripped = raw.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        key_match = re.match(r"^\*{0,2}\s*([a-zA-Z_ ]+)\s*:\s*\*{0,2}$", stripped)
        if key_match:
            label = key_match.group(1).strip().lower().replace(" ", "_")
            if label in {"selected_tools", "why_chosen", "key_results", "remaining_gaps"}:
                if label in seen_sections:
                    continue
                seen_sections.add(label)
                lines.append(f"### {_humanize_fallback_label(label)}")
                continue
        lines.append(raw.rstrip())

    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return lines or ["- No fallback notes were captured."]


def _build_evidence_context(task: WorkflowTask, refs: list[str], max_items: int = 8) -> list[str]:
    if not refs:
        return ["- No explicit citation IDs were detected in the workflow output."]

    context_by_ref: dict[str, str] = {}
    for step_idx, step in enumerate(task.steps, start=1):
        output_lines = [line.strip() for line in (step.output or "").splitlines() if line.strip()]
        for line in output_lines:
            compact = _summarize_step_output(line, max_chars=170)
            for ref in refs:
                if ref in context_by_ref:
                    continue
                if ref in line:
                    context_by_ref[ref] = f"Step {step_idx}: {compact}"

    lines: list[str] = []
    for ref in refs[:max_items]:
        detail = context_by_ref.get(ref, "Referenced in synthesis; see Answer section for interpretation.")
        lines.append(f"- {ref}: {detail}")
    remaining = len(refs) - max_items
    if remaining > 0:
        lines.append(f"- ... {remaining} additional references omitted for brevity.")
    return lines


def _normalize_paragraph_key(text: str) -> str:
    normalized = re.sub(r"[`*_#>-]+", " ", text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    return normalized


def _dedupe_answer_paragraphs(text: str, *, max_paragraphs: int = 10) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    if not paragraphs:
        return ""
    seen: set[str] = set()
    kept: list[str] = []
    for paragraph in paragraphs:
        key = _normalize_paragraph_key(paragraph)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(paragraph)
        if len(kept) >= max_paragraphs:
            break
    return "\n\n".join(kept)


def _strip_legacy_report_sections(text: str) -> str:
    if not text:
        return ""
    raw_lines = str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    kept: list[str] = []
    skip = False
    heading_re = re.compile(r"^\s*(?:#+\s*)?([A-Za-z][A-Za-z0-9 _-]{1,80})\s*:?\s*$")
    blocked = {
        "query",
        "scope",
        "decomposition",
        "diagnostics",
        "methodology",
        "reasoning",
        "actions",
        "observations",
        "fallback recovery notes",
        "tool activity",
    }

    for raw in raw_lines:
        stripped = raw.strip()
        match = heading_re.match(stripped)
        if match:
            heading = re.sub(r"\s+", " ", match.group(1).strip().lower())
            if heading in blocked:
                skip = True
                continue
            if skip:
                skip = False
        if skip:
            continue
        kept.append(raw)
    return "\n".join(kept).strip()


def _sanitize_markdown_artifacts(text: str) -> str:
    if not text:
        return ""
    value = str(text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r":\*{1,2}\s*$", ":", value, flags=re.MULTILINE)
    value = value.replace("**", "")
    value = value.replace("`", "")
    value = re.sub(r"\s+\n", "\n", value)
    return value.strip()


def _extract_next_actions(answer_text: str) -> tuple[str, list[str]]:
    lines = (answer_text or "").splitlines()
    if not lines:
        return "", []

    heading_pattern = re.compile(
        r"^\s*(?:#+\s*)?\*{0,2}\s*next\s*(?:actions?|steps?)\s*:?\s*\*{0,2}\s*$",
        flags=re.IGNORECASE,
    )
    start_idx = next((idx for idx, line in enumerate(lines) if heading_pattern.match(line.strip())), None)
    if start_idx is None:
        return (answer_text or "").strip(), []

    body_lines = lines[:start_idx]
    action_lines = lines[start_idx + 1 :]
    actions: list[str] = []
    for line in action_lines:
        stripped = line.strip()
        if not stripped or stripped == "---":
            continue
        bullet = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", stripped)
        candidate = bullet.group(1).strip() if bullet else stripped
        candidate = candidate.replace("**", "").replace("`", "")
        candidate = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip(" .")
        if not candidate:
            continue
        if candidate not in actions:
            actions.append(candidate)
        if len(actions) >= 6:
            break
    body = "\n".join(body_lines).strip()
    return body, actions


def _extract_answer_line(answer_text: str, label: str) -> str | None:
    pattern = re.compile(
        rf"^\s*(?:[-*]\s+)?\*{{0,2}}{re.escape(label)}\*{{0,2}}\s*:\s*(.+)$",
        flags=re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(answer_text or "")
    if not match:
        return None
    value = match.group(1).strip()
    value = value.replace("**", "").replace("`", "")
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value or None


def _extract_first_narrative_paragraph(text: str) -> str | None:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text or "") if part.strip()]
    for paragraph in paragraphs:
        if re.match(r"^(?:#+\s*)", paragraph):
            continue
        if re.match(r"^(?:[-*]|\d+[.)])\s+", paragraph):
            continue
        if ":" in paragraph and len(paragraph) <= 80:
            # Likely a short section label line.
            continue
        compact = re.sub(r"\s+", " ", paragraph).strip()
        if compact:
            return compact
    return None


def _compact_rationale_body(answer_body: str, recommendation: str | None, confidence: str | None) -> str:
    if not answer_body:
        return ""
    body = answer_body
    for label in ("Recommendation", "Confidence Level", "Confidence"):
        pattern = re.compile(
            rf"^\s*(?:[-*]\s+)?{re.escape(label)}\s*:\s*.+$",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        body = pattern.sub("", body)

    if recommendation:
        escaped = re.escape(recommendation.strip())
        body = re.sub(rf"^\s*{escaped}\s*$", "", body, flags=re.MULTILINE)
    if confidence:
        escaped_conf = re.escape(confidence.strip())
        body = re.sub(rf"^\s*{escaped_conf}\s*$", "", body, flags=re.MULTILINE)

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    filtered: list[str] = []
    blocked_prefixes = (
        "query",
        "scope",
        "decomposition",
        "diagnostics",
        "why this recommendation",
        "rationale narrative",
        "summary of evidence",
        "decision report",
        "methodology",
        "reasoning",
        "actions",
        "observations",
        "next actions",
    )
    for paragraph in paragraphs:
        normalized = re.sub(r"^[#\s]+", "", paragraph).strip().lower()
        if any(normalized.startswith(prefix) for prefix in blocked_prefixes):
            continue
        cleaned = paragraph.strip()
        cleaned = re.sub(
            r"^\s*(?:rationale narrative|why this recommendation|rationale)\s*:\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        if not cleaned:
            continue
        filtered.append(cleaned)
        if len(filtered) >= 6:
            break

    compact = "\n\n".join(filtered).strip()
    compact = re.sub(r"\n{3,}", "\n\n", compact)
    return compact


def _render_methodology_brief(task: WorkflowTask) -> list[str]:
    lines = ["### Methodology"]
    step_titles = [f"{idx + 1}. {step.title}" for idx, step in enumerate(task.steps) if step.status != "pending"]
    if step_titles:
        lines.append(f"This run executed the following workflow steps: {' | '.join(step_titles)}.")
    else:
        lines.append("This run did not execute any workflow steps.")

    trace_entries = [entry for step in task.steps for entry in step.tool_trace]
    trace_entries.extend(task.fallback_tool_trace or [])
    tool_names = sorted({str(entry.get("tool_name", "unknown_tool")) for entry in trace_entries})
    lines.append(f"Tools used in this run: {', '.join(tool_names) if tool_names else 'none'}.")

    if not trace_entries:
        lines.append("No tool-call trace was recorded, so methodology details are limited.")
        return lines

    outcome_counts: dict[str, int] = {}
    notable_errors: list[str] = []
    for entry in trace_entries:
        outcome = str(entry.get("outcome", "unknown"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        if outcome in {"error", "not_found_or_empty", "no_response", "degraded"} and len(notable_errors) < 3:
            tool_name = str(entry.get("tool_name", "unknown_tool"))
            detail = _summarize_step_output(str(entry.get("detail", "")), max_chars=100)
            detail_text = f" ({detail})" if detail and detail != "No output captured." else ""
            notable_errors.append(f"{tool_name}{detail_text}")

    ok_count = outcome_counts.get("ok", 0)
    issue_count = (
        outcome_counts.get("error", 0)
        + outcome_counts.get("not_found_or_empty", 0)
        + outcome_counts.get("no_response", 0)
        + outcome_counts.get("degraded", 0)
    )
    lines.append(
        "Tool activity summary: "
        f"{len(trace_entries)} calls were captured, with {ok_count} successful and {issue_count} returning issues."
    )

    if notable_errors:
        lines.append(f"Notable tool issues included: {'; '.join(notable_errors)}.")

    pivot_notes = _detect_pivots(trace_entries)
    if pivot_notes:
        lines.append(
            f"The workflow adapted {len(pivot_notes)} time(s) after blocked or failed responses during evidence collection."
        )
        if len(pivot_notes) <= 2:
            for note in pivot_notes:
                lines.append(f"- {note}")
        else:
            for note in pivot_notes[:2]:
                lines.append(f"- {note}")
            lines.append(f"- {len(pivot_notes) - 2} additional adaptation(s) omitted for brevity.")
    else:
        lines.append("No tool-switch pivots were required during this run.")
    return lines


def render_final_report(task: WorkflowTask, quality_report: dict | None = None) -> str:
    raw_answer_text = task.steps[-1].output if task.steps else "No answer generated."
    answer_text = raw_answer_text.strip() if raw_answer_text else "No answer generated."
    fallback_text = task.fallback_recovery_notes.strip() or None
    if not fallback_text:
        # Backward compatibility for legacy tasks where fallback notes were appended to step output.
        answer_text, fallback_text = _split_answer_and_fallback(raw_answer_text)
    answer_text = _strip_legacy_report_sections(answer_text)
    answer_text = _sanitize_markdown_artifacts(answer_text)
    answer_text = _dedupe_answer_paragraphs(answer_text)
    answer_body, extracted_actions = _extract_next_actions(answer_text)
    recommendation = _extract_answer_line(answer_body, "Recommendation")
    confidence = _extract_answer_line(answer_body, "Confidence Level") or _extract_answer_line(answer_body, "Confidence")
    narrative_open = _extract_first_narrative_paragraph(answer_body) or "The available evidence supports the recommendation below."
    refs = sorted({ref for step in task.steps for ref in step.evidence_refs})
    source_families = _detect_source_families(task)

    answer_summary = recommendation or narrative_open
    if confidence:
        answer_summary = f"{answer_summary} Confidence: {confidence}."

    lines = ["## Answer", answer_summary]
    if source_families:
        lines.append(f"Sources used in this run: {', '.join(source_families)}.")

    lines.extend(["", "## Rationale"])
    rationale_body = _compact_rationale_body(answer_body, recommendation, confidence)
    if rationale_body:
        lines.append(rationale_body)
    else:
        lines.append("The recommendation is based on the strongest available evidence captured in this run.")

    lines.extend(["", "### Evidence Base"])
    if refs:
        preview = ", ".join(refs[:10])
        suffix = f", +{len(refs) - 10} more" if len(refs) > 10 else ""
        lines.append(f"Evidence IDs captured in this run include: {preview}{suffix}.")
    else:
        lines.append("Evidence IDs captured in this run: none.")
    if quality_report:
        lines.append(
            "Quality snapshot: "
            f"{quality_report.get('tool_call_count', 0)} tool calls, "
            f"{quality_report.get('evidence_count', 0)} evidence IDs."
        )

    lines.extend([""])
    lines.extend(_render_methodology_brief(task))

    lines.extend(["", "### Limitations"])
    gaps = quality_report.get("unresolved_gaps", []) if quality_report else []
    if gaps:
        lines.extend([f"- {gap}" for gap in gaps[:8]])
    elif fallback_text:
        lines.append("- Fallback recovery was used; validate high-stakes claims before external use.")
    else:
        lines.append("- No critical unresolved gaps were detected by current quality checks.")

    if fallback_text and not gaps:
        lines.extend(["", "### Fallback Notes"])
        lines.extend(_format_fallback_notes(fallback_text))

    lines.extend(["", "## Next Actions"])
    next_actions = extracted_actions[:5]
    if not next_actions and gaps:
        next_actions = [f"Resolve: {gap}" for gap in gaps[:3]]
    if not next_actions:
        next_actions = [
            "Validate the recommendation against one independent source.",
            "Run a focused follow-up for the highest-uncertainty evidence gap.",
            "Convert findings into an execution plan with owners and milestones.",
        ]
    lines.extend([f"- {item}" for item in next_actions])
    return "\n".join(lines)


def step_prompt(task: WorkflowTask, step: WorkflowStep) -> str:
    revision = _extract_revision_directive(task.objective)
    revision_directives = _extract_revision_directives(task.objective)
    revision_directive_block = ""
    if revision_directives:
        rendered_directives = "\n".join(f"- {item}" for item in revision_directives[:8])
        revision_directive_block = (
            "\nRevision directives (authoritative):\n"
            f"{rendered_directives}\n"
            "- Apply these directives even when they change tool choice, sequence, or output shape.\n"
            "- If any directive cannot be fully satisfied, explain the blocker and fallback explicitly.\n"
        )
    researcher_ranking_guardrail = ""
    if step.step_id == "step_2" and "researcher_discovery" in task.intent_tags:
        researcher_ranking_guardrail = (
            "\nResearcher ranking requirements:\n"
            "- Treat requests using terms like top/most active/prominent as quantitative ranking tasks.\n"
            "- Prioritize `rank_researchers_by_activity` with topic query and an explicit fromYear.\n"
            "- Cross-check shortlisted names with `search_openalex_works` before final ranking.\n"
            "- Do not build the final ranking from only a few first-authors of review/case papers.\n"
            "- If ranking tools time out or return errors, pivot to publication-based tools (`search_openalex_works`, `search_pubmed_advanced`, `get_pubmed_author_profile`) in the same step.\n"
            "- Do not switch to a clinical-trials-only fallback for top researcher ranking unless the user explicitly asks for trial investigators.\n"
            "- If ranking tools fail, report blocked status with exact error evidence instead of presenting a confident top list.\n"
        )
    step_freshness_guardrail = (
        "\nExecution integrity requirements:\n"
        "- Treat this as a fresh execution of the current step.\n"
        "- Do not state that this step was already completed in a previous turn.\n"
    )
    if step.step_id == "step_1":
        step_freshness_guardrail += (
            "- Output must include `decomposition_subtasks` as a numbered list (3-5 concrete sub-tasks).\n"
            "- Each sub-task must be executable and reflect intended tool/data operations.\n"
        )
        if "researcher_discovery" in task.intent_tags:
            step_freshness_guardrail += (
                "- For researcher discovery, include these operations in order: "
                "query disease/topic context -> identify publications -> find authors -> assess activity.\n"
            )
        if revision:
            step_freshness_guardrail += (
                f"- Explicitly acknowledge this revision and apply it: {revision}\n"
                "- Restate the updated timeframe in explicit year terms when relevant.\n"
            )
        if revision_directives:
            step_freshness_guardrail += (
                "- Include a brief `revision_alignment` mapping from each directive to concrete plan updates.\n"
            )
    elif revision_directives:
        step_freshness_guardrail += (
            "- Carry forward revision directives as execution constraints for this step.\n"
        )
    if task.steps and step.step_id == task.steps[-1].step_id:
        step_freshness_guardrail += (
            "- Final output format: Recommendation first, then Rationale narrative, then Methodology, then Next Actions.\n"
            "- Keep the narrative concise and avoid repeating full evidence lists already stated in prior steps.\n"
            "- Limit Next Actions to 3 concrete items.\n"
        )
    return (
        "You are executing a co-investigator workflow.\n"
        f"Task objective: {task.objective}\n"
        f"Current step: {step.title}\n"
        f"Step instruction: {step.instruction}\n\n"
        "Preferred tools: "
        f"{', '.join(step.recommended_tools) if step.recommended_tools else 'N/A'}\n"
        "Fallback tools: "
        f"{', '.join(step.fallback_tools) if step.fallback_tools else 'N/A'}\n"
        "Allowed tools for this step: "
        f"{', '.join(step.allowed_tools) if step.allowed_tools else 'none'}\n\n"
        "Return concise output for this step only.\n"
        "Do not restate workflow contracts, constraints, or meta-guidelines.\n"
        "Prioritize directly answering the user query.\n"
        "Do not call tools outside the allowed-tools list for this step.\n"
        "When preferred tools are available, execute at least one relevant tool before finalizing the step.\n"
        "If tools are insufficient, state a fallback strategy instead of blocking.\n"
        "Do not introduce factual claims from background/domain memory when they are not supported by captured tool output in this step.\n"
        "If high-priority evidence is missing due to tool failures, mark conclusions as provisional and carry the gap forward explicitly.\n"
        "Use source citations whenever possible (PMID, NCT IDs).\n"
        f"{revision_directive_block}"
        f"{step_freshness_guardrail}"
        f"{researcher_ranking_guardrail}"
    )
