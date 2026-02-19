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
CAPABILITY_PATTERNS = _planning.CAPABILITY_PATTERNS

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


def _source_for_tool_name(tool_name: str) -> str | None:
    tool = str(tool_name or "").strip().lower()
    if not tool:
        return None
    if "openalex" in tool or tool == "rank_researchers_by_activity":
        return "OpenAlex"
    if "pubmed" in tool:
        return "PubMed"
    if "chembl" in tool:
        return "ChEMBL"
    if "gwas" in tool:
        return "GWAS Catalog"
    if "clinvar" in tool:
        return "ClinVar"
    if "reactome" in tool:
        return "Reactome"
    if "string" in tool:
        return "STRING"
    if "target" in tool or "druggability" in tool or "safety" in tool or "expression" in tool:
        return "Open Targets"
    if "clinical" in tool:
        return "ClinicalTrials.gov"
    if "local" in tool:
        return "Local datasets"
    return None


_TOOL_HUMAN_ACTIONS: dict[str, str] = {
    "search_openalex_works": "queried OpenAlex for topic-matched publications",
    "search_openalex_authors": "queried OpenAlex for author and affiliation profiles",
    "rank_researchers_by_activity": "used OpenAlex activity signals to rank researchers by topic relevance",
    "get_researcher_contact_candidates": "looked up public researcher contact and profile signals from OpenAlex-linked metadata",
    "search_pubmed": "searched PubMed for topic-relevant literature",
    "search_pubmed_advanced": "ran targeted PubMed searches with tighter topic/time constraints",
    "get_pubmed_abstract": "reviewed PubMed abstracts for relevance and findings",
    "get_pubmed_paper_details": "extracted publication details from PubMed records",
    "get_pubmed_author_profile": "reviewed PubMed author profiles and publication history",
    "search_diseases": "resolved disease identity and ontology context via Open Targets",
    "expand_disease_context": "expanded disease terminology and context via Open Targets",
    "search_targets": "resolved target identity via Open Targets",
    "search_disease_targets": "linked disease-to-target evidence via Open Targets",
    "get_target_info": "retrieved target biology details via Open Targets",
    "check_druggability": "reviewed target tractability evidence via Open Targets",
    "get_target_drugs": "reviewed known drug programs linked to the target via Open Targets",
    "summarize_target_expression_context": "reviewed target expression context via Open Targets",
    "summarize_target_competitive_landscape": "reviewed target competition and development landscape via Open Targets",
    "summarize_target_safety_liabilities": "reviewed target safety-liability evidence via Open Targets",
    "compare_targets_multi_axis": "compared targets across weighted evidence dimensions via Open Targets",
    "search_clinical_trials": "queried ClinicalTrials.gov for relevant studies",
    "get_clinical_trial": "reviewed detailed records from ClinicalTrials.gov",
    "summarize_clinical_trials_landscape": "summarized trial status and outcome patterns from ClinicalTrials.gov",
    "search_chembl_compounds_for_target": "reviewed compound-level data for the target in ChEMBL",
    "search_gwas_associations": "reviewed GWAS association signals",
    "infer_genetic_effect_direction": "inferred likely direction-of-effect from human genetics evidence",
    "search_clinvar_variants": "reviewed ClinVar variant evidence",
    "get_clinvar_variant_details": "reviewed detailed ClinVar variant annotations",
    "search_reactome_pathways": "reviewed pathway context in Reactome",
    "get_string_interactions": "reviewed protein interaction context in STRING",
    "get_gene_info": "reviewed gene-level context from genomics resources",
}


def _humanize_internal_tool_mentions(text: str) -> str:
    value = str(text or "")
    if not value:
        return value
    replacements: list[tuple[str, str]] = []
    for tool_name in sorted(_TOOL_HUMAN_ACTIONS.keys(), key=len, reverse=True):
        source = _source_for_tool_name(tool_name)
        if not source:
            continue
        replacements.append((tool_name, source))
    repaired = value
    for tool_name, label in replacements:
        repaired = re.sub(
            rf"`?{re.escape(tool_name)}`?",
            label,
            repaired,
            flags=re.IGNORECASE,
        )
    return repaired


def _detect_source_families(task: WorkflowTask) -> list[str]:
    families: set[str] = set()
    all_entries = [entry for step in task.steps for entry in step.tool_trace]
    for entry in all_entries:
        source = _source_for_tool_name(str(entry.get("tool_name", "")))
        if source:
            families.add(source)
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
    subtasks: list[str] = [
        "Define scope, entities, constraints, and success criteria.",
    ]
    evidence_steps = [step.title for step in task.steps[1:-1] if step.recommended_tools or step.fallback_tools]
    for title in evidence_steps[:3]:
        subtasks.append(
            f"Execute the '{title}' stage and capture traceable citations plus unresolved gaps."
        )
    subtasks.append("Synthesize evidence into a recommendation with limitations and next actions.")
    return subtasks


def _decomposition_status(task: WorkflowTask, subtask: str) -> str:
    tool_names = {
        str(entry.get("tool_name", ""))
        for step in task.steps
        for entry in step.tool_trace
    }
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
        detail = context_by_ref.get(ref, "Referenced in synthesis; see final recommendation and rationale for interpretation.")
        lines.append(f"- {ref}: {detail}")
    remaining = len(refs) - max_items
    if remaining > 0:
        lines.append(f"- ... {remaining} additional references omitted for brevity.")
    return lines


def _requested_paper_count(objective: str) -> int | None:
    text = str(objective or "")
    match = re.search(r"\b(\d{1,2})\s+(?:recent\s+)?papers?\b", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        count = int(match.group(1))
    except ValueError:
        return None
    return count if count > 0 else None


def _looks_like_placeholder_summary(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized:
        return True
    if normalized.startswith("(doi:") or normalized.startswith("doi:"):
        return True
    placeholders = (
        "the available evidence supports the recommendation below",
        "the recommendation is based on the strongest available evidence",
        "no answer generated",
    )
    return any(marker in normalized for marker in placeholders)


def _strip_leading_citation_tokens(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return value
    cleaned = re.sub(
        r"^(?:\(?\s*(?:DOI|PMID|OpenAlex)\s*:\s*[^)\s]+(?:\s*\)?|\s+))+",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip(" -:,.")
    return cleaned or value


def _collect_formatted_references(task: WorkflowTask) -> list[str]:
    refs: list[str] = []
    seen: set[str] = set()

    def _append(citation: str, doi: str = "") -> None:
        text = re.sub(r"\s+", " ", str(citation or "")).strip()
        if not text:
            return
        text = re.sub(r"^\s*Citation:\s*", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*\|\s*Cited by:\s*\d+\s*$", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"\s*OpenAlex\s*$", "", text, flags=re.IGNORECASE).strip()
        normalized_doi = str(doi or "").strip().lower()
        if not normalized_doi:
            doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)
            normalized_doi = doi_match.group(0).lower() if doi_match else ""
        if normalized_doi:
            text = re.sub(r"\bDOI:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", "", text, flags=re.IGNORECASE).strip()
            if f"https://doi.org/{normalized_doi}" not in text.lower():
                text += f" [DOI:{normalized_doi}](https://doi.org/{normalized_doi})"
        text = re.sub(r"\s+", " ", text).strip()
        key = normalized_doi or re.sub(r"\s+", " ", text).strip().lower()
        if key in seen:
            return
        seen.add(key)
        refs.append(text)

    for step in task.steps:
        for entry in (step.tool_trace or []):
            for bib in (entry.get("bibliography") or []):
                if not isinstance(bib, dict):
                    continue
                _append(str(bib.get("citation", "")), str(bib.get("doi", "")))

    if refs:
        return refs

    # Fallback: at least expose clickable DOI links when full citations are unavailable.
    doi_refs = [
        ref for ref in extract_evidence_refs("\n".join(step.output for step in task.steps if step.output))
        if str(ref).upper().startswith("DOI:")
    ]
    for doi_ref in doi_refs:
        doi = str(doi_ref).split(":", 1)[1].strip().lower()
        if doi:
            _append(f"DOI only metadata available: {doi}", doi)
    return refs


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
        "structured output",
        "machine readable output",
        "reasoning",
        "actions",
        "observations",
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


def split_report_and_next_actions(markdown: str) -> tuple[str, list[str]]:
    text = str(markdown or "").strip()
    if not text:
        return "", []
    body, actions = _extract_next_actions(text)
    normalized_body = _sanitize_markdown_artifacts(body)
    normalized_body = _humanize_internal_tool_mentions(normalized_body)
    normalized_body = re.sub(r"\n{3,}", "\n\n", normalized_body).strip()
    cleaned_actions: list[str] = []
    for item in actions:
        cleaned = _humanize_internal_tool_mentions(str(item).strip())
        if not cleaned:
            continue
        if cleaned not in cleaned_actions:
            cleaned_actions.append(cleaned)
        if len(cleaned_actions) >= 5:
            break
    return normalized_body, cleaned_actions


def derive_follow_up_suggestions(markdown: str, quality_report: dict | None = None) -> list[str]:
    _, actions = split_report_and_next_actions(markdown)
    if actions:
        return actions[:5]

    gaps = quality_report.get("unresolved_gaps", []) if isinstance(quality_report, dict) else []
    suggestions = [f"Resolve evidence gap: {str(gap).strip()}" for gap in gaps[:3] if str(gap).strip()]
    if not suggestions:
        suggestions = [
            "Run a focused follow-up on the highest-uncertainty claim.",
            "Stress-test the recommendation against one independent evidence source.",
            "Convert findings into an execution plan with owners and milestones.",
        ]
    return suggestions[:5]


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
    body = re.split(
        r"\n\s*(?:#{1,3}\s*)?(?:methodology|limitations|references|next actions?)\s*:?\s*\n",
        body,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    filtered: list[str] = []
    blocked_prefixes = (
        "query",
        "scope",
        "decomposition",
        "diagnostics",
        "summary of evidence",
        "decision report",
        "methodology",
        "reasoning",
        "actions",
        "observations",
        "next actions",
        "references",
        "methodology:",
        "limitations:",
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


def _build_structured_output_contract(
    task: WorkflowTask,
    *,
    recommendation: str | None,
    confidence: str | None,
    answer_summary: str,
    next_actions: list[str],
    source_families: list[str],
    diagnostics_evidence_count: int,
    diagnostics_tool_calls: int,
    contract_ok: int,
    contract_expected: int,
    refs: list[str],
    unverified_refs: list[str],
    gaps: list[str],
    quality_report: dict | None = None,
) -> dict:
    quality_passed = quality_report.get("passed") if quality_report else None
    validated_refs = quality_report.get("validated_evidence_refs", refs) if quality_report else refs
    if not isinstance(validated_refs, list):
        validated_refs = refs
    validated_refs = [str(ref).strip() for ref in validated_refs if str(ref).strip()]
    unresolved = [str(gap).strip() for gap in (gaps or []) if str(gap).strip()]
    def _evidence_ref_sort_key(ref: str) -> tuple[int, str]:
        normalized = str(ref).strip()
        upper = normalized.upper()
        if upper.startswith("PMID:"):
            return (0, normalized)
        if upper.startswith("NCT"):
            return (1, normalized)
        if upper.startswith("DOI:"):
            return (2, normalized)
        if upper.startswith("OPENALEX:") or upper.startswith("HTTP://OPENALEX.") or upper.startswith("HTTPS://OPENALEX."):
            return (3, normalized)
        return (9, normalized)
    prioritized_validated_refs = sorted(set(validated_refs), key=_evidence_ref_sort_key)
    prioritized_unverified_refs = sorted(
        {str(ref).strip() for ref in (unverified_refs or []) if str(ref).strip()},
        key=_evidence_ref_sort_key,
    )

    return {
        "schema_version": "co_scientist_report_v1",
        "generated_at_utc": _utc_now(),
        "task_id": task.task_id,
        "status": task.status,
        "quality_gate_passed": quality_passed,
        "recommendation": recommendation or answer_summary,
        "confidence": confidence,
        "evidence_ref_count": diagnostics_evidence_count,
        "tool_call_count": diagnostics_tool_calls,
        "source_families": source_families,
        "mcp_response_contracts": {
            "valid_count": max(contract_ok, 0),
            "expected_count": max(contract_expected, 0),
            "violation_count": max(contract_expected - contract_ok, 0),
        },
        "claim_provenance": {
            "assertive_claim_count": int(quality_report.get("claim_provenance_claim_count", 0)) if quality_report else 0,
            "uncited_claim_count": int(quality_report.get("claim_provenance_uncited_count", 0)) if quality_report else 0,
        },
        "validated_evidence_refs": prioritized_validated_refs[:40],
        "unverified_output_refs": prioritized_unverified_refs[:40],
        "unresolved_gaps": unresolved[:10],
        "next_actions": [str(item).strip() for item in next_actions if str(item).strip()][:5],
    }


def _methodology_strategy_sentence(task: WorkflowTask) -> str:
    objective_lower = str(task.objective or "").lower()
    focus: list[str] = []
    keyword_focus = (
        (("genetic", "gwas", "variant", "direction-of-effect"), "human genetics direction-of-effect"),
        (("trial", "clinical", "nct", "phase"), "clinical outcome patterns"),
        (("safety", "toxicity", "adverse", "liability"), "safety liabilities"),
        (("chembl", "compound", "potency", "druggability"), "chemical and tractability evidence"),
        (("competitive", "pipeline", "program", "landscape"), "development landscape signals"),
        (("pathway", "interaction", "reactome", "string"), "pathway/network context"),
        (("expression", "tissue", "cell type", "single-cell"), "tissue and cell-context evidence"),
        (("researcher", "author", "investigator", "affiliation"), "research-ecosystem signals"),
    )
    for tokens, label in keyword_focus:
        if any(token in objective_lower for token in tokens):
            focus.append(label)
    if focus:
        if len(focus) == 1:
            focus_text = focus[0]
        elif len(focus) == 2:
            focus_text = f"{focus[0]} and {focus[1]}"
        else:
            focus_text = f"{', '.join(focus[:-1])}, and {focus[-1]}"
        return (
            "The workflow was composed around the detected request scope and prioritized "
            f"{focus_text} before final synthesis."
        )
    return (
        "The workflow first scoped the objective, then assembled evidence modules aligned to the request, "
        "and finally synthesized findings into a decision-oriented recommendation."
    )


def _first_sentence(text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if not value:
        return ""
    match = re.match(r"^(.*?[.!?])(?:\s+.*)?$", value)
    return (match.group(1) if match else value).strip()


def _methodology_reason_for_step(step: WorkflowStep) -> str:
    reason = _first_sentence(step.rationale or "") or _first_sentence(step.instruction or "")
    if not reason:
        return "This step was required to support the final recommendation with traceable evidence."
    reason = _humanize_internal_tool_mentions(reason)
    return reason


def _summarize_tool_actions_for_step(tool_names: list[str]) -> str:
    phrases: list[str] = []
    for name in sorted({str(tool).strip() for tool in tool_names if str(tool).strip()}):
        phrase = _TOOL_HUMAN_ACTIONS.get(name)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    if not phrases:
        sources = sorted(
            {
                source
                for source in (_source_for_tool_name(name) for name in tool_names)
                if source
            }
        )
        if not sources:
            return "reviewed and synthesized previously gathered evidence"
        if len(sources) == 1:
            return f"queried {sources[0]} to gather evidence for this step"
        if len(sources) == 2:
            return f"queried {sources[0]} and {sources[1]} to gather complementary evidence"
        lead = ", ".join(sources[:-1])
        return f"queried {lead}, and {sources[-1]} to gather complementary evidence"

    if len(phrases) <= 2:
        return " and ".join(phrases)
    if len(phrases) == 3:
        return f"{phrases[0]}, {phrases[1]}, and {phrases[2]}"
    return f"{phrases[0]}, {phrases[1]}, and additional supporting evidence lookups"


def _detect_human_readable_pivots(trace_entries: list[dict], max_items: int = 3) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()
    failure_outcomes = {"error", "not_found_or_empty", "no_response", "degraded"}
    for idx in range(len(trace_entries) - 1):
        current = trace_entries[idx]
        nxt = trace_entries[idx + 1]
        outcome = str(current.get("outcome", "unknown")).strip().lower()
        if outcome not in failure_outcomes:
            continue
        current_source = _source_for_tool_name(str(current.get("tool_name", ""))) or "an evidence source"
        next_source = _source_for_tool_name(str(nxt.get("tool_name", ""))) or "an alternate evidence source"
        if current_source == next_source:
            note = (
                f"After a {outcome} response from {current_source}, the workflow retried with an alternate query path."
            )
        else:
            note = f"After a {outcome} response from {current_source}, the workflow pivoted to {next_source}."
        if note in seen:
            continue
        seen.add(note)
        notes.append(note)
        if len(notes) >= max_items:
            break
    return notes


def _render_methodology_brief(task: WorkflowTask) -> list[str]:
    lines = ["## Methodology"]
    executed_steps = [step.title for step in task.steps if step.status != "pending"]
    trace_entries = [entry for step in task.steps for entry in step.tool_trace]
    source_families = _detect_source_families(task)

    lines.append(_methodology_strategy_sentence(task))
    if source_families:
        lines.append(f"Evidence was triangulated across {', '.join(source_families)}.")

    if executed_steps:
        lines.append("Step-by-step execution:")
        for idx, step in enumerate((step for step in task.steps if step.status != "pending"), start=1):
            step_tool_names = [
                str(entry.get("tool_name", "")).strip()
                for entry in (step.tool_trace or [])
                if str(entry.get("tool_name", "")).strip()
            ]
            action_summary = _summarize_tool_actions_for_step(step_tool_names)
            reason = _methodology_reason_for_step(step)
            failure_count = sum(
                1
                for entry in (step.tool_trace or [])
                if str(entry.get("outcome", "")).strip().lower() in {"error", "not_found_or_empty", "no_response", "degraded"}
            )
            line = f"- Step {idx} ({step.title}): {action_summary}. Reasoning: {reason}"
            if failure_count:
                line += (
                    f" {failure_count} call(s) returned partial/error outcomes, so the workflow cross-checked with alternate evidence where available."
                )
            lines.append(line)
    else:
        lines.append("No executable workflow stages were completed.")
        return lines

    if not trace_entries:
        lines.append("No tool-based evidence calls were recorded in this run.")
        return lines

    outcome_counts: dict[str, int] = {}
    for entry in trace_entries:
        outcome = str(entry.get("outcome", "unknown")).strip().lower()
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    ok_count = outcome_counts.get("ok", 0)
    issue_count = (
        outcome_counts.get("error", 0)
        + outcome_counts.get("not_found_or_empty", 0)
        + outcome_counts.get("no_response", 0)
        + outcome_counts.get("degraded", 0)
    )
    lines.append(
        f"Across all steps, {len(trace_entries)} evidence calls were executed: {ok_count} successful and "
        f"{issue_count} with partial/error outcomes."
    )

    pivot_notes = _detect_human_readable_pivots(trace_entries)
    if pivot_notes:
        lines.append("When evidence-retrieval issues occurred, the workflow adapted as follows:")
        lines.extend([f"- {note}" for note in pivot_notes])
    else:
        lines.append("No major evidence-retrieval pivots were required.")
    return lines


def _build_agentic_structured_summary(task: WorkflowTask, quality_report: dict | None = None) -> dict:
    raw_answer_text = task.steps[-1].output if task.steps else "No answer generated."
    answer_text = raw_answer_text.strip() if raw_answer_text else "No answer generated."
    answer_text = _strip_legacy_report_sections(answer_text)
    answer_text = _sanitize_markdown_artifacts(answer_text)
    answer_text = _dedupe_answer_paragraphs(answer_text)
    answer_body, extracted_actions = _extract_next_actions(answer_text)

    recommendation = _extract_answer_line(answer_body, "Recommendation")
    confidence = _extract_answer_line(answer_body, "Confidence Level") or _extract_answer_line(answer_body, "Confidence")
    narrative_open = _extract_first_narrative_paragraph(answer_body) or "The available evidence supports the recommendation below."

    answer_summary = recommendation or narrative_open
    if confidence:
        answer_summary = f"{answer_summary} Confidence: {confidence}."
    answer_summary = _humanize_internal_tool_mentions(answer_summary)
    answer_summary = _summarize_step_output(answer_summary, max_chars=440)
    if _looks_like_placeholder_summary(answer_summary):
        fallback_summary = _humanize_internal_tool_mentions(
            _summarize_step_output(_extract_first_narrative_paragraph(answer_body) or narrative_open, max_chars=440)
        )
        if not _looks_like_placeholder_summary(fallback_summary):
            answer_summary = fallback_summary
    answer_summary = re.sub(r"^\s*Recommendation:\s*", "", answer_summary, flags=re.IGNORECASE).strip()
    answer_summary = _strip_leading_citation_tokens(answer_summary)

    formatted_references = _collect_formatted_references(task)
    requested_papers = _requested_paper_count(task.objective)
    source_families = _detect_source_families(task)
    selected_references = formatted_references[:requested_papers] if requested_papers else formatted_references[:8]

    data_retrieved: list[dict[str, str]] = []
    for source in source_families:
        data_retrieved.append({"source": source, "summary": f"Evidence retrieved from {source} during workflow execution."})
    for reference in selected_references:
        data_retrieved.append({"source": "reference", "summary": reference})
    if not data_retrieved:
        data_retrieved.append(
            {"source": "workflow", "summary": "No tool-validated references were captured in this run; synthesis is based on model output only."}
        )

    executed_steps = [step for step in task.steps if step.status != "pending"]
    steps_taken: list[str] = []
    for idx, step in enumerate(executed_steps, start=1):
        step_tool_names = [
            str(entry.get("tool_name", "")).strip()
            for entry in (step.tool_trace or [])
            if str(entry.get("tool_name", "")).strip()
        ]
        action_summary = _summarize_tool_actions_for_step(step_tool_names)
        reason = _methodology_reason_for_step(step)
        steps_taken.append(f"Step {idx} ({step.title}): {action_summary}. Why: {reason}")
    if not steps_taken:
        steps_taken.append("No executable workflow stages were completed.")

    gaps = quality_report.get("unresolved_gaps", []) if quality_report else []
    limitations = [_humanize_internal_tool_mentions(str(gap)) for gap in gaps[:8] if str(gap).strip()]
    if not limitations:
        limitations = ["No critical unresolved gaps were detected by current quality checks."]

    next_actions = extracted_actions[:5]
    if not next_actions and gaps:
        next_actions = [f"Resolve: {gap}" for gap in gaps[:3]]
    if not next_actions:
        next_actions = [
            "Validate the recommendation against one independent source.",
            "Run a focused follow-up for the highest-uncertainty evidence gap.",
            "Convert findings into an execution plan with owners and milestones.",
        ]
    next_actions = [_humanize_internal_tool_mentions(item) for item in next_actions]

    return {
        "objective": _extract_primary_query(task.objective),
        "answer": answer_summary,
        "data_retrieved": data_retrieved[:20],
        "steps_taken": steps_taken[:12],
        "limitations": limitations,
        "next_actions": next_actions,
        "confidence": confidence or "unspecified",
    }


def _render_agentic_structured_summary_markdown(summary: dict) -> str:
    lines: list[str] = []
    objective = str(summary.get("objective", "")).strip()
    if objective:
        lines.extend(["## Objective", objective, ""])

    lines.extend(["## Structured Summary", str(summary.get("answer", "")).strip() or "No answer generated.", ""])
    lines.append(f"- Confidence: {str(summary.get('confidence', 'unspecified')).strip() or 'unspecified'}")

    data_retrieved = summary.get("data_retrieved", [])
    lines.extend(["", "## Data Retrieved"])
    if isinstance(data_retrieved, list) and data_retrieved:
        for item in data_retrieved:
            if isinstance(item, dict):
                source = str(item.get("source", "source")).strip() or "source"
                text = str(item.get("summary", "")).strip()
                if text:
                    lines.append(f"- {source}: {text}")
            else:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
    else:
        lines.append("- No data retrieval details were captured.")

    steps_taken = summary.get("steps_taken", [])
    lines.extend(["", "## Steps Taken"])
    if isinstance(steps_taken, list) and steps_taken:
        lines.extend([f"- {str(step).strip()}" for step in steps_taken if str(step).strip()])
    else:
        lines.append("- No workflow execution steps were recorded.")

    limitations = summary.get("limitations", [])
    lines.extend(["", "## Limitations"])
    if isinstance(limitations, list) and limitations:
        lines.extend([f"- {str(item).strip()}" for item in limitations if str(item).strip()])
    else:
        lines.append("- No limitations reported.")

    next_actions = summary.get("next_actions", [])
    lines.extend(["", "## Next Actions"])
    if isinstance(next_actions, list) and next_actions:
        lines.extend([f"- {str(item).strip()}" for item in next_actions if str(item).strip()])
    else:
        lines.append("- No immediate next actions were provided.")

    return "\n".join(lines).strip()


def render_final_report(task: WorkflowTask, quality_report: dict | None = None) -> str:
    raw = task.steps[-1].output if task.steps else ""
    text = (raw or "").strip()
    if not text:
        return "No answer generated."
    text = _sanitize_markdown_artifacts(text)
    text = _humanize_internal_tool_mentions(text)
    return text


def _final_response_principles(objective: str) -> list[str]:
    del objective
    return [
        "Adapt structure to the query type and evidence quality.",
        "Lead with a concise and direct answer first, and then elaborate on the rationale and methodology in separate sections.",
        "Methodology should include a provenance summary detailing the search strategy and tools utilized to reach the conclusion.",
        "Literature supporting the answer should be cited and listed as full references at the end, while tools should be mentioned in the text using real-world names.",
        "Use only the sections, bullets, or tables that improve clarity for the specific query.",
        "Ground major claims in executed evidence and cite source IDs or references where possible.",
    ]


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
    decision_quality_guardrail = (
        "\nDecision quality requirements:\n"
        "- Use explicit criteria when comparing or ranking options.\n"
        "- If high-priority evidence is missing, mark conclusions provisional and call out the gap.\n"
        "- Avoid high-confidence claims when primary evidence retrieval is degraded or empty.\n"
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
        final_principles_block = "\n".join(f"- {item}" for item in _final_response_principles(task.objective))
        step_freshness_guardrail += (
            "\nFinal response principles:\n"
            f"{final_principles_block}\n"
            "- After the opening answer, include separate `Rationale` and `Methodology` sections.\n"
            "- `Methodology` must include a provenance summary of search strategy and tools used.\n"
            "- Mention tools in narrative using real-world names (for example, PubMed or OpenAlex), not internal tool identifiers.\n"
            "- Add a final `References` section with full formatted citations for supporting literature.\n"
            "- Include DOI markdown links when available.\n"
            "- Include an optional `Next Actions` section (up to 3 items) only when it materially helps the user.\n"
            "- Keep the narrative concise and avoid repeating full evidence lists already stated in prior steps.\n"
        )
    role_directive = "executor"
    if step.step_id == "step_1" or step.subgoal_id == "sg_scope":
        role_directive = "planner"
    elif step.subgoal_id == "sg_critique":
        role_directive = "critic"
    elif task.steps and step.step_id == task.steps[-1].step_id:
        role_directive = "report_synthesizer"

    tool_shortlist_context = ""
    if step.allowed_tools:
        preview = ", ".join(step.allowed_tools[:10])
        detail_lines = [line for line in step.observations if str(line).strip().startswith("- ")]
        details = "\n".join(detail_lines[:6]) if detail_lines else ""
        tool_shortlist_context = (
            "\nDynamic tool shortlist for this step:\n"
            f"- {preview}\n"
            f"{details}\n"
            "- Use this shortlist as the primary search space unless escalation is explicitly justified.\n"
        )

    evidence_requirements = [str(item).strip() for item in step.evidence_requirements if str(item).strip()]
    evidence_requirements_block = ""
    if evidence_requirements:
        evidence_requirements_block = (
            "\nEvidence requirements for this subgoal:\n"
            f"- {', '.join(evidence_requirements)}\n"
        )

    return (
        "You are executing a co-investigator workflow.\n"
        f"Active role: {role_directive}\n"
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
        "When citing literature, use full citations and include DOI markdown links when available; avoid citation IDs alone.\n"
        "Use source citations whenever possible (PMID, NCT IDs).\n"
        f"{tool_shortlist_context}"
        f"{evidence_requirements_block}"
        f"{revision_directive_block}"
        f"{step_freshness_guardrail}"
        f"{decision_quality_guardrail}"
    )
