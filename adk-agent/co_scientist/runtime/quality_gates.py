"""
Quality gates, adaptive checkpoint policy, and fallback recovery helpers.
"""
from __future__ import annotations

import re

from .event_orchestrator import (
    EVENT_CHECKPOINT_OPENED,
    PHASE_SYNTHESIS,
    append_event,
    checkpoint_payload_for_transition,
    ensure_phase_state,
    infer_phase_for_step,
    should_checkpoint_for_phase_boundary,
)

_ASSERTIVE_CLAIM_RE = re.compile(
    r"\b("
    r"recommend(?:ation|ed)?|"
    r"prioriti[sz]e(?:d|s)?|deprioriti[sz]e(?:d|s)?|"
    r"should|must|"
    r"outperform(?:s|ed)?|superior|inferior|"
    r"trap\b"
    r")\b",
    flags=re.IGNORECASE,
)
_HEDGED_CLAIM_RE = re.compile(
    r"\b("
    r"may|might|could|"
    r"possible|possibly|"
    r"preliminary|"
    r"uncertain|unknown|"
    r"insufficient|limited|"
    r"provisional|tentative"
    r")\b",
    flags=re.IGNORECASE,
)

_GENERIC_RECOMMENDATION_PATTERNS = (
    "the available evidence supports the recommendation below",
    "the recommendation is based on the strongest available evidence",
    "no answer generated",
    "recommendation pending",
)


def _contains_pubmed_ref(refs: set[str]) -> bool:
    return any(re.match(r"^PMID:\d{5,9}$", str(ref).strip(), flags=re.IGNORECASE) for ref in refs)


def _select_inline_citation(refs: set[str]) -> str | None:
    if not refs:
        return None
    normalized = sorted({str(ref).strip() for ref in refs if str(ref).strip()})
    priority_prefixes = ("PMID:", "NCT", "DOI:", "OpenAlex:")
    for prefix in priority_prefixes:
        for ref in normalized:
            if ref.upper().startswith(prefix.upper()):
                return ref
    return normalized[0] if normalized else None


def _append_inline_citation_to_first_sentence(text: str, inline_ref: str) -> str:
    body = str(text or "").strip()
    if not body:
        return body
    first_sentence_match = re.match(r"^(.*?[.!?])(\s+.*)?$", body)
    first_sentence = first_sentence_match.group(1).strip() if first_sentence_match else body
    remainder = first_sentence_match.group(2) or "" if first_sentence_match else ""
    if _extract_evidence_refs(first_sentence):
        return body
    if first_sentence.endswith((".", "!", "?")):
        first_sentence = f"{first_sentence[:-1]} ({inline_ref}){first_sentence[-1]}"
    else:
        first_sentence = f"{first_sentence} ({inline_ref})"
    return f"{first_sentence}{remainder}"


def _extract_recommendation_sections(text: str) -> list[str]:
    if not text:
        return []
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").splitlines()
    inline_label_re = re.compile(
        r"^\s*(?:[-*]\s+)?(?:#+\s*)?(?:\*{0,2})?(?:revised_)?recommendation(?:\*{0,2})\s*:\s*(?:\*{0,2})?\s*(?P<body>.+?)\s*$",
        flags=re.IGNORECASE,
    )
    label_only_re = re.compile(
        r"^\s*(?:[-*]\s+)?(?:#+\s*)?(?:\*{0,2})?(?:revised_)?recommendation(?:\*{0,2})\s*:\s*(?:\*{0,2})?\s*$",
        flags=re.IGNORECASE,
    )
    section_heading_re = re.compile(
        r"^\s*(?:#+\s*)?(?:\*{0,2})?[A-Za-z][A-Za-z0-9 _/\-]{1,60}(?:\*{0,2})?\s*:\s*$",
        flags=re.IGNORECASE,
    )

    extracted: list[str] = []
    idx = 0
    while idx < len(lines):
        line = str(lines[idx]).strip()
        if not label_only_re.match(line):
            inline_match = inline_label_re.match(line)
            if inline_match:
                body = re.sub(r"\s+", " ", inline_match.group("body").strip())
                if body:
                    extracted.append(body)
                idx += 1
                continue
            idx += 1
            continue

        j = idx + 1
        while j < len(lines) and not str(lines[j]).strip():
            j += 1
        section_lines: list[str] = []
        while j < len(lines):
            candidate = str(lines[j]).strip()
            if not candidate:
                if section_lines:
                    break
                j += 1
                continue
            if section_heading_re.match(candidate):
                break
            section_lines.append(candidate)
            j += 1
        if section_lines:
            extracted.append(re.sub(r"\s+", " ", " ".join(section_lines)).strip())
        idx = j

    deduped: list[str] = []
    seen: set[str] = set()
    for section in extracted:
        normalized = section.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _extract_primary_narrative_paragraph(text: str) -> str:
    if not text:
        return ""
    section_heading_re = re.compile(
        r"^\s*(?:#+\s*)?(?:\*{0,2})?[A-Za-z][A-Za-z0-9 _/\-]{1,60}(?:\*{0,2})?\s*:\s*$",
        flags=re.IGNORECASE,
    )
    for paragraph in re.split(r"\n\s*\n", str(text)):
        normalized = str(paragraph).strip()
        if not normalized:
            continue
        if section_heading_re.match(normalized):
            continue
        return normalized
    return ""


def _inject_inline_citation_in_recommendation(text: str, refs: set[str]) -> str:
    if not text:
        return text
    inline_ref = _select_inline_citation(refs)
    if not inline_ref:
        return text

    lines = text.splitlines()
    updated = False
    for idx, raw in enumerate(lines):
        line = str(raw).rstrip()
        label_only = re.match(
            r"^\s*(?:[-*]\s+)?(?:#+\s*)?(?:\*{0,2})?(?:revised_)?recommendation(?:\*{0,2})\s*:\s*(?:\*{0,2})?\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if label_only:
            target_idx = idx + 1
            while target_idx < len(lines) and not str(lines[target_idx]).strip():
                target_idx += 1
            if target_idx >= len(lines):
                continue
            target_line = str(lines[target_idx]).strip()
            repaired_line = _append_inline_citation_to_first_sentence(target_line, inline_ref)
            if repaired_line == target_line:
                continue
            leading_ws = re.match(r"^\s*", str(lines[target_idx])).group(0)
            lines[target_idx] = f"{leading_ws}{repaired_line}"
            updated = True
            break

        match = re.match(
            r"^(?P<prefix>\s*(?:[-*]\s+)?(?:#+\s*)?(?:\*{0,2})?(?:revised_)?recommendation(?:\*{0,2})\s*:\s*(?:\*{0,2})?\s*)(?P<body>.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            prefix = match.group("prefix")
            body = match.group("body").strip()
            if not body:
                continue
            repaired = _append_inline_citation_to_first_sentence(body, inline_ref)
            if repaired == body:
                continue
            lines[idx] = f"{prefix}{repaired}"
            updated = True
            break

    if not updated:
        section_heading_re = re.compile(
            r"^\s*(?:#+\s*)?(?:\*{0,2})?[A-Za-z][A-Za-z0-9 _/\-]{1,60}(?:\*{0,2})?\s*:\s*$",
            flags=re.IGNORECASE,
        )
        for idx, raw in enumerate(lines):
            stripped = str(raw).strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if section_heading_re.match(stripped):
                continue
            if re.match(r"^(?:[-*]|\d+[.)])\s+", stripped):
                continue
            if not _ASSERTIVE_CLAIM_RE.search(stripped):
                continue
            repaired = _append_inline_citation_to_first_sentence(stripped, inline_ref)
            if repaired == stripped:
                continue
            leading_ws = re.match(r"^\s*", str(raw)).group(0)
            lines[idx] = f"{leading_ws}{repaired}"
            updated = True
            break

    if not updated:
        return text
    return "\n".join(lines)


def _extract_evidence_refs(text: str) -> set[str]:
    if not text:
        return set()
    pmids = {f"PMID:{match}" for match in re.findall(r"\bPMID[:\s]*([0-9]{5,9})\b", text, flags=re.IGNORECASE)}
    ncts = {match.upper() for match in re.findall(r"\b(NCT[0-9]{8})\b", text, flags=re.IGNORECASE)}
    dois = {f"DOI:{match.lower()}" for match in re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, flags=re.IGNORECASE)}
    openalex_urls = {match for match in re.findall(r"https?://openalex\.org/[AW]\d+", text, flags=re.IGNORECASE)}
    openalex_ids = {
        f"OpenAlex:{match.upper()}"
        for match in re.findall(r"\bhttps?://openalex\.org/([AW]\d+)\b", text, flags=re.IGNORECASE)
    }
    reactome_ids = {f"Reactome:{match}" for match in re.findall(r"\bR-HSA-\d+\b", text, flags=re.IGNORECASE)}
    string_ids = {f"STRING:{match}" for match in re.findall(r"\b9606\.[A-Za-z0-9_.-]+\b", text, flags=re.IGNORECASE)}
    chembl_ids = {
        f"ChEMBL:{match.upper()}"
        for match in re.findall(r"\b(CHEMBL\d{3,})\b", text, flags=re.IGNORECASE)
    }
    mondo_ids = {
        f"MONDO:{match.upper().replace(':', '_')}"
        for match in re.findall(r"\bMONDO[:_]\d+\b", text, flags=re.IGNORECASE)
    }
    efo_ids = {f"EFO:{match.upper().replace(':', '_')}" for match in re.findall(r"\bEFO[:_]\d+\b", text, flags=re.IGNORECASE)}
    rs_ids = {f"GWAS:{match.lower()}" for match in re.findall(r"\b(rs\d+)\b", text, flags=re.IGNORECASE)}

    return pmids | ncts | dois | openalex_urls | openalex_ids | reactome_ids | string_ids | chembl_ids | mondo_ids | efo_ids | rs_ids


def _collect_trace_entries(task) -> list[dict]:
    entries = [entry for step in task.steps for entry in (step.tool_trace or []) if isinstance(entry, dict)]
    return entries


def _collect_output_evidence_refs(task) -> set[str]:
    refs: set[str] = set()
    for step in task.steps:
        refs.update(_extract_evidence_refs(step.output or ""))
    return refs


def _collect_tool_evidence_refs(trace_entries: list[dict]) -> set[str]:
    refs: set[str] = set()
    for entry in trace_entries:
        for ref in entry.get("evidence_refs") or []:
            normalized = str(ref).strip()
            if normalized:
                refs.add(normalized)
    return refs


def _collect_mcp_contract_violations(trace_entries: list[dict]) -> tuple[int, int, list[str]]:
    expected = 0
    violations: list[str] = []
    for entry in trace_entries:
        outcome = str(entry.get("outcome", "unknown"))
        # Enforce response-contract validation on successful/degraded payloads.
        # Error/empty outcomes may return transport-level failures without a
        # structured payload contract and are handled separately by failure gates.
        if outcome not in {"ok", "degraded"}:
            continue
        expected += 1
        contract = entry.get("response_contract")
        tool_name = str(entry.get("tool_name", "unknown_tool"))
        if not isinstance(contract, dict):
            violations.append(f"{tool_name}: missing response contract metadata.")
            continue
        if not bool(contract.get("valid", False)):
            issues = contract.get("issues")
            issue_text = ""
            if isinstance(issues, list) and issues:
                issue_text = str(issues[0])
            violations.append(f"{tool_name}: {issue_text or 'response contract validation failed.'}")
    return expected, len(violations), violations


def _split_claim_candidates(text: str) -> list[str]:
    if not text:
        return []
    cleaned = str(text).replace("\r\n", "\n").replace("\r", "\n")
    candidates: list[str] = []
    for paragraph in re.split(r"\n\s*\n", cleaned):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for raw_segment in re.split(r"(?<=[.!?])\s+|\n+", paragraph):
            segment = str(raw_segment).strip()
            if not segment:
                continue
            segment = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", segment)
            segment = segment.replace("**", "").replace("`", "")
            segment = re.sub(r"\s+", " ", segment).strip()
            if len(segment) < 24:
                continue
            candidates.append(segment)
    return candidates


def _is_assertive_claim(sentence: str) -> bool:
    normalized = str(sentence or "").strip()
    if not normalized:
        return False
    if normalized.startswith("#"):
        return False
    if re.match(r"^(?:recommendation|rationale narrative|next actions?)\s*[:\-]?\s*$", normalized, flags=re.IGNORECASE):
        return False
    if _HEDGED_CLAIM_RE.search(normalized):
        return False
    if not _ASSERTIVE_CLAIM_RE.search(normalized):
        return False
    if normalized.lower().startswith(("next action", "resolve:", "validate the recommendation")):
        return False
    return True


def _collect_uncited_assertive_claims(task, tool_evidence_refs: set[str]) -> tuple[int, list[str]]:
    if not task.steps:
        return 0, []

    final_output = str(task.steps[-1].output or "").strip()
    synthesis_blocks: list[str] = []
    final_recommendations = _extract_recommendation_sections(final_output)
    if final_recommendations:
        synthesis_blocks.extend(final_recommendations)
    else:
        primary_paragraph = _extract_primary_narrative_paragraph(final_output)
        if primary_paragraph:
            synthesis_blocks.append(primary_paragraph)
    synthesis_text = "\n\n".join(part for part in synthesis_blocks if part)
    if not synthesis_text:
        return 0, []

    uncited_claims: list[str] = []
    assertive_claim_count = 0
    for paragraph in re.split(r"\n\s*\n", synthesis_text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_refs = _extract_evidence_refs(paragraph)
        for candidate in _split_claim_candidates(paragraph):
            if not _is_assertive_claim(candidate):
                continue
            assertive_claim_count += 1
            candidate_refs = _extract_evidence_refs(candidate)
            if (candidate_refs | paragraph_refs).intersection(tool_evidence_refs):
                continue
            excerpt = candidate if len(candidate) <= 180 else f"{candidate[:177].rstrip()}..."
            if excerpt not in uncited_claims:
                uncited_claims.append(excerpt)
    return assertive_claim_count, uncited_claims


def _requested_paper_count(objective: str) -> int | None:
    match = re.search(r"\b(\d{1,2})\s+(?:recent\s+)?papers?\b", str(objective or ""), flags=re.IGNORECASE)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    return value if value > 0 else None


def _count_literature_citations_in_text(text: str) -> int:
    raw_refs = _extract_evidence_refs(text or "")
    keys: set[str] = set()
    for ref in raw_refs:
        value = str(ref).strip()
        upper = value.upper()
        if upper.startswith("DOI:"):
            keys.add(f"DOI:{value.split(':', 1)[1].strip().lower()}")
        elif upper.startswith("PMID:"):
            keys.add(f"PMID:{value.split(':', 1)[1].strip()}")
        elif upper.startswith("OPENALEX:"):
            keys.add(f"OPENALEX:{value.split(':', 1)[1].strip().upper()}")
        elif upper.startswith("HTTP://OPENALEX.ORG/") or upper.startswith("HTTPS://OPENALEX.ORG/"):
            keys.add(value.upper())
    return len(keys)


def _selected_recommendation_text(task) -> str:
    if not task.steps:
        return ""
    final_output = str(task.steps[-1].output or "").strip()
    final_recommendations = _extract_recommendation_sections(final_output)
    if final_recommendations:
        return " ".join(final_recommendations).strip()
    return _extract_primary_narrative_paragraph(final_output).strip()


def _looks_placeholder_recommendation(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip().lower()
    if not normalized:
        return True
    if any(marker in normalized for marker in _GENERIC_RECOMMENDATION_PATTERNS):
        return True
    # Recommendation should communicate an explicit decision, not only generic framing.
    if len(normalized) < 70 and not _ASSERTIVE_CLAIM_RE.search(normalized):
        return True
    return False


def evaluate_quality_gates(task) -> dict:
    trace_entries = _collect_trace_entries(task)
    tool_evidence_refs = _collect_tool_evidence_refs(trace_entries)
    if task.steps and tool_evidence_refs:
        final_step = task.steps[-1]
        final_output = str(final_step.output or "")
        repaired_final_output = _inject_inline_citation_in_recommendation(final_output, tool_evidence_refs)
        if repaired_final_output != final_output:
            final_step.output = repaired_final_output
    output_evidence_refs = _collect_output_evidence_refs(task)
    evidence_count = len(tool_evidence_refs)
    steps_with_output = sum(1 for step in task.steps if step.output and step.output != "(No response generated)")
    coverage_ratio = steps_with_output / len(task.steps) if task.steps else 0.0
    tool_call_count = len(trace_entries)

    unresolved_gaps: list[str] = []

    def _append_gap(message: str) -> None:
        normalized = str(message or "").strip()
        if normalized and normalized not in unresolved_gaps:
            unresolved_gaps.append(normalized)

    combined_output = "\n".join(step.output for step in task.steps if step.output).lower()
    if any(
        marker in combined_output
        for marker in [
            "cannot be fulfilled",
            "cannot be completed",
            "insufficient data",
            "unable to identify",
            "unable to retrieve",
            "tool limitation",
            "service unavailable",
        ]
    ):
        _append_gap("Output contains self-reported evidence limitations that may affect confidence.")

    failed_entries = [
        entry
        for step in task.steps
        for entry in (step.tool_trace or [])
        if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
    ]
    if failed_entries:
        failure_count = len(failed_entries)
        failure_ratio = failure_count / max(tool_call_count, 1)
        if failure_count >= 2 or failure_ratio >= 0.35:
            _append_gap(
                "Tool execution issues were detected "
                f"({failure_count} failed or empty tool calls)."
            )

    critical_marker_patterns = (
        r"\bcritical gap\s*[:\-]",
        r"\bcritical missing evidence\b",
        r"\bservice unavailable\b",
        r"\bfailed due to api error(?:s)?\b",
        r"\bcould not retrieve\b",
        r"\bunable to retrieve\b",
        r"\bpersistent failure\b",
    )
    if any(re.search(pattern, combined_output, flags=re.IGNORECASE) for pattern in critical_marker_patterns):
        _append_gap("Output reports critical missing evidence that affects confidence in the recommendation.")
    if any(marker in combined_output for marker in ["not directly from tool output", "historical knowledge"]):
        _append_gap("Synthesis includes claims that are not directly supported by captured tool output.")

    unverified_output_refs = sorted(output_evidence_refs - tool_evidence_refs)
    if unverified_output_refs:
        tolerance = 1 if evidence_count >= 5 else 0
        if len(unverified_output_refs) > tolerance:
            sample = ", ".join(unverified_output_refs[:3])
            _append_gap(
                "Some citation IDs in synthesis are not backed by captured tool responses "
                f"({sample}{', ...' if len(unverified_output_refs) > 3 else ''})."
            )

    assertive_claim_count, uncited_assertive_claims = _collect_uncited_assertive_claims(task, tool_evidence_refs)
    if uncited_assertive_claims:
        sample = "; ".join(uncited_assertive_claims[:2])
        _append_gap(
            "Assertive recommendation claims are missing inline validated citations "
            f"({sample}{'; ...' if len(uncited_assertive_claims) > 2 else ''})."
        )
    recommendation_text = _selected_recommendation_text(task)
    if not recommendation_text:
        _append_gap("Final synthesis is missing an explicit recommendation statement.")
    elif _looks_placeholder_recommendation(recommendation_text):
        _append_gap("Final recommendation is generic/template-like and not decision-specific.")
    requested_papers = _requested_paper_count(task.objective)
    if requested_papers:
        final_text = str(task.steps[-1].output or "") if task.steps else ""
        found_citations = _count_literature_citations_in_text(final_text)
        if found_citations < requested_papers:
            _append_gap(
                "Final synthesis cites fewer literature references than requested "
                f"({found_citations}/{requested_papers})."
            )

    if evidence_count == 0:
        _append_gap("No tool-validated citation evidence IDs were captured.")
    if tool_call_count == 0:
        _append_gap("No tool calls were captured for the workflow.")
    executed_tools = {
        str(entry.get("tool_name", "")).strip()
        for entry in trace_entries
        if str(entry.get("tool_name", "")).strip()
    }
    missing_tool_steps = [
        step.title
        for step in task.steps
        if (
            step.status == "completed"
            and step.recommended_tools
            and not step.tool_trace
            and not executed_tools.intersection(
                {str(tool).strip() for tool in step.recommended_tools if str(tool).strip()}
            )
        )
    ]
    if missing_tool_steps:
        _append_gap(
            "Completed steps with recommended tools but no recorded tool execution: "
            + ", ".join(missing_tool_steps)
        )

    contract_expected_count, contract_violation_count, contract_violations = _collect_mcp_contract_violations(
        trace_entries
    )
    if contract_violation_count:
        sample = "; ".join(contract_violations[:3])
        _append_gap(
            "MCP response contract validation failed for one or more tool calls "
            f"({sample}{'; ...' if len(contract_violations) > 3 else ''})."
        )

    # Confidence-and-coverage scoring with soft penalties for gaps.
    evidence_score = min(evidence_count / 4.0, 1.0)
    coverage_score = min(max(coverage_ratio, 0.0), 1.0)
    execution_score = min(tool_call_count / 3.0, 1.0)
    contract_score = 0.0 if contract_violation_count > 0 else 1.0
    gap_penalty = min(len(unresolved_gaps) / 5.0, 1.0)
    weighted_score = (
        0.32 * evidence_score
        + 0.24 * coverage_score
        + 0.14 * execution_score
        + 0.18 * contract_score
        + 0.12 * (1.0 - gap_penalty)
    )
    if weighted_score >= 0.75:
        confidence_label = "high"
    elif weighted_score >= 0.5:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    must_fail = contract_violation_count > 0 or recommendation_text == ""
    passed = bool((weighted_score >= 0.5) and not must_fail)
    task.quality_confidence = confidence_label
    if task.steps:
        task.steps[-1].confidence_label = confidence_label
        task.steps[-1].critic_verdict = "pass" if passed else "needs_revision"
    return {
        "passed": passed,
        "quality_score": round(weighted_score, 4),
        "quality_confidence": confidence_label,
        "evidence_count": evidence_count,
        "tool_evidence_count": len(tool_evidence_refs),
        "output_evidence_count": len(output_evidence_refs),
        "validated_evidence_refs": sorted(tool_evidence_refs),
        "unverified_output_refs": unverified_output_refs,
        "claim_provenance_claim_count": assertive_claim_count,
        "claim_provenance_uncited_count": len(uncited_assertive_claims),
        "claim_provenance_uncited_claims": uncited_assertive_claims,
        "coverage_ratio": coverage_ratio,
        "tool_call_count": tool_call_count,
        "mcp_contract_expected_count": contract_expected_count,
        "mcp_contract_ok_count": max(contract_expected_count - contract_violation_count, 0),
        "mcp_contract_violation_count": contract_violation_count,
        "mcp_contract_violations": contract_violations,
        "unresolved_gaps": unresolved_gaps,
    }


def gate_ack_token(reason: str, plan_version_id: str | None) -> str | None:
    normalized_reason = str(reason or "").strip().lower()
    if not normalized_reason:
        return None
    normalized_plan = str(plan_version_id or "none").strip() or "none"
    return f"gate_ack:{normalized_reason}:{normalized_plan}"


def should_open_checkpoint(
    task,
    next_step,
    quality_state: dict | None = None,
    queued_feedback: list[str] | None = None,
    *,
    active_plan_version_fn=None,
    gate_ack_token_fn=gate_ack_token,
) -> tuple[bool, str]:
    if not next_step:
        return False, "none"
    ensure_phase_state(task)
    if next_step.recommended_tools and not any(step.tool_trace for step in task.steps if step.status == "completed"):
        payload = {
            "from_phase": "pre_execution",
            "to_phase": infer_phase_for_step(next_step),
            "summary": "Initial execution gate before running evidence tools.",
            "question": "I have decomposed the plan. Should I begin evidence discovery now?",
        }
        task.checkpoint_payload = payload
        append_event(task, EVENT_CHECKPOINT_OPENED, reason="pre_evidence_execution", payload=payload)
        return True, "pre_evidence_execution"

    open_boundary, reason = should_checkpoint_for_phase_boundary(task, next_step)
    if open_boundary:
        plan_version_id = None
        if callable(active_plan_version_fn):
            try:
                plan_version = active_plan_version_fn(task)
            except Exception:
                plan_version = None
            if plan_version is not None:
                plan_version_id = str(getattr(plan_version, "version_id", "") or "").strip() or None
        ack_token = gate_ack_token_fn(reason, plan_version_id) if callable(gate_ack_token_fn) else None
        if ack_token and ack_token in getattr(task, "hitl_history", []):
            return False, "none"

        transition = str(reason.split(":", 1)[1]) if ":" in reason else ""
        from_phase, to_phase = (transition.split("->", 1) + [""])[:2]
        # Do not interrupt immediately before synthesis/reporting.
        if to_phase.strip() == PHASE_SYNTHESIS:
            return False, "none"
        payload = checkpoint_payload_for_transition(task, from_phase.strip(), to_phase.strip())
        task.checkpoint_payload = payload
        append_event(task, EVENT_CHECKPOINT_OPENED, reason=reason, payload=payload)
        return True, reason

    return False, "none"


def render_quality_gate_message(report: dict) -> str:
    lines = [
        "[Quality Gate Check]",
        f"- Evidence references found: {report['evidence_count']}",
        f"- Output references detected: {report.get('output_evidence_count', 0)}",
        f"- Step coverage ratio: {report['coverage_ratio']:.2f}",
        f"- Tool calls captured: {report.get('tool_call_count', 0)}",
        f"- Quality score/confidence: {report.get('quality_score', 0.0):.2f} / {report.get('quality_confidence', 'unknown')}",
        (
            "- MCP response contracts: "
            f"{report.get('mcp_contract_ok_count', 0)}/{report.get('mcp_contract_expected_count', 0)} valid"
        ),
        (
            "- Claim-level provenance: "
            f"{report.get('claim_provenance_claim_count', 0)} assertive claims scanned, "
            f"{report.get('claim_provenance_uncited_count', 0)} missing inline validated citations"
        ),
    ]
    if report["unresolved_gaps"]:
        lines.append("- Unresolved critical gaps:")
        lines.extend([f"  - {gap}" for gap in report["unresolved_gaps"]])
    else:
        lines.append("- Unresolved critical gaps: none")
    return "\n".join(lines)


def clean_recovery_text(text: str) -> str:
    if not text:
        return text
    seen = set()
    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if normalized in {"**3. key results:**", "3. key results:"}:
            if "3-key-results" in seen:
                continue
            seen.add("3-key-results")
        if normalized and normalized in seen and normalized.startswith("**"):
            continue
        if normalized:
            seen.add(normalized)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


async def complete_remaining_steps(
    runner,
    session_id: str,
    user_id: str,
    task,
    state_store,
    *,
    execute_step_fn,
    evaluate_quality_gates_fn=evaluate_quality_gates,
    render_quality_gate_message_fn=render_quality_gate_message,
    print_fn=print,
) -> dict:
    for idx in range(task.current_step_index + 1, len(task.steps)):
        step_text = await execute_step_fn(runner, session_id, user_id, task, idx)
        state_store.save_task(task, note=f"step_{idx + 1}_completed")
        print_fn(step_text)

    quality = evaluate_quality_gates_fn(task)
    print_fn("\n" + render_quality_gate_message_fn(quality))
    return quality


def format_checkpoint_reason(reason: str) -> str:
    mapping = {
        "pre_evidence_execution": "Before bulk evidence collection",
        "quality_gap_spike": "Quality/uncertainty spike detected",
        "repeated_tool_failures": "Repeated tool failures detected",
        "uncertainty_spike": "Uncertainty spike detected",
        "pre_final_after_intent_change": "Intent changed before final synthesis",
        "feedback_replan": "Plan updated from user feedback",
        "queued_feedback_pending": "Queued feedback pending application",
    }
    key = str(reason or "").strip()
    if key.startswith("phase_boundary:"):
        transition = key.split(":", 1)[1]
        from_phase, to_phase = (transition.split("->", 1) + [""])[:2]
        if from_phase and to_phase:
            return (
                "Phase boundary reached: "
                + from_phase.replace("_", " ")
                + " -> "
                + to_phase.replace("_", " ")
            )
    return mapping.get(key, key.replace("_", " ") if key else "unspecified")


def print_checkpoint_plan(
    task,
    *,
    active_plan_version_fn,
    format_checkpoint_reason_fn=format_checkpoint_reason,
    print_fn=print,
) -> None:
    print_fn("\n[Checkpoint Plan]")
    if task.latest_plan_delta:
        delta = task.latest_plan_delta
        print_fn("What changed:")
        print_fn(f"- {delta.summary or 'No structural changes.'}")
        if delta.added_steps:
            print_fn(f"- Added: {', '.join(delta.added_steps)}")
        if delta.removed_steps:
            print_fn(f"- Removed: {', '.join(delta.removed_steps)}")
        if delta.modified_steps:
            print_fn(f"- Modified: {', '.join(delta.modified_steps)}")
        if delta.reordered_steps:
            print_fn(f"- Reordered: {', '.join(delta.reordered_steps)}")
        print_fn("")

    version = active_plan_version_fn(task)
    if version and version.steps:
        print_fn("Remaining plan:")
        for idx, step in enumerate(version.steps, start=1):
            print_fn(f"{idx}. {step.title}")
    else:
        print_fn("Remaining plan: none")

    if task.checkpoint_reason:
        print_fn(f"\nCheckpoint reason: {format_checkpoint_reason_fn(task.checkpoint_reason)}")
