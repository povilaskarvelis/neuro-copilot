"""
Workflow models and rendering helpers for the Co-Investigator mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import re
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    expected_output_fields: list[str] = field(default_factory=list)

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
            "expected_output_fields": self.expected_output_fields,
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
            expected_output_fields=list(payload.get("expected_output_fields", [])),
        )


@dataclass
class WorkflowTask:
    """A task with explicit planner output and execution state."""

    task_id: str
    objective: str
    request_type: str = "exploration"
    intent_tags: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    status: str = "pending"
    steps: list[WorkflowStep] = field(default_factory=list)
    current_step_index: int = -1
    awaiting_hitl: bool = False
    hitl_history: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "request_type": self.request_type,
            "intent_tags": self.intent_tags,
            "success_criteria": self.success_criteria,
            "status": self.status,
            "steps": [step.to_dict() for step in self.steps],
            "current_step_index": self.current_step_index,
            "awaiting_hitl": self.awaiting_hitl,
            "hitl_history": self.hitl_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkflowTask":
        return cls(
            task_id=payload.get("task_id", ""),
            objective=payload.get("objective", ""),
            request_type=payload.get("request_type", "exploration"),
            intent_tags=list(payload.get("intent_tags", [])),
            success_criteria=list(payload.get("success_criteria", [])),
            status=payload.get("status", "pending"),
            steps=[WorkflowStep.from_dict(step) for step in payload.get("steps", [])],
            current_step_index=payload.get("current_step_index", -1),
            awaiting_hitl=bool(payload.get("awaiting_hitl", False)),
            hitl_history=list(payload.get("hitl_history", [])),
            created_at=payload.get("created_at", _utc_now()),
            updated_at=payload.get("updated_at", _utc_now()),
        )


def classify_request_type(objective: str) -> str:
    lower = objective.lower()
    if any(token in lower for token in ["compare", "versus", "vs", "tradeoff", "difference"]):
        return "comparison"
    if any(token in lower for token in ["rank", "prioritize", "top", "best"]):
        return "prioritization"
    if any(token in lower for token in ["validate", "verify", "check", "plausible", "hypothesis"]):
        return "validation"
    if any(token in lower for token in ["plan", "roadmap", "next steps", "how should", "what should"]):
        return "action_planning"
    return "exploration"


def infer_intent_tags(objective: str) -> list[str]:
    lower = objective.lower()
    tags: list[str] = []
    if any(token in lower for token in ["researcher", "author", "expert", "investigator", "contact"]):
        tags.append("researcher_discovery")
    if any(token in lower for token in ["evidence", "landscape", "literature", "paper", "publication"]):
        tags.append("evidence_landscape")
    if any(token in lower for token in ["variant", "clinvar", "pathogenic", "gwas", "mutation"]):
        tags.append("variant_check")
    if any(token in lower for token in ["pathway", "network", "interaction", "reactome", "string"]):
        tags.append("pathway_context")
    if any(token in lower for token in ["trial", "clinical", "nct", "phase", "terminated", "recruiting"]):
        tags.append("clinical_landscape")
    if any(token in lower for token in ["compound", "chembl", "potency", "ic50", "ki", "ec50", "small molecule", "chemical matter"]):
        tags.append("chemistry_evidence")
    if any(token in lower for token in ["ontology", "synonym", "disease context", "efo", "mondo"]):
        tags.append("ontology_expansion")
    if any(token in lower for token in ["expression", "tissue", "cell type", "single-cell", "cell context", "specificity"]):
        tags.append("expression_context")
    if any(token in lower for token in ["direction", "risk allele", "protective", "causal", "direction-of-effect", "doe"]):
        tags.append("genetics_direction")
    if any(token in lower for token in ["safety", "liability", "toxicity", "adverse", "off-target", "tolerability"]):
        tags.append("safety_assessment")
    if any(token in lower for token in ["competitive", "competition", "crowded", "white space", "novelty", "landscape"]):
        tags.append("competitive_landscape")
    if any(token in lower for token in ["compare", "versus", "vs"]):
        tags.append("comparison")
        tags.append("target_comparison")
    if any(token in lower for token in ["rank", "prioritize", "top"]):
        tags.append("prioritization")
        tags.append("target_comparison")
    if not tags:
        tags.append("evidence_landscape")
    return sorted(set(tags))


def tool_bundle_for_intent(intent_tags: list[str]) -> tuple[list[str], list[str]]:
    preferred: list[str] = []
    fallback: list[str] = []
    if "researcher_discovery" in intent_tags:
        preferred.extend(["search_pubmed_advanced", "get_pubmed_paper_details", "search_openalex_authors"])
        fallback.extend(["search_openalex_works", "search_clinical_trials", "get_clinical_trial"])
    if "evidence_landscape" in intent_tags:
        preferred.extend(["search_pubmed_advanced", "search_openalex_works", "expand_disease_context"])
        fallback.extend(["search_pubmed", "get_pubmed_abstract", "search_clinical_trials"])
    if "variant_check" in intent_tags:
        preferred.extend(["search_clinvar_variants", "get_clinvar_variant_details", "search_gwas_associations"])
        fallback.extend(["search_pubmed_advanced", "get_gene_info"])
    if "pathway_context" in intent_tags:
        preferred.extend(["search_reactome_pathways", "get_string_interactions"])
        fallback.extend(["search_pubmed_advanced", "search_targets"])
    if "clinical_landscape" in intent_tags:
        preferred.extend(["summarize_clinical_trials_landscape", "search_clinical_trials"])
        fallback.extend(["get_clinical_trial"])
    if "chemistry_evidence" in intent_tags:
        preferred.extend(["search_chembl_compounds_for_target", "search_targets"])
        fallback.extend(["get_target_drugs", "check_druggability"])
    if "ontology_expansion" in intent_tags:
        preferred.extend(["expand_disease_context", "search_diseases"])
        fallback.extend(["search_pubmed_advanced"])
    if "expression_context" in intent_tags:
        preferred.extend(["summarize_target_expression_context", "search_targets"])
        fallback.extend(["get_gene_info", "search_pubmed_advanced"])
    if "genetics_direction" in intent_tags:
        preferred.extend(["infer_genetic_effect_direction", "search_gwas_associations"])
        fallback.extend(["search_clinvar_variants", "search_pubmed_advanced"])
    if "safety_assessment" in intent_tags:
        preferred.extend(["summarize_target_safety_liabilities", "summarize_target_expression_context"])
        fallback.extend(["search_clinical_trials", "search_pubmed_advanced"])
    if "competitive_landscape" in intent_tags:
        preferred.extend(["summarize_target_competitive_landscape", "get_target_drugs"])
        fallback.extend(["summarize_clinical_trials_landscape", "search_chembl_compounds_for_target"])
    if "target_comparison" in intent_tags:
        preferred.extend(["compare_targets_multi_axis", "search_targets"])
        fallback.extend(["summarize_target_competitive_landscape", "summarize_target_safety_liabilities"])
    return sorted(set(preferred)), sorted(set(fallback))


def build_success_criteria(request_type: str) -> list[str]:
    criteria = [
        "Plan contains 2-4 executable steps aligned to the request.",
        "Each major claim is tied to at least one source or tool output.",
        "At least one uncertainty or limitation is stated explicitly.",
    ]
    if request_type == "comparison":
        criteria.append("Comparison includes explicit dimensions and trade-offs.")
    elif request_type == "prioritization":
        criteria.append("Ranking includes transparent criteria and confidence.")
    elif request_type == "validation":
        criteria.append("Validation includes supporting and contradictory evidence.")
    elif request_type == "action_planning":
        criteria.append("Next actions include concrete owners, dependencies, or sequence.")
    else:
        criteria.append("Findings summarize key signals and open questions.")
    return criteria


def _plan_for_expert_discovery(intent_tags: list[str]) -> list[WorkflowStep]:
    preferred, fallback = tool_bundle_for_intent(intent_tags)
    return [
        WorkflowStep(
            step_id="step_1",
            title="Scope and decomposition",
            instruction=(
                "Identify disease area, timeframe, and output format. "
                "Define what counts as an expert and list any assumptions."
            ),
            recommended_tools=[],
            fallback_tools=[],
            rationale="Clarify objective and criteria before expensive evidence retrieval.",
            expected_output_fields=["objective", "constraints", "success_criteria"],
        ),
        WorkflowStep(
            step_id="step_2",
            title="Evidence collection",
            instruction=(
                "Collect publication and trial evidence using available tools. "
                "Prioritize recent and high-signal researchers or targets."
            ),
            recommended_tools=preferred,
            fallback_tools=fallback,
            rationale="Use researcher-centric tools first; fallback to trial investigators when author data is sparse.",
            expected_output_fields=["selected_tools", "tool_rationale", "evidence_records", "blockers"],
        ),
        WorkflowStep(
            step_id="step_3",
            title="Structured synthesis",
            instruction=(
                "Return a structured shortlist with supporting citations, "
                "uncertainties, and recommended next actions."
            ),
            recommended_tools=[],
            fallback_tools=[],
            rationale="Produce a decision-ready summary with explicit gaps and next actions.",
            expected_output_fields=["findings", "confidence", "evidence_links", "limitations", "next_actions"],
        ),
    ]


def _plan_for_target_assessment(intent_tags: list[str]) -> list[WorkflowStep]:
    preferred, fallback = tool_bundle_for_intent(intent_tags)
    core_preferred = [
        "search_targets",
        "check_druggability",
        "get_target_drugs",
        "search_chembl_compounds_for_target",
        "summarize_target_expression_context",
        "infer_genetic_effect_direction",
        "compare_targets_multi_axis",
        "summarize_target_competitive_landscape",
        "summarize_target_safety_liabilities",
        "summarize_clinical_trials_landscape",
        "search_clinical_trials",
        "search_pubmed_advanced",
    ]
    core_fallback = ["search_pubmed", "get_pubmed_abstract", "get_clinical_trial"]
    recommended_tools = sorted(set(core_preferred + preferred))
    fallback_tools = sorted(set(core_fallback + fallback))
    return [
        WorkflowStep(
            step_id="step_1",
            title="Request framing",
            instruction=(
                "Extract target/disease entities and define success criteria "
                "for risk assessment."
            ),
            rationale="Frame assessment criteria before tool execution.",
            expected_output_fields=["target", "disease", "risk_criteria"],
        ),
        WorkflowStep(
            step_id="step_2",
            title="Evidence and risk analysis",
            instruction=(
                "Use tools to gather genetics, druggability, clinical trial, "
                "and literature evidence. Highlight contradictory evidence."
            ),
            recommended_tools=recommended_tools,
            fallback_tools=fallback_tools,
            rationale="Combine structured and literature evidence for robust assessment.",
            expected_output_fields=["selected_tools", "supporting_evidence", "contradictory_evidence", "confidence"],
        ),
        WorkflowStep(
            step_id="step_3",
            title="Decision report",
            instruction=(
                "Produce a structured recommendation with confidence and "
                "citations (PMIDs/NCT IDs)."
            ),
            rationale="Convert evidence into an auditable recommendation.",
            expected_output_fields=["recommendation", "confidence", "citations", "risks", "next_actions"],
        ),
    ]


def _plan_generic(request_type: str, intent_tags: list[str]) -> list[WorkflowStep]:
    preferred, fallback = tool_bundle_for_intent(intent_tags)
    step_3_instruction = "Deliver findings, limitations, and next actions in a structured format."
    if request_type == "comparison":
        step_3_instruction = (
            "Produce a side-by-side comparison with explicit trade-offs, "
            "confidence notes, and recommendation."
        )
    elif request_type == "prioritization":
        step_3_instruction = (
            "Produce a ranked shortlist with transparent criteria, "
            "confidence, and rationale for ordering."
        )
    elif request_type == "validation":
        step_3_instruction = (
            "Provide a validation judgment supported by both confirming and "
            "contradictory evidence."
        )
    elif request_type == "action_planning":
        step_3_instruction = (
            "Produce an action plan with sequencing, key dependencies, and "
            "clear next actions."
        )

    return [
        WorkflowStep(
            step_id="step_1",
            title="Scope extraction",
            instruction=(
                "Extract concrete entities and search terms from the user request. "
                "If critical inputs are missing, ask one concise clarification question."
            ),
            rationale="Prevent scope drift and define evaluation target up front.",
            expected_output_fields=["objective", "constraints", "success_criteria", "plan"],
        ),
        WorkflowStep(
            step_id="step_2",
            title="Evidence gathering",
            instruction=(
                "Collect the strongest available evidence with citations and "
                "use fallback tools only when primary tools are insufficient."
            ),
            recommended_tools=preferred,
            fallback_tools=fallback,
            rationale="Use intent-aligned tool bundle and record fallback behavior explicitly.",
            expected_output_fields=["selected_tools", "tool_rationale", "results", "fallback_strategy"],
        ),
        WorkflowStep(
            step_id="step_3",
            title="Direct answer",
            instruction=step_3_instruction,
            rationale="Deliver concise, structured output for actionability.",
            expected_output_fields=["findings", "evidence", "limitations", "next_actions"],
        ),
    ]


def build_plan_steps(objective: str) -> list[WorkflowStep]:
    intent_tags = infer_intent_tags(objective)
    lower = objective.lower()
    if "expert" in lower or "researcher" in lower or "investigator" in lower:
        return _plan_for_expert_discovery(intent_tags)
    if "target" in lower or "druggab" in lower or "trial" in lower:
        return _plan_for_target_assessment(intent_tags)
    return _plan_generic(classify_request_type(objective), intent_tags)


def create_task(objective: str) -> WorkflowTask:
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    request_type = classify_request_type(objective)
    task = WorkflowTask(
        task_id=task_id,
        objective=objective,
        request_type=request_type,
        intent_tags=infer_intent_tags(objective),
        success_criteria=build_success_criteria(request_type),
        status="pending",
    )
    task.steps = build_plan_steps(objective)
    task.touch()
    return task


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


def _detect_pivots(trace_entries: list[dict]) -> list[str]:
    pivots: list[str] = []
    for idx in range(len(trace_entries) - 1):
        current = trace_entries[idx]
        nxt = trace_entries[idx + 1]
        current_outcome = str(current.get("outcome", "unknown"))
        if current_outcome in {"error", "not_found_or_empty", "no_response"}:
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
        return ["- Executed tool trace: no tool calls recorded for this step."]

    lines = ["- Executed tool trace:"]
    for idx, entry in enumerate(trace_entries, start=1):
        tool_name = str(entry.get("tool_name", "unknown_tool"))
        outcome = str(entry.get("outcome", "unknown"))
        call_id = str(entry.get("call_id", "n/a"))
        phase = str(entry.get("phase", "main"))
        args_text = _compact_json(entry.get("args", {}))
        line = f"  {idx}. [{phase}] {tool_name}(call_id={call_id}, args={args_text}) -> {outcome}"
        detail = str(entry.get("detail", "")).strip()
        if detail:
            line += f" | detail: {detail}"
        lines.append(line)

    pivot_notes = _detect_pivots(trace_entries)
    if pivot_notes:
        lines.append("- Pivot behavior:")
        for note in pivot_notes:
            lines.append(f"  - {note}")
    else:
        lines.append("- Pivot behavior: none detected.")
    return lines


def _render_methodology(task: WorkflowTask) -> list[str]:
    lines = ["## Methodology"]
    if not task.steps:
        lines.append("- No workflow steps were created.")
        return lines

    for idx, step in enumerate(task.steps, start=1):
        planned_tools = sorted(set(step.recommended_tools + step.fallback_tools))
        lines.extend(
            [
                f"### Step {idx}: {step.title}",
                f"- Status: {step.status}",
                f"- Goal: {step.instruction}",
                f"- Why this order: {step.rationale or 'Not specified.'}",
                f"- Planned tools: {', '.join(planned_tools) if planned_tools else 'none'}",
                (
                    f"- Enforced step tools: {', '.join(step.allowed_tools)}"
                    if step.allowed_tools
                    else "- Enforced step tools: none (reasoning-only step)"
                ),
                f"- Information gained: {_summarize_step_output(step.output)}",
                (
                    f"- Evidence IDs from step: {', '.join(step.evidence_refs)}"
                    if step.evidence_refs
                    else "- Evidence IDs from step: none detected"
                ),
            ]
        )
        lines.extend(_render_tool_trace(step.tool_trace))
        lines.append("")
    return lines


def render_final_report(task: WorkflowTask, quality_report: dict | None = None) -> str:
    all_refs: list[str] = []
    for step in task.steps:
        all_refs.extend(step.evidence_refs)
    unique_refs = sorted(set(all_refs))

    answer_text = task.steps[-1].output if task.steps else "No answer generated."
    lines = ["## Answer", answer_text, ""]
    lines.extend(_render_methodology(task))
    lines.extend(["", "## Evidence"])
    if unique_refs:
        for ref in unique_refs:
            lines.append(f"- {ref}")
    else:
        lines.append("- No explicit citation IDs detected in model output.")

    lines.extend(
        [
            "",
            "## Diagnostics",
            f"- Request type: {task.request_type}",
            f"- Intent tags: {', '.join(task.intent_tags) if task.intent_tags else 'none'}",
            f"- Steps completed: {sum(1 for s in task.steps if s.status == 'completed')}/{len(task.steps)}",
        ]
    )
    if quality_report:
        lines.append(f"- Quality gate passed: {'yes' if quality_report.get('passed') else 'no'}")
        lines.append(f"- Evidence refs detected: {quality_report.get('evidence_count', 0)}")
        lines.append(f"- Tool calls captured: {quality_report.get('tool_call_count', 0)}")
        gaps = quality_report.get("unresolved_gaps", [])
        if gaps:
            lines.append("- Unresolved gaps:")
            lines.extend([f"  - {gap}" for gap in gaps])
        else:
            lines.append("- Unresolved gaps: none")
    return "\n".join(lines)


def step_prompt(task: WorkflowTask, step: WorkflowStep) -> str:
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
        "Use source citations whenever possible (PMID, NCT IDs).\n"
    )
