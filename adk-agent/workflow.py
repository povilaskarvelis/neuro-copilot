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


VALID_REQUEST_TYPES = {
    "comparison",
    "prioritization",
    "validation",
    "action_planning",
    "exploration",
}

VALID_INTENT_TAGS = {
    "researcher_discovery",
    "evidence_landscape",
    "variant_check",
    "pathway_context",
    "clinical_landscape",
    "chemistry_evidence",
    "ontology_expansion",
    "expression_context",
    "genetics_direction",
    "safety_assessment",
    "competitive_landscape",
    "comparison",
    "prioritization",
    "target_comparison",
}


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
    reasoning_summary: str = ""
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

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
            "reasoning_summary": self.reasoning_summary,
            "actions": self.actions,
            "observations": self.observations,
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
            reasoning_summary=payload.get("reasoning_summary", ""),
            actions=list(payload.get("actions", [])),
            observations=list(payload.get("observations", [])),
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
    fallback_recovery_notes: str = ""
    fallback_tool_trace: list[dict] = field(default_factory=list)
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
            "fallback_recovery_notes": self.fallback_recovery_notes,
            "fallback_tool_trace": self.fallback_tool_trace,
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
            fallback_recovery_notes=payload.get("fallback_recovery_notes", ""),
            fallback_tool_trace=list(payload.get("fallback_tool_trace", [])),
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


def sanitize_request_type(request_type: str | None) -> str | None:
    if not request_type:
        return None
    normalized = request_type.strip().lower()
    return normalized if normalized in VALID_REQUEST_TYPES else None


def sanitize_intent_tags(intent_tags: list[str] | str | None) -> list[str]:
    if not intent_tags:
        return []
    if isinstance(intent_tags, str):
        intent_tags = [item.strip() for item in re.split(r"[,\n;]+", intent_tags) if item.strip()]
    cleaned: set[str] = set()
    for tag in intent_tags:
        normalized = str(tag).strip().lower()
        if normalized in VALID_INTENT_TAGS:
            cleaned.add(normalized)
    return sorted(cleaned)


def _mentions_researcher_role(text: str) -> bool:
    patterns = (
        r"\bresearch(?:er|ers)?\b",
        r"\breseach(?:er|ers)?\b",
        r"\breseacher(?:s)?\b",
        r"\breseachers?\b",
        r"\bauthor(?:s)?\b",
        r"\bexpert(?:s)?\b",
        r"\binvestigator(?:s)?\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _mentions_target_context(text: str) -> bool:
    return any(
        token in text
        for token in [
            "target",
            "gene",
            "protein",
            "druggab",
            "candidate",
            "mechanism",
            "pathway",
        ]
    )


def infer_intent_tags(objective: str) -> list[str]:
    lower = objective.lower()
    tags: list[str] = []
    researcher_role_query = _mentions_researcher_role(lower)
    target_context_query = _mentions_target_context(lower)
    if researcher_role_query or "contact" in lower:
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
        if target_context_query:
            tags.append("target_comparison")
    if any(token in lower for token in ["rank", "prioritize", "top"]):
        tags.append("prioritization")
        if target_context_query and not researcher_role_query:
            tags.append("target_comparison")
    if not tags:
        tags.append("evidence_landscape")
    return sorted(set(tags))


def tool_bundle_for_intent(intent_tags: list[str]) -> tuple[list[str], list[str]]:
    preferred: list[str] = []
    fallback: list[str] = []
    if "researcher_discovery" in intent_tags:
        preferred.extend(
            [
                "rank_researchers_by_activity",
                "search_openalex_works",
                "search_openalex_authors",
                "search_pubmed_advanced",
            ]
        )
        fallback.extend(
            [
                "get_pubmed_author_profile",
                "get_pubmed_paper_details",
                "search_openalex_works",
            ]
        )
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
            expected_output_fields=["objective", "constraints", "success_criteria", "decomposition_subtasks"],
        ),
        WorkflowStep(
            step_id="step_2",
            title="Evidence collection",
            instruction=(
                "Collect quantitative evidence for researcher prominence using topic-specific ranking tools. "
                "Prioritize activity scores, topic-matched publication volume/citations, and stable affiliation signals."
            ),
            recommended_tools=preferred,
            fallback_tools=fallback,
            rationale=(
                "Use topic-based researcher ranking first; "
                "fallback to publication-derived author profiling only when ranking endpoints are blocked."
            ),
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
            expected_output_fields=["target", "disease", "risk_criteria", "decomposition_subtasks"],
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
            expected_output_fields=["objective", "constraints", "success_criteria", "decomposition_subtasks"],
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


def build_plan_steps(
    objective: str,
    *,
    intent_tags_override: list[str] | None = None,
    request_type_override: str | None = None,
) -> list[WorkflowStep]:
    intent_tags = (
        sanitize_intent_tags(intent_tags_override)
        if intent_tags_override is not None
        else infer_intent_tags(objective)
    )
    if not intent_tags:
        intent_tags = infer_intent_tags(objective)
    request_type = sanitize_request_type(request_type_override) or classify_request_type(objective)
    lower = objective.lower()
    if "researcher_discovery" in intent_tags or _mentions_researcher_role(lower):
        return _plan_for_expert_discovery(intent_tags)
    target_assessment_tags = {
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
    if any(tag in target_assessment_tags for tag in intent_tags) or "target" in lower or "druggab" in lower or "trial" in lower:
        return _plan_for_target_assessment(intent_tags)
    return _plan_generic(request_type, intent_tags)


def create_task(
    objective: str,
    *,
    request_type_override: str | None = None,
    intent_tags_override: list[str] | None = None,
) -> WorkflowTask:
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    request_type = sanitize_request_type(request_type_override) or classify_request_type(objective)
    intent_tags = (
        sanitize_intent_tags(intent_tags_override)
        if intent_tags_override is not None
        else infer_intent_tags(objective)
    )
    if not intent_tags:
        intent_tags = infer_intent_tags(objective)
    task = WorkflowTask(
        task_id=task_id,
        objective=objective,
        request_type=request_type,
        intent_tags=intent_tags,
        success_criteria=build_success_criteria(request_type),
        status="pending",
    )
    task.steps = build_plan_steps(
        objective,
        intent_tags_override=intent_tags,
        request_type_override=request_type,
    )
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


def _extract_revision_directive(objective: str) -> str | None:
    match = re.search(r"User revision to scope/decomposition:\s*(.+)", objective or "", flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


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
    for step in task.steps:
        for entry in step.tool_trace:
            tool = str(entry.get("tool_name", ""))
            if "openalex" in tool or tool == "rank_researchers_by_activity":
                families.add("OpenAlex")
            elif "pubmed" in tool:
                families.add("PubMed")
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
        entry = re.sub(r"\s+", " ", bullet_match.group(1).strip())
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
        return ["- Tool activity summary: no tool calls were recorded for this step."]

    outcome_counts: dict[str, int] = {}
    tool_counts: dict[str, int] = {}
    notable_errors: list[str] = []
    for entry in trace_entries:
        outcome = str(entry.get("outcome", "unknown"))
        tool_name = str(entry.get("tool_name", "unknown_tool"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
        if outcome in {"error", "not_found_or_empty", "no_response"} and len(notable_errors) < 3:
            detail = _summarize_step_output(str(entry.get("detail", "")), max_chars=100)
            detail_text = f" ({detail})" if detail and detail != "No output captured." else ""
            notable_errors.append(f"{tool_name}{detail_text}")

    ok_count = outcome_counts.get("ok", 0)
    issue_count = (
        outcome_counts.get("error", 0)
        + outcome_counts.get("not_found_or_empty", 0)
        + outcome_counts.get("no_response", 0)
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
        elif step.output:
            observation_points.append(_summarize_step_output(step.output))
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


def render_final_report(task: WorkflowTask, quality_report: dict | None = None) -> str:
    raw_answer_text = task.steps[-1].output if task.steps else "No answer generated."
    answer_text = raw_answer_text.strip() if raw_answer_text else "No answer generated."
    fallback_text = task.fallback_recovery_notes.strip() or None
    if not fallback_text:
        # Backward compatibility for legacy tasks where fallback notes were appended to step output.
        answer_text, fallback_text = _split_answer_and_fallback(raw_answer_text)
    lines = _render_scope_snapshot(task)
    lines.extend([""])
    lines.extend(_render_decomposition(task))
    lines.extend(["", "## Answer", answer_text, ""])
    lines.extend(_render_methodology(task))
    if fallback_text:
        lines.extend(["## Fallback Recovery Notes"])
        lines.extend(_format_fallback_notes(fallback_text))
        if task.fallback_tool_trace:
            lines.extend(["", "### Tool Activity"])
            lines.extend(_render_tool_trace(task.fallback_tool_trace))
        lines.append("")

    lines.extend(
        [
            "",
            "## Diagnostics",
            f"- Request type: {task.request_type}",
            f"- Intent tags: {', '.join(task.intent_tags) if task.intent_tags else 'none'}",
            f"- Workflow status: {task.status}",
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
        revision = _extract_revision_directive(task.objective)
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
        f"{step_freshness_guardrail}"
        f"{researcher_ranking_guardrail}"
    )
