"""
Planning and revision logic extracted from workflow.py.

This module is intentionally behavior-compatible with legacy workflow exports.
"""
from __future__ import annotations

import re
import uuid

from co_scientist.domain.models import (
    PlanDelta,
    PlanVersion,
    RevisionIntent,
    WorkflowStep,
    WorkflowTask,
    _utc_now,
    generate_chat_title,
)

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

RECOGNIZED_TOOL_NAMES = {
    "search_diseases",
    "search_disease_targets",
    "get_target_info",
    "search_targets",
    "check_druggability",
    "get_target_drugs",
    "search_clinical_trials",
    "get_clinical_trial",
    "summarize_clinical_trials_landscape",
    "search_chembl_compounds_for_target",
    "search_pubmed",
    "get_pubmed_abstract",
    "search_pubmed_advanced",
    "get_pubmed_paper_details",
    "get_pubmed_author_profile",
    "search_openalex_works",
    "search_openalex_authors",
    "rank_researchers_by_activity",
    "get_researcher_contact_candidates",
    "get_gene_info",
    "search_clinvar_variants",
    "get_clinvar_variant_details",
    "search_gwas_associations",
    "search_reactome_pathways",
    "get_string_interactions",
    "expand_disease_context",
    "summarize_target_expression_context",
    "infer_genetic_effect_direction",
    "summarize_target_competitive_landscape",
    "summarize_target_safety_liabilities",
    "compare_targets_multi_axis",
    "list_local_datasets",
    "read_local_dataset",
}

TOOL_CAPABILITIES: dict[str, set[str]] = {
    "search_diseases": {"disease_context"},
    "search_disease_targets": {"disease_context", "target_mapping"},
    "get_target_info": {"target_biology"},
    "search_targets": {"target_mapping"},
    "check_druggability": {"druggability", "chemistry"},
    "get_target_drugs": {"druggability", "clinical"},
    "search_clinical_trials": {"clinical"},
    "get_clinical_trial": {"clinical"},
    "summarize_clinical_trials_landscape": {"clinical", "competitive"},
    "search_chembl_compounds_for_target": {"chemistry", "druggability"},
    "search_pubmed": {"literature"},
    "get_pubmed_abstract": {"literature"},
    "search_pubmed_advanced": {"literature"},
    "get_pubmed_paper_details": {"literature"},
    "get_pubmed_author_profile": {"literature", "researcher"},
    "search_openalex_works": {"literature", "researcher"},
    "search_openalex_authors": {"researcher"},
    "rank_researchers_by_activity": {"researcher"},
    "get_researcher_contact_candidates": {"researcher"},
    "get_gene_info": {"genetics"},
    "search_clinvar_variants": {"genetics", "variants"},
    "get_clinvar_variant_details": {"genetics", "variants"},
    "search_gwas_associations": {"genetics", "gwas", "variants"},
    "search_reactome_pathways": {"pathways"},
    "get_string_interactions": {"pathways"},
    "expand_disease_context": {"disease_context"},
    "summarize_target_expression_context": {"expression", "target_biology"},
    "infer_genetic_effect_direction": {"genetics", "gwas", "directionality"},
    "summarize_target_competitive_landscape": {"competitive", "clinical"},
    "summarize_target_safety_liabilities": {"safety"},
    "compare_targets_multi_axis": {"comparison", "synthesis"},
    "list_local_datasets": {"local_data"},
    "read_local_dataset": {"local_data"},
}

CAPABILITY_PATTERNS: list[tuple[str, str]] = [
    ("gwas", r"\bgwas\b|genome[-\s]*wide association|\bsnp\b|\blocus\b"),
    ("genetics", r"\bgenetic|\bgenomic|\bvariant|\bmutation|\bheritable"),
    ("directionality", r"direction[-\s]*of[-\s]*effect|risk[-\s]*increasing|protective"),
    ("safety", r"\bsafety\b|\btoxicity\b|adverse|liabilit|risk signal"),
    ("clinical", r"\bclinical trial|\bphase\s*[1234]\b|\bpatient\b|clinicaltrials\.gov"),
    ("literature", r"\bpubmed\b|\bliterature\b|\bpaper|\bpublication|\bcitation"),
    ("druggability", r"\bdruggability\b|\btractability\b|\bmodality\b"),
    ("chemistry", r"\bchembl\b|\bcompound\b|chemical matter|potency|ic50"),
    ("competitive", r"\bcompetitive\b|\blandscape\b|\bpipeline\b|\bprogram\b"),
    ("expression", r"\bexpression\b|\btissue\b|\bcell type\b|\banatom"),
    ("pathways", r"\bpathway\b|\breactome\b|\binteraction\b|\bstring\b"),
    ("researcher", r"\bresearcher\b|\bauthor\b|\baffiliation\b|\binvestigator\b"),
    ("local_data", r"\blocal dataset\b|\binternal data\b|\bcsv\b|\bfile\b"),
]


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
            title="Scope and weighted criteria",
            instruction=(
                "Lock the decision question, comparison targets, disease context, and weighting rules. "
                "Convert user priorities into explicit decision criteria and decomposition subtasks."
            ),
            rationale="Frame assessment criteria before tool execution.",
            expected_output_fields=["targets", "disease", "weighted_criteria", "decomposition_subtasks"],
        ),
        WorkflowStep(
            step_id="step_2",
            title="Human genetics evidence",
            instruction=(
                "Collect and compare genetics evidence for both targets, including association strength and "
                "direction-of-effect when available. If directionality tools fail, run fallback genetics paths "
                "and record the exact residual uncertainty."
            ),
            recommended_tools=[
                "infer_genetic_effect_direction",
                "search_gwas_associations",
                "compare_targets_multi_axis",
                "search_pubmed_advanced",
            ],
            fallback_tools=fallback_tools,
            rationale="Human genetics is a high-weight decision axis and must be assessed first.",
            expected_output_fields=["selected_tools", "genetics_findings", "directionality_gaps", "confidence"],
        ),
        WorkflowStep(
            step_id="step_3",
            title="Safety liabilities and risk signals",
            instruction=(
                "Assess safety liabilities and translational risk signals for each target. "
                "Use safety summaries plus trial termination/adverse-signal context and note where evidence is sparse."
            ),
            recommended_tools=[
                "summarize_target_safety_liabilities",
                "summarize_clinical_trials_landscape",
                "search_clinical_trials",
                "search_pubmed_advanced",
            ],
            fallback_tools=fallback_tools,
            rationale="Safety is a high-weight criterion and should be evaluated as a dedicated step.",
            expected_output_fields=["selected_tools", "safety_findings", "risk_signals", "limitations"],
        ),
        WorkflowStep(
            step_id="step_4",
            title="Druggability and development landscape",
            instruction=(
                "Evaluate modality tractability, available chemical matter, current drug programs, and competitive "
                "development stage for each target."
            ),
            recommended_tools=[
                "check_druggability",
                "get_target_drugs",
                "search_chembl_compounds_for_target",
                "summarize_target_competitive_landscape",
            ],
            fallback_tools=fallback_tools,
            rationale="Translate biology into practical program feasibility and competitive context.",
            expected_output_fields=["selected_tools", "druggability_findings", "landscape_findings", "constraints"],
        ),
        WorkflowStep(
            step_id="step_5",
            title="Weighted comparison and trade-offs",
            instruction=(
                "Synthesize the evidence across criteria using the requested weighting. "
                "Resolve contradictions, call out unresolved gaps, and produce a clear winner/loser rationale."
            ),
            recommended_tools=["compare_targets_multi_axis", "search_pubmed_advanced"],
            fallback_tools=fallback_tools,
            rationale="Create an auditable comparison before final recommendation.",
            expected_output_fields=["comparison_matrix", "tradeoffs", "unresolved_gaps", "provisional_conclusion"],
        ),
        WorkflowStep(
            step_id="step_6",
            title="Decision report",
            instruction=(
                "Produce a concise report with recommendation first, then rationale narrative, methodology, "
                "and exactly three next actions. Include explicit reasons for deprioritizing the loser."
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
        steps = _plan_for_expert_discovery(intent_tags)
        return _apply_revision_plan_overrides(steps, objective)

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
        steps = _plan_for_target_assessment(intent_tags)
    else:
        steps = _plan_generic(request_type, intent_tags)
    return _apply_revision_plan_overrides(steps, objective)


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
        title=generate_chat_title(objective),
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


def _extract_revision_directive(objective: str) -> str | None:
    match = re.search(r"User revision to scope/decomposition:\s*(.+)", objective or "", flags=re.IGNORECASE)
    return match.group(1).strip() if match else None


def _extract_revision_directives(objective: str) -> list[str]:
    directives: list[str] = []
    block_match = re.search(
        r"Revision directives to apply:\s*((?:\n-\s+.+)+)",
        objective or "",
        flags=re.IGNORECASE,
    )
    if block_match:
        for line in block_match.group(1).splitlines():
            stripped = line.strip()
            if not stripped.startswith("-"):
                continue
            directive = re.sub(r"\s+", " ", stripped[1:].strip()).strip(" .")
            if directive and directive not in directives:
                directives.append(directive)
    if directives:
        return directives[:8]

    revision = _extract_revision_directive(objective)
    if not revision:
        return []
    fallback = re.sub(r"\s+", " ", revision).strip(" .")
    return [fallback] if fallback else []


def _extract_revision_tool_hints(objective: str) -> list[str]:
    revision = _extract_revision_directive(objective) or ""
    directives = _extract_revision_directives(objective)
    combined = "\n".join([revision, *directives]).lower()
    if not combined.strip():
        return []

    candidates = re.findall(r"\b([a-z][a-z0-9_]{2,})\b", combined)
    tool_hints: list[str] = []
    for candidate in candidates:
        if candidate not in RECOGNIZED_TOOL_NAMES:
            continue
        if candidate in tool_hints:
            continue
        tool_hints.append(candidate)

    requested_capabilities: set[str] = set()
    for capability, pattern in CAPABILITY_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            requested_capabilities.add(capability)

    if requested_capabilities:
        scored: list[tuple[int, str]] = []
        for tool_name in sorted(RECOGNIZED_TOOL_NAMES):
            caps = TOOL_CAPABILITIES.get(tool_name, set())
            overlap = requested_capabilities.intersection(caps)
            if not overlap:
                continue
            # Higher score means better semantic match for user-requested evidence/workflow preference.
            score = len(overlap)
            if "gwas" in requested_capabilities and "gwas" in caps:
                score += 2
            if "safety" in requested_capabilities and "safety" in caps:
                score += 2
            if "clinical" in requested_capabilities and "clinical" in caps:
                score += 1
            scored.append((score, tool_name))
        scored.sort(key=lambda item: (-item[0], item[1]))
        for _, tool_name in scored:
            if tool_name not in tool_hints:
                tool_hints.append(tool_name)
    return tool_hints[:10]


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        output.append(cleaned)
        seen.add(cleaned)
    return output


def _apply_revision_plan_overrides(steps: list[WorkflowStep], objective: str) -> list[WorkflowStep]:
    revision_directives = _extract_revision_directives(objective)
    if not revision_directives:
        return steps

    tool_hints = _extract_revision_tool_hints(objective)
    for idx, step in enumerate(steps):
        if idx == 0:
            step.instruction += (
                " Update decomposition_subtasks to reflect all revision directives and any changed execution path."
            )
            if tool_hints:
                step.instruction += (
                    " If the user specified tool preferences, map them into sub-task sequencing: "
                    f"{', '.join(tool_hints)}."
                )
            continue

        if step.recommended_tools or step.fallback_tools:
            relevant_hints: list[str] = []
            if tool_hints:
                existing_tools = set(step.recommended_tools + step.fallback_tools)
                relevant_hints = [hint for hint in tool_hints if hint in existing_tools]
                if not relevant_hints and idx == 1:
                    relevant_hints = list(tool_hints)

                if relevant_hints:
                    step.recommended_tools = _unique_preserve_order(relevant_hints + step.recommended_tools)
                    prefix = f"Prioritize user-requested tools: {', '.join(relevant_hints)}. "
                    if not step.instruction.lower().startswith("prioritize user-requested tools:"):
                        step.instruction = prefix + step.instruction
                    step.instruction += (
                        " Respect user tool preferences when feasible and prioritize: "
                        f"{', '.join(relevant_hints)}."
                    )
            step.instruction += (
                " Execute this step under the latest revision directives; if any directive cannot be fully met, "
                "state the limitation and fallback explicitly."
            )
        else:
            step.instruction += (
                " Ensure the final synthesis explicitly reflects the latest revision directives and unresolved items."
            )
    return steps


def clone_step(step: WorkflowStep) -> WorkflowStep:
    return WorkflowStep.from_dict(step.to_dict())


def _step_signature(step: WorkflowStep) -> tuple[str, str]:
    title = re.sub(r"\s+", " ", (step.title or "").strip().lower())
    instruction = re.sub(r"\s+", " ", (step.instruction or "").strip().lower())
    return title, instruction


def build_plan_delta(
    previous_steps: list[WorkflowStep],
    next_steps: list[WorkflowStep],
    *,
    from_version_id: str | None,
    to_version_id: str,
) -> PlanDelta:
    prev_titles = [step.title.strip() for step in previous_steps if step.title.strip()]
    next_titles = [step.title.strip() for step in next_steps if step.title.strip()]
    prev_signatures = [_step_signature(step) for step in previous_steps]
    next_signatures = [_step_signature(step) for step in next_steps]

    added_steps = [title for title in next_titles if title not in prev_titles]
    removed_steps = [title for title in prev_titles if title not in next_titles]

    modified_steps: list[str] = []
    reordered_steps: list[str] = []
    shared = [title for title in next_titles if title in prev_titles]
    for title in shared:
        prev_idx = prev_titles.index(title)
        next_idx = next_titles.index(title)
        if prev_idx != next_idx:
            reordered_steps.append(title)
            continue
        if prev_signatures[prev_idx] != next_signatures[next_idx]:
            modified_steps.append(title)

    summary_parts: list[str] = []
    if added_steps:
        summary_parts.append(f"added {len(added_steps)} step(s)")
    if removed_steps:
        summary_parts.append(f"removed {len(removed_steps)} step(s)")
    if modified_steps:
        summary_parts.append(f"modified {len(modified_steps)} step(s)")
    if reordered_steps:
        summary_parts.append(f"reordered {len(reordered_steps)} step(s)")
    if not summary_parts:
        summary_parts.append("no structural changes")

    return PlanDelta(
        from_version_id=from_version_id,
        to_version_id=to_version_id,
        added_steps=added_steps,
        removed_steps=removed_steps,
        modified_steps=modified_steps,
        reordered_steps=reordered_steps,
        summary=", ".join(summary_parts),
    )


def active_plan_version(task: WorkflowTask) -> PlanVersion | None:
    active_id = str(task.active_plan_version_id or "").strip()
    if not active_id:
        return None
    for version in task.plan_versions:
        if version.version_id == active_id:
            return version
    return None


def _normalize_remaining_step(step: WorkflowStep, step_number: int) -> WorkflowStep:
    normalized = clone_step(step)
    normalized.step_id = f"step_{step_number}"
    normalized.status = "pending"
    normalized.output = ""
    normalized.evidence_refs = []
    normalized.allowed_tools = []
    normalized.tool_trace = []
    normalized.reasoning_summary = ""
    normalized.actions = []
    normalized.observations = []
    return normalized


def register_plan_version(
    task: WorkflowTask,
    *,
    base_from_step_index: int,
    request_type: str,
    intent_tags: list[str],
    revision_intent: RevisionIntent | None,
    steps: list[WorkflowStep],
    gate_reason: str,
    from_version_id: str | None,
    previous_steps: list[WorkflowStep] | None = None,
) -> PlanVersion:
    version = PlanVersion(
        version_id=f"plan_{uuid.uuid4().hex[:10]}",
        created_at=_utc_now(),
        base_from_step_index=base_from_step_index,
        request_type=request_type,
        intent_tags=list(intent_tags),
        revision_intent=revision_intent,
        steps=[clone_step(step) for step in steps],
        gate_reason=gate_reason,
    )
    task.plan_versions.append(version)
    task.active_plan_version_id = version.version_id
    task.latest_plan_delta = build_plan_delta(
        previous_steps=[clone_step(step) for step in (previous_steps or [])],
        next_steps=version.steps,
        from_version_id=from_version_id,
        to_version_id=version.version_id,
    )
    return version


def initialize_plan_version(task: WorkflowTask, gate_reason: str) -> PlanVersion:
    base_from_step_index = max(0, task.current_step_index + 1)
    remaining = [clone_step(step) for step in task.steps[base_from_step_index:]]
    prev = active_plan_version(task)
    prev_id = prev.version_id if prev else None
    return register_plan_version(
        task,
        base_from_step_index=base_from_step_index,
        request_type=task.request_type,
        intent_tags=task.intent_tags,
        revision_intent=None,
        steps=remaining,
        gate_reason=gate_reason,
        from_version_id=prev_id,
        previous_steps=prev.steps if prev else [],
    )


def replan_remaining_steps(
    task: WorkflowTask,
    *,
    revised_objective: str,
    request_type: str,
    intent_tags: list[str],
    revision_intent: RevisionIntent | None,
    gate_reason: str,
) -> tuple[PlanVersion, PlanDelta]:
    base_from_step_index = max(0, task.current_step_index + 1)
    existing_remaining = [clone_step(step) for step in task.steps[base_from_step_index:]]
    previous = active_plan_version(task)
    previous_id = previous.version_id if previous else None

    new_full_plan = build_plan_steps(
        revised_objective,
        intent_tags_override=intent_tags,
        request_type_override=request_type,
    )
    new_remaining_raw = new_full_plan[base_from_step_index:] if base_from_step_index < len(new_full_plan) else []
    new_remaining = [
        _normalize_remaining_step(step, base_from_step_index + idx + 1)
        for idx, step in enumerate(new_remaining_raw)
    ]

    frozen_steps = [clone_step(step) for step in task.steps[:base_from_step_index]]
    task.steps = frozen_steps + new_remaining
    task.objective = revised_objective
    task.request_type = request_type
    task.intent_tags = list(intent_tags)
    task.checkpoint_state = "open"
    task.checkpoint_reason = gate_reason

    version = PlanVersion(
        version_id=f"plan_{uuid.uuid4().hex[:10]}",
        created_at=_utc_now(),
        base_from_step_index=base_from_step_index,
        request_type=request_type,
        intent_tags=list(intent_tags),
        revision_intent=revision_intent,
        steps=[clone_step(step) for step in new_remaining],
        gate_reason=gate_reason,
    )
    delta = build_plan_delta(
        existing_remaining,
        version.steps,
        from_version_id=previous_id,
        to_version_id=version.version_id,
    )
    task.plan_versions.append(version)
    task.active_plan_version_id = version.version_id
    task.latest_plan_delta = delta
    return version, delta
