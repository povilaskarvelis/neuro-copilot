"""
Planning and revision logic extracted from workflow.py.

This module is intentionally behavior-compatible with legacy workflow exports.
"""
from __future__ import annotations

import re
import uuid

from co_scientist.runtime.event_orchestrator import ensure_phase_state
from co_scientist.runtime.tool_registry import infer_capabilities_from_text
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

INTENT_CAPABILITY_HINTS: dict[str, set[str]] = {
    "researcher_discovery": {"researcher", "literature"},
    "evidence_landscape": {"literature"},
    "variant_check": {"genetics", "variants", "gwas"},
    "pathway_context": {"pathways"},
    "clinical_landscape": {"clinical"},
    "chemistry_evidence": {"chemistry", "druggability"},
    "ontology_expansion": {"disease_context"},
    "expression_context": {"expression", "target_biology"},
    "genetics_direction": {"genetics", "gwas", "directionality"},
    "safety_assessment": {"safety", "clinical"},
    "competitive_landscape": {"competitive", "clinical"},
    "target_comparison": {"comparison", "synthesis"},
}

def classify_request_type(objective: str) -> str:
    lower = objective.lower()
    if any(token in lower for token in ["compare", "versus", "vs", "tradeoff", "trade-off", "difference"]):
        return "comparison"
    if any(token in lower for token in ["rank", "prioritize", "top", "best", "shortlist", "triage"]):
        return "prioritization"
    if any(token in lower for token in ["validate", "verify", "check", "plausible", "hypothesis", "stress-test"]):
        return "validation"
    if any(token in lower for token in ["plan", "roadmap", "next steps", "how should", "what should", "execution plan"]):
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
    lower_text = str(text or "").lower()
    if any(
        token in lower_text
        for token in [
            "target",
            "gene",
            "protein",
            "druggab",
            "candidate",
            "mechanism",
            "pathway",
        ]
    ):
        return True

    # Heuristic: in comparison phrasing, multiple uppercase gene-like tokens usually imply target context.
    comparison_markers = (" vs ", " versus ", "compare ", "between ")
    if any(marker in lower_text for marker in comparison_markers):
        tokens = re.findall(r"\b[A-Za-z0-9-]{3,12}\b", text)
        gene_like = [
            token
            for token in tokens
            if token.upper() == token
            and any(ch.isalpha() for ch in token)
            and not token.startswith(("NCT", "PMID", "ENSG", "MONDO_", "EFO_"))
        ]
        if len(gene_like) >= 2:
            return True
    return False


def infer_intent_tags(objective: str) -> list[str]:
    lower = objective.lower()
    tags: list[str] = []
    researcher_role_query = _mentions_researcher_role(lower)
    target_context_query = _mentions_target_context(objective)
    if researcher_role_query or "contact" in lower:
        tags.append("researcher_discovery")
    if any(
        token in lower
        for token in [
            "evidence",
            "landscape",
            "literature",
            "paper",
            "publication",
            "systematic review",
            "meta-analysis",
            "benchmark",
        ]
    ):
        tags.append("evidence_landscape")
    if any(token in lower for token in ["variant", "clinvar", "pathogenic", "gwas", "mutation", "allele", "polymorphism"]):
        tags.append("variant_check")
    if any(token in lower for token in ["pathway", "network", "interaction", "reactome", "string", "mechanism", "signaling"]):
        tags.append("pathway_context")
    if any(
        token in lower
        for token in ["trial", "clinical", "nct", "phase", "terminated", "recruiting", "real-world", "rwe"]
    ):
        tags.append("clinical_landscape")
    if any(
        token in lower
        for token in [
            "compound",
            "chembl",
            "potency",
            "ic50",
            "ki",
            "ec50",
            "small molecule",
            "chemical matter",
            "selectivity",
            "binding affinity",
        ]
    ):
        tags.append("chemistry_evidence")
    if any(token in lower for token in ["ontology", "synonym", "disease context", "efo", "mondo"]):
        tags.append("ontology_expansion")
    if any(
        token in lower
        for token in ["expression", "tissue", "cell type", "single-cell", "cell context", "specificity", "atlas", "spatial"]
    ):
        tags.append("expression_context")
    if any(token in lower for token in ["direction", "risk allele", "protective", "causal", "direction-of-effect", "doe"]):
        tags.append("genetics_direction")
    if any(
        token in lower
        for token in ["safety", "liability", "toxicity", "adverse", "off-target", "off target", "on-target", "tolerability"]
    ):
        tags.append("safety_assessment")
    if any(token in lower for token in ["competitive", "competition", "crowded", "white space", "novelty", "landscape"]):
        tags.append("competitive_landscape")
    if any(token in lower for token in ["compare", "versus", "vs"]):
        tags.append("comparison")
        if target_context_query:
            tags.append("target_comparison")
    if any(token in lower for token in ["rank", "prioritize", "top", "shortlist", "triage"]):
        tags.append("prioritization")
        if target_context_query and not researcher_role_query:
            tags.append("target_comparison")
    if not tags:
        tags.append("evidence_landscape")
    return sorted(set(tags))


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


def _is_target_focused_query(objective: str, intent_tags: list[str]) -> bool:
    if "target_comparison" in intent_tags:
        return True
    if re.search(r"\bENSG\d{11}\b", objective or "", flags=re.IGNORECASE):
        return True
    return _mentions_target_context(objective)


def _infer_capability_needs(objective: str, intent_tags: list[str], request_type: str) -> list[str]:
    lower = str(objective or "").lower()
    required: set[str] = set()
    for tag in intent_tags:
        required.update(INTENT_CAPABILITY_HINTS.get(tag, set()))

    for capability, pattern in CAPABILITY_PATTERNS:
        if re.search(pattern, lower, flags=re.IGNORECASE):
            required.add(capability)

    target_focused = _is_target_focused_query(objective, intent_tags)
    if target_focused:
        # Ensure target-focused requests trigger target-grounded evidence stages,
        # not only abstract synthesis/comparison stages.
        required.add("target_mapping")
        if request_type in {"comparison", "prioritization", "validation"}:
            required.update({"genetics", "clinical", "druggability"})
        if request_type in {"comparison", "prioritization"}:
            required.add("safety")

    if request_type in {"comparison", "prioritization"} and target_focused:
        required.update({"comparison", "synthesis"})
    elif request_type in {"comparison", "prioritization"}:
        required.add("synthesis")

    if request_type in {"validation", "action_planning"}:
        required.add("synthesis")

    if not required:
        required.add("literature")
    return sorted(required)


def _build_scope_step() -> WorkflowStep:
    return WorkflowStep(
        step_id="step_1",
        title="Scope and decomposition",
        instruction=(
            "Extract concrete entities, constraints, and success criteria from the request. "
            "Define decomposition subtasks that can be executed with available evidence tools."
        ),
        rationale="Explicit scoping prevents retrieval drift and improves traceable synthesis quality.",
        expected_output_fields=["objective", "constraints", "success_criteria", "decomposition_subtasks"],
    )


def _build_final_stage(request_type: str) -> WorkflowStep:
    final_instruction = (
        "Produce a concise final report with recommendation first, followed by rationale, methodology, limitations, and next actions."
    )
    if request_type == "comparison":
        final_instruction += " Include explicit winner/loser rationale and unresolved trade-offs."
    elif request_type == "prioritization":
        final_instruction += " Include rank order, scoring rationale, and confidence by item."
    elif request_type == "validation":
        final_instruction += " Include both supporting and contradictory evidence in the judgment."
    elif request_type == "action_planning":
        final_instruction += " Include sequenced next actions with dependencies and readiness gates."

    return WorkflowStep(
        step_id="",
        title="Decision report",
        instruction=final_instruction,
        rationale="Convert multi-stage evidence into an auditable, decision-ready output.",
        expected_output_fields=["recommendation", "confidence", "citations", "limitations", "next_actions"],
    )


def _registry_summary_tools_for_capabilities(
    tool_registry_summary: list[dict] | None,
    capabilities: set[str],
    *,
    k: int = 8,
) -> tuple[list[str], list[str]]:
    if not tool_registry_summary:
        return [], []
    ranked: list[str] = []
    fallback: list[str] = []
    caps = {str(item).strip() for item in capabilities if str(item).strip()}
    for item in tool_registry_summary:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        tool_caps = {str(c).strip() for c in item.get("capabilities", []) if str(c).strip()}
        if caps and tool_caps.intersection(caps):
            if name not in ranked:
                ranked.append(name)
        else:
            if name not in fallback:
                fallback.append(name)
    if not ranked:
        ranked = fallback[:k]
        fallback = fallback[k:]
    return ranked[:k], fallback[:k]


def _capability_map_from_registry_summary(tool_registry_summary: list[dict] | None) -> dict[str, set[str]]:
    capability_map: dict[str, set[str]] = {}
    for item in tool_registry_summary or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        caps = {str(cap).strip() for cap in item.get("capabilities", []) if str(cap).strip()}
        capability_map[name] = caps
    return capability_map


def _tool_has_capability(
    tool_name: str,
    capability: str,
    *,
    tool_capability_map: dict[str, set[str]] | None = None,
) -> bool:
    caps = (tool_capability_map or {}).get(tool_name)
    if caps:
        return capability in caps
    inferred = infer_capabilities_from_text(tool_name)
    return capability in inferred


def _split_objective_subgoals(objective: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(objective or "")).strip()
    if not text:
        return []
    parts = [p.strip(" .") for p in re.split(r"[.;]\s+|\bthen\b|\band then\b", text, flags=re.IGNORECASE)]
    unique: list[str] = []
    seen: set[str] = set()
    for part in parts:
        if len(part) < 12:
            continue
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(part)
    return unique[:6]


def build_dynamic_plan_graph(
    objective: str,
    *,
    intent_tags: list[str],
    request_type: str,
    tool_registry_summary: list[dict] | None = None,
) -> list[dict]:
    required_capabilities = _infer_capability_needs(objective, intent_tags, request_type)
    atomic_goals = _split_objective_subgoals(objective)
    graph: list[dict] = []
    if atomic_goals:
        goal_texts = atomic_goals[:4]
    else:
        goal_texts = [f"Investigate {cap.replace('_', ' ')} evidence relevant to the objective." for cap in required_capabilities[:3]]
    if not goal_texts:
        goal_texts = [objective.strip() or "Investigate evidence relevant to the objective."]

    for idx, clause in enumerate(goal_texts, start=1):
        clause_caps = sorted(set(_infer_capability_needs(clause, intent_tags, request_type)))
        subgoal_id = f"sg_dynamic_{idx}"
        deps = [f"sg_dynamic_{idx - 1}"] if idx > 1 else []
        graph.append(
            {
                "subgoal_id": subgoal_id,
                "title": f"Evidence subgoal {idx}",
                "objective": clause,
                "dependencies": deps,
                "evidence_requirements": clause_caps[:5] or required_capabilities[:3] or ["literature"],
                "done_criteria": [
                    "at least one grounded finding",
                    "explicit uncertainty",
                    "captured citations or evidence IDs",
                ],
                "max_calls": 5,
                "phase": _infer_phase_for_node_text(clause),
            }
        )

    # Optional signal from registry scale: when many tools exist, ask planner to keep retrieval selective.
    if tool_registry_summary and len(tool_registry_summary) > 80:
        for node in graph:
            if node["subgoal_id"].startswith("sg_dynamic_"):
                node["done_criteria"].append("tool shortlist remained focused to relevant subset")
    return graph


def _infer_phase_for_node_text(text: str) -> str:
    lowered = str(text or "").lower()
    if any(token in lowered for token in ["researcher", "author", "investigator", "openalex", "affiliation"]):
        return "researcher_scouting"
    if any(
        token in lowered
        for token in ["synthesis", "report", "recommend", "decision", "critique", "contradiction", "stress-test"]
    ):
        return "synthesis_reporting"
    return "evidence_discovery"


def normalize_plan_graph(
    plan_graph: list[dict] | None,
    *,
    objective: str,
    intent_tags: list[str],
    request_type: str,
    max_nodes: int = 6,
) -> list[dict]:
    required_capabilities = _infer_capability_needs(objective, intent_tags, request_type)
    raw_nodes = plan_graph or []
    normalized: list[dict] = []
    seen_ids: set[str] = set()

    for idx, raw in enumerate(raw_nodes[: max_nodes], start=1):
        if not isinstance(raw, dict):
            continue
        candidate_id = str(raw.get("subgoal_id") or raw.get("node_id") or f"sg_dynamic_{idx}").strip().lower()
        candidate_id = re.sub(r"[^a-z0-9_]+", "_", candidate_id).strip("_") or f"sg_dynamic_{idx}"
        if candidate_id in seen_ids:
            candidate_id = f"{candidate_id}_{idx}"
        seen_ids.add(candidate_id)

        title = str(raw.get("title", "")).strip()
        node_objective = str(raw.get("objective") or raw.get("instruction") or title).strip()
        if len(node_objective) < 8:
            continue

        raw_caps = raw.get("evidence_requirements", [])
        if isinstance(raw_caps, str):
            raw_caps = [item.strip() for item in re.split(r"[,\n;]+", raw_caps) if item.strip()]
        node_caps = {str(item).strip() for item in (raw_caps or []) if str(item).strip()}
        node_caps.update(infer_capabilities_from_text(node_objective))
        if not node_caps:
            node_caps.update(required_capabilities[:2] or ["literature"])

        raw_deps = raw.get("dependencies", [])
        if isinstance(raw_deps, str):
            raw_deps = [item.strip() for item in re.split(r"[,\n;]+", raw_deps) if item.strip()]
        deps: list[str] = []
        for dep in raw_deps or []:
            dep_id = re.sub(r"[^a-z0-9_]+", "_", str(dep).strip().lower()).strip("_")
            if dep_id and dep_id in seen_ids and dep_id != candidate_id and dep_id not in deps:
                deps.append(dep_id)

        raw_done = raw.get("done_criteria", [])
        if isinstance(raw_done, str):
            raw_done = [item.strip() for item in re.split(r"[;\n]+", raw_done) if item.strip()]
        done_criteria = [str(item).strip() for item in (raw_done or []) if str(item).strip()]
        if not done_criteria:
            done_criteria = [
                "at least one grounded finding",
                "explicit uncertainty",
                "captured citations or evidence IDs",
            ]

        raw_calls = raw.get("max_calls", 0)
        try:
            max_calls = int(raw_calls or 0)
        except (TypeError, ValueError):
            max_calls = 0
        max_calls = min(max(max_calls, 1), 8)

        raw_phase = str(raw.get("phase", "")).strip().lower()
        phase = raw_phase if raw_phase in {"evidence_discovery", "researcher_scouting", "synthesis_reporting"} else ""
        if not phase:
            phase = _infer_phase_for_node_text(f"{title} {node_objective}")

        normalized.append(
            {
                "subgoal_id": candidate_id,
                "title": title,
                "objective": node_objective,
                "dependencies": deps,
                "evidence_requirements": sorted(node_caps)[:6],
                "done_criteria": done_criteria[:6],
                "max_calls": max_calls,
                "phase": phase,
            }
        )

    if normalized:
        return normalized

    fallback_graph = build_dynamic_plan_graph(
        objective,
        intent_tags=intent_tags,
        request_type=request_type,
        tool_registry_summary=None,
    )
    return fallback_graph[:max_nodes]


def build_dynamic_plan_steps(
    objective: str,
    *,
    intent_tags_override: list[str] | None = None,
    request_type_override: str | None = None,
    tool_registry_summary: list[dict] | None = None,
    plan_graph_override: list[dict] | None = None,
) -> tuple[list[WorkflowStep], list[dict]]:
    intent_tags = (
        sanitize_intent_tags(intent_tags_override)
        if intent_tags_override is not None
        else infer_intent_tags(objective)
    )
    if not intent_tags:
        intent_tags = infer_intent_tags(objective)
    request_type = sanitize_request_type(request_type_override) or classify_request_type(objective)
    using_model_override = bool(plan_graph_override)
    if using_model_override:
        plan_graph = normalize_plan_graph(
            plan_graph_override,
            objective=objective,
            intent_tags=intent_tags,
            request_type=request_type,
        )
    else:
        plan_graph = build_dynamic_plan_graph(
            objective,
            intent_tags=intent_tags,
            request_type=request_type,
            tool_registry_summary=tool_registry_summary,
        )
    if not plan_graph:
        return [], []

    steps: list[WorkflowStep] = []
    researcher_mode = "researcher_discovery" in intent_tags
    tool_capability_map = _capability_map_from_registry_summary(tool_registry_summary)
    for node in plan_graph:
        subgoal_id = str(node.get("subgoal_id", "")).strip()
        evidence_requirements = {
            str(item).strip() for item in node.get("evidence_requirements", []) if str(item).strip()
        }
        recommended_tools, fallback_tools = _registry_summary_tools_for_capabilities(
            tool_registry_summary,
            evidence_requirements or {"literature"},
            k=8,
        )
        if researcher_mode and subgoal_id != "sg_critique":
            recommended_tools = [
                tool
                for tool in recommended_tools
                if not _tool_has_capability(tool, "clinical", tool_capability_map=tool_capability_map)
            ]
            fallback_tools = [
                tool
                for tool in fallback_tools
                if not _tool_has_capability(tool, "clinical", tool_capability_map=tool_capability_map)
            ]
        node_title = str(node.get("title", "")).strip()
        if not using_model_override and subgoal_id == "sg_scope":
            step = _build_scope_step()
        elif not using_model_override and subgoal_id == "sg_critique":
            step = WorkflowStep(
                step_id="",
                title="Critic loop and contradiction check",
                instruction=str(node.get("objective", "")).strip(),
                recommended_tools=recommended_tools[:4],
                fallback_tools=fallback_tools[:4],
                rationale="Before recommendation, validate cross-source consistency and confidence bounds.",
                expected_output_fields=["consistency_findings", "contradictions", "confidence_bounds", "remaining_gaps"],
            )
        elif not using_model_override and subgoal_id == "sg_researchers":
            # Researcher scouting should stay literature/researcher-focused and avoid
            # clinical-trial fallback drift.
            recommended_tools = [
                tool
                for tool in recommended_tools
                if not _tool_has_capability(tool, "clinical", tool_capability_map=tool_capability_map)
            ]
            fallback_tools = [
                tool
                for tool in fallback_tools
                if not _tool_has_capability(tool, "clinical", tool_capability_map=tool_capability_map)
            ]
            step = WorkflowStep(
                step_id="",
                title="Researcher scouting and activity map",
                instruction=str(node.get("objective", "")).strip(),
                recommended_tools=recommended_tools[:8],
                fallback_tools=fallback_tools[:8],
                rationale="Researcher discovery should happen as a distinct phase after evidence context is established.",
                expected_output_fields=[
                    "selected_tools",
                    "researcher_candidates",
                    "activity_level_summary",
                    "contact_signals",
                    "limitations",
                ],
            )
        else:
            instruction = str(node.get("objective", "")).strip() or "Collect the required evidence."
            step = WorkflowStep(
                step_id="",
                title=node_title or f"Dynamic subgoal: {subgoal_id}",
                instruction=instruction,
                recommended_tools=recommended_tools,
                fallback_tools=fallback_tools,
                rationale="Dynamic plan node generated from objective-specific decomposition.",
                expected_output_fields=["selected_tools", "findings", "citations", "limitations"],
            )
        step.subgoal_id = subgoal_id
        step.evidence_requirements = sorted(evidence_requirements)
        step.dependencies = [str(item).strip() for item in node.get("dependencies", []) if str(item).strip()]
        step.max_tool_calls = int(node.get("max_calls", 0) or 0)
        step.done_criteria = [str(item).strip() for item in node.get("done_criteria", []) if str(item).strip()]
        if str(node.get("phase", "")).strip():
            step.observations = list(step.observations) + [f"phase={str(node.get('phase')).strip()}"]
        steps.append(step)

    final_stage = _build_final_stage(request_type)
    final_stage.subgoal_id = "sg_final_report"
    final_stage.dependencies = [step.subgoal_id for step in steps if step.subgoal_id]
    final_stage.done_criteria = ["decision statement present", "confidence stated", "limitations and next actions included"]
    steps.append(final_stage)
    for idx, step in enumerate(steps, start=1):
        step.step_id = f"step_{idx}"
    return _apply_revision_plan_overrides(steps, objective), plan_graph


def build_plan_steps(
    objective: str,
    *,
    intent_tags_override: list[str] | None = None,
    request_type_override: str | None = None,
) -> list[WorkflowStep]:
    # Compatibility wrapper: legacy callers still resolve to the agentic dynamic planner.
    dynamic_steps, _ = build_dynamic_plan_steps(
        objective,
        intent_tags_override=intent_tags_override,
        request_type_override=request_type_override,
    )
    if dynamic_steps:
        return dynamic_steps
    return [_build_scope_step(), _build_final_stage("exploration")]


def create_task(
    objective: str,
    *,
    request_type_override: str | None = None,
    intent_tags_override: list[str] | None = None,
    use_dynamic_planner: bool = True,
    tool_registry_summary: list[dict] | None = None,
    plan_graph_override: list[dict] | None = None,
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
    dynamic_steps, plan_graph = build_dynamic_plan_steps(
        objective,
        intent_tags_override=intent_tags,
        request_type_override=request_type,
        tool_registry_summary=tool_registry_summary,
        plan_graph_override=plan_graph_override,
    )
    task.steps = dynamic_steps or build_plan_steps(
        objective,
        intent_tags_override=intent_tags,
        request_type_override=request_type,
    )
    task.planner_graph = plan_graph
    task.planner_mode = "model_dynamic" if plan_graph_override else "dynamic"
    ensure_phase_state(task)
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
        if "_" not in candidate:
            continue
        if candidate in tool_hints:
            continue
        tool_hints.append(candidate)
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
    use_dynamic_planner: bool = False,
    tool_registry_summary: list[dict] | None = None,
    plan_graph_override: list[dict] | None = None,
) -> tuple[PlanVersion, PlanDelta]:
    base_from_step_index = max(0, task.current_step_index + 1)
    existing_remaining = [clone_step(step) for step in task.steps[base_from_step_index:]]
    previous = active_plan_version(task)
    previous_id = previous.version_id if previous else None

    dynamic_steps, dynamic_graph = build_dynamic_plan_steps(
        revised_objective,
        intent_tags_override=intent_tags,
        request_type_override=request_type,
        tool_registry_summary=tool_registry_summary,
        plan_graph_override=plan_graph_override,
    )
    new_full_plan = dynamic_steps or build_plan_steps(
        revised_objective,
        intent_tags_override=intent_tags,
        request_type_override=request_type,
    )
    task.planner_graph = dynamic_graph
    task.planner_mode = "model_dynamic" if plan_graph_override else "dynamic"
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
    ensure_phase_state(task)

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
