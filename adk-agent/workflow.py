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
class RevisionIntent:
    raw_feedback: str
    objective_adjustments: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    priorities: list[str] = field(default_factory=list)
    exclusions: list[str] = field(default_factory=list)
    evidence_preferences: list[str] = field(default_factory=list)
    output_preferences: list[str] = field(default_factory=list)
    confidence: float = 0.0
    parser_source: str = "fallback"

    def to_dict(self) -> dict:
        return {
            "raw_feedback": self.raw_feedback,
            "objective_adjustments": list(self.objective_adjustments),
            "constraints": list(self.constraints),
            "priorities": list(self.priorities),
            "exclusions": list(self.exclusions),
            "evidence_preferences": list(self.evidence_preferences),
            "output_preferences": list(self.output_preferences),
            "confidence": float(self.confidence),
            "parser_source": self.parser_source,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "RevisionIntent | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            raw_feedback=str(payload.get("raw_feedback", "")).strip(),
            objective_adjustments=[str(x).strip() for x in payload.get("objective_adjustments", []) if str(x).strip()],
            constraints=[str(x).strip() for x in payload.get("constraints", []) if str(x).strip()],
            priorities=[str(x).strip() for x in payload.get("priorities", []) if str(x).strip()],
            exclusions=[str(x).strip() for x in payload.get("exclusions", []) if str(x).strip()],
            evidence_preferences=[str(x).strip() for x in payload.get("evidence_preferences", []) if str(x).strip()],
            output_preferences=[str(x).strip() for x in payload.get("output_preferences", []) if str(x).strip()],
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            parser_source=str(payload.get("parser_source", "fallback") or "fallback"),
        )


@dataclass
class PlanDelta:
    from_version_id: str | None
    to_version_id: str
    added_steps: list[str] = field(default_factory=list)
    removed_steps: list[str] = field(default_factory=list)
    modified_steps: list[str] = field(default_factory=list)
    reordered_steps: list[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "from_version_id": self.from_version_id,
            "to_version_id": self.to_version_id,
            "added_steps": list(self.added_steps),
            "removed_steps": list(self.removed_steps),
            "modified_steps": list(self.modified_steps),
            "reordered_steps": list(self.reordered_steps),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PlanDelta | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            from_version_id=str(payload.get("from_version_id", "")).strip() or None,
            to_version_id=str(payload.get("to_version_id", "")).strip(),
            added_steps=[str(x).strip() for x in payload.get("added_steps", []) if str(x).strip()],
            removed_steps=[str(x).strip() for x in payload.get("removed_steps", []) if str(x).strip()],
            modified_steps=[str(x).strip() for x in payload.get("modified_steps", []) if str(x).strip()],
            reordered_steps=[str(x).strip() for x in payload.get("reordered_steps", []) if str(x).strip()],
            summary=str(payload.get("summary", "")).strip(),
        )


@dataclass
class PlanVersion:
    version_id: str
    created_at: str
    base_from_step_index: int
    request_type: str
    intent_tags: list[str] = field(default_factory=list)
    revision_intent: RevisionIntent | None = None
    steps: list[WorkflowStep] = field(default_factory=list)
    gate_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "created_at": self.created_at,
            "base_from_step_index": self.base_from_step_index,
            "request_type": self.request_type,
            "intent_tags": list(self.intent_tags),
            "revision_intent": self.revision_intent.to_dict() if self.revision_intent else None,
            "steps": [step.to_dict() for step in self.steps],
            "gate_reason": self.gate_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict | None) -> "PlanVersion | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            version_id=str(payload.get("version_id", "")).strip(),
            created_at=str(payload.get("created_at", "")).strip() or _utc_now(),
            base_from_step_index=int(payload.get("base_from_step_index", 0) or 0),
            request_type=str(payload.get("request_type", "exploration") or "exploration"),
            intent_tags=[str(x).strip() for x in payload.get("intent_tags", []) if str(x).strip()],
            revision_intent=RevisionIntent.from_dict(payload.get("revision_intent")),
            steps=[WorkflowStep.from_dict(step) for step in payload.get("steps", []) if isinstance(step, dict)],
            gate_reason=str(payload.get("gate_reason", "")).strip(),
        )


@dataclass
class WorkflowTask:
    """A task with explicit planner output and execution state."""

    task_id: str
    objective: str
    title: str = ""
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
    base_objective: str = ""
    plan_versions: list[PlanVersion] = field(default_factory=list)
    active_plan_version_id: str | None = None
    pending_feedback_queue: list[str] = field(default_factory=list)
    latest_plan_delta: PlanDelta | None = None
    checkpoint_state: str = "closed"
    checkpoint_reason: str = ""
    progress_events: list[dict] = field(default_factory=list)
    progress_summaries: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict:
        title = (self.title or "").strip() or generate_chat_title(self.objective)
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "title": title,
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
            "base_objective": self.base_objective or self.objective,
            "plan_versions": [version.to_dict() for version in self.plan_versions],
            "active_plan_version_id": self.active_plan_version_id,
            "pending_feedback_queue": list(self.pending_feedback_queue),
            "latest_plan_delta": self.latest_plan_delta.to_dict() if self.latest_plan_delta else None,
            "checkpoint_state": self.checkpoint_state,
            "checkpoint_reason": self.checkpoint_reason,
            "progress_events": list(self.progress_events),
            "progress_summaries": list(self.progress_summaries),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "WorkflowTask":
        checkpoint_state_raw = str(
            payload.get("checkpoint_state", "open" if payload.get("awaiting_hitl") else "closed")
        ).strip().lower()
        checkpoint_state = "open" if checkpoint_state_raw == "open" else "closed"
        return cls(
            task_id=payload.get("task_id", ""),
            objective=payload.get("objective", ""),
            title=(str(payload.get("title", "")).strip() or generate_chat_title(payload.get("objective", ""))),
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
            base_objective=payload.get("base_objective", "") or payload.get("objective", ""),
            plan_versions=[
                version
                for version in (PlanVersion.from_dict(item) for item in payload.get("plan_versions", []))
                if version is not None
            ],
            active_plan_version_id=payload.get("active_plan_version_id"),
            pending_feedback_queue=[str(x).strip() for x in payload.get("pending_feedback_queue", []) if str(x).strip()],
            latest_plan_delta=PlanDelta.from_dict(payload.get("latest_plan_delta")),
            checkpoint_state=checkpoint_state,
            checkpoint_reason=str(payload.get("checkpoint_reason", "")),
            progress_events=[item for item in payload.get("progress_events", []) if isinstance(item, dict)],
            progress_summaries=[item for item in payload.get("progress_summaries", []) if isinstance(item, dict)],
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


def generate_chat_title(objective: str, *, max_words: int = 8) -> str:
    text = str(objective or "").strip()
    if not text:
        return "Untitled Research"

    markers = [
        "\nUser revision to scope/decomposition:",
        "\nRevision directives to apply:",
        "\nRequired revision constraints:",
        "\nUser clarification:",
    ]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    compact = re.sub(r"\s+", " ", first_line).strip(" .")
    compact = re.sub(
        r"^(?:please\s+|can you\s+|could you\s+|i need\s+|find me\s+|show me\s+|tell me\s+)",
        "",
        compact,
        flags=re.IGNORECASE,
    )
    compact = compact.strip(" .")
    if not compact:
        return "Untitled Research"

    words = compact.split()
    title = " ".join(words[:max(3, max_words)])
    if len(words) > max_words:
        title = f"{title}..."
    if len(title) > 72:
        title = f"{title[:69].rstrip()}..."
    return title or "Untitled Research"


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
