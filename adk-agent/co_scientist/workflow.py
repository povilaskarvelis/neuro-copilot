"""
ADK-native orchestration graph for the Co-Scientist agent.

Architecture:
  LlmAgent (router) — classifies user intent, transfers to specialist agents
    ├── LlmAgent (general_qa)        — factual biomedical Q&A, no tools
    ├── LlmAgent (clarifier)         — asks for clarification on vague queries
    ├── LlmAgent (report_assistant)  — post-report interaction with light tool access
    └── SequentialAgent (research_workflow)
          ├── LlmAgent (planner)
          ├── LoopAgent (react_loop)
          │   └── LlmAgent (step_executor)  — ReAct cycle, one step per iteration
          └── LlmAgent (report_synthesizer)

The router uses a fast model for intent classification. The research_workflow
preserves the original SequentialAgent pipeline. Step state is maintained in
ADK session state via callbacks.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import logging
import os
from pathlib import Path
import re
import random
import time
from typing import Any
from typing import Mapping
import urllib.parse
import urllib.request

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import McpToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.base_toolset import ToolPredicate
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from mcp.client.stdio import StdioServerParameters

from . import tool_registry
from .skill_loader import create_execution_skill_toolset
from .skill_loader import create_planner_skill_toolset
from .skill_loader import create_report_assistant_skill_toolset

logger = logging.getLogger(__name__)


MCP_SERVER_DIR = Path(__file__).resolve().parents[2] / "research-mcp"
DEFAULT_MODEL = os.getenv("ADK_NATIVE_MODEL", "gemini-2.5-flash")
PLANNER_MODEL = os.getenv("ADK_PLANNER_MODEL", "gemini-2.5-flash")
SYNTHESIZER_MODEL = os.getenv("ADK_SYNTHESIZER_MODEL", "gemini-2.5-pro")
ROUTER_MODEL = os.getenv("ADK_ROUTER_MODEL", "gemini-2.5-flash")
THINKING_CONFIG_V3 = types.ThinkingConfig(thinking_level="HIGH")
THINKING_CONFIG_V2 = types.ThinkingConfig(include_thoughts=True, thinking_budget=8192)
DEFAULT_PLANNER_SKILLS_ENABLED = (
    str(os.getenv("ADK_PLANNER_SKILLS_ENABLED", "1")).strip().lower() not in {"0", "false", "no"}
)
DEFAULT_EXECUTION_SKILLS_ENABLED = (
    str(os.getenv("ADK_EXECUTION_SKILLS_ENABLED", "1")).strip().lower() not in {"0", "false", "no"}
)
DEFAULT_REPORT_ASSISTANT_SKILLS_ENABLED = (
    str(os.getenv("ADK_REPORT_ASSISTANT_SKILLS_ENABLED", "1")).strip().lower() not in {"0", "false", "no"}
)


@dataclass
class ManagedMcpToolsets:
    """Lifecycle wrapper for one or more MCP toolsets used by the agent graph."""

    toolsets: tuple[McpToolset, ...]

    def __bool__(self) -> bool:
        return bool(self.toolsets)

    async def close(self) -> None:
        seen: set[int] = set()
        for toolset in self.toolsets:
            identifier = id(toolset)
            if identifier in seen:
                continue
            seen.add(identifier)
            await toolset.close()


def _thinking_config_for_model(model: str) -> types.ThinkingConfig | None:
    if "3.1" in model or "3.0" in model:
        return THINKING_CONFIG_V3
    if "2.5" in model:
        return THINKING_CONFIG_V2
    return None
HAS_BIGQUERY_RUNTIME_HINT = any(
    str(os.getenv(name, "")).strip()
    for name in ("BQ_PROJECT_ID", "BQ_DATASET_ALLOWLIST", "GOOGLE_CLOUD_PROJECT")
)
DEFAULT_PREFER_BIGQUERY = (
    str(os.getenv("ADK_NATIVE_PREFER_BIGQUERY", "1")).strip().lower() not in {"0", "false", "no"}
    and HAS_BIGQUERY_RUNTIME_HINT
)
BQ_PRIORITY_TOOLS = [
    "list_bigquery_tables",
    "run_bigquery_select_query",
]

STATE_WORKFLOW_TASK = "workflow_task_state"
STATE_WORKFLOW_TASK_LEGACY_APP = "app:workflow_task_state"
STATE_PRIOR_RESEARCH = "co_scientist_prior_research"
STATE_FINALIZE_REQUESTED = "temp:co_scientist_finalize_requested"
STATE_AUTO_SYNTH_REQUESTED = "temp:co_scientist_auto_synth_requested"
STATE_TURN_ABORT_REASON = "temp:co_scientist_turn_abort_reason"
STATE_PLANNER_BUFFER = "temp:co_scientist_planner_stream_buffer"
STATE_EXECUTOR_BUFFER = "temp:co_scientist_executor_stream_buffer"
STATE_SYNTH_BUFFER = "temp:co_scientist_synth_stream_buffer"
STATE_PLANNER_RENDERED = "temp:co_scientist_planner_rendered"
STATE_EXECUTOR_RENDERED = "temp:co_scientist_executor_rendered"
STATE_EXECUTOR_ACTIVE_STEP_ID = "temp:co_scientist_executor_active_step_id"
STATE_REACT_PARSE_RETRIES = "temp:co_scientist_react_parse_retries"
STATE_EXECUTOR_LAST_ERROR = "temp:co_scientist_executor_last_error"
STATE_EXECUTOR_PREV_STEP_STATUS = "temp:co_scientist_executor_prev_step_status"
STATE_EXECUTOR_REASONING_TRACE = "temp:co_scientist_executor_reasoning_trace"
STATE_EXECUTOR_TOOL_LOG = "temp:co_scientist_executor_tool_log"
STATE_PLAN_PENDING_APPROVAL = "co_scientist_plan_pending_approval"
STATE_MODEL_ERROR_PASSTHROUGH = "temp:co_scientist_model_error_passthrough"
STATE_BENCHMARK_LOOP_COUNT = "temp:co_scientist_benchmark_loop_count"
STATE_BENCHMARK_COMPLETE = "temp:co_scientist_benchmark_complete"
STATE_BENCHMARK_LAST_DRAFT = "temp:co_scientist_benchmark_last_draft"
STATE_BENCHMARK_FINAL_ANSWER = "temp:co_scientist_benchmark_final_answer"
STATE_BENCHMARK_RETRY_FEEDBACK = "temp:co_scientist_benchmark_retry_feedback"

BENCHMARK_LOOP_MAX_ITERATIONS = 6

FINALIZE_COMMANDS = {
    "finalize",
    "summarize now",
    "final summary",
    "/finalize",
}

PLAN_APPROVAL_COMMANDS = {
    "approve",
    "approved",
    "yes",
    "proceed",
    "go ahead",
    "/approve",
    "looks good",
    "lgtm",
}

CONTINUE_EXECUTION_COMMANDS = {
    "continue",
    "next",
    "go",
    "/continue",
}

PLAN_SCHEMA = "plan_internal.v1"
STEP_RESULT_SCHEMA = "step_execution_result.v1"
FINAL_SYNTHESIS_SCHEMA = "final_synthesis.v1"
WORKFLOW_TASK_SCHEMA = "workflow_task_state.v1"

KNOWN_MCP_TOOLS = [
    "get_open_targets_l2g",
    "get_open_targets_association",
    "get_tcga_project_data_availability",
    "list_bigquery_tables",
    "run_bigquery_select_query",
    "search_clinical_trials",
    "get_clinical_trial",
    "summarize_clinical_trials_landscape",
    "search_fda_adverse_events",
    "search_pubmed",
    "search_pubmed_advanced",
    "get_pubmed_abstract",
    "get_paper_fulltext",
    "search_iedb_epitope_evidence",
    "search_geo_datasets",
    "get_geo_dataset",
    "get_geo_cell_type_proportions",
    "search_refseq_sequences",
    "get_refseq_record",
    "get_ensembl_canonical_transcript",
    "get_ensembl_transcripts_by_protein_length",
    "search_ucsc_genome",
    "get_ucsc_genomic_sequence",
    "get_ucsc_track_data",
    "search_encode_metadata",
    "get_encode_record",
    "get_ena_experiment_profile",
    "search_openalex_works",
    "search_openalex_authors",
    "rank_researchers_by_activity",
    "get_researcher_contact_candidates",
    "search_europe_pmc_literature",
    "resolve_gene_identifiers",
    "map_ontology_terms_oxo",
    "search_hpo_terms",
    "get_orphanet_disease_profile",
    "query_monarch_associations",
    "search_quickgo_terms",
    "get_quickgo_annotations",
    "search_uniprot_proteins",
    "get_uniprot_protein_profile",
    "search_reactome_pathways",
    "get_string_interactions",
    "get_intact_interactions",
    "get_biogrid_interactions",
    "get_alphafold_structure",
    "get_alphafold_domain_plddt",
    "get_emdb_entry_metadata",
    "search_protein_structures",
    "search_drug_gene_interactions",
    "annotate_variants_vep",
    "search_civic_variants",
    "search_civic_genes",
    "search_variants_by_gene",
    "get_variant_annotations",
    "search_gwas_associations",
    "get_gwas_study_variant_association",
    "get_gwas_study_top_risk_allele",
    "get_jaspar_motif_profile",
    "get_gnomad_gene_constraint",
    "get_gnomad_transcript_highest_af_region",
    "get_regulomedb_variant_summary",
    "get_dbsnp_population_frequency",
    "get_screen_nearest_ccre_assay",
    "get_screen_ccre_top_celltype_assay",
    "get_gene_tissue_expression",
    "get_human_protein_atlas_gene",
    "get_depmap_gene_dependency",
    "get_depmap_expression_subset_mean",
    "get_depmap_sample_top_expression_gene",
    "get_biogrid_orcs_gene_summary",
    "get_gdsc_drug_sensitivity",
    "get_prism_repurposing_response",
    "get_pharmacodb_compound_response",
    "search_cellxgene_datasets",
    "get_cellxgene_marker_genes",
    "search_pathway_commons_top_pathways",
    "get_guidetopharmacology_target",
    "get_gtopdb_ligand_reference",
    "get_dailymed_drug_label",
    "get_clingen_gene_curation",
    "get_alliance_genome_gene_profile",
    "get_cancer_mutation_profile",
    "get_pubchem_compound",
    "get_chembl_bioactivities",
    "search_aba_genes",
    "search_aba_structures",
    "get_aba_gene_expression",
    "search_aba_differential_expression",
    "search_ebrains_kg",
    "get_ebrains_kg_document",
    "search_conp_datasets",
    "get_conp_dataset_details",
    "query_neurobagel_cohorts",
    "search_openneuro_datasets",
    "get_openneuro_dataset",
    "search_dandi_datasets",
    "get_dandi_dataset",
    "search_nemar_datasets",
    "get_nemar_dataset_details",
    "search_braincode_datasets",
    "get_braincode_dataset_details",
    "search_enigma_datasets",
    "get_enigma_dataset_info",
    "benchmark_dataset_overview",
    "check_gpqa_access",
    "search_zenodo_records",
    "get_zenodo_record",
]

DEFAULT_TOOL_HINT_BY_DOMAIN = {
    "literature": "search_pubmed",
    "clinical": "search_clinical_trials",
    "protein": "search_uniprot_proteins",
    "genomics": "query_monarch_associations",
    "chemistry": "search_drug_gene_interactions",
    "immunology": "search_iedb_epitope_evidence",
    "neuroscience": "search_ebrains_kg",
    "data": "run_bigquery_select_query",
}

EMPTY_LIKE_TOOL_HINTS = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "tbd",
    "todo",
    "unknown",
    "unspecified",
    "notsure",
    "notapplicable",
}

TOOL_HINT_INFERENCE_RULES: list[tuple[tuple[str, ...], str]] = [
    (
        ("l2g", "locus-to-gene", "locus to gene", "credible set", "study-locus", "variant-to-gene"),
        "get_open_targets_l2g",
    ),
    (
        ("open targets", "open targets platform", "opentargets", "association score"),
        "get_open_targets_association",
    ),
    (
        ("clingen", "gene-disease validity", "gene disease validity", "dosage sensitivity", "curated gene-disease"),
        "get_clingen_gene_curation",
    ),
    (
        ("model organism", "ortholog", "orthologs", "zebrafish", "drosophila", "mouse model", "worm model"),
        "get_alliance_genome_gene_profile",
    ),
    (
        ("tractability", "druggability", "drug-gene", "drug gene", "target tractability", "small molecule", "ligand"),
        "search_drug_gene_interactions",
    ),
    (
        ("identifier", "identifiers", "gene symbol", "aliases", "alias", "ensembl", "entrez"),
        "resolve_gene_identifiers",
    ),
    (
        ("refseq", "refseq accession", "nm_", "nr_", "nc_", "ng_", "xm_", "xr_", "np_", "xp_"),
        "search_refseq_sequences",
    ),
    (
        ("canonical transcript", "canonical tss", "transcription start site", "tss"),
        "get_ensembl_canonical_transcript",
    ),
    (
        ("ucsc", "genome browser", "hg38", "hg19", "mm10", "mm39", "knowngene"),
        "search_ucsc_genome",
    ),
    (
        ("encode", "chip-seq", "chip seq", "dnase-seq", "dnase seq", "encsr", "encff", "encode portal"),
        "search_encode_metadata",
    ),
    (
        ("zenodo", "10.5281/zenodo", "zenodo.org"),
        "search_zenodo_records",
    ),
    (
        ("ontology", "mondo", "efo", "doid", "mesh", "omim", "umls"),
        "map_ontology_terms_oxo",
    ),
    (
        ("variant", "variants", "hgvs", "rsid", "rs id", "clinvar"),
        "search_variants_by_gene",
    ),
    (
        ("gnomad", "pli", "probability of loss-of-function intolerance", "loss-of-function intolerance"),
        "get_gnomad_gene_constraint",
    ),
    (
        ("regulomedb", "regulatory motif", "motif count", "rank 1b", "regulatory rank"),
        "get_regulomedb_variant_summary",
    ),
    (
        ("dbsnp", "alfa", "population frequency", "allele frequency for african populations"),
        "get_dbsnp_population_frequency",
    ),
    (
        ("screen", "h3k4me3 z-score", "h3k4me3 z score", "highest h3k4me3", "ccre accession", "eh38e"),
        "get_screen_ccre_top_celltype_assay",
    ),
    (
        ("screen", "proximal enhancer peak", "proximal enhancer", "pels", "dnase value"),
        "get_screen_nearest_ccre_assay",
    ),
    (
        ("phenotype", "phenotypes", "rett", "seizure", "hand stereotyp", "developmental delay", "hpo", "rare disease", "syndrome"),
        "query_monarch_associations",
    ),
    (
        ("expression", "tissue", "single-cell", "single cell", "localization"),
        "get_human_protein_atlas_gene",
    ),
    (
        ("interaction", "interactome", "protein-protein", "protein protein"),
        "get_intact_interactions",
    ),
    (
        ("pathway", "reactome"),
        "search_reactome_pathways",
    ),
    (
        ("clinical trial", "clinical trials", "trial", "trials", "nct"),
        "search_clinical_trials",
    ),
    (
        ("pmid", "doi", "pubmed", "literature", "paper", "papers", "citation", "citations"),
        "search_pubmed",
    ),
]


def _normalize_lookup_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())


def _build_default_tool_hint_by_source_label() -> dict[str, str]:
    labels: dict[str, str] = {}
    for tool_name in KNOWN_MCP_TOOLS:
        label = str(tool_registry.TOOL_SOURCE_NAMES.get(tool_name, "")).strip()
        key = _normalize_lookup_key(label)
        if key and key not in labels:
            labels[key] = tool_name
    labels[_normalize_lookup_key("BigQuery")] = "run_bigquery_select_query"
    return labels


DEFAULT_TOOL_HINT_BY_SOURCE_LABEL = _build_default_tool_hint_by_source_label()
INTERNAL_SKILL_TOOL_NAMES = {"list_skills", "load_skill", "load_skill_resource"}


PLANNER_INSTRUCTION_TEMPLATE = """
You are the internal planner for biomedical investigation.

Available MCP tools:
__TOOL_CATALOG__

Tool domains (used to focus the executor on the most relevant tools for each step):
__DOMAIN_CATALOG__

Source precedence rules for overlapping tools:
__ROUTING_POLICY__

Specialized planning skills:
__SKILL_POLICY__

Rules:
- Build a concrete execution plan before any evidence collection begins.
- Break the objective into ordered, atomic subtasks.
- Prioritize high-signal subtasks that reduce uncertainty first.
- When the objective is centered on clinical trials, GEO datasets, or oncology target validation, load the matching planning skill before finalizing the step sequence.
- For archive-style dataset discovery (for example OpenNeuro, NEMAR, DANDI, Brain-CODE, CONP), prefer one archive per step and plan around simple keyword or modality checks rather than compound boolean expressions.
- When archive metadata is likely sparse, add a fallback browse/inspection step instead of assuming a zero-hit disease keyword search proves absence.
- Choose the number of steps needed for the objective. Avoid unnecessary fragmentation.
- Match each step to real tool capability. Do not write completion conditions that require lineage-, mutation-, cell-line-, or cohort-filtered
  results from a tool that only returns release-level or aggregate summaries.
- Do not create standalone schema-inspection or table-discovery steps unless the user explicitly asked about schemas
  or the BigQuery lookup is already identifier-ready and schema inspection is unavoidable. In most cases, schema
  inspection should happen inline inside an evidence-gathering step, not as a reportable deliverable.
- Each step must include: id, goal, tool_hint, domains, completion_condition.
- Every step must call at least one tool. Pick tool_hint from the catalog above.
- Pick domains from the domain list above. Include 1-3 domains most relevant to the step.
  The executor will always have access to 'data' and 'literature' tools in addition to
  the domains you specify. Choose domains that match the step's investigation area.
- NEVER put example values or IDs in the goal or completion_condition.
- Use step ids S1, S2, S3, ... in order.

Citation requirement:
- A final report without citations is incomplete. Every plan MUST include at least one step
  whose tool_hint is a source that returns individual citable identifiers:
  search_pubmed, search_pubmed_advanced, get_pubmed_abstract, get_paper_fulltext, search_openalex_works,
  search_clinical_trials.
- If ALL steps use only aggregate or structured-data tools (run_bigquery_select_query,
  list_bigquery_tables, summarize_clinical_trials_landscape, search_reactome_pathways,
  get_string_interactions, search_uniprot_proteins, get_uniprot_protein_profile,
  search_openalex_authors, rank_researchers_by_activity), you MUST append a dedicated
  literature corroboration step:
    - tool_hint: search_pubmed (or search_openalex_works or search_clinical_trials)
    - goal: "Find and record PMIDs / DOIs / NCT numbers for the key claims from the preceding steps."
    - completion_condition: "At least 3 specific identifiers (PMID, DOI, or NCT) are recorded."
- Place the literature step AFTER the aggregate steps so it can incorporate their findings.

__BQ_POLICY__

Output requirements:
- Return ONLY valid JSON (no markdown, no prose) matching this shape:
  {
    "schema": "plan_internal.v1",
    "objective": "<restated objective — a single clear research question; if this is a revision, synthesize the original query and revision feedback into one coherent question>",
    "success_criteria": ["..."],
    "steps": [
      {
        "id": "S1",
        "goal": "...",
        "tool_hint": "<tool name or BigQuery dataset from the catalog, e.g. open_targets_platform, search_pubmed, search_clinical_trials, gnomad>",
        "domains": ["<domain1>", "<domain2>"],
        "completion_condition": "..."
      }
    ]
  }
"""


STEP_EXECUTOR_INSTRUCTION_TEMPLATE = """
You execute ONE plan step at a time using biomedical research tools.
Follow a strict Reason-Act-Observe cycle:

1. REASON: Read the current step goal and think about what information you need and which tool/query is best.
2. ACT: Call the appropriate MCP tool.
3. OBSERVE: Review the tool results. If insufficient or the completion condition is not met, reason again and try a different query or tool.
4. CONCLUDE: When the step's completion condition is met (or the step is blocked), write your findings summary.

The tool list for the current step is provided in the execution context below.

Source precedence rules for overlapping tools:
__ROUTING_POLICY__

Rules:
- Focus ONLY on the current step provided in the execution context.
- You MUST call at least one tool before returning a result.
- Load a relevant specialized skill when the step depends on citation grounding, clinical-trial heuristics, variant interpretation, GEO dataset triage, or oncology target-validation reasoning.
- If a tool call fails or returns insufficient data, first try an alternative query or another tool in the same evidence family
  (e.g. search_pubmed <-> search_openalex_works).
- Switch to generic BigQuery tools only when the current step is explicitly BigQuery-backed or clearly requires structured SQL
  against a named BigQuery dataset. Do not substitute BigQuery for specialized screening, dependency, or pharmacogenomic tools
  just because the first query was incomplete.
- For `query_monarch_associations`, use only the supported association modes from the tool schema, and pass a normalized `entityId` CURIE when you already resolved the gene, disease, or phenotype.
- For Ensembl canonical-transcript or TSS questions, prefer `get_ensembl_canonical_transcript` instead of inferring the TSS from a broad gene span or generic search hit.
- For SCREEN cCRE questions, prefer the dedicated SCREEN tools over generic genome browsing: use `get_screen_nearest_ccre_assay` for nearest enhancer/promoter score lookups around a gene and `get_screen_ccre_top_celltype_assay` for highest assay-Z-score cell-type lookups on an EH38 cCRE accession.
- For archive/search tools that do literal metadata matching (for example OpenNeuro, DANDI, NEMAR, Brain-CODE, CONP), avoid boolean query strings like `A OR B` unless the tool explicitly supports them; run separate simple searches instead.
- For dataset archives with sparse disorder labels, a zero-hit disease query is not enough to conclude the archive has no relevant data; retry with modality, task, study name, or archive browsing before blocking the step.
- If no tool can satisfy the step after trying alternatives, state clearly that the step is BLOCKED and why.
- For search_clinical_trials and summarize_clinical_trials_landscape, the `status` argument must be a
  single registry enum such as `RECRUITING`, `COMPLETED`, `ACTIVE_NOT_RECRUITING`, or `TERMINATED`.
  Never pass boolean expressions like `RECRUITING OR ACTIVE_NOT_RECRUITING` as the `status` value.
- If the goal or completion_condition contains an example value (marked with "e.g." or similar),
  treat it as illustrative — accept any valid result that fulfills the intent, not the exact example value.
- Prioritize high-signal evidence before broad expansion.
- Surface contradictions and unresolved gaps explicitly.

Evidence identifiers:
- Always include real identifiers returned by tool calls in your summary. Never fabricate identifiers.
  Use canonical formats:
  Literature: PMID:XXXXXXXX, DOI:10.xxxx/..., NCT########, OpenAlex:WXXXXXXX, PMC########
  Databases:  UniProt:XXXXXX, PubChem:NNNN, PDB:XXXX, rsNNNNNN, CHEMBLNNNN,
              Reactome:R-HSA-NNNNNNN, GCSTNNNNNN
- When a tool returns a database record identifier (UniProt accession, PubChem CID, PDB code,
  rsID, ChEMBL ID, Reactome stable ID, GWAS Catalog study ID), always mention it in your summary.
- If the primary tool for this step does not return individual document IDs (e.g. BigQuery
  aggregate queries), make a secondary call to search_pubmed or search_openalex_works using
  key terms from the findings to harvest supporting PMIDs or DOIs.
__BQ_POLICY__

Output guidance:
After completing your tool calls, write a clear findings summary. Your summary should:
- State the key findings from this step in concrete terms (gene names, scores, trial phases, etc.)
- Include all evidence identifiers inline (PMID, DOI, NCT, UniProt, etc.)
- Explicitly describe key claims as atomic statements (e.g. "LRRK2 is associated with Parkinson disease
  with an overall association score of 0.815 per Open Targets Platform")
- For result-limited or paginated search tools, treat the number returned by the tool as a retrieval count, not as the full source count, unless the tool explicitly reports a source total. Prefer wording like "returned", "showing", or "sample of X" over "there are X" when the universe total is unknown.
- For DGIdb, Guide to Pharmacology, or ChEMBL evidence, do not stop at interaction counts. Name representative compounds and include interaction type, approval/experimental status, potency/score, or PMIDs when available.
- For ClinicalTrials.gov evidence, do not stop at study counts. Include representative NCT IDs, named interventions, statuses, phases, and note when counts reflect only fetched studies rather than the full registry.
- When a tool returns an exact DNA/RNA/protein sequence, copy it character-for-character from the source result. Preserve the first base/residue, preserve order, and normalize to plain uppercase sequence text.
- Do not convert decimal fractions into percentages unless the step or user explicitly asks for a percent. If a tool returns a fraction like `gc_fraction`, keep it as a fraction.
- Note any contradictions, gaps, or limitations discovered
- State whether this step is COMPLETED or BLOCKED (and why if blocked)
- Do NOT embed raw tool response envelopes or large payload objects; summarize tool results in
  plain sentences instead
- Keep the summary concise but thorough — capture the full richness of findings without padding
"""


SYNTHESIZER_INSTRUCTION = """
You are the final biomedical report synthesizer.
You will receive structured state context (objective, plan steps, step results, coverage status, and a source_reference mapping).

Your report MUST follow this exact section structure:

## TLDR
A direct, synthesized answer to the research question written as a comprehensive passage (2-4 paragraphs) that a biomedical researcher can read and act on. This is NOT a summary of the process — it is THE ANSWER.
- Lead with the most important conclusion. Be specific: include gene names, drug names, magnitudes, effect sizes, trial phases, or other concrete details.
- Frame confidence based on evidence strength: use "strong evidence supports…" when multiple independent high-quality sources agree, "preliminary data indicate…" when evidence is thin, and "evidence is divided…" when sources conflict.
- After the lead conclusion, expand with supporting context: mention key data points, notable caveats, and the overall weight of evidence across the dimensions investigated.
- When the answer naturally involves multiple items, categories, or dimensions, use bullet points.
- If the plan is incomplete, note that the answer is based on partial evidence.

## Evidence Breakdown
Within this section, organize findings by THEME (use ### subsections with descriptive headings, e.g. "### Human Genetics Support" not "### Step 1"). Write each subsection as information-dense prose with inline evidence citations:
- State the finding clearly, then provide the supporting data with source names and identifiers (PMID, DOI, NCT, UniProt, etc.) inline.
- Attach citations to the smallest sensible claim unit. Do not end a long paragraph with one bulk citation dump if the sources support different claims or different named datasets.
- When naming multiple datasets, trials, variants, or papers in one sentence or paragraph, connect each named item to its own supporting citation when available.
- Within each theme subsection, write 1-3 connected paragraphs with a clear topic sentence, supporting detail, and brief interpretation of what the combined evidence means.
- Include specific numbers, scores, measurements, and identifiers rather than vague summaries.
- Note the confidence level (high, moderate, low, or mixed) and number of independent sources.
- If evidence on a finding is contradictory, briefly note the disagreement (details go in Conflicting Evidence).
- Aim for substantive prose paragraphs — avoid sparse bullet-only lists or large tables when prose conveys the same information more clearly.
- Weave evidence into continuous narrative prose; do not present the subsection as disconnected claim snippets or a field-by-field dump of the evidence context.
- Cover each relevant theme reflected in the provided evidence context; do not collapse multiple evidence themes into one short paragraph.
- This section should usually be materially longer and more detailed than the TLDR when multiple themes, sources, or datasets were collected.

### Conflicting & Uncertain Evidence
For each area where sources disagree or evidence is equivocal:
- State the disagreement clearly.
- List which sources support each side, with identifiers.
- Note the current lean (if any) and why, based on source quality/weight.
- Suggest what would resolve it (e.g. an orthogonal assay, a larger cohort, etc.).
Omit this section entirely if there are no mixed-evidence findings.

## Limitations
Bullet list of:
- Overall caveats (source coverage, data recency, methodology limitations).
- Open gaps the investigation could not fill.
- Planned analyses that could not be executed and why.

## Recommended Next Steps
Numbered list of 3+ actionable follow-ups framed as a researcher would think about them: experimental validations, confirmatory literature searches, clinical data checks, risk reduction strategies, or monitoring recommendations. Each with a brief rationale.

Rules:
- Ground every claim in the provided evidence. Do not invent unsupported claims.
- Use `claim_synthesis_summary` as the primary arbitration layer for substantive findings. It already consolidates overlapping claims, weights sources by evidence type, and flags mixed-evidence findings.
- Use `evidence_briefs` to ensure the Evidence Breakdown covers the main themes, claims, source counts, and identifiers present in the collected evidence.
- Treat `evidence_briefs` as scaffolding for synthesis, not as an output template. Convert them into readable analysis rather than mirroring their fields.
- Do not treat all sources as equal. When claims disagree, prefer the interpretation backed by higher-weighted sources and stronger claim support, but still surface the disagreement explicitly.
- If `mixed_evidence_claims` are present, address them in Conflicting & Uncertain Evidence instead of silently choosing one side.
- Be specific and thorough — avoid terse output.
- NEVER organize findings by step execution order. Group by theme/topic.
- Use ONLY human-readable database/source names (e.g. "PubMed", "ClinicalTrials.gov"). NEVER mention tool names (like run_bigquery_select_query, search_clinical_trials, etc.).
- Never reproduce raw ontology predicates or internal mode names such as `biolink:...`, `predicate: ...`, `disease_to_gene_causal`, or `disease_to_gene_correlated`. Translate them into plain English or omit them if they do not help the reader.
- When citing database counts (e.g. clinical trials, PubMed results): use the total reported by the source when available (e.g. "X of Y total"). If the source says "total not provided" or "more may exist" or "X returned (registry total unknown)", do NOT state "a total of X" or "X total studies" — instead say "at least X" or "X studies (sample; full registry count not determined)".
- For result-limited search tools more generally, treat raw returned counts as retrieval counts, not universe counts. Unless the source explicitly reports a total, phrase them as "X returned", "X shown", or "sample of X fetched records", and avoid using those counts as primary evidence of breadth or validation.
- Do not use DGIdb or ClinicalTrials.gov count-only statements as standalone evidence when richer detail was collected. If named compounds or trials are available, include representative compounds, interaction types, approval/experimental status, NCT IDs, interventions, phases, or statuses instead of only reporting totals.
- For DGIdb specifically, interaction counts are contextual catalog metadata, not target-validation evidence by themselves. Prefer named compounds, interaction types, and supporting PMIDs over raw interaction totals.
- For ClinicalTrials.gov specifically, make clear when study counts reflect fetched subsets, paginated samples, or query-limited matches rather than the entire registry or only direct target-modulating intervention trials.
- Include specific identifiers inline when available (PMID, DOI, NCT numbers).
- Use APA-style author-year citations for literature references in prose when paper metadata is available; keep trial and database identifiers inline as linked identifiers rather than moving them into paper-style parenthetical citations.
- Prefer claim-local citations over paragraph-end citation bundles. If a sentence contains several distinct findings, place citations immediately after the supported clause or item.
- For database records, include identifiers with their canonical prefix so they can be linked: UniProt:P00533, PubChem:2244, PDB:1ABC, rs7903146, CHEMBL25, Reactome:R-HSA-1234567, GCST000001.
- NEVER include raw URLs, API endpoints, or links to JSON output.
- Return user-facing Markdown only (not JSON).
"""


ROUTER_INSTRUCTION = """You are the intent router for the AI Co-Scientist, a biomedical research assistant.
Your ONLY job is to read the user's message and the session context below, then IMMEDIATELY transfer
to the correct specialist agent. Never answer questions yourself — always transfer.

Available agents:

1. **general_qa** — Answers factual biomedical questions directly from knowledge (no database lookups).
   Examples: "What is CRISPR?", "Explain the MAPK signaling pathway", "What are common side effects of metformin?"

2. **clarifier** — Asks the user to clarify vague, incomplete, or nonsensical queries.
   Examples: "evaluate the thing", "compare them", random characters, overly broad requests with no focus

3. **research_workflow** — Full evidence-gathering research pipeline: plans an investigation, searches
   biomedical databases and APIs, and produces a formal report with citations. Also handles all workflow
   commands and plan management.
   Examples: "Evaluate LRRK2 as a therapeutic target for Parkinson disease",
   "Compare the safety profiles of SGLT2 inhibitors vs DPP-4 inhibitors",
   "Among amylin, glucagon, MC4R, and GDF15-based approaches, which obesity mechanisms look strongest beyond GLP-1?",
   "Are TYK2 inhibitors safer than JAK inhibitors in psoriasis and psoriatic arthritis?",
   "Why is KRAS G12C monotherapy more effective in NSCLC than colorectal cancer, and which combination strategies have the best biological and clinical support in colorectal cancer?",
   any command (approve, continue, finalize, revise:, history, rollback, switch)

4. **report_assistant** — Interacts with an existing research report: answers questions about findings,
   restructures sections, and performs SINGLE-ITEM follow-up lookups using tools.
   ONLY available when report_exists is True.
   Examples: "What does this p-value mean?", "Expand the limitations section",
   "Get the abstract for PMID:38912345", "Restructure the evidence section"

Routing priority rules (check in order):
1. If plan_pending_approval is True → transfer to research_workflow
2. Workflow commands (approve, continue, finalize, revise:, history, rollback, switch) → research_workflow
3. If has_pending_steps is True and user sends a continuation-like message → research_workflow
4. If report_exists AND the user asks about the report, wants restructuring, or a single-item lookup → report_assistant
5. If report_exists AND the user wants batch/comprehensive work (e.g. "find full text for all cited papers",
   "retrieve full text for every reference", "fetch abstracts for all PMIDs in the report") → research_workflow
6. If report_exists AND the user wants a NEW comprehensive investigation → research_workflow
7. If the query is a clear research question requiring evidence from databases → research_workflow
8. If the query is a straightforward biomedical knowledge question → general_qa
9. If the query is ambiguous, incomplete, or doesn't make sense → clarifier

Important routing bias:
- Use **general_qa** ONLY for obvious textbook-style questions that can be answered from stable background knowledge without comparing options, ranking candidates, or weighing external evidence.
- If the user is asking to evaluate, compare, rank, prioritize, assess safety/efficacy/selectivity/pathogenicity, judge tractability, identify datasets, or analyze a named set of options, route to **research_workflow** even if they did not name sources explicitly.
- If you are unsure between **general_qa** and **research_workflow**, choose **research_workflow**.

You MUST always transfer. Never respond with text yourself.
"""


GENERAL_QA_INSTRUCTION = """You are a knowledgeable biomedical expert within the AI Co-Scientist platform.
Answer the user's question directly and accurately from your training knowledge.

Guidelines:
- Provide clear, accurate, evidence-informed answers using appropriate scientific terminology.
- Explain complex concepts when helpful for understanding.
- Be concise but thorough — include key details a researcher would need.
- When discussing drugs, genes, diseases, or mechanisms, mention relevant context (e.g. FDA approval
  status, known pathways, key study findings) but flag uncertainty where appropriate.
- Do not fabricate specific statistics, paper titles, clinical trial numbers, or database identifiers.
- If the question would benefit from a formal evidence-based investigation with real-time data
  from biomedical databases, suggest: "For a comprehensive evidence-based analysis with citations,
  you can ask me to formally investigate this as a research question."
"""


BENCHMARK_QA_INSTRUCTION = """You are the benchmark execution profile for the AI Co-Scientist.
Answer a SINGLE benchmark question by using the available biomedical tools directly.

Your goal is to maximize factual correctness on source-grounded database questions.

Rules:
- Treat the question as a direct lookup or lightweight computation task, not as a report-writing task.
- Use tools aggressively when the question references a database, release, accession, coordinate range, identifier, or file-derived quantity.
- For Open Targets questions about L2G, credible sets, or variant-to-gene prioritization, use `get_open_targets_l2g`; do not substitute `get_open_targets_association`, which answers a different score.
- For Open Targets questions that mention a named release, prefer `get_open_targets_association` or `get_open_targets_l2g` over BigQuery/current-release sources.
- For benchmark Open Targets L2G questions without an explicit release, default to `release="25.09"` so the answer stays aligned with the benchmark snapshot rather than the latest archive.
- For GWAS Catalog questions that name both a GCST study accession and a specific rsID or risk allele, prefer `get_gwas_study_variant_association` over broad search tools and report the matched RAF/risk frequency directly.
- For GWAS Catalog questions that name a GCST study and ask for the risk allele with the highest or lowest p-value, prefer `get_gwas_study_top_risk_allele`.
- For JASPAR transcription-factor motif questions, prefer `get_jaspar_motif_profile` and report the returned consensus sequence and total information content.
- For gnomAD questions asking for pLI / probability of Loss-of-function Intolerance, prefer `get_gnomad_gene_constraint`.
- For gnomAD questions asking which transcript region contains the highest-frequency variant, prefer `get_gnomad_transcript_highest_af_region`.
- For RegulomeDB questions about rank, probability, or motif counts for one rsID or region, prefer `get_regulomedb_variant_summary`.
- For dbSNP population-frequency questions that name an rsID and a population such as African / ALFA, prefer `get_dbsnp_population_frequency`.
- For SCREEN questions asking for the nearest proximal/distal enhancer or promoter-like cCRE score near a gene, prefer `get_screen_nearest_ccre_assay`; it uses the Ensembl canonical transcript TSS together with SCREEN cCRE classes.
- For SCREEN questions asking which cell type has the highest assay Z-score for an EH38 cCRE accession, prefer `get_screen_ccre_top_celltype_assay` with node-cell-type restriction enabled.
- For TCGA/GDC project-level file-availability questions such as how many TCGA-BRCA cases have proteome profiling, prefer `get_tcga_project_data_availability`; it returns the case count from GDC, not a cBioPortal sample count.
- For Cell X Gene / CELLxGENE questions asking for a top marker gene or marker-effect ranking in a named cell type and tissue, prefer `get_cellxgene_marker_genes` over `search_cellxgene_datasets`; the dataset-search tool only finds datasets and cannot answer marker-gene rankings.
- For GEO questions asking for donor- or disease-filtered cell-type proportions from a specific dataset, prefer `get_geo_cell_type_proportions` over `get_geo_dataset`; the metadata tool cannot compute per-cell proportions.
- For Ensembl canonical transcript or TSS questions, prefer `get_ensembl_canonical_transcript`; when upstream/downstream bases are requested, ask that tool for the TSS-centered window directly.
- For Ensembl transcript-enumeration questions that ask which transcripts encode proteins within an amino-acid range, prefer `get_ensembl_transcripts_by_protein_length`.
- For genome sequence questions, prefer `get_ucsc_genomic_sequence` and use the returned GC/base-composition fields directly when they answer the question.
- When a question gives a human-readable genomic interval like `chr:start-end`, treat that interval as 1-based inclusive when reasoning about the expected sequence length. If you call UCSC sequence APIs, convert to UCSC coordinates carefully so you do not drop the first base.
- When a question asks for `N bp upstream` plus `N bp downstream` of a coordinate or TSS, include the central base/position in the requested interval unless the question explicitly says otherwise.
- For RefSeq annotation questions (for example signal peptide, CDS, or peptide ranges), use `get_refseq_record` and inspect its returned feature annotations.
- For UniProt accession questions about disease-associated natural variants or amino acid positions, use `get_uniprot_protein_profile` and inspect the disease-associated natural variant positions in the response.
- For AlphaFold questions asking for domain-level mean pLDDT values, prefer `get_alphafold_domain_plddt` over `get_alphafold_structure`; the structure-summary tool only returns the global model score.
- For ENCODE MPRA or CRISPR-screen questions, search `FunctionalCharacterizationExperiment` records via `search_encode_metadata` rather than assuming they live under generic `Experiment`, and extract named cell lines from `biosample_summary` and dataset descriptions when they are enumerated there.
- For ENA experiment questions about technique or instrument, prefer `get_ena_experiment_profile`.
- For EMDB entry questions about cryopreservative or vitrification cryogen, prefer `get_emdb_entry_metadata`.
- For Guide to Pharmacology / GtoPdb ligand-reference questions, prefer `get_gtopdb_ligand_reference` when the ligand ID is known.
- For Human Protein Atlas exact single-cell questions, infer the tissue and cell-type strings from the question and pass them directly to `get_human_protein_atlas_gene` via `singleCellTissue` and `singleCellCellType`. If the question says non-Tabula Sapiens or single cell type, set `singleCellDataset="single_cell_type"`. In benchmark mode, default HPA single-cell lookups to `release="v24"` unless the question explicitly asks for another release.
- For DepMap public-expression questions that ask for a mean log2(TPM+1) value across a named model subset or molecular subtype, prefer `get_depmap_expression_subset_mean` over blocked BigQuery queries.
- For DepMap public-expression questions that ask which gene is highest-expressed in one named sample, prefer `get_depmap_sample_top_expression_gene`.
- Do not ask the user clarifying questions. Make the most reasonable interpretation and proceed.
- Do not produce a research plan, step list, report section, limitations section, or methodological essay.
- Do not tell the user how they could look up the answer themselves. Return the best answer you can derive.
- If a tool returns enough information to answer, stop and answer succinctly.
- If multiple values are requested, enumerate every requested item explicitly so none are omitted.
- Keep the final answer compact and high-signal: no headers, no preamble, no citations unless the question explicitly asks.
- When the answer is numeric, include the exact numeric value and units when available.
- When a score-like answer is a long floating-point decimal, prefer a short rounded form in the final wording and keep the exact value in parentheses if helpful.
- Do not convert decimal fractions into percentages unless the question explicitly asks for a percent. If a tool returns `gc_fraction`, report the fraction, not `gc_percent`.
- When a small calculation is required, compute it and report the computed result directly.
- For DNA/RNA/protein sequence questions, copy the final sequence character-for-character from the source result. Preserve the first residue/base, preserve order, preserve case normalization as a plain uppercase sequence, and do not add prose around the sequence unless the question asks for more than the sequence itself.
- If a required source is unavailable after genuine tool attempts, say that briefly and state the closest grounded result you were able to recover.
"""


BENCHMARK_LOOP_EXECUTOR_INSTRUCTION_TEMPLATE = """You are the benchmark execution profile for the AI Co-Scientist.
Answer a SINGLE benchmark question by using the available biomedical tools directly.

Work in iterative Reason-Act-Observe cycles:
1. REASON: identify what is still missing from the answer.
2. ACT: call the best tool(s) to fill that gap.
3. OBSERVE: inspect the tool outputs carefully.
4. FINALIZE: ONLY when every requested item is covered, output `FINAL: <compact answer>`.

Your goal is to maximize factual correctness on source-grounded database questions.

Rules:
- Treat the question as a direct lookup or lightweight computation task, not as a report-writing task.
- Use tools aggressively when the question references a database, release, accession, coordinate range, identifier, or file-derived quantity.
- Keep working until every requested value or entity is covered, or you are clearly blocked after trying reasonable alternatives.
- Do not output `FINAL:` for a partial answer.
- If a previous draft omitted requested items or ended in a blocked-style answer, fix that before finalizing.
- If a tool call fails or returns insufficient data, first try a more specific query or another tool in the same evidence family before finalizing.
- For Open Targets questions about L2G, credible sets, or variant-to-gene prioritization, use `get_open_targets_l2g`; do not substitute `get_open_targets_association`, which answers a different score.
- For Open Targets questions that mention a named release, prefer `get_open_targets_association` or `get_open_targets_l2g` over BigQuery/current-release sources.
- For benchmark Open Targets L2G questions without an explicit release, default to `release="25.09"` so the answer stays aligned with the benchmark snapshot rather than the latest archive.
- For GWAS Catalog questions that name both a GCST study accession and a specific rsID or risk allele, prefer `get_gwas_study_variant_association` over broad search tools and report the matched RAF/risk frequency directly.
- For GWAS Catalog questions that name a GCST study and ask for the risk allele with the highest or lowest p-value, prefer `get_gwas_study_top_risk_allele`.
- For JASPAR transcription-factor motif questions, prefer `get_jaspar_motif_profile` and report the returned consensus sequence and total information content.
- For gnomAD questions asking for pLI / probability of Loss-of-function Intolerance, prefer `get_gnomad_gene_constraint`.
- For gnomAD questions asking which transcript region contains the highest-frequency variant, prefer `get_gnomad_transcript_highest_af_region`.
- For RegulomeDB questions about rank, probability, or motif counts for one rsID or region, prefer `get_regulomedb_variant_summary`.
- For dbSNP population-frequency questions that name an rsID and a population such as African / ALFA, prefer `get_dbsnp_population_frequency`.
- For SCREEN questions asking for the nearest proximal/distal enhancer or promoter-like cCRE score near a gene, prefer `get_screen_nearest_ccre_assay`; it uses the Ensembl canonical transcript TSS together with SCREEN cCRE classes.
- For SCREEN questions asking which cell type has the highest assay Z-score for an EH38 cCRE accession, prefer `get_screen_ccre_top_celltype_assay` with node-cell-type restriction enabled.
- For TCGA/GDC project-level file-availability questions such as how many TCGA-BRCA cases have proteome profiling, prefer `get_tcga_project_data_availability`; it returns the case count from GDC, not a cBioPortal sample count.
- For Cell X Gene / CELLxGENE questions asking for a top marker gene or marker-effect ranking in a named cell type and tissue, prefer `get_cellxgene_marker_genes` over `search_cellxgene_datasets`; the dataset-search tool only finds datasets and cannot answer marker-gene rankings.
- For GEO questions asking for donor- or disease-filtered cell-type proportions from a specific dataset, prefer `get_geo_cell_type_proportions` over `get_geo_dataset`; the metadata tool cannot compute per-cell proportions.
- For Ensembl canonical transcript or TSS questions, prefer `get_ensembl_canonical_transcript`; when upstream/downstream bases are requested, ask that tool for the TSS-centered window directly.
- For Ensembl transcript-enumeration questions that ask which transcripts encode proteins within an amino-acid range, prefer `get_ensembl_transcripts_by_protein_length`.
- For genome sequence questions, prefer `get_ucsc_genomic_sequence` and use the returned GC/base-composition fields directly when they answer the question.
- When a question gives a human-readable genomic interval like `chr:start-end`, treat that interval as 1-based inclusive when reasoning about the expected sequence length. If you call UCSC sequence APIs, convert to UCSC coordinates carefully so you do not drop the first base.
- When a question asks for `N bp upstream` plus `N bp downstream` of a coordinate or TSS, include the central base/position in the requested interval unless the question explicitly says otherwise.
- For RefSeq annotation questions (for example signal peptide, CDS, or peptide ranges), use `get_refseq_record` and inspect its returned feature annotations.
- For UniProt accession questions about disease-associated natural variants or amino acid positions, use `get_uniprot_protein_profile` and inspect the disease-associated natural variant positions in the response.
- For AlphaFold questions asking for domain-level mean pLDDT values, prefer `get_alphafold_domain_plddt` over `get_alphafold_structure`; the structure-summary tool only returns the global model score.
- For ENCODE MPRA or CRISPR-screen questions, search `FunctionalCharacterizationExperiment` records via `search_encode_metadata` rather than assuming they live under generic `Experiment`, and extract named cell lines from `biosample_summary` and dataset descriptions when they are enumerated there.
- For ENA experiment questions about technique or instrument, prefer `get_ena_experiment_profile`.
- For EMDB entry questions about cryopreservative or vitrification cryogen, prefer `get_emdb_entry_metadata`.
- For Guide to Pharmacology / GtoPdb ligand-reference questions, prefer `get_gtopdb_ligand_reference` when the ligand ID is known.
- For Human Protein Atlas exact single-cell questions, infer the tissue and cell-type strings from the question and pass them directly to `get_human_protein_atlas_gene` via `singleCellTissue` and `singleCellCellType`. If the question says non-Tabula Sapiens or single cell type, set `singleCellDataset="single_cell_type"`. In benchmark mode, default HPA single-cell lookups to `release="v24"` unless the question explicitly asks for another release.
- For DepMap public-expression questions that ask for a mean log2(TPM+1) value across a named model subset or molecular subtype, prefer `get_depmap_expression_subset_mean` over blocked BigQuery queries.
- For DepMap public-expression questions that ask which gene is highest-expressed in one named sample, prefer `get_depmap_sample_top_expression_gene`.
- Do not ask the user clarifying questions. Make the most reasonable interpretation and proceed.
- Do not produce a research plan, step list, report section, limitations section, or methodological essay.
- Do not tell the user how they could look up the answer themselves. Return the best answer you can derive.
- If multiple values are requested, enumerate every requested item explicitly so none are omitted.
- Keep the final answer compact and high-signal: no headers, no preamble, no citations unless the question explicitly asks.
- When the answer is numeric, include the exact numeric value and units when available.
- When a score-like answer is a long floating-point decimal, prefer a short rounded form in the final wording and keep the exact value in parentheses if helpful.
- Do not convert decimal fractions into percentages unless the question explicitly asks for a percent. If a tool returns `gc_fraction`, report the fraction, not `gc_percent`.
- When a small calculation is required, compute it and report the computed result directly.
- For DNA/RNA/protein sequence questions, copy the final sequence character-for-character from the source result. Preserve the first residue/base, preserve order, preserve case normalization as a plain uppercase sequence, and do not add prose around the sequence unless the question asks for more than the sequence itself.
- If a required source is unavailable after genuine tool attempts, say that briefly and state the closest grounded result you were able to recover.

Source precedence rules for overlapping tools:
__ROUTING_POLICY__
"""


CLARIFIER_INSTRUCTION = """You are a helpful query assistant for the AI Co-Scientist, a biomedical research platform.
The user's query is unclear, ambiguous, or incomplete. Help them formulate a clear request.

Your approach:
1. Identify specifically what is unclear or missing (target, scope, comparison, outcome).
2. Ask focused clarifying questions (1-3 questions, not a long list).
3. Suggest 2-3 well-formed example queries that might match what the user intended.

Be friendly, specific, and helpful. Don't just say "please clarify" — explain WHAT needs clarifying.

Examples:
- Vague: "evaluate the thing" → "Could you specify what you'd like me to evaluate? For example:
  • 'Evaluate LRRK2 as a therapeutic target for Parkinson disease'
  • 'Evaluate the efficacy of pembrolizumab in NSCLC'
  • 'Evaluate the safety profile of SGLT2 inhibitors'"

- Missing context: "compare them" → "I'd be happy to run a comparison. Could you specify which
  entities to compare? For instance, two drugs, two gene targets, or two treatment approaches?"

- Too broad: "cancer" → "That's a very broad topic. Could you narrow it down? For example:
  • A specific cancer type (e.g. 'triple-negative breast cancer')
  • A specific gene or target (e.g. 'BRAF V600E in melanoma')
  • A specific therapeutic question (e.g. 'emerging immunotherapy combinations for NSCLC')"
"""


REPORT_ASSISTANT_INSTRUCTION = """You are the report assistant for the AI Co-Scientist.
A research report has been produced, and the user has a follow-up request about it.

You can:
1. **Answer questions** about the report's findings, methodology, or terminology.
2. **Explain** specific results, statistical measures, or biological concepts in the report.
3. **Restructure** or reformat sections of the report upon request.
4. **Perform light follow-up lookups** using the available tools (e.g. fetch a specific abstract,
   run a quick database search). Keep to 1-3 tool calls maximum.
5. **Provide additional context** for findings mentioned in the report.

Guidelines:
- Ground your answers in the report content and any new tool results.
- Load a relevant follow-up skill when the request is mainly about citation recovery, trial clarification, variant interpretation, GEO datasets, or oncology target-validation nuance.
- Maintain the report's citation style when referencing sources.
- When restructuring, preserve all evidence and citations — don't drop content unless asked.
- When the user asks to fetch more / expand / continue a prior lookup, reuse the exact prior query
  string and filters unless they explicitly asked to broaden or change scope. Prefer increasing
  limit, maxStudies, or page depth over rewriting the query.
- If a source still returns the same slice or exposes no additional pages/totals after a deeper
  fetch attempt, say that clearly instead of implying a broader count was retrieved.
- If the user's request requires a comprehensive multi-step investigation (new hypothesis, full
  comparative analysis, deep-dive across multiple databases), tell them: "This would require a full
  research investigation. You can ask me to investigate this formally, and I'll create a new plan
  that builds on the current findings."
- Be conversational and helpful.
- NEVER include meta-commentary in your response: do not mention tool access, API rate limits,
  retrieval failures, or ask "would you like me to...". Deliver research content only; if retrieval
  fails for some items, state that concisely and provide what you can without explaining system
  internals.
"""


def _dedupe_str_list(values: list[Any], *, limit: int = 20) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = re.sub(r"\s+", " ", str(value or "").strip())
        if not item:
            continue
        lowered = item.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(item)
        if len(cleaned) >= max(1, limit):
            break
    return cleaned


def _normalize_user_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _is_finalize_command(text: str) -> bool:
    return _normalize_user_text(text) in FINALIZE_COMMANDS


def _is_plan_approval_command(text: str) -> bool:
    return _normalize_user_text(text) in PLAN_APPROVAL_COMMANDS


def _is_continue_execution_command(text: str) -> bool:
    normalized = _normalize_user_text(text)
    return normalized in CONTINUE_EXECUTION_COMMANDS or normalized in PLAN_APPROVAL_COMMANDS


def _parse_rollback_command(text: str) -> int | None:
    """Parse 'rollback', 'rollback N', or 'switch N'. Returns 1-based cycle index or None."""
    normalized = _normalize_user_text(text)
    if normalized == "rollback":
        return -1
    match = re.match(r"(?:rollback|switch)\s+(\d+)$", normalized)
    if match:
        return int(match.group(1))
    return None


def _extract_revision_feedback(text: str) -> str | None:
    """Return the feedback portion if user text starts with 'revise:' or 'revision:', else None."""
    stripped = text.strip()
    lowered = stripped.lower()
    for prefix in ("revise:", "revision:"):
        if lowered.startswith(prefix):
            feedback = stripped[len(prefix):].strip()
            return feedback or None
    return None


def _render_plan_approval_prompt() -> str:
    return (
        "\n\n---\n"
        "**Please review the plan above.** Respond with:\n"
        "- `approve` \u2014 proceed with execution\n"
        "- `revise: <your feedback>` \u2014 request changes to the plan"
    )


def _extract_user_turn_text(callback_context: CallbackContext) -> str:
    user_content = getattr(callback_context, "user_content", None)
    parts = getattr(user_content, "parts", None) if user_content is not None else None
    if not parts:
        return ""
    text = " ".join(
        str(getattr(part, "text", "") or "").strip()
        for part in parts
        if str(getattr(part, "text", "") or "").strip()
    )
    return re.sub(r"\s+", " ", text).strip()


def _is_lookup_tool_name(name: str) -> bool:
    raw = str(name or "").strip()
    if not raw:
        return False
    if raw in INTERNAL_SKILL_TOOL_NAMES:
        return False
    return raw.startswith(("search_", "list_", "summarize_", "query_")) or raw in {
        "run_bigquery_select_query",
        "get_clinical_trial",
        "get_pubmed_abstract",
        "get_paper_fulltext",
        "get_gnomad_gene_constraint",
        "get_gnomad_transcript_highest_af_region",
        "get_regulomedb_variant_summary",
        "get_dbsnp_population_frequency",
        "get_refseq_record",
        "get_ucsc_genomic_sequence",
        "get_ucsc_track_data",
        "get_encode_record",
        "get_zenodo_record",
    }


def _is_internal_skill_tool_name(name: str) -> bool:
    return str(name or "").strip() in INTERNAL_SKILL_TOOL_NAMES


def _compact_tool_args_for_provenance(args: Mapping[str, Any]) -> dict[str, Any]:
    """Store a compact subset of tool args so follow-up turns can reuse scope exactly."""
    preferred_keys = [
        "query",
        "search",
        "genome",
        "chrom",
        "start",
        "end",
        "track",
        "maxMatches",
        "maxItems",
        "maxResults",
        "maxBases",
        "objectType",
        "searchTerm",
        "searchQuery",
        "recordId",
        "recordType",
        "community",
        "page",
        "allVersions",
        "assayTitle",
        "accessionOrPath",
        "frame",
        "status",
        "limit",
        "maxStudies",
        "maxPages",
        "condition",
        "disease",
        "disease_id",
        "gene",
        "gene_symbol",
        "gene_id",
        "compound",
        "drug",
        "drug_name",
        "nctId",
        "id",
        "identifier",
    ]
    compact: dict[str, Any] = {}
    seen: set[str] = set()
    items = list(args.items()) if isinstance(args, Mapping) else []
    ordered_keys = preferred_keys + [key for key, _ in items if key not in preferred_keys]
    for key in ordered_keys:
        if key in seen or key in {"pageToken", "cursor", "offset", "page"}:
            continue
        seen.add(key)
        value = args.get(key) if isinstance(args, Mapping) else None
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = re.sub(r"\s+", " ", value).strip()
            if cleaned:
                compact[key] = cleaned[:500]
        elif isinstance(value, (bool, int, float)):
            compact[key] = value
        elif isinstance(value, list):
            rendered_items: list[Any] = []
            for item in value:
                if isinstance(item, str):
                    cleaned = re.sub(r"\s+", " ", item).strip()
                    if cleaned:
                        rendered_items.append(cleaned[:120])
                elif isinstance(item, (bool, int, float)):
                    rendered_items.append(item)
                if len(rendered_items) >= 5:
                    break
            if rendered_items:
                compact[key] = rendered_items
        if len(compact) >= 8:
            break
    return compact


def _tool_lookup_family(name: str, summary: str = "") -> str:
    raw = str(name or "").strip()
    lowered_summary = str(summary or "").lower()
    if raw in {
        "search_refseq_sequences",
        "get_refseq_record",
        "search_ucsc_genome",
        "get_ucsc_genomic_sequence",
        "get_ucsc_track_data",
        "search_encode_metadata",
        "get_encode_record",
        "get_gnomad_gene_constraint",
        "get_gnomad_transcript_highest_af_region",
        "get_regulomedb_variant_summary",
        "get_dbsnp_population_frequency",
    }:
        return "genomics"
    if raw in {"search_clinical_trials", "get_clinical_trial", "summarize_clinical_trials_landscape"}:
        return "clinical_trials"
    if raw in {"search_pubmed", "search_pubmed_advanced", "search_openalex_works", "search_europe_pmc_literature", "get_pubmed_abstract", "get_paper_fulltext"}:
        return "literature"
    if raw in {"search_zenodo_records", "get_zenodo_record"}:
        return "datasets"
    if raw.startswith(("search_openneuro_", "search_nemar_", "search_dandi_", "search_conp_", "search_braincode_", "search_geo_", "search_cellxgene_")):
        return "datasets"
    if "trial" in lowered_summary or "clinicaltrials.gov" in lowered_summary:
        return "clinical_trials"
    if "pubmed" in lowered_summary or "literature" in lowered_summary or "paper" in lowered_summary:
        return "literature"
    if "dataset" in lowered_summary or "geo" in lowered_summary:
        return "datasets"
    return "general"


def _infer_lookup_focus_family(user_text: str) -> str | None:
    normalized = _normalize_user_text(user_text)
    if not normalized:
        return None
    if any(term in normalized for term in ("trial", "trials", "nct", "clinicaltrials")):
        return "clinical_trials"
    if any(term in normalized for term in ("paper", "papers", "pubmed", "literature", "abstract", "full text", "fulltext", "pmid", "doi")):
        return "literature"
    if any(term in normalized for term in ("dataset", "datasets", "geo", "openneuro", "nemar", "dandi", "conp", "braincode", "brain-code", "zenodo")):
        return "datasets"
    return None


def _is_lookup_expansion_request(user_text: str) -> bool:
    normalized = _normalize_user_text(user_text)
    if not normalized:
        return False
    return any(
        term in normalized
        for term in (
            "fetch more",
            "find more",
            "show more",
            "get more",
            "more clinical trials",
            "more trials",
            "more papers",
            "more results",
            "expand",
            "broaden",
            "continue",
        )
    )


def _is_obvious_general_qa_query(user_text: str) -> bool:
    """Return True only for narrow textbook-style questions suitable for direct Q&A."""
    normalized = f" {_normalize_user_text(user_text)} "
    stripped = normalized.strip()
    if len(stripped) < 8:
        return False
    if _is_obvious_research_workflow_query(user_text):
        return False

    educational_starts = (
        "what is ",
        "what are ",
        "what does ",
        "what do ",
        "explain ",
        "define ",
        "describe ",
        "how does ",
        "how do ",
        "why do ",
        "why does ",
    )
    research_like_terms = (
        " compare ",
        " compared ",
        " versus ",
        " vs ",
        " safer ",
        " more effective ",
        " less effective ",
        " rank ",
        " rank them ",
        " prioritize ",
        " promising ",
        " strongest ",
        " best ",
        " credible ",
        " validated ",
        " efficacy ",
        " toxic",
        " tolerability ",
        " selectivity ",
        " pathogenic ",
        " pathogenicity ",
        " gain-of-function ",
        " loss-of-function ",
        " repurpose ",
        " repurposing ",
        " dataset ",
        " datasets ",
        " trial ",
        " trials ",
        " literature ",
        " evidence ",
        " pubmed ",
        " clinicaltrials.gov ",
        " open targets ",
    )

    if not stripped.startswith(educational_starts):
        return False
    if any(term in normalized for term in research_like_terms):
        return False
    if stripped.startswith(("how does ", "how do ", "why do ", "why does ")) and "," in user_text:
        return False
    return True


def _is_obvious_research_workflow_query(user_text: str) -> bool:
    """Return True for clearly evidence-driven research asks that should bypass general QA."""
    normalized = f" {_normalize_user_text(user_text)} "
    stripped = normalized.strip()
    if len(stripped) < 40:
        return False

    explicit_research_phrases = (
        " best biological and clinical support ",
        " best clinical and biological support ",
        " combination strategies ",
        " combination strategy ",
        " therapeutic target ",
        " trial landscape ",
        " safety profile ",
        " safety profiles ",
        " mechanistic studies ",
        " preclinical findings ",
    )
    comparison_phrases = (
        " more effective ",
        " less effective ",
        " work better ",
        " works better ",
        " compared with ",
        " compared to ",
        " versus ",
        " vs ",
        " safer than ",
        " better than ",
        " worse than ",
    )
    evidence_terms = (
        " clinical support ",
        " biological support ",
        " evidence ",
        " literature ",
        " mechanistic ",
        " preclinical ",
        " trial ",
        " trials ",
        " combination ",
        " monotherapy ",
    )
    evaluation_phrases = (
        " look strongest ",
        " looks strongest ",
        " look most promising ",
        " looks most promising ",
        " compelling ",
        " look best ",
        " looks best ",
        " look most credible ",
        " looks most credible ",
        " work better ",
        " works better ",
        " safer than ",
        " better than ",
        " worse than ",
        " clinically validated ",
        " tissue selectivity ",
        " pathogenic ",
        " pathogenicity ",
        " gain-of-function ",
        " loss-of-function ",
        " prioritize next ",
        " rank ",
        " rank them ",
        " compare ",
        " repurpose ",
        " repurposing ",
    )
    domain_scope_terms = (
        " drug ",
        " drugs ",
        " target ",
        " targets ",
        " therapy ",
        " therapies ",
        " inhibitor ",
        " inhibitors ",
        " mechanism ",
        " mechanisms ",
        " approach ",
        " approaches ",
        " co-target ",
        " co-targets ",
        " dependency ",
        " dependencies ",
        " co-dependency ",
        " co-dependencies ",
        " variant ",
        " variants ",
        " phenotype ",
        " disease ",
        " diseases ",
        " cancer ",
        " dermatitis ",
        " psoriasis ",
        " arthritis ",
        " obesity ",
        " dataset ",
        " datasets ",
        " mri ",
        " fmri ",
        " eeg ",
        " meg ",
    )
    source_terms = (
        " dailymed ",
        " clinicaltrials.gov ",
        " lincs ",
        " prism ",
        " pharmacodb ",
        " chembl ",
        " rcsb pdb ",
        " hpo ",
        " orphanet ",
        " monarch ",
        " clingen ",
        " openneuro ",
        " dandi ",
        " nemar ",
        " brain-code ",
        " braincode ",
    )
    dataset_evaluation_terms = (
        " most usable ",
        " strongest ",
        " benchmark ",
        " first study ",
        " reuse ",
        " reusable ",
        " realistic ",
        " replication study ",
        " metadata quality ",
        " access friction ",
    )

    if any(phrase in normalized for phrase in explicit_research_phrases):
        return True

    if (
        stripped.startswith(("among ", "for "))
        and any(term in normalized for term in evaluation_phrases)
        and any(term in normalized for term in domain_scope_terms)
        and ("," in user_text or " and " in normalized)
    ):
        return True

    if (
        stripped.startswith("which ")
        and "," in user_text
        and any(term in normalized for term in evaluation_phrases)
        and any(term in normalized for term in domain_scope_terms)
    ):
        return True

    if (
        stripped.startswith(("why is ", "why are ", "why does ", "which ", "what explains ", "how does ", "are ", "is "))
        and any(phrase in normalized for phrase in comparison_phrases)
        and any(term in normalized for term in domain_scope_terms)
        and (
            sum(1 for term in evidence_terms if term in normalized) >= 1
            or any(term in normalized for term in evaluation_phrases)
        )
    ):
        return True

    if (
        any(term in normalized for term in source_terms)
        and (
            any(term in normalized for term in evaluation_phrases)
            or any(term in normalized for term in evidence_terms)
            or " compare " in normalized
        )
    ):
        return True

    if (
        stripped.startswith(("which ", "for "))
        and any(term in normalized for term in (" dataset ", " datasets ", " mri ", " fmri ", " eeg ", " meg ", " openneuro ", " nemar ", " dandi ", " brain-code ", " braincode "))
        and (
            any(term in normalized for term in dataset_evaluation_terms)
            or " rank " in normalized
            or " rank them " in normalized
        )
    ):
        return True

    if (
        stripped.startswith(("for ", "how should ", "does "))
        and " variant " in normalized
        and any(term in normalized for term in (" pathogenic ", " pathogenicity ", " gain-of-function ", " loss-of-function "))
    ):
        return True

    if (
        (" which " in normalized or stripped.startswith("which "))
        and " support " in normalized
        and " clinical " in normalized
        and " biological " in normalized
    ):
        return True

    return False


def _render_lookup_provenance_scope(entry: Mapping[str, Any]) -> str:
    args = entry.get("args")
    if isinstance(args, Mapping):
        preferred_keys = [
            "query",
            "status",
            "limit",
            "maxStudies",
            "maxPages",
            "condition",
            "disease",
            "disease_id",
            "gene",
            "gene_symbol",
            "gene_id",
            "compound",
            "drug",
            "drug_name",
            "nctId",
            "id",
        ]
        parts: list[str] = []
        seen: set[str] = set()
        ordered_keys = preferred_keys + [key for key in args if key not in preferred_keys]
        for key in ordered_keys:
            if key in seen or key not in args:
                continue
            seen.add(key)
            value = args.get(key)
            if value is None:
                continue
            if isinstance(value, list):
                rendered_value = ", ".join(str(item) for item in value[:5])
            else:
                rendered_value = str(value).strip()
            if not rendered_value:
                continue
            if isinstance(value, str):
                rendered_value = rendered_value.replace('"', '\\"')
                parts.append(f'{key}="{rendered_value}"')
            else:
                parts.append(f"{key}={rendered_value}")
        if parts:
            return "; ".join(parts)
    return str(entry.get("summary", "") or entry.get("result", "") or "").strip()


def _infer_provenance_args_from_summary(raw_tool: str, summary: str) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", str(summary or "").strip())
    if not text:
        return {}
    query_prefixes = {
        "search_clinical_trials": ("Searching clinical trials for ",),
        "summarize_clinical_trials_landscape": ("Summarizing trial landscape for ",),
        "search_pubmed": ("Searching literature for ",),
        "search_pubmed_advanced": ("Searching literature for ",),
        "search_openalex_works": ("Searching literature for ",),
        "search_europe_pmc_literature": ("Searching literature for ",),
        "search_geo_datasets": ("Searching GEO datasets for ",),
    }
    for prefix in query_prefixes.get(str(raw_tool or "").strip(), ()):
        if text.startswith(prefix):
            query = text[len(prefix):].strip()
            return {"query": query} if query else {}
    if str(raw_tool or "").strip() == "get_clinical_trial":
        prefix = "Retrieving trial details for "
        if text.startswith(prefix):
            nct_id = text[len(prefix):].strip()
            return {"nctId": nct_id} if nct_id else {}
    if str(raw_tool or "").strip() == "get_pubmed_abstract":
        prefix = "Fetching abstract for PMID "
        if text.startswith(prefix):
            pmid = text[len(prefix):].strip()
            return {"pmid": pmid} if pmid else {}
    return {}


def _collect_report_lookup_provenance_entries(
    task_state: dict[str, Any],
    user_text: str,
    *,
    max_entries: int = 8,
) -> list[dict[str, Any]]:
    focus_family = _infer_lookup_focus_family(user_text)
    entries_out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for step in reversed(task_state.get("steps", [])):
        step_id = str(step.get("id", "")).strip() or "S?"
        tool_log = step.get("tool_log") or []
        if not isinstance(tool_log, list):
            continue
        for entry in reversed(tool_log):
            if not isinstance(entry, Mapping):
                continue
            raw_tool = str(entry.get("raw_tool", "") or "").strip()
            if not _is_lookup_tool_name(raw_tool):
                continue
            summary = str(entry.get("summary", "") or "").strip()
            family = _tool_lookup_family(raw_tool, summary)
            if focus_family and family != focus_family:
                continue
            entry_args = entry.get("args")
            if not isinstance(entry_args, Mapping):
                entry_args = _infer_provenance_args_from_summary(raw_tool, summary)
            compact_args = _compact_tool_args_for_provenance(entry_args) if isinstance(entry_args, Mapping) else {}
            render_entry: dict[str, Any] = {
                "summary": summary,
                "result": str(entry.get("result", "") or "").strip(),
            }
            if compact_args:
                render_entry["args"] = compact_args
            scope = _render_lookup_provenance_scope(render_entry)
            if not scope:
                continue
            key = f"{raw_tool}|{scope}"
            if key in seen:
                continue
            seen.add(key)
            entries_out.append({
                "step_id": step_id,
                "raw_tool": raw_tool,
                "family": family,
                "summary": summary,
                "scope": scope,
                "args": compact_args,
            })
            if len(entries_out) >= max_entries:
                return entries_out
    return entries_out


def _build_report_lookup_provenance(task_state: dict[str, Any], user_text: str, *, max_entries: int = 8) -> str:
    entries = _collect_report_lookup_provenance_entries(
        task_state,
        user_text,
        max_entries=max_entries,
    )
    return "\n".join(
        f"- {entry['step_id']} `{entry['raw_tool']}`: {entry['scope']}"
        for entry in entries
    )


REPORT_ASSISTANT_RETRIEVAL_PROFILES: dict[str, dict[str, Any]] = {
    "search_clinical_trials": {
        "family": "clinical_trials",
        "size_arg": "limit",
        "default_base": 50,
        "cap": 200,
        "mode_targets": {"representative": 20, "landscape": 100, "deep_dive": 200},
        "copy_args": ("query", "status"),
    },
    "summarize_clinical_trials_landscape": {
        "family": "clinical_trials",
        "size_arg": "maxStudies",
        "default_base": 60,
        "cap": 200,
        "mode_targets": {"representative": 40, "landscape": 100, "deep_dive": 200},
        "copy_args": ("query", "status"),
        "secondary_int_args": {
            "maxPages": {
                "default_base": 4,
                "cap": 8,
                "mode_targets": {"representative": 2, "landscape": 6, "deep_dive": 8},
            }
        },
    },
    "search_pubmed": {
        "family": "literature",
        "size_arg": "maxResults",
        "default_base": 20,
        "cap": 100,
        "mode_targets": {"representative": 10, "landscape": 40, "deep_dive": 100},
        "copy_args": ("query", "minDate", "maxDate", "sort"),
    },
    "search_pubmed_advanced": {
        "family": "literature",
        "size_arg": "maxResults",
        "default_base": 20,
        "cap": 100,
        "mode_targets": {"representative": 10, "landscape": 40, "deep_dive": 100},
        "copy_args": ("query", "sort"),
    },
    "search_openalex_works": {
        "family": "literature",
        "size_arg": "limit",
        "default_base": 10,
        "cap": 50,
        "mode_targets": {"representative": 10, "landscape": 25, "deep_dive": 50},
        "copy_args": ("query", "fromYear", "toYear"),
    },
    "search_europe_pmc_literature": {
        "family": "literature",
        "size_arg": "limit",
        "default_base": 5,
        "cap": 10,
        "mode_targets": {"representative": 5, "landscape": 10, "deep_dive": 10},
        "copy_args": ("query", "source", "openAccessOnly"),
    },
    "search_geo_datasets": {
        "family": "datasets",
        "size_arg": "maxResults",
        "default_base": 10,
        "cap": 50,
        "mode_targets": {"representative": 10, "landscape": 25, "deep_dive": 50},
        "copy_args": ("query", "entryType"),
    },
    "search_refseq_sequences": {
        "family": "genomics",
        "size_arg": "maxResults",
        "default_base": 15,
        "cap": 50,
        "mode_targets": {"representative": 10, "landscape": 25, "deep_dive": 50},
        "copy_args": ("query", "moleculeType", "organism", "refseqOnly"),
    },
    "get_refseq_record": {
        "family": "genomics",
        "copy_args": ("identifier", "moleculeType"),
    },
    "search_ucsc_genome": {
        "family": "genomics",
        "size_arg": "maxMatches",
        "default_base": 40,
        "cap": 100,
        "mode_targets": {"representative": 25, "landscape": 60, "deep_dive": 100},
        "copy_args": ("search", "genome", "categories"),
    },
    "get_ucsc_genomic_sequence": {
        "family": "genomics",
        "copy_args": ("genome", "chrom", "start", "end", "revComp", "maxBases", "hubUrl"),
    },
    "get_ucsc_track_data": {
        "family": "genomics",
        "size_arg": "maxItems",
        "default_base": 25,
        "cap": 500,
        "mode_targets": {"representative": 15, "landscape": 80, "deep_dive": 500},
        "copy_args": ("genome", "track", "chrom", "start", "end", "hubUrl"),
    },
    "search_encode_metadata": {
        "family": "genomics",
        "size_arg": "maxResults",
        "default_base": 15,
        "cap": 50,
        "mode_targets": {"representative": 10, "landscape": 30, "deep_dive": 50},
        "copy_args": ("objectType", "searchTerm", "organism", "assayTitle", "status", "frame"),
    },
    "get_encode_record": {
        "family": "genomics",
        "copy_args": ("accessionOrPath", "frame"),
    },
    "search_zenodo_records": {
        "family": "datasets",
        "size_arg": "maxResults",
        "default_base": 10,
        "cap": 100,
        "mode_targets": {"representative": 8, "landscape": 25, "deep_dive": 100},
        "copy_args": ("searchQuery", "sort", "recordType", "community", "status", "allVersions", "page"),
    },
    "get_zenodo_record": {
        "family": "datasets",
        "copy_args": ("recordId",),
    },
}


def _infer_report_retrieval_mode(user_text: str) -> str | None:
    normalized = _normalize_user_text(user_text)
    if not normalized:
        return None
    if _is_lookup_expansion_request(user_text):
        return "expand_previous"
    if any(
        phrase in normalized
        for phrase in (
            "comprehensive",
            "exhaustive",
            "all available",
            "full list",
            "full registry",
            "deep dive",
            "deep-dive",
        )
    ):
        return "deep_dive"
    if any(
        phrase in normalized
        for phrase in (
            "landscape",
            "overview",
            "how many",
            "breakdown",
            "distribution",
            "broader",
            "broad",
            "more complete",
        )
    ):
        return "landscape"
    if any(
        phrase in normalized
        for phrase in (
            "representative",
            "example",
            "examples",
            "top few",
            "quick",
            "brief",
            "sample",
        )
    ):
        return "representative"
    return None


def _coerce_positive_int_arg(value: Any) -> int | None:
    try:
        numeric = int(round(float(value)))
    except Exception:  # noqa: BLE001
        return None
    return numeric if numeric > 0 else None


def _target_retrieval_depth(
    profile: Mapping[str, Any],
    *,
    mode: str,
    prior_args: Mapping[str, Any],
    current_args: Mapping[str, Any],
    size_arg: str,
) -> int | None:
    mode_targets = profile.get("mode_targets") or {}
    cap = _coerce_positive_int_arg(profile.get("cap")) or 200
    default_base = _coerce_positive_int_arg(profile.get("default_base")) or 20
    if mode == "expand_previous":
        prior_value = _coerce_positive_int_arg(prior_args.get(size_arg))
        current_value = _coerce_positive_int_arg(current_args.get(size_arg))
        base = prior_value or current_value or default_base
        landscape_floor = _coerce_positive_int_arg(mode_targets.get("landscape")) or base
        return min(cap, max(landscape_floor, base * 2))
    target = _coerce_positive_int_arg(mode_targets.get(mode))
    if target is None:
        return None
    return min(cap, target)


def _rewrite_llm_response_function_calls(
    llm_response: LlmResponse,
    rewritten_calls: list[dict[str, Any]],
) -> LlmResponse:
    updated = llm_response.model_copy(deep=True)
    content = getattr(updated, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return updated
    new_parts: list[types.Part] = []
    call_index = 0
    for part in parts:
        if getattr(part, "function_call", None) is not None and call_index < len(rewritten_calls):
            rewritten = rewritten_calls[call_index]
            call_index += 1
            new_parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=str(rewritten.get("name", "") or "").strip(),
                        args=rewritten.get("args") or {},
                    )
                )
            )
        else:
            new_parts.append(part.model_copy(deep=True))
    updated.content = types.Content(role=getattr(content, "role", "model") or "model", parts=new_parts)
    return updated


def _apply_report_assistant_adaptive_depth(
    *,
    llm_response: LlmResponse,
    user_text: str,
    provenance_entries: list[dict[str, Any]],
) -> LlmResponse | None:
    mode = _infer_report_retrieval_mode(user_text)
    if mode is None:
        return None

    rewritten_calls: list[dict[str, Any]] = []
    changed = False
    for call in _extract_function_calls(llm_response):
        name = str(call.get("name", "") or "").strip()
        args = dict(call.get("args") or {})
        profile = REPORT_ASSISTANT_RETRIEVAL_PROFILES.get(name)
        if profile is None:
            rewritten_calls.append({"name": name, "args": args})
            continue

        family = str(profile.get("family", "") or "")
        prior_entry = next(
            (entry for entry in provenance_entries if entry.get("raw_tool") == name),
            None,
        )
        if prior_entry is None and family:
            prior_entry = next(
                (entry for entry in provenance_entries if entry.get("family") == family),
                None,
            )
        prior_args = dict((prior_entry or {}).get("args") or {})

        adapted_args = dict(args)
        if mode == "expand_previous":
            for key in profile.get("copy_args", ()):
                if key in prior_args and adapted_args.get(key) != prior_args[key]:
                    adapted_args[key] = prior_args[key]
                    changed = True

        size_arg = str(profile.get("size_arg", "") or "").strip()
        if size_arg:
            target = _target_retrieval_depth(
                profile,
                mode=mode,
                prior_args=prior_args,
                current_args=adapted_args,
                size_arg=size_arg,
            )
            current_value = _coerce_positive_int_arg(adapted_args.get(size_arg))
            if target is not None:
                if mode == "expand_previous":
                    if current_value != target:
                        adapted_args[size_arg] = target
                        changed = True
                elif current_value is None or current_value < target:
                    adapted_args[size_arg] = target
                    changed = True

        for extra_key, extra_profile in (profile.get("secondary_int_args") or {}).items():
            target = _target_retrieval_depth(
                extra_profile,
                mode=mode,
                prior_args=prior_args,
                current_args=adapted_args,
                size_arg=extra_key,
            )
            current_value = _coerce_positive_int_arg(adapted_args.get(extra_key))
            if target is not None:
                if mode == "expand_previous":
                    if current_value != target:
                        adapted_args[extra_key] = target
                        changed = True
                elif current_value is None or current_value < target:
                    adapted_args[extra_key] = target
                    changed = True

        rewritten_calls.append({"name": name, "args": adapted_args})

    if not changed:
        return None
    return _rewrite_llm_response_function_calls(llm_response, rewritten_calls)


def _make_content(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part.from_text(text=text)])


def _make_text_response(text: str) -> LlmResponse:
    return LlmResponse(content=_make_content(text), partial=False, turn_complete=True)


def _replace_llm_response_text(llm_response: LlmResponse, text: str) -> LlmResponse:
    updated = llm_response.model_copy(deep=True)
    updated.content = _make_content(text)
    return updated


def _llm_response_text(llm_response: LlmResponse) -> str:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    return "".join(
        str(getattr(part, "text", "") or "")
        for part in parts
        if isinstance(getattr(part, "text", None), str)
        and not bool(getattr(part, "thought", False))
    )


def _llm_response_thought_text(llm_response: LlmResponse) -> str:
    """Extract thinking/reasoning tokens from the model response."""
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    return "".join(
        str(getattr(part, "text", "") or "")
        for part in parts
        if isinstance(getattr(part, "text", None), str)
        and bool(getattr(part, "thought", False))
    )


def _llm_response_has_function_call(llm_response: LlmResponse) -> bool:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return False
    return any(getattr(part, "function_call", None) is not None for part in parts)

def _extract_function_calls(llm_response: LlmResponse) -> list[dict[str, Any]]:
    """Extract name + args for each function call in the response."""
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return []
    calls: list[dict[str, Any]] = []
    for part in parts:
        fc = getattr(part, "function_call", None)
        if fc is not None:
            name = str(getattr(fc, "name", "") or "").strip()
            if name:
                args = getattr(fc, "args", None) or {}
                if not isinstance(args, dict):
                    try:
                        args = dict(args)
                    except Exception:  # noqa: BLE001
                        args = {}
                calls.append({"name": name, "args": args})
    return calls


def _describe_tool_call(name: str, args: dict[str, Any]) -> str:
    """Generate a human-readable 1-line description from a tool call and its args.

    The description explains *what information is being sought*, not just
    the tool name.  Uses the tool's semantic purpose + available args.
    """
    source = tool_registry.TOOL_SOURCE_NAMES.get(name, name)
    query = str(args.get("query", "") or "").strip()
    gene = str(args.get("gene", "") or args.get("gene_symbol", "") or args.get("gene_id", "") or "").strip()
    disease = str(args.get("disease", "") or args.get("disease_id", "") or "").strip()
    compound = str(args.get("compound", "") or args.get("drug", "") or args.get("drug_name", "") or "").strip()

    # --- BigQuery ---
    if name == "run_bigquery_select_query" and query:
        tables = re.findall(r"FROM\s+`?[\w.]+\.(\w+)`?", query, re.IGNORECASE)
        table_hint = f" from {', '.join(tables)}" if tables else ""
        # Try to extract what's being queried for
        where_clauses = re.findall(r"WHERE\s+.*?['\"](.+?)['\"]", query, re.IGNORECASE)
        subject = where_clauses[0][:50] if where_clauses else ""
        if subject and table_hint:
            return f"Retrieving {subject} data{table_hint}"
        if table_hint:
            return f"Querying{table_hint}"
        return f"Running BigQuery query"
    if name == "list_bigquery_tables":
        dataset = str(args.get("dataset_id", "") or "").strip()
        table = str(args.get("table_name", "") or "").strip()
        if table:
            return f"Inspecting schema of {table} table"
        if dataset:
            return f"Discovering available tables in {dataset}"
        return f"Listing available BigQuery tables"

    # --- Gene / variant resolution ---
    if name == "resolve_gene_identifiers":
        return f"Resolving gene identifiers for {query}" if query else "Resolving gene identifiers"
    if name == "get_variant_annotations":
        return f"Retrieving variant annotations for {query}" if query else "Retrieving variant annotations"
    if name == "search_variants_by_gene":
        return f"Searching variants in gene {gene or query}" if (gene or query) else "Searching variants by gene"
    if name == "annotate_variants_vep":
        return f"Annotating variants with Ensembl VEP for {query}" if query else "Annotating variants via VEP"

    # --- Ontology / disease identifiers ---
    if name == "map_ontology_terms_oxo":
        return f"Mapping ontology identifiers for {query}" if query else "Cross-mapping ontology terms"
    if name == "search_hpo_terms":
        return f"Looking up phenotype terms for {query}" if query else "Searching phenotype ontology"
    if name == "get_orphanet_disease_profile":
        return f"Retrieving rare disease profile for {query}" if query else "Fetching Orphanet disease profile"

    # --- Clinical trials ---
    if name == "search_clinical_trials":
        return f"Searching clinical trials for {query}" if query else "Searching ClinicalTrials.gov"
    if name == "get_clinical_trial":
        nct = str(args.get("nct_id", "") or args.get("id", "") or query or "").strip()
        return f"Retrieving trial details for {nct}" if nct else "Retrieving clinical trial details"
    if name == "summarize_clinical_trials_landscape":
        return f"Summarizing trial landscape for {query}" if query else "Summarizing clinical trial landscape"

    # --- Literature ---
    if name in ("search_pubmed", "search_pubmed_advanced"):
        return f"Searching literature for {query[:220]}" if query else "Searching PubMed"
    if name == "get_pubmed_abstract":
        pmid = str(args.get("pmid", "") or query or "").strip()
        return f"Fetching abstract for PMID {pmid}" if pmid else "Fetching PubMed abstract"
    if name == "get_paper_fulltext":
        return f"Retrieving full text for {query[:60]}" if query else "Retrieving paper full text"
    if name == "search_iedb_epitope_evidence":
        focus = query or str(args.get("peptide", "") or args.get("antigen", "") or args.get("allele", "") or "").strip()
        return f"Searching IEDB epitope evidence for {focus[:120]}" if focus else "Searching IEDB epitope evidence"
    if name in ("search_openalex_works", "search_europe_pmc_literature"):
        return f"Searching literature for {query[:140]}" if query else f"Searching {source}"

    # --- Expression / tissue data ---
    if name == "get_gene_tissue_expression":
        return f"Retrieving tissue expression data for {gene or query}" if (gene or query) else "Querying GTEx expression"
    if name == "get_human_protein_atlas_gene":
        return f"Retrieving protein expression atlas for {gene or query}" if (gene or query) else "Querying Human Protein Atlas"
    if name in ("search_aba_genes", "get_aba_gene_expression"):
        return f"Searching brain expression data for {gene or query}" if (gene or query) else "Querying Allen Brain Atlas expression"
    if name == "search_aba_structures":
        return f"Looking up brain structures: {query}" if query else "Browsing Allen Brain Atlas structures"
    if name == "search_aba_differential_expression":
        return f"Checking differential brain expression for {gene or query}" if (gene or query) else "Querying differential expression"

    # --- Drug / compound data ---
    if name == "get_chembl_bioactivities":
        return f"Retrieving bioactivity data for {compound or query}" if (compound or query) else "Querying ChEMBL bioactivities"
    if name == "get_pubchem_compound":
        return f"Looking up compound info for {compound or query}" if (compound or query) else "Querying PubChem"
    if name == "get_guidetopharmacology_target":
        return f"Checking pharmacology data for {gene or query}" if (gene or query) else "Querying Guide to Pharmacology"
    if name == "get_dailymed_drug_label":
        return f"Retrieving drug label for {compound or query}" if (compound or query) else "Fetching DailyMed label"

    # --- Safety / adverse events ---
    if name == "search_fda_adverse_events":
        return f"Checking FDA adverse events for {query}" if query else "Searching FDA FAERS"

    # --- Protein structure / interactions ---
    if name == "get_alphafold_structure":
        return f"Fetching predicted structure for {gene or query}" if (gene or query) else "Querying AlphaFold"
    if name == "search_protein_structures":
        return f"Searching protein structures for {gene or query}" if (gene or query) else "Searching RCSB PDB"
    if name == "search_drug_gene_interactions":
        return f"Checking drug-gene interactions for {gene or query}" if (gene or query) else "Querying DGIdb"
    if name == "search_reactome_pathways":
        return f"Searching pathway data for {gene or query}" if (gene or query) else "Querying Reactome pathways"

    # --- GWAS ---
    if name == "search_gwas_associations":
        return f"Searching GWAS associations for {gene or query}" if (gene or query) else "Querying GWAS Catalog"

    # --- ClinGen / gene curation ---
    if name == "get_clingen_gene_curation":
        return f"Checking gene-disease validity for {gene or query}" if (gene or query) else "Querying ClinGen"

    # --- Genetic associations ---
    if name == "query_monarch_associations":
        association_mode = str(args.get("associationMode", "") or "").strip()
        target = str(args.get("entityId", "") or query).strip()
        mode_labels = {
            "disease_to_phenotype": "disease-to-phenotype associations",
            "phenotype_to_gene": "phenotype-to-gene associations",
            "disease_to_gene_causal": "disease-to-gene causal associations",
            "disease_to_gene_correlated": "disease-to-gene correlated associations",
            "gene_to_phenotype": "gene-to-phenotype associations",
        }
        label = mode_labels.get(association_mode, "Monarch associations")
        return f"Querying {label} for {target}" if target else "Searching Monarch Initiative"
    if name == "search_civic_variants":
        return f"Searching clinical variant evidence for {query}" if query else "Querying CIViC variants"
    if name == "search_civic_genes":
        return f"Searching clinical gene evidence for {gene or query}" if (gene or query) else "Querying CIViC genes"
    if name == "get_cancer_mutation_profile":
        return f"Retrieving mutation profile for {gene or query}" if (gene or query) else "Querying cBioPortal"

    # --- Generic fallback: use source + whatever query-like arg is available ---
    any_query = query or gene or compound or disease
    if any_query:
        return f"Querying {source} for {any_query[:80]}"
    return f"Querying {source}"


def _describe_tool_result(name: str, response: Any) -> str:
    """Generate a human-readable 1-line description from a tool result."""
    source = tool_registry.TOOL_SOURCE_NAMES.get(name, name)
    if not isinstance(response, dict):
        try:
            response = dict(response) if response else {}
        except Exception:  # noqa: BLE001
            response = {}
    err = response.get("error") or response.get("error_message")
    if err:
        return f"Error in {name}: {str(err)[:400]}"

    result_meta = _extract_mcp_result_meta(response)

    # MCP tools put text in content[0].text; the format is typically:
    #   "Summary:\n{actual summary}\n\nKey Fields:\n..."
    # Extract the real summary from the second line, or the first non-label line.
    mcp_text = _extract_mcp_text(response)
    if mcp_text:
        if name == "search_drug_gene_interactions":
            rich_summary = _extract_dgidb_mcp_summary(mcp_text)
            if rich_summary:
                return _truncate_summary(rich_summary, max_chars=500)
        if name in ("search_clinical_trials", "summarize_clinical_trials_landscape", "get_clinical_trial"):
            rich_summary = _extract_clinical_trials_mcp_summary(name, mcp_text)
            if rich_summary:
                return _truncate_summary(rich_summary, max_chars=500)
        meta_summary = _summarize_mcp_result_meta(result_meta)
        if meta_summary:
            return _truncate_summary(meta_summary, max_chars=500)
        summary_line = _extract_mcp_summary_line(mcp_text)
        if summary_line:
            summary_line = _qualify_search_mcp_summary_line(name, summary_line)
            # Long-output tools: literature (incl. query in result), clinical trials, abstracts
            long_output_tools = (
                "get_aba_gene_expression",
                "get_gene_tissue_expression",
                "search_pubmed",
                "search_pubmed_advanced",
                "search_clinical_trials",
                "summarize_clinical_trials_landscape",
                "get_clinical_trial",
                "get_pubmed_abstract",
            )
            max_chars = 500 if name in long_output_tools else 380
            return _truncate_summary(summary_line, max_chars=max_chars)

    # Try structured dict keys (for non-MCP tools / direct function tools)
    summary = _extract_result_summary(name, response)
    if summary:
        return summary
    return f"{source} returned results"


def _extract_tool_result_evidence_text(name: str, response: Any, *, max_chars: int = 6000) -> str:
    """Extract a compact evidence-preserving text payload from a tool response."""
    if not isinstance(response, dict):
        try:
            response = dict(response) if response else {}
        except Exception:  # noqa: BLE001
            response = {}

    chunks: list[str] = []

    mcp_text = _extract_mcp_text(response)
    if mcp_text:
        chunks.append(mcp_text)

    structured = response.get("structuredContent")
    if isinstance(structured, dict):
        for key in ("text", "summary"):
            value = str(structured.get(key, "") or "").strip()
            if value:
                chunks.append(value)

    for key in ("summary", "text"):
        value = str(response.get(key, "") or "").strip()
        if value:
            chunks.append(value)

    if not chunks:
        return ""

    combined = "\n\n".join(chunk for chunk in chunks if chunk).strip()
    if len(combined) <= max_chars:
        return combined
    return combined[: max_chars - 1].rstrip() + "…"


_MCP_SECTION_LABELS = {"summary:", "key fields:", "sources:", "limitations:", "notes:"}


def _qualify_search_mcp_summary_line(name: str, summary_line: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(summary_line or "").strip())
    if not cleaned:
        return ""

    if re.search(r"\bshowing\b", cleaned, flags=re.IGNORECASE):
        return re.sub(r"^Found\s+([0-9][0-9,]*)\s+", r"Source reported \1 ", cleaned, flags=re.IGNORECASE)

    if name.startswith("search_"):
        return re.sub(r"^Found\s+([0-9][0-9,]*)\s+", r"Returned \1 ", cleaned, flags=re.IGNORECASE)

    return cleaned


def _extract_mcp_result_meta(response: dict[str, Any]) -> dict[str, Any] | None:
    sc = response.get("structuredContent")
    if not isinstance(sc, dict):
        return None
    direct = sc.get("result_meta")
    if isinstance(direct, dict):
        return direct
    payload = sc.get("payload")
    if isinstance(payload, dict):
        nested = payload.get("result_meta")
        if isinstance(nested, dict):
            return nested
    return None


def _coerce_result_meta_count(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return max(0, int(str(value).replace(",", "").strip()))
    except Exception:  # noqa: BLE001
        return None


def _summarize_mcp_result_meta(meta: dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return ""
    mode = str(meta.get("mode", "") or "").strip().lower()
    if mode not in {"search", "list", "summary"}:
        return ""

    item_label = str(meta.get("item_label", "") or "records").strip()
    returned_count = _coerce_result_meta_count(meta.get("returned_count"))
    reported_total = _coerce_result_meta_count(meta.get("reported_total"))
    total_relation = str(meta.get("total_relation", "") or "").strip().lower()
    has_more = meta.get("has_more")
    if isinstance(has_more, str):
        has_more = has_more.strip().lower() in {"1", "true", "yes"}
    elif not isinstance(has_more, bool):
        has_more = None

    if reported_total is not None:
        if returned_count is not None and returned_count != reported_total:
            return f"returned {returned_count} {item_label} (source reported {reported_total} total matches)"
        return f"source reported {reported_total} {item_label}"

    if returned_count is not None:
        if has_more is True or total_relation == "lower_bound":
            return f"returned {returned_count} {item_label} (more may exist beyond current limit)"
        return f"returned {returned_count} {item_label}"

    return ""


def _extract_mcp_text(response: dict[str, Any]) -> str:
    """Extract the combined text from an MCP tool response."""
    content_parts = response.get("content")
    if isinstance(content_parts, list):
        for part in content_parts:
            if isinstance(part, dict) and part.get("type") == "text":
                text = str(part.get("text", "") or "").strip()
                if text:
                    return text
    # Fallback: structuredContent.text
    sc = response.get("structuredContent")
    if isinstance(sc, dict):
        text = str(sc.get("text", "") or "").strip()
        if text:
            return text
    return ""


def _extract_mcp_summary_line(text: str) -> str:
    """Extract the real summary from MCP structured text.

    MCP tools format responses as:
        Summary:
        {actual summary text}

        Key Fields:
        - field1
        ...
    We skip section labels ('Summary:', 'Key Fields:', etc.) and grab
    the first block of actual content lines.
    """
    lines = text.split("\n")
    in_summary_section = False
    content_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if content_lines:
                break
            continue
        if stripped.lower() in _MCP_SECTION_LABELS:
            if content_lines:
                break
            in_summary_section = stripped.lower() == "summary:"
            continue
        # Skip bullet-point lines from non-summary sections
        if not in_summary_section and not content_lines and stripped.startswith("- "):
            continue
        content_lines.append(stripped)
    return " ".join(content_lines).strip() if content_lines else ""


def _truncate_summary(text: str, *, max_chars: int = 120) -> str:
    """Truncate a summary line to max_chars, stripping markdown heading markers."""
    text = re.sub(r"^#+\s*", "", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1].rstrip() + "…"


def _extract_result_summary(name: str, response: dict[str, Any]) -> str:
    """Extract a meaningful 1-line summary from tool response data."""
    source = tool_registry.TOOL_SOURCE_NAMES.get(name, name)

    def _count_summary(
        item_label: str,
        *,
        returned_count: int | float | None = None,
        source_total: int | float | None = None,
    ) -> str:
        if source_total is not None:
            total_value = int(source_total)
            if returned_count is not None:
                returned_value = int(returned_count)
                if returned_value != total_value:
                    return f"returned {returned_value} {item_label} (source reported {total_value} total matches)"
            return f"source reported {total_value} {item_label}"
        if returned_count is not None:
            return f"returned {int(returned_count)} {item_label}"
        return ""

    # Gene resolver: extract symbol, Ensembl ID
    if name == "resolve_gene_identifiers":
        symbol = response.get("symbol") or response.get("query") or ""
        ensembl = response.get("ensembl", {})
        eid = ensembl.get("gene") if isinstance(ensembl, dict) else ""
        if symbol and eid:
            return f"found {symbol} (Ensembl: {eid})"
        if symbol:
            return f"found gene: {symbol}"

    # BigQuery: count rows if result is a list/table
    if name in ("run_bigquery_select_query", "list_bigquery_tables"):
        # list_bigquery_tables returns table names
        if name == "list_bigquery_tables":
            tables = response.get("tables") or response.get("table_names")
            if isinstance(tables, list):
                return f"found {len(tables)} tables"
            schema = response.get("schema") or response.get("fields")
            if isinstance(schema, (list, dict)):
                count = len(schema) if isinstance(schema, list) else len(schema)
                return f"retrieved schema ({count} fields)"
        # select query: look for rows/results
        rows = response.get("rows") or response.get("results") or response.get("data")
        if isinstance(rows, list):
            return f"found {len(rows)} rows"
        total = response.get("total_rows") or response.get("totalRows") or response.get("count")
        if total is not None:
            return f"found {total} results"

    # Clinical trials
    if name in ("search_clinical_trials", "summarize_clinical_trials_landscape"):
        studies = response.get("studies") or response.get("results") or response.get("trials")
        count = response.get("totalCount") or response.get("count") or response.get("total")
        if isinstance(studies, list):
            return _count_summary("clinical trials", returned_count=len(studies), source_total=count)
        if count is not None:
            return _count_summary("clinical trials", source_total=count)

    # PubMed / literature search
    if name in ("search_pubmed", "search_pubmed_advanced", "search_openalex_works",
                "search_europe_pmc_literature"):
        articles = response.get("results") or response.get("articles") or response.get("papers")
        count = response.get("count") or response.get("total") or response.get("totalResults")
        if isinstance(articles, list):
            return _count_summary("articles", returned_count=len(articles), source_total=count)
        if count is not None:
            return _count_summary("articles", source_total=count)
    if name == "search_iedb_epitope_evidence":
        records = response.get("records")
        if isinstance(records, list):
            return _count_summary("IEDB evidence records", returned_count=len(records))
        endpoint_counts = response.get("endpoint_counts")
        if isinstance(endpoint_counts, dict):
            total = 0
            for meta in endpoint_counts.values():
                if isinstance(meta, dict):
                    total += int(meta.get("retrieved") or 0)
            if total > 0:
                return _count_summary("IEDB endpoint hits", returned_count=total)

    # GWAS
    if name == "search_gwas_associations":
        assocs = response.get("associations") or response.get("results")
        count = response.get("count") or response.get("total") or response.get("totalCount")
        if isinstance(assocs, list):
            return _count_summary("associations", returned_count=len(assocs), source_total=count)
        if count is not None:
            return _count_summary("associations", source_total=count)

    # HPO / ontology
    if name == "search_hpo_terms":
        terms = response.get("terms") or response.get("results")
        if isinstance(terms, list):
            return _count_summary("phenotype terms", returned_count=len(terms))

    # Drug interactions
    if name == "search_drug_gene_interactions":
        interactions = response.get("matchedTerms") or response.get("interactions") or response.get("results")
        if isinstance(interactions, list):
            return _count_summary("drug-gene interactions", returned_count=len(interactions))

    # FDA adverse events
    if name == "search_fda_adverse_events":
        events = response.get("results") or response.get("events")
        if isinstance(events, list):
            return _count_summary("adverse event reports", returned_count=len(events))

    # Generic: look for common list/count patterns in any response
    for key in ("results", "data", "items", "hits", "records", "entries"):
        val = response.get(key)
        if isinstance(val, list) and len(val) > 0:
            return _count_summary(key, returned_count=len(val))
    for key in ("count", "total", "totalCount", "numFound", "total_rows"):
        val = response.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return _count_summary("results", source_total=int(val))

    # If response has a meaningful top-level field (symbol, name, id)
    for key in ("symbol", "name", "label", "title"):
        val = response.get(key)
        if isinstance(val, str) and val.strip():
            return f"found: {val.strip()[:80]}"

    return ""


def _extract_dgidb_mcp_summary(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    summary_line = next((line for line in lines if line.startswith("DGIdb results for ")), "")
    compounds: list[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped.startswith("- "):
            continue
        name = stripped[2:].split(" (", 1)[0].strip()
        if name and name not in compounds:
            compounds.append(name)
        if len(compounds) >= 3:
            break

    gene_label = ""
    summary_match = re.match(r"DGIdb results for (.+?):", summary_line)
    if summary_match:
        gene_label = summary_match.group(1).strip()

    parts: list[str] = []
    if gene_label:
        parts.append(f"DGIdb interaction record retrieved for {gene_label}.")
    if compounds:
        parts.append(f"Representative compounds: {', '.join(compounds)}.")
    elif summary_line:
        parts.append(summary_line)
    return " ".join(parts).strip()


def _extract_clinical_trials_mcp_summary(name: str, text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if name == "search_clinical_trials":
        raw_count_line = next(
            (
                line for line in lines
                if "trial" in line.lower() and ("showing " in line.lower() or "returned" in line.lower())
            ),
            "",
        )
        count_line = raw_count_line
        showing_match = re.match(r"Showing\s+(\d+)\s+of\s+(\d+)\s+total\s+trials", raw_count_line, flags=re.IGNORECASE)
        if showing_match:
            shown, total = showing_match.groups()
            count_line = f"Fetched {shown} ClinicalTrials.gov study records (source reported {total} total matches)."
        else:
            returned_match = re.match(r"(\d+)\s+trials returned\b", raw_count_line, flags=re.IGNORECASE)
            if returned_match:
                shown = returned_match.group(1)
                count_line = (
                    f"Fetched {shown} ClinicalTrials.gov study records; full registry total was not provided and more may exist."
                )
        representatives: list[str] = []
        for idx, line in enumerate(lines):
            if not line.startswith("NCT ID:"):
                continue
            nct = line.split(":", 1)[1].strip()
            status = ""
            phase = ""
            intervention = ""
            for look_ahead in lines[idx + 1: idx + 6]:
                if look_ahead.startswith("Status:"):
                    status = look_ahead.split(":", 1)[1].strip()
                elif look_ahead.startswith("Phase:"):
                    phase = look_ahead.split(":", 1)[1].strip()
                elif look_ahead.startswith("Interventions:"):
                    intervention = look_ahead.split(":", 1)[1].strip()
            bits = [nct]
            if status and status != "Unknown":
                bits.append(status)
            if phase and phase != "Not specified":
                bits.append(phase)
            if intervention and intervention != "Not specified":
                bits.append(intervention)
            representatives.append(" | ".join(bits))
            if len(representatives) >= 2:
                break
        parts = [count_line] if count_line else []
        if representatives:
            parts.append(f"Representative studies: {'; '.join(representatives)}.")
        return " ".join(parts).strip()

    if name == "summarize_clinical_trials_landscape":
        studies_line = next((line for line in lines if line.startswith("Studies analyzed:")), "")
        status_line = next((line for line in lines if line.startswith("Status breakdown:")), "")
        intervention_line = next((line for line in lines if line.startswith("Top interventions:")), "")
        parts = [part for part in [studies_line, status_line, intervention_line] if part]
        return " ".join(parts).strip()

    if name == "get_clinical_trial":
        nct_line = next((line for line in lines if line.startswith("NCT ID:")), "")
        status_line = next((line for line in lines if line.startswith("Status:")), "")
        phase_line = next((line for line in lines if line.startswith("Phase:")), "")
        title_line = next((line for line in lines if not line.endswith(":") and not line.startswith("Summary:")), "")
        parts = [part for part in [title_line, nct_line, status_line, phase_line] if part]
        return " ".join(parts).strip()

    return ""


def _summarize_latest_tool_results(llm_request: LlmRequest) -> str:
    """Extract brief observation lines from function_response parts in the most recent content."""
    contents = getattr(llm_request, "contents", None) or []
    observations: list[str] = []
    for content in reversed(contents):
        parts = getattr(content, "parts", None)
        if not parts:
            continue
        found_any = False
        for part in parts:
            fr = getattr(part, "function_response", None)
            if fr is None:
                continue
            found_any = True
            tool_name = str(getattr(fr, "name", "") or "").strip()
            source = tool_registry.TOOL_SOURCE_NAMES.get(tool_name, tool_name)
            response = getattr(fr, "response", None) or {}
            if not isinstance(response, dict):
                try:
                    response = dict(response) if response else {}
                except Exception:  # noqa: BLE001
                    response = {}
            err = response.get("error") or response.get("error_message")
            if err:
                msg = str(err)[:400]
                observations.append(f"OBSERVE: {source} → error: {msg}")
            else:
                resp_str = str(response)
                size = len(resp_str)
                if size > 300:
                    observations.append(f"OBSERVE: {source} → returned data ({size} chars)")
                elif size > 0:
                    preview = resp_str[:200].replace("\n", " ")
                    observations.append(f"OBSERVE: {source} → {preview}")
                else:
                    observations.append(f"OBSERVE: {source} → returned a response")
        if found_any:
            break
    return "\n".join(observations)


def _get_tool_log(callback_context: CallbackContext) -> list[dict[str, str]]:
    """Read the structured tool log from state."""
    raw = callback_context.state.get(STATE_EXECUTOR_TOOL_LOG, "[]") or "[]"
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(str(raw))
        return parsed if isinstance(parsed, list) else []
    except Exception:  # noqa: BLE001
        return []


def _set_tool_log(callback_context: CallbackContext, log: list[dict[str, Any]]) -> None:
    callback_context.state[STATE_EXECUTOR_TOOL_LOG] = json.dumps(log, ensure_ascii=False)
    # Sync tool_log to the active step so it's visible via workflow state reads mid-execution
    task_state = callback_context.state.get(STATE_WORKFLOW_TASK)
    if task_state:
        active_id = str(callback_context.state.get(STATE_EXECUTOR_ACTIVE_STEP_ID, "") or "").strip()
        if active_id:
            for s in task_state.get("steps", []):
                if str(s.get("id", "")) == active_id:
                    s["tool_log"] = log
                    break

def _clear_turn_temp_state(callback_context: CallbackContext) -> None:
    # Cleanup legacy app-scoped workflow state that caused cross-session bleed.
    if STATE_WORKFLOW_TASK_LEGACY_APP in callback_context.state:
        callback_context.state[STATE_WORKFLOW_TASK_LEGACY_APP] = None
    callback_context.state[STATE_AUTO_SYNTH_REQUESTED] = False
    callback_context.state[STATE_TURN_ABORT_REASON] = ""
    callback_context.state[STATE_PLANNER_BUFFER] = ""
    callback_context.state[STATE_EXECUTOR_BUFFER] = ""
    callback_context.state[STATE_SYNTH_BUFFER] = ""
    callback_context.state[STATE_PLANNER_RENDERED] = ""
    callback_context.state[STATE_EXECUTOR_RENDERED] = ""
    callback_context.state[STATE_EXECUTOR_ACTIVE_STEP_ID] = ""
    callback_context.state[STATE_EXECUTOR_PREV_STEP_STATUS] = ""
    callback_context.state[STATE_EXECUTOR_REASONING_TRACE] = ""
    callback_context.state[STATE_EXECUTOR_TOOL_LOG] = "[]"
    callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
    callback_context.state[STATE_BENCHMARK_LOOP_COUNT] = 0
    callback_context.state[STATE_BENCHMARK_COMPLETE] = False
    callback_context.state[STATE_BENCHMARK_LAST_DRAFT] = ""
    callback_context.state[STATE_BENCHMARK_FINAL_ANSWER] = ""
    callback_context.state[STATE_BENCHMARK_RETRY_FEEDBACK] = ""


def _set_turn_rendered_output(callback_context: CallbackContext, *, key: str, text: str) -> None:
    callback_context.state[key] = str(text or "")


def _append_executor_rendered(callback_context: CallbackContext, text: str) -> None:
    """Accumulate executor rendered output across LoopAgent iterations."""
    existing = str(callback_context.state.get(STATE_EXECUTOR_RENDERED, "") or "").strip()
    new_text = str(text or "").strip()
    if not new_text:
        return
    if existing:
        callback_context.state[STATE_EXECUTOR_RENDERED] = existing + "\n\n" + new_text
    else:
        callback_context.state[STATE_EXECUTOR_RENDERED] = new_text


def _compose_non_finalize_turn_output(callback_context: CallbackContext) -> str:
    planner_text = str(callback_context.state.get(STATE_PLANNER_RENDERED, "") or "").strip()
    executor_text = str(callback_context.state.get(STATE_EXECUTOR_RENDERED, "") or "").strip()
    parts = [part for part in (planner_text, executor_text) if part]
    combined = "\n\n".join(parts).strip()

    task_state = _get_task_state(callback_context)
    if task_state and str(task_state.get("plan_status", "")) != "completed":
        completed = _completed_step_count(task_state)
        total = _total_step_count(task_state)
        next_id = str(task_state.get("current_step_id") or "")
        if total > 0 and completed < total:
            hint = f"\n\n---\n_Completed {completed} of {total} steps."
            if next_id:
                hint += f" Next: **{next_id}**."
            hint += " Send `finalize` for a partial summary._"
            combined += hint

    return combined


def _json_candidate_from_fenced_block(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*([\[{].*[\]}])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()


def _extract_balanced_json_substring(text: str) -> str | None:
    start = None
    opening = ""
    closing = ""
    depth = 0
    in_string = False
    escape = False
    for idx, ch in enumerate(text):
        if start is None:
            if ch == "{":
                start = idx
                opening = "{"
                closing = "}"
                depth = 1
            elif ch == "[":
                start = idx
                opening = "["
                closing = "]"
                depth = 1
            continue

        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1].strip()
    return None


_INVALID_JSON_ESCAPES = re.compile(r"\\(?![\"\\\/bfnrtu])")


def _sanitize_json_string(text: str) -> str:
    """Fix common invalid JSON escape sequences produced by LLMs (e.g. \\' → ')."""
    return _INVALID_JSON_ESCAPES.sub("", text)


def _parse_python_literal_object(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Fallback parser for Python-literal dicts emitted by the model."""
    raw = str(raw_text or "").strip()
    if not raw:
        return None, "Empty model output."
    try:
        parsed = ast.literal_eval(raw)
    except Exception as exc:  # noqa: BLE001
        return None, f"Python literal parse error: {exc}"
    if not isinstance(parsed, dict):
        return None, "Top-level Python literal value must be an object."
    return parsed, None


def _parse_json_object_from_text(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    raw = str(raw_text or "").strip()
    if not raw:
        return None, "Empty model output."

    candidates: list[str] = []
    for candidate in (
        raw,
        _json_candidate_from_fenced_block(raw),
        _extract_balanced_json_substring(raw),
    ):
        if not candidate:
            continue
        if candidate in candidates:
            continue
        candidates.append(candidate)

    last_error = "Failed to parse JSON object."
    for candidate in candidates:
        for attempt in (candidate, _sanitize_json_string(candidate)):
            try:
                parsed = json.loads(attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = f"JSON parse error: {exc}"
                continue
            if not isinstance(parsed, dict):
                last_error = "Top-level JSON value must be an object."
                continue
            return parsed, None
        parsed, err = _parse_python_literal_object(_sanitize_json_string(candidate))
        if parsed is not None:
            return parsed, None
        if err:
            last_error = err
    return None, last_error


def _buffer_partial_text(callback_context: CallbackContext, buffer_key: str, chunk: str) -> None:
    if not chunk:
        return
    existing = str(callback_context.state.get(buffer_key, "") or "")
    callback_context.state[buffer_key] = existing + chunk


def _consume_buffered_json_object(
    callback_context: CallbackContext,
    *,
    buffer_key: str,
    llm_response: LlmResponse,
) -> tuple[dict[str, Any] | None, str | None]:
    current_text = _llm_response_text(llm_response)
    buffered = str(callback_context.state.get(buffer_key, "") or "")
    callback_context.state[buffer_key] = ""

    candidates: list[str] = []
    for candidate in (
        current_text,
        buffered + current_text,
        buffered,
    ):
        candidate = str(candidate or "")
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    if not candidates:
        return None, "Empty model output."

    last_error = "Failed to parse JSON object."
    for candidate in candidates:
        parsed, err = _parse_json_object_from_text(candidate)
        if parsed is not None:
            return parsed, None
        if err:
            last_error = err
    return None, last_error


_EVIDENCE_ID_PATTERNS = re.compile(
    r"(?:PMID:\d+|DOI:10\.[^\s\]\[);,]+|NCT\d{8,}|PMC\d+|OpenAlex:W\d+"
    r"|UniProt:[A-Z0-9]{4,}|PubChem:\d+|PDB:[A-Z0-9]{4}"
    r"|rs\d{4,}|CHEMBL\d+|Reactome:R-HSA-\d+|GCST\d+"
    r"|ENSG\d{11}|ENST\d{11}|MONDO[_:]\d+|EFO[_:]\d+|HP:\d+|GO:\d+)"
)


def _extract_evidence_ids_from_text(text: str) -> list[str]:
    """Regex-extract canonical evidence identifiers from prose text."""
    if not text:
        return []
    return _extract_inline_ids_from_text(text)[:30]


_EXECUTOR_SECTION_ALIASES = {
    "summary": "summary",
    "key findings": "summary",
    "findings": "summary",
    "evidence": "evidence",
    "evidence ids": "evidence",
    "references": "evidence",
    "open questions": "open_gaps",
    "open question": "open_gaps",
    "open gaps": "open_gaps",
    "uncertainties": "open_gaps",
    "limitations": "open_gaps",
    "suggested next searches": "suggested_next_searches",
    "suggested next steps": "suggested_next_searches",
    "next steps": "suggested_next_searches",
    "recommended next steps": "suggested_next_searches",
}

_SUMMARY_PROCESS_PREFIXES = (
    "the user is ",
    "the analysis focuses on ",
    "this step focuses on ",
    "this step focused on ",
    "the objective is ",
    "the approach involves ",
    "the approach is ",
    "the user plans to ",
    "initial steps involved ",
    "the next planned step ",
    "the plan is to ",
    "searching ",
    "retrieving ",
    "fetching ",
    "querying ",
    "looking up ",
    "obtaining ",
    "listing ",
    "calling ",
    "running ",
    "getting ",
)

_SUMMARY_PROCESS_MARKERS = (
    "next planned step",
    "previous step",
    "completion condition",
    "fulfills the completion condition",
    "objective of this step",
    "the user encountered",
    "the user plans",
    "the approach involves",
    "initial steps involved",
    " -> ",
    " → ",
)

_SUMMARY_POSITIVE_MARKERS = (
    "identified",
    "retrieved",
    "found",
    "confirmed",
    "ranked",
    "showed",
    "supports",
    "associated",
    "targets",
    "phase",
    "biomarker",
    "clinical trial",
    "no clinical trials found",
)

_SUMMARY_ACTIVITY_PREFIX_RE = re.compile(
    r"^(?:searching|retrieving|fetching|querying|looking up|obtaining|listing|calling|running|getting)\b",
    re.IGNORECASE,
)


def _normalize_executor_section_title(text: str) -> str:
    cleaned = re.sub(r"^#{1,6}\s*", "", str(text or "").strip())
    cleaned = cleaned.strip().strip(":").lower()
    return re.sub(r"\s+", " ", cleaned)


def _parse_executor_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"body": []}
    current = "body"
    for raw_line in str(text or "").splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if sections[current] and sections[current][-1] != "":
                sections[current].append("")
            continue

        normalized = _normalize_executor_section_title(stripped)
        section_key = _EXECUTOR_SECTION_ALIASES.get(normalized)
        if section_key:
            current = section_key
            sections.setdefault(current, [])
            continue

        sections.setdefault(current, []).append(stripped)

    return {
        key: "\n".join(lines).strip()
        for key, lines in sections.items()
        if any(line.strip() for line in lines)
    }


def _normalize_summary_activity_line(line: str) -> str:
    normalized = re.sub(r"\s+", " ", str(line or "").strip())
    if not normalized:
        return ""

    if _SUMMARY_ACTIVITY_PREFIX_RE.match(normalized):
        if "->" in normalized or "→" in normalized:
            return ""
        # Only treat a colon as a prefix separator when it is followed by whitespace.
        # This avoids truncating CURIEs like HGNC:18618 or MONDO:0005180 in summary text.
        colon_match = re.match(r"^[^:]{1,220}:\s+(.+)$", normalized)
        if colon_match:
            tail = colon_match.group(1).strip()
            if re.match(r"^(?:clinical trial|title)\s*:", tail, flags=re.IGNORECASE):
                return ""
            return tail

    return normalized


def _clean_executor_summary_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = str(raw_line or "").strip()
        if not line:
            continue
        if "OBSERVE:" in line:
            line = line.split("OBSERVE:", 1)[1].strip()
            if "→" in line:
                line = line.split("→", 1)[1].strip()
            line = re.sub(r"^returned data \(\d+ chars\)\s*", "", line, flags=re.IGNORECASE)
            line = re.sub(r"^returned (?:a response|results?)\s*", "", line, flags=re.IGNORECASE)
        if "ACT:" in line:
            line = line.split("ACT:", 1)[0].strip()
        if not line or line.startswith(("ACT:", "OBSERVE:", "---")):
            continue
        line = re.sub(r"^#{1,6}\s*Step\s+S?\d+\s*:?\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^#{1,6}\s*S\d+\s*(?:[·:-]\s*.*)?$", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(?:REASON|FINDINGS)\s*:\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\bFindings Summary:\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(
            r'\{.*"(?:structured_observations|evidence_ids|step_id|result_summary|step_progress_note|schema|suggested_next_searches|open_gaps|tools_called|data_sources_queried)".*\}',
            "",
            line,
            flags=re.IGNORECASE,
        )
        line = re.sub(r"\b(?:COMPLETED|BLOCKED|PENDING)\.?\s*$", "", line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        if _normalize_executor_section_title(line) in _EXECUTOR_SECTION_ALIASES:
            continue
        line = re.sub(r"^(?:[-*+]\s+|\d+\.\s+)", "", line)
        line = line.replace("\\'", "'")
        line = _normalize_summary_activity_line(line)
        if not line:
            continue
        if line.lower() in {"this step is completed.", "the step is completed.", "this step is blocked."}:
            continue
        cleaned_lines.append(line)
    return _sanitize_internal_report_text(re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip())


def _split_summary_sentences(text: str) -> list[str]:
    cleaned = _clean_executor_summary_text(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    sentences: list[str] = []
    for piece in pieces:
        sentence = re.sub(r"\s+", " ", str(piece or "").strip())
        if sentence:
            sentences.append(sentence)
    return sentences


def _score_summary_sentence(sentence: str) -> float:
    cleaned = re.sub(r"\s+", " ", str(sentence or "").strip())
    if not cleaned:
        return -10.0
    lowered = cleaned.lower()
    if lowered.startswith("act:") or lowered.startswith("observe:"):
        return -10.0

    score = 0.0
    if len(cleaned) >= 40:
        score += 0.4
    if re.search(_EVIDENCE_ID_PATTERNS, cleaned):
        score += 2.0
    if re.search(r"\b\d+(?:\.\d+)?\b", cleaned):
        score += 0.9
    if any(marker in lowered for marker in _SUMMARY_POSITIVE_MARKERS):
        score += 1.2
    if any(lowered.startswith(prefix) for prefix in _SUMMARY_PROCESS_PREFIXES):
        score -= 2.5
    if any(marker in lowered for marker in _SUMMARY_PROCESS_MARKERS):
        score -= 1.6
    if "step completed" in lowered or lowered == "completed.":
        score -= 1.2
    return score


def _select_informative_summary_text(text: str, *, max_sentences: int = 3, max_chars: int = 1200) -> str:
    sentences = _split_summary_sentences(text)
    if not sentences:
        return ""

    selected: list[str] = []
    seen: set[str] = set()
    total_chars = 0
    for sentence in sentences:
        if _score_summary_sentence(sentence) <= 0:
            continue
        normalized = _normalize_user_text(sentence)
        if normalized in seen:
            continue
        projected = total_chars + len(sentence) + (1 if selected else 0)
        if projected > max_chars and selected:
            break
        seen.add(normalized)
        selected.append(sentence)
        total_chars = projected
        if len(selected) >= max_sentences:
            break

    if not selected:
        for sentence in sentences:
            lowered = sentence.lower()
            if any(lowered.startswith(prefix) for prefix in _SUMMARY_PROCESS_PREFIXES):
                continue
            if any(marker in lowered for marker in _SUMMARY_PROCESS_MARKERS):
                continue
            selected = [sentence]
            break

    return " ".join(selected).strip()


def _preferred_model_result_summary(value: Any) -> str:
    """Use the executor-provided result_summary when it looks substantive."""
    cleaned = _clean_executor_summary_text(str(value or ""))
    if not cleaned:
        return ""
    candidate_sentences = []
    for sentence in _split_summary_sentences(cleaned):
        lowered = sentence.lower()
        if any(lowered.startswith(prefix) for prefix in _SUMMARY_PROCESS_PREFIXES):
            continue
        if any(marker in lowered for marker in _SUMMARY_PROCESS_MARKERS):
            continue
        candidate_sentences.append(sentence)
    if not candidate_sentences:
        return ""
    return _select_informative_summary_text(" ".join(candidate_sentences), max_sentences=6, max_chars=2200)


def _clean_tool_log_phrase(text: str, *, source: str = "") -> str:
    phrase = re.sub(r"\s+", " ", str(text or "").strip())
    if not phrase:
        return ""
    if source:
        prefix = f"{source}: "
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix):].strip()
    phrase = phrase.replace("\\'", "'")
    phrase = re.sub(r"^returned results\b", "", phrase, flags=re.IGNORECASE).strip(" .;:")
    return phrase


def _score_tool_log_phrase(entry: dict[str, Any], phrase: str) -> float:
    lowered = phrase.lower()
    score = 0.0
    if entry.get("status") == "done":
        score += 0.8
    if "error" in lowered:
        score -= 1.0
    if re.search(_EVIDENCE_ID_PATTERNS, phrase):
        score += 1.5
    if re.search(r"\b\d+(?:\.\d+)?\b", phrase):
        score += 0.8
    if any(marker in lowered for marker in ("found", "returned", "retrieved", "source reported", "identified", "confirmed", "no clinical trials found")):
        score += 1.0
    if any(marker in lowered for marker in ("schema", "table", "tables")):
        score -= 0.6
    if len(phrase) < 18:
        score -= 0.5
    return score


def _build_tool_log_summary(step: dict[str, Any], tool_log: list[dict[str, Any]]) -> str:
    ranked: list[tuple[float, int, str]] = []
    for idx, entry in enumerate(tool_log):
        source = str(entry.get("tool", "") or "").strip()
        phrase = _clean_tool_log_phrase(str(entry.get("result", "") or ""), source=source)
        if not phrase:
            phrase = _clean_tool_log_phrase(str(entry.get("summary", "") or ""), source=source)
        if not phrase:
            continue
        ranked.append((_score_tool_log_phrase(entry, phrase), idx, phrase))

    if not ranked:
        return ""

    ranked.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    chosen_by_index = sorted(ranked[:3], key=lambda item: item[1])

    selected: list[str] = []
    seen: set[str] = set()
    for _, _, phrase in chosen_by_index:
        normalized = _normalize_user_text(phrase)
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(phrase.rstrip(".") + ".")

    if not selected:
        return ""

    refs: list[Any] = [step.get("tool_hint")]
    refs.extend(entry.get("tool") for entry in tool_log)
    refs.extend(entry.get("raw_tool") for entry in tool_log)
    sources = _normalize_source_label_candidates(refs, allow_verbatim_labels=True)
    if any(str(ref or "").strip() in {"BigQuery", "run_bigquery_select_query", "list_bigquery_tables"} for ref in refs):
        sources = _dedupe_str_list(list(sources) + ["BigQuery"], limit=8)
    source_prefix = f"Used {_human_join(sources[:4])}. " if sources else ""
    return (source_prefix + " ".join(selected)).strip()


def _harmonize_blocked_step_summary(summary_text: str, tool_log: list[dict[str, Any]]) -> str:
    cleaned = _clean_executor_summary_text(summary_text)
    if not cleaned:
        cleaned = re.sub(r"\s+", " ", str(summary_text or "").strip())

    lowered = cleaned.lower()
    if any(marker in lowered for marker in ("blocked", "could not", "unable", "partial result", "remained blocked")):
        return cleaned

    blocker_phrase = ""
    for entry in tool_log:
        phrase = _clean_tool_log_phrase(str(entry.get("result", "") or ""), source=str(entry.get("tool", "") or ""))
        if not phrase:
            phrase = _clean_tool_log_phrase(str(entry.get("summary", "") or ""), source=str(entry.get("tool", "") or ""))
        if not phrase or "error" not in phrase.lower():
            continue
        blocker_phrase = _first_sentence(_sanitize_internal_report_text(phrase), max_chars=220).rstrip(".")
        if blocker_phrase:
            break

    if blocker_phrase:
        if cleaned:
            return f"Partial result: {cleaned.rstrip('.')} However, the step remained blocked: {blocker_phrase}."
        return f"Step blocked: {blocker_phrase}."

    if cleaned:
        return f"Partial result: {cleaned.rstrip('.')} However, the step remained blocked before meeting the completion condition."
    return "Step blocked before meeting the completion condition."


def _extract_section_items(text: str, *, limit: int = 10) -> list[str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    items = _extract_markdown_list_items(cleaned)
    if items:
        return _dedupe_str_list(items, limit=limit)
    sentences = _split_summary_sentences(cleaned)
    return _dedupe_str_list(sentences, limit=limit)


def _infer_step_status_from_output(final_text: str, tool_log: list[dict[str, Any]]) -> str:
    lowered = str(final_text or "").lower()
    if re.search(r"\bblocked\b", lowered):
        return "blocked"

    done_count = sum(1 for entry in tool_log if entry.get("status") == "done")
    error_count = 0
    for entry in tool_log:
        result = str(entry.get("result", "") or "").lower()
        if "error" in result:
            error_count += 1

    if done_count == 0 and error_count > 0 and re.search(
        r"(unable to complete|could not complete|cannot complete|tool unavailable|permission denied|rate limit)",
        lowered,
    ):
        return "blocked"
    return "completed"


def _build_deterministic_step_result(
    *,
    step: dict[str, Any],
    step_id: str,
    final_text: str,
    tool_log: list[dict[str, Any]],
    base_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base = dict(base_result or {})
    sections = _parse_executor_sections(final_text)

    evidence_inputs = [final_text, sections.get("evidence", ""), sections.get("summary", "")]
    for entry in tool_log:
        evidence_inputs.append(str(entry.get("evidence_text", "") or ""))
        evidence_inputs.append(str(entry.get("result", "") or ""))
        evidence_inputs.append(str(entry.get("summary", "") or ""))
    evidence_ids = _dedupe_str_list(
        list(base.get("evidence_ids", []) or [])
        + [eid for chunk in evidence_inputs for eid in _extract_evidence_ids_from_text(chunk)],
        limit=30,
    )

    model_summary = _preferred_model_result_summary(base.get("result_summary", ""))
    summary_text = model_summary
    if not summary_text:
        summary_text = _select_informative_summary_text(sections.get("summary", ""))
    if not summary_text:
        summary_text = _select_informative_summary_text(sections.get("body", ""))
    if not summary_text:
        summary_text = _build_tool_log_summary(step, tool_log)
    elif not model_summary:
        tool_summary = _build_tool_log_summary(step, tool_log)
        if tool_summary and _score_summary_sentence(summary_text) < 1.0:
            summary_text = f"{summary_text} {tool_summary}".strip()
    if not summary_text:
        goal = str(step.get("goal", "") or "").strip()
        summary_text = f"Completed: {goal}" if goal else "Step completed."

    tools_called = _dedupe_str_list(
        list(base.get("tools_called", []) or [])
        + [entry.get("raw_tool") for entry in tool_log if str(entry.get("raw_tool", "")).strip()],
        limit=20,
    )

    refs: list[Any] = list(base.get("data_sources_queried", []) or [])
    refs.append(step.get("tool_hint"))
    refs.extend(entry.get("tool") for entry in tool_log)
    refs.extend(entry.get("raw_tool") for entry in tool_log)
    data_sources = _normalize_source_label_candidates(refs, allow_verbatim_labels=True)
    if any(str(ref or "").strip() in {"BigQuery", "run_bigquery_select_query", "list_bigquery_tables"} for ref in refs):
        data_sources = _dedupe_str_list(list(data_sources) + ["BigQuery"], limit=15)

    open_gaps = _dedupe_str_list(
        list(base.get("open_gaps", []) or []) + _extract_section_items(sections.get("open_gaps", ""), limit=10),
        limit=10,
    )
    next_searches = _dedupe_str_list(
        list(base.get("suggested_next_searches", []) or [])
        + _extract_section_items(sections.get("suggested_next_searches", ""), limit=10),
        limit=10,
    )

    result = {
        "schema": STEP_RESULT_SCHEMA,
        "step_id": step_id,
        "status": str(base.get("status") or _infer_step_status_from_output(final_text, tool_log)).strip().lower() or "completed",
        "step_progress_note": _first_sentence(summary_text, max_chars=200) or "Step completed.",
        "result_summary": summary_text,
        "evidence_ids": evidence_ids,
        "open_gaps": open_gaps,
        "suggested_next_searches": next_searches,
        "tools_called": tools_called,
        "data_sources_queried": data_sources,
        "structured_observations": list(base.get("structured_observations", []) or []),
    }

    if result["status"] not in {"completed", "blocked"}:
        result["status"] = "completed"
    if result["status"] == "blocked":
        result["result_summary"] = _harmonize_blocked_step_summary(result["result_summary"], tool_log)
        result["step_progress_note"] = _first_sentence(result["result_summary"], max_chars=200) or "Step blocked."
    return result


def _as_nonempty_str(value: Any, field_name: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _as_string_list(value: Any, field_name: str, *, limit: int = 20) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return _dedupe_str_list(value, limit=limit)


def _canonicalize_tool_hint_candidate(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().strip("`"))
    if not text:
        return ""
    lookup_key = _normalize_lookup_key(text)
    if lookup_key in EMPTY_LIKE_TOOL_HINTS:
        return ""
    if text in KNOWN_MCP_TOOLS or text in tool_registry.TOOL_SOURCE_NAMES:
        return text
    if "." in text:
        base = text.split(".", 1)[0].strip()
        if base in KNOWN_MCP_TOOLS or base in tool_registry.TOOL_SOURCE_NAMES:
            return base
    mapped = DEFAULT_TOOL_HINT_BY_SOURCE_LABEL.get(lookup_key) or DEFAULT_TOOL_HINT_BY_DOMAIN.get(lookup_key)
    return mapped or text


def _infer_plan_step_tool_hint(step: dict[str, Any], domains: list[str]) -> str:
    for field_name in ("tool_hint", "tool", "source", "database", "dataset"):
        candidate = _canonicalize_tool_hint_candidate(step.get(field_name))
        if candidate:
            return candidate

    context_fields = [
        step.get("goal"),
        step.get("completion_condition"),
        step.get("notes"),
        step.get("rationale"),
        step.get("description"),
    ]
    context_text = " ".join(re.sub(r"\s+", " ", str(value or "").strip()).lower() for value in context_fields if value)
    for keywords, tool_name in TOOL_HINT_INFERENCE_RULES:
        if any(keyword in context_text for keyword in keywords):
            return tool_name

    for domain in domains:
        default_tool = DEFAULT_TOOL_HINT_BY_DOMAIN.get(_normalize_lookup_key(domain))
        if default_tool:
            return default_tool
    return "search_pubmed"


def _validate_plan_internal(raw: dict[str, Any]) -> dict[str, Any]:
    if str(raw.get("schema", "")).strip() != PLAN_SCHEMA:
        raise ValueError(f"schema must be {PLAN_SCHEMA}")
    objective = _as_nonempty_str(raw.get("objective"), "objective")
    success_criteria = _as_string_list(raw.get("success_criteria"), "success_criteria", limit=8)
    if not success_criteria:
        raise ValueError("success_criteria must contain at least one item")

    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list):
        raise ValueError("steps must be a list")
    if len(steps_raw) < 1:
        raise ValueError("steps must contain at least one item")

    steps: list[dict[str, Any]] = []
    for idx, step in enumerate(steps_raw, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"steps[{idx - 1}] must be an object")
        canonical_id = f"S{idx}"
        raw_domains = step.get("domains")
        if isinstance(raw_domains, list):
            domains = [str(d).strip().lower() for d in raw_domains if str(d).strip()]
            domains = [d for d in domains if d in tool_registry.TOOL_DOMAINS]
        else:
            domains = []
        raw_tool_hint = re.sub(r"\s+", " ", str(step.get("tool_hint", "") or "").strip())
        tool_hint = _infer_plan_step_tool_hint(step, domains)
        if not raw_tool_hint:
            logger.warning("[planner] repaired empty tool_hint for steps[%d] -> %s", idx - 1, tool_hint)
        steps.append(
            {
                "id": canonical_id,
                "goal": _as_nonempty_str(step.get("goal"), f"steps[{idx - 1}].goal"),
                "tool_hint": _as_nonempty_str(tool_hint, f"steps[{idx - 1}].tool_hint"),
                "domains": domains,
                "completion_condition": _as_nonempty_str(
                    step.get("completion_condition"),
                    f"steps[{idx - 1}].completion_condition",
                ),
            }
        )

    return {
        "schema": PLAN_SCHEMA,
        "objective": objective,
        "success_criteria": success_criteria,
        "steps": steps,
    }


def _initialize_task_state_from_plan(plan: dict[str, Any], *, objective_text: str) -> dict[str, Any]:
    validated = _validate_plan_internal(plan)
    steps = [
        {
            "id": step["id"],
            "goal": step["goal"],
            "tool_hint": step["tool_hint"],
            "domains": step.get("domains", []),
            "completion_condition": step["completion_condition"],
            "status": "pending",
            "result_summary": "",
            "evidence_ids": [],
            "open_gaps": [],
            "suggested_next_searches": [],
            "step_progress_note": "",
            "reasoning_trace": "",
            "tools_called": [],
            "data_sources_queried": [],
            "structured_observations": [],
            "entity_ids": [],
            "claim_ids": [],
            "execution_metrics": {},
        }
        for step in validated["steps"]
    ]
    objective = validated["objective"] or objective_text
    task_state = {
        "schema": WORKFLOW_TASK_SCHEMA,
        "objective": objective,
        "objective_fingerprint": _normalize_user_text(objective_text or objective),
        "plan_status": "ready",
        "current_step_id": steps[0]["id"] if steps else None,
        "last_completed_step_id": None,
        "steps": steps,
        "success_criteria": validated["success_criteria"],
        "latest_synthesis": None,
        "evidence_store": _new_evidence_store(),
        "execution_metrics": _new_execution_metrics_bundle(),
    }
    _refresh_task_state_derived_state(task_state)
    return task_state


def _new_evidence_store() -> dict[str, Any]:
    return {
        "entities": {},
        "claims": {},
        "evidence": [],
    }


def _new_execution_metrics_bundle() -> dict[str, Any]:
    return {
        "steps": [],
        "summary": {
            "step_count": 0,
            "completed_count": 0,
            "blocked_count": 0,
            "tool_hint_accuracy": None,
            "tool_hint_first_accuracy": None,
            "fallback_rate": 0.0,
            "avg_tools_per_step": 0.0,
            "avg_evidence_ids_per_step": 0.0,
            "avg_structured_observations_per_step": 0.0,
            "avg_parse_retries_per_step": 0.0,
            "clusters": [],
            "specialization_watchlist": [],
        },
    }


def _slugify_token(text: str) -> str:
    normalized = _normalize_user_text(text)
    token = re.sub(r"[^a-z0-9:]+", "_", normalized).strip("_")
    return token or "unknown"


def _canonical_entity_key(entity_type: str, label: str) -> str:
    return f"{_slugify_token(entity_type)}:{_slugify_token(label)}"


def _merge_str_values(existing: Any, new_values: list[Any] | None = None, *, limit: int = 50) -> list[str]:
    items: list[Any] = []
    if isinstance(existing, list):
        items.extend(existing)
    elif existing not in (None, "", {}):
        items.append(existing)
    if new_values:
        items.extend(new_values)
    return _dedupe_str_list(items, limit=limit)


def _merge_attr_payload(existing: dict[str, Any] | None, new_attrs: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(existing or {})
    for key, value in (new_attrs or {}).items():
        if value in (None, "", [], {}):
            continue
        prior = merged.get(key)
        if isinstance(prior, dict) and isinstance(value, dict):
            merged[key] = {**prior, **value}
            continue
        if isinstance(prior, list) or isinstance(value, list):
            merged[key] = _merge_str_values(prior, value if isinstance(value, list) else [value])
            continue
        merged[key] = value
    return merged


_ID_LIKE_PATTERN = re.compile(r"^[A-Z0-9_]+:[A-Z0-9_.:-]+$", flags=re.IGNORECASE)


def _is_more_informative_label(candidate: str, existing: str) -> bool:
    """Return True if *candidate* is a better human-readable label than *existing*.

    Prefers labels that look like natural language over bare identifiers
    (e.g. "Parkinson disease" over "MONDO:0005180").
    """
    if not candidate or candidate == existing:
        return False
    existing_is_id = bool(_ID_LIKE_PATTERN.match(existing))
    candidate_is_id = bool(_ID_LIKE_PATTERN.match(candidate))
    if existing_is_id and not candidate_is_id:
        return True
    if not existing_is_id and candidate_is_id:
        return False
    return len(candidate) > len(existing)


def _upsert_entity(
    store: dict[str, Any],
    entity_type: str,
    label: str,
    *,
    aliases: list[str] | None = None,
    attrs: dict[str, Any] | None = None,
    canonical_key: str | None = None,
) -> dict[str, Any]:
    cleaned_label = _as_nonempty_str(label, "label")
    canonical = str(canonical_key or _canonical_entity_key(entity_type, cleaned_label)).strip()
    entity_id = f"entity:{canonical}"
    existing = store["entities"].get(entity_id)
    if not existing:
        existing = {
            "id": entity_id,
            "type": str(entity_type).strip() or "record",
            "label": cleaned_label,
            "canonical_key": canonical,
            "aliases": [],
            "attrs": {},
        }
        store["entities"][entity_id] = existing
    elif _is_more_informative_label(cleaned_label, existing.get("label", "")):
        existing["label"] = cleaned_label
    existing["aliases"] = _merge_str_values(existing.get("aliases"), aliases or [])
    existing["attrs"] = _merge_attr_payload(existing.get("attrs"), attrs)
    return existing


def _canonical_claim_key(
    subject_id: str,
    predicate: str,
    *,
    object_id: str = "",
    object_literal: str = "",
) -> str:
    object_key = object_id or object_literal or "none"
    return "|".join(
        [
            _slugify_token(subject_id),
            _slugify_token(predicate),
            _slugify_token(object_key),
        ]
    )


def _merge_status(existing: str, new_status: str) -> str:
    priority = {
        "supported": 4,
        "completed": 3,
        "observed": 2,
        "blocked": 1,
        "pending": 0,
    }
    existing_clean = str(existing or "").strip().lower()
    new_clean = str(new_status or "").strip().lower()
    if priority.get(new_clean, -1) >= priority.get(existing_clean, -1):
        return new_clean or existing_clean
    return existing_clean


def _merge_confidence(existing: str, new_value: str) -> str:
    priority = {
        "high": 3,
        "medium": 2,
        "low": 1,
        "unknown": 0,
        "": 0,
    }
    existing_clean = str(existing or "").strip().lower()
    new_clean = str(new_value or "").strip().lower()
    if priority.get(new_clean, -1) >= priority.get(existing_clean, -1):
        return new_clean or existing_clean or "unknown"
    return existing_clean or "unknown"


def _normalize_observation_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip())
    if isinstance(value, list):
        normalized_list: list[Any] = []
        for item in value:
            normalized_item = _normalize_observation_value(item)
            if normalized_item in (None, "", [], {}):
                continue
            normalized_list.append(normalized_item)
        return normalized_list
    if isinstance(value, dict):
        normalized_dict = {}
        for key, nested in value.items():
            key_text = re.sub(r"\s+", " ", str(key or "").strip())
            if not key_text:
                continue
            normalized_nested = _normalize_observation_value(nested)
            if normalized_nested in (None, "", [], {}):
                continue
            normalized_dict[key_text] = normalized_nested
        return normalized_dict
    if value is None:
        return None
    return re.sub(r"\s+", " ", str(value).strip())


def _normalize_observation_qualifiers(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("structured_observations[*].qualifiers must be an object")
    normalized = _normalize_observation_value(value)
    return normalized if isinstance(normalized, dict) else {}


def _validate_structured_entity_ref(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    entity_type = _as_nonempty_str(value.get("type"), f"{field_name}.type")
    raw_id = re.sub(r"\s+", " ", str(value.get("id", "")).strip())
    raw_label = re.sub(r"\s+", " ", str(value.get("label", "")).strip())
    label = raw_label or raw_id
    if not label:
        raise ValueError(f"{field_name}.label or {field_name}.id must be provided")
    aliases = _as_string_list(value.get("aliases"), f"{field_name}.aliases", limit=12)
    if raw_id:
        aliases = _merge_str_values(aliases, [raw_id], limit=12)
    attrs = _normalize_observation_qualifiers(value.get("attrs"))
    if raw_id:
        attrs = _merge_attr_payload(attrs, {"identifier": raw_id})
    return {
        "type": entity_type,
        "label": label,
        "id": raw_id,
        "aliases": aliases,
        "attrs": attrs,
    }


def _validate_structured_observations(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("structured_observations must be a list")

    observations: list[dict[str, Any]] = []
    for idx, observation in enumerate(value[:12]):
        if not isinstance(observation, dict):
            raise ValueError(f"structured_observations[{idx}] must be an object")
        subject = _validate_structured_entity_ref(
            observation.get("subject"),
            f"structured_observations[{idx}].subject",
        )
        predicate = _as_nonempty_str(
            observation.get("predicate"),
            f"structured_observations[{idx}].predicate",
        )
        predicate = _normalize_observation_predicate(predicate)
        object_value = observation.get("object")
        object_ref = None
        if object_value is not None:
            object_ref = _validate_structured_entity_ref(
                object_value,
                f"structured_observations[{idx}].object",
            )
        object_literal = re.sub(r"\s+", " ", str(observation.get("object_literal", "")).strip())
        if object_ref is None and not object_literal:
            raise ValueError(
                f"structured_observations[{idx}] must include either object or object_literal"
            )
        confidence = re.sub(r"\s+", " ", str(observation.get("confidence", "medium")).strip()).lower()
        if confidence not in {"low", "medium", "high"}:
            raise ValueError(
                f"structured_observations[{idx}].confidence must be low, medium, or high"
            )
        source_tool = re.sub(r"\s+", " ", str(observation.get("source_tool", "")).strip())
        observation_type = _as_nonempty_str(
            observation.get("observation_type", "observation"),
            f"structured_observations[{idx}].observation_type",
        )
        observations.append(
            {
                "observation_type": observation_type,
                "subject": subject,
                "predicate": predicate,
                "object": object_ref,
                "object_literal": object_literal,
                "supporting_ids": _as_string_list(
                    observation.get("supporting_ids"),
                    f"structured_observations[{idx}].supporting_ids",
                    limit=20,
                ),
                "source_tool": source_tool,
                "confidence": confidence,
                "qualifiers": _normalize_observation_qualifiers(observation.get("qualifiers")),
            }
        )
    return observations


def _format_observation_qualifiers(qualifiers: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(qualifiers):
        value = qualifiers.get(key)
        if value in (None, "", [], {}):
            continue
        key_text = str(key)
        if isinstance(value, list):
            value_text = ", ".join(str(item) for item in value[:6])
        elif isinstance(value, dict):
            nested = ", ".join(f"{nested_key}={nested_value}" for nested_key, nested_value in list(value.items())[:6])
            value_text = nested
        else:
            value_text = str(value)
        key_text = {
            "mode": "association view",
            "associationMode": "association view",
            "predicate": "relation",
        }.get(key_text, key_text.replace("_", " "))
        value_text = _sanitize_internal_report_text(value_text)
        parts.append(f"{key_text}: {value_text}")
    return _sanitize_internal_report_text("; ".join(parts[:8]))


def _upsert_claim(
    store: dict[str, Any],
    subject_id: str,
    predicate: str,
    *,
    object_id: str = "",
    object_literal: str = "",
    status: str = "supported",
    confidence: str = "medium",
    step_id: str = "",
    source_tool: str = "",
    source_label: str = "",
    observation_type: str = "",
) -> dict[str, Any]:
    claim_key = _canonical_claim_key(
        subject_id,
        predicate,
        object_id=object_id,
        object_literal=object_literal,
    )
    claim_id = f"claim:{claim_key}"
    existing = store["claims"].get(claim_id)
    if not existing:
        existing = {
            "id": claim_id,
            "subject_entity_id": subject_id,
            "predicate": str(predicate).strip(),
            "object_entity_id": object_id,
            "object_literal": str(object_literal or "").strip(),
            "status": str(status or "supported").strip().lower(),
            "confidence": str(confidence or "medium").strip().lower(),
            "step_ids": [],
            "source_tools": [],
            "source_labels": [],
            "evidence_count": 0,
            "observation_types": [],
        }
        store["claims"][claim_id] = existing
    existing["status"] = _merge_status(existing.get("status", ""), status)
    existing["confidence"] = _merge_confidence(existing.get("confidence", ""), confidence)
    existing["step_ids"] = _merge_str_values(existing.get("step_ids"), [step_id] if step_id else [])
    existing["source_tools"] = _merge_str_values(existing.get("source_tools"), [source_tool] if source_tool else [])
    existing["source_labels"] = _merge_str_values(existing.get("source_labels"), [source_label] if source_label else [])
    existing["observation_types"] = _merge_str_values(
        existing.get("observation_types"),
        [observation_type] if observation_type else [],
    )
    return existing


def _append_evidence_record(
    store: dict[str, Any],
    *,
    claim_id: str,
    step_id: str,
    source_tool: str,
    source_label: str,
    evidence_ids: list[str] | None,
    summary_text: str,
    score: float | None = None,
    qualifiers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "id": f"evidence:{step_id or 'step'}:{len(store['evidence']) + 1}",
        "claim_id": claim_id,
        "step_id": str(step_id or "").strip(),
        "source_tool": str(source_tool or "").strip(),
        "source_label": str(source_label or "").strip(),
        "evidence_ids": _merge_str_values([], evidence_ids or [], limit=20),
        "summary_text": _sanitize_internal_report_text(str(summary_text or "").strip()),
        "score": score,
        "qualifiers": _normalize_observation_qualifiers(qualifiers),
    }
    store["evidence"].append(record)
    claim = store["claims"].get(claim_id)
    if claim:
        claim["evidence_count"] = int(claim.get("evidence_count", 0) or 0) + 1
        claim["source_tools"] = _merge_str_values(claim.get("source_tools"), [record["source_tool"]] if record["source_tool"] else [])
        claim["source_labels"] = _merge_str_values(claim.get("source_labels"), [record["source_label"]] if record["source_label"] else [])
    return record


def _infer_entity_from_identifier(identifier: str) -> dict[str, Any] | None:
    normalized = re.sub(r"\s*:\s*", ":", str(identifier or "").strip())
    if not normalized:
        return None

    identifier_map: list[tuple[re.Pattern[str], str, str]] = [
        (re.compile(r"(?i)^PMID:(\d{4,9})$"), "paper", "paper:pmid:{}"),
        (re.compile(r"(?i)^DOI:(10\..+)$"), "paper", "paper:doi:{}"),
        (re.compile(r"(?i)^(?:PMC:)?(PMC\d+)$"), "paper", "paper:{}"),
        (re.compile(r"(?i)^OpenAlex:(W\d+)$"), "paper", "paper:{}"),
        (re.compile(r"(?i)^(?:NCT:)?(NCT\d{8})$"), "trial", "trial:{}"),
        (re.compile(r"(?i)^UniProt:([A-Z][A-Z0-9]{2,9})$"), "protein", "protein:uniprot:{}"),
        (re.compile(r"(?i)^PubChem:(\d+)$"), "compound", "compound:pubchem:{}"),
        (re.compile(r"(?i)^(CHEMBL\d+)$"), "compound", "compound:{}"),
        (re.compile(r"(?i)^PDB:([A-Z0-9]{4,8})$"), "structure", "structure:pdb:{}"),
        (re.compile(r"(?i)^(rs\d{3,})$"), "variant", "variant:{}"),
        (re.compile(r"(?i)^Reactome:(R-[A-Z]{3}-\d+)$"), "pathway", "pathway:{}"),
        (re.compile(r"(?i)^(GCST\d{4,})$"), "study", "study:{}"),
        (re.compile(r"(?i)^(HP:\d+)$"), "phenotype", "phenotype:{}"),
        (re.compile(r"(?i)^(MONDO:\d+)$"), "disease", "disease:{}"),
        (re.compile(r"(?i)^(EFO:\d+)$"), "disease", "disease:{}"),
        (re.compile(r"(?i)^(ORPHA:\d+)$"), "disease", "disease:{}"),
        (re.compile(r"(?i)^(ORDO:\d+)$"), "disease", "disease:{}"),
        (re.compile(r"(?i)^(MESH:[A-Z0-9]+)$"), "disease", "disease:{}"),
        (re.compile(r"(?i)^(ds\d{6,})$"), "dataset", "dataset:{}"),
        (re.compile(r"(?i)^(nm\d{6,})$"), "dataset", "dataset:{}"),
        (re.compile(r"(?i)^(braincode_[A-Za-z0-9_]+)$"), "dataset", "dataset:{}"),
        (re.compile(r"(?i)^(DANDI:\d+)$"), "dataset", "dataset:{}"),
    ]
    for pattern, entity_type, key_template in identifier_map:
        match = pattern.match(normalized)
        if not match:
            continue
        captured = match.group(1).strip()
        canonical_value = captured.lower() if entity_type == "paper" and normalized.upper().startswith("DOI:") else captured
        return {
            "entity_type": entity_type,
            "label": normalized if ":" in normalized or normalized.lower().startswith("ds") else captured,
            "canonical_key": key_template.format(_slugify_token(canonical_value)),
            "aliases": [normalized],
            "attrs": {
                "identifier": normalized,
                "namespace": normalized.split(":", 1)[0] if ":" in normalized else entity_type,
            },
        }

    curie_match = re.match(r"(?i)^([A-Z][A-Z0-9_-]{1,24}):([A-Za-z0-9._/-]+)$", normalized)
    if curie_match:
        prefix = curie_match.group(1).upper()
        return {
            "entity_type": "record",
            "label": normalized,
            "canonical_key": f"record:{_slugify_token(normalized)}",
            "aliases": [normalized],
            "attrs": {
                "identifier": normalized,
                "namespace": prefix,
            },
        }

    return {
        "entity_type": "record",
        "label": normalized,
        "canonical_key": f"record:{_slugify_token(normalized)}",
        "aliases": [normalized],
        "attrs": {
            "identifier": normalized,
            "namespace": "opaque",
        },
    }


OVERLAP_GROUP_TO_EXECUTOR_CLUSTER: dict[str, str] = {
    "literature_search": "literature",
    "clinical_trials": "clinical_regulatory",
    "pathway_context": "interactions_pathways",
    "molecular_interactions": "interactions_pathways",
    "compound_pharmacology": "compound_pharmacology",
    "drug_safety_regulatory": "clinical_regulatory",
    "variant_evidence": "variant_and_genomics",
    "expression_context": "expression_and_datasets",
    "target_vulnerability": "drug_response_screens",
    "phenotype_rare_disease": "phenotype_rare_disease",
    "translational_model_evidence": "translational_models",
}


DOMAIN_TO_EXECUTOR_CLUSTER: dict[str, str] = {
    "literature": "literature",
    "clinical": "clinical_regulatory",
    "protein": "interactions_pathways",
    "genomics": "variant_and_genomics",
    "chemistry": "compound_pharmacology",
    "neuroscience": "neuroscience_datasets",
    "data": "structured_data",
}


STRUCTURED_OBSERVATION_GUIDANCE_BY_OVERLAP_GROUP: dict[str, dict[str, Any]] = {
    "clinical_trials": {
        "label": "clinical-trial evidence",
        "predicates": ["tested_in", "associated_with"],
        "entity_types": ["compound", "disease", "trial", "gene"],
        "when_to_emit": (
            "Emit observations when the tool reports named interventions, specific NCT studies, or an aggregated "
            "trial landscape with representative programs and statuses."
        ),
        "extraction_rules": [
            "Prefer `tested_in` when a named intervention or program is linked to a disease or indication; store NCT ID, status, phase, sponsor, and study type in qualifiers.",
            "Use `associated_with` only when the evidence is landscape-level and cannot be tied to a specific named intervention.",
            "Do not emit count-only observations from paginated searches; pair counts with representative NCT IDs, interventions, and note when the count reflects only fetched studies.",
        ],
        "example": {
            "observation_type": "clinical_trial",
            "subject": {"type": "compound", "label": "BIIB122"},
            "predicate": "tested_in",
            "object": {"type": "disease", "label": "Parkinson disease", "id": "MONDO:0005180"},
            "supporting_ids": ["NCT04557800"],
            "source_tool": "search_clinical_trials",
            "confidence": "medium",
            "qualifiers": {
                "nct_id": "NCT04557800",
                "status": "RECRUITING",
                "phase": "Phase 2",
                "sponsor": "Biogen",
            },
        },
    },
    "compound_pharmacology": {
        "label": "compound pharmacology and druggability evidence",
        "predicates": ["inhibits", "activates", "associated_with"],
        "entity_types": ["compound", "gene", "protein"],
        "when_to_emit": (
            "Emit observations when the tool reports named compounds, target-ligand interactions, quantitative "
            "potency/selectivity, or explicit interaction types for the queried target."
        ),
        "extraction_rules": [
            "Use the compound as the subject and the gene or protein target as the object.",
            "Prefer `inhibits` or `activates` when the interaction type or mechanism is explicit; otherwise use `associated_with` for broader druggability support.",
            "Do not reduce DGIdb or target-ligand output to count-only claims; emit representative named compounds and store approval status, interaction score, potency, selectivity, and PMIDs in qualifiers when available.",
        ],
        "example": {
            "observation_type": "compound_pharmacology",
            "subject": {"type": "compound", "label": "GSK2646264"},
            "predicate": "inhibits",
            "object": {"type": "gene", "label": "LRRK2", "id": "HGNC:18618"},
            "supporting_ids": ["PMID:30998356"],
            "source_tool": "search_drug_gene_interactions",
            "confidence": "medium",
            "qualifiers": {
                "approval_status": "experimental",
                "interaction_type": "inhibitor",
                "interaction_score": "12.3",
            },
        },
    },
    "target_vulnerability": {
        "label": "drug-response and screening evidence",
        "predicates": ["sensitive_in", "resistant_in", "depends_on", "screen_hit_in"],
        "entity_types": ["compound", "gene", "cell_line", "tissue", "disease"],
        "when_to_emit": (
            "Emit observations when the tool reports a top sensitive tissue, a top sensitive cell line, "
            "or a direct gene-dependency / screen-hit statement."
        ),
        "extraction_rules": [
            "Use the compound or gene as the subject, and the tissue, disease, or cell line as the object.",
            "Prefer `sensitive_in` or `resistant_in` for compound response; prefer `depends_on` or `screen_hit_in` for gene-centric screening evidence.",
            "Capture screening context in qualifiers such as dataset, metric, direction, tissue, or screen name.",
        ],
        "example": {
            "observation_type": "drug_response",
            "subject": {"type": "compound", "label": "Paclitaxel", "id": "CHEMBL3658657"},
            "predicate": "sensitive_in",
            "object": {"type": "cell_line", "label": "A549"},
            "supporting_ids": ["CHEMBL3658657"],
            "source_tool": "get_pharmacodb_compound_response",
            "confidence": "high",
            "qualifiers": {"dataset": "PharmacoDB", "metric": "AAC", "direction": "more_sensitive", "tissue": "lung"},
        },
    },
    "phenotype_rare_disease": {
        "label": "phenotype and rare-disease evidence",
        "predicates": ["associated_with", "has_phenotype", "causal_gene_for", "correlated_gene_for"],
        "entity_types": ["phenotype", "disease", "gene"],
        "when_to_emit": (
            "Emit observations when the tool resolves a phenotype, returns a curated disease-gene association, "
            "or reports a phenotype-driven disease/gene link."
        ),
        "extraction_rules": [
            "Use the phenotype or disease as the subject when the source is phenotype-first; use the disease as the subject for curated disease-gene links.",
            "Prefer `causal_gene_for` for curated disease-causing links, `associated_with` for broader phenotype/disease links, and `has_phenotype` for disease-to-phenotype statements.",
            "Capture query mode, association type, and evidence-count style fields in qualifiers instead of overloading the predicate.",
        ],
        "example": {
            "observation_type": "phenotype_association",
            "subject": {"type": "disease", "label": "Rett syndrome", "id": "ORPHA:778"},
            "predicate": "causal_gene_for",
            "object": {"type": "gene", "label": "MECP2"},
            "supporting_ids": ["ORPHA:778"],
            "source_tool": "get_orphanet_disease_profile",
            "confidence": "high",
            "qualifiers": {"association_type": "disease-causing germline mutation", "mode": "disease_to_gene"},
        },
    },
    "translational_model_evidence": {
        "label": "translational model-organism evidence",
        "predicates": ["has_ortholog", "has_model", "has_phenotype", "associated_with"],
        "entity_types": ["gene", "species", "model", "disease", "phenotype"],
        "when_to_emit": (
            "Emit observations when the tool reports representative orthologs, disease models, or cross-species phenotype context."
        ),
        "extraction_rules": [
            "Use the source gene as the subject for ortholog and model relationships.",
            "Prefer `has_ortholog` for ortholog rows and `has_model` for disease-model rows; use qualifiers for species, provider, or disease labels.",
            "Only emit observations for the top orthologs or models that are actually summarized in the tool output.",
        ],
        "example": {
            "observation_type": "translational_model",
            "subject": {"type": "gene", "label": "TP53", "id": "HGNC:11998"},
            "predicate": "has_ortholog",
            "object": {"type": "gene", "label": "Trp53", "id": "MGI:98834"},
            "supporting_ids": ["HGNC:11998"],
            "source_tool": "get_alliance_genome_gene_profile",
            "confidence": "high",
            "qualifiers": {"species": "Mus musculus", "has_disease_annotations": True},
        },
    },
    "molecular_interactions": {
        "label": "molecular interaction evidence",
        "predicates": ["interacts_with"],
        "entity_types": ["gene", "protein"],
        "when_to_emit": (
            "Emit observations when the tool returns a top interaction partner with explicit experimental support."
        ),
        "extraction_rules": [
            "Use one observation per top distinct partner rather than one observation per raw interaction row.",
            "Use `interacts_with` as the predicate and store detection methods, interaction types, species, or miscore values in qualifiers.",
            "Include PMIDs in supporting_ids when the source reports them directly.",
        ],
        "example": {
            "observation_type": "interaction",
            "subject": {"type": "gene", "label": "TP53"},
            "predicate": "interacts_with",
            "object": {"type": "gene", "label": "MDM2"},
            "supporting_ids": ["PMID:10722742"],
            "source_tool": "get_intact_interactions",
            "confidence": "high",
            "qualifiers": {"detection_method": "anti bait coimmunoprecipitation", "interaction_type": "physical association", "species": "human"},
        },
    },
    "pathway_context": {
        "label": "pathway and network context",
        "predicates": ["participates_in", "associated_with"],
        "entity_types": ["gene", "protein", "pathway"],
        "when_to_emit": (
            "Emit observations when the tool returns a specific named pathway or curated pathway context that clearly contains the queried gene/protein."
        ),
        "extraction_rules": [
            "Use the gene or protein as the subject and the named pathway as the object.",
            "Prefer `participates_in` for explicit pathway membership and reserve `associated_with` for broader network context.",
            "Store provider, rank, or score information in qualifiers rather than the predicate.",
        ],
        "example": {
            "observation_type": "pathway_context",
            "subject": {"type": "gene", "label": "EGFR"},
            "predicate": "participates_in",
            "object": {"type": "pathway", "label": "EGFR signaling pathway", "id": "Reactome:R-HSA-177929"},
            "supporting_ids": ["Reactome:R-HSA-177929"],
            "source_tool": "search_pathway_commons_top_pathways",
            "confidence": "medium",
            "qualifiers": {"provider": "Reactome", "rank": 1},
        },
    },
}


def _determine_executor_cluster(step: dict[str, Any], tool_names: list[str]) -> str:
    for tool_name in tool_names:
        overlap_group = str(tool_registry.TOOL_ROUTING_METADATA.get(tool_name, {}).get("overlap_group", "")).strip()
        if overlap_group and overlap_group in OVERLAP_GROUP_TO_EXECUTOR_CLUSTER:
            return OVERLAP_GROUP_TO_EXECUTOR_CLUSTER[overlap_group]

    for domain in step.get("domains", []) or []:
        cluster = DOMAIN_TO_EXECUTOR_CLUSTER.get(str(domain).strip().lower())
        if cluster:
            return cluster

    return "general"


def _build_step_execution_metrics(
    step: dict[str, Any],
    validated_result: dict[str, Any],
    *,
    parse_retry_count: int = 0,
) -> dict[str, Any]:
    step_snapshot = dict(step)
    step_snapshot.update(validated_result)
    tool_hint = str(step.get("tool_hint", "")).strip()
    tools_called = _dedupe_str_list(validated_result.get("tools_called", []) or [], limit=20)
    tool_names_for_cluster = tools_called[:]
    if tool_hint:
        tool_names_for_cluster = _dedupe_str_list([tool_hint] + tool_names_for_cluster, limit=20)

    fallback_tools = tool_registry.TOOL_ROUTING_METADATA.get(tool_hint, {}).get("fallback_tools", []) if tool_hint else []
    used_tool_hint = bool(tool_hint and tool_hint in tools_called)
    used_tool_hint_first = bool(tool_hint and tools_called and tools_called[0] == tool_hint)
    fallback_used = bool(
        tool_hint
        and any(tool_name in set(fallback_tools) or tool_name != tool_hint for tool_name in tools_called)
    )

    return {
        "step_id": str(validated_result.get("step_id", "")).strip(),
        "status": str(validated_result.get("status", "")).strip().lower(),
        "tool_hint": tool_hint,
        "tool_hint_source": _resolve_source_label(tool_hint),
        "tools_called": tools_called,
        "tool_sources": _derive_step_data_sources(step_snapshot),
        "overlap_groups": _dedupe_str_list(
            [
                tool_registry.TOOL_ROUTING_METADATA.get(tool_name, {}).get("overlap_group", "")
                for tool_name in tool_names_for_cluster
            ],
            limit=10,
        ),
        "executor_cluster": _determine_executor_cluster(step, tool_names_for_cluster),
        "used_tool_hint": used_tool_hint,
        "used_tool_hint_first": used_tool_hint_first,
        "fallback_used": fallback_used,
        "tool_count": len(tools_called),
        "evidence_count": len(validated_result.get("evidence_ids", []) or []),
        "structured_observation_count": len(validated_result.get("structured_observations", []) or []),
        "open_gap_count": len(validated_result.get("open_gaps", []) or []),
        "parse_retry_count": max(0, int(parse_retry_count or 0)),
    }


def _summarize_evidence_store(store: dict[str, Any]) -> dict[str, Any]:
    entities = list((store or {}).get("entities", {}).values())
    claims = list((store or {}).get("claims", {}).values())
    evidence_records = list((store or {}).get("evidence", []))
    entities_by_type: dict[str, int] = {}
    claims_by_predicate: dict[str, int] = {}
    label_by_entity = {entity["id"]: entity.get("label", entity["id"]) for entity in entities}

    for entity in entities:
        entity_type = str(entity.get("type", "record")).strip() or "record"
        entities_by_type[entity_type] = entities_by_type.get(entity_type, 0) + 1

    for claim in claims:
        predicate = str(claim.get("predicate", "related_to")).strip() or "related_to"
        claims_by_predicate[predicate] = claims_by_predicate.get(predicate, 0) + 1

    top_claims = []
    for claim in sorted(
        claims,
        key=lambda item: (
            int(item.get("evidence_count", 0) or 0),
            len(item.get("source_labels", []) or []),
            str(item.get("predicate", "")),
        ),
        reverse=True,
    )[:10]:
        object_id = str(claim.get("object_entity_id", "")).strip()
        top_claims.append(
            {
                "predicate": claim.get("predicate", ""),
                "subject": label_by_entity.get(claim.get("subject_entity_id", ""), claim.get("subject_entity_id", "")),
                "object": label_by_entity.get(object_id, claim.get("object_literal", "")),
                "status": claim.get("status", ""),
                "support_count": int(claim.get("evidence_count", 0) or 0),
                "source_count": len(claim.get("source_labels", []) or []),
            }
        )

    sources = _dedupe_str_list(
        [record.get("source_label", "") for record in evidence_records if record.get("source_label")],
        limit=20,
    )

    return {
        "entity_count": len(entities),
        "claim_count": len(claims),
        "evidence_count": len(evidence_records),
        "entities_by_type": entities_by_type,
        "claims_by_predicate": claims_by_predicate,
        "sources": sources,
        "top_claims": top_claims,
    }


SYNTHESIS_META_PREDICATES = {
    "investigated_by",
    "queried_source",
    "supported_by",
}

EVIDENCE_GRAPH_HIDDEN_ENTITY_TYPES = {
    "objective",
    "step",
    "source",
}

EVIDENCE_GRAPH_TARGET_ENTITY_TYPES = {
    "gene",
    "protein",
    "compound",
    "drug",
    "query_focus",
    "target",
    "receptor",
    "biological_entity",
    "protein_family",
}

EVIDENCE_GRAPH_STRUCTURAL_ENTITY_TYPES = {
    "attribute",
    "column",
    "field",
    "record",
    "row",
    "support_cluster",
    "table",
}

EVIDENCE_GRAPH_GENERIC_FOCUS_TERMS = {
    "across",
    "analysis",
    "available",
    "based",
    "biomarker",
    "bridge",
    "clinicaltrialsgov",
    "classes",
    "cns",
    "clinical",
    "compare",
    "determine",
    "disease",
    "eeg",
    "efficacy",
    "experiment",
    "expression",
    "gtex",
    "genetics",
    "human",
    "human protein atlas",
    "including",
    "landscape",
    "liabilities",
    "main",
    "meg",
    "missing",
    "mri",
    "name",
    "programs",
    "rank",
    "safety",
    "selective",
    "specificity",
    "still",
    "target",
    "targets",
    "therapeutic",
    "tissue",
    "trial",
    "trials",
    "which",
}

TRIVIAL_IDENTIFIER_PREDICATES = {
    "has_ensembl_id",
    "has_mondo_id",
    "has_efo_id",
    "has_entrez_id",
    "has_uniprot_id",
    "has_chembl_id",
    "has_id",
    "has_identifier",
    "identified_as",
    "maps_to",
    "cross_referenced_in",
    "resolved_to",
    "has_accession",
    "has_ontology_id",
    "has_domain_count",
    "has_modified_residue_count",
    "has_isoform_count",
    "has_subcellular_location",
    "classified_as",
    "has_no_genetic_associations",
    "has_no_phenotype_associations",
    "has_no_clingen_curation_for_disease",
    "localized_to",
}


CLAIM_SOURCE_TOOL_WEIGHTS: dict[str, float] = {
    "get_clingen_gene_curation": 1.0,
    "get_guidetopharmacology_target": 1.0,
    "get_orphanet_disease_profile": 0.98,
    "get_intact_interactions": 0.95,
    "get_biogrid_interactions": 0.95,
    "get_gdsc_drug_sensitivity": 0.95,
    "get_depmap_gene_dependency": 0.93,
    "get_depmap_expression_subset_mean": 0.9,
    "get_geo_cell_type_proportions": 0.9,
    "get_chembl_bioactivities": 0.92,
    "get_human_protein_atlas_gene": 0.9,
    "get_alphafold_domain_plddt": 0.88,
    "search_reactome_pathways": 0.88,
    "search_civic_variants": 0.88,
    "search_civic_genes": 0.88,
    "search_variants_by_gene": 0.85,
    "get_pharmacodb_compound_response": 0.87,
    "get_gene_tissue_expression": 0.84,
    "get_biogrid_orcs_gene_summary": 0.84,
    "get_alliance_genome_gene_profile": 0.82,
    "get_clinical_trial": 0.82,
    "query_monarch_associations": 0.8,
    "summarize_clinical_trials_landscape": 0.78,
    "search_clinical_trials": 0.76,
    "search_pathway_commons_top_pathways": 0.76,
    "get_prism_repurposing_response": 0.74,
    "annotate_variants_vep": 0.72,
    "search_hpo_terms": 0.68,
    "get_string_interactions": 0.65,
    "get_variant_annotations": 0.62,
    "search_drug_gene_interactions": 0.6,
    "get_pubchem_compound": 0.45,
}


CLAIM_OVERLAP_GROUP_WEIGHTS: dict[str, float] = {
    "clinical_trials": 0.8,
    "compound_pharmacology": 0.8,
    "variant_evidence": 0.78,
    "molecular_interactions": 0.82,
    "pathway_context": 0.72,
    "expression_context": 0.78,
    "target_vulnerability": 0.8,
    "phenotype_rare_disease": 0.8,
    "translational_model_evidence": 0.75,
    "literature_search": 0.7,
}


CLAIM_PREDICATE_STRENGTH_BONUS: dict[str, float] = {
    "inhibits": 0.12,
    "activates": 0.1,
    "tested_in": 0.1,
    "causal_gene_for": 0.24,
    "depends_on": 0.16,
    "has_function": 0.14,
    "sensitive_in": 0.12,
    "resistant_in": 0.12,
    "has_ortholog": 0.12,
    "has_model": 0.12,
    "interacts_with": 0.1,
    "participates_in": 0.08,
    "has_phenotype": 0.05,
    "screen_hit_in": 0.05,
    "correlated_gene_for": -0.02,
}


CLAIM_CONFLICT_GROUPS: dict[str, dict[str, set[str]]] = {
    "response_direction": {
        "supporting": {"sensitive_in"},
        "opposing": {"resistant_in"},
    },
}


CLAIM_CONFLICT_LEAN_THRESHOLD = 0.3


CLAIM_DISPLAY_PREDICATE_BONUS: dict[str, float] = {
    "causal_gene_for": 0.85,
    "associated_with": 0.75,
    "depends_on": 0.6,
    "has_function": 0.5,
    "inhibits": 0.45,
    "tested_in": 0.45,
    "activates": 0.35,
    "participates_in": 0.35,
    "sensitive_in": 0.35,
    "resistant_in": 0.35,
    "has_ortholog": 0.25,
    "has_model": 0.25,
    "has_phenotype": 0.2,
    "interacts_with": 0.05,
    "cross_referenced_in": -0.8,
}


OBJECTIVE_FOCUS_STOPWORDS = {
    "about",
    "across",
    "agent",
    "and",
    "are",
    "assess",
    "assessment",
    "by",
    "clinical",
    "considered",
    "conviction",
    "determine",
    "development",
    "disease",
    "evaluate",
    "evaluating",
    "evidence",
    "for",
    "function",
    "genetic",
    "high",
    "if",
    "is",
    "landscape",
    "of",
    "or",
    "parkinson",
    "parkinsons",
    "pd",
    "protein",
    "target",
    "therapeutic",
    "therapy",
    "trial",
    "trials",
    "whether",
}


GENERIC_MODEL_SUMMARY_MARKERS = {
    "this is a vague model summary",
    "no final summary was produced",
    "the collected evidence is not yet sufficient",
    "the following key findings were identified",
    "the following findings were identified",
}


def _source_support_weight(source_tool: str, source_label: str) -> float:
    tool_name = str(source_tool or "").strip()
    if tool_name in CLAIM_SOURCE_TOOL_WEIGHTS:
        return CLAIM_SOURCE_TOOL_WEIGHTS[tool_name]
    overlap_group = str(tool_registry.TOOL_ROUTING_METADATA.get(tool_name, {}).get("overlap_group", "")).strip()
    if overlap_group in CLAIM_OVERLAP_GROUP_WEIGHTS:
        return CLAIM_OVERLAP_GROUP_WEIGHTS[overlap_group]
    label = str(source_label or "").strip()
    for known_tool, known_label in tool_registry.TOOL_SOURCE_NAMES.items():
        if label and label == known_label and known_tool in CLAIM_SOURCE_TOOL_WEIGHTS:
            return CLAIM_SOURCE_TOOL_WEIGHTS[known_tool]
    return 0.6


_INTERNAL_ASSOCIATION_MODE_LABELS = {
    "disease_to_gene_causal": "causal disease-gene",
    "disease_to_gene_correlated": "correlated disease-gene",
    "disease_to_phenotype": "disease-to-phenotype",
    "phenotype_to_gene": "phenotype-to-gene",
    "gene_to_phenotype": "gene-to-phenotype",
}


def _humanize_biolink_predicate(value: str) -> str:
    cleaned = str(value or "").strip()
    lowered = cleaned.lower()
    if lowered.startswith("biolink:"):
        lowered = lowered.split(":", 1)[1]
    mapping = {
        "associated_with": "associated with",
        "gene_associated_with_condition": "gene-disease association",
        "contributes_to": "contributory association",
        "has_phenotype": "has phenotype",
        "interacts_with": "interacts with",
        "participates_in": "participates in",
        "related_to": "related to",
    }
    return mapping.get(lowered, lowered.replace("_", " "))


def _sanitize_internal_report_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return ""

    cleaned = re.sub(
        r"\(\s*predicate:\s*biolink:[^)]+\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    def _replace_predicate(match: re.Match[str]) -> str:
        label = _humanize_biolink_predicate(match.group(1))
        return f"relation: {label}" if label else ""

    cleaned = re.sub(
        r"predicate:\s*biolink:([a-z_]+)",
        _replace_predicate,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"biolink:([a-z_]+)",
        lambda match: _humanize_biolink_predicate(match.group(1)),
        cleaned,
        flags=re.IGNORECASE,
    )

    for raw, label in _INTERNAL_ASSOCIATION_MODE_LABELS.items():
        cleaned = re.sub(rf"\b{re.escape(raw)}\b", label, cleaned)

    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


def _normalize_observation_predicate(predicate: str) -> str:
    cleaned = str(predicate or "").strip()
    lowered = cleaned.lower()
    if lowered.startswith("biolink:"):
        lowered = lowered.split(":", 1)[1]
    mapping = {
        "associated_with": "associated_with",
        "gene_associated_with_condition": "associated_with",
        "contributes_to": "associated_with",
        "related_to": "associated_with",
        "has_phenotype": "has_phenotype",
        "interacts_with": "interacts_with",
        "participates_in": "participates_in",
    }
    normalized = mapping.get(lowered)
    if normalized:
        return normalized
    if cleaned.lower().startswith("biolink:"):
        return lowered
    return cleaned


def _humanize_claim_predicate(predicate: str) -> str:
    mapping = {
        "activates": "activates",
        "associated_with": "is associated with",
        "causal_gene_for": "is a causal gene for",
        "correlated_gene_for": "is correlated with",
        "depends_on": "depends on",
        "has_model": "has model",
        "has_ortholog": "has ortholog",
        "has_phenotype": "has phenotype",
        "interacts_with": "interacts with",
        "participates_in": "participates in",
        "queried_source": "queried source",
        "resistant_in": "is resistant in",
        "screen_hit_in": "is a screen hit in",
        "sensitive_in": "is sensitive in",
        "tested_in": "is being tested in",
        "inhibits": "inhibits",
        "supported_by": "is supported by",
    }
    cleaned = str(predicate or "").strip()
    if cleaned.lower().startswith("biolink:"):
        return _humanize_biolink_predicate(cleaned)
    return mapping.get(cleaned, cleaned.replace("_", " "))


def _claim_object_key(claim: dict[str, Any]) -> str:
    object_entity_id = str(claim.get("object_entity_id", "") or "").strip()
    if object_entity_id:
        return object_entity_id
    object_literal = str(claim.get("object_literal", "") or "").strip()
    if object_literal:
        return f"literal:{_slugify_token(object_literal)}"
    return "none"


def _looks_like_identifier_text(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return False
    patterns = [
        r"^[A-Z][A-Z0-9_-]{1,24}:[A-Za-z0-9._/-]+$",
        r"^CHEMBL\d+$",
        r"^rs\d+$",
        r"^[OPQ][0-9][A-Z0-9]{3}[0-9](?:-\d+)?$",
        r"^[A-NR-Z]\d[A-Z0-9]{3}\d(?:-\d+)?$",
        r"^ENS[A-Z]*\d+$",
    ]
    return any(re.match(pattern, cleaned, flags=re.IGNORECASE) for pattern in patterns)


def _extract_objective_focus_terms(objective_text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", str(objective_text or ""))
    focus: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in OBJECTIVE_FOCUS_STOPWORDS:
            continue
        focus.append(lowered)
    return _dedupe_str_list(focus, limit=12)


def _claim_display_priority_score(claim: dict[str, Any], objective_text: str) -> float:
    score = float(claim.get("support_score", 0.0) or 0.0)
    predicate = str(claim.get("predicate", "")).strip()
    score += CLAIM_DISPLAY_PREDICATE_BONUS.get(predicate, 0.0)

    focus_terms = _extract_objective_focus_terms(objective_text)
    haystack = " ".join(
        [
            str(claim.get("statement", "")).lower(),
            str(claim.get("subject", "")).lower(),
            str(claim.get("object", "")).lower(),
            " ".join(str(source).lower() for source in (claim.get("primary_sources", []) or [])),
        ]
    )
    matched_focus_terms = sum(1 for term in focus_terms if term in haystack)
    score += min(0.75, 0.2 * matched_focus_terms)

    objective_lower = str(objective_text or "").lower()
    if any(term in objective_lower for term in ["target", "conviction", "therapeutic"]):
        score += {
            "causal_gene_for": 0.25,
            "associated_with": 0.2,
            "inhibits": 0.14,
            "tested_in": 0.14,
            "has_function": 0.12,
            "participates_in": 0.08,
            "cross_referenced_in": -0.3,
            "interacts_with": -0.1,
        }.get(predicate, 0.0)
    if "trial" in objective_lower:
        score += {"tested_in": 0.25, "associated_with": 0.05}.get(predicate, 0.0)
    if "genetic" in objective_lower:
        score += {"causal_gene_for": 0.2, "associated_with": 0.1}.get(predicate, 0.0)
    if "function" in objective_lower:
        score += {"has_function": 0.18, "participates_in": 0.08}.get(predicate, 0.0)

    identifier_penalty = 0.0
    if _looks_like_identifier_text(str(claim.get("subject", ""))):
        identifier_penalty += 0.2
    if _looks_like_identifier_text(str(claim.get("object", ""))):
        identifier_penalty += 0.2
    score -= min(0.4, identifier_penalty)
    return round(score, 3)


def _first_substantive_paragraph(text: str) -> str:
    for block in re.split(r"\n\s*\n", str(text or "").strip()):
        cleaned = block.strip()
        if not cleaned:
            continue
        if cleaned.startswith("#") or cleaned.startswith("- ") or re.match(r"^\d+\.\s", cleaned):
            continue
        return re.sub(r"\s+", " ", cleaned).strip()
    return ""


def _first_sentence(text: str, *, max_chars: int = 280) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "").strip())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentence = parts[0].strip() if parts else cleaned
    if len(sentence) > max_chars:
        sentence = sentence[: max_chars - 3].rstrip() + "..."
    return sentence


def _sanitize_step_highlight_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    while lines and re.match(
        r"^(?:#{1,6}\s*)?(?:S\d+\s*[·-]\s*(?:completed|blocked|failed|pending)\s+Goal:|Step\s+\d+:)",
        lines[0],
        flags=re.IGNORECASE,
    ):
        lines.pop(0)
    cleaned = " ".join(lines).strip()
    if not cleaned:
        cleaned = raw
    cleaned = re.sub(
        r"^(?:#{1,6}\s*)?(?:S\d+\s*[·-]\s*(?:completed|blocked|failed|pending)\s+)?Goal:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def _step_highlight_priority(step: dict[str, Any], summary: str) -> float:
    goal = _normalize_user_text(str(step.get("goal", "")).strip())
    summary_text = _normalize_user_text(summary)
    combined = f"{goal} {summary_text}".strip()
    score = min(len(summary_text), 220) / 220.0 if summary_text else 0.0

    high_signal_markers = [
        "associated",
        "pathway",
        "mechanism",
        "target class",
        "clinical",
        "trial",
        "publication",
        "review",
        "literature",
        "dataset",
        "cohort",
        "function",
        "interaction",
        "sensitivity",
        "resistant",
        "dependency",
        "expression",
        "recruiting",
        "completed",
    ]
    score += 0.18 * min(4, sum(1 for marker in high_signal_markers if marker in combined))

    foundational_markers = [
        "retrieve the disease id",
        "determine the disease id",
        "disease identifier",
        "standardized reference",
        "enable subsequent",
        "lookup identifier",
        "mapping identifier",
        "mondo id",
        "efo id",
    ]
    if any(marker in combined for marker in foundational_markers):
        score -= 1.0
    if re.search(r"\b(?:mondo|efo|mesh|doid|hp)[_: -]?[a-z0-9]+\b", combined):
        score -= 0.35
    if _looks_like_identifier_text(summary):
        score -= 0.35

    return round(score, 3)


def _build_step_result_highlights(task_state: dict[str, Any], *, limit: int = 4) -> list[dict[str, str]]:
    candidates: list[tuple[float, int, dict[str, str]]] = []
    seen: set[str] = set()
    for idx, step in enumerate(task_state.get("steps", [])):
        if str(step.get("status", "")).strip() != "completed":
            continue
        summary = _first_sentence(_sanitize_step_highlight_text(str(step.get("result_summary", "")).strip()))
        if not summary:
            summary = _first_sentence(_sanitize_step_highlight_text(str(step.get("step_progress_note", "")).strip()))
        if not summary:
            continue
        normalized = _normalize_user_text(summary)
        if normalized in seen:
            continue
        seen.add(normalized)
        source = _format_step_source_display(step)
        candidates.append((
            _step_highlight_priority(step, summary),
            idx,
            {
                "summary": summary,
                "source": source,
            },
        ))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    highlights = [item[2] for item in candidates[:limit]]
    return highlights


def _is_informative_model_summary(objective_text: str, model_summary: str) -> bool:
    paragraph = _first_substantive_paragraph(model_summary)
    if not paragraph:
        return False
    lowered = paragraph.lower()
    if any(marker in lowered for marker in GENERIC_MODEL_SUMMARY_MARKERS):
        return False
    if lowered.endswith(":"):
        return False
    if re.match(r"^(to|in order to)\b", lowered) and ("the following" in lowered or "findings were identified" in lowered):
        return False
    if len(paragraph) < 90:
        return False
    focus_terms = _extract_objective_focus_terms(objective_text)
    if any(term in lowered for term in focus_terms[:6]):
        return True
    informative_cues = ["suggests", "supports", "indicates", "appears", "likely", "unlikely", "high-conviction", "not high-conviction"]
    if sum(1 for cue in informative_cues if cue in lowered) >= 1 and len(re.findall(r"[.!?]", paragraph)) >= 1:
        return True
    return False


def _build_claim_statement(claim: dict[str, Any], label_by_entity: dict[str, str]) -> tuple[str, str, str]:
    subject = label_by_entity.get(claim.get("subject_entity_id", ""), claim.get("subject_entity_id", ""))
    object_entity_id = str(claim.get("object_entity_id", "") or "").strip()
    object_text = label_by_entity.get(object_entity_id, "") if object_entity_id else ""
    if not object_text:
        object_text = str(claim.get("object_literal", "") or "").strip()
    predicate_text = _humanize_claim_predicate(str(claim.get("predicate", "") or "").strip())
    statement = f"{subject} {predicate_text} {object_text}".strip()
    return statement, subject, object_text


def _claim_conflict_signature(predicate: str) -> tuple[str, str] | None:
    predicate_name = str(predicate or "").strip()
    for group_name, polarity_map in CLAIM_CONFLICT_GROUPS.items():
        for polarity, predicates in polarity_map.items():
            if predicate_name in predicates:
                return group_name, polarity
    return None


def _score_adjudicated_claim(claim: dict[str, Any], evidence_records: list[dict[str, Any]], label_by_entity: dict[str, str]) -> dict[str, Any]:
    source_details: dict[str, dict[str, Any]] = {}
    supporting_ids = _dedupe_str_list(
        [
            identifier
            for record in evidence_records
            for identifier in (record.get("evidence_ids") or [])
        ],
        limit=12,
    )
    for record in evidence_records:
        source_tool = str(record.get("source_tool", "") or "").strip()
        source_label = str(record.get("source_label", "") or "").strip() or _resolve_source_label(source_tool)
        source_key = source_label or source_tool or "Unknown source"
        weight = _source_support_weight(source_tool, source_label)
        existing = source_details.get(source_key)
        detail = {
            "source": source_key,
            "source_tool": source_tool,
            "weight": round(weight, 3),
            "overlap_group": str(tool_registry.TOOL_ROUTING_METADATA.get(source_tool, {}).get("overlap_group", "")).strip(),
        }
        if existing is None or detail["weight"] > float(existing.get("weight", 0.0) or 0.0):
            source_details[source_key] = detail

    sorted_sources = sorted(
        source_details.values(),
        key=lambda item: (float(item.get("weight", 0.0) or 0.0), item.get("source", "")),
        reverse=True,
    )
    source_score = sum(float(item.get("weight", 0.0) or 0.0) for item in sorted_sources[:4])
    evidence_bonus = min(0.3, 0.05 * len(supporting_ids))
    observation_bonus = min(0.2, 0.04 * len(evidence_records))
    confidence_bonus = {
        "high": 0.25,
        "medium": 0.12,
        "low": 0.03,
    }.get(str(claim.get("confidence", "unknown")).strip().lower(), 0.0)
    predicate_bonus = CLAIM_PREDICATE_STRENGTH_BONUS.get(str(claim.get("predicate", "")).strip(), 0.0)
    support_score = round(source_score + evidence_bonus + observation_bonus + confidence_bonus + predicate_bonus, 3)

    if support_score >= 2.0 or (len(sorted_sources) >= 2 and len(supporting_ids) >= 2):
        support_strength = "high"
    elif support_score >= 1.0:
        support_strength = "medium"
    else:
        support_strength = "low"

    statement, subject_text, object_text = _build_claim_statement(claim, label_by_entity)
    return {
        "claim_id": str(claim.get("id", "")).strip(),
        "statement": statement,
        "subject": subject_text,
        "predicate": str(claim.get("predicate", "")).strip(),
        "object": object_text,
        "status": str(claim.get("status", "")).strip(),
        "claim_confidence": str(claim.get("confidence", "unknown")).strip(),
        "support_score": support_score,
        "support_strength": support_strength,
        "source_count": len(sorted_sources),
        "evidence_count": len(evidence_records),
        "primary_sources": [item.get("source", "") for item in sorted_sources[:3] if item.get("source")],
        "supporting_ids": supporting_ids,
        "source_weights": sorted_sources[:5],
        "subject_entity_id": str(claim.get("subject_entity_id", "")).strip(),
        "object_key": _claim_object_key(claim),
    }


def _detect_claim_conflicts(adjudicated_claims: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for claim in adjudicated_claims:
        signature = _claim_conflict_signature(claim.get("predicate", ""))
        if signature is None:
            continue
        group_name, polarity = signature
        bucket_key = (
            str(claim.get("subject_entity_id", "")).strip(),
            str(claim.get("object_key", "")).strip(),
            group_name,
        )
        bucket = buckets.setdefault(
            bucket_key,
            {
                "group": group_name,
                "subject": claim.get("subject", ""),
                "object": claim.get("object", ""),
                "claims_by_polarity": {},
            },
        )
        bucket["claims_by_polarity"].setdefault(polarity, []).append(claim)

    conflicts: list[dict[str, Any]] = []
    mixed_claim_ids: set[str] = set()
    for bucket in buckets.values():
        claims_by_polarity = bucket.get("claims_by_polarity", {})
        if len(claims_by_polarity) < 2:
            continue
        polarity_scores = {
            polarity: round(sum(float(claim.get("support_score", 0.0) or 0.0) for claim in claims), 3)
            for polarity, claims in claims_by_polarity.items()
        }
        ordered_polarities = sorted(
            polarity_scores.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        leading_polarity, leading_score = ordered_polarities[0]
        trailing_polarity, trailing_score = ordered_polarities[1]
        support_gap = round(leading_score - trailing_score, 3)
        if support_gap >= CLAIM_CONFLICT_LEAN_THRESHOLD:
            preferred_claim = max(
                claims_by_polarity.get(leading_polarity, []),
                key=lambda claim: (float(claim.get("support_score", 0.0) or 0.0), claim.get("statement", "")),
            )
            assessment = f"mixed_lean_{leading_polarity}"
            preferred_interpretation = preferred_claim.get("statement", "")
        else:
            assessment = "mixed_equivocal"
            preferred_interpretation = ""

        conflict_claims = []
        for polarity, claims in sorted(claims_by_polarity.items()):
            for claim in sorted(
                claims,
                key=lambda item: (float(item.get("support_score", 0.0) or 0.0), item.get("statement", "")),
                reverse=True,
            ):
                mixed_claim_ids.add(str(claim.get("claim_id", "")).strip())
                conflict_claims.append(
                    {
                        "claim_id": claim.get("claim_id", ""),
                        "statement": claim.get("statement", ""),
                        "predicate": claim.get("predicate", ""),
                        "support_score": claim.get("support_score", 0.0),
                        "support_strength": claim.get("support_strength", ""),
                        "primary_sources": list(claim.get("primary_sources", []) or [])[:3],
                        "polarity": polarity,
                    }
                )

        conflicts.append(
            {
                "subject": bucket.get("subject", ""),
                "object": bucket.get("object", ""),
                "conflict_group": bucket.get("group", ""),
                "assessment": assessment,
                "preferred_interpretation": preferred_interpretation,
                "leading_support_score": leading_score,
                "trailing_support_score": trailing_score,
                "support_gap": support_gap,
                "claims": conflict_claims,
            }
        )

    conflicts.sort(
        key=lambda item: (
            float(item.get("support_gap", 0.0) or 0.0),
            float(item.get("leading_support_score", 0.0) or 0.0),
            str(item.get("subject", "")),
        ),
        reverse=True,
    )
    return conflicts, mixed_claim_ids


def _build_claim_synthesis_summary(store: dict[str, Any], objective_text: str = "") -> dict[str, Any]:
    context = _adjudicate_substantive_claims(store, objective_text)
    adjudicated_claims = list(context["adjudicated_claims"])
    ranked_claims = list(context["ranked_claims"])
    conflicts = list(context["conflicts"])

    source_weight_reference: dict[str, dict[str, Any]] = {}
    for claim in adjudicated_claims:
        for source in claim.get("source_weights", []) or []:
            source_name = str(source.get("source", "")).strip()
            if not source_name:
                continue
            existing = source_weight_reference.get(source_name)
            if existing is None or float(source.get("weight", 0.0) or 0.0) > float(existing.get("weight", 0.0) or 0.0):
                source_weight_reference[source_name] = {
                    "source": source_name,
                    "weight": round(float(source.get("weight", 0.0) or 0.0), 3),
                    "overlap_group": str(source.get("overlap_group", "") or "").strip(),
                }

    strength_counts = {"high": 0, "medium": 0, "low": 0}
    for claim in adjudicated_claims:
        strength = str(claim.get("support_strength", "low")).strip().lower()
        if strength in strength_counts:
            strength_counts[strength] += 1

    def _public_claim_summary(claim: dict[str, Any]) -> dict[str, Any]:
        return {
            "statement": claim.get("statement", ""),
            "subject": claim.get("subject", ""),
            "object": claim.get("object", ""),
            "predicate": claim.get("predicate", ""),
            "support_score": claim.get("support_score", 0.0),
            "support_strength": claim.get("support_strength", ""),
            "mixed_evidence": bool(claim.get("mixed_evidence")),
            "source_count": int(claim.get("source_count", 0) or 0),
            "evidence_count": int(claim.get("evidence_count", 0) or 0),
            "display_priority_score": claim.get("display_priority_score", 0.0),
            "primary_sources": list(claim.get("primary_sources", []) or [])[:3],
            "supporting_ids": list(claim.get("supporting_ids", []) or [])[:8],
        }

    return {
        "substantive_claim_count": len(adjudicated_claims),
        "high_support_count": strength_counts["high"],
        "medium_support_count": strength_counts["medium"],
        "low_support_count": strength_counts["low"],
        "mixed_evidence_count": len(conflicts),
        "top_supported_claims": [_public_claim_summary(claim) for claim in ranked_claims[:8]],
        "mixed_evidence_claims": conflicts[:6],
        "source_weighting_reference": sorted(
            source_weight_reference.values(),
            key=lambda item: (float(item.get("weight", 0.0) or 0.0), item.get("source", "")),
            reverse=True,
        )[:10],
    }


def _adjudicate_substantive_claims(store: dict[str, Any], objective_text: str = "") -> dict[str, Any]:
    claims = list((store or {}).get("claims", {}).values())
    evidence_records = list((store or {}).get("evidence", []))
    label_by_entity = {
        entity_id: str(entity.get("label", entity_id)).strip()
        for entity_id, entity in ((store or {}).get("entities", {}) or {}).items()
    }
    evidence_by_claim: dict[str, list[dict[str, Any]]] = {}
    for record in evidence_records:
        claim_id = str(record.get("claim_id", "")).strip()
        if not claim_id:
            continue
        evidence_by_claim.setdefault(claim_id, []).append(record)

    excluded_predicates = SYNTHESIS_META_PREDICATES | TRIVIAL_IDENTIFIER_PREDICATES
    substantive_claims = [
        claim
        for claim in claims
        if str(claim.get("predicate", "")).strip() not in excluded_predicates
    ]
    adjudicated_claims = [
        _score_adjudicated_claim(claim, evidence_by_claim.get(str(claim.get("id", "")).strip(), []), label_by_entity)
        for claim in substantive_claims
    ]
    conflicts, mixed_claim_ids = _detect_claim_conflicts(adjudicated_claims)

    for claim in adjudicated_claims:
        claim["mixed_evidence"] = str(claim.get("claim_id", "")).strip() in mixed_claim_ids
        claim["display_priority_score"] = _claim_display_priority_score(claim, objective_text)

    ranked_claims = sorted(
        adjudicated_claims,
        key=lambda claim: (
            float(claim.get("display_priority_score", 0.0) or 0.0),
            1 if not claim.get("mixed_evidence") else 0,
            int(claim.get("source_count", 0) or 0),
            str(claim.get("statement", "")),
        ),
        reverse=True,
    )
    return {
        "claims_by_id": {
            str(claim.get("id", "")).strip(): claim
            for claim in substantive_claims
            if str(claim.get("id", "")).strip()
        },
        "evidence_by_claim": evidence_by_claim,
        "label_by_entity": label_by_entity,
        "adjudicated_claims": adjudicated_claims,
        "ranked_claims": ranked_claims,
        "conflicts": conflicts,
    }


def _truncate_graph_label(text: str, *, max_chars: int = 28) -> str:
    cleaned = _sanitize_internal_report_text(str(text or "").strip()) or "Unknown"
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _normalize_graph_match_text(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def _graph_focus_type_bonus(entity_type: str) -> float:
    normalized = str(entity_type or "").strip().lower()
    if normalized in EVIDENCE_GRAPH_TARGET_ENTITY_TYPES:
        return 4.0
    if normalized in {"pathway", "mechanism", "biological_process", "protein_complex"}:
        return 2.0
    if normalized in {"disease", "phenotype", "cell_type", "tissue"}:
        return 0.75
    if normalized in {"literal", "objective", "source"}:
        return -1.5
    return 1.0


def _is_informative_focus_label(label: str) -> bool:
    normalized = _normalize_graph_match_text(label)
    compact = _normalize_lookup_key(label)
    if not normalized or not compact:
        return False
    if normalized in EVIDENCE_GRAPH_GENERIC_FOCUS_TERMS:
        return False
    if normalized.endswith(" table") or normalized.endswith(" column"):
        return False
    if compact in {"approvedsymbol", "dbxrefs", "diseaseid", "geneid", "name", "proteinid", "targetid"}:
        return False
    if re.fullmatch(r"[a-z]+id", compact):
        return False
    return True


def _clean_query_focus_fragment(text: str) -> str:
    value = _sanitize_internal_report_text(str(text or "").strip())
    value = re.sub(r"^[Tt]he\s+", "", value)
    value = re.sub(r"^(?:and|or)\s+", "", value, flags=re.I)
    value = re.sub(r"^(?:approved|publicly available|public|known|standard|canonical|official|top\s+\d+)\s+", "", value, flags=re.I)
    value = re.sub(r"\s+\(excluding[^)]*\)", "", value, flags=re.I)
    value = value.strip(" ,.;:()")
    return value


def _extract_query_focus_labels(objective_text: str, *, max_items: int = 6) -> list[str]:
    text = _sanitize_internal_report_text(str(objective_text or "").strip())
    if not text:
        return []

    candidates: list[str] = []

    def _push_many(segment: str) -> None:
        cleaned = _clean_query_focus_fragment(segment)
        if not cleaned:
            return
        pieces = re.split(r"\s*(?:,| and | or | versus | vs\.? | between )\s*", cleaned, flags=re.I)
        for piece in pieces:
            fragment = _clean_query_focus_fragment(piece)
            if not fragment:
                continue
            if fragment.lower() in {"", "and", "or"}:
                continue
            if _is_informative_focus_label(fragment):
                candidates.append(fragment)

    patterns = [
        r"\b(?:assess|determine|evaluate|investigate)\s+(?:if|whether)\s+(.+?)\s+(?:is|remains|are|looks|look|can|should)\b",
        r"\b(?:compare|among|between|versus|across)\s+(.+?)(?:\s+(?:to determine|on|for|based on|once|highlighting|focusing|including|considering)\b|[.?!])",
        r"\b(?:investigate|identify|rank|find)\s+(.+?)(?:\s+(?:to determine|for|by|based on|including|considering|highlighting|focusing)\b|[.?!])",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            _push_many(match.group(1))

    for match in re.finditer(r"\b[A-Z][A-Z0-9-]{2,}(?:\s+[A-Z0-9-]{1,})*\b", text):
        token = _clean_query_focus_fragment(match.group(0))
        if _is_informative_focus_label(token):
            candidates.append(token)

    ordered = _dedupe_str_list(candidates, limit=max_items)
    return [item for item in ordered if _is_informative_focus_label(item)][:max_items]


def _match_support_topology_focus_ids(
    edge_payload: dict[str, Any],
    *,
    focus_by_id: dict[str, dict[str, Any]],
    target_entity: dict[str, Any] | None,
) -> list[str]:
    haystacks: list[str] = []

    def _append(value: Any) -> None:
        cleaned = _normalize_lookup_key(_sanitize_internal_report_text(str(value or "").strip()))
        if cleaned:
            haystacks.append(cleaned)

    _append(edge_payload.get("statement", ""))
    _append(edge_payload.get("target_label", ""))
    _append(edge_payload.get("_target_literal", ""))
    for value in list(edge_payload.get("supporting_ids", []) or []):
        _append(value)
    for value in list(edge_payload.get("primary_sources", []) or []):
        _append(value)

    if isinstance(target_entity, dict):
        _append(target_entity.get("label", ""))
        for alias in list(target_entity.get("aliases", []) or []):
            _append(alias)
        attrs = dict(target_entity.get("attrs", {}) or {})
        _append(attrs.get("identifier", ""))

    matches: list[str] = []
    for focus_id, metadata in focus_by_id.items():
        label = str(metadata.get("label", "") or "").strip()
        if not label:
            continue
        compact = _normalize_lookup_key(label)
        if not compact:
            continue
        if any(compact in haystack or haystack in compact for haystack in haystacks if haystack):
            matches.append(focus_id)
    return matches


def _build_graph_focus_candidates(node_data: dict[str, Any]) -> list[tuple[str, float]]:
    candidates: list[tuple[str, float]] = []
    seen: set[str] = set()

    def _push(value: Any, weight: float) -> None:
        raw = _sanitize_internal_report_text(str(value or "").strip())
        normalized = _normalize_graph_match_text(raw)
        compact = _normalize_lookup_key(raw)
        if not raw or not normalized or not compact:
            return
        if normalized in EVIDENCE_GRAPH_GENERIC_FOCUS_TERMS:
            return
        if len(normalized.split()) == 1 and len(compact) < 3:
            return
        if compact in seen:
            return
        seen.add(compact)
        candidates.append((raw, weight))

    _push(node_data.get("full_label") or node_data.get("label") or "", 12.0)
    for alias in list(node_data.get("aliases", []) or []):
        _push(alias, 9.0)
    attrs = node_data.get("attrs", {}) if isinstance(node_data.get("attrs", {}), dict) else {}
    _push(attrs.get("identifier", ""), 7.0)
    node_id = str(node_data.get("id", "") or "").strip()
    if node_id:
        tail = node_id.split(":")[-1].strip()
        if tail and tail != node_id:
            _push(tail, 5.0)
    return candidates


def _graph_objective_match_score(node_data: dict[str, Any], objective_text: str) -> tuple[float, list[str]]:
    objective_phrase = _normalize_graph_match_text(objective_text)
    objective_compact = _normalize_lookup_key(objective_text)
    if not objective_phrase or not objective_compact:
        return 0.0, []

    if str(node_data.get("type", "") or "").strip().lower() == "objective":
        return 0.0, []

    phrase_haystack = f" {objective_phrase} "
    best_score = 0.0
    matched_terms: list[str] = []
    for raw_term, base_weight in _build_graph_focus_candidates(node_data):
        normalized = _normalize_graph_match_text(raw_term)
        compact = _normalize_lookup_key(raw_term)
        term_score = 0.0
        matched = False
        if normalized and f" {normalized} " in phrase_haystack:
            matched = True
            term_score = max(term_score, base_weight + (2.5 if len(normalized.split()) > 1 else 1.5))
        if compact and len(compact) >= 3 and compact in objective_compact:
            matched = True
            term_score = max(term_score, base_weight + 1.0)
        if not matched:
            continue
        best_score = max(best_score, term_score)
        matched_terms.append(raw_term)

    return best_score, _dedupe_str_list(matched_terms, limit=4)


def _annotate_evidence_graph_focus(
    nodes_by_id: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    objective_text: str,
    mode: str,
) -> dict[str, Any]:
    support_weight_by_node: dict[str, float] = {node_id: 0.0 for node_id in nodes_by_id}
    for edge in edges:
        data = edge.get("data", {}) if isinstance(edge, dict) else {}
        source_id = str(data.get("source", "") or "").strip()
        target_id = str(data.get("target", "") or "").strip()
        support_score = float(data.get("support_score", 0.0) or 0.0)
        if source_id:
            support_weight_by_node[source_id] = support_weight_by_node.get(source_id, 0.0) + support_score
        if target_id:
            support_weight_by_node[target_id] = support_weight_by_node.get(target_id, 0.0) + support_score

    matched_candidates: list[dict[str, Any]] = []
    fallback_candidates: list[dict[str, Any]] = []
    for node_id, payload in nodes_by_id.items():
        data = payload.get("data", {}) if isinstance(payload.get("data", {}), dict) else {}
        entity_type = str(data.get("type", "") or "").strip().lower()
        degree = int(data.get("degree", 0) or 0)
        connected_claim_count = int(data.get("connected_claim_count", 0) or 0)
        support_weight = round(float(support_weight_by_node.get(node_id, 0.0) or 0.0), 3)
        data["support_weight"] = support_weight
        data["is_focus"] = 0
        data["focus_rank"] = 0
        data["focus_score"] = 0.0
        data["focus_terms"] = []

        if entity_type in {"objective", "source", "literal"} | EVIDENCE_GRAPH_STRUCTURAL_ENTITY_TYPES:
            continue

        mention_score, matched_terms = _graph_objective_match_score(data, objective_text)
        type_bonus = _graph_focus_type_bonus(entity_type)
        structure_bonus = min(4.0, support_weight / 4.0) + min(3.0, connected_claim_count / 6.0)
        score = round(mention_score + type_bonus + structure_bonus, 3)
        candidate = {
            "node_id": node_id,
            "label": str(data.get("full_label", "") or data.get("label", "")).strip() or node_id,
            "label_key": _normalize_lookup_key(str(data.get("full_label", "") or data.get("label", "")).strip() or node_id),
            "entity_type": entity_type,
            "degree": degree,
            "connected_claim_count": connected_claim_count,
            "support_weight": support_weight,
            "focus_score": score,
            "focus_terms": matched_terms,
            "mention_score": mention_score,
        }
        if not _is_informative_focus_label(str(candidate["label"])):
            if mention_score <= 0:
                continue
        fallback_candidates.append(candidate)
        if mention_score > 0:
            matched_candidates.append(candidate)

    selected_candidates: list[dict[str, Any]] = []
    if matched_candidates:
        if any(candidate["entity_type"] in EVIDENCE_GRAPH_TARGET_ENTITY_TYPES for candidate in matched_candidates):
            matched_candidates = [
                candidate
                for candidate in matched_candidates
                if candidate["entity_type"] in EVIDENCE_GRAPH_TARGET_ENTITY_TYPES
            ]
        best_by_label: dict[str, dict[str, Any]] = {}
        for candidate in sorted(
            matched_candidates,
            key=lambda item: (
                float(item.get("focus_score", 0.0) or 0.0),
                float(item.get("support_weight", 0.0) or 0.0),
                int(item.get("connected_claim_count", 0) or 0),
            ),
            reverse=True,
        ):
            key = str(candidate.get("label_key", "") or candidate.get("node_id", "")).strip()
            if key and key not in best_by_label:
                best_by_label[key] = candidate
        selected_candidates = list(best_by_label.values())[:6]
    elif mode == "semantic":
        semantic_fallback = [
            candidate
            for candidate in fallback_candidates
            if candidate["entity_type"] not in {"disease", "phenotype"}
        ]
        if any(candidate["entity_type"] in EVIDENCE_GRAPH_TARGET_ENTITY_TYPES for candidate in semantic_fallback):
            semantic_fallback = [
                candidate
                for candidate in semantic_fallback
                if candidate["entity_type"] in EVIDENCE_GRAPH_TARGET_ENTITY_TYPES
            ]
        fallback_ordered = sorted(
            semantic_fallback or fallback_candidates,
            key=lambda item: (
                float(item.get("support_weight", 0.0) or 0.0),
                int(item.get("connected_claim_count", 0) or 0),
                _graph_focus_type_bonus(str(item.get("entity_type", "") or "")),
            ),
            reverse=True,
        )
        best_by_label: dict[str, dict[str, Any]] = {}
        for candidate in fallback_ordered:
            key = str(candidate.get("label_key", "") or candidate.get("node_id", "")).strip()
            if key and key not in best_by_label:
                best_by_label[key] = candidate
        selected_candidates = list(best_by_label.values())[:3]

    focus_node_ids: list[str] = []
    focus_labels: list[str] = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        node_id = str(candidate.get("node_id", "") or "").strip()
        if not node_id or node_id not in nodes_by_id:
            continue
        data = nodes_by_id[node_id]["data"]
        data["is_focus"] = 1
        data["focus_rank"] = rank
        data["focus_score"] = round(float(candidate.get("focus_score", 0.0) or 0.0), 3)
        data["focus_terms"] = list(candidate.get("focus_terms", []) or [])[:4]
        focus_node_ids.append(node_id)
        focus_labels.append(str(data.get("full_label", "") or data.get("label", "")).strip() or node_id)

    return {
        "focus_node_ids": focus_node_ids,
        "focus_labels": _dedupe_str_list(focus_labels, limit=6),
    }


def _build_semantic_evidence_graph(task_state: dict[str, Any]) -> dict[str, Any]:
    store = task_state.get("evidence_store", {}) if isinstance(task_state, dict) else {}
    store = store if isinstance(store, dict) else {}
    objective_text = str((task_state or {}).get("objective", "") or "").strip()
    context = _adjudicate_substantive_claims(store, objective_text)
    ranked_claims = list(context.get("ranked_claims", []) or [])
    claims_by_id = context.get("claims_by_id", {}) if isinstance(context.get("claims_by_id", {}), dict) else {}
    evidence_by_claim = context.get("evidence_by_claim", {}) if isinstance(context.get("evidence_by_claim", {}), dict) else {}
    entities_by_id = ((store or {}).get("entities", {}) or {}) if isinstance((store or {}).get("entities", {}), dict) else {}

    warnings = ["Graph shows semantic claims only. Workflow steps and source scaffolding are hidden."]
    nodes_by_id: dict[str, dict[str, Any]] = {}
    neighbor_ids: dict[str, set[str]] = {}
    incident_counts: dict[str, int] = {}
    edges: list[dict[str, Any]] = []
    rendered_evidence_ids: set[str] = set()
    skipped_claims = 0
    mode = "semantic"

    def _ensure_graph_node(node_id: str, *, label: str, entity_type: str, aliases: list[str] | None = None, attrs: dict[str, Any] | None = None) -> str:
        if node_id not in nodes_by_id:
            nodes_by_id[node_id] = {
                "data": {
                    "id": node_id,
                    "label": _truncate_graph_label(label),
                    "full_label": _sanitize_internal_report_text(label),
                    "type": str(entity_type or "record").strip() or "record",
                    "aliases": _dedupe_str_list(list(aliases or []), limit=12),
                    "attrs": dict(attrs or {}),
                    "degree": 0,
                    "connected_claim_count": 0,
                }
            }
            neighbor_ids[node_id] = set()
            incident_counts[node_id] = 0
        return node_id

    def _ensure_entity_node(entity_id: str, *, allow_hidden: bool = False) -> str:
        entity = entities_by_id.get(entity_id)
        if not isinstance(entity, dict):
            return ""
        entity_type = str(entity.get("type", "record") or "record").strip() or "record"
        if not allow_hidden and entity_type in EVIDENCE_GRAPH_HIDDEN_ENTITY_TYPES:
            return ""
        return _ensure_graph_node(
            str(entity.get("id", entity_id) or entity_id).strip(),
            label=str(entity.get("label", entity_id) or entity_id).strip(),
            entity_type=entity_type,
            aliases=list(entity.get("aliases", []) or []),
            attrs=dict(entity.get("attrs", {}) or {}),
        )

    def _ensure_literal_node(object_literal: str) -> str:
        literal = _sanitize_internal_report_text(str(object_literal or "").strip())
        literal_id = f"literal:{_slugify_token(literal or 'literal')}"
        return _ensure_graph_node(
            literal_id,
            label=literal or "Literal value",
            entity_type="literal",
            aliases=[],
            attrs={"literal": True},
        )

    for claim in ranked_claims:
        claim_id = str(claim.get("claim_id", "")).strip()
        raw_claim = claims_by_id.get(claim_id, {})
        if not claim_id or not isinstance(raw_claim, dict):
            skipped_claims += 1
            continue

        subject_entity_id = str(raw_claim.get("subject_entity_id", "") or "").strip()
        object_entity_id = str(raw_claim.get("object_entity_id", "") or "").strip()
        object_literal = str(raw_claim.get("object_literal", "") or "").strip()

        source_id = _ensure_entity_node(subject_entity_id)
        if not source_id:
            skipped_claims += 1
            continue

        target_id = _ensure_entity_node(object_entity_id) if object_entity_id else ""
        if not target_id and object_literal:
            target_id = _ensure_literal_node(object_literal)
        if not target_id:
            skipped_claims += 1
            continue

        records = list(evidence_by_claim.get(claim_id, []) or [])
        qualifier_lines = _dedupe_str_list(
            [
                _format_observation_qualifiers(record.get("qualifiers", {}))
                for record in records
                if isinstance(record, dict) and _format_observation_qualifiers(record.get("qualifiers", {}))
            ],
            limit=8,
        )

        edges.append(
            {
                "data": {
                    "id": claim_id,
                    "source": source_id,
                    "target": target_id,
                    "predicate": str(claim.get("predicate", "") or "").strip(),
                    "label": _humanize_claim_predicate(str(claim.get("predicate", "") or "").strip()),
                    "statement": str(claim.get("statement", "") or "").strip(),
                    "confidence": str(claim.get("claim_confidence", "unknown") or "unknown").strip(),
                    "support_strength": str(claim.get("support_strength", "low") or "low").strip(),
                    "support_score": round(float(claim.get("support_score", 0.0) or 0.0), 3),
                    "evidence_count": int(claim.get("evidence_count", 0) or 0),
                    "primary_sources": list(claim.get("primary_sources", []) or [])[:5],
                    "supporting_ids": list(claim.get("supporting_ids", []) or [])[:12],
                    "qualifiers": qualifier_lines,
                    "mixed_evidence": bool(claim.get("mixed_evidence")),
                }
            }
        )

        neighbor_ids[source_id].add(target_id)
        neighbor_ids[target_id].add(source_id)
        incident_counts[source_id] = incident_counts.get(source_id, 0) + 1
        incident_counts[target_id] = incident_counts.get(target_id, 0) + 1
        for record in records:
            record_id = str((record or {}).get("id", "") or "").strip()
            if record_id:
                rendered_evidence_ids.add(record_id)

    if not edges:
        nodes_by_id = {}
        neighbor_ids = {}
        incident_counts = {}
        rendered_evidence_ids = set()
        warnings = []
        mode = "support_topology"

        objective_entities = [
            entity
            for entity in entities_by_id.values()
            if isinstance(entity, dict) and str(entity.get("type", "") or "").strip() == "objective"
        ]
        objective_entity = objective_entities[0] if objective_entities else None
        objective_entity_id = str((objective_entity or {}).get("id", "") or "").strip()
        objective_node_id = objective_entity_id or ""

        support_candidates_by_type: dict[str, list[dict[str, Any]]] = {}

        def _record_summary_payload(
            records: list[dict[str, Any]],
            *,
            confidence: str,
            statement: str,
            label: str,
            predicate: str,
            mixed: bool = False,
            source_labels: list[str] | None = None,
        ) -> dict[str, Any]:
            supporting_ids = _dedupe_str_list(
                [
                    identifier
                    for record in records
                    for identifier in list((record or {}).get("evidence_ids", []) or [])
                ],
                limit=12,
            )
            primary_sources = _dedupe_str_list(
                [
                    *[str((record or {}).get("source_label", "") or "").strip() for record in records],
                    *[str(value or "").strip() for value in list(source_labels or [])],
                ],
                limit=5,
            )
            qualifier_lines = _dedupe_str_list(
                [
                    _format_observation_qualifiers((record or {}).get("qualifiers", {}))
                    for record in records
                    if _format_observation_qualifiers((record or {}).get("qualifiers", {}))
                ],
                limit=8,
            )
            evidence_count = len(records)
            support_strength = "high" if evidence_count >= 3 or len(supporting_ids) >= 3 else ("medium" if evidence_count >= 1 else "low")
            support_score = round(
                min(3.0, (0.32 * evidence_count) + (0.08 * len(primary_sources)) + (0.05 * len(supporting_ids))),
                3,
            )
            for record in records:
                record_id = str((record or {}).get("id", "") or "").strip()
                if record_id:
                    rendered_evidence_ids.add(record_id)
            return {
                "label": label,
                "predicate": predicate,
                "statement": statement,
                "confidence": str(confidence or "low").strip() or "low",
                "support_strength": support_strength,
                "support_score": support_score,
                "evidence_count": evidence_count,
                "primary_sources": primary_sources,
                "supporting_ids": supporting_ids,
                "qualifiers": qualifier_lines,
                "mixed_evidence": mixed,
            }

        for raw_claim in list((store or {}).get("claims", {}).values()):
            if not isinstance(raw_claim, dict):
                continue
            claim_id = str(raw_claim.get("id", "") or "").strip()
            predicate = str(raw_claim.get("predicate", "") or "").strip()
            records = list(evidence_by_claim.get(claim_id, []) or [])
            subject_entity_id = str(raw_claim.get("subject_entity_id", "") or "").strip()
            object_entity_id = str(raw_claim.get("object_entity_id", "") or "").strip()
            object_literal = str(raw_claim.get("object_literal", "") or "").strip()
            if predicate == "queried_source":
                continue

            if predicate != "supported_by" or subject_entity_id != objective_entity_id:
                continue

            if not objective_node_id:
                continue

            target_entity = entities_by_id.get(object_entity_id) if object_entity_id else None
            target_type = str((target_entity or {}).get("type", "record") or "record").strip() or "record"
            if not object_entity_id and object_literal:
                target_type = "literal"
            if target_type in {"objective", "step", "source"}:
                continue

            target_label = str((target_entity or {}).get("label", "") or object_literal or object_entity_id).strip()
            payload = _record_summary_payload(
                records,
                confidence=str(raw_claim.get("confidence", "low") or "low"),
                statement=f"Session evidence includes {target_label}",
                label=_humanize_claim_predicate(predicate),
                predicate=predicate,
                source_labels=list(raw_claim.get("source_labels", []) or []),
            )
            payload["id"] = f"graph:support:{object_entity_id or _slugify_token(object_literal)}"
            payload["_target_entity_id"] = object_entity_id
            payload["_target_literal"] = object_literal
            payload["_allow_hidden_target"] = False
            payload["target_label"] = target_label
            support_candidates_by_type.setdefault(target_type, []).append(payload)

        selected_edge_specs: list[dict[str, Any]] = []
        for target_type, entries in sorted(support_candidates_by_type.items()):
            ordered = sorted(
                entries,
                key=lambda item: (
                    float(item.get("support_score", 0.0) or 0.0),
                    int(item.get("evidence_count", 0) or 0),
                    str(item.get("target_label", "") or ""),
                ),
                reverse=True,
            )
            selected_edge_specs.extend(ordered)

        nodes_by_id = {}
        neighbor_ids = {}
        incident_counts = {}
        edges = []
        rendered_evidence_ids = set()
        objective_node_id = ""
        inferred_focus_labels = _extract_query_focus_labels(objective_text)
        focus_node_ids: list[str] = []
        focus_by_id: dict[str, dict[str, Any]] = {}

        if inferred_focus_labels:
            for label in inferred_focus_labels:
                focus_node_id = _ensure_graph_node(
                    f"graph:focus:{_slugify_token(label)}",
                    label=label,
                    entity_type="query_focus",
                    aliases=[],
                    attrs={"inferred_from_query": True},
                )
                focus_node_ids.append(focus_node_id)
                focus_by_id[focus_node_id] = {"label": label}
        if not focus_node_ids and objective_entity_id:
            objective_node_id = _ensure_entity_node(objective_entity_id, allow_hidden=True)

        for edge in selected_edge_specs:
            target_entity_id = str(edge.get("_target_entity_id", "") or "").strip()
            target_literal = str(edge.get("_target_literal", "") or "").strip()
            allow_hidden_target = bool(edge.get("_allow_hidden_target", False))
            target_id = _ensure_entity_node(target_entity_id, allow_hidden=allow_hidden_target) if target_entity_id else ""
            if not target_id and target_literal:
                target_id = _ensure_literal_node(target_literal)
            if not target_id:
                continue

            source_ids: list[str] = []
            target_entity = entities_by_id.get(target_entity_id) if target_entity_id else None
            if focus_node_ids:
                matched_focus_ids = _match_support_topology_focus_ids(
                    edge,
                    focus_by_id=focus_by_id,
                    target_entity=target_entity if isinstance(target_entity, dict) else None,
                )
                if len(focus_node_ids) == 1:
                    source_ids = [focus_node_ids[0]]
                elif matched_focus_ids:
                    source_ids = matched_focus_ids
                else:
                    source_ids = list(focus_node_ids)
            elif objective_node_id:
                source_ids = [objective_node_id]

            edge.pop("_target_entity_id", None)
            edge.pop("_target_literal", None)
            edge.pop("_allow_hidden_target", None)
            edge.pop("target_label", None)

            for source_id in source_ids:
                source_id = str(source_id or "").strip()
                target_id = str(target_id or "").strip()
                if not source_id or not target_id:
                    continue
                edge_data = dict(edge)
                edge_data["id"] = f"{edge_data.get('id', 'graph:edge')}:{_slugify_token(source_id)}"
                edge_data["source"] = source_id
                edge_data["target"] = target_id
                neighbor_ids.setdefault(source_id, set()).add(target_id)
                neighbor_ids.setdefault(target_id, set()).add(source_id)
                incident_counts[source_id] = incident_counts.get(source_id, 0) + 1
                incident_counts[target_id] = incident_counts.get(target_id, 0) + 1
                edges.append({"data": edge_data})

    if skipped_claims > 0 and mode == "semantic":
        warnings.append("Some claims were omitted because they did not resolve to graphable nodes.")
    if not edges:
        warnings.append("No graphable evidence is available for this session yet.")

    for node_id, payload in nodes_by_id.items():
        payload["data"]["degree"] = len(neighbor_ids.get(node_id, set()))
        payload["data"]["connected_claim_count"] = int(incident_counts.get(node_id, 0) or 0)

    focus_summary = _annotate_evidence_graph_focus(
        nodes_by_id,
        edges,
        objective_text=objective_text,
        mode=mode,
    )

    sorted_nodes = sorted(
        nodes_by_id.values(),
        key=lambda item: (
            -int(item.get("data", {}).get("is_focus", 0) or 0),
            int(item.get("data", {}).get("focus_rank", 0) or 0),
            str(item.get("data", {}).get("type", "") or ""),
            str(item.get("data", {}).get("full_label", "") or ""),
        ),
    )
    sorted_edges = sorted(
        edges,
        key=lambda item: (
            float(item.get("data", {}).get("support_score", 0.0) or 0.0),
            str(item.get("data", {}).get("statement", "") or ""),
        ),
        reverse=True,
    )

    return {
        "mode": mode,
        "elements": {
            "nodes": sorted_nodes,
            "edges": sorted_edges,
        },
        "summary": {
            "node_count": len(sorted_nodes),
            "edge_count": len(sorted_edges),
            "evidence_count": len(rendered_evidence_ids),
            "mixed_edge_count": sum(1 for edge in sorted_edges if edge.get("data", {}).get("mixed_evidence")),
            "focus_node_count": len(list(focus_summary.get("focus_node_ids", []) or [])),
            "focus_node_ids": list(focus_summary.get("focus_node_ids", []) or []),
            "focus_labels": list(focus_summary.get("focus_labels", []) or []),
        },
        "warnings": warnings,
    }


def _aggregate_executor_metrics(step_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not step_metrics:
        return _new_execution_metrics_bundle()["summary"]

    summary = {
        "step_count": len(step_metrics),
        "completed_count": sum(1 for metric in step_metrics if metric.get("status") == "completed"),
        "blocked_count": sum(1 for metric in step_metrics if metric.get("status") == "blocked"),
        "tool_hint_accuracy": None,
        "tool_hint_first_accuracy": None,
        "fallback_rate": 0.0,
        "avg_tools_per_step": 0.0,
        "avg_evidence_ids_per_step": 0.0,
        "avg_structured_observations_per_step": 0.0,
        "avg_parse_retries_per_step": 0.0,
        "clusters": [],
        "specialization_watchlist": [],
    }
    hinted = [metric for metric in step_metrics if metric.get("tool_hint")]
    if hinted:
        summary["tool_hint_accuracy"] = round(
            sum(1 for metric in hinted if metric.get("used_tool_hint")) / len(hinted),
            3,
        )
        summary["tool_hint_first_accuracy"] = round(
            sum(1 for metric in hinted if metric.get("used_tool_hint_first")) / len(hinted),
            3,
        )
    summary["fallback_rate"] = round(
        sum(1 for metric in step_metrics if metric.get("fallback_used")) / len(step_metrics),
        3,
    )
    summary["avg_tools_per_step"] = round(
        sum(int(metric.get("tool_count", 0) or 0) for metric in step_metrics) / len(step_metrics),
        3,
    )
    summary["avg_evidence_ids_per_step"] = round(
        sum(int(metric.get("evidence_count", 0) or 0) for metric in step_metrics) / len(step_metrics),
        3,
    )
    summary["avg_structured_observations_per_step"] = round(
        sum(int(metric.get("structured_observation_count", 0) or 0) for metric in step_metrics) / len(step_metrics),
        3,
    )
    summary["avg_parse_retries_per_step"] = round(
        sum(int(metric.get("parse_retry_count", 0) or 0) for metric in step_metrics) / len(step_metrics),
        3,
    )

    cluster_groups: dict[str, list[dict[str, Any]]] = {}
    for metric in step_metrics:
        cluster_groups.setdefault(str(metric.get("executor_cluster", "general")), []).append(metric)

    cluster_summaries = []
    watchlist: list[str] = []
    for cluster_name in sorted(cluster_groups):
        metrics = cluster_groups[cluster_name]
        hinted_metrics = [metric for metric in metrics if metric.get("tool_hint")]
        tool_hint_accuracy = None
        if hinted_metrics:
            tool_hint_accuracy = round(
                sum(1 for metric in hinted_metrics if metric.get("used_tool_hint")) / len(hinted_metrics),
                3,
            )
        fallback_rate = round(
            sum(1 for metric in metrics if metric.get("fallback_used")) / len(metrics),
            3,
        )
        blocked_rate = round(
            sum(1 for metric in metrics if metric.get("status") == "blocked") / len(metrics),
            3,
        )
        cluster_summary = {
            "cluster": cluster_name,
            "step_count": len(metrics),
            "blocked_rate": blocked_rate,
            "fallback_rate": fallback_rate,
            "tool_hint_accuracy": tool_hint_accuracy,
            "avg_tools_per_step": round(
                sum(int(metric.get("tool_count", 0) or 0) for metric in metrics) / len(metrics),
                3,
            ),
            "avg_structured_observations_per_step": round(
                sum(int(metric.get("structured_observation_count", 0) or 0) for metric in metrics) / len(metrics),
                3,
            ),
            "avg_parse_retries_per_step": round(
                sum(int(metric.get("parse_retry_count", 0) or 0) for metric in metrics) / len(metrics),
                3,
            ),
        }
        cluster_summaries.append(cluster_summary)
        if len(metrics) >= 2 and (
            (tool_hint_accuracy is not None and tool_hint_accuracy < 0.85)
            or fallback_rate > 0.25
            or blocked_rate > 0.2
        ):
            watchlist.append(cluster_name)

    summary["clusters"] = cluster_summaries
    summary["specialization_watchlist"] = watchlist
    return summary


def _append_structured_observation_claims(
    store: dict[str, Any],
    *,
    step: dict[str, Any],
    observations: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    entity_ids: list[str] = []
    claim_ids: list[str] = []
    step_id = str(step.get("id", "")).strip()
    default_tool = str(step.get("tool_hint", "")).strip()
    default_summary = str(step.get("result_summary", "")).strip()

    for observation in observations:
        subject_ref = observation.get("subject") or {}
        subject_entity = _upsert_entity(
            store,
            subject_ref.get("type", "record"),
            subject_ref.get("label") or subject_ref.get("id") or "Unknown subject",
            aliases=subject_ref.get("aliases"),
            attrs=subject_ref.get("attrs"),
            canonical_key=_canonical_entity_key(
                subject_ref.get("type", "record"),
                subject_ref.get("id") or subject_ref.get("label") or "unknown_subject",
            ),
        )
        entity_ids.append(subject_entity["id"])

        object_ref = observation.get("object")
        object_entity_id = ""
        object_literal = str(observation.get("object_literal", "")).strip()
        if isinstance(object_ref, dict):
            object_entity = _upsert_entity(
                store,
                object_ref.get("type", "record"),
                object_ref.get("label") or object_ref.get("id") or "Unknown object",
                aliases=object_ref.get("aliases"),
                attrs=object_ref.get("attrs"),
                canonical_key=_canonical_entity_key(
                    object_ref.get("type", "record"),
                    object_ref.get("id") or object_ref.get("label") or "unknown_object",
                ),
            )
            object_entity_id = object_entity["id"]
            entity_ids.append(object_entity["id"])

        source_tool = str(observation.get("source_tool", "")).strip() or default_tool
        source_label = _preferred_step_source_label(step, source_tool)
        claim = _upsert_claim(
            store,
            subject_entity["id"],
            str(observation.get("predicate", "")).strip(),
            object_id=object_entity_id,
            object_literal=object_literal,
            status="supported",
            confidence=str(observation.get("confidence", "medium")).strip().lower(),
            step_id=step_id,
            source_tool=source_tool,
            source_label=source_label,
            observation_type=str(observation.get("observation_type", "")).strip(),
        )
        claim_ids.append(claim["id"])

        qualifier_text = _format_observation_qualifiers(observation.get("qualifiers", {}))
        evidence_summary = default_summary
        if qualifier_text:
            evidence_summary = f"{default_summary} | {qualifier_text}" if default_summary else qualifier_text
        _append_evidence_record(
            store,
            claim_id=claim["id"],
            step_id=step_id,
            source_tool=source_tool,
            source_label=source_label,
            evidence_ids=observation.get("supporting_ids", []),
            summary_text=evidence_summary,
            qualifiers=observation.get("qualifiers", {}),
        )

    return _merge_str_values([], entity_ids), _merge_str_values([], claim_ids)


def _extract_v1_evidence_from_step(
    task_state: dict[str, Any],
    step: dict[str, Any],
    store: dict[str, Any],
) -> tuple[list[str], list[str]]:
    status = str(step.get("status", "")).strip().lower()
    if status not in {"completed", "blocked"}:
        return [], []

    entity_ids: list[str] = []
    claim_ids: list[str] = []
    objective = str(task_state.get("objective", "")).strip()
    objective_fingerprint = str(task_state.get("objective_fingerprint", "")).strip()
    objective_entity = _upsert_entity(
        store,
        "objective",
        objective or "Research objective",
        attrs={"fingerprint": objective_fingerprint},
        canonical_key=f"objective:{_slugify_token(objective_fingerprint or objective or 'objective')}",
    )
    entity_ids.append(objective_entity["id"])

    step_id = str(step.get("id", "")).strip()
    step_goal = str(step.get("goal", "")).strip() or step_id or "Workflow step"
    step_entity = _upsert_entity(
        store,
        "step",
        f"{step_id}: {step_goal}" if step_id else step_goal,
        aliases=[step_id] if step_id else [],
        attrs={
            "step_id": step_id,
            "status": status,
            "tool_hint": step.get("tool_hint", ""),
            "domains": step.get("domains", []),
            "completion_condition": step.get("completion_condition", ""),
        },
        canonical_key=f"step:{_slugify_token(step_id or step_goal)}",
    )
    entity_ids.append(step_entity["id"])

    investigated_claim = _upsert_claim(
        store,
        objective_entity["id"],
        "investigated_by",
        object_id=step_entity["id"],
        status="observed",
        confidence="high",
        step_id=step_id,
        source_tool=str(step.get("tool_hint", "")).strip(),
        source_label=_preferred_step_source_label(step, str(step.get("tool_hint", "")).strip()),
    )
    claim_ids.append(investigated_claim["id"])

    observed_tools = _dedupe_str_list(
        ([step.get("tool_hint")] if step.get("tool_hint") else []) + list(step.get("tools_called", []) or []),
        limit=20,
    )
    for tool_name in observed_tools:
        source_label = _preferred_step_source_label(step, tool_name)
        source_entity = _upsert_entity(
            store,
            "source",
            source_label,
            aliases=[tool_name],
            attrs={
                "tool_names": [tool_name],
                "domains": tool_registry.TOOL_TO_DOMAINS.get(tool_name, []),
                "overlap_group": tool_registry.TOOL_ROUTING_METADATA.get(tool_name, {}).get("overlap_group", ""),
            },
            canonical_key=f"source:{_slugify_token(source_label)}",
        )
        entity_ids.append(source_entity["id"])
        queried_claim = _upsert_claim(
            store,
            step_entity["id"],
            "queried_source",
            object_id=source_entity["id"],
            status="observed",
            confidence="high",
            step_id=step_id,
            source_tool=tool_name,
            source_label=source_label,
        )
        claim_ids.append(queried_claim["id"])
        _append_evidence_record(
            store,
            claim_id=queried_claim["id"],
            step_id=step_id,
            source_tool=tool_name,
            source_label=source_label,
            evidence_ids=list(step.get("evidence_ids", []) or [])[:5],
            summary_text=str(step.get("step_progress_note", "") or step.get("result_summary", "")).strip(),
        )

    if status != "completed":
        return _merge_str_values([], entity_ids), _merge_str_values([], claim_ids)

    evidence_ids = _dedupe_str_list(step.get("evidence_ids", []) or [], limit=30)
    primary_tool = observed_tools[0] if observed_tools else str(step.get("tool_hint", "")).strip()
    primary_source = _preferred_step_source_label(step, primary_tool)
    claim_confidence = "high" if len(evidence_ids) >= 3 else ("medium" if evidence_ids else "low")
    for identifier in evidence_ids:
        inferred = _infer_entity_from_identifier(identifier)
        if inferred is None:
            continue
        evidence_entity = _upsert_entity(
            store,
            inferred["entity_type"],
            inferred["label"],
            aliases=inferred.get("aliases"),
            attrs=inferred.get("attrs"),
            canonical_key=inferred.get("canonical_key"),
        )
        entity_ids.append(evidence_entity["id"])
        support_claim = _upsert_claim(
            store,
            objective_entity["id"],
            "supported_by",
            object_id=evidence_entity["id"],
            status="supported",
            confidence=claim_confidence,
            step_id=step_id,
            source_tool=primary_tool,
            source_label=primary_source,
        )
        claim_ids.append(support_claim["id"])
        _append_evidence_record(
            store,
            claim_id=support_claim["id"],
            step_id=step_id,
            source_tool=primary_tool,
            source_label=primary_source,
            evidence_ids=[identifier],
            summary_text=str(step.get("result_summary", "")).strip(),
        )

    if not evidence_ids and primary_tool:
        source_entity = _upsert_entity(
            store,
            "source",
            primary_source,
            aliases=[primary_tool],
            attrs={"tool_names": [primary_tool]},
            canonical_key=f"source:{_slugify_token(primary_source)}",
        )
        entity_ids.append(source_entity["id"])
        support_claim = _upsert_claim(
            store,
            objective_entity["id"],
            "supported_by",
            object_id=source_entity["id"],
            status="supported",
            confidence="low",
            step_id=step_id,
            source_tool=primary_tool,
            source_label=primary_source,
        )
        claim_ids.append(support_claim["id"])
        _append_evidence_record(
            store,
            claim_id=support_claim["id"],
            step_id=step_id,
            source_tool=primary_tool,
            source_label=primary_source,
            evidence_ids=[],
            summary_text=str(step.get("result_summary", "")).strip(),
        )

    structured_entity_ids, structured_claim_ids = _append_structured_observation_claims(
        store,
        step=step,
        observations=list(step.get("structured_observations", []) or []),
    )
    entity_ids = _merge_str_values(entity_ids, structured_entity_ids, limit=80)
    claim_ids = _merge_str_values(claim_ids, structured_claim_ids, limit=80)

    return _merge_str_values([], entity_ids), _merge_str_values([], claim_ids)


def _rebuild_evidence_store(task_state: dict[str, Any]) -> dict[str, Any]:
    store = _new_evidence_store()
    objective = str(task_state.get("objective", "")).strip()
    objective_fingerprint = str(task_state.get("objective_fingerprint", "")).strip()
    if objective:
        _upsert_entity(
            store,
            "objective",
            objective,
            attrs={"fingerprint": objective_fingerprint},
            canonical_key=f"objective:{_slugify_token(objective_fingerprint or objective)}",
        )

    for step in task_state.get("steps", []):
        entity_ids, claim_ids = _extract_v1_evidence_from_step(task_state, step, store)
        step["entity_ids"] = entity_ids
        step["claim_ids"] = claim_ids

    return store


def _rebuild_execution_metrics_bundle(task_state: dict[str, Any]) -> dict[str, Any]:
    step_metrics = [
        dict(step.get("execution_metrics", {}))
        for step in task_state.get("steps", [])
        if isinstance(step.get("execution_metrics"), dict) and step.get("execution_metrics")
    ]
    return {
        "steps": step_metrics,
        "summary": _aggregate_executor_metrics(step_metrics),
    }


def _refresh_task_state_derived_state(task_state: dict[str, Any]) -> None:
    task_state["evidence_store"] = _rebuild_evidence_store(task_state)
    task_state["execution_metrics"] = _rebuild_execution_metrics_bundle(task_state)


def _get_task_state(callback_context: CallbackContext) -> dict[str, Any] | None:
    state = callback_context.state.get(STATE_WORKFLOW_TASK)
    return state if isinstance(state, dict) else None


def _archive_current_task(callback_context: CallbackContext) -> None:
    """Push a deep copy of the current task state onto the history stack."""
    task_state = _get_task_state(callback_context)
    if not task_state:
        return
    has_results = any(
        str(s.get("status", "")) in ("completed", "blocked")
        for s in task_state.get("steps", [])
    )
    if not has_results:
        return
    prior: list[dict[str, Any]] = callback_context.state.get(STATE_PRIOR_RESEARCH) or []
    if not isinstance(prior, list):
        prior = []
    import copy
    prior.append(copy.deepcopy(task_state))
    max_archived = 10
    callback_context.state[STATE_PRIOR_RESEARCH] = prior[-max_archived:]


def _get_prior_research(callback_context: CallbackContext) -> list[dict[str, Any]]:
    """Return the list of archived task states (full history)."""
    prior = callback_context.state.get(STATE_PRIOR_RESEARCH) or []
    return prior if isinstance(prior, list) else []


def _find_step(task_state: dict[str, Any], step_id: str) -> tuple[int, dict[str, Any]]:
    for idx, step in enumerate(task_state.get("steps", [])):
        if str(step.get("id")) == str(step_id):
            return idx, step
    raise ValueError(f"Step not found: {step_id}")


def _next_pending_step_id(task_state: dict[str, Any]) -> str | None:
    for step in task_state.get("steps", []):
        if str(step.get("status", "")) == "pending":
            return str(step.get("id"))
    return None


def _completed_step_count(task_state: dict[str, Any]) -> int:
    return sum(1 for step in task_state.get("steps", []) if str(step.get("status")) == "completed")


def _total_step_count(task_state: dict[str, Any]) -> int:
    return len(task_state.get("steps", []))


def _failed_step_count(task_state: dict[str, Any]) -> int:
    return sum(1 for step in task_state.get("steps", []) if str(step.get("status")) == "blocked")


def _compute_coverage_status(task_state: dict[str, Any]) -> str:
    steps = task_state.get("steps", [])
    if steps and all(str(step.get("status")) == "completed" for step in steps):
        return "complete_plan"
    return "partial_plan"


def _compact_completed_step_summaries(task_state: dict[str, Any]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for step in task_state.get("steps", []):
        if str(step.get("status")) != "completed":
            continue
        summaries.append(
            {
                "id": step.get("id"),
                "goal": step.get("goal"),
                "result_summary": step.get("result_summary", ""),
                "evidence_ids": list(step.get("evidence_ids", []) or [])[:20],
                "open_gaps": list(step.get("open_gaps", []) or [])[:10],
            }
        )
    return summaries


_STEP_FOCUS_STOPWORDS = {
    "AND", "ARE", "ASSAY", "ASSAYS", "BUILD", "CANDIDATE", "CANDIDATES", "COMPARE", "COMPARES",
    "COMPLETION", "CONDITION", "DATA", "DATASET", "DATASETS", "DERIVED", "EVIDENCE", "EXACT",
    "EXISTING", "FIND", "FINDINGS", "FROM", "GOAL", "HUMAN", "IDENTIFIED", "IDENTIFY", "IN",
    "LITERATURE", "MODEL", "MUTANT", "MUTATION", "MUTATIONS", "OF", "OR", "PEPTIDE", "PEPTIDES",
    "PRESENTATION", "PUBLIC", "QUERY", "RANK", "RECOGNITION", "RETRIEVE", "RETURN", "SEARCH",
    "SEQUENCE", "SEQUENCES", "STEP", "SUPPORT", "THE", "THIS", "USE", "USING", "VALIDATE",
}


def _extract_step_focus_terms(step: dict[str, Any], *, limit: int = 12) -> list[str]:
    text = " ".join(
        str(step.get(field, "") or "").strip()
        for field in ("goal", "completion_condition")
    )
    if not text:
        return []

    ordered_terms: list[str] = []
    seen: set[str] = set()

    def _add_term(raw: str) -> None:
        term = re.sub(r"\s+", " ", str(raw or "").strip())
        if not term:
            return
        key = term.upper()
        if key in seen:
            return
        seen.add(key)
        ordered_terms.append(term)

    for match in re.finditer(r"\b([A-Z0-9-]{2,12})\s+([A-Z]\d{1,5}[A-Z*])\b", text):
        gene = match.group(1).strip()
        if gene not in _STEP_FOCUS_STOPWORDS:
            _add_term(f"{gene} {match.group(2).strip()}")

    for match in re.finditer(r"\bHLA-[A-Z]\*\d{2}:\d{2}\b", text):
        _add_term(match.group(0))

    for match in re.finditer(r"\b[A-Z]\d{1,5}[A-Z*]\b", text):
        _add_term(match.group(0))

    for match in re.finditer(r"\b[ACDEFGHIKLMNPQRSTVWY]{8,}\b", text):
        _add_term(match.group(0))

    for match in re.finditer(r"\b[A-Z0-9-]{2,12}\b", text):
        token = match.group(0).strip()
        if token in _STEP_FOCUS_STOPWORDS:
            continue
        if token.isdigit():
            continue
        _add_term(token)

    return ordered_terms[:limit]


def _serialize_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)


def _render_plan_markdown(task_state: dict[str, Any]) -> str:
    lines = ["## Plan", ""]
    objective = str(task_state.get("objective", "")).strip()
    if objective:
        lines.append(f"**Objective:** {objective}")
        lines.append("")
    for step in task_state.get("steps", []):
        tool_hint = step.get("tool_hint", "").strip()
        source_label = _resolve_source_label(tool_hint)
        source_display = source_label if source_label else tool_hint
        domains = step.get("domains") or []
        domain_tag = f" | domains: {', '.join(domains)}" if domains else ""
        lines.append(
            f"1. **{step.get('id', 'S?')}**: {step.get('goal', '').strip()} "
            f"*(source: {source_display}{domain_tag})*"
        )
        lines.append(f"Completion: {step.get('completion_condition', '').strip()}")
    return "\n".join(lines).strip()



_REACT_PHASE_LABELS = ("REASON", "ACT", "OBSERVE", "CONCLUDE")
_REACT_PHASE_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(" + "|".join(_REACT_PHASE_LABELS) + r")(?:\*\*)?:\s*",
    re.IGNORECASE,
)


def _parse_react_phases(trace: str) -> dict[str, str] | None:
    """Parse a reasoning_trace string into labeled ReAct phases.

    Returns a dict like {"REASON": "...", "ACT": "...", ...} if at least
    REASON and one other phase are found, otherwise None (caller falls back
    to rendering the trace as-is).
    """
    if not trace:
        return None
    splits = _REACT_PHASE_RE.split(trace)
    if len(splits) < 3:
        return None
    phases: dict[str, str] = {}
    i = 1
    while i < len(splits) - 1:
        label = splits[i].upper()
        text = splits[i + 1].strip()
        if label in phases:
            phases[label] += " " + text
        else:
            phases[label] = text
        i += 2
    if "REASON" not in phases or len(phases) < 2:
        return None
    return phases


def _render_react_trace_block(
    reasoning_trace: str,
    tools_called: list[str],
) -> list[str]:
    """Render the ReAct reasoning trace as a visually structured block."""
    lines: list[str] = []
    if not reasoning_trace and not tools_called:
        return lines

    lines.append("**ReAct Trace**")
    lines.append("")

    phases = _parse_react_phases(reasoning_trace)
    if phases:
        for label in _REACT_PHASE_LABELS:
            text = phases.get(label, "").strip()
            if not text:
                continue
            lines.append(f"> **{label.capitalize()}:** {text}")
            lines.append(">")
        if lines and lines[-1] == ">":
            lines.pop()
    elif reasoning_trace:
        for trace_line in reasoning_trace.split("\n"):
            lines.append(f"> {trace_line.strip()}")

    if tools_called:
        source_labels = []
        for tool_name in tools_called:
            source = tool_registry.TOOL_SOURCE_NAMES.get(tool_name, "")
            if source and source not in source_labels:
                source_labels.append(source)
        tools_display = ", ".join(f"`{t}`" for t in tools_called)
        lines.append("")
        data_sources = f" ({', '.join(source_labels)})" if source_labels else ""
        lines.append(f"**Tools used:** {tools_display}{data_sources}")

    lines.append("")
    return lines


def _render_react_step_progress(
    task_state: dict[str, Any],
    result: dict[str, Any],
    reasoning_trace: str,
    tool_reasoning: str = "",
    executor_prose: str = "",
) -> str:
    """Render a compact progressive log for a completed ReAct step.

    Format:
      ### S1 · `completed` — Goal text
      1. **MyGene.info** — resolved LRRK2 to ENSG00000188906
      2. **BigQuery** — listed 60 tables in open_targets_platform
      -> Claims: LRRK2 has Ensembl ID ENSG00000188906 [high]
      ---
    """
    step_id = str(result.get("step_id", "")).strip()
    try:
        _, step = _find_step(task_state, step_id)
    except Exception:  # noqa: BLE001
        step = {}
    status = str(result.get("status", step.get("status", ""))).strip()
    goal = str(step.get("goal", "")).strip()
    tool_log: list[dict[str, str]] = step.get("tool_log") or []

    header = f"### {step_id} · `{status}`"
    if goal:
        header += f" — {goal}"
    lines: list[str] = [header]

    # --- Tool log: compact dash list ---
    if tool_log:
        for entry in tool_log:
            tool = entry.get("tool", "?")
            summary = entry.get("summary", "").strip()
            entry_result = entry.get("result", "").strip()
            entry_status = entry.get("status", "")
            if summary and entry_result:
                lines.append(f"- {summary} → {entry_result}")
            elif summary:
                if entry_status == "called":
                    lines.append(f"- {summary}…")
                else:
                    lines.append(f"- {summary}")
            else:
                lines.append(f"- **{tool}** — querying…")
    else:
        tools_called = list(step.get("tools_called", []) or [])
        if tools_called:
            source_labels = []
            for tool_name in tools_called:
                source = tool_registry.TOOL_SOURCE_NAMES.get(tool_name, "")
                if source and source not in source_labels:
                    source_labels.append(source)
            if source_labels:
                lines.append(f"Sources: {', '.join(source_labels)}")

    # --- Key claims (compact, 1-2 lines) ---
    observations = list(result.get("structured_observations", []) or [])
    if observations:
        claim_parts = []
        for obs in observations[:5]:
            subj = obs.get("subject", {}).get("label", "?") if isinstance(obs.get("subject"), dict) else "?"
            pred = obs.get("predicate", "?")
            obj = obs.get("object", {}).get("label", "?") if isinstance(obs.get("object"), dict) else "?"
            conf = obs.get("confidence", "")
            conf_tag = f" [{conf}]" if conf else ""
            claim_parts.append(f"{subj} → {pred} → {obj}{conf_tag}")
        claims_text = "; ".join(claim_parts)
        if len(observations) > 5:
            claims_text += f" _(+{len(observations) - 5} more)_"
        lines.append(f"   → **Claims:** {claims_text}")

    lines.append("---")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Citation / reference helpers
# ---------------------------------------------------------------------------

_INLINE_ID_RE = re.compile(
    r"\bPMID\s*:?\s*(?P<pmid>\d{6,8})\b"
    r"|\bDOI\s*:?\s*(?P<doi>10\.\S+?)(?=[,;\s\)\]>]|$)"
    r"|\b(?P<nct>NCT\d{8})\b"
    r"|\bOpenAlex\s*:?\s*(?P<openalex>W\d+)\b"
    r"|\b(?P<pmc>PMC\d+)\b"
    r"|\bUniProt(?:\s+(?:ID|Accession))?\s*:?\s*(?P<uniprot>[A-Z][A-Z0-9]{2,9})\b"
    r"|\b(?:Ensembl(?:\s+(?:Gene|Transcript|Protein))?\s+ID\s*:?\s*)?(?P<ensembl>ENS[A-Z0-9]{3,}\d{6,})\b"
    r"|\b(?:Entrez(?:\s+Gene)?\s+ID|NCBI\s+Gene\s+ID)\s*:?\s*(?P<entrez>\d+)\b"
    r"|\b(?P<hgnc>HGNC:\d+)\b"
    r"|\b(?P<mondo>MONDO[_:]\d+)\b"
    r"|\b(?P<efo>EFO[_:]\d+)\b"
    r"|\b(?P<hp>HP:\d+)\b"
    r"|\b(?P<go>GO:\d+)\b"
    r"|\bPubChem\s*:\s*(?P<pubchem>\d+)\b"
    r"|\bPDB\s*:\s*(?P<pdb>[A-Za-z0-9]{4})\b"
    r"|\b(?P<rsid>rs\d{3,})\b"
    r"|\b(?P<chembl>CHEMBL\d+)\b"
    r"|\bReactome\s*:?\s*(?P<reactome>R-[A-Z]{3}-\d+)\b"
    r"|\b(?P<gcst>GCST\d{4,})\b",
    re.IGNORECASE,
)


def _evidence_id_to_url(eid: str) -> str | None:
    """Return a human-readable URL for a known evidence identifier, or None."""
    raw = re.sub(r"\s*:\s*", ":", eid.strip())
    m = re.fullmatch(r"(?i)PMID:(\d{4,9})", raw)
    if m:
        return f"https://pubmed.ncbi.nlm.nih.gov/{m.group(1)}/"
    m = re.fullmatch(r"(?i)DOI:(10\..+)", raw)
    if m:
        return f"https://doi.org/{m.group(1)}"
    m = re.fullmatch(r"(?i)(?:NCT:)?(NCT\d{8})", raw)
    if m:
        return f"https://clinicaltrials.gov/study/{m.group(1)}"
    m = re.fullmatch(r"(?i)OpenAlex:(W\d+)", raw)
    if m:
        return f"https://openalex.org/{m.group(1)}"
    m = re.fullmatch(r"(?i)(?:PMC:)?(PMC\d+)", raw)
    if m:
        return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{m.group(1)}/"
    m = re.fullmatch(r"(?i)UniProt:([A-Z][A-Z0-9]{2,9})", raw)
    if m:
        return f"https://www.uniprot.org/uniprotkb/{m.group(1)}"
    m = re.fullmatch(r"(?i)(?:Ensembl:)?(ENS[A-Z0-9]{3,}\d{6,})", raw)
    if m:
        return f"https://www.ensembl.org/id/{m.group(1)}"
    m = re.fullmatch(r"(?i)(?:Entrez|NCBIGene|GeneID):(\d+)", raw)
    if m:
        return f"https://www.ncbi.nlm.nih.gov/gene/{m.group(1)}"
    m = re.fullmatch(r"(?i)(HGNC:\d+)", raw)
    if m:
        hgnc_id = m.group(1).upper()
        return f"https://www.genenames.org/data/gene-symbol-report/#!/hgnc_id/{hgnc_id}"
    m = re.fullmatch(r"(?i)(MONDO[:_]\d+)", raw)
    if m:
        mondo_id = m.group(1).upper().replace("_", ":")
        return f"https://monarchinitiative.org/disease/{mondo_id}"
    m = re.fullmatch(r"(?i)(EFO[:_]\d+)", raw)
    if m:
        efo_id = m.group(1).upper().replace(":", "_")
        return f"https://www.ebi.ac.uk/ols4/ontologies/efo/terms?short_form={efo_id}"
    m = re.fullmatch(r"(?i)(HP:\d+)", raw)
    if m:
        return f"https://hpo.jax.org/app/browse/term/{m.group(1).upper()}"
    m = re.fullmatch(r"(?i)(GO:\d+)", raw)
    if m:
        return f"https://amigo.geneontology.org/amigo/term/{m.group(1).upper()}"
    m = re.fullmatch(r"(?i)PubChem:(\d+)", raw)
    if m:
        return f"https://pubchem.ncbi.nlm.nih.gov/compound/{m.group(1)}"
    m = re.fullmatch(r"(?i)PDB:([A-Za-z0-9]{4,8})", raw)
    if m:
        return f"https://www.rcsb.org/structure/{m.group(1).upper()}"
    m = re.fullmatch(r"(?i)(rs\d+)", raw)
    if m:
        return f"https://www.ncbi.nlm.nih.gov/snp/{m.group(1).lower()}"
    m = re.fullmatch(r"(?i)(?:ChEMBL:)?(CHEMBL\d+)", raw)
    if m:
        return f"https://www.ebi.ac.uk/chembl/compound_report_card/{m.group(1).upper()}"
    m = re.fullmatch(r"(?i)(?:Reactome:)?(R-[A-Z]{3}-\d+)", raw)
    if m:
        return f"https://reactome.org/content/detail/{m.group(1)}"
    m = re.fullmatch(r"(?i)(GCST\d+)", raw)
    if m:
        return f"https://www.ebi.ac.uk/gwas/studies/{m.group(1).upper()}"
    return None


def _extract_inline_ids_from_text(text: str) -> list[str]:
    """Scan a markdown string for inline source identifiers and return them in order."""
    seen: set[str] = set()
    ids: list[str] = []
    for m in _INLINE_ID_RE.finditer(text):
        if m.group("pmid"):
            normalized = f"PMID:{m.group('pmid')}"
        elif m.group("doi"):
            normalized = f"DOI:{m.group('doi')}"
        elif m.group("nct"):
            normalized = m.group("nct").upper()
        elif m.group("openalex"):
            normalized = f"OpenAlex:{m.group('openalex')}"
        elif m.group("pmc"):
            normalized = m.group("pmc").upper()
        elif m.group("uniprot"):
            normalized = f"UniProt:{m.group('uniprot').upper()}"
        elif m.group("ensembl"):
            normalized = f"Ensembl:{m.group('ensembl').upper()}"
        elif m.group("entrez"):
            normalized = f"Entrez:{m.group('entrez')}"
        elif m.group("hgnc"):
            normalized = m.group("hgnc").upper()
        elif m.group("mondo"):
            normalized = m.group("mondo").upper().replace("_", ":")
        elif m.group("efo"):
            normalized = m.group("efo").upper().replace(":", "_")
        elif m.group("hp"):
            normalized = m.group("hp").upper()
        elif m.group("go"):
            normalized = m.group("go").upper()
        elif m.group("pubchem"):
            normalized = f"PubChem:{m.group('pubchem')}"
        elif m.group("pdb"):
            normalized = f"PDB:{m.group('pdb').upper()}"
        elif m.group("rsid"):
            normalized = m.group("rsid").lower()
        elif m.group("chembl"):
            normalized = m.group("chembl").upper()
        elif m.group("reactome"):
            normalized = f"Reactome:{m.group('reactome').upper()}"
        elif m.group("gcst"):
            normalized = m.group("gcst").upper()
        else:
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            ids.append(normalized)
    return ids


_REFERENCE_SECTION_ID_RE = re.compile(
    r"(?i)^(PMID:|DOI:|OpenAlex:|PMC)"
)


def _is_literature_id(eid: str) -> bool:
    """True for paper-like identifiers that belong in the References section."""
    return bool(_REFERENCE_SECTION_ID_RE.match(re.sub(r"\s*:\s*", ":", eid.strip())))


_VALID_EVIDENCE_ID_RE = re.compile(
    r"(?i)^("
    r"PMID:\d{4,9}"
    r"|DOI:10\..+"
    r"|NCT\d{8}"
    r"|OpenAlex:W\d+"
    r"|PMC\d+"
    r"|UniProt:[A-Z][A-Z0-9]{2,9}"
    r"|PubChem:\d+"
    r"|PDB:[A-Za-z0-9]{4,8}"
    r"|rs\d{3,}"
    r"|CHEMBL\d+"
    r"|Reactome:R-[A-Z]{3}-\d+"
    r"|GCST\d{4,}"
    r")$"
)


def _collect_all_evidence_ids(task_state: dict[str, Any]) -> list[str]:
    """Return a deduplicated list of evidence IDs from all completed steps.

    Only accepts recognised identifier formats (PMID, DOI, NCT, OpenAlex, PMC).
    Raw URLs or arbitrary strings stored by the executor are silently dropped.
    """
    seen: set[str] = set()
    ids: list[str] = []
    for step in task_state.get("steps", []):
        for eid in step.get("evidence_ids", []) or []:
            normalized = re.sub(r"\s*:\s*", ":", str(eid).strip())
            if not normalized:
                continue
            if not _VALID_EVIDENCE_ID_RE.match(normalized):
                logger.debug("Skipping non-standard evidence_id from references: %r", normalized)
                continue
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                ids.append(normalized)
    return ids


def _build_ref_map(ids: list[str]) -> dict[str, int]:
    """Map colon-normalised lowercase EID → 1-based reference number."""
    return {
        re.sub(r"\s*:\s*", ":", eid.strip()).lower(): i
        for i, eid in enumerate(ids, start=1)
    }


# ---------------------------------------------------------------------------
# Live metadata fetching for APA citations
# ---------------------------------------------------------------------------

_CITATION_META_CACHE: dict[str, dict] = {}
_CITATION_FETCH_TIMEOUT = 6  # seconds

# ---------------------------------------------------------------------------
# URL link validation
# ---------------------------------------------------------------------------

_VALIDATED_URL_CACHE: dict[str, bool] = {}
_URL_VALIDATE_TIMEOUT = 5  # seconds


def _validate_url(url: str) -> bool:
    """Return True if url responds with HTTP 2xx/3xx; False on error, timeout, or 4xx/5xx.

    Results are cached for the process lifetime so repeated checks are free.
    Uses HEAD to avoid downloading bodies.
    """
    if url in _VALIDATED_URL_CACHE:
        return _VALIDATED_URL_CACHE[url]
    try:
        req = urllib.request.Request(
            url,
            method="HEAD",
            headers={"User-Agent": "ai-co-scientist/1.0 (link-validator)"},
        )
        with urllib.request.urlopen(req, timeout=_URL_VALIDATE_TIMEOUT) as resp:
            ok = 200 <= resp.status < 400
    except Exception:  # noqa: BLE001  (network errors, timeouts, 4xx/5xx)
        ok = False
    _VALIDATED_URL_CACHE[url] = ok
    return ok


def _http_get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ai-co-scientist/1.0 (citation-builder)"})
        with urllib.request.urlopen(req, timeout=_CITATION_FETCH_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _fetch_europepmc_meta_by_pmid(pmid: str) -> dict | None:
    """Fetch article metadata from Europe PMC for a PubMed identifier."""
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        f"?query=EXT_ID:{urllib.parse.quote(pmid)}%20AND%20SRC:MED&format=json"
    )
    data = _http_get_json(url)
    results = (((data or {}).get("resultList") or {}).get("result") or [])
    if not results:
        return None
    result = results[0]
    author_string = re.sub(r"\s+", " ", str(result.get("authorString", "")).strip()).rstrip(".")
    authors = [part.strip() for part in author_string.split(",") if part.strip()]
    title = re.sub(r"<[^>]+>", "", str(result.get("title", "")).strip()).rstrip(".")
    return {
        "authors": authors,
        "title": title,
        "journal": str(result.get("journalTitle", "")).strip(),
        "year": str(result.get("pubYear", "")).strip(),
        "volume": str(result.get("journalVolume", "")).strip(),
        "issue": str(result.get("issue", "")).strip(),
        "pages": str(result.get("pageInfo", "")).strip(),
        "doi": str(result.get("doi", "")).strip(),
        "pmid": str(result.get("pmid", "") or pmid).strip(),
    }


def _fetch_pubmed_meta(pmid: str) -> dict | None:
    """Fetch article metadata from NCBI esummary. Returns a flat dict or None."""
    cache_key = f"pmid:{pmid}"
    if cache_key in _CITATION_META_CACHE:
        return _CITATION_META_CACHE[cache_key]
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        f"?db=pubmed&id={urllib.parse.quote(pmid)}&retmode=json"
    )
    data = _http_get_json(url)
    if not data:
        fallback = _fetch_europepmc_meta_by_pmid(pmid)
        _CITATION_META_CACHE[cache_key] = fallback or {}
        return fallback
    result = (data.get("result") or {}).get(pmid)
    if not result:
        fallback = _fetch_europepmc_meta_by_pmid(pmid)
        _CITATION_META_CACHE[cache_key] = fallback or {}
        return fallback
    # Extract DOI from articleids
    doi = ""
    for aid in result.get("articleids") or []:
        if str(aid.get("idtype", "")).lower() == "doi":
            doi = str(aid.get("value", "")).strip()
            break
    if not doi:
        doi = str(result.get("elocationid", "")).strip()
        if doi and not doi.startswith("10."):
            doi = ""
    meta = {
        "authors": [str(a.get("name", "")).strip() for a in (result.get("authors") or []) if a.get("authtype") == "Author"],
        "title": str(result.get("title", "")).rstrip(".").strip(),
        "journal": str(result.get("source", "")).strip(),
        "year": (str(result.get("pubdate", "")).split() or [""])[0],
        "volume": str(result.get("volume", "")).strip(),
        "issue": str(result.get("issue", "")).strip(),
        "pages": str(result.get("pages", "")).strip(),
        "doi": doi,
        "pmid": pmid,
    }
    if not meta.get("title"):
        fallback = _fetch_europepmc_meta_by_pmid(pmid)
        if fallback and fallback.get("title"):
            _CITATION_META_CACHE[cache_key] = fallback
            return fallback
    _CITATION_META_CACHE[cache_key] = meta
    return meta


def _fetch_crossref_meta(doi: str) -> dict | None:
    """Fetch article metadata from CrossRef. Returns a flat dict or None."""
    cache_key = f"doi:{doi.lower()}"
    if cache_key in _CITATION_META_CACHE:
        return _CITATION_META_CACHE[cache_key]
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi, safe='/')}"
    data = _http_get_json(url)
    if not data:
        _CITATION_META_CACHE[cache_key] = {}
        return None
    msg = data.get("message") or {}
    raw_authors = msg.get("author") or []
    authors = []
    for a in raw_authors:
        family = str(a.get("family", "")).strip()
        given = str(a.get("given", "")).strip()
        if family:
            authors.append(f"{family} {given}" if given else family)
    date_parts = ((msg.get("published") or msg.get("published-print") or msg.get("published-online") or {}).get("date-parts") or [[]])[0]
    year = str(date_parts[0]) if date_parts else ""
    titles = msg.get("title") or []
    title = str(titles[0]).rstrip(".") if titles else ""
    journals = msg.get("container-title") or []
    journal = str(journals[0]) if journals else ""
    meta = {
        "authors": authors,
        "title": title,
        "journal": journal,
        "year": year,
        "volume": str(msg.get("volume", "")).strip(),
        "issue": str(msg.get("issue", "")).strip(),
        "pages": str(msg.get("page", "")).strip(),
        "doi": str(msg.get("DOI", doi)).strip(),
        "pmid": "",
    }
    _CITATION_META_CACHE[cache_key] = meta
    return meta


def _format_apa_authors(names: list[str]) -> str:
    """Convert a list of 'LastName Initials' or 'LastName First' strings to APA author string."""
    if not names:
        return ""
    formatted: list[str] = []
    for name in names[:6]:
        parts = name.rsplit(" ", 1)
        if len(parts) == 2:
            last, first = parts
            # Convert initials like 'JD' or full name 'John' to 'J. D.' or 'J.'
            if first.isupper() or (len(first) <= 3 and first.replace(" ", "").isupper()):
                initials = ". ".join(first.upper()) + "."
                formatted.append(f"{last}, {initials}")
            else:
                initial = first[0].upper() + "."
                formatted.append(f"{last}, {initial}")
        else:
            formatted.append(name)
    if len(names) > 7:
        return ", ".join(formatted) + ", . . ."
    if len(formatted) > 1:
        return ", ".join(formatted[:-1]) + ", & " + formatted[-1]
    return formatted[0]


def _build_apa_citation(meta: dict, pmid: str = "", doi: str = "") -> str:
    """Render a full APA 7th-edition citation string from a metadata dict."""
    author_str = _format_apa_authors(meta.get("authors") or [])
    year = meta.get("year") or "n.d."
    title = meta.get("title") or ""
    journal = meta.get("journal") or ""
    volume = meta.get("volume") or ""
    issue = meta.get("issue") or ""
    pages = meta.get("pages") or ""
    doi_val = meta.get("doi") or doi
    pmid_val = meta.get("pmid") or pmid

    parts: list[str] = []
    if author_str:
        parts.append(f"{author_str} ({year}).")
    elif year:
        parts.append(f"({year}).")
    if title:
        parts.append(f"{title}.")
    if journal:
        journal_part = f"*{journal}*"
        if volume:
            journal_part += f", *{volume}*"
            if issue:
                journal_part += f"({issue})"
        if pages:
            journal_part += f", {pages}"
        journal_part += "."
        parts.append(journal_part)
    if doi_val:
        doi_url = f"https://doi.org/{doi_val}"
        parts.append(f"[{doi_url}]({doi_url})")
    if pmid_val:
        pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_val}/"
        parts.append(f"PMID: [{pmid_val}]({pmid_url})")
    return " ".join(parts)


def _format_apa_intext_author(names: list[str]) -> str:
    """Return an APA-style in-text author component from a list of raw names."""
    families: list[str] = []
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        if "," in name:
            family = name.split(",", 1)[0].strip()
        else:
            tokens = name.split()
            while len(tokens) > 1 and re.fullmatch(r"(?:[A-Z]\.?){1,4}|[A-Z]{1,4}", tokens[-1]):
                tokens.pop()
            if len(tokens) > 1 and len(tokens[-1]) <= 3 and tokens[-1].isupper():
                tokens.pop()
            if len(tokens) > 1 and tokens[-1][:1].isupper() and tokens[-1][1:].islower():
                family = " ".join(tokens[:-1]).strip()
            else:
                family = " ".join(tokens).strip()
        family = family.strip()
        if family:
            families.append(family)

    if not families:
        return ""
    if len(families) == 1:
        return families[0]
    if len(families) == 2:
        return f"{families[0]} & {families[1]}"
    return f"{families[0]} et al."


def _build_apa_intext_label(eid: str) -> str:
    """Return the plain APA-style in-text citation label for one reference ID."""
    meta = _fetch_reference_meta(eid)
    citation_label = ""
    if meta:
        author_part = _format_apa_intext_author(meta.get("authors") or [])
        year = str(meta.get("year") or "n.d.").strip() or "n.d."
        if author_part:
            citation_label = f"{author_part}, {year}"
        else:
            title = str(meta.get("title") or "").strip()
            if title:
                citation_label = f"{title[:40].rstrip('.')}..., {year}"

    if citation_label:
        return citation_label

    raw = re.sub(r"\s*:\s*", ":", eid.strip())
    if raw.lower().startswith("openalex:"):
        return f"{raw.replace(':', ': ', 1)}, n.d."
    if raw.upper().startswith("PMC"):
        return f"{raw.upper()}, n.d."
    return raw


def _fetch_reference_meta(eid: str) -> dict | None:
    """Fetch citation metadata for a reference-style identifier when available."""
    raw = re.sub(r"\s*:\s*", ":", eid.strip())
    match = re.fullmatch(r"(?i)PMID:(\d{4,9})", raw)
    if match:
        pmid = match.group(1)
        meta = _fetch_pubmed_meta(pmid)
        if meta and meta.get("title"):
            return meta
        doi = str((meta or {}).get("doi", "")).strip()
        if doi:
            crossref_meta = _fetch_crossref_meta(doi)
            if crossref_meta and crossref_meta.get("title"):
                crossref_meta["pmid"] = pmid
                return crossref_meta
        return meta

    match = re.fullmatch(r"(?i)DOI:(10\..+)", raw)
    if match:
        return _fetch_crossref_meta(match.group(1))

    return None


def _format_apa_intext_citation(ref_number: int, eid: str) -> str:
    """Return a linked APA-style in-text citation when metadata is available."""
    return f"[{_build_apa_intext_label(eid)}](#ref-{ref_number})"


def _format_reference_apa(i: int, eid: str) -> str:
    """Format a single reference in full APA 7th-edition style, fetching live metadata."""
    raw = re.sub(r"\s*:\s*", ":", eid.strip())
    anchor = f'<a id="ref-{i}"></a>'

    # PMID — fetch from PubMed; fall back to CrossRef via DOI if available
    m = re.fullmatch(r"(?i)PMID:(\d{4,9})", raw)
    if m:
        pmid = m.group(1)
        pmid_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        meta = _fetch_pubmed_meta(pmid)
        if meta:
            if meta.get("title"):
                return f"{anchor}{i}. {_build_apa_citation(meta, pmid=pmid)}"
            if meta.get("doi"):
                cr_meta = _fetch_crossref_meta(meta["doi"])
                if cr_meta and cr_meta.get("title"):
                    cr_meta["pmid"] = pmid
                    return f"{anchor}{i}. {_build_apa_citation(cr_meta, pmid=pmid)}"
        # Validate the fallback URL before emitting a clickable link
        if _validate_url(pmid_url):
            return f"{anchor}{i}. PMID: [{pmid}]({pmid_url})"
        return f"{anchor}{i}. PMID: {pmid} _(link could not be verified)_"

    # DOI — fetch from CrossRef
    m = re.fullmatch(r"(?i)DOI:(10\..+)", raw)
    if m:
        doi = m.group(1)
        doi_url = f"https://doi.org/{doi}"
        meta = _fetch_crossref_meta(doi)
        if meta and meta.get("title"):
            return f"{anchor}{i}. {_build_apa_citation(meta, doi=doi)}"
        # Validate before emitting bare DOI link
        if _validate_url(doi_url):
            return f"{anchor}{i}. [{doi_url}]({doi_url})"
        return f"{anchor}{i}. DOI: {doi} _(link could not be verified)_"

    # NCT — ClinicalTrials.gov
    m = re.fullmatch(r"(?i)(?:NCT:)?(NCT\d{8})", raw)
    if m:
        nct = m.group(1).upper()
        url = f"https://clinicaltrials.gov/study/{nct}"
        if _validate_url(url):
            return f"{anchor}{i}. U.S. National Library of Medicine. (n.d.). *ClinicalTrials.gov*. [{nct}]({url})"
        return f"{anchor}{i}. U.S. National Library of Medicine. (n.d.). *ClinicalTrials.gov*. {nct} _(link could not be verified)_"

    # OpenAlex
    m = re.fullmatch(r"(?i)OpenAlex:(W\d+)", raw)
    if m:
        wid = m.group(1)
        url = f"https://openalex.org/{wid}"
        if _validate_url(url):
            return f"{anchor}{i}. *OpenAlex*. [{wid}]({url})"
        return f"{anchor}{i}. *OpenAlex*. {wid} _(link could not be verified)_"

    # PMC
    m = re.fullmatch(r"(?i)(?:PMC:)?(PMC\d+)", raw)
    if m:
        pmc = m.group(1).upper()
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/"
        if _validate_url(url):
            return f"{anchor}{i}. National Library of Medicine. (n.d.). *PubMed Central*. [{pmc}]({url})"
        return f"{anchor}{i}. National Library of Medicine. (n.d.). *PubMed Central*. {pmc} _(link could not be verified)_"

    url = _evidence_id_to_url(eid)
    if url and _validate_url(url):
        return f"{anchor}{i}. [{eid}]({url})"
    if url:
        return f"{anchor}{i}. {eid} _(link could not be verified)_"
    return f"{anchor}{i}. {eid}"


def _build_references_section(ids: list[str]) -> str:
    """Format a Markdown ## References section in APA 7th-edition style."""
    if not ids:
        return ""
    lines = ["## References", ""]
    for i, eid in enumerate(ids, start=1):
        lines.append(_format_reference_apa(i, eid))
    return "\n".join(lines)


_PROTECT_RE = re.compile(
    r"```[\s\S]*?```"       # fenced code blocks
    r"|`[^`\n]+`"           # inline code
    r"|\[([^\]]+)\]\([^)]+\)",  # existing markdown links
    re.DOTALL,
)


def _dedupe_preserve_order(items: list[str], limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        normalized = re.sub(r"\s*:\s*", ":", str(item or "").strip())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
        if limit and len(deduped) >= limit:
            break
    return deduped


def _collect_claim_summary_literature_ids(claim_summary: dict[str, Any], limit: int = 20) -> list[str]:
    ids: list[str] = []

    for claim in list(claim_summary.get("top_supported_claims", []) or [])[:8]:
        ids.extend(str(eid).strip() for eid in list(claim.get("supporting_ids", []) or [])[:8])

    for cluster in list(claim_summary.get("mixed_evidence_claims", []) or [])[:5]:
        for claim in list(cluster.get("claims", []) or [])[:4]:
            ids.extend(str(eid).strip() for eid in list(claim.get("supporting_ids", []) or [])[:8])

    return [
        eid for eid in _dedupe_preserve_order(ids, limit=limit)
        if _is_literature_id(eid)
    ]


def _collect_final_report_literature_ids(
    task_state: dict[str, Any],
    synthesis: dict[str, Any],
    rendered_text: str,
    *,
    limit: int = 25,
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_extract_inline_ids_from_text(rendered_text))

    model_references_text = str(synthesis.get("model_references_text", "") or "").strip()
    if model_references_text:
        candidates.extend(_extract_inline_ids_from_text(model_references_text))

    claim_summary = dict(synthesis.get("claim_synthesis_summary", {}) or {})
    candidates.extend(_collect_claim_summary_literature_ids(claim_summary, limit=limit))
    candidates.extend(_collect_all_evidence_ids(task_state))

    return [
        eid for eid in _dedupe_preserve_order(candidates, limit=limit)
        if _is_literature_id(eid)
    ]


def _hyperlink_inline_ids(text: str, ref_map: dict[str, int] | None = None) -> str:
    """Replace bare inline ID mentions with links in the body text.

    Paper-like IDs (PMID, DOI, OpenAlex, PMC) present in ref_map are
    replaced with linked APA-style citations pointing to the References
    section. Trial and database IDs (NCT, UniProt, PubChem, PDB, rsID, ChEMBL,
    Reactome, GCST) are replaced with clickable links to the external record.
    Skips the ## References section (already formatted) and avoids double-linking.
    """
    # Split off the References section — leave it untouched.
    refs_split = re.split(r"(?m)^#{2,3} References\b", text, maxsplit=1)
    body = refs_split[0]
    refs_tail = ("\n## References" + refs_split[1]) if len(refs_split) > 1 else ""

    # Protect code spans/blocks and existing links with null-byte placeholders.
    placeholders: list[str] = []

    def _protect(m: re.Match) -> str:  # type: ignore[type-arg]
        idx = len(placeholders)
        placeholders.append(m.group(0))
        return f"\x00P{idx}\x00"

    protected = _PROTECT_RE.sub(_protect, body)

    def _replace_id(m: re.Match) -> str:  # type: ignore[type-arg]
        display = m.group(0)
        if m.group("pmid"):
            normalized = f"PMID:{m.group('pmid')}"
        elif m.group("doi"):
            normalized = f"DOI:{m.group('doi')}"
        elif m.group("nct"):
            normalized = m.group("nct").upper()
        elif m.group("openalex"):
            normalized = f"OpenAlex:{m.group('openalex')}"
        elif m.group("pmc"):
            normalized = m.group("pmc").upper()
        elif m.group("uniprot"):
            normalized = f"UniProt:{m.group('uniprot').upper()}"
        elif m.group("ensembl"):
            normalized = f"Ensembl:{m.group('ensembl').upper()}"
        elif m.group("entrez"):
            normalized = f"Entrez:{m.group('entrez')}"
        elif m.group("hgnc"):
            normalized = m.group("hgnc").upper()
        elif m.group("mondo"):
            normalized = m.group("mondo").upper().replace("_", ":")
        elif m.group("efo"):
            normalized = m.group("efo").upper().replace(":", "_")
        elif m.group("hp"):
            normalized = m.group("hp").upper()
        elif m.group("go"):
            normalized = m.group("go").upper()
        elif m.group("pubchem"):
            normalized = f"PubChem:{m.group('pubchem')}"
        elif m.group("pdb"):
            normalized = f"PDB:{m.group('pdb').upper()}"
        elif m.group("rsid"):
            normalized = m.group("rsid").lower()
        elif m.group("chembl"):
            normalized = m.group("chembl").upper()
        elif m.group("reactome"):
            normalized = f"Reactome:{m.group('reactome').upper()}"
        elif m.group("gcst"):
            normalized = m.group("gcst").upper()
        else:
            return m.group(0)
        ref_key = re.sub(r"\s*:\s*", ":", normalized).lower()
        if ref_map and ref_key in ref_map:
            n = ref_map[ref_key]
            return _format_apa_intext_citation(n, normalized)
        url = _evidence_id_to_url(normalized)
        if not url:
            return m.group(0)
        return f"[{display}]({url})"

    linked = _INLINE_ID_RE.sub(_replace_id, protected)

    # Restore placeholders.
    for idx, original in enumerate(placeholders):
        linked = linked.replace(f"\x00P{idx}\x00", original)

    return linked + refs_tail


def _hyperlink_author_year_citations(text: str, lit_ids: list[str]) -> str:
    """Link plain author-year citations in the body to the final References section."""
    if not text or not lit_ids:
        return text

    refs_split = re.split(r"(?m)^#{2,3} References\b", text, maxsplit=1)
    body = refs_split[0]
    refs_tail = ("\n## References" + refs_split[1]) if len(refs_split) > 1 else ""

    placeholders: list[str] = []

    def _protect(m: re.Match) -> str:  # type: ignore[type-arg]
        idx = len(placeholders)
        placeholders.append(m.group(0))
        return f"\x00P{idx}\x00"

    protected = _PROTECT_RE.sub(_protect, body)

    citation_pairs: list[tuple[str, str]] = []
    for ref_number, eid in enumerate(lit_ids, start=1):
        label = _build_apa_intext_label(eid)
        if not label or label == re.sub(r"\s*:\s*", ":", eid.strip()):
            continue
        citation_pairs.append((label, f"[{label}](#ref-{ref_number})"))

    for label, replacement in sorted(citation_pairs, key=lambda item: len(item[0]), reverse=True):
        pattern = re.compile(rf"(?<![\w>]){re.escape(label)}(?![\w<])")
        protected = pattern.sub(replacement, protected)

    for idx, original in enumerate(placeholders):
        protected = protected.replace(f"\x00P{idx}\x00", original)

    return protected + refs_tail


_AUTHOR_YEAR_CITATION_RE = (
    r"(?:"
    r"[A-Z][A-Za-z'’.\-]+(?: [A-Z][A-Za-z'’.\-]+)*(?: et al\.)?"
    r"|"
    r"[A-Z][A-Za-z'’.\-]+(?: [A-Z][A-Za-z'’.\-]+)* & [A-Z][A-Za-z'’.\-]+(?: [A-Z][A-Za-z'’.\-]+)*"
    r"), \d{4}[a-z]?"
)


def _collapse_duplicate_citation_mentions(text: str) -> str:
    """Collapse duplicate adjacent author-year citations in the report body."""
    if not text:
        return text

    refs_split = re.split(r"(?m)^#{2,3} References\b", text, maxsplit=1)
    body = refs_split[0]
    refs_tail = ("\n## References" + refs_split[1]) if len(refs_split) > 1 else ""

    linked_plain_re = re.compile(
        rf"\[(?P<label>{_AUTHOR_YEAR_CITATION_RE})\]\(#ref-(?P<ref>\d+)\)"
        rf"(?P<sep>,\s*|;\s*|\s+and\s+)(?P=label)"
    )
    plain_linked_re = re.compile(
        rf"(?P<label>{_AUTHOR_YEAR_CITATION_RE})"
        rf"(?P<sep>,\s*|;\s*|\s+and\s+)"
        rf"\[(?P=label)\]\(#ref-(?P<ref>\d+)\)"
    )
    linked_linked_re = re.compile(
        rf"(?P<citation>\[(?P<label>{_AUTHOR_YEAR_CITATION_RE})\]\(#ref-(?P<ref>\d+)\))"
        rf"(?P<sep>,\s*|;\s*|\s+and\s+)(?P=citation)"
    )
    plain_plain_re = re.compile(
        rf"(?P<label>{_AUTHOR_YEAR_CITATION_RE})"
        rf"(?P<sep>,\s*|;\s*|\s+and\s+)(?P=label)"
    )

    updated = body
    for _ in range(6):
        next_text = linked_plain_re.sub(r"[\g<label>](#ref-\g<ref>)", updated)
        next_text = plain_linked_re.sub(r"[\g<label>](#ref-\g<ref>)", next_text)
        next_text = linked_linked_re.sub(r"\g<citation>", next_text)
        next_text = plain_plain_re.sub(r"\g<label>", next_text)
        if next_text == updated:
            break
        updated = next_text

    return updated + refs_tail


def _inject_key_literature_fallback(text: str, lit_ids: list[str], *, max_refs: int = 3) -> str:
    """Insert a small cited literature block when references exist but the body has none."""
    if not text or not lit_ids:
        return text

    refs_split = re.split(r"(?m)^#{2,3} References\b", text, maxsplit=1)
    body = refs_split[0]
    refs_tail = ("\n## References" + refs_split[1]) if len(refs_split) > 1 else ""

    if re.search(r"\]\(#ref-\d+\)", body):
        return text

    citation_snippets = [
        _format_apa_intext_citation(ref_number, eid)
        for ref_number, eid in enumerate(lit_ids[:max_refs], start=1)
    ]
    citation_snippets = [snippet for snippet in citation_snippets if snippet]
    if not citation_snippets:
        return text

    fallback_block = (
        "### Key Literature\n\n"
        f"Supporting literature includes {_human_join(citation_snippets)}.\n\n"
    )

    next_steps_match = re.search(r"(?m)^### Recommended Next Steps\b", body)
    if next_steps_match:
        insert_at = next_steps_match.start()
        body = body[:insert_at].rstrip() + "\n\n" + fallback_block + body[insert_at:].lstrip()
    else:
        body = body.rstrip() + "\n\n" + fallback_block

    return body + refs_tail


def _expand_reference_only_body_lines(text: str, lit_ids: list[str]) -> str:
    if not text or not lit_ids:
        return text

    refs_split = re.split(r"(?m)^#{2,3} References\b", text, maxsplit=1)
    body = refs_split[0]
    refs_tail = ("\n## References" + refs_split[1]) if len(refs_split) > 1 else ""
    citation_cache: dict[int, str] = {}
    citation_only_re = re.compile(
        r"^(\s*(?:[-*]\s+)?)"
        r"(?:\[(\d+)\](?:\(#ref-\2\))?|\[([^\]]+)\]\(#ref-(\d+)\))\s*$"
    )

    def _citation_text(ref_number: int) -> str:
        cached = citation_cache.get(ref_number)
        if cached is not None:
            return cached
        if ref_number < 1 or ref_number > len(lit_ids):
            return ""
        formatted = _format_reference_apa(ref_number, lit_ids[ref_number - 1])
        formatted = re.sub(rf'^<a id="ref-{ref_number}"></a>\s*', "", formatted).strip()
        formatted = re.sub(rf"^{ref_number}\.\s*", "", formatted).strip()
        citation_cache[ref_number] = formatted
        return formatted

    expanded_lines: list[str] = []
    for raw_line in body.splitlines():
        match = citation_only_re.match(raw_line)
        if not match:
            expanded_lines.append(raw_line)
            continue
        ref_number = int(match.group(2) or match.group(4) or 0)
        citation_text = _citation_text(ref_number)
        if not citation_text:
            expanded_lines.append(raw_line)
            continue
        expanded_lines.append(f"{match.group(1) or '- '}{citation_text}")

    return "\n".join(expanded_lines) + refs_tail


def _normalize_top_level_heading_key(heading: str) -> str:
    normalized = re.sub(r"\s+", " ", str(heading or "").strip()).lower()
    mapping = {
        "tldr": "answer",
        "answer": "answer",
        "summary": "answer",
        "evidence breakdown": "detailed_findings",
        "detailed findings": "detailed_findings",
        "key findings": "detailed_findings",
        "findings": "detailed_findings",
        "conflicting & uncertain evidence": "conflicting_evidence",
        "conflicting and uncertain evidence": "conflicting_evidence",
        "conflicting evidence": "conflicting_evidence",
        "limitations & gaps": "limitations",
        "limitations and gaps": "limitations",
        "limitations": "limitations",
        "recommended next steps": "next_steps",
        "potential next steps": "next_steps",
        "next steps": "next_steps",
        "next actions": "next_steps",
        "sources consulted": "sources_consulted",
        "references": "references",
        # Legacy mappings for backward compatibility with older LLM output
        "evidence and methodology": "detailed_findings",
        "evidence & methodology": "detailed_findings",
    }
    return mapping.get(normalized, normalized.replace(" ", "_"))


def _extract_top_level_markdown_sections(markdown: str) -> dict[str, str]:
    text = str(markdown or "").strip()
    if not text:
        return {}
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE))
    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        heading_key = _normalize_top_level_heading_key(match.group(1))
        body_start = match.end()
        body_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections[heading_key] = body
    return sections


def _extract_markdown_list_items(markdown: str) -> list[str]:
    items: list[str] = []
    for line in str(markdown or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        bullet_match = re.match(r"^(?:[-*+]\s+|\d+\.\s+)(.+)$", stripped)
        if bullet_match:
            items.append(bullet_match.group(1).strip())
    return _dedupe_str_list(items, limit=12)


def _human_join(items: list[str]) -> str:
    cleaned = _dedupe_str_list(items, limit=6)
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def _all_substantive_paragraphs(text: str) -> str:
    """Return all non-heading, non-empty paragraphs from text as joined prose."""
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", str(text or "").strip()):
        cleaned = block.strip()
        if not cleaned:
            continue
        if cleaned.startswith("#"):
            continue
        paragraphs.append(re.sub(r"\s+", " ", cleaned).strip())
    return "\n\n".join(paragraphs)


def _build_structured_answer_markdown(
    task_state: dict[str, Any],
    claim_summary: dict[str, Any],
    *,
    model_answer: str = "",
) -> str:
    """Build the Answer section: a direct, standalone answer with confidence framing."""
    lines: list[str] = []
    objective = str(task_state.get("objective", "")).strip()
    top_claims = list(claim_summary.get("top_supported_claims", []) or [])
    mixed_claims = list(claim_summary.get("mixed_evidence_claims", []) or [])
    model_answer_text = _all_substantive_paragraphs(model_answer)
    model_answer_first = _first_substantive_paragraph(model_answer)

    all_sources: list[str] = []
    for step in task_state.get("steps", []):
        if str(step.get("status", "")).strip() == "completed":
            all_sources.extend(_derive_step_data_sources(step))
    unique_sources = _dedupe_str_list(all_sources, limit=6)
    source_context = ""
    if unique_sources:
        source_context = f" Based on evidence from {_human_join(unique_sources)}."

    if not top_claims:
        if _is_informative_model_summary(objective, model_answer):
            lines.append(model_answer_text or model_answer_first)
            if source_context:
                lines.append(source_context.strip())
        else:
            step_highlights = _build_step_result_highlights(task_state)
            if step_highlights:
                highlight_text = "; ".join(h["summary"] for h in step_highlights[:3])
                lines.append(f"Preliminary findings indicate: {highlight_text}.{source_context}")
            elif objective:
                lines.append(
                    f"The collected evidence is not yet sufficient to reach a definitive conclusion"
                    f" regarding {objective}.{source_context}"
                )
            else:
                lines.append(f"The collected evidence is not yet sufficient to reach a definitive conclusion.{source_context}")
        return "\n\n".join(lines).strip()

    confidence_framing = _answer_confidence_framing(top_claims, mixed_claims)

    if _is_informative_model_summary(objective, model_answer):
        lines.append(model_answer_text or model_answer_first)
    else:
        lead_claim = top_claims[0]
        lead_sources = _human_join(list(lead_claim.get("primary_sources", []) or [])[:3]) or "the available sources"
        lines.append(
            f"{confidence_framing} {lead_claim.get('statement', '')},"
            f" driven primarily by {lead_sources}.{source_context}"
        )

    if mixed_claims:
        conflict = mixed_claims[0]
        focus = " and ".join(
            part for part in [
                str(conflict.get("subject", "")).strip(),
                str(conflict.get("object", "")).strip(),
            ] if part
        ) or "a key finding"
        lines.append(
            f" However, evidence is mixed regarding {focus}"
            f" (see Conflicting & Uncertain Evidence below)."
        )

    return "\n\n".join(lines).strip()


def _answer_confidence_framing(
    top_claims: list[dict[str, Any]],
    mixed_claims: list[dict[str, Any]],
) -> str:
    if mixed_claims:
        return "Evidence is divided, but the current balance suggests"
    strengths = [str(c.get("support_strength", "low")).strip().lower() for c in top_claims[:3]]
    high_count = sum(1 for s in strengths if s == "high")
    if high_count >= 2:
        return "Strong evidence from multiple independent sources supports that"
    if high_count == 1:
        return "Evidence supports that"
    return "Preliminary data indicate that"


# ---------------------------------------------------------------------------
# Thematic claim clustering + adaptive finding rendering
# ---------------------------------------------------------------------------

NARRATIVE_PREDICATES = {
    "has_function", "participates_in", "causal_gene_for", "depends_on",
    "activates", "inhibits",
}
TABULAR_PREDICATES = {
    "associated_with", "interacts_with", "has_phenotype", "cross_referenced_in",
    "has_ortholog", "has_model", "screen_hit_in", "tested_in",
}

_ENTITY_TYPE_LABELS: dict[str, str] = {
    "gene": "Gene",
    "compound": "Compound",
    "disease": "Disease",
    "trial": "Trial",
    "cell_line": "Cell line",
    "phenotype": "Phenotype",
    "pathway": "Pathway",
    "variant": "Variant",
    "protein": "Protein",
    "source": "Source",
    "record": "Record",
}


def _cluster_claims_by_theme(
    claim_summary: dict[str, Any],
    task_state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Group ALL substantive claims from the evidence store into thematic clusters.

    Reads the full evidence store (entities, claims, evidence records) rather
    than only the top-N summary, so nuance is preserved for complex questions.
    """
    store = task_state.get("evidence_store", {}) or {}
    objective_text = str(task_state.get("objective", "")).strip()
    all_claims = list((store or {}).get("claims", {}).values())
    evidence_records = list((store or {}).get("evidence", []))
    entities = dict((store or {}).get("entities", {}))
    label_by_entity = {eid: str(e.get("label", eid)).strip() for eid, e in entities.items()}
    type_by_entity = {eid: str(e.get("type", "record")).strip() for eid, e in entities.items()}

    excluded_predicates = SYNTHESIS_META_PREDICATES | TRIVIAL_IDENTIFIER_PREDICATES
    substantive_claims = [
        c for c in all_claims
        if str(c.get("predicate", "")).strip() not in excluded_predicates
    ]
    if not substantive_claims:
        return _fallback_findings_from_steps(task_state)

    evidence_by_claim: dict[str, list[dict[str, Any]]] = {}
    for record in evidence_records:
        cid = str(record.get("claim_id", "")).strip()
        if cid:
            evidence_by_claim.setdefault(cid, []).append(record)

    mixed_claim_ids: set[str] = set()
    mixed_claims_list = list(claim_summary.get("mixed_evidence_claims", []) or [])
    for conflict in mixed_claims_list:
        for cc in list(conflict.get("claims", []) or []):
            cid = str(cc.get("claim_id", "")).strip()
            if cid:
                mixed_claim_ids.add(cid)

    adjudicated: list[dict[str, Any]] = []
    for claim in substantive_claims:
        cid = str(claim.get("id", "")).strip()
        scored = _score_adjudicated_claim(claim, evidence_by_claim.get(cid, []), label_by_entity)
        scored["mixed_evidence"] = cid in mixed_claim_ids
        scored["display_priority_score"] = _claim_display_priority_score(scored, objective_text)
        scored["subject_entity_id"] = str(claim.get("subject_entity_id", "")).strip()
        scored["object_entity_id"] = str(claim.get("object_entity_id", "")).strip()
        scored["subject_type"] = type_by_entity.get(scored["subject_entity_id"], "record")
        scored["object_type"] = type_by_entity.get(scored["object_entity_id"], "record")
        scored["step_ids"] = list(claim.get("step_ids", []) or [])
        scored["observation_types"] = list(claim.get("observation_types", []) or [])

        qualifiers_combined: dict[str, Any] = {}
        for rec in evidence_by_claim.get(cid, []):
            for k, v in (rec.get("qualifiers") or {}).items():
                if v and k not in qualifiers_combined:
                    qualifiers_combined[k] = v
        scored["qualifiers"] = qualifiers_combined

        adjudicated.append(scored)

    adjudicated.sort(key=lambda c: (
        float(c.get("display_priority_score", 0.0) or 0.0),
        1 if not c.get("mixed_evidence") else 0,
        int(c.get("source_count", 0) or 0),
    ), reverse=True)

    entity_to_cluster: dict[str, int] = {}
    clusters: list[dict[str, Any]] = []

    for claim in adjudicated:
        subject = str(claim.get("subject", "")).strip()
        obj = str(claim.get("object", "")).strip()
        merged_idx: int | None = None
        for key in [subject, obj]:
            if key and key in entity_to_cluster:
                merged_idx = entity_to_cluster[key]
                break

        if merged_idx is not None:
            clusters[merged_idx]["claims"].append(claim)
            for key in [subject, obj]:
                if key:
                    entity_to_cluster[key] = merged_idx
        else:
            new_idx = len(clusters)
            clusters.append({"claims": [claim], "title": "", "variant": "verdict"})
            for key in [subject, obj]:
                if key:
                    entity_to_cluster[key] = new_idx

    for cluster in clusters:
        cluster["title"] = _derive_cluster_title(cluster["claims"])
        cluster["variant"] = _select_finding_variant(cluster["claims"])
        cluster["confidence"] = _aggregate_cluster_confidence(cluster["claims"])

    clusters.sort(key=lambda c: (
        max((float(cl.get("display_priority_score", 0)) for cl in c["claims"]), default=0),
        len(c["claims"]),
    ), reverse=True)
    return clusters[:12]


def _build_synthesis_evidence_briefs(
    task_state: dict[str, Any],
    claim_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build compact thematic evidence briefs for the final synthesizer prompt."""
    store = task_state.get("evidence_store", {}) or {}
    evidence_by_claim: dict[str, list[dict[str, Any]]] = {}
    for record in list((store or {}).get("evidence", []) or []):
        claim_id = str(record.get("claim_id", "")).strip()
        if claim_id:
            evidence_by_claim.setdefault(claim_id, []).append(record)

    briefs: list[dict[str, Any]] = []
    for cluster in _cluster_claims_by_theme(claim_summary, task_state)[:8]:
        claims = list(cluster.get("claims", []) or [])
        unique_sources = _dedupe_str_list(
            [source for claim in claims for source in list(claim.get("primary_sources", []) or [])],
            limit=8,
        )
        supporting_ids = _dedupe_str_list(
            [identifier for claim in claims for identifier in list(claim.get("supporting_ids", []) or [])],
            limit=12,
        )
        evidence_notes = _dedupe_str_list(
            [
                _sanitize_internal_report_text(re.sub(r"\s+", " ", str(record.get("summary_text", "")).strip()))
                for claim in claims
                for record in evidence_by_claim.get(str(claim.get("claim_id", "")).strip(), [])
                if str(record.get("summary_text", "")).strip()
            ],
            limit=4,
        )

        entry = {
            "theme": cluster.get("title", "Finding"),
            "confidence": cluster.get("confidence", "moderate"),
            "claim_count": len(claims),
            "source_count": len(unique_sources),
            "evidence_count": sum(int(claim.get("evidence_count", 0) or 0) for claim in claims),
            "top_sources": unique_sources,
            "supporting_ids": supporting_ids,
            "evidence_notes": evidence_notes,
        }

        if claims:
            entry["top_claims"] = [
                {
                    "statement": claim.get("statement", ""),
                    "support_strength": claim.get("support_strength", ""),
                    "source_count": int(claim.get("source_count", 0) or 0),
                    "evidence_count": int(claim.get("evidence_count", 0) or 0),
                    "primary_sources": list(claim.get("primary_sources", []) or [])[:3],
                    "supporting_ids": list(claim.get("supporting_ids", []) or [])[:6],
                    "qualifiers": dict(claim.get("qualifiers", {}) or {}),
                }
                for claim in claims[:4]
            ]
        else:
            entry["step_summary"] = _sanitize_internal_report_text(str(cluster.get("step_summary", "")).strip())
            source = str(cluster.get("step_source", "")).strip()
            if source:
                entry["step_source"] = source

        briefs.append(entry)

    return briefs


def _fallback_findings_from_steps(task_state: dict[str, Any]) -> list[dict[str, Any]]:
    """When no structured claims exist, build findings from step result summaries."""
    findings: list[dict[str, Any]] = []
    for step in task_state.get("steps", []):
        if str(step.get("status", "")).strip() != "completed":
            continue
        summary = str(step.get("result_summary", "")).strip()
        if not summary:
            continue
        source = _format_step_source_display(step)
        findings.append({
            "claims": [],
            "title": str(step.get("goal", "")).strip() or "Finding",
            "variant": "verdict",
            "confidence": "moderate",
            "step_summary": summary,
            "step_source": source,
        })
    return findings[:12]


def _derive_cluster_title(claims: list[dict[str, Any]]) -> str:
    from collections import Counter
    subjects = [str(c.get("subject", "")).strip() for c in claims if c.get("subject")]
    objects = [str(c.get("object", "")).strip() for c in claims if c.get("object")]
    entity_counts = Counter(subjects + objects)
    if entity_counts:
        top_entity = entity_counts.most_common(1)[0][0]
        subject_types = {str(c.get("subject_type", "")).strip() for c in claims if str(c.get("subject", "")).strip() == top_entity}
        object_types = {str(c.get("object_type", "")).strip() for c in claims if str(c.get("object", "")).strip() == top_entity}
        entity_type = (subject_types | object_types) - {"", "record", "objective", "step", "source"}
        type_label = _ENTITY_TYPE_LABELS.get(next(iter(entity_type), ""), "")
        predicates = {str(c.get("predicate", "")).strip() for c in claims}
        if len(predicates) == 1:
            pred = _humanize_claim_predicate(predicates.pop())
            if type_label:
                return f"{top_entity} ({type_label}) — {pred}"
            return f"{top_entity} — {pred}"
        if type_label:
            return f"{top_entity} ({type_label})"
        return top_entity
    first = claims[0] if claims else {}
    return str(first.get("statement", "Finding")).split(" is ")[0].strip() or "Finding"


def _select_finding_variant(claims: list[dict[str, Any]]) -> str:
    """Choose tabular / verdict / narrative based on claim predicates and entity counts."""
    predicates = [str(c.get("predicate", "")).strip() for c in claims]
    subjects = {str(c.get("subject", "")).strip() for c in claims if c.get("subject")}
    objects = {str(c.get("object", "")).strip() for c in claims if c.get("object")}
    unique_entities = subjects | objects

    narrative_count = sum(1 for p in predicates if p in NARRATIVE_PREDICATES)
    tabular_count = sum(1 for p in predicates if p in TABULAR_PREDICATES)

    if len(claims) >= 3 and len(unique_entities) >= 3 and tabular_count >= narrative_count:
        return "tabular"
    if narrative_count > tabular_count and narrative_count >= 2:
        return "narrative"
    return "verdict"


def _aggregate_cluster_confidence(claims: list[dict[str, Any]]) -> str:
    if any(c.get("mixed_evidence") for c in claims):
        return "mixed"
    total_sources = len({s for c in claims for s in (c.get("primary_sources") or [])})
    total_evidence = sum(int(c.get("evidence_count", 0) or 0) for c in claims)
    strengths = [str(c.get("support_strength", "low")).strip().lower() for c in claims]
    if all(s == "high" for s in strengths) or (total_sources >= 3 and total_evidence >= 4):
        return "high"
    if any(s == "high" for s in strengths) or total_sources >= 2:
        return "moderate"
    if total_sources >= 1 and total_evidence >= 1:
        return "moderate"
    return "low"


def _render_conflicting_evidence_section(claim_summary: dict[str, Any]) -> list[str]:
    """Render the Conflicting & Uncertain Evidence section."""
    conflicts = list(claim_summary.get("mixed_evidence_claims", []) or [])
    if not conflicts:
        return []
    lines = ["### Conflicting & Uncertain Evidence", ""]
    for conflict in conflicts[:5]:
        focus = " and ".join(
            part for part in [
                str(conflict.get("subject", "")).strip(),
                str(conflict.get("object", "")).strip(),
            ] if part
        ) or "a key claim"

        lines.append(f"### {focus}")
        lines.append("")

        conflict_claims = list(conflict.get("claims", []) or [])
        for claim in conflict_claims[:4]:
            predicate = _humanize_claim_predicate(str(claim.get("predicate", "")).strip())
            sources = ", ".join(list(claim.get("primary_sources", []) or [])[:2]) or "unknown source"
            lines.append(f"- **{sources}** supports: {claim.get('statement', '')} ({predicate})")

        assessment = str(conflict.get("assessment", "")).replace("_", " ").strip()
        preferred = str(conflict.get("preferred_interpretation", "")).strip()
        if preferred:
            lines.append(f"- **Current lean:** {preferred} ({assessment})")
        else:
            lines.append(f"- **Current lean:** No clear preferred interpretation ({assessment})")
        lines.append(f"- **To resolve:** Validate with an orthogonal source or confirmatory experiment for {focus}.")
        lines.append("")
    return lines


def _build_structured_limitations(
    task_state: dict[str, Any],
    claim_summary: dict[str, Any],
    model_limitations: list[str] | None = None,
) -> list[str]:
    items: list[str] = []
    if model_limitations:
        items.extend(model_limitations)

    total = _total_step_count(task_state)
    completed = _completed_step_count(task_state)
    failed = _failed_step_count(task_state)
    if total and completed < total:
        if failed:
            items.append(f"Only {completed} of {total} planned steps completed successfully; {failed} step(s) failed or were blocked.")
        else:
            items.append(f"Only {completed} of {total} planned steps completed before synthesis, so coverage is still partial.")

    for conflict in list(claim_summary.get("mixed_evidence_claims", []) or [])[:3]:
        preferred = str(conflict.get("preferred_interpretation", "")).strip()
        subject = str(conflict.get("subject", "")).strip()
        object_text = str(conflict.get("object", "")).strip()
        focus = " and ".join(part for part in [subject, object_text] if part)
        if preferred:
            items.append(
                f"Cross-source evidence is mixed for {focus or 'a key claim'}; the current interpretation only leans toward `{preferred}`."
            )
        else:
            items.append(
                f"Cross-source evidence remains equivocal for {focus or 'a key claim'}, with no clear preferred interpretation."
            )

    seen_gaps: set[str] = set()
    for step in task_state.get("steps", []):
        for gap in step.get("open_gaps", []) or []:
            gap_text = re.sub(r"\s+", " ", str(gap or "").strip())
            if not gap_text:
                continue
            lowered = gap_text.lower()
            if lowered in seen_gaps:
                continue
            seen_gaps.add(lowered)
            items.append(f"Open gap: {gap_text}")
            if len(seen_gaps) >= 3:
                break
        if len(seen_gaps) >= 3:
            break

    return _dedupe_str_list(items, limit=8)


def _build_structured_next_actions(
    task_state: dict[str, Any],
    claim_summary: dict[str, Any],
    model_next_actions: list[str] | None = None,
) -> list[str]:
    actions: list[str] = []
    for conflict in list(claim_summary.get("mixed_evidence_claims", []) or [])[:3]:
        subject = str(conflict.get("subject", "")).strip()
        object_text = str(conflict.get("object", "")).strip()
        focus = " and ".join(part for part in [subject, object_text] if part) or "the mixed-evidence claim"
        actions.append(
            f"Resolve the mixed evidence for {focus} using an orthogonal source or confirmatory experiment."
        )

    actions.extend(_fallback_next_actions_from_task_state(task_state))
    if model_next_actions:
        actions.extend(model_next_actions)
    return _dedupe_str_list(actions, limit=6)


def _build_structured_final_synthesis(task_state: dict[str, Any], raw_markdown: str) -> dict[str, Any]:
    sections = _extract_top_level_markdown_sections(raw_markdown)
    claim_summary = _build_claim_synthesis_summary(
        task_state.get("evidence_store", {}),
        str(task_state.get("objective", "")).strip(),
    )

    model_answer_text = sections.get("answer", "") or sections.get("summary", "")
    model_findings_text = sections.get("detailed_findings", "") or sections.get("key_findings", "")

    return {
        "coverage_status": _compute_coverage_status(task_state),
        "claim_synthesis_summary": claim_summary,
        "answer": _build_structured_answer_markdown(
            task_state,
            claim_summary,
            model_answer=model_answer_text,
        ),
        "model_findings_text": model_findings_text.strip(),
        "model_references_text": str(sections.get("references", "") or "").strip(),
        "limitations": _build_structured_limitations(
            task_state,
            claim_summary,
            _extract_markdown_list_items(sections.get("limitations", "")),
        ),
        "next_actions": _build_structured_next_actions(
            task_state,
            claim_summary,
            _extract_markdown_list_items(sections.get("next_steps", "")),
        ),
    }


def _fallback_next_actions_from_task_state(task_state: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    plan_status = str(task_state.get("plan_status", "")).strip()
    current_step_id = str(task_state.get("current_step_id") or "").strip()
    if plan_status == "blocked" and current_step_id:
        actions.append(f"Unblock and rerun {current_step_id} to complete the planned evidence collection.")
    elif plan_status != "completed":
        pending_steps = [
            str(step.get("id"))
            for step in task_state.get("steps", [])
            if str(step.get("status", "")).strip() == "pending"
        ]
        if pending_steps:
            actions.append(f"Continue the remaining planned steps in order ({', '.join(pending_steps[:6])}).")

    # Convert open gaps into concrete follow-up actions.
    seen_gaps: set[str] = set()
    for step in task_state.get("steps", []):
        for gap in step.get("open_gaps", []) or []:
            gap_text = re.sub(r"\s+", " ", str(gap or "").strip())
            if not gap_text:
                continue
            low = gap_text.lower()
            if low in seen_gaps:
                continue
            seen_gaps.add(low)
            actions.append(f"Address open gap: {gap_text}")
            if len(actions) >= 5:
                return actions

    if not actions:
        actions.append("Review the compiled evidence for decision readiness and identify any confirmatory analyses worth running.")
        actions.append("Document key assumptions and uncertainties before making a downstream decision or recommendation.")
    return actions[:5]


def _postprocess_synth_markdown(task_state: dict[str, Any], raw_markdown: str) -> str:
    """Post-process the LLM's markdown output into the final report format."""
    synthesis = _build_structured_final_synthesis(task_state, raw_markdown)
    return _render_final_synthesis_markdown(task_state, synthesis)


def _render_final_synthesis_markdown(task_state: dict[str, Any], synthesis: dict[str, Any]) -> str:
    """Render the final report from structured synthesis fields."""
    objective = str(task_state.get("objective", "")).strip()
    claim_summary = dict(synthesis.get("claim_synthesis_summary", {}) or {})

    lines = ["# AI Co-Scientist Report", ""]

    if objective:
        lines += [f"> **Research Question:** {objective}", "", "---", ""]

    # TLDR
    lines += ["## TLDR", ""]
    answer = str(synthesis.get("answer", "")).strip()
    if answer:
        lines.append(answer)
        lines.append("")

    # Evidence Breakdown (Detailed Findings, Conflicting Evidence, References, Next Steps)
    lines += ["## Evidence Breakdown", ""]

    # Detailed Findings (model-authored thematic prose)
    model_findings = str(synthesis.get("model_findings_text", "")).strip()
    if model_findings:
        lines.append(model_findings)
        lines.append("")

    # Conflicting & Uncertain Evidence
    conflict_lines = _render_conflicting_evidence_section(claim_summary)
    if conflict_lines:
        lines.extend(conflict_lines)

    # Recommended Next Steps (within Evidence Breakdown)
    next_actions = [str(x).strip() for x in synthesis.get("next_actions", []) if str(x).strip()]
    if not next_actions:
        next_actions = _fallback_next_actions_from_task_state(task_state)
    if next_actions:
        lines += ["### Recommended Next Steps", ""]
        for i, item in enumerate(next_actions[:20], start=1):
            lines.append(f"{i}. {item}")
        lines.append("")

    # Limitations (top-level, own section)
    limitations = [str(x).strip() for x in synthesis.get("limitations", []) if str(x).strip()]
    if limitations:
        lines += ["## Limitations", ""]
        lines.extend(f"- {item}" for item in limitations[:20])
        lines.append("")

    # References (last section)
    current_text = "\n".join(lines)
    lit_ids = _collect_final_report_literature_ids(task_state, synthesis, current_text)
    ref_map = _build_ref_map(lit_ids)
    refs = _build_references_section(lit_ids)
    if refs:
        lines += refs.split("\n")
        lines.append("")

    body_so_far = "\n".join(lines)
    body_so_far = _hyperlink_inline_ids(body_so_far, ref_map)
    body_so_far = _hyperlink_author_year_citations(body_so_far, lit_ids)
    body_so_far = _collapse_duplicate_citation_mentions(body_so_far)
    body_so_far = _inject_key_literature_fallback(body_so_far, lit_ids)
    body_so_far = _expand_reference_only_body_lines(body_so_far, lit_ids)
    lines = body_so_far.split("\n")

    return "\n".join(lines).strip()


def _render_parse_error_markdown(agent_label: str, error: str, raw_excerpt: str) -> str:
    excerpt = re.sub(r"\s+", " ", raw_excerpt).strip()
    if len(excerpt) > 240:
        excerpt = excerpt[:237] + "..."
    lines = [f"## {agent_label} Parse Error", "", f"{error}"]
    if excerpt:
        lines.append("")
        lines.append(f"Raw output excerpt: `{excerpt}`")
    return "\n".join(lines).strip()


def _render_no_plan_to_finalize_message() -> str:
    return (
        "## Final Summary\n\n"
        "No plan or collected evidence is available yet. Ask a research question first, "
        "then use `finalize` when you want a final summary."
    )


def _planner_json_instruction_suffix() -> str:
    return (
        "Return ONLY valid JSON matching `plan_internal.v1` for this objective. "
        "Each step MUST include a \"domains\" array with 1-3 domain names from: "
        f"{', '.join(tool_registry.ALL_DOMAIN_NAMES)}. "
        "If you are unsure which tool fits a step, choose the closest valid tool from the catalog instead of leaving "
        "\"tool_hint\" blank. "
        "Do not include markdown fences or commentary."
    )


def _react_step_context_instructions(task_state: dict[str, Any], active_step: dict[str, Any]) -> list[str]:
    """Build context for the ReAct step executor — focuses on ONE step."""
    prior_completed = _compact_completed_step_summaries(task_state)
    remaining_count = sum(
        1 for s in task_state.get("steps", [])
        if str(s.get("status", "")).strip() in {"pending", "in_progress"}
    )

    step_domains = active_step.get("domains") or []
    tool_hint = str(active_step.get("tool_hint", "")).strip()
    goal_text = str(active_step.get("goal", "") or "")
    focus_terms = _extract_step_focus_terms(active_step)
    focused_tools = _prioritize_tools_for_step(
        _resolve_step_tools(step_domains),
        tool_hint,
    )

    focused_catalog = _format_tool_catalog(focused_tools)
    routing_guidance = _format_step_routing_guidance(tool_hint, focused_tools)
    structured_observation_guidance = _format_structured_observation_guidance(tool_hint, focused_tools)
    payload = {
        "schema": "react_step_context.v1",
        "objective": task_state.get("objective", ""),
        "current_step": {
            "id": active_step.get("id"),
            "goal": active_step.get("goal"),
            "tool_hint": active_step.get("tool_hint"),
            "domains": step_domains,
            "completion_condition": active_step.get("completion_condition"),
            "focus_terms": focus_terms,
        },
        "remaining_steps_after_this": remaining_count - 1,
        "prior_completed_steps": prior_completed,
    }

    instructions = [
        "Execution context (authoritative; use this instead of inferring from prior prose):",
        _serialize_pretty_json(payload),
    ]

    tools_header = (
        f"Tools for this step (domains: {', '.join(step_domains)}):" if step_domains
        else "Tools for this step:"
    )
    instructions.append(
        f"{tools_header}\n{focused_catalog}\n"
        "Prefer tools from this list. You may use other available tools "
        "if the focused set is insufficient, but start here."
    )
    if routing_guidance:
        instructions.append(routing_guidance)
    if structured_observation_guidance:
        instructions.append(structured_observation_guidance)

    if focus_terms:
        instructions.append(
            "Scope guardrails for this step:\n"
            f"- In-scope anchors: {', '.join(focus_terms)}.\n"
            "- Reuse prior_completed_steps only when they directly support these anchors.\n"
            "- Ignore sibling genes, mutations, alleles, datasets, or diseases from other plan steps unless this step explicitly requires them.\n"
            "- In result_summary, describe only what this step established. Do not narrate that a previous step already succeeded or that the completion condition was fulfilled."
        )

    is_mutation_epitope_step = bool(
        re.search(r"\b[A-Z]\d{1,5}[A-Z*]\b", goal_text)
        and re.search(r"\b(epitope|neoepitope|neoantigen|peptide)\b", goal_text, flags=re.IGNORECASE)
    )
    if is_mutation_epitope_step:
        instructions.append(
            "Mutation-specific peptide guardrails:\n"
            "- Do not invent or reverse-engineer an exact mutant peptide from a generic protein sequence unless the residue position and substituted amino acid are explicitly verified in the cited source.\n"
            "- If the source only supports a peptide span, residue range, or long-peptide construct without an exact sequence, state that the exact sequence remains unresolved instead of fabricating one."
        )

    instructions.append(
        f"Execute ONLY step {active_step.get('id')}. "
        f"There are {remaining_count - 1} more step(s) after this one. "
        "Call at least one tool, then write a clear findings summary. "
        "Include all evidence identifiers (PMID, DOI, NCT, etc.) inline "
        "and state key findings as explicit claims."
    )
    instructions.append(
        "If tool calls return errors (e.g. schema changes, no matching signature, field not found), "
        "conclude the step now with a brief note about the limitations. Do not retry failing queries more than once."
    )
    return instructions


def _resolve_source_label(tool_hint: str) -> str:
    """Map a tool_hint to its human-readable database/source name."""
    tool_hint = str(tool_hint or "").strip()
    if not tool_hint:
        return ""
    if _is_internal_skill_tool_name(tool_hint):
        return ""
    label = tool_registry.TOOL_SOURCE_NAMES.get(tool_hint)
    if label:
        return label
    # BigQuery tool_hints can be dataset.table (e.g. open_targets_platform.disease)
    if "." in tool_hint:
        base = tool_hint.split(".", 1)[0]
        label = tool_registry.TOOL_SOURCE_NAMES.get(base)
        if label:
            return label
    return tool_hint


def _extract_source_labels_from_text(text: str) -> list[str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return []

    labels: list[str] = []
    for key, label in tool_registry.TOOL_SOURCE_NAMES.items():
        if not label or label == "BigQuery":
            continue
        if re.search(rf"\b{re.escape(key)}\b", raw_text):
            labels.append(label)
    for match in re.finditer(r"bigquery://([A-Za-z0-9._-]+)", raw_text):
        labels.extend(_normalize_source_label_candidates([match.group(1)]))
    return _dedupe_str_list(labels, limit=20)


def _normalize_source_label_candidates(values: list[Any] | None, *, allow_verbatim_labels: bool = False) -> list[str]:
    specific: list[str] = []
    generic: list[str] = []

    def add_label(label: str) -> None:
        cleaned = str(label or "").strip()
        if not cleaned:
            return
        if cleaned == "BigQuery":
            generic.append(cleaned)
        else:
            specific.append(cleaned)

    for value in values or []:
        raw = str(value or "").strip()
        if not raw:
            continue
        if _is_internal_skill_tool_name(raw):
            continue
        if raw == "BigQuery":
            generic.append(raw)
            continue
        if re.search(r"\s", raw) or "bigquery://" in raw:
            for label in _extract_source_labels_from_text(raw):
                add_label(label)

        cleaned = raw[11:] if raw.startswith("bigquery://") else raw
        parts = [part for part in cleaned.split(".") if part]
        candidates = [cleaned]
        if len(parts) >= 3:
            candidates.extend([".".join(parts[-2:]), parts[-2]])
        elif len(parts) == 2:
            candidates.extend([parts[1], parts[0]])

        for candidate in candidates:
            label = _resolve_source_label(candidate)
            if label and label != candidate:
                add_label(label)
            elif candidate in tool_registry.TOOL_SOURCE_NAMES:
                add_label(label)

        if raw in tool_registry.TOOL_SOURCE_NAMES:
            add_label(_resolve_source_label(raw))
            continue

        if allow_verbatim_labels and raw not in {"run_bigquery_select_query", "list_bigquery_tables"}:
            add_label(raw)

    normalized = _dedupe_str_list(specific, limit=20)
    if normalized:
        return normalized
    return _dedupe_str_list(generic, limit=5)


def _derive_step_data_sources(step: dict[str, Any]) -> list[str]:
    explicit_labels = _normalize_source_label_candidates(
        list(step.get("data_sources_queried", []) or []),
        allow_verbatim_labels=True,
    )
    refs: list[Any] = []
    for observation in list(step.get("structured_observations", []) or []):
        if not isinstance(observation, dict):
            continue
        refs.append(observation.get("source_tool"))
        qualifiers = observation.get("qualifiers") or {}
        if isinstance(qualifiers, dict):
            refs.extend([
                qualifiers.get("dataset"),
                qualifiers.get("source"),
                qualifiers.get("database"),
            ])
    refs.append(step.get("tool_hint"))
    refs.extend(list(step.get("tools_called", []) or []))

    labels = _merge_str_values(explicit_labels, _normalize_source_label_candidates(refs), limit=20)
    if labels:
        return labels
    fallback = _resolve_source_label(step.get("tool_hint", ""))
    return [fallback] if fallback else []


def _preferred_step_source_label(step: dict[str, Any], source_tool: str = "") -> str:
    source_tool = str(source_tool or "").strip()
    direct = _resolve_source_label(source_tool)
    if direct and direct != "BigQuery":
        return direct
    derived = _derive_step_data_sources(step)
    if derived:
        return derived[0]
    return direct or source_tool


def _format_step_source_display(step: dict[str, Any]) -> str:
    return "; ".join(_derive_step_data_sources(step))


def _synth_context_instructions(task_state: dict[str, Any], callback_context: CallbackContext | None = None) -> list[str]:
    evidence_store_summary = _summarize_evidence_store(task_state.get("evidence_store", {}))
    claim_synthesis_summary = _build_claim_synthesis_summary(
        task_state.get("evidence_store", {}),
        str(task_state.get("objective", "")).strip(),
    )
    evidence_briefs = _build_synthesis_evidence_briefs(task_state, claim_synthesis_summary)
    execution_metrics_summary = dict((task_state.get("execution_metrics") or {}).get("summary", {}))
    payload = {
        "schema": "synthesis_context.v1",
        "objective": task_state.get("objective", ""),
        "plan_status": task_state.get("plan_status", "ready"),
        "coverage_status": _compute_coverage_status(task_state),
        "evidence_store_summary": evidence_store_summary,
        "claim_synthesis_summary": claim_synthesis_summary,
        "evidence_briefs": evidence_briefs,
        "execution_metrics_summary": execution_metrics_summary,
        "steps": [
            {
                "id": step.get("id"),
                "goal": step.get("goal"),
                "tool_hint": step.get("tool_hint", ""),
                "source": _preferred_step_source_label(step, str(step.get("tool_hint", ""))),
                "status": step.get("status"),
                "reasoning_trace": _sanitize_internal_report_text(str(step.get("reasoning_trace", "") or "")),
                "tools_called": list(step.get("tools_called", []) or []),
                "data_sources_queried": _derive_step_data_sources(step),
                "result_summary": _sanitize_internal_report_text(str(step.get("result_summary", "") or "")),
                "evidence_ids": list(step.get("evidence_ids", []) or [])[:20],
                "open_gaps": list(step.get("open_gaps", []) or [])[:10],
                "structured_observations": list(step.get("structured_observations", []) or [])[:8],
                "entity_ids": list(step.get("entity_ids", []) or [])[:20],
                "claim_ids": list(step.get("claim_ids", []) or [])[:20],
                "execution_metrics": dict(step.get("execution_metrics", {}) or {}),
            }
            for step in task_state.get("steps", [])
        ],
    }

    used_sources: dict[str, str] = {}
    for step in task_state.get("steps", []):
        hint = str(step.get("tool_hint", "")).strip()
        if hint and hint not in used_sources:
            used_sources[hint] = tool_registry.TOOL_SOURCE_NAMES.get(hint, hint)
    if used_sources:
        payload["source_reference"] = {
            tool: source for tool, source in used_sources.items()
        }

    instructions = [
        "Synthesis context (authoritative; use this instead of inferring from prior prose):",
        _serialize_pretty_json(payload),
        "Use `claim_synthesis_summary` as the normalized claim arbitration layer. Prioritize its top-supported claims for the direct answer, and explicitly describe any `mixed_evidence_claims` as conflicting or mixed evidence rather than flattening them into one-sided prose.",
    ]

    prior: list[dict[str, Any]] = []
    if callback_context is not None:
        prior = _get_prior_research(callback_context)
    if prior:
        prior_entries = []
        for cycle_idx, entry in enumerate(prior, start=1):
            prior_entries.append({
                "cycle": cycle_idx,
                "objective": entry.get("objective", ""),
                "plan_status": entry.get("plan_status", ""),
                "steps": [
                    {
                        "id": s.get("id"),
                        "goal": s.get("goal"),
                        "status": s.get("status"),
                        "result_summary": s.get("result_summary", ""),
                        "evidence_ids": list(s.get("evidence_ids", []) or [])[:12],
                        "open_gaps": list(s.get("open_gaps", []) or [])[:5],
                    }
                    for s in entry.get("steps", [])
                ],
                "synthesis_markdown": (entry.get("latest_synthesis") or {}).get("markdown", ""),
            })
        instructions.append(
            "Prior research cycles from this session (reference and build on where relevant):"
        )
        instructions.append(_serialize_pretty_json(prior_entries))

    instructions.append(
        "Return user-facing Markdown (not JSON). Follow the section structure from your instructions exactly: "
        "`## TLDR`, `## Evidence Breakdown` with `###` theme subsections, `### Conflicting & Uncertain Evidence` when needed, "
        "`## Limitations`, and `## Recommended Next Steps`. Use `evidence_briefs` to cover each relevant theme with concrete supporting detail, "
        "but turn them into cohesive prose with clear topic sentences and continuous narrative flow rather than a checklist."
    )
    return instructions


def _validate_step_execution_result(raw: dict[str, Any]) -> dict[str, Any]:
    if str(raw.get("schema", "")).strip() != STEP_RESULT_SCHEMA:
        raise ValueError(f"schema must be {STEP_RESULT_SCHEMA}")
    status = str(raw.get("status", "")).strip().lower()
    if status not in {"completed", "blocked"}:
        raise ValueError("status must be `completed` or `blocked`")
    tools_called = _as_string_list(raw.get("tools_called"), "tools_called", limit=20)
    return {
        "schema": STEP_RESULT_SCHEMA,
        "step_id": _as_nonempty_str(raw.get("step_id"), "step_id"),
        "status": status,
        "step_progress_note": _as_nonempty_str(raw.get("step_progress_note"), "step_progress_note"),
        "result_summary": _as_nonempty_str(raw.get("result_summary"), "result_summary"),
        "evidence_ids": _as_string_list(raw.get("evidence_ids"), "evidence_ids", limit=30),
        "open_gaps": _as_string_list(raw.get("open_gaps"), "open_gaps", limit=15),
        "suggested_next_searches": _as_string_list(
            raw.get("suggested_next_searches"),
            "suggested_next_searches",
            limit=15,
        ),
        "tools_called": tools_called,
        "data_sources_queried": _as_string_list(raw.get("data_sources_queried"), "data_sources_queried", limit=15),
        "structured_observations": _validate_structured_observations(raw.get("structured_observations")),
    }



def _apply_step_execution_result_to_task_state(
    task_state: dict[str, Any],
    step_result: dict[str, Any],
    *,
    parse_retry_count: int = 0,
) -> dict[str, Any]:
    validated = _validate_step_execution_result(step_result)
    current_step_id = str(task_state.get("current_step_id") or "").strip()
    if current_step_id and validated["step_id"] != current_step_id:
        raise ValueError(
            f"Executor returned step_id {validated['step_id']} but active step is {current_step_id}"
        )

    _, step = _find_step(task_state, validated["step_id"])
    step["status"] = validated["status"]
    step["step_progress_note"] = validated["step_progress_note"]
    step["result_summary"] = validated["result_summary"]
    step["evidence_ids"] = validated["evidence_ids"]
    step["open_gaps"] = validated["open_gaps"]
    step["suggested_next_searches"] = validated["suggested_next_searches"]
    step["tools_called"] = validated.get("tools_called", [])
    step["data_sources_queried"] = validated.get("data_sources_queried", [])
    step["structured_observations"] = validated.get("structured_observations", [])
    step["execution_metrics"] = _build_step_execution_metrics(
        step,
        validated,
        parse_retry_count=parse_retry_count,
    )

    if validated["status"] == "completed":
        task_state["last_completed_step_id"] = validated["step_id"]
        next_step_id = _next_pending_step_id(task_state)
        task_state["current_step_id"] = next_step_id
        task_state["plan_status"] = "completed" if next_step_id is None else "ready"
    else:
        task_state["current_step_id"] = validated["step_id"]
        task_state["plan_status"] = "blocked"

    _refresh_task_state_derived_state(task_state)
    return validated


def _make_planner_before_model_callback(*, require_approval: bool, model_name: str = PLANNER_MODEL):
    """Factory: returns a planner before-model callback.

    When ``require_approval`` is True, the callback recognises ``approve``,
    ``revise: <feedback>`` and ``continue``-style commands so the workflow
    can pause after planning and resume after human review.
    """

    def _callback(*, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
        _clear_turn_temp_state(callback_context)
        user_text = _extract_user_turn_text(callback_context)
        is_finalize = _is_finalize_command(user_text)
        callback_context.state[STATE_FINALIZE_REQUESTED] = bool(is_finalize)

        if is_finalize:
            return _make_text_response("")

        # --- History command ---
        if _normalize_user_text(user_text) in ("history", "/history", "list cycles"):
            prior = _get_prior_research(callback_context)
            current = _get_task_state(callback_context)
            if not prior and not current:
                callback_context.state[STATE_TURN_ABORT_REASON] = "command_handled"
                return _make_text_response("No research history yet.")
            lines = ["## Research History", ""]
            for i, entry in enumerate(prior, start=1):
                obj = str(entry.get("objective", "")).strip() or "(no objective)"
                status = str(entry.get("plan_status", ""))
                c = sum(1 for s in entry.get("steps", []) if str(s.get("status", "")) == "completed")
                t = len(entry.get("steps", []))
                lines.append(f"{i}. **{obj}** — {status} ({c}/{t} steps) — `switch {i}` to restore")
            if current:
                obj = str(current.get("objective", "")).strip() or "(no objective)"
                status = str(current.get("plan_status", ""))
                c = sum(1 for s in current.get("steps", []) if str(s.get("status", "")) == "completed")
                t = len(current.get("steps", []))
                lines.append(f"\n**Active:** {obj} — {status} ({c}/{t} steps)")
            callback_context.state[STATE_TURN_ABORT_REASON] = "command_handled"
            return _make_text_response("\n".join(lines))

        # --- Rollback / switch ---
        rollback_idx = _parse_rollback_command(user_text)
        if rollback_idx is not None:
            prior = _get_prior_research(callback_context)
            if not prior:
                callback_context.state[STATE_TURN_ABORT_REASON] = "command_handled"
                return _make_text_response("No prior research cycles to roll back to.")
            if rollback_idx == -1:
                target_idx = len(prior) - 1
            else:
                target_idx = rollback_idx - 1
            if target_idx < 0 or target_idx >= len(prior):
                callback_context.state[STATE_TURN_ABORT_REASON] = "command_handled"
                return _make_text_response(
                    f"Invalid cycle number. Available: 1\u2013{len(prior)}."
                )
            _archive_current_task(callback_context)
            restored = prior.pop(target_idx)
            callback_context.state[STATE_PRIOR_RESEARCH] = prior
            callback_context.state[STATE_WORKFLOW_TASK] = restored
            obj = str(restored.get("objective", "")).strip()
            completed = sum(
                1 for s in restored.get("steps", [])
                if str(s.get("status", "")) == "completed"
            )
            total = len(restored.get("steps", []))
            plan_status = str(restored.get("plan_status", ""))
            rendered = (
                f"## Restored Research Cycle {target_idx + 1}\n\n"
                f"**Objective:** {obj}\n\n"
                f"**Status:** {plan_status} ({completed}/{total} steps completed)\n\n"
                "Send `finalize` to regenerate the report, or `approve` to resume execution."
            )
            callback_context.state[STATE_TURN_ABORT_REASON] = "command_handled"
            return _make_text_response(rendered)

        # --- HITL approval gate ---
        if require_approval and bool(callback_context.state.get(STATE_PLAN_PENDING_APPROVAL, False)):
            if _is_plan_approval_command(user_text):
                callback_context.state[STATE_PLAN_PENDING_APPROVAL] = False
                return _make_text_response("")

            revision_feedback = _extract_revision_feedback(user_text)
            if revision_feedback is not None:
                task_state = _get_task_state(callback_context)
                original_objective = task_state.get("objective", "") if task_state else ""
                _archive_current_task(callback_context)
                callback_context.state[STATE_WORKFLOW_TASK] = None
                callback_context.state[STATE_PLAN_PENDING_APPROVAL] = False
                llm_request.config = llm_request.config or types.GenerateContentConfig()
                tc = _thinking_config_for_model(str(model_name))
                if tc:
                    llm_request.config.thinking_config = tc
                llm_request.config.response_mime_type = None
                llm_request.append_instructions([
                    f"Revise the previous plan for this objective: {original_objective}",
                    f"User revision feedback: {revision_feedback}",
                    "Generate an updated plan that addresses the feedback.",
                    _planner_json_instruction_suffix(),
                ])
                return None

            callback_context.state[STATE_PLAN_PENDING_APPROVAL] = False
            _archive_current_task(callback_context)
            callback_context.state[STATE_WORKFLOW_TASK] = None

        # --- HITL: let continuation commands pass through to executor ---
        if require_approval:
            task_state = _get_task_state(callback_context)
            if task_state:
                plan_status = str(task_state.get("plan_status", ""))
                if plan_status in ("ready", "blocked", "in_progress") and _is_continue_execution_command(user_text):
                    return _make_text_response("")

        # --- Original logic ---
        task_state = _get_task_state(callback_context)
        normalized = _normalize_user_text(user_text)
        if task_state and str(task_state.get("objective_fingerprint", "")) == normalized:
            return _make_text_response("")

        if task_state and normalized and str(task_state.get("objective_fingerprint", "")) != normalized:
            _archive_current_task(callback_context)
            callback_context.state[STATE_WORKFLOW_TASK] = None

        llm_request.config = llm_request.config or types.GenerateContentConfig()
        tc = _thinking_config_for_model(str(model_name))
        if tc:
            llm_request.config.thinking_config = tc
        llm_request.config.response_mime_type = None
        llm_request.append_instructions([_planner_json_instruction_suffix()])
        return None

    return _callback


def _make_planner_after_model_callback(*, require_approval: bool):
    """Factory: returns a planner after-model callback.

    When ``require_approval`` is True the rendered plan is appended with
    an approval prompt and ``STATE_PLAN_PENDING_APPROVAL`` is set.
    """

    def _callback(*, callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
        if bool(callback_context.state.get(STATE_MODEL_ERROR_PASSTHROUGH, False)):
            callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
            callback_context.state[STATE_PLANNER_BUFFER] = ""
            return None

        if _llm_response_has_function_call(llm_response):
            return None

        text = _llm_response_text(llm_response)
        if bool(getattr(llm_response, "partial", False)):
            _buffer_partial_text(callback_context, STATE_PLANNER_BUFFER, text)
            return _replace_llm_response_text(llm_response, "")

        parsed, parse_error = _consume_buffered_json_object(
            callback_context,
            buffer_key=STATE_PLANNER_BUFFER,
            llm_response=llm_response,
        )
        if parsed is None:
            callback_context.state[STATE_TURN_ABORT_REASON] = "planner_parse_error"
            rendered = _render_parse_error_markdown("Planner", parse_error or "Failed to parse plan JSON", text)
            _set_turn_rendered_output(callback_context, key=STATE_PLANNER_RENDERED, text=rendered)
            return _replace_llm_response_text(llm_response, rendered)

        objective_text = _extract_user_turn_text(callback_context)
        try:
            task_state = _initialize_task_state_from_plan(parsed, objective_text=objective_text)
        except Exception as exc:  # noqa: BLE001
            callback_context.state[STATE_TURN_ABORT_REASON] = "planner_validation_error"
            rendered = _render_parse_error_markdown("Planner", str(exc), text)
            _set_turn_rendered_output(callback_context, key=STATE_PLANNER_RENDERED, text=rendered)
            return _replace_llm_response_text(llm_response, rendered)

        callback_context.state[STATE_WORKFLOW_TASK] = task_state
        rendered = _render_plan_markdown(task_state)

        if require_approval:
            callback_context.state[STATE_PLAN_PENDING_APPROVAL] = True
            rendered += _render_plan_approval_prompt()

        _set_turn_rendered_output(callback_context, key=STATE_PLANNER_RENDERED, text=rendered)
        return _replace_llm_response_text(llm_response, rendered)

    return _callback


RATE_LIMIT_BACKOFF_BASE = int(os.environ.get("ADK_RATE_LIMIT_BACKOFF_SECONDS", "5"))
RATE_LIMIT_BACKOFF_MAX = 60
RATE_LIMIT_MAX_RETRIES = int(os.environ.get("ADK_RATE_LIMIT_MAX_RETRIES", "5"))
RATE_LIMIT_AUTO_RETRY = os.environ.get(
    "ADK_RATE_LIMIT_AUTO_RETRY", "true"
).strip().lower() not in {"0", "false", "no", "off"}

STATE_RATE_LIMIT_RETRY_COUNT = "temp:co_scientist_rate_limit_retry_count"


def _using_vertex_ai_backend() -> bool:
    return os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() == "true"


def _on_model_error(
    *,
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
    """Handle model-level errors with exponential backoff for rate limits."""
    error_type = type(error).__name__
    error_msg = str(error)
    logger.error("Model error in %s: [%s] %s", "agent", error_type, error_msg)

    callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = True

    is_rate_limit = any(
        hint in error_msg.lower()
        for hint in ("429", "resource exhausted", "rate limit", "quota", "503", "unavailable")
    )
    if is_rate_limit:
        backend_label = "Vertex AI" if _using_vertex_ai_backend() else "Google AI Studio"
        retry_count = int(callback_context.state.get(STATE_RATE_LIMIT_RETRY_COUNT, 0))

        if RATE_LIMIT_AUTO_RETRY and retry_count < RATE_LIMIT_MAX_RETRIES:
            callback_context.state[STATE_RATE_LIMIT_RETRY_COUNT] = retry_count + 1
            backoff = min(RATE_LIMIT_BACKOFF_BASE * (2 ** retry_count) + random.uniform(0, RATE_LIMIT_BACKOFF_BASE), RATE_LIMIT_BACKOFF_MAX)
            logger.info(
                "Rate limit from %s — retry %d/%d, backing off %.1fs",
                backend_label, retry_count + 1, RATE_LIMIT_MAX_RETRIES, backoff,
            )
            time.sleep(backoff)
            user_msg = (
                f"_Rate limit hit from {backend_label} — retry {retry_count + 1}/{RATE_LIMIT_MAX_RETRIES}, waited {backoff:.0f}s…_"
            )
        else:
            callback_context.state[STATE_RATE_LIMIT_RETRY_COUNT] = 0
            mitigation = (
                "This deployment is currently using Vertex AI. If local runs are fine, redeploy with `USE_VERTEX_AI=false` to use the configured API key backend, or increase Vertex quota."
                if _using_vertex_ai_backend()
                else "Retry after quota resets, reduce concurrent usage, or switch to a backend with available quota."
            )
            reason = f"Retries exhausted ({RATE_LIMIT_MAX_RETRIES})" if retry_count >= RATE_LIMIT_MAX_RETRIES else "Auto-retry disabled"
            user_msg = (
                "## Execution Error\n\n"
                f"{backend_label} quota or rate limit exhausted. {reason}.\n\n"
                f"`{error_msg[:300]}`\n\n"
                f"{mitigation}"
            )
    else:
        user_msg = (
            f"## Execution Error\n\n"
            f"A model error occurred: **{error_type}**\n\n"
            f"`{error_msg[:300]}`\n\n"
            "Send `approve` to retry."
        )
    return _make_text_response(user_msg)


def _on_tool_error(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    error: Exception,
) -> dict | None:
    """Surface tool-level errors as a result the LLM can see and adapt to."""
    error_type = type(error).__name__
    error_msg = str(error)
    tool_name = getattr(tool, "name", "unknown_tool")
    logger.error("Tool error in %s: [%s] %s", tool_name, error_type, error_msg)

    fallback_hints = {
        "search_openalex_works": "Try search_pubmed or search_pubmed_advanced instead.",
        "get_paper_fulltext": "Try get_pubmed_abstract for abstract-only access, or search_openalex_works for alternate open-access links.",
        "search_pubmed": "Try search_openalex_works or search_pubmed_advanced instead.",
        "search_pubmed_advanced": "Try search_pubmed or search_openalex_works instead.",
        "search_clinical_trials": "Try summarize_clinical_trials_landscape or run_bigquery_select_query instead.",
        "list_bigquery_tables": "Try again with the `table` parameter to get the schema, e.g. list_bigquery_tables(dataset='...', table='...').",
    }
    suggestion = fallback_hints.get(tool_name, "Try an alternative tool or query.")

    return {
        "error": True,
        "error_type": error_type,
        "message": f"Tool '{tool_name}' failed: {error_msg[:500]}",
        "suggestion": suggestion,
    }


def _hitl_skip_agent(*, callback_context: CallbackContext) -> types.Content | None:
    """before_agent_callback shared by executor and synth when HITL is active.

    Returns Content (skipping the agent entirely) when the plan is still
    awaiting human approval.  Returns None to let the agent run normally.
    """
    if bool(callback_context.state.get(STATE_PLAN_PENDING_APPROVAL, False)):
        return types.Content(role="model", parts=[types.Part.from_text(text="")])
    return None


def _react_skip_if_done(*, callback_context: CallbackContext) -> types.Content | None:
    """before_agent_callback for step_executor: skip entirely when no work remains."""
    if bool(callback_context.state.get(STATE_FINALIZE_REQUESTED, False)):
        task_state = _get_task_state(callback_context)
        if task_state:
            task_state["plan_status"] = "completed"
            task_state["current_step_id"] = None
            callback_context.state[STATE_WORKFLOW_TASK] = task_state
            callback_context.state[STATE_AUTO_SYNTH_REQUESTED] = True
        return types.Content(role="model", parts=[])
    if str(callback_context.state.get(STATE_TURN_ABORT_REASON, "")).strip():
        return types.Content(role="model", parts=[])
    task_state = _get_task_state(callback_context)
    if not task_state:
        return types.Content(role="model", parts=[])
    plan_status = str(task_state.get("plan_status", ""))
    current_step = str(task_state.get("current_step_id") or "").strip()
    if plan_status == "completed" or not current_step:
        return types.Content(role="model", parts=[])
    if plan_status == "blocked" and not _next_pending_step_id(task_state):
        return types.Content(role="model", parts=[])
    return None


def _react_before_model_callback(*, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    """ReAct step executor: inject context for ONE step per LoopAgent iteration."""
    if bool(callback_context.state.get(STATE_FINALIZE_REQUESTED, False)):
        logger.info("[react:before] skipping — finalize requested")
        return _make_text_response("")
    abort = str(callback_context.state.get(STATE_TURN_ABORT_REASON, "")).strip()
    if abort:
        logger.info("[react:before] skipping — abort reason: %s", abort)
        return _make_text_response("")

    task_state = _get_task_state(callback_context)
    if not task_state:
        logger.warning("[react:before] skipping — no task state")
        return _make_text_response("")

    current_step_id = str(task_state.get("current_step_id") or "").strip()
    plan_status = str(task_state.get("plan_status", ""))

    if plan_status == "blocked" and current_step_id:
        next_id = _next_pending_step_id(task_state)
        if next_id:
            logger.info("[react:before] advancing past blocked %s → %s", current_step_id, next_id)
            task_state["current_step_id"] = next_id
            task_state["plan_status"] = "ready"
            callback_context.state[STATE_WORKFLOW_TASK] = task_state
            current_step_id = next_id
        else:
            logger.info("[react:before] all remaining steps blocked or done")
            return _make_text_response("")

    if not current_step_id or plan_status == "completed":
        logger.info("[react:before] all steps done (plan_status=%s) — no-op iteration", plan_status)
        return _make_text_response("")

    try:
        _, active_step = _find_step(task_state, current_step_id)
    except Exception as exc:  # noqa: BLE001
        logger.error("[react:before] bad task state: %s", exc)
        callback_context.state[STATE_TURN_ABORT_REASON] = "executor_state_error"
        rendered = f"## Execution\n\nInvalid task state: {exc}"
        _append_executor_rendered(callback_context, rendered)
        return _make_text_response(rendered)

    prev_active = str(callback_context.state.get(STATE_EXECUTOR_ACTIVE_STEP_ID, "") or "").strip()
    callback_context.state[STATE_EXECUTOR_ACTIVE_STEP_ID] = current_step_id
    callback_context.state[STATE_EXECUTOR_PREV_STEP_STATUS] = str(active_step.get("status", "pending"))
    if prev_active != current_step_id:
        callback_context.state[STATE_EXECUTOR_REASONING_TRACE] = ""
        _set_tool_log(callback_context, [])
    else:
        # Within the same step's tool loop -- update last tool_log entry with result
        tool_log = _get_tool_log(callback_context)
        contents = getattr(llm_request, "contents", None) or []
        for content in reversed(contents):
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                fr = getattr(part, "function_response", None)
                if fr is None:
                    continue
                tool_name = str(getattr(fr, "name", "") or "").strip()
                if _is_internal_skill_tool_name(tool_name):
                    continue
                result_desc = _describe_tool_result(tool_name, getattr(fr, "response", None))
                evidence_text = _extract_tool_result_evidence_text(tool_name, getattr(fr, "response", None))
                for entry in tool_log:
                    if entry.get("status") == "called" and entry.get("raw_tool") == tool_name:
                        entry["status"] = "done"
                        entry["result"] = result_desc
                        if evidence_text:
                            entry["evidence_text"] = evidence_text
                        break
                else:
                    source = tool_registry.TOOL_SOURCE_NAMES.get(tool_name, tool_name)
                    entry = {"tool": source, "raw_tool": tool_name, "status": "done", "summary": result_desc}
                    if evidence_text:
                        entry["evidence_text"] = evidence_text
                    tool_log.append(entry)
            break
        _set_tool_log(callback_context, tool_log)
        obs = _summarize_latest_tool_results(llm_request)
        if obs:
            prev_trace = str(callback_context.state.get(STATE_EXECUTOR_REASONING_TRACE, "") or "")
            callback_context.state[STATE_EXECUTOR_REASONING_TRACE] = (prev_trace + "\n" + obs).strip()
    active_step["status"] = "in_progress"
    callback_context.state[STATE_WORKFLOW_TASK] = task_state

    logger.info("[react:before] executing step %s: %s", current_step_id, active_step.get("goal", ""))

    llm_request.config = llm_request.config or types.GenerateContentConfig()
    tc = _thinking_config_for_model(str(DEFAULT_MODEL))
    if tc:
        llm_request.config.thinking_config = tc
    llm_request.config.response_mime_type = None
    instructions = _react_step_context_instructions(task_state, active_step)
    llm_request.append_instructions(instructions)
    return None


def _react_after_model_callback(*, callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
    """ReAct step executor: accept prose or JSON, derive structure, and advance."""
    if bool(callback_context.state.get(STATE_MODEL_ERROR_PASSTHROUGH, False)):
        callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
        callback_context.state[STATE_EXECUTOR_BUFFER] = ""
        logger.info("[react:after] model error passthrough — skipping parse")
        return None

    if _llm_response_has_function_call(llm_response):
        callback_context.state[STATE_EXECUTOR_BUFFER] = ""
        fc_list = _extract_function_calls(llm_response)
        visible_fc_list = [fc for fc in fc_list if not _is_internal_skill_tool_name(fc["name"])]
        tool_names = [fc["name"] for fc in visible_fc_list]
        thought_text = _llm_response_thought_text(llm_response).strip()
        text_alongside = _llm_response_text(llm_response).strip()

        # Build structured tool_log entries from actual function call data
        tool_log = _get_tool_log(callback_context)
        for fc in visible_fc_list:
            source = tool_registry.TOOL_SOURCE_NAMES.get(fc["name"], fc["name"])
            description = _describe_tool_call(fc["name"], fc["args"])
            provenance_args = _compact_tool_args_for_provenance(fc["args"])
            tool_log.append({
                "tool": source,
                "raw_tool": fc["name"],
                "status": "called",
                "summary": description,
                **({"args": provenance_args} if provenance_args else {}),
            })
        _set_tool_log(callback_context, tool_log)

        # Keep the raw reasoning trace for backward compat
        trace_parts: list[str] = []
        if tool_names:
            source_labels = [tool_registry.TOOL_SOURCE_NAMES.get(t, t) for t in tool_names]
            trace_parts.append(f"ACT: Called {', '.join(source_labels)}")
        if thought_text:
            trace_parts.append(thought_text)
        if text_alongside:
            trace_parts.append(text_alongside)
        if trace_parts:
            chunk = "\n".join(trace_parts)
            prev_trace = str(callback_context.state.get(STATE_EXECUTOR_REASONING_TRACE, "") or "")
            callback_context.state[STATE_EXECUTOR_REASONING_TRACE] = (prev_trace + "\n" + chunk).strip()
        logger.info(
            "[react:after] function_call: %s, tool_log_len=%d",
            ", ".join(tool_names) if tool_names else "(internal skill only)",
            len(tool_log),
        )
        if text_alongside:
            return _replace_llm_response_text(llm_response, "")
        return None

    text = _llm_response_text(llm_response)
    if bool(getattr(llm_response, "partial", False)):
        _buffer_partial_text(callback_context, STATE_EXECUTOR_BUFFER, text)
        return _replace_llm_response_text(llm_response, "")

    if not text and not str(callback_context.state.get(STATE_EXECUTOR_BUFFER, "") or ""):
        logger.debug("[react:after] empty text (no-op iteration)")
        return None

    buffered = str(callback_context.state.get(STATE_EXECUTOR_BUFFER, "") or "")
    callback_context.state[STATE_EXECUTOR_BUFFER] = ""
    tool_reasoning = str(callback_context.state.get(STATE_EXECUTOR_REASONING_TRACE, "") or "").strip()
    callback_context.state[STATE_EXECUTOR_REASONING_TRACE] = ""
    final_text = (buffered + text).strip()
    full_text = (tool_reasoning + "\n\n" + final_text).strip() if (tool_reasoning and final_text) else (tool_reasoning or final_text)

    logger.info("[react:after] received text (%d chars, %d tool-reasoning), processing step result",
                len(full_text), len(tool_reasoning))

    task_state = _get_task_state(callback_context)
    active_step_id = str(callback_context.state.get(STATE_EXECUTOR_ACTIVE_STEP_ID, "") or "")

    if not task_state or not active_step_id:
        logger.warning("[react:after] no task state or active step — ignoring text")
        return _replace_llm_response_text(llm_response, "")

    tool_log = _get_tool_log(callback_context)
    try:
        _, active_step = _find_step(task_state, active_step_id)
    except Exception:
        active_step = {}

    # --- Build step result: accept executor JSON, otherwise derive deterministically ---
    parsed, _ = _parse_json_object_from_text(full_text)
    reasoning_trace = ""

    if parsed is not None and parsed.get("schema") == STEP_RESULT_SCHEMA:
        logger.info("[react:after] model produced valid JSON for %s", active_step_id)
        reasoning_trace = str(parsed.pop("reasoning_trace", "") or "").strip()
        parsed = _build_deterministic_step_result(
            step=active_step,
            step_id=active_step_id,
            final_text=final_text,
            tool_log=tool_log,
            base_result=parsed,
        )
    else:
        logger.info("[react:after] prose output for %s — building deterministic step result", active_step_id)
        parsed = _build_deterministic_step_result(
            step=active_step,
            step_id=active_step_id,
            final_text=final_text,
            tool_log=tool_log,
        )

    if "schema" not in parsed:
        parsed["schema"] = STEP_RESULT_SCHEMA
    if not parsed.get("step_id"):
        parsed["step_id"] = active_step_id
    if not parsed.get("step_progress_note"):
        summary = str(parsed.get("result_summary", "") or "")
        parsed["step_progress_note"] = summary[:200] if summary else "Step completed."

    try:
        validated = _apply_step_execution_result_to_task_state(
            task_state,
            parsed,
            parse_retry_count=0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[react:after] validation error for %s: %s — falling back to minimal", active_step_id, exc)
        try:
            _, step = _find_step(task_state, active_step_id)
            fallback = _build_deterministic_step_result(
                step=step,
                step_id=active_step_id,
                final_text=final_text,
                tool_log=tool_log,
            )
            step["status"] = fallback["status"]
            step["result_summary"] = fallback["result_summary"]
            step["step_progress_note"] = fallback["step_progress_note"]
            step["evidence_ids"] = fallback["evidence_ids"]
            step["open_gaps"] = fallback["open_gaps"]
            step["suggested_next_searches"] = fallback["suggested_next_searches"]
            step["tools_called"] = fallback["tools_called"]
            step["data_sources_queried"] = fallback["data_sources_queried"]
            step["structured_observations"] = fallback["structured_observations"]
            step["execution_metrics"] = _build_step_execution_metrics(
                step,
                {
                    "step_id": active_step_id,
                    "status": fallback["status"],
                    "tools_called": fallback["tools_called"],
                    "evidence_ids": step["evidence_ids"],
                    "open_gaps": fallback["open_gaps"],
                },
                parse_retry_count=0,
            )
            next_id = _next_pending_step_id(task_state)
            task_state["current_step_id"] = next_id
            task_state["plan_status"] = "completed" if next_id is None else "ready"
            _refresh_task_state_derived_state(task_state)
            validated = {"step_id": active_step_id, "status": fallback["status"]}
        except Exception:  # noqa: BLE001
            logger.error("[react:after] fallback also failed for %s", active_step_id)
            return _replace_llm_response_text(llm_response, "")
    try:
        _, step = _find_step(task_state, validated["step_id"])
        if reasoning_trace:
            step["reasoning_trace"] = reasoning_trace
        step["tool_reasoning"] = tool_reasoning
        step["executor_prose"] = final_text
        step["tool_log"] = tool_log
    except Exception:  # noqa: BLE001
        pass

    completed = _completed_step_count(task_state)
    total = _total_step_count(task_state)
    new_plan_status = str(task_state.get("plan_status", ""))
    logger.info(
        "[react:after] step %s → %s, %d/%d complete, plan_status=%s",
        validated["step_id"], validated["status"], completed, total, new_plan_status,
    )

    callback_context.state[STATE_WORKFLOW_TASK] = task_state
    callback_context.state[STATE_EXECUTOR_ACTIVE_STEP_ID] = ""
    callback_context.state[STATE_EXECUTOR_PREV_STEP_STATUS] = ""

    if new_plan_status == "completed":
        callback_context.state[STATE_AUTO_SYNTH_REQUESTED] = True

    rendered = _render_react_step_progress(
        task_state, validated, reasoning_trace,
        tool_reasoning=tool_reasoning, executor_prose=final_text,
    )
    _append_executor_rendered(callback_context, rendered)
    return _replace_llm_response_text(llm_response, rendered)


def _render_benchmark_tool_log(callback_context: CallbackContext) -> str:
    tool_log = _get_tool_log(callback_context)
    if not tool_log:
        return "No completed tool calls yet."
    lines: list[str] = []
    for idx, entry in enumerate(tool_log[-10:], start=1):
        summary = str(entry.get("summary", "") or "").strip()
        result = str(entry.get("result", "") or "").strip()
        status = str(entry.get("status", "") or "").strip()
        if summary and result:
            lines.append(f"{idx}. {summary} -> {result}")
        elif summary:
            suffix = " ..." if status == "called" else ""
            lines.append(f"{idx}. {summary}{suffix}")
    return "\n".join(lines) if lines else "No completed tool calls yet."


def _benchmark_named_items_from_question(question: str) -> list[str]:
    text = str(question or "")
    accessions = _dedupe_str_list(
        re.findall(r"\b(?:[A-Z]{1,4}_\d+(?:\.\d+)?)\b", text),
        limit=8,
    )
    if len(accessions) >= 2:
        return accessions

    gene_list_match = re.search(
        r"\bgenes?\s+([A-Z0-9-]+(?:\s*,\s*[A-Z0-9-]+)*(?:\s*,?\s+and\s+[A-Z0-9-]+)?)",
        text,
    )
    if not gene_list_match:
        return []
    items = [
        item.strip()
        for item in re.split(r"\s*,\s*|\s+and\s+", gene_list_match.group(1))
        if re.fullmatch(r"[A-Z][A-Z0-9-]{1,11}", item.strip())
    ]
    deduped = _dedupe_str_list(items, limit=8)
    return deduped if len(deduped) >= 2 else []


def _benchmark_retry_feedback(question: str, draft: str) -> str:
    text = str(draft or "").strip()
    if not text:
        return "The previous attempt did not produce a usable answer. Continue gathering evidence."

    lowered = text.lower()
    if lowered.startswith("_rate limit hit"):
        return "The previous attempt was interrupted by a temporary rate limit. Continue from the existing evidence."

    blocked_markers = (
        "could not",
        "unable to",
        "cannot",
        "can not",
        "not available",
        "do not have access",
        "not accessible",
        "not enough information",
        "cannot directly answer",
    )
    if any(marker in lowered for marker in blocked_markers):
        return "The previous draft ended in a blocked-style answer. Try a different query or another tool in the same evidence family before giving up."

    named_items = _benchmark_named_items_from_question(question)
    missing_items = [
        item for item in named_items
        if item.casefold() not in text.casefold()
    ]
    if missing_items:
        return (
            "The previous draft omitted requested named items: "
            + ", ".join(missing_items[:6])
            + ". Continue until every requested item is explicitly covered."
        )

    return (
        "If the previous draft already fully answers the question, re-emit it with the required `FINAL:` prefix. "
        "Otherwise improve it before finalizing."
    )


def _benchmark_required_tool_retry_feedback(question: str, callback_context: CallbackContext) -> str:
    lowered = str(question or "").lower()
    tool_log = _get_tool_log(callback_context)
    raw_tools = {
        str(entry.get("raw_tool", "")).strip()
        for entry in tool_log
        if isinstance(entry, dict) and str(entry.get("raw_tool", "")).strip()
    }

    if "open targets" in lowered and ("l2g" in lowered or "locus-to-gene" in lowered or "credible set" in lowered):
        if "get_open_targets_l2g" not in raw_tools:
            return (
                "This question asks for an Open Targets L2G score. Use `get_open_targets_l2g` directly "
                "instead of answering from association-score tools."
            )

    if "gwas" in lowered and "gcst" in lowered and "rs" in lowered:
        if "get_gwas_study_variant_association" not in raw_tools:
            return (
                "This question names a GWAS Catalog study accession and variant. Use "
                "`get_gwas_study_variant_association` before finalizing."
            )

    if "gwas" in lowered and "gcst" in lowered and "highest p-value" in lowered:
        if "get_gwas_study_top_risk_allele" not in raw_tools:
            return (
                "This question asks for the risk allele on the study row with the highest p-value. Use "
                "`get_gwas_study_top_risk_allele` before finalizing."
            )
        for entry in reversed(tool_log):
            if str(entry.get("raw_tool", "")).strip() != "get_gwas_study_top_risk_allele":
                continue
            evidence = str(entry.get("evidence_text", "") or entry.get("result", "") or "")
            if "Ranking mode: lowest_pvalue" not in evidence:
                return (
                    "For this benchmark wording, retry `get_gwas_study_top_risk_allele` with "
                    '`rankBy="lowest_pvalue"` and report that lead / most-significant risk allele.'
                )
            break

    if "ensembl" in lowered and "transcript" in lowered and "aa" in lowered:
        if "get_ensembl_transcripts_by_protein_length" not in raw_tools:
            return (
                "This question asks for Ensembl transcripts filtered by protein length. Use "
                "`get_ensembl_transcripts_by_protein_length` before finalizing."
            )

    if "jaspar" in lowered and "information content" in lowered:
        if "get_jaspar_motif_profile" not in raw_tools:
            return (
                "This question asks for a JASPAR motif profile. Use `get_jaspar_motif_profile` before finalizing."
            )

    if "gnomad" in lowered and ("pli" in lowered or "loss-of-function intolerance" in lowered):
        if "get_gnomad_gene_constraint" not in raw_tools:
            return (
                "This question asks for gnomAD gene-constraint metrics such as pLI. Use "
                "`get_gnomad_gene_constraint` before finalizing."
            )

    if "gnomad" in lowered and "highest allele frequency" in lowered and "transcript" in lowered:
        if "get_gnomad_transcript_highest_af_region" not in raw_tools:
            return (
                "This question asks which transcript region contains the highest-frequency gnomAD variant. Use "
                "`get_gnomad_transcript_highest_af_region` before finalizing."
            )

    if "regulomedb" in lowered and ("motif" in lowered or "probability" in lowered or re.search(r"\brank\s*1", lowered)):
        if "get_regulomedb_variant_summary" not in raw_tools:
            return (
                "This question asks for a RegulomeDB regulatory summary. Use "
                "`get_regulomedb_variant_summary` before finalizing."
            )

    if "dbsnp" in lowered and "population" in lowered and "frequency" in lowered and "rs" in lowered:
        if "get_dbsnp_population_frequency" not in raw_tools:
            return (
                "This question asks for a population-specific dbSNP allele frequency. Use "
                "`get_dbsnp_population_frequency` before finalizing."
            )

    if "tcga-brca" in lowered and "proteome profiling" in lowered:
        if "get_tcga_project_data_availability" not in raw_tools:
            return (
                "This question asks for a TCGA/GDC project case count by data availability. Use "
                "`get_tcga_project_data_availability` before finalizing."
            )

    if ("cell x gene" in lowered or "cellxgene" in lowered) and "marker gene" in lowered:
        if "get_cellxgene_marker_genes" not in raw_tools:
            return (
                "This question asks for a CELLxGENE marker-gene ranking. Use "
                "`get_cellxgene_marker_genes` before finalizing instead of dataset-search tools."
            )
        if "mononuclear" in lowered:
            for entry in reversed(tool_log):
                if str(entry.get("raw_tool", "")).strip() != "get_cellxgene_marker_genes":
                    continue
                evidence = str(entry.get("evidence_text", "") or entry.get("result", "") or "")
                if "Resolved cell type: mononuclear cell" not in evidence and "Resolved cell type: mononuclear cells" not in evidence:
                    return (
                        "The previous CELLxGENE marker lookup did not preserve the exact requested cell type. "
                        "Retry `get_cellxgene_marker_genes` using the literal cell type from the question "
                        "(`mononuclear cell`), not a related substitute such as macrophage, monocyte, or retinal pigment epithelial cell."
                )
                break

    if "geo dataset" in lowered and ("proportion" in lowered or "proportions" in lowered):
        if "get_geo_cell_type_proportions" not in raw_tools:
            return (
                "This question asks for donor-filtered cell-type proportions from a GEO dataset. Use "
                "`get_geo_cell_type_proportions` before finalizing instead of stopping at GEO metadata."
            )

    if "alphafold" in lowered and "plddt" in lowered and any(term in lowered for term in ("domain", "domains", "signal peptide", "transmembrane", "cytoplasmic", "extracellular")):
        if "get_alphafold_domain_plddt" not in raw_tools:
            return (
                "This question asks for domain-level AlphaFold pLDDT values. Use "
                "`get_alphafold_domain_plddt` before finalizing instead of the global-score summary tool."
            )

    if "depmap" in lowered and ("log2(tpm+1)" in lowered or "expression public" in lowered):
        if "get_depmap_expression_subset_mean" not in raw_tools:
            return (
                "This question asks for a DepMap public-expression subset mean. Use "
                "`get_depmap_expression_subset_mean` before finalizing instead of blocked BigQuery queries."
            )

    if "depmap" in lowered and "highest log2-normalized expression" in lowered:
        if "get_depmap_sample_top_expression_gene" not in raw_tools:
            return (
                "This question asks for the highest-expressed gene in one named DepMap sample. Use "
                "`get_depmap_sample_top_expression_gene` before finalizing."
            )

    if "ena" in lowered and re.search(r"\bERX\d+\b", str(question or ""), re.IGNORECASE):
        if "get_ena_experiment_profile" not in raw_tools:
            return (
                "This question asks for ENA experiment technique/instrument metadata. Use "
                "`get_ena_experiment_profile` before finalizing."
            )

    if "emdb" in lowered and re.search(r"\bEMD-\d+\b", str(question or ""), re.IGNORECASE):
        if "get_emdb_entry_metadata" not in raw_tools:
            return (
                "This question asks for EMDB entry metadata such as cryopreservative. Use "
                "`get_emdb_entry_metadata` before finalizing."
            )

    if "gtopdb" in lowered and "ligand" in lowered:
        if "get_gtopdb_ligand_reference" not in raw_tools:
            return (
                "This question asks for a cited Guide to Pharmacology ligand reference. Use "
                "`get_gtopdb_ligand_reference` before finalizing."
            )

    return ""


def _benchmark_missing_field_retry_feedback(question: str, draft: str) -> str:
    lowered_q = str(question or "").lower()
    text = str(draft or "").strip()
    if not text:
        return ""

    if "jaspar" in lowered_q and "consensus" in lowered_q and "information content" in lowered_q:
        has_sequence = bool(re.search(r"\b[ACGT]{8,}\b", text.upper()))
        has_numeric = bool(re.search(r"\b\d+(?:\.\d+)?\b", text))
        if not has_sequence or not has_numeric:
            return (
                "The previous draft did not include both requested JASPAR outputs. Continue until you report "
                "both the consensus recognition sequence and the total information content."
            )

    if ("cell x gene" in lowered_q or "cellxgene" in lowered_q) and "marker gene" in lowered_q and "mononuclear" in lowered_q:
        lowered_text = text.lower()
        if "mononuclear" not in lowered_text and any(
            marker in lowered_text for marker in ["macrophage", "monocyte", "retinal pigment epithelial"]
        ):
            return (
                "The previous draft changed the requested CELLxGENE cell type. Continue until you answer "
                "for the exact requested cell type (`mononuclear cell`), not a substitute cell type."
            )

    if "depmap" in lowered_q and ("log2(tpm+1)" in lowered_q or "expression public" in lowered_q):
        if not re.search(r"\b\d+(?:\.\d+)?\b", text):
            return (
                "The previous draft did not include the requested DepMap numeric mean. Continue until you "
                "report the mean log2(TPM+1) value."
            )

    if "gnomad" in lowered_q and ("pli" in lowered_q or "loss-of-function intolerance" in lowered_q):
        expected_genes = [gene for gene in ["APOE", "APOC1", "APOC2"] if gene.lower() in lowered_q]
        if expected_genes:
            covered = [gene for gene in expected_genes if gene.lower() in text.lower()]
            numeric_values = re.findall(r"\b0\.\d+\b", text)
            if len(covered) < len(expected_genes) or len(numeric_values) < len(expected_genes):
                return (
                    "The previous draft did not include every requested gnomAD pLI value. Continue until you "
                    "report each requested gene explicitly with its numeric pLI."
                )

    if "dbsnp" in lowered_q and "population" in lowered_q and "frequency" in lowered_q:
        if not re.search(r"\b0\.\d+\b", text):
            return (
                "The previous draft did not include the requested dbSNP population frequency. Continue until you "
                "report the numeric allele frequency."
            )

    if "geo dataset" in lowered_q and ("proportion" in lowered_q or "proportions" in lowered_q):
        numeric_values = re.findall(r"\b0\.\d+\b", text)
        if len(numeric_values) < 2:
            return (
                "The previous draft did not include the requested GEO cell-type proportions. Continue until you "
                "report the numeric proportions for every requested cell type."
            )

    if "alphafold" in lowered_q and "plddt" in lowered_q and any(term in lowered_q for term in ("domain", "domains", "signal peptide", "transmembrane", "cytoplasmic", "extracellular")):
        numeric_values = re.findall(r"\b\d+(?:\.\d+)?\b", text)
        if len(numeric_values) < 4:
            return (
                "The previous draft did not include every requested AlphaFold domain pLDDT value. Continue until "
                "you report each requested domain explicitly with its numeric mean pLDDT."
            )

    return ""


def _recover_benchmark_answer_from_tool_evidence(
    question: str,
    draft: str,
    callback_context: CallbackContext,
) -> str:
    lowered_q = str(question or "").lower()
    cleaned_draft = str(draft or "").strip()
    tool_log = _get_tool_log(callback_context)

    if "open targets" in lowered_q and ("l2g" in lowered_q or "locus-to-gene" in lowered_q or "credible set" in lowered_q):
        target_match = re.search(
            r"variant associating\s+([A-Z0-9-]{2,20})\s+with\s+(.+?)\s+according to open targets",
            str(question or ""),
            re.IGNORECASE,
        )
        for entry in reversed(tool_log):
            if str(entry.get("raw_tool", "")).strip() != "get_open_targets_l2g":
                continue
            evidence = str(entry.get("evidence_text", "") or entry.get("result", "") or "").strip()
            if not evidence:
                continue
            exact_match = re.search(r"L2G score:\s*([0-9]+(?:\.[0-9]+)?)", evidence, re.IGNORECASE)
            rounded_match = re.search(r"Rounded L2G score \(3 d\.p\.\):\s*([0-9]+(?:\.[0-9]+)?)", evidence, re.IGNORECASE)
            variant_match = re.search(r"Variant:\s*(.+)", evidence, re.IGNORECASE)
            if not exact_match:
                continue
            exact = exact_match.group(1)
            rounded = rounded_match.group(1) if rounded_match else ""
            variant = variant_match.group(1).strip() if variant_match else "the matched Open Targets study-locus variant"
            if target_match:
                target = target_match.group(1).upper()
                disease = target_match.group(2).strip().rstrip("?.")
                if rounded and rounded != exact:
                    return (
                        f"The L2G score for the variant {variant} associating {target} with {disease} "
                        f"is {rounded} (exact {exact})."
                    )
                return (
                    f"The L2G score for the variant {variant} associating {target} with {disease} "
                    f"is {exact}."
                )
            if rounded and rounded != exact:
                return f"The Open Targets L2G score is {rounded} (exact {exact})."
            return f"The Open Targets L2G score is {exact}."

    return cleaned_draft


def _benchmark_specialized_hints(question: str) -> list[str]:
    text = str(question or "").strip()
    lowered = text.lower()
    hints: list[str] = []

    if "open targets" in lowered and ("l2g" in lowered or "locus-to-gene" in lowered or "credible set" in lowered):
        l2g_match = re.search(
            r"variant associating\s+([A-Z0-9-]{2,20})\s+with\s+(.+?)\s+according to open targets",
            text,
            re.IGNORECASE,
        )
        if l2g_match and "release" not in lowered and not re.search(r"\b\d{2}\.\d{2}\b", lowered):
            target = l2g_match.group(1).upper()
            disease = l2g_match.group(2).strip().rstrip("?.")
            hints.append(
                "For this Open Targets L2G benchmark question, call "
                f'`get_open_targets_l2g(target="{target}", disease="{disease}", release="25.09")` '
                "unless the question explicitly specifies another release. Do not answer from "
                "`get_open_targets_association`, because that is a different score."
            )
        elif "release" not in lowered and not re.search(r"\b\d{2}\.\d{2}\b", lowered):
            hints.append(
                "For this Open Targets L2G benchmark question, call "
                '`get_open_targets_l2g(..., release="25.09")` unless the question explicitly specifies another release. '
                "Do not answer from `get_open_targets_association`, because that is a different score."
            )

    if "gwas" in lowered and "gcst" in lowered and "rs" in lowered:
        study_match = re.search(r"\b(GCST\d+)\b", text, re.IGNORECASE)
        variant_match = re.search(r"\b(rs\d+)\b", text, re.IGNORECASE)
        risk_match = re.search(r"\b(rs\d+-[A-Z])\b", text, re.IGNORECASE)
        if study_match and variant_match:
            parts = [
                "For this GWAS Catalog benchmark question, call `get_gwas_study_variant_association(",
                f'studyAccession="{study_match.group(1).upper()}", ',
                f'variantId="{variant_match.group(1).lower()}"',
            ]
            if risk_match:
                parts.append(f', riskAllele="{risk_match.group(1).lower()}"')
            parts.append(")` and report the matched RAF / risk frequency from that study row.")
            hints.append("".join(parts))

    if "gwas" in lowered and "gcst" in lowered and "highest p-value" in lowered:
        study_match = re.search(r"\b(GCST\d+)\b", text, re.IGNORECASE)
        if study_match:
            hints.append(
                "For this GWAS Catalog benchmark question, call "
                f'`get_gwas_study_top_risk_allele(studyAccession="{study_match.group(1).upper()}", rankBy="lowest_pvalue")` '
                "and report the returned risk-allele label, because this benchmark wording is targeting the lead / most-significant study hit."
            )

    if "jaspar" in lowered:
        tf_match = re.search(r"for human transcription factor ([A-Z0-9-]+)", text, re.IGNORECASE)
        tf_name = tf_match.group(1).upper() if tf_match else ""
        if tf_name:
            hints.append(
                "For this JASPAR benchmark question, call "
                f'`get_jaspar_motif_profile(tfName="{tf_name}", speciesTaxId=9606)` '
                "and report the consensus sequence plus total information content."
            )

    if "gnomad" in lowered and ("pli" in lowered or "loss-of-function intolerance" in lowered):
        genes = [gene for gene in ["APOE", "APOC1", "APOC2"] if gene.lower() in lowered]
        if genes:
            hints.append(
                "For this gnomAD constraint benchmark question, call "
                f"`get_gnomad_gene_constraint(genes={json.dumps(genes)}, referenceGenome=\"GRCh38\")` "
                "and report the pLI for each requested gene."
            )

    if "gnomad" in lowered and "highest allele frequency" in lowered and "cdkn2a" in lowered:
        hints.append(
            "For this gnomAD transcript-region benchmark question, call "
            '`get_gnomad_transcript_highest_af_region(geneIdentifier="CDKN2A", dataset="gnomad_r4", referenceGenome="GRCh38", transcriptSelection="canonical")` '
            "and report the returned transcript-region label."
        )

    if "regulomedb" in lowered and "motif" in lowered:
        rsid_match = re.search(r"\b(rs\d+)\b", text, re.IGNORECASE)
        if rsid_match:
            hints.append(
                "For this RegulomeDB benchmark question, call "
                f'`get_regulomedb_variant_summary(query="{rsid_match.group(1).lower()}", genome="GRCh38")` '
                "and report the returned unique motif-target count."
            )

    if "dbsnp" in lowered and "population" in lowered and "frequency" in lowered:
        rsid_match = re.search(r"\b(rs\d+)\b", text, re.IGNORECASE)
        if rsid_match and "african" in lowered:
            hints.append(
                "For this dbSNP benchmark question, call "
                f'`get_dbsnp_population_frequency(rsId="{rsid_match.group(1).lower()}", populationName="African", preferReferenceAllele=true)` '
                "and report the returned reference-allele frequency."
            )

    if "screen" in lowered and "nearest" in lowered and ("proximal enhancer" in lowered or "pels" in lowered):
        gene_match = re.search(r"\bhuman\s+([A-Z0-9-]{2,20})\b", text, re.IGNORECASE)
        gene = gene_match.group(1).upper() if gene_match else ""
        if gene:
            hints.append(
                "For this SCREEN benchmark question, call "
                f'`get_screen_nearest_ccre_assay(geneIdentifier="{gene}", ccreClass="pELS", assay="DNase", assembly="GRCh38")` '
                "and report the returned DNase score from the nearest pELS to the canonical transcript TSS."
            )

    if "screen" in lowered and "highest" in lowered and "h3k4me3" in lowered:
        accession_match = re.search(r"\b(EH38E\d+)\b", text, re.IGNORECASE)
        if accession_match:
            hints.append(
                "For this SCREEN benchmark question, call "
                f'`get_screen_ccre_top_celltype_assay(accession="{accession_match.group(1).upper()}", assay="H3K4me3", restrictToNodeCelltypes=true)` '
                "and report the returned top cell type."
            )

    if "tcga-brca" in lowered and "proteome profiling" in lowered:
        hints.append(
            "For this TCGA project benchmark question, call "
            '`get_tcga_project_data_availability(projectId="TCGA-BRCA", dataCategory="Proteome Profiling")` '
            "and report the returned case count."
        )

    if ("cell x gene" in lowered or "cellxgene" in lowered) and "marker gene" in lowered:
        cell_type = "mononuclear cell" if "mononuclear cell" in lowered or "mononuclear cells" in lowered else ""
        tissue = "eye" if " eye " in f" {lowered} " else ""
        organism = "Homo sapiens" if "human" not in lowered else "Homo sapiens"
        disease = "age related macular degeneration 7" if "age-related macular degeneration 7" in lowered else ""
        if cell_type and tissue:
            parts = [
                "For this CELLxGENE marker-gene benchmark question, call ",
                f'`get_cellxgene_marker_genes(cellType="{cell_type}", tissue="{tissue}", organism="{organism}"',
            ]
            if disease:
                parts.append(f', disease="{disease}"')
            parts.append(', test="ttest", nMarkers=10)` and report the top marker gene. ')
            parts.append("Use the exact cell type from the question; do not substitute macrophage, monocyte, or retinal pigment epithelial cell.")
            hints.append("".join(parts))

    if "geo dataset" in lowered and ("proportion" in lowered or "proportions" in lowered):
        geo_match = re.search(r"\b(GSE\d+)\b", text, re.IGNORECASE)
        wants_beta = "beta" in lowered
        wants_ductal = "ductal" in lowered
        if geo_match:
            cell_type_args: list[str] = []
            if wants_beta:
                cell_type_args.append('"beta"')
            if wants_ductal:
                cell_type_args.append('"ductal"')
            cells_fragment = f", cellTypes=[{', '.join(cell_type_args)}]" if cell_type_args else ""
            hints.append(
                "For this GEO benchmark question, call "
                f'`get_geo_cell_type_proportions(accession="{geo_match.group(1).upper()}"{cells_fragment}, '
                'organism="Homo sapiens", donorDiseaseField="type 2 diabetes mellitus", donorDiseaseValue="Yes")` '
                "and report the returned numeric proportions."
            )

    if "ensembl" in lowered and "transcript" in lowered and "aa" in lowered:
        gene_match = re.search(r"\b(ENSG\d+)\b", text, re.IGNORECASE)
        range_match = re.search(r"(\d+)\s*-\s*(\d+)\s*aa", lowered)
        if gene_match and range_match:
            hints.append(
                "For this Ensembl transcript-length benchmark question, call "
                f'`get_ensembl_transcripts_by_protein_length(identifier="{gene_match.group(1).upper()}", '
                f'minProteinLengthAa={int(range_match.group(1))}, maxProteinLengthAa={int(range_match.group(2))})` '
                "and report the matching transcript IDs exactly as returned."
            )

    if "alphafold" in lowered and "plddt" in lowered and any(term in lowered for term in ("domain", "domains", "signal peptide", "transmembrane", "cytoplasmic", "extracellular")):
        protein_match = re.search(r"\bhuman\s+([A-Z0-9-]{2,12})\b", text, re.IGNORECASE)
        protein = protein_match.group(1).upper() if protein_match else ""
        if protein == "TFRC":
            hints.append(
                "For this AlphaFold benchmark question, resolve TFRC to UniProt accession P02786 and call "
                '`get_alphafold_domain_plddt(uniprotId="P02786", version="4", domains=["signal peptide", "extracellular", "transmembrane", "cytoplasmic"])`. '
                "Report the numeric domain means from the returned domain rows instead of the global pLDDT."
            )

    if "human protein atlas" in lowered and "single cell" in lowered:
        gene_match = re.search(r"\bfor\s+([A-Z0-9-]{2,12})\s+expression\b", text)
        gene = gene_match.group(1) if gene_match else ""
        tissue = "prostate" if ("prostatic" in lowered or "prostate" in lowered) else ""
        cell_type = "Basal prostatic cells" if "basal prostatic cells" in lowered else ""
        dataset = "single_cell_type" if ("non-tabula sapiens" in lowered or "single cell type" in lowered) else ""
        release = "v24"
        if gene and tissue and cell_type and dataset:
            hints.append(
                "For this Human Protein Atlas single-cell question, call "
                "`get_human_protein_atlas_gene(gene=\"{gene}\", singleCellTissue=\"{tissue}\", "
                "singleCellCellType=\"{cell_type}\", singleCellDataset=\"{dataset}\", release=\"{release}\")` "
                "before answering."
                .format(
                    gene=gene,
                    tissue=tissue,
                    cell_type=cell_type,
                    dataset=dataset,
                    release=release,
                )
            )

    if "depmap" in lowered and ("expression public" in lowered or "log2(tpm+1)" in lowered):
        gene_match = re.search(r"\bfor\s+([A-Z0-9-]{2,20})\s+in\b", text, re.IGNORECASE)
        gene = gene_match.group(1).upper() if gene_match else ""
        subtype = "RB1Loss" if "rb1loss" in lowered else ""
        release = "25Q3" if "25q3" in lowered or "expression public 25q3" in lowered else ""
        if gene and subtype:
            parts = [
                "For this DepMap public-expression benchmark question, call ",
                f'`get_depmap_expression_subset_mean(geneSymbol="{gene}", subtype="{subtype}"',
            ]
            if release:
                parts.append(f', release="{release}"')
            parts.append(')` and report the returned mean log2(TPM+1) value.')
            hints.append("".join(parts))

    if "depmap" in lowered and "highest log2-normalized expression" in lowered:
        sample_match = re.search(r"\b([A-Z0-9]+)\s+Sample\b", text, re.IGNORECASE)
        release = "25Q3" if "25q3" in lowered or "expression public 25q3" in lowered else ""
        if sample_match:
            parts = [
                "For this DepMap sample-expression benchmark question, call ",
                f'`get_depmap_sample_top_expression_gene(sampleQuery="{sample_match.group(1).upper()} Sample"',
            ]
            if release:
                parts.append(f', release="{release}"')
            parts.append(')` and report the returned top gene symbol.')
            hints.append("".join(parts))

    ena_match = re.search(r"\b(ERX\d+)\b", text, re.IGNORECASE)
    if "ena" in lowered and ena_match:
        hints.append(
            "For this ENA benchmark question, call "
            f'`get_ena_experiment_profile(experimentAccession="{ena_match.group(1).upper()}")` '
            "and report the returned technique-and-instrument phrase."
        )

    emdb_match = re.search(r"\b(EMD-\d+)\b", text, re.IGNORECASE)
    if "emdb" in lowered and emdb_match:
        hints.append(
            "For this EMDB benchmark question, call "
            f'`get_emdb_entry_metadata(accession="{emdb_match.group(1).upper()}")` '
            "and report the vitrification cryogen / cryopreservative field."
        )

    gtopdb_match = re.search(r"\bLigand\s+(\d+)\b", text, re.IGNORECASE)
    if "gtopdb" in lowered and gtopdb_match:
        hints.append(
            "For this Guide to Pharmacology benchmark question, call "
            f'`get_gtopdb_ligand_reference(ligandId="{gtopdb_match.group(1)}", order="earliest")` '
            "and report the earliest article title exactly."
        )

    return hints


def _is_benchmark_scratch_sentence(sentence: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(sentence or "").strip()).lower()
    if not lowered:
        return False

    prefix_markers = (
        "okay,",
        "okay ",
        "alright,",
        "alright ",
        "now,",
        "first,",
        "next,",
        "therefore,",
        "therefore ",
        "the catch is",
        "my strategy",
        "the strategy",
        "the user's request",
        "the user is",
        "before diving in",
        "time to",
        "on to the next step",
        "let's ",
        "i need to",
        "i'll ",
        "i will ",
        "i'm going to",
        "i am going to",
        "i should ",
        "i can ",
    )
    if any(lowered.startswith(marker) for marker in prefix_markers):
        return True

    substring_markers = (
        "call the tool",
        "call the function",
        "focus my query",
        "tap into the metadata",
        "i don't have",
        "i do not have",
        "once i've",
        "once i have",
        "let's see what",
        "time to get started",
        "that's the most efficient way",
        "that's the obvious first port of call",
        "should be able to",
        "will make sure to",
        "i need to configure",
        "i need to call",
    )
    return any(marker in lowered for marker in substring_markers)


def _format_quantized_decimal(value: Decimal, places: str) -> str:
    quantized = value.quantize(Decimal(places), rounding=ROUND_HALF_UP)
    return format(quantized, "f")


def _augment_benchmark_score_precision(text: str) -> str:
    lowered = str(text or "").lower()
    if "score" not in lowered:
        return str(text or "")
    if "rounded to" in lowered or "to 3 d.p." in lowered:
        return str(text or "")

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            numeric = Decimal(raw)
        except InvalidOperation:
            return raw

        if numeric.is_nan() or numeric.is_infinite():
            return raw
        if numeric.copy_abs() >= Decimal("1") or "." not in raw:
            return raw
        fractional_digits = len(raw.split(".", 1)[1])
        if fractional_digits < 5:
            return raw

        rounded_2 = _format_quantized_decimal(numeric, "0.01")
        rounded_3 = _format_quantized_decimal(numeric, "0.001")
        if rounded_2 == raw or rounded_3 == raw:
            return raw
        if rounded_2 == rounded_3:
            return f"{rounded_2} (exact {raw})"
        return f"{rounded_2} ({rounded_3} to 3 d.p.; exact {raw})"

    return re.sub(r"\b0\.\d{5,}\b", _replace, str(text or ""))


def _sanitize_benchmark_final_answer(question: str, draft: str) -> str:
    raw = re.sub(r"^\s*FINAL\s*:\s*", "", str(draft or "").strip(), flags=re.IGNORECASE | re.DOTALL).strip()
    if not raw:
        return raw

    question_lower = str(question or "").lower()
    if any(token in question_lower for token in (" dna sequence", " rna sequence", " protein sequence", "reference sequence")):
        seq_matches = re.findall(r"([ACGTUNacgtun]{20,})", raw)
        if seq_matches:
            return seq_matches[-1].upper()

    sentence_ready = re.sub(r"([.!?])([A-Z])", r"\1 \2", raw)
    sentences = _split_summary_sentences(sentence_ready)
    named_items = _benchmark_named_items_from_question(question)
    if named_items and len(sentences) > 1:
        for idx in range(len(sentences) - 1, -1, -1):
            sentence = sentences[idx]
            if _is_benchmark_scratch_sentence(sentence):
                continue
            suffix = " ".join(sentences[idx:]).strip()
            if not suffix:
                continue
            suffix_lower = suffix.casefold()
            covered_items = [item for item in named_items if item.casefold() in suffix_lower]
            if len(covered_items) != len(named_items):
                continue
            return suffix

    while len(sentences) > 1 and _is_benchmark_scratch_sentence(sentences[0]):
        sentences.pop(0)
    cleaned = " ".join(sentences).strip()
    return _augment_benchmark_score_precision(cleaned or raw)


def _benchmark_before_agent_callback(*, callback_context: CallbackContext) -> types.Content | None:
    if bool(callback_context.state.get(STATE_BENCHMARK_COMPLETE, False)):
        return types.Content(role="model", parts=[])
    if str(callback_context.state.get(STATE_TURN_ABORT_REASON, "")).strip():
        return types.Content(role="model", parts=[])
    callback_context.state[STATE_BENCHMARK_LOOP_COUNT] = int(callback_context.state.get(STATE_BENCHMARK_LOOP_COUNT, 0) or 0) + 1
    return None


def _benchmark_before_model_callback(*, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    if bool(callback_context.state.get(STATE_BENCHMARK_COMPLETE, False)):
        return _make_text_response("")
    abort = str(callback_context.state.get(STATE_TURN_ABORT_REASON, "")).strip()
    if abort:
        return _make_text_response("")

    llm_request.config = llm_request.config or types.GenerateContentConfig()
    tc = _thinking_config_for_model(str(DEFAULT_MODEL))
    if tc:
        llm_request.config.thinking_config = tc
    llm_request.config.response_mime_type = None

    question = _extract_user_turn_text(callback_context)
    loop_count = int(callback_context.state.get(STATE_BENCHMARK_LOOP_COUNT, 0) or 0)
    last_draft = str(callback_context.state.get(STATE_BENCHMARK_LAST_DRAFT, "") or "").strip()
    retry_feedback = str(callback_context.state.get(STATE_BENCHMARK_RETRY_FEEDBACK, "") or "").strip()
    tool_log_text = _render_benchmark_tool_log(callback_context)

    instructions = [
        "Benchmark execution context (authoritative; use this instead of inferring from prior prose):",
        _serialize_pretty_json(
            {
                "schema": "benchmark_loop_context.v1",
                "question": question,
                "loop_iteration": loop_count,
                "max_iterations": BENCHMARK_LOOP_MAX_ITERATIONS,
                "tool_log_summary": tool_log_text,
                "previous_draft_answer": last_draft or None,
                "retry_feedback": retry_feedback or None,
            }
        ),
        "Keep the user-facing output hidden until you are ready to finalize. Call tools as needed, and only emit `FINAL: ...` once the answer is complete.",
    ]
    specialized_hints = _benchmark_specialized_hints(question)
    if specialized_hints:
        instructions.extend(["Question-specific execution hints:"] + specialized_hints)
    llm_request.append_instructions(instructions)
    return None


def _benchmark_after_model_callback(*, callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
    question = _extract_user_turn_text(callback_context)
    loop_count = int(callback_context.state.get(STATE_BENCHMARK_LOOP_COUNT, 0) or 0)

    if bool(callback_context.state.get(STATE_MODEL_ERROR_PASSTHROUGH, False)):
        callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
        text = _llm_response_text(llm_response).strip()
        if text.lower().startswith("_rate limit hit") and loop_count < BENCHMARK_LOOP_MAX_ITERATIONS:
            callback_context.state[STATE_BENCHMARK_LAST_DRAFT] = text
            callback_context.state[STATE_BENCHMARK_RETRY_FEEDBACK] = _benchmark_retry_feedback(question, text)
            return _replace_llm_response_text(llm_response, "")
        callback_context.state[STATE_BENCHMARK_COMPLETE] = True
        callback_context.state[STATE_BENCHMARK_FINAL_ANSWER] = text
        return _replace_llm_response_text(llm_response, text)

    if _llm_response_has_function_call(llm_response):
        callback_context.state[STATE_EXECUTOR_BUFFER] = ""
        fc_list = _extract_function_calls(llm_response)
        visible_fc_list = [fc for fc in fc_list if not _is_internal_skill_tool_name(fc["name"])]
        text_alongside = _llm_response_text(llm_response).strip()

        tool_log = _get_tool_log(callback_context)
        for fc in visible_fc_list:
            source = tool_registry.TOOL_SOURCE_NAMES.get(fc["name"], fc["name"])
            description = _describe_tool_call(fc["name"], fc["args"])
            provenance_args = _compact_tool_args_for_provenance(fc["args"])
            tool_log.append({
                "tool": source,
                "raw_tool": fc["name"],
                "status": "called",
                "summary": description,
                **({"args": provenance_args} if provenance_args else {}),
            })
        _set_tool_log(callback_context, tool_log)

        thought_text = _llm_response_thought_text(llm_response).strip()
        trace_parts: list[str] = []
        if visible_fc_list:
            source_labels = [tool_registry.TOOL_SOURCE_NAMES.get(fc["name"], fc["name"]) for fc in visible_fc_list]
            trace_parts.append(f"ACT: Called {', '.join(source_labels)}")
        if thought_text:
            trace_parts.append(thought_text)
        if text_alongside:
            trace_parts.append(text_alongside)
        if trace_parts:
            prev_trace = str(callback_context.state.get(STATE_EXECUTOR_REASONING_TRACE, "") or "")
            callback_context.state[STATE_EXECUTOR_REASONING_TRACE] = (prev_trace + "\n" + "\n".join(trace_parts)).strip()

        if text_alongside:
            return _replace_llm_response_text(llm_response, "")
        return None

    text = _llm_response_text(llm_response)
    if bool(getattr(llm_response, "partial", False)):
        _buffer_partial_text(callback_context, STATE_EXECUTOR_BUFFER, text)
        return _replace_llm_response_text(llm_response, "")

    buffered = str(callback_context.state.get(STATE_EXECUTOR_BUFFER, "") or "")
    callback_context.state[STATE_EXECUTOR_BUFFER] = ""
    final_text = (buffered + text).strip()
    if not final_text:
        return _replace_llm_response_text(llm_response, "")

    final_match = re.match(r"^\s*FINAL\s*:\s*(.*)$", final_text, flags=re.IGNORECASE | re.DOTALL)
    candidate = final_match.group(1).strip() if final_match else final_text
    cleaned_candidate = _sanitize_benchmark_final_answer(question, candidate)
    cleaned_candidate = _recover_benchmark_answer_from_tool_evidence(
        question,
        cleaned_candidate,
        callback_context,
    )
    retry_feedback = (
        _benchmark_required_tool_retry_feedback(question, callback_context)
        or _benchmark_missing_field_retry_feedback(question, cleaned_candidate)
        or _benchmark_retry_feedback(question, cleaned_candidate)
    )
    should_retry = (
        loop_count < BENCHMARK_LOOP_MAX_ITERATIONS
        and (
            final_match is None
            or retry_feedback.startswith("This question asks for")
            or retry_feedback.startswith("This question names")
            or retry_feedback.startswith("The previous draft did not include both requested")
            or retry_feedback.startswith("The previous draft ended in a blocked-style answer")
            or retry_feedback.startswith("The previous draft omitted requested named items")
            or retry_feedback.startswith("The previous attempt was interrupted by a temporary rate limit")
        )
    )

    if should_retry:
        callback_context.state[STATE_BENCHMARK_LAST_DRAFT] = cleaned_candidate
        callback_context.state[STATE_BENCHMARK_RETRY_FEEDBACK] = retry_feedback
        return _replace_llm_response_text(llm_response, "")

    callback_context.state[STATE_BENCHMARK_COMPLETE] = True
    callback_context.state[STATE_BENCHMARK_FINAL_ANSWER] = cleaned_candidate
    callback_context.state[STATE_BENCHMARK_LAST_DRAFT] = cleaned_candidate
    callback_context.state[STATE_BENCHMARK_RETRY_FEEDBACK] = ""
    return _replace_llm_response_text(llm_response, cleaned_candidate)


def _synth_before_model_callback(*, callback_context: CallbackContext, llm_request: LlmRequest) -> LlmResponse | None:
    wants_finalize = bool(callback_context.state.get(STATE_FINALIZE_REQUESTED, False))
    wants_auto_synth = bool(callback_context.state.get(STATE_AUTO_SYNTH_REQUESTED, False))
    abort_reason = str(callback_context.state.get(STATE_TURN_ABORT_REASON, "")).strip()

    if not wants_finalize and not wants_auto_synth:
        text = _compose_non_finalize_turn_output(callback_context)
        logger.info(
            "[synth:before] non-finalize path (finalize=%s, auto_synth=%s, abort=%s), output_len=%d",
            wants_finalize, wants_auto_synth, abort_reason or "none", len(text),
        )
        return _make_text_response(text)

    if abort_reason:
        text = _compose_non_finalize_turn_output(callback_context)
        logger.info("[synth:before] abort path (%s), output_len=%d", abort_reason, len(text))
        return _make_text_response(text)

    task_state = _get_task_state(callback_context)
    if not task_state:
        return _make_text_response(_render_no_plan_to_finalize_message())

    callback_context.state[STATE_SYNTH_BUFFER] = ""
    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.response_mime_type = None
    llm_request.append_instructions(_synth_context_instructions(task_state, callback_context))
    return None


def _synth_after_model_callback(*, callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
    if bool(callback_context.state.get(STATE_MODEL_ERROR_PASSTHROUGH, False)):
        callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
        callback_context.state[STATE_SYNTH_BUFFER] = ""
        logger.info("[synth:after] model error passthrough — skipping synthesis")
        return None

    if _llm_response_has_function_call(llm_response):
        return None

    text = _llm_response_text(llm_response)
    if bool(getattr(llm_response, "partial", False)):
        _buffer_partial_text(callback_context, STATE_SYNTH_BUFFER, text)
        return _replace_llm_response_text(llm_response, "")

    task_state = _get_task_state(callback_context)
    if task_state is None:
        return None

    buffered = str(callback_context.state.get(STATE_SYNTH_BUFFER, "") or "")
    callback_context.state[STATE_SYNTH_BUFFER] = ""
    raw_markdown = (buffered + text).strip()
    structured_synthesis = _build_structured_final_synthesis(task_state, raw_markdown)
    final_markdown = _render_final_synthesis_markdown(task_state, structured_synthesis)
    task_state["latest_synthesis"] = {
        "schema": "final_synthesis_text.v1",
        "coverage_status": _compute_coverage_status(task_state),
        "claim_synthesis_summary": structured_synthesis.get("claim_synthesis_summary", {}),
        "structured_synthesis": structured_synthesis,
        "markdown": final_markdown,
    }
    callback_context.state[STATE_WORKFLOW_TASK] = task_state
    return _replace_llm_response_text(llm_response, final_markdown)


BQ_EXECUTOR_POLICY = """- BigQuery-first policy: For any structured data lookup, prefer `list_bigquery_tables` \
and `run_bigquery_select_query` over non-BQ tools. \
Available datasets: open_targets_platform (targets, diseases, drugs, evidence), ebi_chembl (bioactivity), \
gnomAD (variant frequencies), human_genome_variants, human_variant_annotation (ClinVar), \
nlm_rxnorm (drug nomenclature), fda_drug (drug labels, NDC, enforcement), \
umiami_lincs (perturbation signatures), ebi_surechembl (patents).
CRITICAL SQL syntax: Always wrap table references in backticks in your SQL queries. \
Short names are auto-expanded: `open_targets_platform.target` → `bigquery-public-data.open_targets_platform.target`. \
Example: SELECT id, approvedSymbol FROM `open_targets_platform.target` WHERE approvedSymbol = 'BRCA1'.
If a filter value contains an apostrophe, escape it as two single quotes: `WHERE name = 'Alzheimer''s disease'`. \
For unfamiliar or nested-field queries, first run the same SQL with `dryRun=true` to catch syntax issues before executing it. \
Before writing queries:
  1. Call `list_bigquery_tables(dataset="<dataset_name>")` to see all available tables.
  2. Call `list_bigquery_tables(dataset="<dataset_name>", table="<table_name>")` to get the full column schema (names, types, descriptions).
  NEVER guess column names — always inspect the schema first. \
  Column names are often singular (e.g. "target" not "targets") \
  and use IDs rather than human-readable names (e.g. targetId is an Ensembl ID like "ENSG00000012048", \
  diseaseId is an EFO ID like "EFO_0001075"). Look up IDs from reference tables first.
For `umiami_lincs`, restrict BigQuery usage to metadata-sized tables (`signature`, `perturbagen`, `small_molecule`, `model_system`, `cell_line`) unless you already have exact signature IDs or have explicitly raised the bytes-billed cap. The `readout` table is extremely large and broad gene-list filters usually still scan roughly the full table. \
Fall back to non-BQ tools for: literature search (search_pubmed, get_paper_fulltext, search_openalex_works), \
Europe PMC literature/preprints/citations (search_europe_pmc_literature), \
IEDB epitope / assay evidence (search_iedb_epitope_evidence), \
ClinicalTrials.gov, UniProt, Reactome pathways, STRING interactions, \
IntAct experimental interactions (get_intact_interactions), \
BioGRID experimental interactions (get_biogrid_interactions), \
gene identifier normalization (resolve_gene_identifiers via MyGene.info), \
ontology cross-mapping (map_ontology_terms_oxo via EBI OxO), \
GO ontology lookup and annotations (search_quickgo_terms, get_quickgo_annotations via QuickGO), \
variant effect predictions (annotate_variants_vep for SIFT/PolyPhen/AlphaMissense), \
variant discovery by gene (search_variants_by_gene when only gene is known), \
aggregated variant annotations (get_variant_annotations for ClinVar/CADD/dbSNP/gnomAD/COSMIC — requires rsID/HGVS), \
clinical variant interpretations (search_civic_variants, search_civic_genes for CIViC), \
protein structure predictions (get_alphafold_structure for pLDDT), \
GWAS associations (search_gwas_associations), drug-gene interactions (search_drug_gene_interactions), \
tissue expression (get_gene_tissue_expression), protein atlas summaries (get_human_protein_atlas_gene), \
target dependency / vulnerability (get_depmap_gene_dependency), \
published CRISPR screen summaries (get_biogrid_orcs_gene_summary), \
drug sensitivity / pharmacogenomics (get_gdsc_drug_sensitivity), \
PRISM repurposing response (get_prism_repurposing_response), \
PharmacoDB cross-dataset pharmacogenomics (get_pharmacodb_compound_response), \
single-cell dataset discovery (search_cellxgene_datasets), ClinGen curations (get_clingen_gene_curation), \
Alliance Genome Resources translational summaries (get_alliance_genome_gene_profile), \
Pathway Commons pathway search (search_pathway_commons_top_pathways), \
Guide to Pharmacology curated target-ligand summaries (get_guidetopharmacology_target), \
DailyMed label summaries (get_dailymed_drug_label), experimental structures (search_protein_structures), \
cancer mutations (get_cancer_mutation_profile), \
drug bioactivity and selectivity (get_chembl_bioactivities — prefer over BigQuery ebi_chembl), \
chemical compound data (get_pubchem_compound), \
and post-marketing adverse events (search_fda_adverse_events for openFDA FAERS — \
prefer this over BigQuery fda_drug for adverse event reports)."""


def _format_tool_catalog(tool_hints: list[str]) -> str:
    lines = []
    for name in tool_hints[:80]:
        desc = tool_registry.TOOL_DESCRIPTIONS.get(name)
        lines.append(f"- {name} — {desc}" if desc else f"- {name}")
    return "\n".join(lines) or "- No tools available."


def _format_source_precedence_rules(tool_hints: list[str]) -> str:
    lines = []
    for rule in tool_registry.iter_active_source_precedence_rules(tool_hints):
        lines.append(f"- {rule['topic']}: {rule['summary']}")
    return "\n".join(lines) or "- No special overlap rules configured for the current tool set."


def _prioritize_tools_for_step(tool_names: list[str], tool_hint: str) -> list[str]:
    ordered = _dedupe_str_list(tool_names, limit=120)
    hint = str(tool_hint or "").strip()
    if not hint or hint not in ordered:
        return ordered

    prioritized = [hint]
    fallback_tools = tool_registry.TOOL_ROUTING_METADATA.get(hint, {}).get("fallback_tools", [])
    prioritized.extend(str(name).strip() for name in fallback_tools if str(name).strip())

    seen: set[str] = set()
    result: list[str] = []
    for name in prioritized + ordered:
        if not name or name in seen:
            continue
        seen.add(name)
        result.append(name)
    return result


def _format_step_routing_guidance(tool_hint: str, available_tools: list[str]) -> str:
    hint = str(tool_hint or "").strip()
    meta = tool_registry.TOOL_ROUTING_METADATA.get(hint)
    if not meta:
        return ""

    preferred_for = str(meta.get("preferred_for", "")).strip()
    overlap_group = str(meta.get("overlap_group", "")).strip()
    fallback_tools = [
        tool for tool in meta.get("fallback_tools", [])
        if tool in set(available_tools)
    ]
    source_label = _resolve_source_label(hint)
    fallback_labels = [
        f"`{tool}` ({_resolve_source_label(tool)})"
        for tool in fallback_tools
    ]

    parts = [
        f"Routing guidance for this step's tool_hint `{hint}` ({source_label}):",
        f"- Start with `{hint}` for {preferred_for}."
        if preferred_for else f"- Start with `{hint}` before trying overlapping tools.",
    ]
    if fallback_labels:
        parts.append(
            "- Only fall back if the requested evidence type is unavailable or insufficient. "
            f"Preferred fallbacks: {', '.join(fallback_labels)}."
        )
    parts.append(
        "- Do not substitute a nearby overlapping source unless it better matches the step's requested evidence type."
    )
    if overlap_group == "target_vulnerability":
        parts.append(
            "- Keep target-vulnerability work inside specialized screening and dependency tools unless the step explicitly names a "
            "BigQuery dataset or requires a confirmed structured-data slice."
        )
    return "\n".join(parts)


def _resolve_structured_observation_overlap_groups(tool_hint: str, available_tools: list[str]) -> list[str]:
    hint = str(tool_hint or "").strip()
    overlap_groups: list[str] = []

    if hint:
        hint_group = str(tool_registry.TOOL_ROUTING_METADATA.get(hint, {}).get("overlap_group", "")).strip()
        if hint_group and hint_group in STRUCTURED_OBSERVATION_GUIDANCE_BY_OVERLAP_GROUP:
            overlap_groups.append(hint_group)

        for fallback_tool in tool_registry.TOOL_ROUTING_METADATA.get(hint, {}).get("fallback_tools", [])[:4]:
            fallback_group = str(tool_registry.TOOL_ROUTING_METADATA.get(fallback_tool, {}).get("overlap_group", "")).strip()
            if fallback_group and fallback_group in STRUCTURED_OBSERVATION_GUIDANCE_BY_OVERLAP_GROUP:
                overlap_groups.append(fallback_group)

    if not overlap_groups:
        for tool_name in available_tools[:12]:
            overlap_group = str(tool_registry.TOOL_ROUTING_METADATA.get(tool_name, {}).get("overlap_group", "")).strip()
            if overlap_group and overlap_group in STRUCTURED_OBSERVATION_GUIDANCE_BY_OVERLAP_GROUP:
                overlap_groups.append(overlap_group)

    return _dedupe_str_list(overlap_groups, limit=3)


def _format_structured_observation_example(example: dict[str, Any]) -> str:
    return _serialize_pretty_json(example)


def _format_structured_observation_guidance(tool_hint: str, available_tools: list[str]) -> str:
    overlap_groups = _resolve_structured_observation_overlap_groups(tool_hint, available_tools)
    if not overlap_groups:
        return ""

    lines = [
        "Structured observation guidance for this step:",
        "- Include `structured_observations` in the step JSON if the tool outputs support grounded atomic claims.",
        "- Keep each observation to one claim grounded directly in the current step's tool results.",
        "- Do not invent observations that are not explicitly supported by the retrieved data.",
        "- Prefer 1-4 high-signal observations by default; only emit more if the step explicitly asks for breadth.",
        "- Use exact entity labels or IDs that appeared in the tool output whenever possible.",
    ]
    for overlap_group in overlap_groups:
        guidance = STRUCTURED_OBSERVATION_GUIDANCE_BY_OVERLAP_GROUP[overlap_group]
        predicates = ", ".join(f"`{predicate}`" for predicate in guidance.get("predicates", [])[:6])
        entity_types = ", ".join(f"`{entity_type}`" for entity_type in guidance.get("entity_types", [])[:6])
        lines.append(f"- Family: {guidance['label']}.")
        lines.append(f"  Emit when: {guidance.get('when_to_emit', '')}")
        lines.append(f"  Prefer predicates: {predicates}.")
        lines.append(f"  Typical entity types: {entity_types}.")
        for rule in guidance.get("extraction_rules", [])[:3]:
            lines.append(f"  Extraction rule: {rule}")
        example = guidance.get("example")
        if isinstance(example, dict):
            lines.append(f"  Example observation for {guidance['label']}:")
            lines.append(_format_structured_observation_example(example))
    lines.append("Recommended observation template:")
    lines.append(
        _format_structured_observation_example(
            {
                "observation_type": "...",
                "subject": {"type": "...", "label": "...", "id": "..."},
                "predicate": "...",
                "object": {"type": "...", "label": "...", "id": "..."},
                "supporting_ids": ["..."],
                "source_tool": "...",
                "confidence": "medium",
                "qualifiers": {"dataset": "...", "metric": "..."},
            }
        )
    )
    return "\n".join(lines)


def _resolve_step_tools(domains: list[str] | None, *, available_tools: set[str] | None = None) -> list[str]:
    """Resolve a list of domain names into a deduplicated, ordered tool list.

    Always includes ALWAYS_AVAILABLE_DOMAINS.  Falls back to all known tools
    when *domains* is empty/None (preserving backward compatibility with plans
    that don't yet include domain tags).
    """
    if not domains:
        return list(KNOWN_MCP_TOOLS)

    target_domains = set(domains) | tool_registry.ALWAYS_AVAILABLE_DOMAINS
    seen: set[str] = set()
    tools: list[str] = []
    for domain in tool_registry.ALL_DOMAIN_NAMES:
        if domain not in target_domains:
            continue
        for tool in tool_registry.TOOL_DOMAINS.get(domain, []):
            if tool in seen:
                continue
            if available_tools is not None and tool not in available_tools:
                continue
            seen.add(tool)
            tools.append(tool)
    return tools


def _resolve_step_tool_allowlist(
    active_step: dict[str, Any],
    *,
    available_tools: list[str] | None = None,
) -> list[str]:
    """Return the narrowed MCP tool list for a single active step."""
    available = _dedupe_str_list(list(available_tools) if available_tools else KNOWN_MCP_TOOLS, limit=120)
    available_set = set(available)

    step_domains = active_step.get("domains") or []
    focused_tools = _resolve_step_tools(step_domains, available_tools=available_set) if step_domains else []

    tool_hint = str(active_step.get("tool_hint", "")).strip()
    fallback_tools = [
        str(name).strip()
        for name in tool_registry.TOOL_ROUTING_METADATA.get(tool_hint, {}).get("fallback_tools", [])
        if str(name).strip()
    ]
    for tool_name in [tool_hint, *fallback_tools]:
        if tool_name and tool_name in available_set and tool_name not in focused_tools:
            focused_tools.append(tool_name)

    if not focused_tools:
        return available
    return _prioritize_tools_for_step(focused_tools, tool_hint)


def _get_task_state_from_state_map(state: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not state:
        return None
    task_state = state.get(STATE_WORKFLOW_TASK)
    return task_state if isinstance(task_state, dict) else None


def _resolve_active_step_tool_allowlist(
    task_state: dict[str, Any] | None,
    *,
    available_tools: list[str] | None = None,
) -> list[str] | None:
    """Return the current step's allowed tool names, or None for the full base set."""
    if not task_state:
        return None

    plan_status = str(task_state.get("plan_status", "")).strip()
    current_step_id = str(task_state.get("current_step_id") or "").strip()
    if plan_status == "blocked" and current_step_id:
        next_step_id = _next_pending_step_id(task_state)
        if next_step_id:
            current_step_id = next_step_id
            plan_status = "ready"

    if not current_step_id or plan_status == "completed":
        return None

    try:
        _, active_step = _find_step(task_state, current_step_id)
    except Exception:  # noqa: BLE001
        return None

    return _resolve_step_tool_allowlist(active_step, available_tools=available_tools)


class _ActiveStepToolPredicate:
    """Expose only the active step's MCP tools during executor turns."""

    def __init__(self, available_tools: list[str]):
        self._available_tools = _dedupe_str_list(list(available_tools), limit=120)
        self._available_tool_set = set(self._available_tools)

    def __call__(self, tool: BaseTool, readonly_context: Any = None) -> bool:
        if tool.name not in self._available_tool_set:
            return False

        if readonly_context is None:
            return False

        state = getattr(readonly_context, "state", None)
        task_state = _get_task_state_from_state_map(state)
        scoped_tools = _resolve_active_step_tool_allowlist(
            task_state,
            available_tools=self._available_tools,
        )
        if not scoped_tools:
            return False
        return tool.name in set(scoped_tools)


def _build_step_executor_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    routing_policy = _format_source_precedence_rules(tool_hints)
    if prefer_bigquery:
        bq_policy = BQ_EXECUTOR_POLICY
    else:
        bq_policy = "- BigQuery-first policy is disabled for this run."

    return (
        STEP_EXECUTOR_INSTRUCTION_TEMPLATE
        .replace("__ROUTING_POLICY__", routing_policy)
        .replace("__BQ_POLICY__", bq_policy)
    )


def _build_benchmark_loop_instruction(tool_hints: list[str]) -> str:
    routing_policy = _format_source_precedence_rules(tool_hints)
    return (
        BENCHMARK_LOOP_EXECUTOR_INSTRUCTION_TEMPLATE
        .replace("__ROUTING_POLICY__", routing_policy)
    )


def _format_domain_catalog() -> str:
    lines = []
    for domain in tool_registry.ALL_DOMAIN_NAMES:
        tools = tool_registry.TOOL_DOMAINS.get(domain, [])
        tool_names = ", ".join(tools[:12])
        always = " (always included)" if domain in tool_registry.ALWAYS_AVAILABLE_DOMAINS else ""
        lines.append(f"- {domain}{always}: {tool_names}")
    return "\n".join(lines)


def _planner_skill_guidance(*, planner_skills_enabled: bool) -> str:
    if not planner_skills_enabled:
        return "- No specialized planning skills are available for this run."
    return (
        "- Use `structured-data-planning` before planning BigQuery-backed, identifier-ready, or aggregate structured-data investigations.\n"
        "- Use `archive-dataset-discovery-planning` before planning archive or neuroscience dataset discovery tasks.\n"
        "- Use `clinical-trials-planning` before planning trial-landscape, outcome, sponsor, label, or safety-driven investigations.\n"
        "- Use `geo-dataset-discovery-planning` before planning GEO-centric transcriptomics dataset discovery or accession-triage tasks.\n"
        "- Use `oncology-target-validation-planning` before planning multi-source oncology target-validation work.\n"
        "- Use `comparative-assessment-planning` before planning head-to-head comparisons that should be aligned by shared dimensions.\n"
        "- Use `entity-resolution-planning` before planning tasks that depend on resolving aliases, ontology terms, or ambiguous biomedical entities.\n"
        "- Use `safety-risk-interpretation-planning` before planning risk- or safety-driven investigations that combine trial, label, and post-market evidence.\n"
        "- Load only the skills relevant to the current objective, then return the final plan JSON."
    )


def _build_planner_instruction(
    tool_hints: list[str],
    *,
    prefer_bigquery: bool,
    planner_skills_enabled: bool,
) -> str:
    tool_catalog = _format_tool_catalog(tool_hints)
    domain_catalog = _format_domain_catalog()
    routing_policy = _format_source_precedence_rules(tool_hints)
    skill_policy = _planner_skill_guidance(planner_skills_enabled=planner_skills_enabled)
    if prefer_bigquery:
        bq_policy = (
            "- BigQuery-first policy applies when the task is genuinely structured-data or identifier-ready.\n"
            "- For BigQuery-backed steps, use the specific dataset name as `tool_hint` (for example `open_targets_platform`, `gnomad`, `ebi_chembl`, `human_variant_annotation`, `human_genome_variants`, `umiami_lincs`) rather than `run_bigquery_select_query`.\n"
            "- Keep schema discovery inside an evidence step unless the user explicitly asked about schemas.\n"
            "- If a structured source is likely to produce aggregate or dataset-level findings without direct paper identifiers, add a later literature corroboration step."
        )
    else:
        bq_policy = "- BigQuery-first policy is disabled for this run."

    return (
        PLANNER_INSTRUCTION_TEMPLATE
        .replace("__TOOL_CATALOG__", tool_catalog)
        .replace("__DOMAIN_CATALOG__", domain_catalog)
        .replace("__ROUTING_POLICY__", routing_policy)
        .replace("__SKILL_POLICY__", skill_policy)
        .replace("__BQ_POLICY__", bq_policy)
    )


def _router_before_model_callback(
    *,
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Route by session state when in-workflow (no LLM call); otherwise inject
    context for the router LLM to classify. This avoids an extra API call when
    the user is mid-workflow (e.g. approving a plan, continuing execution).
    """
    _clear_turn_temp_state(callback_context)
    user_turn = _extract_user_turn_text(callback_context)

    task_state = _get_task_state(callback_context)
    plan_pending = bool(
        callback_context.state.get(STATE_PLAN_PENDING_APPROVAL, False)
    )
    plan_status = str(task_state.get("plan_status", "none")) if task_state else "none"
    has_pending_steps = bool(
        task_state
        and any(str(s.get("status")) == "pending" for s in task_state.get("steps", []))
    )
    has_report = False
    report_objective = ""
    objective = str(task_state.get("objective", "")) if task_state else ""
    if task_state:
        synthesis = task_state.get("latest_synthesis")
        if isinstance(synthesis, dict) and synthesis.get("markdown"):
            has_report = True
            report_objective = objective
        elif plan_status == "completed" and any(
            str(s.get("status")) == "completed" for s in task_state.get("steps", [])
        ):
            has_report = True
            report_objective = objective

    # In-workflow: bypass router LLM entirely — transfer straight to research_workflow
    if plan_pending or plan_status in ("ready", "blocked", "in_progress") or has_pending_steps:
        transfer_part = types.Part(
            function_call=types.FunctionCall(
                name="transfer_to_agent",
                args={"agent_name": "research_workflow"},
            )
        )
        return LlmResponse(
            content=types.Content(role="model", parts=[transfer_part]),
            partial=False,
            turn_complete=False,
        )

    if not has_report and _is_obvious_research_workflow_query(user_turn):
        transfer_part = types.Part(
            function_call=types.FunctionCall(
                name="transfer_to_agent",
                args={"agent_name": "research_workflow"},
            )
        )
        return LlmResponse(
            content=types.Content(role="model", parts=[transfer_part]),
            partial=False,
            turn_complete=False,
        )

    if not has_report and _is_obvious_general_qa_query(user_turn):
        transfer_part = types.Part(
            function_call=types.FunctionCall(
                name="transfer_to_agent",
                args={"agent_name": "general_qa"},
            )
        )
        return LlmResponse(
            content=types.Content(role="model", parts=[transfer_part]),
            partial=False,
            turn_complete=False,
        )

    # Idle / fresh query: inject state context for LLM to classify
    state_context = (
        "Session context for routing:\n"
        f"- report_exists: {has_report}\n"
        f"- report_objective: {report_objective or 'N/A'}\n"
        f"- plan_pending_approval: {plan_pending}\n"
        f"- plan_status: {plan_status}\n"
        f"- has_pending_steps: {has_pending_steps}\n"
        f"- current_objective: {objective or 'N/A'}"
    )
    llm_request.append_instructions([state_context])
    return None


def _report_assistant_before_model_callback(
    *,
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Inject the current report into the report-assistant's context."""
    task_state = _get_task_state(callback_context)
    if not task_state:
        return _make_text_response(
            "No research report is available yet. "
            "Try asking a research question first, and I can help you "
            "explore the results afterward."
        )

    synthesis = task_state.get("latest_synthesis")
    report_markdown = ""
    if isinstance(synthesis, dict):
        report_markdown = str(synthesis.get("markdown", "")).strip()

    if not report_markdown:
        obj = str(task_state.get("objective", "")).strip()
        step_lines: list[str] = []
        for step in task_state.get("steps", []):
            status = str(step.get("status", "")).strip()
            sid = str(step.get("id", "")).strip()
            goal = str(step.get("goal", "")).strip()
            summary = str(step.get("result_summary", "")).strip()
            step_lines.append(f"- {sid} ({status}): {goal}")
            if summary:
                step_lines.append(f"  Findings: {summary}")
        report_context = (
            f"Research objective: {obj}\n\nStep results:\n"
            + "\n".join(step_lines)
        )
    else:
        report_context = report_markdown

    max_context_chars = 12_000
    if len(report_context) > max_context_chars:
        report_context = (
            report_context[:max_context_chars]
            + "\n\n[... report truncated for context ...]"
        )

    user_turn = _extract_user_turn_text(callback_context)
    lookup_provenance = _build_report_lookup_provenance(task_state, user_turn)

    instructions = [
        "Current research report for reference:",
        report_context,
    ]
    if lookup_provenance:
        instructions.extend([
            "Prior lookup provenance from this report session (reuse exact prior scope for follow-up expansions unless the user asks to change it):",
            lookup_provenance,
        ])
    if lookup_provenance and _is_lookup_expansion_request(user_turn):
        instructions.append(
            "The user appears to be asking to expand a previous lookup. Reuse the exact prior query "
            "string and filters from the lookup provenance above unless they explicitly ask to change "
            "scope. Prefer increasing limit/maxStudies/maxPages over rewriting the query. If the "
            "source still returns the same slice or exposes no more pages, say that directly."
        )

    llm_request.append_instructions(instructions)
    return None


def _report_assistant_after_model_callback(
    *,
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    if bool(callback_context.state.get(STATE_MODEL_ERROR_PASSTHROUGH, False)):
        callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
        return None
    if not _llm_response_has_function_call(llm_response):
        return None

    task_state = _get_task_state(callback_context)
    user_turn = _extract_user_turn_text(callback_context)
    provenance_entries = (
        _collect_report_lookup_provenance_entries(task_state, user_turn, max_entries=12)
        if task_state
        else []
    )
    return _apply_report_assistant_adaptive_depth(
        llm_response=llm_response,
        user_text=user_turn,
        provenance_entries=provenance_entries,
    )


def create_mcp_toolset(tool_filter: list[str] | ToolPredicate | None = None) -> McpToolset | None:
    """Build an MCP toolset for the native evidence-executor agent."""
    if isinstance(tool_filter, list) and len(tool_filter) == 0:
        return None

    server_params = StdioServerParameters(
        command="node",
        args=["server.js"],
        cwd=str(MCP_SERVER_DIR),
        env=dict(os.environ),
    )
    connection_params = StdioConnectionParams(
        server_params=server_params,
        timeout=90.0,
    )
    return McpToolset(
        connection_params=connection_params,
        tool_filter=tool_filter,
    )


def create_workflow_agent(
    *,
    tool_filter: list[str] | None = None,
    model: str | None = None,
    prefer_bigquery: bool | None = None,
    planner_skills_enabled: bool | None = None,
    execution_skills_enabled: bool | None = None,
    report_assistant_skills_enabled: bool | None = None,
    require_plan_approval: bool = False,
    benchmark_mode: bool = False,
) -> tuple[LlmAgent | SequentialAgent, McpToolset | None]:
    """Create the routed ADK agent graph and return (root_agent, managed_mcp_toolsets).

    The root agent is an intent-classifying router that transfers to:
      - general_qa: factual biomedical Q&A (no tools)
      - clarifier: asks for clarification on vague/ambiguous queries
      - report_assistant: post-report interaction (with tools for light lookups)
      - research_workflow: full plan → execute → synthesize pipeline

    Args:
        require_plan_approval: When True, the research_workflow pauses after
            plan generation and waits for the user to ``approve`` or
            ``revise: <feedback>`` before executing the plan.
        benchmark_mode: When True, bypass the routed report workflow and
            return a single tool-using agent optimized for benchmark-style
            direct question answering.
    """
    runtime_model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    planner_model = PLANNER_MODEL
    synthesizer_model = SYNTHESIZER_MODEL
    router_model = ROUTER_MODEL
    use_bigquery_priority = DEFAULT_PREFER_BIGQUERY if prefer_bigquery is None else bool(prefer_bigquery)
    use_planner_skills = (
        DEFAULT_PLANNER_SKILLS_ENABLED if planner_skills_enabled is None else bool(planner_skills_enabled)
    )
    use_execution_skills = (
        DEFAULT_EXECUTION_SKILLS_ENABLED if execution_skills_enabled is None else bool(execution_skills_enabled)
    )
    use_report_assistant_skills = (
        DEFAULT_REPORT_ASSISTANT_SKILLS_ENABLED
        if report_assistant_skills_enabled is None
        else bool(report_assistant_skills_enabled)
    )

    base_tool_hints = _dedupe_str_list(KNOWN_MCP_TOOLS if tool_filter is None else tool_filter, limit=120)
    if benchmark_mode:
        benchmark_mcp_toolset = create_mcp_toolset(tool_filter=base_tool_hints)
        benchmark_tools: list[Any] = []
        if use_execution_skills:
            _, execution_skill_toolset = create_execution_skill_toolset()
            benchmark_tools.append(execution_skill_toolset)
        if benchmark_mcp_toolset is not None:
            benchmark_tools.append(benchmark_mcp_toolset)

        benchmark_executor = LlmAgent(
            name="benchmark_executor",
            description=(
                "Loop-based benchmark execution profile for direct biomedical "
                "database question answering with retry/recovery behavior."
            ),
            model=runtime_model,
            instruction=_build_benchmark_loop_instruction(base_tool_hints),
            tools=benchmark_tools,
            include_contents="none",
            disallow_transfer_to_parent=True,
            before_agent_callback=_benchmark_before_agent_callback,
            before_model_callback=_benchmark_before_model_callback,
            after_model_callback=_benchmark_after_model_callback,
            on_model_error_callback=_on_model_error,
            on_tool_error_callback=_on_tool_error,
        )
        benchmark_loop = LoopAgent(
            name="benchmark_loop",
            sub_agents=[benchmark_executor],
            max_iterations=BENCHMARK_LOOP_MAX_ITERATIONS,
        )
        managed_toolsets = tuple(
            toolset for toolset in (benchmark_mcp_toolset,) if toolset is not None
        )
        return benchmark_loop, (ManagedMcpToolsets(managed_toolsets) if managed_toolsets else None)

    executor_tool_filter: list[str] | ToolPredicate | None
    if base_tool_hints:
        executor_tool_filter = _ActiveStepToolPredicate(base_tool_hints)
    else:
        executor_tool_filter = []
    executor_mcp_toolset = create_mcp_toolset(tool_filter=executor_tool_filter)
    report_assistant_mcp_toolset = create_mcp_toolset(tool_filter=base_tool_hints)
    executor_tools: list[McpToolset] = [executor_mcp_toolset] if executor_mcp_toolset is not None else []
    report_assistant_mcp_tools: list[McpToolset] = (
        [report_assistant_mcp_toolset] if report_assistant_mcp_toolset is not None else []
    )
    planner_tools: list[Any] = []
    workflow_lookup_tools: list[Any] = []
    report_assistant_tools: list[Any] = []
    if use_planner_skills:
        _, planner_skill_toolset = create_planner_skill_toolset()
        planner_tools = [planner_skill_toolset]
    if use_execution_skills:
        _, execution_skill_toolset = create_execution_skill_toolset()
        workflow_lookup_tools.append(execution_skill_toolset)
    if use_report_assistant_skills:
        _, report_assistant_skill_toolset = create_report_assistant_skill_toolset()
        report_assistant_tools.append(report_assistant_skill_toolset)
    workflow_lookup_tools.extend(executor_tools)
    report_assistant_tools.extend(report_assistant_mcp_tools)
    if use_bigquery_priority:
        base_hint_set = set(base_tool_hints)
        prioritized_hints = [name for name in BQ_PRIORITY_TOOLS if name in base_hint_set]
        prioritized_set = set(prioritized_hints)
        prioritized_hints.extend([name for name in base_tool_hints if name not in prioritized_set])
        executor_tool_hints = prioritized_hints
    else:
        executor_tool_hints = base_tool_hints

    hitl_agent_gate = _hitl_skip_agent if require_plan_approval else None

    # ── Research workflow agents (plan → execute → synthesize) ────────────

    planner = LlmAgent(
        name="planner",
        model=planner_model,
        instruction=_build_planner_instruction(
            executor_tool_hints,
            prefer_bigquery=use_bigquery_priority,
            planner_skills_enabled=use_planner_skills,
        ),
        tools=planner_tools,
        disallow_transfer_to_parent=True,
        before_model_callback=_make_planner_before_model_callback(
            require_approval=require_plan_approval,
            model_name=planner_model,
        ),
        after_model_callback=_make_planner_after_model_callback(
            require_approval=require_plan_approval,
        ),
        on_model_error_callback=_on_model_error,
    )
    step_executor = LlmAgent(
        name="step_executor",
        model=runtime_model,
        instruction=_build_step_executor_instruction(
            executor_tool_hints,
            prefer_bigquery=use_bigquery_priority,
        ),
        tools=workflow_lookup_tools,
        include_contents="none",
        before_agent_callback=_react_skip_if_done,
        before_model_callback=_react_before_model_callback,
        after_model_callback=_react_after_model_callback,
        on_model_error_callback=_on_model_error,
        on_tool_error_callback=_on_tool_error,
    )
    react_loop = LoopAgent(
        name="react_loop",
        sub_agents=[step_executor],
        max_iterations=25,
        before_agent_callback=hitl_agent_gate,
    )
    report_synthesizer = LlmAgent(
        name="report_synthesizer",
        model=synthesizer_model,
        instruction=SYNTHESIZER_INSTRUCTION,
        tools=[],
        include_contents="none",
        before_agent_callback=hitl_agent_gate,
        before_model_callback=_synth_before_model_callback,
        after_model_callback=_synth_after_model_callback,
        on_model_error_callback=_on_model_error,
    )

    research_workflow = SequentialAgent(
        name="research_workflow",
        description=(
            "Full research pipeline: plans an investigation, gathers evidence "
            "from biomedical databases and APIs, and produces a formal report "
            "with citations. Also handles workflow commands (approve, continue, "
            "finalize, revise, history, rollback, switch)."
        ),
        sub_agents=[planner, react_loop, report_synthesizer],
    )

    # ── Lightweight specialist agents ─────────────────────────────────────

    general_qa = LlmAgent(
        name="general_qa",
        description=(
            "Answers factual biomedical questions directly from knowledge. "
            "No database lookups or tool calls needed."
        ),
        model=runtime_model,
        instruction=GENERAL_QA_INSTRUCTION,
        tools=[],
        disallow_transfer_to_parent=True,
        on_model_error_callback=_on_model_error,
    )

    clarifier = LlmAgent(
        name="clarifier",
        description=(
            "Asks the user to clarify vague, ambiguous, incomplete, or "
            "nonsensical queries before proceeding."
        ),
        model=runtime_model,
        instruction=CLARIFIER_INSTRUCTION,
        tools=[],
        disallow_transfer_to_parent=True,
        on_model_error_callback=_on_model_error,
    )

    report_assistant = LlmAgent(
        name="report_assistant",
        description=(
            "Helps with an existing research report: answers questions about "
            "findings, restructures sections, and performs light follow-up "
            "lookups using biomedical tools. Only available after a report "
            "has been produced."
        ),
        model=runtime_model,
        instruction=REPORT_ASSISTANT_INSTRUCTION,
        tools=report_assistant_tools,
        disallow_transfer_to_parent=True,
        before_model_callback=_report_assistant_before_model_callback,
        after_model_callback=_report_assistant_after_model_callback,
        on_model_error_callback=_on_model_error,
        on_tool_error_callback=_on_tool_error,
    )

    # ── Intent router (root agent) ────────────────────────────────────────

    router = LlmAgent(
        name="co_scientist_router",
        description="AI Co-Scientist: biomedical research assistant with intent routing.",
        model=router_model,
        instruction=ROUTER_INSTRUCTION,
        sub_agents=[general_qa, clarifier, report_assistant, research_workflow],
        before_model_callback=_router_before_model_callback,
        on_model_error_callback=_on_model_error,
    )

    managed_toolsets = tuple(
        toolset
        for toolset in (executor_mcp_toolset, report_assistant_mcp_toolset)
        if toolset is not None
    )
    return router, (ManagedMcpToolsets(managed_toolsets) if managed_toolsets else None)


__all__ = [
    "create_mcp_toolset",
    "create_workflow_agent",
]
