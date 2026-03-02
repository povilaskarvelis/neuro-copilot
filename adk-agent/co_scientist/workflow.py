"""
ADK-native orchestration graph for the Co-Scientist agent.

Architecture: SequentialAgent[planner → LoopAgent[step_executor] → synthesizer].
The LoopAgent implements a ReAct (Reason-Act-Observe) cycle, executing one plan
step per iteration with explicit reasoning traces. Step state is maintained in
ADK session state via callbacks.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any
import urllib.parse
import urllib.request

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import McpToolset
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from mcp.client.stdio import StdioServerParameters

logger = logging.getLogger(__name__)


MCP_SERVER_DIR = Path(__file__).resolve().parents[2] / "research-mcp"
DEFAULT_MODEL = os.getenv("ADK_NATIVE_MODEL", "gemini-2.5-flash")
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
STATE_EXECUTOR_LAST_PROSE = "temp:co_scientist_executor_last_prose"
MAX_REACT_PARSE_RETRIES = 2
STATE_EXECUTOR_PREV_STEP_STATUS = "temp:co_scientist_executor_prev_step_status"
STATE_PLAN_PENDING_APPROVAL = "co_scientist_plan_pending_approval"
STATE_MODEL_ERROR_PASSTHROUGH = "temp:co_scientist_model_error_passthrough"

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
    "list_bigquery_tables",
    "run_bigquery_select_query",
    "search_clinical_trials",
    "get_clinical_trial",
    "summarize_clinical_trials_landscape",
    "search_pubmed",
    "search_pubmed_advanced",
    "get_pubmed_abstract",
    "search_openalex_works",
    "search_openalex_authors",
    "rank_researchers_by_activity",
    "get_researcher_contact_candidates",
    "search_uniprot_proteins",
    "get_uniprot_protein_profile",
    "search_reactome_pathways",
    "get_string_interactions",
    "get_chembl_bioactivities",
    "search_fda_adverse_events",
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
    "benchmark_dataset_overview",
    "check_gpqa_access",
]

TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_bigquery_tables": "List tables in a BigQuery dataset, or inspect column schema for a specific table",
    "run_bigquery_select_query": "Run SQL on BigQuery public datasets",
    "search_clinical_trials": "Search ClinicalTrials.gov (returns NCT IDs)",
    "get_clinical_trial": "Get details of a specific clinical trial by NCT ID",
    "summarize_clinical_trials_landscape": "Aggregate trial landscape stats for a condition",
    "search_pubmed": "Search PubMed literature (returns PMIDs, titles, authors)",
    "search_pubmed_advanced": "Advanced PubMed search with field-specific queries (MeSH, author, journal)",
    "get_pubmed_abstract": "Fetch full abstract for a PMID",
    "search_openalex_works": "Search OpenAlex for papers, preprints, and citations (returns DOIs)",
    "search_openalex_authors": "Find researchers and their publication profiles",
    "rank_researchers_by_activity": "Rank authors by recent publication activity",
    "get_researcher_contact_candidates": "Get contact/affiliation info for researchers",
    "search_uniprot_proteins": "Search UniProt for protein entries",
    "get_uniprot_protein_profile": "Detailed protein profile (isoforms, PTMs, function)",
    "search_reactome_pathways": "Search biological pathway hierarchies",
    "get_string_interactions": "Get protein-protein interaction networks from STRING",
    "get_chembl_bioactivities": "Get bioactivity data (IC50, Ki, Kd) for a drug from ChEMBL — selectivity profiling",
    "search_fda_adverse_events": "Search FDA FAERS for post-marketing adverse event reports by drug name",
    "search_aba_genes": "Search Allen Brain Atlas for genes by name or acronym (mouse, human, developing mouse)",
    "search_aba_structures": "Search Allen Brain Atlas structure ontology for brain regions",
    "get_aba_gene_expression": "Get quantified gene expression across brain structures from Allen Brain Atlas ISH data",
    "search_aba_differential_expression": "Find genes differentially expressed between two brain structures (Allen Mouse Brain Atlas)",
    "search_ebrains_kg": "Search EBRAINS Knowledge Graph for neuroscience datasets, models, software, and contributors",
    "get_ebrains_kg_document": "Get detailed metadata for a specific EBRAINS Knowledge Graph resource (dataset, model, etc.)",
    "search_conp_datasets": "Search CONP dataset repositories by modality/method keywords (e.g. 'EEG', 'fMRI', 'MRI') or study names — NOT disease names. Disease queries rarely match; use broad neuroscience terms instead",
    "get_conp_dataset_details": "Get detailed metadata (README, license, topics) for a specific CONP dataset repository returned by search_conp_datasets",
    "query_neurobagel_cohorts": "Query Neurobagel public cohorts by age, sex, and imaging modality. Start broad (no filters = browse all). Avoid diagnosis filters — most public datasets lack diagnosis annotations. Use image_modal URIs for modality filtering",
    "search_openneuro_datasets": "Search OpenNeuro neuroimaging datasets by modality (MRI, MEG, EEG, PET, iEEG, behavioral). Omit modality to browse all. Returns dataset IDs for get_openneuro_dataset",
    "get_openneuro_dataset": "Get detailed metadata (name, DOI, modalities, snapshot) for an OpenNeuro dataset by ID (e.g. ds000224)",
    "search_dandi_datasets": "Search DANDI Archive neurophysiology datasets (electrophysiology, calcium imaging, behavioral). Use keywords like 'hippocampus', 'electrophysiology'. Omit query to browse recent dandisets",
    "get_dandi_dataset": "Get detailed metadata (name, version, assets, size, embargo) for a DANDI dandiset by identifier (e.g. 000003)",
    "search_nemar_datasets": "Search NEMAR EEG/MEG/iEEG datasets (nemarDatasets GitHub org). Use 'EEG', 'MEG', 'iEEG', 'resting state', 'visual'. Omit query to browse. BIDS data from OpenNeuro at SDSC",
    "get_nemar_dataset_details": "Get detailed metadata for a NEMAR dataset by repo name (e.g. nm000104)",
    "search_braincode_datasets": "Search Brain-CODE (Ontario Brain Institute) datasets in CONP. Use 'mouse', 'fBIRN', 'NDD', 'epilepsy', or omit to list all. braincode_* repos",
    "get_braincode_dataset_details": "Get detailed metadata for a Brain-CODE dataset by repo name (e.g. braincode_Mouse_Image)",
    "benchmark_dataset_overview": "Overview of available benchmark datasets",
    "check_gpqa_access": "Check access to GPQA benchmark",
}

TOOL_SOURCE_NAMES: dict[str, str] = {
    # Generic BQ tool names (fallback when no dataset hint is available)
    "list_bigquery_tables": "BigQuery",
    "run_bigquery_select_query": "BigQuery",
    # BigQuery dataset names — used as tool_hint by the planner for precise source attribution
    "open_targets_platform": "Open Targets Platform",
    "ebi_chembl": "ChEMBL",
    "gnomad": "gnomAD",
    "human_genome_variants": "Human Genome Variants",
    "human_variant_annotation": "ClinVar (BigQuery)",
    "immune_epitope_db": "IEDB",
    "nlm_rxnorm": "RxNorm",
    "fda_drug": "FDA Drug (BigQuery)",
    "umiami_lincs": "LINCS L1000",
    "ebi_surechembl": "SureChEMBL",
    # Variant annotation APIs
    "annotate_variants_vep": "Ensembl VEP",
    "get_variant_annotations": "MyVariant.info",
    # CIViC (clinical variant interpretations)
    "search_civic_variants": "CIViC",
    "search_civic_genes": "CIViC",
    # AlphaFold (protein structure predictions)
    "get_alphafold_structure": "AlphaFold API",
    # GWAS Catalog (trait-variant associations)
    "search_gwas_associations": "GWAS Catalog",
    # DGIdb (drug-gene interactions)
    "search_drug_gene_interactions": "DGIdb",
    # GTEx (tissue expression)
    "get_gene_tissue_expression": "GTEx",
    # RCSB PDB (experimental protein structures)
    "search_protein_structures": "RCSB PDB",
    # cBioPortal (cancer mutation profiles)
    "get_cancer_mutation_profile": "cBioPortal",
    # ChEMBL REST API (bioactivity & selectivity)
    "get_chembl_bioactivities": "ChEMBL API",
    # PubChem (chemical compound data)
    "get_pubchem_compound": "PubChem",
    # Allen Brain Atlas (gene expression, brain structures, differential expression)
    "search_aba_genes": "Allen Brain Atlas",
    "search_aba_structures": "Allen Brain Atlas",
    "get_aba_gene_expression": "Allen Brain Atlas",
    "search_aba_differential_expression": "Allen Brain Atlas",
    # EBRAINS Knowledge Graph (neuroscience datasets, models, software)
    "search_ebrains_kg": "EBRAINS Knowledge Graph",
    "get_ebrains_kg_document": "EBRAINS Knowledge Graph",
    # CONP (Canadian Open Neuroscience Platform) dataset catalog
    "search_conp_datasets": "CONP Datasets",
    "get_conp_dataset_details": "CONP Datasets",
    # Neurobagel public node (harmonized cohort discovery)
    "query_neurobagel_cohorts": "Neurobagel",
    # OpenNeuro (BIDS neuroimaging datasets)
    "search_openneuro_datasets": "OpenNeuro",
    "get_openneuro_dataset": "OpenNeuro",
    # DANDI Archive (neurophysiology NWB/BIDS)
    "search_dandi_datasets": "DANDI Archive",
    "get_dandi_dataset": "DANDI Archive",
    # NEMAR (EEG/MEG/iEEG from OpenNeuro)
    "search_nemar_datasets": "NEMAR",
    "get_nemar_dataset_details": "NEMAR",
    # Brain-CODE (Ontario Brain Institute, via CONP)
    "search_braincode_datasets": "Brain-CODE",
    "get_braincode_dataset_details": "Brain-CODE",
    # FDA FAERS (post-marketing adverse events)
    "search_fda_adverse_events": "FDA FAERS (openFDA)",
    # PubMed (NCBI E-utilities)
    "search_pubmed": "PubMed",
    "search_pubmed_advanced": "PubMed",
    "get_pubmed_abstract": "PubMed",
    # Remaining MCP tools (live APIs with no BQ equivalent)
    "benchmark_dataset_overview": "Benchmark Datasets",
    "check_gpqa_access": "GPQA",
    "search_clinical_trials": "ClinicalTrials.gov",
    "get_clinical_trial": "ClinicalTrials.gov",
    "summarize_clinical_trials_landscape": "ClinicalTrials.gov",
    "search_openalex_works": "OpenAlex",
    "search_openalex_authors": "OpenAlex",
    "rank_researchers_by_activity": "OpenAlex",
    "get_researcher_contact_candidates": "OpenAlex",
    "search_reactome_pathways": "Reactome",
    "get_string_interactions": "STRING",
    "search_uniprot_proteins": "UniProt",
    "get_uniprot_protein_profile": "UniProt",
}


PLANNER_INSTRUCTION_TEMPLATE = """
You are the internal planner for biomedical investigation.

Available MCP tools:
__TOOL_CATALOG__

Rules:
- Build a concrete execution plan before any evidence collection begins.
- Break the objective into ordered, atomic subtasks.
- Prioritize high-signal subtasks that reduce uncertainty first.
- Choose the number of steps needed for the objective. Avoid unnecessary fragmentation.
- Each step must include: id, goal, tool_hint, completion_condition.
- Every step must call at least one tool. Pick tool_hint from the catalog above.
- NEVER put example values or IDs in the goal or completion_condition.
- Use step ids S1, S2, S3, ... in order.

Citation requirement:
- A final report without citations is incomplete. Every plan MUST include at least one step
  whose tool_hint is a source that returns individual citable identifiers:
  search_pubmed, search_pubmed_advanced, get_pubmed_abstract, search_openalex_works,
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
4. CONCLUDE: When the step's completion condition is met (or the step is blocked), return your result.

Available MCP tools:
__TOOL_CATALOG__

Rules:
- Focus ONLY on the current step provided in the execution context.
- You MUST call at least one tool before returning a result.
- If a tool call fails or returns insufficient data, try an alternative tool or query
  (e.g. search_pubmed <-> search_openalex_works, or fall back to run_bigquery_select_query).
- If no tool can satisfy the step after trying alternatives, mark it as blocked with a clear reason.
- If the goal or completion_condition contains an example value (marked with "e.g." or similar),
  treat it as illustrative — accept any valid result that fulfills the intent, not the exact example value.
- Prioritize high-signal evidence before broad expansion.
- Surface contradictions and unresolved gaps explicitly.
- Include source identifiers when available (PMID, DOI, NCT, OpenAlex IDs, UniProt accessions, PubChem CIDs, PDB IDs, dbSNP rsIDs, ChEMBL IDs, Reactome IDs, GWAS Catalog IDs).

Evidence ID requirements:
- evidence_ids MUST be populated with real, specific identifiers returned directly by tool
  calls. Never fabricate identifiers. Use these canonical formats:
  Literature: PMID:XXXXXXXX, DOI:10.xxxx/..., NCT########, OpenAlex:WXXXXXXX, PMC########
  Databases:  UniProt:XXXXXX, PubChem:NNNN, PDB:XXXX, rsNNNNNN, CHEMBLNNNN,
              Reactome:R-HSA-NNNNNNN, GCSTNNNNNN
- When a tool returns a database record identifier (UniProt accession, PubChem CID, PDB code,
  rsID, ChEMBL ID, Reactome stable ID, GWAS Catalog study ID), always include it in evidence_ids.
- If the primary tool for this step does not return individual document IDs (e.g. BigQuery
  aggregate queries), make a secondary call to search_pubmed or search_openalex_works using
  key terms from the findings to harvest supporting PMIDs or DOIs.
- A completed step with an empty evidence_ids array is only acceptable when the step is a
  pure calculation or data transformation with no literature or trial backing.
__BQ_POLICY__

Output requirements:
- Return ONLY valid JSON (no markdown, no prose) matching this shape:
  {
    "schema": "step_execution_result.v1",
    "step_id": "S1",
    "reasoning_trace": "REASON: <why you chose this approach>\\nACT: <what tool you called and with what parameters>\\nOBSERVE: <what the results showed>\\nCONCLUDE: <your conclusion and whether the step goal is met>",
    "tools_called": ["tool_name_1", "tool_name_2"],
    "status": "completed" | "blocked",
    "step_progress_note": "<1-2 sentence progress update>",
    "result_summary": "<concise findings summary>",
    "evidence_ids": ["PMID:...", "NCT:...", "UniProt:...", "rs...", "CHEMBL..."],
    "open_gaps": ["..."],
    "suggested_next_searches": ["..."]
  }
- The reasoning_trace MUST use labeled phases (REASON, ACT, OBSERVE, CONCLUDE). Keep it concise:
  summarize intermediate tool calls briefly (one line each) and only expand on the final conclusion.
  Do NOT repeat full query results in the trace — just note what was found. This avoids output truncation.
- The tools_called list MUST contain the names of every MCP tool you invoked during this step.
"""


SYNTHESIZER_INSTRUCTION = """
You are the final biomedical report synthesizer.
You will receive structured state context (objective, plan steps, step results, coverage status, and a source_reference mapping).

Your report MUST follow this exact section structure:

## Summary
A direct, evidence-grounded answer to the research question — not just a high-level statement but an informative synthesis that conveys the key takeaways a researcher needs. Include the most important specific findings, magnitudes, or conclusions so the reader learns something substantive without having to read the full report. When the answer naturally involves multiple items, categories, or dimensions, use bullet points to make the presentation clearer. If the plan is incomplete, note that the summary is partial.

## Evidence and Methodology
Start with ONE short overview paragraph that:
- Lists the investigative steps taken and what type of evidence each provided at a high level.
- Notes how many planned steps completed vs. total (e.g. "5 of 5 planned steps completed" or "3 of 5 planned steps completed; 2 steps could not be executed due to data unavailability").
Then provide a subsection for EACH step with a status indicator in the heading (e.g. "### Step 1: <goal> — COMPLETED" or "### Step 2: <goal> — FAILED"), detailing:
- The data source queried (use the human-readable source name, not tool names).
- Key findings with specific identifiers (PMID, DOI, NCT numbers) inline.
- Why the findings matter for the research question.
- Any gaps or limitations specific to that step.
Mark failed/blocked steps clearly with what went wrong.

## Limitations
Bullet list of overall limitations and caveats.

## Potential Next Steps
Numbered list of 3+ actionable follow-ups (confirmatory checks, risk reduction, monitoring, or decision-oriented follow-up).

Rules:
- Ground every claim in the provided evidence. Do not invent unsupported claims.
- Be specific and thorough — avoid terse output.
- Use ONLY human-readable database/source names (e.g. "PubMed", "ClinicalTrials.gov"). NEVER mention tool names (like run_bigquery_select_query, search_clinical_trials, etc.).
- Include specific identifiers inline when available (PMID, DOI, NCT numbers).
- For database records, include identifiers with their canonical prefix so they can be linked: UniProt:P00533, PubChem:2244, PDB:1ABC, rs7903146, CHEMBL25, Reactome:R-HSA-1234567, GCST000001.
- NEVER include raw URLs, API endpoints, or links to JSON output.
- Return user-facing Markdown only (not JSON).
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
    )


def _llm_response_has_function_call(llm_response: LlmResponse) -> bool:
    content = getattr(llm_response, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return False
    return any(getattr(part, "function_call", None) is not None for part in parts)


def _set_temp_state(callback_context: CallbackContext, key: str, value: Any) -> None:
    callback_context.state[key] = value


def _get_temp_state(callback_context: CallbackContext, key: str, default: Any = None) -> Any:
    return callback_context.state.get(key, default)


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
    callback_context.state[STATE_EXECUTOR_LAST_PROSE] = ""
    callback_context.state[STATE_EXECUTOR_RENDERED] = ""
    callback_context.state[STATE_EXECUTOR_ACTIVE_STEP_ID] = ""
    callback_context.state[STATE_EXECUTOR_PREV_STEP_STATUS] = ""
    callback_context.state[STATE_REACT_PARSE_RETRIES] = 0
    callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False


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
            hint = (
                f"\n\n---\n_Completed {completed} of {total} steps."
            )
            if next_id:
                hint += f" Next: **{next_id}**."
            hint += " Send `continue` to execute remaining steps, or `finalize` for a partial summary._"
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
        steps.append(
            {
                "id": canonical_id,
                "goal": _as_nonempty_str(step.get("goal"), f"steps[{idx - 1}].goal"),
                "tool_hint": _as_nonempty_str(step.get("tool_hint"), f"steps[{idx - 1}].tool_hint"),
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
            "completion_condition": step["completion_condition"],
            "status": "pending",
            "result_summary": "",
            "evidence_ids": [],
            "open_gaps": [],
            "suggested_next_searches": [],
            "step_progress_note": "",
            "reasoning_trace": "",
            "tools_called": [],
        }
        for step in validated["steps"]
    ]
    objective = validated["objective"] or objective_text
    return {
        "schema": WORKFLOW_TASK_SCHEMA,
        "objective": objective,
        "objective_fingerprint": _normalize_user_text(objective_text or objective),
        "plan_status": "ready",
        "current_step_id": steps[0]["id"] if steps else None,
        "last_completed_step_id": None,
        "steps": steps,
        "success_criteria": validated["success_criteria"],
        "latest_synthesis": None,
    }


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
        lines.append(
            f"1. **{step.get('id', 'S?')}**: {step.get('goal', '').strip()} "
            f"*(source: {source_display})*"
        )
        lines.append(f"Completion: {step.get('completion_condition', '').strip()}")
    return "\n".join(lines).strip()


def _render_executor_progress_markdown(task_state: dict[str, Any], step_result: dict[str, Any]) -> str:
    step_id = str(step_result.get("step_id", "")).strip()
    _, step = _find_step(task_state, step_id)
    status = str(step.get("status", "")).strip()
    lines = [f"## Step {step_id}", ""]
    lines.append(f"**Goal:** {step.get('goal', '').strip()}")
    lines.append(f"**Status:** `{status}`")
    progress_note = str(step.get("step_progress_note", "")).strip()
    if progress_note:
        lines.append("")
        lines.append(progress_note)
    result_summary = str(step.get("result_summary", "")).strip()
    if result_summary:
        lines.append("")
        lines.append("**Key Findings**")
        lines.append(result_summary)
    evidence_ids = [str(x).strip() for x in step.get("evidence_ids", []) if str(x).strip()]
    if evidence_ids:
        lines.append("")
        lines.append("**Evidence IDs**")
        lines.extend(f"- `{eid}`" for eid in evidence_ids[:20])
    open_gaps = [str(x).strip() for x in step.get("open_gaps", []) if str(x).strip()]
    if open_gaps:
        lines.append("")
        lines.append("**Open Gaps / Uncertainty**")
        lines.extend(f"- {gap}" for gap in open_gaps[:10])
    next_searches = [str(x).strip() for x in step.get("suggested_next_searches", []) if str(x).strip()]
    if next_searches:
        lines.append("")
        lines.append("**Suggested Next Searches / Tool Calls**")
        lines.extend(f"- {item}" for item in next_searches[:10])

    completed = _completed_step_count(task_state)
    total = _total_step_count(task_state)
    next_step_id = task_state.get("current_step_id")
    footer = f"Completed {completed}/{total} steps"
    if next_step_id:
        footer += f"; next: {next_step_id}"
    elif str(task_state.get("plan_status", "")) == "completed":
        footer += "; all planned steps complete — reply `finalize` for a final summary"
    lines.append("")
    lines.append(f"_Progress: {footer}_")
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
            source = TOOL_SOURCE_NAMES.get(tool_name, "")
            if source and source not in source_labels:
                source_labels.append(source)
        tools_display = ", ".join(f"`{t}`" for t in tools_called)
        lines.append("")
        data_sources = f" ({', '.join(source_labels)})" if source_labels else ""
        lines.append(f"**Tools used:** {tools_display}{data_sources}")

    lines.append("")
    return lines


def _render_react_step_progress(task_state: dict[str, Any], result: dict[str, Any], reasoning_trace: str) -> str:
    """Render progress for a single ReAct step iteration."""
    step_id = str(result.get("step_id", "")).strip()
    try:
        _, step = _find_step(task_state, step_id)
    except Exception:  # noqa: BLE001
        step = {}
    status = str(result.get("status", step.get("status", ""))).strip()
    goal = str(step.get("goal", "")).strip()
    tools_called = list(step.get("tools_called", []) or [])

    lines = [f"### {step_id} · `{status}`", ""]
    if goal:
        lines.extend([f"**Goal:** {goal}", ""])

    react_block = _render_react_trace_block(reasoning_trace, tools_called)
    if react_block:
        lines.extend(react_block)

    progress_note = str(result.get("step_progress_note", "")).strip()
    if progress_note:
        lines.extend([progress_note, ""])
    result_summary = str(result.get("result_summary", "")).strip()
    if result_summary:
        lines.extend(["**Key Findings**", "", result_summary, ""])
    evidence_ids = [str(x).strip() for x in result.get("evidence_ids", []) if str(x).strip()]
    if evidence_ids:
        lines.extend(["**Evidence IDs**", ""])
        lines.extend(f"- `{eid}`" for eid in evidence_ids[:12])
        lines.append("")
    open_gaps = [str(x).strip() for x in result.get("open_gaps", []) if str(x).strip()]
    if open_gaps:
        lines.extend(["**Open Gaps**", ""])
        lines.extend(f"- {gap}" for gap in open_gaps[:6])
        lines.append("")

    completed = _completed_step_count(task_state)
    total = _total_step_count(task_state)
    plan_status = str(task_state.get("plan_status", ""))
    if plan_status == "completed":
        lines.append(f"_Progress: {completed}/{total} steps complete — generating final summary._")
    else:
        next_id = str(task_state.get("current_step_id") or "")
        lines.append(f"_Progress: {completed}/{total} steps complete. Next: {next_id}_")
    lines.append("")
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
    r"|\bUniProt\s*:\s*(?P<uniprot>[A-Z][A-Z0-9]{2,9})\b"
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


_LITERATURE_ID_RE = re.compile(
    r"(?i)^(PMID:|DOI:|NCT|OpenAlex:|PMC)"
)


def _is_literature_id(eid: str) -> bool:
    """True for literature/trial identifiers that belong in the References section."""
    return bool(_LITERATURE_ID_RE.match(re.sub(r"\s*:\s*", ":", eid.strip())))


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
        _CITATION_META_CACHE[cache_key] = {}
        return None
    result = (data.get("result") or {}).get(pmid)
    if not result:
        _CITATION_META_CACHE[cache_key] = {}
        return None
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


def _hyperlink_inline_ids(text: str, ref_map: dict[str, int] | None = None) -> str:
    """Replace bare inline ID mentions with links in the body text.

    Literature IDs (PMID, DOI, NCT, OpenAlex, PMC) present in ref_map are
    replaced with compact numbered citations [N] pointing to the References
    section.  Database IDs (UniProt, PubChem, PDB, rsID, ChEMBL, Reactome,
    GCST) are replaced with clickable links to the external database record.
    Skips the ## References section (already formatted) and avoids double-linking.
    """
    # Split off the References section — leave it untouched.
    refs_split = re.split(r"(?m)^## References\b", text, maxsplit=1)
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
        if m.group("pmid"):
            normalized = f"PMID:{m.group('pmid')}"
            display = f"PMID: {m.group('pmid')}"
        elif m.group("doi"):
            normalized = f"DOI:{m.group('doi')}"
            display = f"DOI: {m.group('doi')}"
        elif m.group("nct"):
            normalized = m.group("nct").upper()
            display = normalized
        elif m.group("openalex"):
            normalized = f"OpenAlex:{m.group('openalex')}"
            display = f"OpenAlex: {m.group('openalex')}"
        elif m.group("pmc"):
            normalized = m.group("pmc").upper()
            display = normalized
        elif m.group("uniprot"):
            normalized = f"UniProt:{m.group('uniprot').upper()}"
            display = f"UniProt: {m.group('uniprot').upper()}"
        elif m.group("pubchem"):
            normalized = f"PubChem:{m.group('pubchem')}"
            display = f"PubChem: {m.group('pubchem')}"
        elif m.group("pdb"):
            normalized = f"PDB:{m.group('pdb').upper()}"
            display = f"PDB: {m.group('pdb').upper()}"
        elif m.group("rsid"):
            normalized = m.group("rsid").lower()
            display = normalized
        elif m.group("chembl"):
            normalized = m.group("chembl").upper()
            display = normalized
        elif m.group("reactome"):
            normalized = f"Reactome:{m.group('reactome').upper()}"
            display = m.group("reactome").upper()
        elif m.group("gcst"):
            normalized = m.group("gcst").upper()
            display = normalized
        else:
            return m.group(0)
        ref_key = re.sub(r"\s*:\s*", ":", normalized).lower()
        if ref_map and ref_key in ref_map:
            n = ref_map[ref_key]
            return f"[{n}]"
        url = _evidence_id_to_url(normalized)
        if not url:
            return m.group(0)
        return f"[{display}]({url})"

    linked = _INLINE_ID_RE.sub(_replace_id, protected)

    # Restore placeholders.
    for idx, original in enumerate(placeholders):
        linked = linked.replace(f"\x00P{idx}\x00", original)

    return linked + refs_tail


# ---------------------------------------------------------------------------
# Evidence & Methodology helpers (used by _render_final_synthesis_markdown)
# ---------------------------------------------------------------------------


def _build_methodology_overview(task_state: dict[str, Any], completed: int, total: int, failed: int) -> str:
    """Build the one-paragraph overview for the Evidence and Methodology section."""
    step_summaries: list[str] = []
    for step in task_state.get("steps", []):
        goal = str(step.get("goal", "")).strip()
        status = str(step.get("status", "")).strip()
        source = _resolve_source_label(str(step.get("tool_hint", "")))
        source_note = f" via {source}" if source else ""
        if status == "completed":
            step_summaries.append(f"{goal}{source_note}")
        elif status == "blocked":
            step_summaries.append(f"{goal} (failed)")
        else:
            step_summaries.append(f"{goal} (not executed)")

    overview_parts = [f"This investigation comprised {total} planned steps"]
    if step_summaries:
        overview_parts.append(": " + "; ".join(step_summaries) + ".")
    else:
        overview_parts.append(".")

    if failed:
        overview_parts.append(
            f" {completed} of {total} steps completed successfully; {failed} could not be executed."
        )
    elif completed < total:
        pending = total - completed
        overview_parts.append(
            f" {completed} of {total} steps completed; {pending} remaining when the summary was requested."
        )
    else:
        overview_parts.append(f" All {total} planned steps completed successfully.")

    return "".join(overview_parts)


def _render_step_subsection(step: dict[str, Any]) -> list[str]:
    """Render a ### subsection for one step in the Evidence and Methodology section."""
    step_id = str(step.get("id", "")).strip()
    goal = str(step.get("goal", "")).strip()
    status = str(step.get("status", "")).strip()
    summary = str(step.get("result_summary", "")).strip()
    source = _resolve_source_label(str(step.get("tool_hint", "")))
    evidence_ids = [str(x).strip() for x in (step.get("evidence_ids") or []) if str(x).strip()]
    open_gaps = [str(x).strip() for x in (step.get("open_gaps") or []) if str(x).strip()]

    if status == "completed":
        status_tag = "COMPLETED"
    elif status == "blocked":
        status_tag = "FAILED"
    else:
        status_tag = (status.upper() if status else "PENDING")
    heading = f"### {step_id}: {goal} — {status_tag}" if step_id else f"### {goal} — {status_tag}"
    lines = [heading, ""]

    if source:
        lines.append(f"**Source:** {source}")
        lines.append("")

    if status == "blocked":
        lines.append("*This step could not be completed.*")
        if summary:
            lines.append(f" {summary}")
        lines.append("")
        return lines

    if status != "completed":
        lines.append(f"*Status: {status}*")
        lines.append("")
        return lines

    if summary:
        lines.append(summary)
        lines.append("")

    if evidence_ids:
        lines.append("**Key identifiers:** " + ", ".join(evidence_ids[:10]))
        lines.append("")

    if open_gaps:
        lines.append("**Open gaps:** " + "; ".join(open_gaps[:5]))
        lines.append("")

    return lines


# ---------------------------------------------------------------------------


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
        actions.append("Document confidence level and assumptions before making a downstream decision or recommendation.")
    return actions[:5]


def _postprocess_synth_markdown(task_state: dict[str, Any], raw_markdown: str) -> str:
    """Post-process the LLM's markdown output into the final report format."""
    text = str(raw_markdown or "").strip()
    if not text:
        text = "# AI Co-Scientist Report\n\n## Summary\n\nNo final summary was produced."

    # Ensure title
    has_title = "# AI Co-Scientist Report" in text
    has_old_heading = "## Final Summary" in text or "# Final Summary" in text
    if not has_title and not has_old_heading:
        text = "# AI Co-Scientist Report\n\n" + text
    elif has_old_heading and not has_title:
        text = text.replace("## Final Summary", "## Summary").replace("# Final Summary", "## Summary")
        text = "# AI Co-Scientist Report\n\n" + text

    # Inject research question callout (query only, no coverage) beneath the title
    objective = str(task_state.get("objective", "")).strip()
    if objective and "**Research Question:**" not in text:
        callout = f"\n\n> **Research Question:** {objective}\n\n---"
        title_pos = text.find("# AI Co-Scientist Report")
        if title_pos != -1:
            title_end = text.find("\n", title_pos)
            if title_end != -1:
                text = text[:title_end] + callout + text[title_end:]
            else:
                text += callout

    # Strip any old-style coverage lines the LLM may have placed at the top callout
    text = re.sub(r">\s*\n>\s*\*\*Coverage:\*\*[^\n]*\n?", "", text)

    # References: only literature IDs get numbered entries in the References section;
    # database IDs (UniProt, PubChem, PDB, rs, ChEMBL, Reactome, GCST) become
    # clickable inline links handled by _hyperlink_inline_ids.
    inline_ids = _extract_inline_ids_from_text(text)
    lit_ids = [eid for eid in inline_ids if _is_literature_id(eid)]
    ref_map = _build_ref_map(lit_ids)
    if "## References" not in text:
        refs = _build_references_section(lit_ids)
        if refs:
            next_steps_pattern = re.compile(
                r"(\n#{1,3}\s+(?:Potential\s+)?Next\s+(?:Steps?|Actions?)\b)",
                re.IGNORECASE,
            )
            m = next_steps_pattern.search(text)
            if m:
                text = text[: m.start()] + "\n\n" + refs + text[m.start() :]
            else:
                text += "\n\n" + refs

    text = _hyperlink_inline_ids(text, ref_map)

    # Ensure Next Steps section exists
    lowered = text.lower()
    if "potential next steps" not in lowered and "next steps" not in lowered and "next actions" not in lowered:
        fallback_next = _fallback_next_actions_from_task_state(task_state)
        if fallback_next:
            text += "\n\n## Potential Next Steps\n\n"
            text += "\n".join(f"{i}. {item}" for i, item in enumerate(fallback_next[:20], start=1))

    return text.strip()


def _render_final_synthesis_markdown(task_state: dict[str, Any], synthesis: dict[str, Any]) -> str:
    """Fallback renderer: builds the report from structured synthesis fields."""
    objective = str(task_state.get("objective", "")).strip()
    coverage = str(synthesis.get("coverage_status", "partial_plan"))
    completed = _completed_step_count(task_state)
    total = _total_step_count(task_state)
    failed = _failed_step_count(task_state)

    lines = ["# AI Co-Scientist Report", ""]

    if objective:
        lines += [f"> **Research Question:** {objective}", "", "---", ""]

    # Summary
    lines += ["## Summary", ""]
    direct_answer = str(synthesis.get("direct_answer", "")).strip()
    if direct_answer:
        lines.append(direct_answer)
        lines.append("")

    # Evidence and Methodology
    lines += ["## Evidence and Methodology", ""]
    lines.append(_build_methodology_overview(task_state, completed, total, failed))
    lines.append("")
    for step in task_state.get("steps", []):
        lines += _render_step_subsection(step)

    # Limitations
    limitations = [str(x).strip() for x in synthesis.get("limitations", []) if str(x).strip()]
    if limitations:
        lines += ["## Limitations", ""]
        lines.extend(f"- {item}" for item in limitations[:20])
        lines.append("")

    # References — only literature IDs go into the numbered References section;
    # database IDs get clickable inline links via _hyperlink_inline_ids.
    current_text = "\n".join(lines)
    inline_ids = _extract_inline_ids_from_text(current_text)
    lit_ids = [eid for eid in inline_ids if _is_literature_id(eid)]
    ref_map = _build_ref_map(lit_ids)
    refs = _build_references_section(lit_ids)
    if refs:
        lines += refs.split("\n")
        lines.append("")

    body_so_far = "\n".join(lines)
    body_so_far = _hyperlink_inline_ids(body_so_far, ref_map)
    lines = body_so_far.split("\n")

    # Next Steps (after References so _strip_next_steps_section in ui_server preserves References)
    next_actions = [str(x).strip() for x in synthesis.get("next_actions", []) if str(x).strip()]
    if not next_actions:
        next_actions = _fallback_next_actions_from_task_state(task_state)
    if next_actions:
        lines += ["## Potential Next Steps", ""]
        for i, item in enumerate(next_actions[:20], start=1):
            lines.append(f"{i}. {item}")
        lines.append("")

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
        "Do not include markdown fences or commentary."
    )


def _react_step_context_instructions(task_state: dict[str, Any], active_step: dict[str, Any]) -> list[str]:
    """Build context for the ReAct step executor — focuses on ONE step."""
    prior_completed = _compact_completed_step_summaries(task_state)
    remaining_count = sum(
        1 for s in task_state.get("steps", [])
        if str(s.get("status", "")).strip() in {"pending", "in_progress"}
    )
    payload = {
        "schema": "react_step_context.v1",
        "objective": task_state.get("objective", ""),
        "current_step": {
            "id": active_step.get("id"),
            "goal": active_step.get("goal"),
            "tool_hint": active_step.get("tool_hint"),
            "completion_condition": active_step.get("completion_condition"),
        },
        "remaining_steps_after_this": remaining_count - 1,
        "prior_completed_steps": prior_completed,
    }
    return [
        "Execution context (authoritative; use this instead of inferring from prior prose):",
        _serialize_pretty_json(payload),
        (
            f"Execute ONLY step {active_step.get('id')}. "
            f"There are {remaining_count - 1} more step(s) after this one. "
            "Call at least one tool, then return ONLY valid JSON matching "
            "`step_execution_result.v1`. Include your reasoning_trace. "
            "Do not include markdown fences or extra commentary."
        ),
    ]


def _resolve_source_label(tool_hint: str) -> str:
    """Map a tool_hint to its human-readable database/source name."""
    tool_hint = str(tool_hint or "").strip()
    if not tool_hint:
        return ""
    return TOOL_SOURCE_NAMES.get(tool_hint, tool_hint)


def _synth_context_instructions(task_state: dict[str, Any], callback_context: CallbackContext | None = None) -> list[str]:
    payload = {
        "schema": "synthesis_context.v1",
        "objective": task_state.get("objective", ""),
        "plan_status": task_state.get("plan_status", "ready"),
        "coverage_status": _compute_coverage_status(task_state),
        "steps": [
            {
                "id": step.get("id"),
                "goal": step.get("goal"),
                "tool_hint": step.get("tool_hint", ""),
                "source": _resolve_source_label(step.get("tool_hint", "")),
                "status": step.get("status"),
                "reasoning_trace": step.get("reasoning_trace", ""),
                "tools_called": list(step.get("tools_called", []) or []),
                "data_sources_queried": _dedupe_str_list(
                    [TOOL_SOURCE_NAMES.get(t, t) for t in (step.get("tools_called") or [])],
                    limit=10,
                ),
                "result_summary": step.get("result_summary", ""),
                "evidence_ids": list(step.get("evidence_ids", []) or [])[:20],
                "open_gaps": list(step.get("open_gaps", []) or [])[:10],
            }
            for step in task_state.get("steps", [])
        ],
    }

    used_sources: dict[str, str] = {}
    for step in task_state.get("steps", []):
        hint = str(step.get("tool_hint", "")).strip()
        if hint and hint not in used_sources:
            used_sources[hint] = TOOL_SOURCE_NAMES.get(hint, hint)
    if used_sources:
        payload["source_reference"] = {
            tool: source for tool, source in used_sources.items()
        }

    instructions = [
        "Synthesis context (authoritative; use this instead of inferring from prior prose):",
        _serialize_pretty_json(payload),
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
        "Summary, Evidence and Methodology (with per-step subsections), Limitations, Potential Next Steps."
    )
    return instructions


def _validate_step_execution_result(raw: dict[str, Any]) -> dict[str, Any]:
    if str(raw.get("schema", "")).strip() != STEP_RESULT_SCHEMA:
        raise ValueError(f"schema must be {STEP_RESULT_SCHEMA}")
    status = str(raw.get("status", "")).strip().lower()
    if status not in {"completed", "blocked"}:
        raise ValueError("status must be `completed` or `blocked`")
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
        "tools_called": _as_string_list(raw.get("tools_called"), "tools_called", limit=20),
    }



def _apply_step_execution_result_to_task_state(
    task_state: dict[str, Any],
    step_result: dict[str, Any],
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

    if validated["status"] == "completed":
        task_state["last_completed_step_id"] = validated["step_id"]
        next_step_id = _next_pending_step_id(task_state)
        task_state["current_step_id"] = next_step_id
        task_state["plan_status"] = "completed" if next_step_id is None else "ready"
    else:
        task_state["current_step_id"] = validated["step_id"]
        task_state["plan_status"] = "blocked"

    return validated



def _validate_final_synthesis(raw: dict[str, Any]) -> dict[str, Any]:
    if str(raw.get("schema", "")).strip() != FINAL_SYNTHESIS_SCHEMA:
        raise ValueError(f"schema must be {FINAL_SYNTHESIS_SCHEMA}")
    mode = str(raw.get("mode", "")).strip().lower()
    if mode != "final":
        raise ValueError("mode must be `final`")
    coverage_status = str(raw.get("coverage_status", "")).strip().lower()
    if coverage_status not in {"complete_plan", "partial_plan"}:
        raise ValueError("coverage_status must be `complete_plan` or `partial_plan`")
    return {
        "schema": FINAL_SYNTHESIS_SCHEMA,
        "mode": "final",
        "coverage_status": coverage_status,
        "direct_answer": _as_nonempty_str(raw.get("direct_answer"), "direct_answer"),
        "supporting_evidence": _as_string_list(raw.get("supporting_evidence"), "supporting_evidence", limit=30),
        "limitations": _as_string_list(raw.get("limitations"), "limitations", limit=20),
        "next_actions": _as_string_list(raw.get("next_actions"), "next_actions", limit=20),
    }


def _make_planner_before_model_callback(*, require_approval: bool):
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
                "Send `finalize` to regenerate the report, or `continue` to resume execution."
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
                llm_request.config.response_mime_type = "application/json"
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
        llm_request.config.response_mime_type = "application/json"
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


RATE_LIMIT_BACKOFF_SECONDS = 30


def _on_model_error(
    *,
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
    """Handle model-level errors. Auto-retries rate limits after a backoff."""
    error_type = type(error).__name__
    error_msg = str(error)
    logger.error("Model error in %s: [%s] %s", "agent", error_type, error_msg)

    callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = True

    is_rate_limit = any(
        hint in error_msg.lower()
        for hint in ("429", "resource exhausted", "rate limit", "quota")
    )
    if is_rate_limit:
        logger.info(
            "Rate limit detected — waiting %ds before auto-retry",
            RATE_LIMIT_BACKOFF_SECONDS,
        )
        time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
        user_msg = (
            f"_Rate limit hit — waited {RATE_LIMIT_BACKOFF_SECONDS}s, retrying…_"
        )
    else:
        user_msg = (
            f"## Execution Error\n\n"
            f"A model error occurred: **{error_type}**\n\n"
            f"`{error_msg[:300]}`\n\n"
            "Send `continue` to retry."
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

    callback_context.state[STATE_EXECUTOR_ACTIVE_STEP_ID] = current_step_id
    callback_context.state[STATE_EXECUTOR_PREV_STEP_STATUS] = str(active_step.get("status", "pending"))
    active_step["status"] = "in_progress"
    callback_context.state[STATE_WORKFLOW_TASK] = task_state

    retries = int(callback_context.state.get(STATE_REACT_PARSE_RETRIES, 0) or 0)
    logger.info("[react:before] executing step %s (retry=%d): %s", current_step_id, retries, active_step.get("goal", ""))

    llm_request.config = llm_request.config or types.GenerateContentConfig()
    llm_request.config.response_mime_type = None
    instructions = _react_step_context_instructions(task_state, active_step)
    if retries > 0:
        last_prose = str(callback_context.state.get(STATE_EXECUTOR_LAST_PROSE, "") or "").strip()
        if last_prose:
            instructions.append(
                "CRITICAL — JSON FORMAT REQUIRED: Your previous response was plain prose, not JSON. "
                "Your previous output was:\n"
                f"---\n{last_prose}\n---\n"
                "Do NOT make any more tool calls. Do NOT repeat the prose. "
                "Using ONLY the findings above, output a single raw JSON object matching "
                "`step_execution_result.v1`. The very first character of your response MUST be `{`."
            )
        else:
            instructions.append(
                "CRITICAL — JSON FORMAT REQUIRED: Your previous response could not be parsed as JSON. "
                "You MUST return ONLY a raw JSON object matching `step_execution_result.v1`. "
                "Do NOT make additional tool calls. Do NOT write prose or markdown. "
                "The very first character of your response MUST be `{`."
            )
    llm_request.append_instructions(instructions)
    return None


def _react_after_model_callback(*, callback_context: CallbackContext, llm_response: LlmResponse) -> LlmResponse | None:
    """ReAct step executor: parse single-step result, store trace, advance."""
    if bool(callback_context.state.get(STATE_MODEL_ERROR_PASSTHROUGH, False)):
        callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
        callback_context.state[STATE_EXECUTOR_BUFFER] = ""
        logger.info("[react:after] model error passthrough — skipping JSON parse")
        return None

    if _llm_response_has_function_call(llm_response):
        logger.debug("[react:after] response has function_call — continuing tool loop")
        callback_context.state[STATE_EXECUTOR_BUFFER] = ""
        text_alongside = _llm_response_text(llm_response)
        if text_alongside.strip():
            return _replace_llm_response_text(llm_response, "")
        return None

    text = _llm_response_text(llm_response)
    if bool(getattr(llm_response, "partial", False)):
        _buffer_partial_text(callback_context, STATE_EXECUTOR_BUFFER, text)
        return _replace_llm_response_text(llm_response, "")

    if not text and not str(callback_context.state.get(STATE_EXECUTOR_BUFFER, "") or ""):
        logger.debug("[react:after] empty text (no-op iteration)")
        return None

    logger.info("[react:after] received text (%d chars), parsing step result", len(text))

    parsed, parse_error = _consume_buffered_json_object(
        callback_context,
        buffer_key=STATE_EXECUTOR_BUFFER,
        llm_response=llm_response,
    )

    task_state = _get_task_state(callback_context)
    active_step_id = str(callback_context.state.get(STATE_EXECUTOR_ACTIVE_STEP_ID, "") or "")
    prev_status = str(callback_context.state.get(STATE_EXECUTOR_PREV_STEP_STATUS, "") or "pending")

    def _restore_step_status_on_failure() -> None:
        if not task_state or not active_step_id:
            return
        try:
            _, step = _find_step(task_state, active_step_id)
        except Exception:  # noqa: BLE001
            return
        step["status"] = prev_status or "pending"
        callback_context.state[STATE_WORKFLOW_TASK] = task_state

    retries = int(callback_context.state.get(STATE_REACT_PARSE_RETRIES, 0) or 0)

    def _handle_step_error(error_label: str, error_msg: str) -> LlmResponse:
        """Retry the step or mark it blocked after max retries."""
        _restore_step_status_on_failure()
        if retries < MAX_REACT_PARSE_RETRIES:
            callback_context.state[STATE_REACT_PARSE_RETRIES] = retries + 1
            logger.warning(
                "[react:after] %s for step %s (retry %d/%d) — will retry",
                error_label, active_step_id, retries + 1, MAX_REACT_PARSE_RETRIES,
            )
            return _replace_llm_response_text(llm_response, "")

        callback_context.state[STATE_REACT_PARSE_RETRIES] = 0
        last_prose = str(callback_context.state.get(STATE_EXECUTOR_LAST_PROSE, "") or "").strip()
        callback_context.state[STATE_EXECUTOR_LAST_PROSE] = ""

        # If the model returned actual prose (it gathered data but couldn't format JSON),
        # salvage it as a completed step rather than blocking entirely.
        if last_prose and task_state and active_step_id:
            logger.warning(
                "[react:after] %s for step %s — max retries exhausted but prose output found; "
                "salvaging as completed step",
                error_label, active_step_id,
            )
            try:
                _, step = _find_step(task_state, active_step_id)
                step["status"] = "completed"
                step["result_summary"] = last_prose[:1500]
                step["reasoning_trace"] = (
                    f"Partial result (JSON formatting failed after {MAX_REACT_PARSE_RETRIES + 1} "
                    f"attempts). {error_label}: {error_msg}"
                )
                next_id = _next_pending_step_id(task_state)
                task_state["current_step_id"] = next_id
                new_plan_status = "completed" if next_id is None else "ready"
                task_state["plan_status"] = new_plan_status
                callback_context.state[STATE_WORKFLOW_TASK] = task_state
                if new_plan_status == "completed":
                    callback_context.state[STATE_AUTO_SYNTH_REQUESTED] = True
                rendered = (
                    f"### {active_step_id} · `partial` _(output recovered from prose)_\n\n"
                    f"{last_prose[:800]}{'…' if len(last_prose) > 800 else ''}\n\n---"
                )
                _append_executor_rendered(callback_context, rendered)
                return _replace_llm_response_text(llm_response, rendered)
            except Exception:  # noqa: BLE001
                pass

        logger.error(
            "[react:after] %s for step %s — max retries exhausted, marking blocked",
            error_label, active_step_id,
        )
        if task_state and active_step_id:
            try:
                _, step = _find_step(task_state, active_step_id)
                step["status"] = "blocked"
                step["result_summary"] = f"Step failed after {MAX_REACT_PARSE_RETRIES + 1} attempts: {error_msg}"
                step["reasoning_trace"] = f"Execution failed: {error_label}. {error_msg}"
                next_id = _next_pending_step_id(task_state)
                task_state["current_step_id"] = next_id
                new_plan_status = "completed" if next_id is None else "blocked"
                task_state["plan_status"] = new_plan_status
                callback_context.state[STATE_WORKFLOW_TASK] = task_state
                if new_plan_status == "completed":
                    callback_context.state[STATE_AUTO_SYNTH_REQUESTED] = True
            except Exception:  # noqa: BLE001
                pass
        rendered = (
            f"### {active_step_id} · `blocked`\n\n"
            f"**Reason:** {error_label} — {error_msg}\n\n"
            f"_Attempted {MAX_REACT_PARSE_RETRIES + 1} times. Moving to next step._\n\n---"
        )
        _append_executor_rendered(callback_context, rendered)
        return _replace_llm_response_text(llm_response, rendered)

    if parsed is None or task_state is None:
        logger.warning("[react:after] parse failed (parse_error=%s)", parse_error)
        raw_combined = (str(callback_context.state.get(STATE_EXECUTOR_BUFFER, "") or "") + text).strip()
        if raw_combined:
            callback_context.state[STATE_EXECUTOR_LAST_PROSE] = raw_combined[:3000]
        return _handle_step_error("JSON parse error", parse_error or "Failed to parse step result JSON")

    reasoning_trace = str(parsed.pop("reasoning_trace", "") or "").strip()

    if "schema" not in parsed:
        parsed["schema"] = STEP_RESULT_SCHEMA

    try:
        validated = _apply_step_execution_result_to_task_state(task_state, parsed)
    except Exception as exc:  # noqa: BLE001
        logger.error("[react:after] validation error: %s", exc)
        return _handle_step_error("Validation error", str(exc))

    callback_context.state[STATE_REACT_PARSE_RETRIES] = 0

    if reasoning_trace:
        _, step = _find_step(task_state, validated["step_id"])
        step["reasoning_trace"] = reasoning_trace

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

    rendered = _render_react_step_progress(task_state, validated, reasoning_trace)
    _append_executor_rendered(callback_context, rendered)
    return _replace_llm_response_text(llm_response, rendered)


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
    final_markdown = _postprocess_synth_markdown(task_state, (buffered + text).strip())
    task_state["latest_synthesis"] = {
        "schema": "final_synthesis_text.v1",
        "coverage_status": _compute_coverage_status(task_state),
        "markdown": final_markdown,
    }
    callback_context.state[STATE_WORKFLOW_TASK] = task_state
    return _replace_llm_response_text(llm_response, final_markdown)


BQ_DATASET_CATALOG = """Available BigQuery datasets (query via `list_bigquery_tables` and `run_bigquery_select_query`):

  === Drug targets, diseases & evidence ===
  **bigquery-public-data.open_targets_platform** (61 tables) — Comprehensive drug target data.
    IMPORTANT: Always run `list_bigquery_tables(dataset="open_targets_platform")` first to see all
    tables and then `SELECT * FROM open_targets_platform.<table> LIMIT 1` to inspect column names
    before writing queries. Table and column names are NOT obvious — do NOT guess them.

    Key tables and their primary columns:
    - **target**: id (Ensembl gene ID, e.g. "ENSG00000012048"), approvedSymbol ("BRCA1"),
      approvedName, biotype, tractability, safetyLiabilities, pathways, proteinIds
    - **disease**: id (MONDO or EFO ID — many common diseases use MONDO_* as primary ID,
      e.g. "MONDO_0004975" for Alzheimer's, "EFO_0001075" for ovarian carcinoma),
      name, therapeuticAreas, synonyms, ontology. Always query by name first, accept whatever ID is returned.
    - **evidence**: targetId (Ensembl ID), diseaseId (MONDO or EFO ID), datasourceId, datatypeId,
      score, literature, drugId. Filter by targetId + diseaseId, NOT gene symbols.
    - **known_drug**, **drug_molecule**, **drug_mechanism_of_action**, **drug_warning**: drug data
    - **evidence_*** (evidence_gwas_credible_sets, evidence_chembl, evidence_clingen, etc.): typed evidence
    - **openfda_significant_adverse_drug_reactions**: post-marketing safety signals
    - **go**, **mouse_phenotype**, **literature**: gene ontology, phenotypes, literature mining
    - **l2g_prediction**, **credible_set**, **colocalisation**: genetic evidence, variant-to-gene mapping
    - **expression**: tissue/cell expression data
    - **interaction**, **interaction_evidence**: protein-protein interactions

    Query pattern: To find evidence for a gene + disease, first look up the Ensembl ID from
    `target` (WHERE approvedSymbol = 'BRCA1') and the EFO ID from `disease` (WHERE name LIKE
    '%ovarian%'), then query `evidence` using those IDs.

  === Chemistry & bioactivity ===
  **bigquery-public-data.ebi_chembl** — Bioactive compounds, target bioactivity (IC50/Ki/EC50), assay data,
    mechanism of action, drug indications. Tables: activities, compound_records, target_dictionary, etc.
  **bigquery-public-data.ebi_surechembl** — Chemical structures extracted from patents. Tables: map, match.
    Use for chemical IP landscape and freedom-to-operate analysis.

  === Genomics & variants ===
  **bigquery-public-data.gnomAD** — Population allele frequencies by chromosome (v2_1_1_exomes__chr*,
    v2_1_1_genomes__chr*). Use for variant frequency lookups and gene constraint analysis.
  **bigquery-public-data.human_genome_variants** — 1000 Genomes Phase 3 variants, Platinum Genomes,
    Simons Genome Diversity Project. Tables: *_variants_*, *_sample_info, *_pedigree.
  **bigquery-public-data.human_variant_annotation** — ClinVar variant annotations (hg19/hg38 builds).
    Contains clinical significance classifications (pathogenic, benign, etc.), variant types, and condition
    associations. Does NOT contain SIFT/PolyPhen prediction scores.

  === Immunology ===
  **bigquery-public-data.immune_epitope_db** — IEDB: immune epitopes, B-cell assays, MHC ligand binding,
    T-cell receptor data. Tables: epitope_full_v3, bcell_full_v3, mhc_ligand_full, receptor_full_v3, etc.
    Use for immunotherapy target assessment and vaccine design.

  === Drug nomenclature & regulatory ===
  **bigquery-public-data.nlm_rxnorm** — RxNorm drug nomenclature, ingredient relationships, and
    clinical drug pathways. Use for standardizing drug names across sources.
  **bigquery-public-data.fda_drug** — FDA drug labels, NDC product listings, enforcement actions.
    Use for label information and regulatory data. NOTE: for post-marketing adverse event reports
    (FAERS), use the search_fda_adverse_events tool instead — it queries the openFDA API directly
    and returns richer data than the BigQuery tables.

  === Perturbation biology ===
  **bigquery-public-data.umiami_lincs** — LINCS L1000 perturbation signatures: cell lines, small molecules,
    nucleic acid reagents, readouts, and perturbation signatures. Use for drug repurposing hypotheses
    and mechanism-of-action characterization.

  **How to query**: Always wrap table names in backticks in SQL. Short names are auto-expanded:
  `open_targets_platform.target` → `bigquery-public-data.open_targets_platform.target`.
  Example: SELECT id, approvedSymbol FROM `open_targets_platform.target` WHERE approvedSymbol = 'BRCA1'.

  Start every structured data lookup with BigQuery. Use `list_bigquery_tables` to discover tables, \
and `list_bigquery_tables(dataset="...", table="...")` to inspect column schemas before writing queries.
  Write Standard SQL via `run_bigquery_select_query`. Use non-BigQuery MCP tools for:
    - Literature search: search_pubmed, search_pubmed_advanced, get_pubmed_abstract (PubMed/NCBI)
    - Literature search: search_openalex_works (OpenAlex — broader coverage, preprints)
    - Clinical trials: search_clinical_trials, get_clinical_trial, summarize_clinical_trials_landscape
    - Researcher discovery: search_openalex_authors, rank_researchers_by_activity
    - Protein profiles: search_uniprot_proteins, get_uniprot_protein_profile
    - Pathways: search_reactome_pathways
    - Protein interactions: get_string_interactions
    - Variant effect predictions (SIFT, PolyPhen, AlphaMissense): annotate_variants_vep (Ensembl VEP)
    - Aggregated variant annotations (ClinVar, CADD, dbSNP, gnomAD, COSMIC): get_variant_annotations (MyVariant.info)
    - Clinical variant interpretations in oncology: search_civic_variants, search_civic_genes (CIViC)
    - Protein structure predictions: get_alphafold_structure (AlphaFold — pLDDT confidence, PDB/CIF downloads)
    - GWAS trait-variant associations: search_gwas_associations (GWAS Catalog — p-values, odds ratios, mapped genes)
    - Drug-gene interactions & druggability: search_drug_gene_interactions (DGIdb — approved/experimental drugs)
    - Tissue-level gene expression: get_gene_tissue_expression (GTEx v8 — median TPM across 54 tissues)
    - Experimental protein structures: search_protein_structures (RCSB PDB — resolution, method, ligands)
    - Cancer mutation profiles: get_cancer_mutation_profile (cBioPortal — TCGA pan-cancer mutation frequencies)
    - Drug bioactivity & selectivity: get_chembl_bioactivities (ChEMBL API — IC50/Ki/Kd by target, kinase selectivity profiling. Prefer over BigQuery ebi_chembl for bioactivity lookups)
    - Chemical compound data: get_pubchem_compound (PubChem — molecular properties, SMILES, drug-likeness)
    - Post-marketing adverse events: search_fda_adverse_events (openFDA FAERS — reaction counts, seriousness, indications)
"""


BQ_EXECUTOR_POLICY = """- BigQuery-first policy: For any structured data lookup, prefer `list_bigquery_tables` \
and `run_bigquery_select_query` over non-BQ tools. \
Available datasets: open_targets_platform (targets, diseases, drugs, evidence), ebi_chembl (bioactivity), \
gnomAD (variant frequencies), human_genome_variants, human_variant_annotation (ClinVar), \
immune_epitope_db (IEDB), \
nlm_rxnorm (drug nomenclature), fda_drug (drug labels, NDC, enforcement), \
umiami_lincs (perturbation signatures), ebi_surechembl (patents).
CRITICAL SQL syntax: Always wrap table references in backticks in your SQL queries. \
Short names are auto-expanded: `open_targets_platform.target` → `bigquery-public-data.open_targets_platform.target`. \
Example: SELECT id, approvedSymbol FROM `open_targets_platform.target` WHERE approvedSymbol = 'BRCA1'.
Before writing queries:
  1. Call `list_bigquery_tables(dataset="<dataset_name>")` to see all available tables.
  2. Call `list_bigquery_tables(dataset="<dataset_name>", table="<table_name>")` to get the full column schema (names, types, descriptions).
  NEVER guess column names — always inspect the schema first. \
  Column names are often singular (e.g. "target" not "targets") \
  and use IDs rather than human-readable names (e.g. targetId is an Ensembl ID like "ENSG00000012048", \
  diseaseId is an EFO ID like "EFO_0001075"). Look up IDs from reference tables first.
Fall back to non-BQ tools for: literature search (search_pubmed, search_openalex_works), \
ClinicalTrials.gov, UniProt, Reactome pathways, STRING interactions, \
variant effect predictions (annotate_variants_vep for SIFT/PolyPhen/AlphaMissense), \
aggregated variant annotations (get_variant_annotations for ClinVar/CADD/dbSNP/gnomAD/COSMIC), \
clinical variant interpretations (search_civic_variants, search_civic_genes for CIViC), \
protein structure predictions (get_alphafold_structure for pLDDT), \
GWAS associations (search_gwas_associations), drug-gene interactions (search_drug_gene_interactions), \
tissue expression (get_gene_tissue_expression), experimental structures (search_protein_structures), \
cancer mutations (get_cancer_mutation_profile), \
drug bioactivity and selectivity (get_chembl_bioactivities — prefer over BigQuery ebi_chembl), \
chemical compound data (get_pubchem_compound), \
and post-marketing adverse events (search_fda_adverse_events for openFDA FAERS — \
prefer this over BigQuery fda_drug for adverse event reports)."""


def _format_tool_catalog(tool_hints: list[str]) -> str:
    lines = []
    for name in tool_hints[:80]:
        desc = TOOL_DESCRIPTIONS.get(name)
        lines.append(f"- {name} — {desc}" if desc else f"- {name}")
    return "\n".join(lines) or "- No tools available."


def _build_step_executor_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    tool_catalog = _format_tool_catalog(tool_hints)
    if prefer_bigquery:
        bq_policy = BQ_EXECUTOR_POLICY
    else:
        bq_policy = "- BigQuery-first policy is disabled for this run."

    return (
        STEP_EXECUTOR_INSTRUCTION_TEMPLATE
        .replace("__TOOL_CATALOG__", tool_catalog)
        .replace("__BQ_POLICY__", bq_policy)
    )


def _build_planner_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    tool_catalog = _format_tool_catalog(tool_hints)
    if prefer_bigquery:
        bq_policy = (
            "- BigQuery-first policy:\n"
            f"{BQ_DATASET_CATALOG}"
            "\n- tool_hint for BigQuery steps: use the specific dataset name (e.g. open_targets_platform,"
            " gnomad, ebi_chembl)"
            " rather than run_bigquery_select_query, so the plan clearly shows which source is being accessed.\n"
            "- For variant pathogenicity predictions, use annotate_variants_vep (Ensembl VEP — SIFT, PolyPhen, AlphaMissense).\n"
            "- For aggregated variant annotations (ClinVar, CADD, dbSNP, gnomAD, COSMIC), use get_variant_annotations (MyVariant.info).\n"
            "- For clinical variant interpretations in oncology, use search_civic_variants or search_civic_genes (CIViC).\n"
            "- For protein structure predictions and confidence scores, use get_alphafold_structure (AlphaFold API).\n"
            "- For GWAS trait-variant associations and genetic evidence, use search_gwas_associations (GWAS Catalog).\n"
            "- For druggability assessment and known drug-gene interactions, use search_drug_gene_interactions (DGIdb).\n"
            "- For tissue-level gene expression and target safety, use get_gene_tissue_expression (GTEx).\n"
            "- For experimental protein structures (X-ray, cryo-EM), use search_protein_structures (RCSB PDB).\n"
            "- For cancer mutation frequencies across tumor types, use get_cancer_mutation_profile (cBioPortal).\n"
            "- For drug bioactivity, IC50/Ki/Kd values, and kinase selectivity profiling, use get_chembl_bioactivities (ChEMBL API — prefer over BigQuery ebi_chembl).\n"
            "- For chemical compound properties, SMILES, drug-likeness, use get_pubchem_compound (PubChem).\n"
            "- For post-marketing adverse events (FAERS), use search_fda_adverse_events (openFDA) — not BigQuery fda_drug."
        )
    else:
        bq_policy = "- BigQuery-first policy is disabled for this run."

    return (
        PLANNER_INSTRUCTION_TEMPLATE
        .replace("__TOOL_CATALOG__", tool_catalog)
        .replace("__BQ_POLICY__", bq_policy)
    )


def create_mcp_toolset(tool_filter: list[str] | None = None) -> McpToolset | None:
    """Build an MCP toolset for the native evidence-executor agent."""
    if tool_filter is not None and len(tool_filter) == 0:
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
    require_plan_approval: bool = False,
) -> tuple[SequentialAgent, McpToolset | None]:
    """Create an ADK-native workflow graph and return (root_agent, mcp_toolset).

    Args:
        require_plan_approval: When True, the workflow pauses after plan
            generation and waits for the user to ``approve`` or
            ``revise: <feedback>`` before executing the plan.
    """
    runtime_model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    use_bigquery_priority = DEFAULT_PREFER_BIGQUERY if prefer_bigquery is None else bool(prefer_bigquery)

    mcp_toolset = create_mcp_toolset(tool_filter=tool_filter)
    executor_tools = [mcp_toolset] if mcp_toolset is not None else []
    base_tool_hints = _dedupe_str_list(tool_filter if tool_filter else KNOWN_MCP_TOOLS, limit=120)
    if use_bigquery_priority:
        base_hint_set = set(base_tool_hints)
        prioritized_hints = [name for name in BQ_PRIORITY_TOOLS if name in base_hint_set]
        prioritized_set = set(prioritized_hints)
        prioritized_hints.extend([name for name in base_tool_hints if name not in prioritized_set])
        executor_tool_hints = prioritized_hints
    else:
        executor_tool_hints = base_tool_hints

    hitl_agent_gate = _hitl_skip_agent if require_plan_approval else None

    planner = LlmAgent(
        name="planner",
        model=runtime_model,
        instruction=_build_planner_instruction(
            executor_tool_hints,
            prefer_bigquery=use_bigquery_priority,
        ),
        tools=[],
        disallow_transfer_to_parent=True,
        before_model_callback=_make_planner_before_model_callback(
            require_approval=require_plan_approval,
        ),
        after_model_callback=_make_planner_after_model_callback(
            require_approval=require_plan_approval,
        ),
    )
    step_executor = LlmAgent(
        name="step_executor",
        model=runtime_model,
        instruction=_build_step_executor_instruction(
            executor_tool_hints,
            prefer_bigquery=use_bigquery_priority,
        ),
        tools=executor_tools,
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
        model=runtime_model,
        instruction=SYNTHESIZER_INSTRUCTION,
        tools=[],
        before_agent_callback=hitl_agent_gate,
        before_model_callback=_synth_before_model_callback,
        after_model_callback=_synth_after_model_callback,
        on_model_error_callback=_on_model_error,
    )

    root = SequentialAgent(
        name="co_scientist_workflow",
        description="ADK-native biomedical workflow: planner, ReAct executor loop, synthesis.",
        sub_agents=[planner, react_loop, report_synthesizer],
    )
    return root, mcp_toolset


__all__ = [
    "create_mcp_toolset",
    "create_workflow_agent",
]
