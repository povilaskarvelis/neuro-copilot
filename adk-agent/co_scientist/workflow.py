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
from google import genai as _genai_module
from google.genai import types
from mcp.client.stdio import StdioServerParameters

from . import tool_registry

logger = logging.getLogger(__name__)


MCP_SERVER_DIR = Path(__file__).resolve().parents[2] / "research-mcp"
DEFAULT_MODEL = os.getenv("ADK_NATIVE_MODEL", "gemini-2.5-flash")
PLANNER_MODEL = os.getenv("ADK_PLANNER_MODEL", "gemini-2.5-flash")
SYNTHESIZER_MODEL = os.getenv("ADK_SYNTHESIZER_MODEL", "gemini-2.5-pro")
ROUTER_MODEL = os.getenv("ADK_ROUTER_MODEL", "gemini-2.5-flash")
THINKING_CONFIG_V3 = types.ThinkingConfig(thinking_level="HIGH")
THINKING_CONFIG_V2 = types.ThinkingConfig(include_thoughts=True, thinking_budget=8192)


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
STATE_EXECUTOR_PREV_STEP_STATUS = "temp:co_scientist_executor_prev_step_status"
STATE_EXECUTOR_REASONING_TRACE = "temp:co_scientist_executor_reasoning_trace"
STATE_EXECUTOR_TOOL_LOG = "temp:co_scientist_executor_tool_log"
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
    "search_fda_adverse_events",
    "search_pubmed",
    "search_pubmed_advanced",
    "get_pubmed_abstract",
    "get_paper_fulltext",
    "search_geo_datasets",
    "get_geo_dataset",
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
    "search_protein_structures",
    "search_drug_gene_interactions",
    "annotate_variants_vep",
    "search_civic_variants",
    "search_civic_genes",
    "get_variant_annotations",
    "search_gwas_associations",
    "get_gene_tissue_expression",
    "get_human_protein_atlas_gene",
    "get_depmap_gene_dependency",
    "get_biogrid_orcs_gene_summary",
    "get_gdsc_drug_sensitivity",
    "get_prism_repurposing_response",
    "get_pharmacodb_compound_response",
    "search_cellxgene_datasets",
    "search_pathway_commons_top_pathways",
    "get_guidetopharmacology_target",
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
]


PLANNER_INSTRUCTION_TEMPLATE = """
You are the internal planner for biomedical investigation.

Available MCP tools:
__TOOL_CATALOG__

Tool domains (used to focus the executor on the most relevant tools for each step):
__DOMAIN_CATALOG__

Source precedence rules for overlapping tools:
__ROUTING_POLICY__

Rules:
- Build a concrete execution plan before any evidence collection begins.
- Break the objective into ordered, atomic subtasks.
- Prioritize high-signal subtasks that reduce uncertainty first.
- Choose the number of steps needed for the objective. Avoid unnecessary fragmentation.
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

Available MCP tools:
__TOOL_CATALOG__

Source precedence rules for overlapping tools:
__ROUTING_POLICY__

Rules:
- Focus ONLY on the current step provided in the execution context.
- You MUST call at least one tool before returning a result.
- If a tool call fails or returns insufficient data, try an alternative tool or query
  (e.g. search_pubmed <-> search_openalex_works, or fall back to run_bigquery_select_query).
- If no tool can satisfy the step after trying alternatives, state clearly that the step is BLOCKED and why.
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
- Include specific numbers, scores, measurements, and identifiers rather than vague summaries.
- Note the confidence level (high, moderate, low, or mixed) and number of independent sources.
- If evidence on a finding is contradictory, briefly note the disagreement (details go in Conflicting Evidence).
- Aim for substantive prose paragraphs — avoid sparse bullet-only lists or large tables when prose conveys the same information more clearly.

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
- Do not treat all sources as equal. When claims disagree, prefer the interpretation backed by higher-weighted sources and stronger claim support, but still surface the disagreement explicitly.
- If `mixed_evidence_claims` are present, address them in Conflicting & Uncertain Evidence instead of silently choosing one side.
- Be specific and thorough — avoid terse output.
- NEVER organize findings by step execution order. Group by theme/topic.
- Use ONLY human-readable database/source names (e.g. "PubMed", "ClinicalTrials.gov"). NEVER mention tool names (like run_bigquery_select_query, search_clinical_trials, etc.).
- When citing database counts (e.g. clinical trials, PubMed results): use the total reported by the source when available (e.g. "X of Y total"). If the source says "total not provided" or "more may exist" or "X returned (registry total unknown)", do NOT state "a total of X" or "X total studies" — instead say "at least X" or "X studies (sample; full registry count not determined)".
- Include specific identifiers inline when available (PMID, DOI, NCT numbers).
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
- Maintain the report's citation style when referencing sources.
- When restructuring, preserve all evidence and citations — don't drop content unless asked.
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


def _extract_function_call_names(llm_response: LlmResponse) -> list[str]:
    """Extract the names of all function calls from a model response."""
    return [fc["name"] for fc in _extract_function_calls(llm_response)]


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
        return f"Querying gene-phenotype associations for {query}" if query else "Searching Monarch Initiative"
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
        return f"{source}: error — {str(err)[:100]}"

    # MCP tools put text in content[0].text; the format is typically:
    #   "Summary:\n{actual summary}\n\nKey Fields:\n..."
    # Extract the real summary from the second line, or the first non-label line.
    mcp_text = _extract_mcp_text(response)
    if mcp_text:
        summary_line = _extract_mcp_summary_line(mcp_text)
        if summary_line:
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
            max_chars = 380 if name in long_output_tools else 220
            return _truncate_summary(summary_line, max_chars=max_chars)

    # Try structured dict keys (for non-MCP tools / direct function tools)
    summary = _extract_result_summary(name, response)
    if summary:
        return summary
    return f"{source} returned results"


_MCP_SECTION_LABELS = {"summary:", "key fields:", "sources:", "limitations:", "notes:"}


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
        if isinstance(studies, list):
            return f"found {len(studies)} clinical trials"
        count = response.get("totalCount") or response.get("count") or response.get("total")
        if count is not None:
            return f"found {count} clinical trials"

    # PubMed / literature search
    if name in ("search_pubmed", "search_pubmed_advanced", "search_openalex_works",
                "search_europe_pmc_literature"):
        articles = response.get("results") or response.get("articles") or response.get("papers")
        if isinstance(articles, list):
            return f"found {len(articles)} articles"
        count = response.get("count") or response.get("total") or response.get("totalResults")
        if count is not None:
            return f"found {count} articles"

    # GWAS
    if name == "search_gwas_associations":
        assocs = response.get("associations") or response.get("results")
        if isinstance(assocs, list):
            return f"found {len(assocs)} associations"

    # HPO / ontology
    if name == "search_hpo_terms":
        terms = response.get("terms") or response.get("results")
        if isinstance(terms, list):
            return f"found {len(terms)} phenotype terms"

    # Drug interactions
    if name == "search_drug_gene_interactions":
        interactions = response.get("matchedTerms") or response.get("interactions") or response.get("results")
        if isinstance(interactions, list):
            return f"found {len(interactions)} drug-gene interactions"

    # FDA adverse events
    if name == "search_fda_adverse_events":
        events = response.get("results") or response.get("events")
        if isinstance(events, list):
            return f"found {len(events)} adverse event reports"

    # Generic: look for common list/count patterns in any response
    for key in ("results", "data", "items", "hits", "records", "entries"):
        val = response.get(key)
        if isinstance(val, list) and len(val) > 0:
            return f"found {len(val)} {key}"
    for key in ("count", "total", "totalCount", "numFound", "total_rows"):
        val = response.get(key)
        if isinstance(val, (int, float)) and val > 0:
            return f"found {int(val)} results"

    # If response has a meaningful top-level field (symbol, name, id)
    for key in ("symbol", "name", "label", "title"):
        val = response.get(key)
        if isinstance(val, str) and val.strip():
            return f"found: {val.strip()[:80]}"

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
                msg = str(err)[:120]
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


def _set_tool_log(callback_context: CallbackContext, log: list[dict[str, str]]) -> None:
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
    callback_context.state[STATE_EXECUTOR_RENDERED] = ""
    callback_context.state[STATE_EXECUTOR_ACTIVE_STEP_ID] = ""
    callback_context.state[STATE_EXECUTOR_PREV_STEP_STATUS] = ""
    callback_context.state[STATE_EXECUTOR_REASONING_TRACE] = ""
    callback_context.state[STATE_EXECUTOR_TOOL_LOG] = "[]"
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


# ---------------------------------------------------------------------------
# Post-hoc structure extraction from prose (primary extraction path)
# ---------------------------------------------------------------------------

_EXTRACTION_CLIENT: _genai_module.Client | None = None
EXTRACTION_MODEL = os.getenv("ADK_EXTRACTION_MODEL", "gemini-2.0-flash")
EXTRACTION_FALLBACK_MODEL = os.getenv("ADK_EXTRACTION_FALLBACK_MODEL", "gemini-2.5-flash-lite")

_EXTRACTION_PROMPT_TEMPLATE = """\
Extract structured data from the following research step output.
Focus on identifying substantive scientific claims as structured_observations with proper
subject/predicate/object triples, and collecting all evidence identifiers.

Step {step_id} output:
---
{prose}
---

Return a JSON object with these fields:
{{
  "status": "completed" or "blocked",
  "result_summary": "<concise but thorough findings summary preserving key details, scores, and identifiers>",
  "evidence_ids": ["PMID:...", "NCT:...", "DOI:...", "UniProt:...", "rs...", "CHEMBL..."],
  "tools_called": ["tool_name_1"],
  "data_sources_queried": ["Open Targets Platform", "ClinicalTrials.gov", ...],
  "open_gaps": ["..."],
  "suggested_next_searches": ["..."],
  "structured_observations": [
    {{
      "observation_type": "<claim family: e.g. phenotype_association, drug_response, interaction, pathway_involvement>",
      "subject": {{ "type": "<entity type>", "label": "<name>", "id": "<optional canonical ID>" }},
      "predicate": "<atomic predicate: e.g. associated_with, sensitive_in, inhibits, involved_in>",
      "object": {{ "type": "<entity type>", "label": "<name>", "id": "<optional canonical ID>" }},
      "supporting_ids": ["PMID:...", "CHEMBL..."],
      "source_tool": "<tool name that produced this finding>",
      "confidence": "low" | "medium" | "high"
    }}
  ]
}}

Rules:
- Extract ALL substantive scientific claims as structured_observations (usually 1-12 per step).
- Each observation must be grounded in the text — do not invent unsupported claims.
- Include every evidence identifier mentioned in the text in evidence_ids.
- The result_summary should capture the full richness of findings without padding.
- If the text indicates the step failed or was blocked, set status to "blocked"."""


def _get_extraction_client() -> _genai_module.Client | None:
    global _EXTRACTION_CLIENT
    if _EXTRACTION_CLIENT is not None:
        return _EXTRACTION_CLIENT
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        _EXTRACTION_CLIENT = _genai_module.Client(api_key=api_key)
    except Exception:  # noqa: BLE001
        logger.warning("[extract] failed to create genai client")
        return None
    return _EXTRACTION_CLIENT


def _call_extraction_model(client: _genai_module.Client, model: str, prompt: str, step_id: str) -> dict[str, Any] | None:
    """Call a single extraction model and return parsed dict or None."""
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )
    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        return None
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        return None
    parsed["step_id"] = step_id
    parsed["schema"] = STEP_RESULT_SCHEMA
    if not parsed.get("step_progress_note"):
        summary = str(parsed.get("result_summary", "") or "")
        parsed["step_progress_note"] = summary[:200] if summary else "Step completed."
    return parsed


def _extract_structure_from_prose(prose: str, step_id: str) -> dict[str, Any] | None:
    """Extract structured step data from prose. Tries primary model, falls back on failure."""
    client = _get_extraction_client()
    if client is None:
        return None
    prompt = _EXTRACTION_PROMPT_TEMPLATE.format(step_id=step_id, prose=prose[:6000])
    models_to_try = [EXTRACTION_MODEL]
    if EXTRACTION_FALLBACK_MODEL and EXTRACTION_FALLBACK_MODEL != EXTRACTION_MODEL:
        models_to_try.append(EXTRACTION_FALLBACK_MODEL)
    for model in models_to_try:
        try:
            parsed = _call_extraction_model(client, model, prompt, step_id)
            if parsed is not None:
                logger.info(
                    "[extract] %s extracted for %s: %d observations, %d evidence_ids",
                    model, step_id,
                    len(parsed.get("structured_observations") or []),
                    len(parsed.get("evidence_ids") or []),
                )
                return parsed
            logger.warning("[extract] %s returned empty for %s", model, step_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[extract] %s failed for %s: %s", model, step_id, exc)
    return None


_EVIDENCE_ID_PATTERNS = re.compile(
    r"(?:PMID:\d+|DOI:10\.\S+|NCT\d{8,}|PMC\d+|OpenAlex:W\d+"
    r"|UniProt:[A-Z0-9]{4,}|PubChem:\d+|PDB:[A-Z0-9]{4}"
    r"|rs\d{4,}|CHEMBL\d+|Reactome:R-HSA-\d+|GCST\d+)"
)


def _extract_evidence_ids_from_text(text: str) -> list[str]:
    """Regex-extract canonical evidence identifiers from prose text."""
    if not text:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for match in _EVIDENCE_ID_PATTERNS.finditer(text):
        eid = match.group(0)
        if eid not in seen:
            seen.add(eid)
            result.append(eid)
    return result[:30]


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
        raw_domains = step.get("domains")
        if isinstance(raw_domains, list):
            domains = [str(d).strip().lower() for d in raw_domains if str(d).strip()]
            domains = [d for d in domains if d in tool_registry.TOOL_DOMAINS]
        else:
            domains = []
        steps.append(
            {
                "id": canonical_id,
                "goal": _as_nonempty_str(step.get("goal"), f"steps[{idx - 1}].goal"),
                "tool_hint": _as_nonempty_str(step.get("tool_hint"), f"steps[{idx - 1}].tool_hint"),
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
        if isinstance(value, list):
            value_text = ", ".join(str(item) for item in value[:6])
        elif isinstance(value, dict):
            nested = ", ".join(f"{nested_key}={nested_value}" for nested_key, nested_value in list(value.items())[:6])
            value_text = nested
        else:
            value_text = str(value)
        parts.append(f"{key}: {value_text}")
    return "; ".join(parts[:8])


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
        "summary_text": str(summary_text or "").strip(),
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
    "get_chembl_bioactivities": 0.92,
    "get_human_protein_atlas_gene": 0.9,
    "search_reactome_pathways": 0.88,
    "search_civic_variants": 0.88,
    "search_civic_genes": 0.88,
    "get_pharmacodb_compound_response": 0.87,
    "get_gene_tissue_expression": 0.84,
    "get_biogrid_orcs_gene_summary": 0.84,
    "get_alliance_genome_gene_profile": 0.82,
    "query_monarch_associations": 0.8,
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


def _humanize_claim_predicate(predicate: str) -> str:
    mapping = {
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
        "supported_by": "is supported by",
    }
    cleaned = str(predicate or "").strip()
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


_STEP_PROSE_NOISE_PATTERNS = [
    re.compile(r"^#+\s*S\d+\s*[·\-].*$", re.MULTILINE),
    re.compile(r"^\*\*Goal:\*\*.*$", re.MULTILINE),
    re.compile(r"^\*\*(?:Key |Detailed )?Findings\*\*\s*$", re.MULTILINE),
    re.compile(r"^\*\*(?:Evidence|Evidence IDs|Structured Claims|Claims|Open Gaps|Tool Activity|Sources queried|ReAct Trace|Tool Trace)\*\*.*$", re.MULTILINE),
    re.compile(r"^_Progress:.*_\s*$", re.MULTILINE),
    re.compile(r"^\s*(?:COMPLETED|BLOCKED)\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^-{3,}\s*$", re.MULTILINE),
    re.compile(r"^This step is (?:COMPLETED|BLOCKED)\.?\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^ACT:\s+Called\s+.+$", re.MULTILINE),
    re.compile(r"^OBSERVE:\s+.+$", re.MULTILINE),
]


def _clean_step_prose(text: str) -> str:
    """Remove redundant headings, status lines, and progress lines from model prose."""
    cleaned = text
    for pattern in _STEP_PROSE_NOISE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


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


def _expand_reference_only_body_lines(text: str, lit_ids: list[str]) -> str:
    if not text or not lit_ids:
        return text

    refs_split = re.split(r"(?m)^#{2,3} References\b", text, maxsplit=1)
    body = refs_split[0]
    refs_tail = ("\n## References" + refs_split[1]) if len(refs_split) > 1 else ""
    citation_cache: dict[int, str] = {}
    citation_only_re = re.compile(r"^(\s*(?:[-*]\s+)?)\[(\d+)\](?:\(#ref-\2\))?\s*$")

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
        citation_text = _citation_text(int(match.group(2)))
        if not citation_text:
            expanded_lines.append(raw_line)
            continue
        expanded_lines.append(f"{match.group(1) or '- '}{citation_text}")

    return "\n".join(expanded_lines) + refs_tail


# ---------------------------------------------------------------------------
# Evidence & Methodology helpers (used by _render_final_synthesis_markdown)
# ---------------------------------------------------------------------------


def _build_methodology_overview(task_state: dict[str, Any], completed: int, total: int, failed: int) -> str:
    """Build the one-paragraph overview for the Evidence and Methodology section."""
    step_summaries: list[str] = []
    for step in task_state.get("steps", []):
        goal = str(step.get("goal", "")).strip()
        status = str(step.get("status", "")).strip()
        source = _format_step_source_display(step)
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

    evidence_summary = _summarize_evidence_store(task_state.get("evidence_store", {}))
    if evidence_summary.get("evidence_count") or evidence_summary.get("claim_count"):
        source_count = len(evidence_summary.get("sources", []) or [])
        overview_parts.append(
            " The workflow normalized "
            f"{evidence_summary.get('entity_count', 0)} entities, "
            f"{evidence_summary.get('claim_count', 0)} claims, and "
            f"{evidence_summary.get('evidence_count', 0)} evidence records"
            + (f" across {source_count} source(s)." if source_count else ".")
        )
    claim_summary = _build_claim_synthesis_summary(
        task_state.get("evidence_store", {}),
        str(task_state.get("objective", "")).strip(),
    )
    if claim_summary.get("mixed_evidence_count"):
        overview_parts.append(
            f" Claim adjudication flagged {claim_summary.get('mixed_evidence_count', 0)} mixed-evidence finding(s) that require cautious interpretation."
        )
    top_claims = claim_summary.get("top_supported_claims", []) or []
    if top_claims:
        lead_claim = top_claims[0]
        overview_parts.append(
            " The strongest adjudicated claim was "
            f"'{lead_claim.get('statement', '')}' "
            f"({lead_claim.get('source_count', 0)} source(s))."
        )

    return "".join(overview_parts)


def _render_step_subsection(step: dict[str, Any]) -> list[str]:
    """Render a ### subsection for one step in the Evidence and Methodology section."""
    step_id = str(step.get("id", "")).strip()
    goal = str(step.get("goal", "")).strip()
    status = str(step.get("status", "")).strip()
    summary = str(step.get("result_summary", "")).strip()
    source = _format_step_source_display(step)
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
        source_label = "Sources" if "; " in source else "Source"
        lines.append(f"**{source_label}:** {source}")
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


STEP_SECTION_HEADING_RE = re.compile(
    r"^(?:###\s*)?(Step\s+\d+:\s+.+?)(?:\s*[—-]\s*(COMPLETED|FAILED|PENDING|BLOCKED))?\s*$",
    flags=re.IGNORECASE,
)
STEP_FIELD_LABEL_PATTERN = r"(?:Data Source|Data source|Key Findings|Key findings|Significance|Limitations|Limiatations)"


def _clean_step_field_text(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"^\*+\s*", "", cleaned)
    cleaned = re.sub(r"^\-\s*", "", cleaned)
    cleaned = re.sub(r"\s+\*\s+", " ", cleaned)
    cleaned = re.sub(r"\s*\*+$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not re.search(r"[A-Za-z0-9]", cleaned):
        return ""
    return cleaned


def _normalize_evidence_step_block(lines: list[str]) -> str:
    if not lines:
        return ""
    heading_match = STEP_SECTION_HEADING_RE.match(str(lines[0] or "").strip())
    if not heading_match:
        return "\n".join(line.rstrip() for line in lines).strip()

    heading = heading_match.group(1).strip()
    status = str(heading_match.group(2) or "").strip().upper()
    rendered: list[str] = [f"### {heading}{f' — {status}' if status else ''}", ""]

    field_map = {
        "data source": "data_source",
        "key findings": "key_findings",
        "significance": "significance",
        "limitations": "limitations",
        "limiatations": "limitations",
    }
    fields: dict[str, list[str]] = {
        "data_source": [],
        "key_findings": [],
        "significance": [],
        "limitations": [],
    }
    preamble: list[str] = []
    active_field = ""

    body_text = "\n".join(str(line or "").rstrip() for line in lines[1:])
    body_text = re.sub(
        rf"\*+\s*({STEP_FIELD_LABEL_PATTERN})\s*:\s*\*+",
        r"\1:",
        body_text,
        flags=re.IGNORECASE,
    )
    body_text = re.sub(
        rf"\*+\s*({STEP_FIELD_LABEL_PATTERN})\s*:",
        r"\1:",
        body_text,
        flags=re.IGNORECASE,
    )
    body_text = re.sub(
        rf"({STEP_FIELD_LABEL_PATTERN})\s*:\s*\*+",
        r"\1:",
        body_text,
        flags=re.IGNORECASE,
    )
    body_text = re.sub(
        rf"(?<!\n)(?=(?:{STEP_FIELD_LABEL_PATTERN}):)",
        "\n",
        body_text,
        flags=re.IGNORECASE,
    )

    for raw_line in body_text.splitlines():
        stripped = str(raw_line or "").strip()
        if not stripped:
            continue
        field_match = re.match(r"^([A-Za-z ]+):\s*(.*)$", stripped)
        if field_match:
            label_key = field_map.get(field_match.group(1).strip().lower())
            if label_key:
                active_field = label_key
                value = _clean_step_field_text(field_match.group(2))
                if value:
                    fields[label_key].append(value)
                continue
        if active_field:
            cleaned_value = _clean_step_field_text(stripped)
            if cleaned_value:
                fields[active_field].append(cleaned_value)
        else:
            cleaned_value = _clean_step_field_text(stripped)
            if cleaned_value:
                preamble.append(cleaned_value)

    if preamble:
        rendered.append(" ".join(preamble))
        rendered.append("")

    data_source = " ".join(fields["data_source"]).strip()
    if data_source:
        rendered.append(f"**Data Source:** {data_source}")
        rendered.append("")

    key_findings = [item for item in fields["key_findings"] if item]
    if key_findings:
        rendered.append("**Key Findings:**")
        for item in key_findings:
            rendered.append(f"- {item}")
        rendered.append("")

    significance = " ".join(fields["significance"]).strip()
    if significance:
        rendered.append(f"**Significance:** {significance}")
        rendered.append("")

    limitations = " ".join(fields["limitations"]).strip()
    if limitations:
        rendered.append(f"**Limitations:** {limitations}")
        rendered.append("")

    return "\n".join(rendered).strip()


def _normalize_evidence_section_markdown(markdown: str) -> str:
    text = str(markdown or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    first_step_idx = -1
    for idx, raw_line in enumerate(lines):
        if STEP_SECTION_HEADING_RE.match(str(raw_line or "").strip()):
            first_step_idx = idx
            break
    if first_step_idx < 0:
        return text

    preamble = "\n".join(line.rstrip() for line in lines[:first_step_idx]).strip()
    blocks: list[list[str]] = []
    current: list[str] = []
    for raw_line in lines[first_step_idx:]:
        stripped = str(raw_line or "").strip()
        if STEP_SECTION_HEADING_RE.match(stripped):
            if current:
                blocks.append(current)
            current = [raw_line]
        else:
            current.append(raw_line)
    if current:
        blocks.append(current)

    rendered: list[str] = []
    if preamble:
        rendered.append(preamble)
        rendered.append("")
    for block in blocks:
        normalized = _normalize_evidence_step_block(block)
        if normalized:
            rendered.append(normalized)
            rendered.append("")
    return "\n".join(rendered).strip()


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


def _format_markdown_table_cell(value: Any) -> str:
    if value is None:
        return "-"
    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return "-"
    return text.replace("|", "\\|")


def _render_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not headers or not rows:
        return ""
    cleaned_headers = [_format_markdown_table_cell(header) for header in headers]
    lines = [
        "| " + " | ".join(cleaned_headers) + " |",
        "| " + " | ".join("---" for _ in cleaned_headers) + " |",
    ]
    for row in rows:
        cells = [_format_markdown_table_cell(value) for value in row[: len(cleaned_headers)]]
        if len(cells) < len(cleaned_headers):
            cells.extend("-" for _ in range(len(cleaned_headers) - len(cells)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_top_supported_claims_markdown(claim_summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for claim in list(claim_summary.get("top_supported_claims", []) or [])[:5]:
        source_text = ", ".join(list(claim.get("primary_sources", []) or [])[:3]) or "not specified"
        suffix = "; mixed evidence" if claim.get("mixed_evidence") else ""
        lines.append(f"- {claim.get('statement', '')} (sources: {source_text}{suffix})")
    return "\n".join(lines)


def _build_mixed_evidence_claims_table(claim_summary: dict[str, Any]) -> str:
    rows: list[list[Any]] = []
    for conflict in list(claim_summary.get("mixed_evidence_claims", []) or [])[:5]:
        focus = " and ".join(
            part
            for part in [
                str(conflict.get("subject", "")).strip(),
                str(conflict.get("object", "")).strip(),
            ]
            if part
        ) or "Claim"
        source_fragments: list[str] = []
        for claim in list(conflict.get("claims", []) or [])[:2]:
            sources = ", ".join(list(claim.get("primary_sources", []) or [])[:2])
            if sources:
                source_fragments.append(f"{claim.get('predicate', '')}: {sources}")
        rows.append(
            [
                focus,
                str(conflict.get("assessment", "")).replace("_", " "),
                conflict.get("preferred_interpretation", "") or "No clear lean",
                "; ".join(source_fragments) or "-",
            ]
        )
    return _render_markdown_table(
        ["Claim focus", "Assessment", "Current lean", "Leading sources"],
        rows,
    )


# ---------------------------------------------------------------------------
# Thematic claim clustering + adaptive finding rendering
# ---------------------------------------------------------------------------

NARRATIVE_PREDICATES = {
    "has_function", "participates_in", "causal_gene_for", "depends_on",
    "activates", "inhibits",
}
TABULAR_PREDICATES = {
    "associated_with", "interacts_with", "has_phenotype", "cross_referenced_in",
    "has_ortholog", "has_model", "screen_hit_in",
}

_ENTITY_TYPE_LABELS: dict[str, str] = {
    "gene": "Gene",
    "compound": "Compound",
    "disease": "Disease",
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
            pred = predicates.pop().replace("_", " ")
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


def _format_qualifier_context(qualifiers: dict[str, Any]) -> str:
    """Render non-empty qualifiers as a compact context string."""
    parts: list[str] = []
    for key in ["tissue", "disease", "species", "assay", "phase", "evidence"]:
        val = str(qualifiers.get(key, "") or "").strip()
        if val:
            parts.append(f"{key}: {val}")
    return "; ".join(parts)


def _render_finding_cluster(cluster: dict[str, Any]) -> list[str]:
    """Render a single thematic finding cluster with adaptive presentation."""
    variant = cluster.get("variant", "verdict")
    title = cluster.get("title", "Finding")
    confidence = cluster.get("confidence", "moderate")
    claims = cluster.get("claims", [])
    lines: list[str] = [f"### {title}", ""]

    if not claims:
        step_summary = cluster.get("step_summary", "")
        source = cluster.get("step_source", "")
        if step_summary:
            source_suffix = f" (source: {source})" if source else ""
            lines.append(f"{step_summary}{source_suffix}")
            lines.append("")
        return lines

    source_count = len({s for c in claims for s in (c.get("primary_sources") or [])})
    evidence_count = sum(int(c.get("evidence_count", 0) or 0) for c in claims)
    evidence_meta: list[str] = []
    if source_count > 0:
        evidence_meta.append(f"{source_count} source{'s' if source_count != 1 else ''}")
    if evidence_count > 0:
        evidence_meta.append(f"{evidence_count} evidence record{'s' if evidence_count != 1 else ''}")
    confidence_labels = {"high": "Strong", "moderate": "Moderate", "low": "Limited", "mixed": "Mixed"}
    conf_label = confidence_labels.get(confidence, confidence.title())
    meta_suffix = f" ({', '.join(evidence_meta)})" if evidence_meta else ""
    lines.append(f"**Evidence strength:** {conf_label}{meta_suffix}")
    lines.append("")

    if variant == "tabular":
        lines.extend(_render_finding_tabular(claims))
    elif variant == "narrative":
        lines.extend(_render_finding_narrative(claims))
    else:
        lines.extend(_render_finding_verdict(claims))

    lines.append("")
    return lines


def _render_finding_verdict(claims: list[dict[str, Any]]) -> list[str]:
    supporting = [c for c in claims if not c.get("mixed_evidence")]
    conflicting = [c for c in claims if c.get("mixed_evidence")]
    lines: list[str] = []
    if supporting:
        lines.append("**Supporting evidence:**")
        for claim in supporting:
            lines.append(_format_claim_evidence_line(claim))
    if conflicting:
        lines.append("")
        lines.append("**Conflicting evidence:**")
        for claim in conflicting:
            source_text = ", ".join(list(claim.get("primary_sources", []) or [])[:3]) or "not specified"
            lines.append(f"- {claim.get('statement', '')} — {source_text} (mixed evidence)")
    return lines


def _format_claim_evidence_line(claim: dict[str, Any]) -> str:
    """Format a single claim as a rich evidence bullet with provenance."""
    source_text = ", ".join(list(claim.get("primary_sources", []) or [])[:3]) or "not specified"
    ids_text = ", ".join(list(claim.get("supporting_ids", []) or [])[:4])
    id_suffix = f" ({ids_text})" if ids_text else ""
    strength = str(claim.get("support_strength", "")).strip()
    strength_suffix = f" | strength: {strength}" if strength else ""
    qualifier_ctx = _format_qualifier_context(claim.get("qualifiers") or {})
    qualifier_suffix = f" [{qualifier_ctx}]" if qualifier_ctx else ""
    return f"- {claim.get('statement', '')} — {source_text}{id_suffix}{strength_suffix}{qualifier_suffix}"


def _render_finding_tabular(claims: list[dict[str, Any]]) -> list[str]:
    rows: list[list[Any]] = []
    for claim in claims:
        source_text = ", ".join(list(claim.get("primary_sources", []) or [])[:2]) or "-"
        ids_text = ", ".join(list(claim.get("supporting_ids", []) or [])[:3]) or "-"
        strength = str(claim.get("support_strength", "")).strip() or "-"
        mixed = " (mixed)" if claim.get("mixed_evidence") else ""
        qualifier_ctx = _format_qualifier_context(claim.get("qualifiers") or {})
        subject_type = _ENTITY_TYPE_LABELS.get(str(claim.get("subject_type", "")).strip(), "")
        subject_label = claim.get("subject", "-")
        if subject_type and subject_type not in subject_label:
            subject_label = f"{subject_label} ({subject_type})"
        rows.append([
            subject_label,
            claim.get("predicate", "").replace("_", " "),
            claim.get("object", "-"),
            source_text,
            ids_text,
            f"{strength}{mixed}",
            qualifier_ctx or "-",
        ])
    table = _render_markdown_table(
        ["Subject", "Relationship", "Object", "Source", "Identifiers", "Strength", "Context"],
        rows,
    )
    return [table] if table else []


def _render_finding_narrative(claims: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for claim in claims:
        source_text = ", ".join(list(claim.get("primary_sources", []) or [])[:3]) or "not specified"
        ids_text = ", ".join(list(claim.get("supporting_ids", []) or [])[:4])
        id_suffix = f" ({ids_text})" if ids_text else ""
        qualifier_ctx = _format_qualifier_context(claim.get("qualifiers") or {})
        qualifier_suffix = f" [{qualifier_ctx}]" if qualifier_ctx else ""
        lines.append(f"{claim.get('statement', '')} ({source_text}{id_suffix}){qualifier_suffix}.")
    return [" ".join(lines)]


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
            predicate = str(claim.get("predicate", "")).replace("_", " ")
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


def _render_sources_consulted(task_state: dict[str, Any]) -> list[str]:
    """Build Sources Consulted section from step metadata."""
    seen: dict[str, dict[str, Any]] = {}
    for step in task_state.get("steps", []):
        sources = _derive_step_data_sources(step)
        status = str(step.get("status", "")).strip()
        goal = str(step.get("goal", "")).strip()
        for source_name in sources:
            if source_name in seen:
                seen[source_name]["steps"].append(goal)
                if status == "completed":
                    seen[source_name]["completed"] = True
            else:
                seen[source_name] = {
                    "source": source_name,
                    "steps": [goal],
                    "completed": status == "completed",
                }
    if not seen:
        return []
    rows: list[list[Any]] = []
    for entry in seen.values():
        status_text = "Queried" if entry["completed"] else "Attempted"
        contribution = "; ".join(entry["steps"][:2])
        if len(entry["steps"]) > 2:
            contribution += f" (+{len(entry['steps']) - 2} more)"
        rows.append([entry["source"], status_text, contribution])
    lines = ["## Sources Consulted", ""]
    table = _render_markdown_table(["Source", "Status", "Contribution"], rows)
    if table:
        lines.append(table)
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
    inline_ids = _extract_inline_ids_from_text(current_text)
    lit_ids = [eid for eid in inline_ids if _is_literature_id(eid)]
    ref_map = _build_ref_map(lit_ids)
    refs = _build_references_section(lit_ids)
    if refs:
        lines += refs.split("\n")
        lines.append("")

    body_so_far = "\n".join(lines)
    body_so_far = _hyperlink_inline_ids(body_so_far, ref_map)
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
    focused_tools = _prioritize_tools_for_step(
        _resolve_step_tools(step_domains),
        tool_hint,
    )

    focused_catalog = _format_tool_catalog(focused_tools)
    routing_guidance = _format_step_routing_guidance(tool_hint, focused_tools)
    payload = {
        "schema": "react_step_context.v1",
        "objective": task_state.get("objective", ""),
        "current_step": {
            "id": active_step.get("id"),
            "goal": active_step.get("goal"),
            "tool_hint": active_step.get("tool_hint"),
            "domains": step_domains,
            "completion_condition": active_step.get("completion_condition"),
        },
        "remaining_steps_after_this": remaining_count - 1,
        "prior_completed_steps": prior_completed,
    }

    instructions = [
        "Execution context (authoritative; use this instead of inferring from prior prose):",
        _serialize_pretty_json(payload),
    ]

    if step_domains:
        instructions.append(
            f"Focused tools for this step (domains: {', '.join(step_domains)}):\n"
            f"{focused_catalog}\n"
            "Prefer tools from this focused list. You may use other available tools "
            "if the focused set is insufficient, but start here."
        )
    if routing_guidance:
        instructions.append(routing_guidance)

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
    execution_metrics_summary = dict((task_state.get("execution_metrics") or {}).get("summary", {}))
    payload = {
        "schema": "synthesis_context.v1",
        "objective": task_state.get("objective", ""),
        "plan_status": task_state.get("plan_status", "ready"),
        "coverage_status": _compute_coverage_status(task_state),
        "evidence_store_summary": evidence_store_summary,
        "claim_synthesis_summary": claim_synthesis_summary,
        "execution_metrics_summary": execution_metrics_summary,
        "steps": [
            {
                "id": step.get("id"),
                "goal": step.get("goal"),
                "tool_hint": step.get("tool_hint", ""),
                "source": _preferred_step_source_label(step, str(step.get("tool_hint", ""))),
                "status": step.get("status"),
                "reasoning_trace": step.get("reasoning_trace", ""),
                "tools_called": list(step.get("tools_called", []) or []),
                "data_sources_queried": _derive_step_data_sources(step),
                "result_summary": step.get("result_summary", ""),
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
        "Answer (comprehensive, 2-4 paragraphs), Detailed Findings (organized by theme with ### subsections, not by step), "
        "Conflicting & Uncertain Evidence (if any), Recommended Next Steps, Limitations."
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
        tc = _thinking_config_for_model(str(model_name))
        if tc:
            llm_request.config.thinking_config = tc
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
                result_desc = _describe_tool_result(tool_name, getattr(fr, "response", None))
                for entry in tool_log:
                    if entry.get("status") == "called" and entry.get("raw_tool") == tool_name:
                        entry["status"] = "done"
                        entry["result"] = result_desc
                        break
                else:
                    source = tool_registry.TOOL_SOURCE_NAMES.get(tool_name, tool_name)
                    tool_log.append({"tool": source, "raw_tool": tool_name, "status": "done",
                                     "summary": result_desc})
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
    """ReAct step executor: accept prose or JSON, extract structure post-hoc, advance."""
    if bool(callback_context.state.get(STATE_MODEL_ERROR_PASSTHROUGH, False)):
        callback_context.state[STATE_MODEL_ERROR_PASSTHROUGH] = False
        callback_context.state[STATE_EXECUTOR_BUFFER] = ""
        logger.info("[react:after] model error passthrough — skipping parse")
        return None

    if _llm_response_has_function_call(llm_response):
        callback_context.state[STATE_EXECUTOR_BUFFER] = ""
        fc_list = _extract_function_calls(llm_response)
        tool_names = [fc["name"] for fc in fc_list]
        thought_text = _llm_response_thought_text(llm_response).strip()
        text_alongside = _llm_response_text(llm_response).strip()

        # Build structured tool_log entries from actual function call data
        tool_log = _get_tool_log(callback_context)
        for fc in fc_list:
            source = tool_registry.TOOL_SOURCE_NAMES.get(fc["name"], fc["name"])
            description = _describe_tool_call(fc["name"], fc["args"])
            tool_log.append({"tool": source, "raw_tool": fc["name"], "status": "called",
                             "summary": description})
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
        logger.info("[react:after] function_call: %s, tool_log_len=%d",
                    ", ".join(tool_names), len(tool_log))
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

    # --- Build step result: try JSON first, then post-hoc extraction, then minimal ---
    parsed, _ = _parse_json_object_from_text(full_text)
    reasoning_trace = ""

    if parsed is not None and parsed.get("schema") == STEP_RESULT_SCHEMA:
        logger.info("[react:after] model produced valid JSON for %s", active_step_id)
        reasoning_trace = str(parsed.pop("reasoning_trace", "") or "").strip()
    else:
        parsed = None
        logger.info("[react:after] prose output for %s — running post-hoc extraction", active_step_id)
        extracted = _extract_structure_from_prose(full_text, active_step_id)

        regex_ids = _extract_evidence_ids_from_text(full_text)

        if extracted is not None:
            existing_ids = set(extracted.get("evidence_ids") or [])
            for eid in regex_ids:
                if eid not in existing_ids:
                    extracted.setdefault("evidence_ids", []).append(eid)
                    existing_ids.add(eid)
            parsed = extracted
            reasoning_trace = str(parsed.pop("reasoning_trace", "") or "").strip()
        else:
            logger.warning("[react:after] extraction failed for %s — using minimal result", active_step_id)
            parsed = {
                "schema": STEP_RESULT_SCHEMA,
                "step_id": active_step_id,
                "status": "completed",
                "step_progress_note": "Step completed (structure extraction unavailable).",
                "result_summary": full_text[:2000],
                "evidence_ids": regex_ids,
                "open_gaps": [],
                "suggested_next_searches": [],
                "tools_called": [],
                "data_sources_queried": [],
                "structured_observations": [],
            }

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
            step["status"] = "completed"
            step["result_summary"] = full_text[:2000]
            step["step_progress_note"] = "Step completed (validation fallback)."
            step["evidence_ids"] = _extract_evidence_ids_from_text(full_text)
            step["execution_metrics"] = _build_step_execution_metrics(
                step,
                {"step_id": active_step_id, "status": "completed", "tools_called": [], "evidence_ids": step["evidence_ids"], "open_gaps": []},
                parse_retry_count=0,
            )
            next_id = _next_pending_step_id(task_state)
            task_state["current_step_id"] = next_id
            task_state["plan_status"] = "completed" if next_id is None else "ready"
            _refresh_task_state_derived_state(task_state)
            validated = {"step_id": active_step_id, "status": "completed"}
        except Exception:  # noqa: BLE001
            logger.error("[react:after] fallback also failed for %s", active_step_id)
            return _replace_llm_response_text(llm_response, "")

    tool_log = _get_tool_log(callback_context)
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
    # Synthesizer uses 2.5 Flash by default; skip thinking_config (3.1-only feature)
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
    - **known_drug**, **drug_molecule**, **drug_mechanism_of_action**, **drug_warning**: drug data.
      IMPORTANT: Always run `list_bigquery_tables(dataset='open_targets_platform', table='known_drug')` (and same for drug_mechanism_of_action) to get exact column names before querying — schema can change. Use targetId (Ensembl ID) to join known_drug to targets; there is no drugId field in the target table. For IN UNNEST, use correct types: `col IN UNNEST(ARRAY<STRING>['a','b'])` not `col IN UNNEST(nested_struct)`.
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
    - Literature search: search_pubmed, search_pubmed_advanced, get_pubmed_abstract, get_paper_fulltext (PubMed/NCBI/PMC)
    - Literature search: search_openalex_works (OpenAlex — broader coverage, preprints)
    - Literature enrichment with preprints and citation metadata: search_europe_pmc_literature (Europe PMC)
    - Clinical trials: search_clinical_trials, get_clinical_trial, summarize_clinical_trials_landscape
    - Researcher discovery: search_openalex_authors, rank_researchers_by_activity
    - Gene identifier normalization: resolve_gene_identifiers (MyGene.info)
    - Ontology cross-mapping: map_ontology_terms_oxo (EBI OxO — MONDO/EFO/DOID/MeSH/OMIM/UMLS)
    - Gene Ontology lookup and annotations: search_quickgo_terms, get_quickgo_annotations (QuickGO)
    - Protein profiles: search_uniprot_proteins, get_uniprot_protein_profile
    - Pathways: search_reactome_pathways
    - Protein interactions: get_string_interactions
    - Curated experimental molecular interactions: get_intact_interactions (IntAct)
    - Broader experimental physical/genetic interaction evidence: get_biogrid_interactions (BioGRID)
    - Variant effect predictions (SIFT, PolyPhen, AlphaMissense): annotate_variants_vep (Ensembl VEP)
    - Aggregated variant annotations (ClinVar, CADD, dbSNP, gnomAD, COSMIC): get_variant_annotations (MyVariant.info)
    - Clinical variant interpretations in oncology: search_civic_variants, search_civic_genes (CIViC)
    - Protein structure predictions: get_alphafold_structure (AlphaFold — pLDDT confidence, PDB/CIF downloads)
    - GWAS trait-variant associations: search_gwas_associations (GWAS Catalog — p-values, odds ratios, mapped genes)
    - Drug-gene interactions & druggability: search_drug_gene_interactions (DGIdb — approved/experimental drugs)
    - Tissue-level gene expression: get_gene_tissue_expression (GTEx v8 — median TPM across 54 tissues)
    - Protein-level tissue, single-cell, and localization summaries: get_human_protein_atlas_gene (Human Protein Atlas)
    - Cancer-cell dependency and target vulnerability: get_depmap_gene_dependency (DepMap)
    - Published CRISPR screen evidence with phenotype/cell-line context: get_biogrid_orcs_gene_summary (BioGRID ORCS)
    - Compound sensitivity and pharmacogenomics: get_gdsc_drug_sensitivity (GDSC / CancerRxGene)
    - Broad repurposing single-dose response: get_prism_repurposing_response (PRISM Repurposing)
    - Cross-dataset public pharmacogenomics: get_pharmacodb_compound_response (PharmacoDB)
    - Single-cell dataset discovery by cell type/tissue/disease: search_cellxgene_datasets (CELLxGENE Discover / Census metadata)
    - Integrated top pathways across multiple pathway providers: search_pathway_commons_top_pathways (Pathway Commons)
    - Curated target-ligand pharmacology: get_guidetopharmacology_target (Guide to Pharmacology)
    - Current US drug label sections and boxed warnings: get_dailymed_drug_label (DailyMed)
    - Curated gene-disease validity and dosage sensitivity: get_clingen_gene_curation (ClinGen)
    - Model-organism orthologs and translational gene context: get_alliance_genome_gene_profile (Alliance Genome Resources)
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
Fall back to non-BQ tools for: literature search (search_pubmed, get_paper_fulltext, search_openalex_works), \
Europe PMC literature/preprints/citations (search_europe_pmc_literature), \
ClinicalTrials.gov, UniProt, Reactome pathways, STRING interactions, \
IntAct experimental interactions (get_intact_interactions), \
BioGRID experimental interactions (get_biogrid_interactions), \
gene identifier normalization (resolve_gene_identifiers via MyGene.info), \
ontology cross-mapping (map_ontology_terms_oxo via EBI OxO), \
GO ontology lookup and annotations (search_quickgo_terms, get_quickgo_annotations via QuickGO), \
variant effect predictions (annotate_variants_vep for SIFT/PolyPhen/AlphaMissense), \
aggregated variant annotations (get_variant_annotations for ClinVar/CADD/dbSNP/gnomAD/COSMIC), \
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


def _build_step_executor_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    tool_catalog = _format_tool_catalog(tool_hints)
    routing_policy = _format_source_precedence_rules(tool_hints)
    if prefer_bigquery:
        bq_policy = BQ_EXECUTOR_POLICY
    else:
        bq_policy = "- BigQuery-first policy is disabled for this run."

    return (
        STEP_EXECUTOR_INSTRUCTION_TEMPLATE
        .replace("__TOOL_CATALOG__", tool_catalog)
        .replace("__ROUTING_POLICY__", routing_policy)
        .replace("__BQ_POLICY__", bq_policy)
    )


def _format_domain_catalog() -> str:
    lines = []
    for domain in tool_registry.ALL_DOMAIN_NAMES:
        tools = tool_registry.TOOL_DOMAINS.get(domain, [])
        tool_names = ", ".join(tools[:12])
        always = " (always included)" if domain in tool_registry.ALWAYS_AVAILABLE_DOMAINS else ""
        lines.append(f"- {domain}{always}: {tool_names}")
    return "\n".join(lines)


def _build_planner_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    tool_catalog = _format_tool_catalog(tool_hints)
    domain_catalog = _format_domain_catalog()
    routing_policy = _format_source_precedence_rules(tool_hints)
    if prefer_bigquery:
        bq_policy = (
            "- BigQuery-first policy:\n"
            f"{BQ_DATASET_CATALOG}"
            "\n- tool_hint for BigQuery steps: use the specific dataset name (e.g. open_targets_platform,"
            " gnomad, ebi_chembl)"
            " rather than run_bigquery_select_query, so the plan clearly shows which source is being accessed.\n"
            "- For gene symbol / alias / Ensembl / Entrez normalization, use resolve_gene_identifiers (MyGene.info).\n"
            "- For ontology crosswalks across MONDO/EFO/DOID/MeSH/OMIM/UMLS, use map_ontology_terms_oxo (EBI OxO).\n"
            "- For GO term search and GO annotations, use search_quickgo_terms and get_quickgo_annotations (QuickGO).\n"
            "- For literature search that needs preprints or Europe PMC citation metadata, use search_europe_pmc_literature (Europe PMC).\n"
            "- For variant pathogenicity predictions, use annotate_variants_vep (Ensembl VEP — SIFT, PolyPhen, AlphaMissense).\n"
            "- For aggregated variant annotations (ClinVar, CADD, dbSNP, gnomAD, COSMIC), use get_variant_annotations (MyVariant.info).\n"
            "- For clinical variant interpretations in oncology, use search_civic_variants or search_civic_genes (CIViC).\n"
            "- For protein structure predictions and confidence scores, use get_alphafold_structure (AlphaFold API).\n"
            "- For GWAS trait-variant associations and genetic evidence, use search_gwas_associations (GWAS Catalog).\n"
            "- For druggability assessment and known drug-gene interactions, use search_drug_gene_interactions (DGIdb).\n"
            "- For curated experimental interaction evidence, use get_intact_interactions (IntAct).\n"
            "- For broader experimental physical/genetic interaction coverage and throughput context, use get_biogrid_interactions (BioGRID).\n"
            "- For curated target-ligand pharmacology and mechanism summaries, use get_guidetopharmacology_target (Guide to Pharmacology).\n"
            "- For current US label warnings, contraindications, and indications, use get_dailymed_drug_label (DailyMed).\n"
            "- For tissue-level gene expression and target safety, use get_gene_tissue_expression (GTEx).\n"
            "- For protein-level tissue specificity, single-cell specificity, and subcellular localization, use get_human_protein_atlas_gene (Human Protein Atlas).\n"
            "- For target dependency and cancer-cell vulnerability, use get_depmap_gene_dependency (DepMap).\n"
            "- For published CRISPR screen evidence with phenotype and cell-line context, use get_biogrid_orcs_gene_summary (BioGRID ORCS).\n"
            "- For compound sensitivity and cancer pharmacogenomics, use get_gdsc_drug_sensitivity (GDSC / CancerRxGene).\n"
            "- For Broad single-dose repurposing response across pooled cell lines, use get_prism_repurposing_response (PRISM Repurposing).\n"
            "- For harmonized cross-dataset pharmacogenomics across public screens, use get_pharmacodb_compound_response (PharmacoDB).\n"
            "- For single-cell dataset discovery by disease, tissue, cell type, assay, and organism, use search_cellxgene_datasets (CELLxGENE Discover / Census metadata).\n"
            "- For integrated pathway context across multiple pathway databases, use search_pathway_commons_top_pathways (Pathway Commons).\n"
            "- For curated gene-disease validity and dosage sensitivity, use get_clingen_gene_curation (ClinGen).\n"
            "- For model-organism orthologs, disease models, and translational gene summaries, use get_alliance_genome_gene_profile (Alliance Genome Resources).\n"
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
        .replace("__DOMAIN_CATALOG__", domain_catalog)
        .replace("__ROUTING_POLICY__", routing_policy)
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

    llm_request.append_instructions([
        "Current research report for reference:",
        report_context,
    ])
    return None


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
) -> tuple[LlmAgent | SequentialAgent, McpToolset | None]:
    """Create the routed ADK agent graph and return (root_agent, mcp_toolset).

    The root agent is an intent-classifying router that transfers to:
      - general_qa: factual biomedical Q&A (no tools)
      - clarifier: asks for clarification on vague/ambiguous queries
      - report_assistant: post-report interaction (with tools for light lookups)
      - research_workflow: full plan → execute → synthesize pipeline

    Args:
        require_plan_approval: When True, the research_workflow pauses after
            plan generation and waits for the user to ``approve`` or
            ``revise: <feedback>`` before executing the plan.
    """
    runtime_model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    planner_model = PLANNER_MODEL
    synthesizer_model = SYNTHESIZER_MODEL
    router_model = ROUTER_MODEL
    use_bigquery_priority = DEFAULT_PREFER_BIGQUERY if prefer_bigquery is None else bool(prefer_bigquery)

    mcp_toolset = create_mcp_toolset(tool_filter=tool_filter)
    executor_tools: list[McpToolset] = [mcp_toolset] if mcp_toolset is not None else []
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

    # ── Research workflow agents (plan → execute → synthesize) ────────────

    planner = LlmAgent(
        name="planner",
        model=planner_model,
        instruction=_build_planner_instruction(
            executor_tool_hints,
            prefer_bigquery=use_bigquery_priority,
        ),
        tools=[],
        disallow_transfer_to_parent=True,
        before_model_callback=_make_planner_before_model_callback(
            require_approval=require_plan_approval,
            model_name=planner_model,
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
        model=synthesizer_model,
        instruction=SYNTHESIZER_INSTRUCTION,
        tools=[],
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
        tools=executor_tools,
        disallow_transfer_to_parent=True,
        before_model_callback=_report_assistant_before_model_callback,
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
    )

    return router, mcp_toolset


__all__ = [
    "create_mcp_toolset",
    "create_workflow_agent",
]
