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
from typing import Any
import urllib.error
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
    # BigQuery (primary data source — covers gene info, disease ontology,
    # drug labels, literature, preprints, variant annotations, and more)
    "list_bigquery_tables",
    "run_bigquery_select_query",
    # Clinical trials (live registry, no BQ equivalent)
    "search_clinical_trials",
    "get_clinical_trial",
    "summarize_clinical_trials_landscape",
    # Post-marketing safety aggregations (richer event-level rollups than BQ)
    "summarize_openfda_adverse_events",
    # Researcher discovery (live index, no BQ equivalent)
    "search_openalex_works",
    "search_openalex_authors",
    "rank_researchers_by_activity",
    "get_researcher_contact_candidates",
    # Protein profiles (detailed isoforms/PTMs beyond AlphaFold metadata)
    "search_uniprot_proteins",
    "get_uniprot_protein_profile",
    # Pathway hierarchy (Reactome API returns full pathway trees)
    "search_reactome_pathways",
    # Protein-protein interactions (no STRING data in BQ)
    "get_string_interactions",
    # Benchmarks
    "benchmark_dataset_overview",
    "check_gpqa_access",
    # Local data
    "list_local_datasets",
    "read_local_dataset",
]

TOOL_SOURCE_NAMES: dict[str, str] = {
    # Generic BQ tool names (fallback when no dataset hint is available)
    "list_bigquery_tables": "BigQuery",
    "run_bigquery_select_query": "BigQuery",
    # BigQuery dataset names — used as tool_hint by the planner for precise source attribution
    "open_targets_platform": "Open Targets Platform",
    "ebi_chembl": "ChEMBL",
    "gnomad": "gnomAD",
    "human_genome_variants": "Human Genome Variants",
    "human_variant_annotation": "Ensembl Variant Annotations",
    "deepmind_alphafold": "AlphaFold",
    "immune_epitope_db": "IEDB",
    "nlm_rxnorm": "RxNorm",
    "fda_drug": "FDA Drug Labels",
    "umiami_lincs": "LINCS L1000",
    "ebi_surechembl": "SureChEMBL",
    "pmc_open_access_commercial": "PubMed Central",
    "breathe": "bioRxiv / arXiv",
    "hackathon_data": "CIViC / ClinGen",
    "civic": "CIViC",
    "clingen": "ClinGen",
    "benchmark_dataset_overview": "Benchmark Datasets",
    "sample_pubmedqa_examples": "PubMedQA",
    "sample_bioasq_examples": "BioASQ",
    "check_gpqa_access": "GPQA",
    "search_diseases": "Open Targets Platform",
    "expand_disease_context": "Open Targets Platform",
    "search_targets": "Open Targets Platform",
    "search_disease_targets": "Open Targets Platform",
    "get_target_info": "Open Targets Platform",
    "check_druggability": "Open Targets Platform",
    "get_target_drugs": "Open Targets Platform",
    "summarize_target_expression_context": "Open Targets Platform",
    "summarize_target_competitive_landscape": "Open Targets Platform",
    "summarize_target_safety_liabilities": "Open Targets Platform",
    "compare_targets_multi_axis": "Open Targets Platform",
    "search_clinical_trials": "ClinicalTrials.gov",
    "get_clinical_trial": "ClinicalTrials.gov",
    "summarize_clinical_trials_landscape": "ClinicalTrials.gov",
    "search_pubmed": "PubMed",
    "search_pubmed_advanced": "PubMed",
    "get_pubmed_abstract": "PubMed",
    "get_pubmed_paper_details": "PubMed",
    "get_pubmed_author_profile": "PubMed",
    "search_openalex_works": "OpenAlex",
    "search_openalex_authors": "OpenAlex",
    "rank_researchers_by_activity": "OpenAlex",
    "get_researcher_contact_candidates": "OpenAlex",
    "search_chembl_compounds_for_target": "ChEMBL",
    "search_gwas_associations": "GWAS Catalog",
    "infer_genetic_effect_direction": "GWAS Catalog",
    "search_clinvar_variants": "ClinVar",
    "get_clinvar_variant_details": "ClinVar",
    "search_reactome_pathways": "Reactome",
    "get_string_interactions": "STRING",
    "get_gene_info": "NCBI Gene",
    "list_local_datasets": "Local Datasets",
    "read_local_dataset": "Local Datasets",
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
- Use step ids S1, S2, S3, ... in order.
- Do not call tools.

__BQ_POLICY__

Output requirements:
- Return ONLY valid JSON (no markdown, no prose) matching this shape:
  {
    "schema": "plan_internal.v1",
    "objective": "<restated objective>",
    "success_criteria": ["..."],
    "steps": [
      {
        "id": "S1",
        "goal": "...",
        "tool_hint": "<single best tool, tool family, or BigQuery dataset name (e.g. open_targets_platform, pmc_open_access_commercial, gnomad)>",
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
- If a tool call fails or returns insufficient data, try an alternative tool or query.
- If no tool can satisfy the step, mark it as blocked with a clear reason.
- Prioritize high-signal evidence before broad expansion.
- Surface contradictions and unresolved gaps explicitly.
- Include source identifiers when available (PMID, DOI, NCT, OpenAlex IDs).
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
    "evidence_ids": ["PMID:...", "NCT:..."],
    "open_gaps": ["..."],
    "suggested_next_searches": ["..."]
  }
- The reasoning_trace MUST use labeled phases (REASON, ACT, OBSERVE, CONCLUDE) to clearly show each step of the Reason-Act-Observe cycle. If you made multiple tool calls, include multiple ACT/OBSERVE pairs.
- The tools_called list MUST contain the names of every MCP tool you invoked during this step.
"""


SYNTHESIZER_INSTRUCTION = """
You are the final biomedical report synthesizer.
You will receive structured state context (objective, plan steps, step results, coverage status, and a source_reference mapping).

Rules:
- Produce a final summary grounded only in the provided evidence/results.
- If the plan is incomplete, clearly state that the summary is partial.
- Do not invent unsupported claims.
- Avoid terse output. Be specific and useful.
- For each supporting evidence item, include:
  - the evidence-backed claim,
  - why it matters for the objective (rationale),
  - source identifiers when available.
- Always include 3 potential next steps, even when the plan is complete (e.g., confirmatory checks, risk reduction, monitoring, or decision-oriented follow-up).

Source citation rules:
- Each step in the context has a `source` field with the database/source name (e.g. "PubMed", "ClinicalTrials.gov").
- A `source_reference` legend maps internal tool names to database names.
- When citing findings, use ONLY the database/source name. NEVER mention tool names (like search_pubmed, get_target_info, etc.) in the report.
  Good: "According to PubMed, three RCTs showed..."
  Good: "Open Targets Platform data indicates BRCA1 has high genetic association..."
  Good: "ClinicalTrials.gov lists 12 active Phase II trials..."
  Bad:  "search_pubmed returned three RCTs..."
  Bad:  "Using the search_pubmed tool..."
  Bad:  "PubMed (search_pubmed) data shows..."
- Include specific identifiers inline when available (PMID, DOI, NCT numbers, etc.).
- NEVER include raw URLs, API endpoints, or links to JSON output in the report. Only use human-readable source names and identifiers.

Output requirements:
- Return user-facing Markdown.
- Start with `## Final Summary`.
- Present findings grouped by step. For each step, state the source explicitly on its own line before the findings, e.g.:
  
  **Source:** PubMed
  Three RCTs (PMID: 12345678, PMID: 23456789) demonstrated...
- End with `Limitations` and `Potential Next Steps` sections.
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
        try:
            parsed = json.loads(candidate)
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
    return {
        "schema": WORKFLOW_TASK_SCHEMA,
        "objective": objective_text or validated["objective"],
        "objective_fingerprint": _normalize_user_text(objective_text or validated["objective"]),
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
    r"|\b(?P<pmc>PMC\d+)\b",
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
        else:
            continue
        key = normalized.lower()
        if key not in seen:
            seen.add(key)
            ids.append(normalized)
    return ids


def _collect_all_evidence_ids(task_state: dict[str, Any]) -> list[str]:
    """Return a deduplicated list of evidence IDs from all completed steps."""
    seen: set[str] = set()
    ids: list[str] = []
    for step in task_state.get("steps", []):
        for eid in step.get("evidence_ids", []) or []:
            normalized = re.sub(r"\s*:\s*", ":", str(eid).strip())
            if normalized and normalized.lower() not in seen:
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
        "year": str(result.get("pubdate", "")).split()[0],
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
        return f"{anchor}{i}. PMID: [{pmid}]({pmid_url})"

    # DOI — fetch from CrossRef
    m = re.fullmatch(r"(?i)DOI:(10\..+)", raw)
    if m:
        doi = m.group(1)
        doi_url = f"https://doi.org/{doi}"
        meta = _fetch_crossref_meta(doi)
        if meta and meta.get("title"):
            return f"{anchor}{i}. {_build_apa_citation(meta, doi=doi)}"
        return f"{anchor}{i}. [{doi_url}]({doi_url})"

    # NCT — ClinicalTrials.gov
    m = re.fullmatch(r"(?i)(?:NCT:)?(NCT\d{8})", raw)
    if m:
        nct = m.group(1).upper()
        url = f"https://clinicaltrials.gov/study/{nct}"
        return f"{anchor}{i}. U.S. National Library of Medicine. (n.d.). *ClinicalTrials.gov*. [{nct}]({url})"

    # OpenAlex
    m = re.fullmatch(r"(?i)OpenAlex:(W\d+)", raw)
    if m:
        wid = m.group(1)
        url = f"https://openalex.org/{wid}"
        return f"{anchor}{i}. *OpenAlex*. [{wid}]({url})"

    # PMC
    m = re.fullmatch(r"(?i)(?:PMC:)?(PMC\d+)", raw)
    if m:
        pmc = m.group(1).upper()
        url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/"
        return f"{anchor}{i}. National Library of Medicine. (n.d.). *PubMed Central*. [{pmc}]({url})"

    url = _evidence_id_to_url(eid)
    if url:
        return f"{anchor}{i}. [{eid}]({url})"
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

    When ref_map is provided, links point to the local #ref-N anchor so readers
    can click through to the APA entry at the bottom of the document.
    Otherwise falls back to the external URL for the identifier.
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
        else:
            return m.group(0)
        ref_key = re.sub(r"\s*:\s*", ":", normalized).lower()
        if ref_map and ref_key in ref_map:
            n = ref_map[ref_key]
            return f"[{n}](#ref-{n})"
        url = _evidence_id_to_url(normalized)
        if not url:
            return m.group(0)
        return f"[{display}]({url})"

    linked = _INLINE_ID_RE.sub(_replace_id, protected)

    # Restore placeholders.
    for idx, original in enumerate(placeholders):
        linked = linked.replace(f"\x00P{idx}\x00", original)

    return linked + refs_tail


def _merge_reference_ids(inline_ids: list[str], task_ids: list[str]) -> list[str]:
    """Merge inline-extracted IDs (first) with task-state IDs, deduplicating."""
    seen: set[str] = set()
    combined: list[str] = []
    for eid in inline_ids + task_ids:
        key = re.sub(r"\s*:\s*", ":", eid).lower()
        if key not in seen:
            seen.add(key)
            combined.append(eid)
    return combined


# ---------------------------------------------------------------------------


def _fallback_supporting_evidence_from_task_state(task_state: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for step in task_state.get("steps", []):
        if str(step.get("status", "")).strip() != "completed":
            continue
        goal = str(step.get("goal", "")).strip()
        summary = str(step.get("result_summary", "")).strip()
        if not summary:
            continue
        evidence_ids = [str(x).strip() for x in step.get("evidence_ids", []) if str(x).strip()]
        evidence_suffix = f" (IDs: {', '.join(evidence_ids[:6])})" if evidence_ids else ""
        if goal:
            items.append(f"{summary} This matters because it addresses the step goal: {goal}.{evidence_suffix}")
        else:
            items.append(summary + evidence_suffix)
        if len(items) >= 8:
            break
    return items


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


def _coverage_note_from_task_state(task_state: dict[str, Any]) -> str:
    coverage = _compute_coverage_status(task_state)
    completed = _completed_step_count(task_state)
    total = _total_step_count(task_state)
    coverage_label = "Complete plan" if coverage == "complete_plan" else "Partial plan"
    if coverage == "partial_plan":
        return (
            f"_Coverage: {coverage_label} ({completed} of {total} planned steps completed when final summary was requested)._"
        )
    return f"_Coverage: {coverage_label} ({completed} of {total} planned steps completed)._"



def _postprocess_synth_markdown(task_state: dict[str, Any], raw_markdown: str) -> str:
    text = str(raw_markdown or "").strip()
    if not text:
        text = "# AI Co-Scientist Report\n\n## Summary\n\nNo final summary was produced."

    has_title = "# AI Co-Scientist Report" in text
    has_old_heading = "## Final Summary" in text or "# Final Summary" in text
    if not has_title and not has_old_heading:
        text = "# AI Co-Scientist Report\n\n## Summary\n\n" + text
    elif has_old_heading and not has_title:
        # Upgrade legacy heading
        text = text.replace("## Final Summary", "## Summary").replace("# Final Summary", "## Summary")
        text = "# AI Co-Scientist Report\n\n" + text

    lowered = text.lower()

    # References: inject before Potential Next Steps so _strip_next_steps_section in ui_server preserves them
    inline_ids = _extract_inline_ids_from_text(text)
    task_ids = _collect_all_evidence_ids(task_state)
    all_ids = _merge_reference_ids(inline_ids, task_ids)
    ref_map = _build_ref_map(all_ids)
    if "## References" not in text:
        refs = _build_references_section(all_ids)
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

    # Hyperlink bare inline IDs — link to local #ref-N anchors when available
    text = _hyperlink_inline_ids(text, ref_map)

    lowered = text.lower()
    if "potential next steps" not in lowered and "next steps" not in lowered and "next actions" not in lowered:
        fallback_next = _fallback_next_actions_from_task_state(task_state)
        if fallback_next:
            text += "\n\n## Potential Next Steps\n\n"
            text += "\n".join(f"{i}. {item}" for i, item in enumerate(fallback_next[:20], start=1))

    lowered = text.lower()
    if "_coverage:" not in lowered and "coverage:" not in lowered:
        text += "\n\n" + _coverage_note_from_task_state(task_state)

    return text.strip()


def _render_final_synthesis_markdown(task_state: dict[str, Any], synthesis: dict[str, Any]) -> str:
    objective = str(task_state.get("objective", "")).strip()
    coverage = str(synthesis.get("coverage_status", "partial_plan"))
    completed = _completed_step_count(task_state)
    total = _total_step_count(task_state)
    coverage_label = "Complete" if coverage == "complete_plan" else "Partial"

    lines = ["# AI Co-Scientist Report", ""]

    # Research question + coverage callout
    if objective:
        lines += [
            f"> **Research Question:** {objective}",
            ">",
            f"> **Coverage:** {coverage_label} \u2014 {completed} of {total} planned steps completed",
            "",
            "---",
            "",
        ]

    # Summary
    lines += ["## Summary", ""]
    direct_answer = str(synthesis.get("direct_answer", "")).strip()
    if direct_answer:
        lines.append(direct_answer)
        lines.append("")

    # Supporting Evidence
    supporting = [str(x).strip() for x in synthesis.get("supporting_evidence", []) if str(x).strip()]
    if not supporting:
        supporting = _fallback_supporting_evidence_from_task_state(task_state)
    if supporting:
        lines += ["## Supporting Evidence", ""]
        for i, item in enumerate(supporting[:20], start=1):
            lines.append(f"**{i}.** {item}")
            lines.append("")

    # Limitations
    limitations = [str(x).strip() for x in synthesis.get("limitations", []) if str(x).strip()]
    if limitations:
        lines += ["## Limitations", ""]
        lines.extend(f"- {item}" for item in limitations[:20])
        lines.append("")

    # Collect IDs, build ref_map, then render References and hyperlink inline mentions
    current_text = "\n".join(lines)
    inline_ids = _extract_inline_ids_from_text(current_text)
    task_ids = _collect_all_evidence_ids(task_state)
    all_ids = _merge_reference_ids(inline_ids, task_ids)
    ref_map = _build_ref_map(all_ids)
    refs = _build_references_section(all_ids)
    if refs:
        lines += refs.split("\n")
        lines.append("")

    # Hyperlink inline IDs — internal anchors to #ref-N where available
    body_so_far = "\n".join(lines)
    body_so_far = _hyperlink_inline_ids(body_so_far, ref_map)
    lines = body_so_far.split("\n")

    # Next Steps (after References so _strip_next_steps_section in ui_server won't remove References)
    next_actions = [str(x).strip() for x in synthesis.get("next_actions", []) if str(x).strip()]
    if not next_actions:
        next_actions = _fallback_next_actions_from_task_state(task_state)
    if next_actions:
        lines += ["## Potential Next Steps", ""]
        for i, item in enumerate(next_actions[:20], start=1):
            lines.append(f"{i}. {item}")
        lines.append("")

    lines.append("---")
    lines.append("")
    if coverage == "partial_plan":
        lines.append(
            f"_Coverage: Partial plan \u2014 {completed} of {total} planned steps completed when final summary was requested._"
        )
    else:
        lines.append(
            f"_Coverage: Complete plan \u2014 all {total} planned steps completed._"
        )

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


def _render_all_steps_complete_message() -> str:
    return (
        "## Execution\n\n"
        "All planned steps are already complete. Reply `finalize` to generate the final summary."
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
        "Return user-facing Markdown (not JSON). Include a direct answer, supporting evidence with rationale, "
        "limitations, potential next steps, and a coverage note (complete vs partial plan)."
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


def _on_model_error(
    *,
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
    """Surface model-level errors (rate limits, network, etc.) as visible output."""
    error_type = type(error).__name__
    error_msg = str(error)
    logger.error("Model error in %s: [%s] %s", "agent", error_type, error_msg)

    is_rate_limit = any(
        hint in error_msg.lower()
        for hint in ("429", "resource exhausted", "rate limit", "quota")
    )
    if is_rate_limit:
        user_msg = (
            "## Execution Paused — Rate Limit\n\n"
            "The model API returned a rate-limit error. "
            "Please wait a moment and send `continue` to resume."
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

    return {
        "error": True,
        "error_type": error_type,
        "message": f"Tool '{tool_name}' failed: {error_msg[:500]}",
        "suggestion": "Try an alternative tool or skip this step.",
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
                    f"Step completed via prose fallback after {MAX_REACT_PARSE_RETRIES + 1} "
                    f"JSON formatting attempts. {error_label}: {error_msg}"
                )
                next_id = _next_pending_step_id(task_state)
                task_state["current_step_id"] = next_id
                new_plan_status = "completed" if next_id is None else "ready"
                task_state["plan_status"] = new_plan_status
                callback_context.state[STATE_WORKFLOW_TASK] = task_state
                if new_plan_status == "completed":
                    callback_context.state[STATE_AUTO_SYNTH_REQUESTED] = True
                rendered = (
                    f"### {active_step_id} · completed _(via prose fallback)_\n\n"
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
  **bigquery-public-data.open_targets_platform** — Comprehensive drug target data:
    - disease, target, interaction, expression: disease-target associations, tractability/druggability
    - known_drug, drug_molecule, drug_mechanism_of_action, drug_warning: drugs and safety
    - evidence_* tables (evidence_gwas_credible_sets, evidence_chembl, evidence_clingen, evidence_reactome,
      evidence_eva, evidence_intogen, evidence_gene_burden, etc.): all evidence types
    - openfda_significant_adverse_drug_reactions: post-marketing safety signals
    - go, mouse_phenotype, literature: gene ontology, phenotypes, literature mining
    - l2g_prediction, credible_set, colocalisation: genetic evidence and variant-to-gene mapping

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
  **bigquery-public-data.human_variant_annotation** — Ensembl variant annotations (1000 Genomes, ESP6500)
    across hg19/hg38 builds (release 89 & 93). Functional consequence, SIFT, PolyPhen scores.

  === Protein structure ===
  **bigquery-public-data.deepmind_alphafold** — AlphaFold protein structure predictions. Tables: metadata
    (UniProt accession, organism, confidence metrics, PDB/CIF URLs). Use to assess structural druggability
    and binding-site availability for targets lacking experimental crystal structures.

  === Immunology ===
  **bigquery-public-data.immune_epitope_db** — IEDB: immune epitopes, B-cell assays, MHC ligand binding,
    T-cell receptor data. Tables: epitope_full_v3, bcell_full_v3, mhc_ligand_full, receptor_full_v3, etc.
    Use for immunotherapy target assessment and vaccine design.

  === Drug nomenclature & regulatory ===
  **bigquery-public-data.nlm_rxnorm** — RxNorm drug nomenclature, ingredient relationships, and
    clinical drug pathways. Use for standardizing drug names across sources.
  **bigquery-public-data.fda_drug** — FDA drug labels and enforcement actions. Tables: drug_label,
    drug_enforcement. Structured alternative to openFDA API for label lookups.

  === Perturbation biology ===
  **bigquery-public-data.umiami_lincs** — LINCS L1000 perturbation signatures: cell lines, small molecules,
    nucleic acid reagents, readouts, and perturbation signatures. Use for drug repurposing hypotheses
    and mechanism-of-action characterization.

  === Literature & preprints ===
  **bigquery-public-data.pmc_open_access_commercial** — PubMed Central full-text articles with vector
    embeddings. Tables: articles (pmid, pmc_id, title, author, article_text, pmc_link,
    ml_generate_embedding_result), pmc_metadata. Use SEARCH() for keyword search, VECTOR_SEARCH()
    for semantic similarity. Provides complete article text, not just abstracts.
  **bigquery-public-data.breathe** — Full-text bioRxiv preprints, arxiv papers, and BioASQ QA data.
    Tables: biorxiv, arxiv, bioasq. Complements PMC for preprint literature.

  === Hackathon project data ===
  **hackathon_data** (project dataset) — Contains:
    - civic_* tables: Cancer variant clinical interpretations (assertions, evidence, variants, molecular profiles)
    - clingen_* tables: Gene-disease validity, variant pathogenicity, dosage sensitivity, actionability

  Start every structured data lookup with BigQuery. Use `list_bigquery_tables` to discover table schemas.
  Write Standard SQL via `run_bigquery_select_query`. Fall back to non-BigQuery MCP tools only for:
    - ClinicalTrials.gov (search_clinical_trials, get_clinical_trial, etc.)
    - openFDA adverse event aggregations (summarize_openfda_adverse_events — richer rollups than fda_drug tables)
    - OpenAlex researcher discovery (search_openalex_works, search_openalex_authors, etc.)
    - UniProt protein profiles (detailed isoforms, PTMs beyond AlphaFold metadata)
    - Reactome pathway hierarchies (search_reactome_pathways)
    - STRING protein-protein interactions (get_string_interactions)
  Do NOT use external APIs for: gene info (use open_targets_platform.target), disease ontology
    (use open_targets_platform.disease), drug labels (use fda_drug.drug_label), literature search
    (use pmc_open_access_commercial or breathe), preprints (use breathe.biorxiv/medrxiv),
    variant annotations (use gnomAD, human_variant_annotation, human_genome_variants)."""


BQ_EXECUTOR_POLICY = """- BigQuery-first policy: For any structured data lookup, prefer `list_bigquery_tables` \
and `run_bigquery_select_query` over non-BQ tools. Available datasets include: \
open_targets_platform (targets, diseases, drugs, evidence), ebi_chembl (bioactivity), \
gnomAD (variant frequencies), human_genome_variants, human_variant_annotation (Ensembl), \
deepmind_alphafold (protein structures), immune_epitope_db (IEDB), fda_drug (labels), \
nlm_rxnorm (drug nomenclature), umiami_lincs (perturbation signatures), ebi_surechembl (patents), \
pmc_open_access_commercial (PubMed Central full text), breathe (bioRxiv/arxiv/BioASQ), \
and project hackathon_data (CIViC, ClinGen). Use `list_bigquery_tables` to discover schemas. \
Fall back to non-BQ tools only for ClinicalTrials.gov, openFDA aggregations, OpenAlex, \
UniProt, Reactome pathways, and STRING interactions."""


def _build_step_executor_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    tool_catalog = "\n".join(f"- {name}" for name in tool_hints[:80]) or "- No tools available."
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
    tool_catalog = "\n".join(f"- {name}" for name in tool_hints[:80]) or "- No tools available."
    if prefer_bigquery:
        bq_policy = (
            "- BigQuery-first policy:\n"
            f"{BQ_DATASET_CATALOG}"
            "\n- tool_hint for BigQuery steps: use the specific dataset name (e.g. open_targets_platform,"
            " pmc_open_access_commercial, gnomad, ebi_chembl, deepmind_alphafold, fda_drug, hackathon_data)"
            " rather than run_bigquery_select_query, so the plan clearly shows which source is being accessed."
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
    max_plan_iterations: int | None = None,
    prefer_bigquery: bool | None = None,
    require_plan_approval: bool = False,
) -> tuple[SequentialAgent, McpToolset | None]:
    """Create an ADK-native workflow graph and return (root_agent, mcp_toolset).

    Args:
        require_plan_approval: When True, the workflow pauses after plan
            generation and waits for the user to ``approve`` or
            ``revise: <feedback>`` before executing the plan.
    """
    del max_plan_iterations

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
