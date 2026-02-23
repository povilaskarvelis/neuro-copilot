"""
ADK-native orchestration graph for the Co-Scientist agent.

This workflow keeps planning internal to the model and does not expose
or require human confirmation of execution plans.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Any

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from mcp.client.stdio import StdioServerParameters


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

KNOWN_MCP_TOOLS = [
    "list_bigquery_tables",
    "run_bigquery_select_query",
    "benchmark_dataset_overview",
    "sample_pubmedqa_examples",
    "sample_bioasq_examples",
    "check_gpqa_access",
    "search_diseases",
    "expand_disease_context",
    "search_targets",
    "search_disease_targets",
    "get_target_info",
    "check_druggability",
    "get_target_drugs",
    "summarize_target_expression_context",
    "summarize_target_competitive_landscape",
    "summarize_target_safety_liabilities",
    "compare_targets_multi_axis",
    "search_clinical_trials",
    "get_clinical_trial",
    "summarize_clinical_trials_landscape",
    "search_pubmed",
    "search_pubmed_advanced",
    "get_pubmed_abstract",
    "get_pubmed_paper_details",
    "get_pubmed_author_profile",
    "search_openalex_works",
    "search_openalex_authors",
    "rank_researchers_by_activity",
    "get_researcher_contact_candidates",
    "search_chembl_compounds_for_target",
    "search_gwas_associations",
    "infer_genetic_effect_direction",
    "search_clinvar_variants",
    "get_clinvar_variant_details",
    "search_reactome_pathways",
    "get_string_interactions",
    "get_gene_info",
    "list_local_datasets",
    "read_local_dataset",
]


PLANNER_INSTRUCTION_TEMPLATE = """
You are the internal planner for biomedical investigation.

Available MCP tools:
__TOOL_CATALOG__

Rules:
- Build a concrete execution plan before any evidence collection begins.
- Break the objective into ordered, atomic subtasks.
- Prioritize high-signal subtasks that reduce uncertainty first.
- Each subtask must have:
  - a short goal,
  - the intended evidence source/tool family,
  - a clear completion condition.

__BQ_POLICY__
- Do not call tools.

The output should be very succinct. 
Output format:
To answer your query I will:
1) <goal> using <tool>
2) <goal> using <tool>
3) <goal> using <tool>
...

"""


EVIDENCE_EXECUTOR_INSTRUCTION_TEMPLATE = """
You execute biomedical evidence collection and validation.
Follow the planner's subtask breakdown from earlier in this turn before taking actions.
Use a ReAct-style workflow internally: reason about the next action, call tools when needed, observe results, then reassess.

Available MCP tools:
__TOOL_CATALOG__

Rules:
- Execute one subtask at a time and reassess after each subtask.
- Continue to the next subtask only after summarizing what was learned from the current one.
- Prioritize high-signal evidence before broad expansion.
- Use MCP tools only when they directly improve evidence quality.
__BQ_POLICY__
- Surface contradictions and unresolved gaps explicitly.
- Include source identifiers when available (PMID, DOI, NCT, OpenAlex IDs).

Return only:
Short updates on your progress and evidence gathered.
End with:
- Open gaps / uncertainty
- Suggested next searches or tool calls (if more work is needed)
"""

SYNTHESIZER_INSTRUCTION = """
You are the final biomedical report synthesizer.
Use the conversation evidence gathered in prior workflow steps.

Write a concise final answer by stating the direct answer upfront and highlighting the sources of evidence. Explicitly list:
Limitations and uncertainty.
Practical next actions.

Do not invent unsupported claims.
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


def _build_evidence_executor_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    tool_catalog = "\n".join(f"- {name}" for name in tool_hints[:80]) or "- No tools available."
    if prefer_bigquery:
        bq_policy = (
            "- BigQuery-first policy:\n"
            "  - For structured/tabular analysis, start with `list_bigquery_tables` and `run_bigquery_select_query`.\n"
            "  - Use non-BigQuery tools for enrichment, freshness gaps, or unavailable data."
        )
    else:
        bq_policy = "- BigQuery-first policy is disabled for this run."

    return (
        EVIDENCE_EXECUTOR_INSTRUCTION_TEMPLATE
        .replace("__TOOL_CATALOG__", tool_catalog)
        .replace("__BQ_POLICY__", bq_policy)
    )


def _build_planner_instruction(tool_hints: list[str], *, prefer_bigquery: bool) -> str:
    tool_catalog = "\n".join(f"- {name}" for name in tool_hints[:80]) or "- No tools available."
    if prefer_bigquery:
        bq_policy = (
            "- BigQuery-first policy:\n"
            "  - Prefer `list_bigquery_tables` and `run_bigquery_select_query` for structured/tabular subtasks.\n"
            "  - Use non-BigQuery tools for enrichment, freshness gaps, or unavailable data."
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
) -> tuple[SequentialAgent, McpToolset | None]:
    """Create an ADK-native workflow graph and return (root_agent, mcp_toolset)."""
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

    planner = LlmAgent(
        name="planner",
        model=runtime_model,
        instruction=_build_planner_instruction(
            executor_tool_hints,
            prefer_bigquery=use_bigquery_priority,
        ),
        tools=[],
        disallow_transfer_to_parent=True,
    )
    evidence_executor = LlmAgent(
        name="evidence_executor",
        model=runtime_model,
        instruction=_build_evidence_executor_instruction(
            executor_tool_hints,
            prefer_bigquery=use_bigquery_priority,
        ),
        tools=executor_tools,
    )
    report_synthesizer = LlmAgent(
        name="report_synthesizer",
        model=runtime_model,
        instruction=SYNTHESIZER_INSTRUCTION,
        tools=[],
    )

    root = SequentialAgent(
        name="co_scientist_workflow",
        description="ADK-native biomedical workflow: planner, executor, synthesis.",
        sub_agents=[planner, evidence_executor, report_synthesizer],
    )
    return root, mcp_toolset


__all__ = [
    "create_mcp_toolset",
    "create_workflow_agent",
]
