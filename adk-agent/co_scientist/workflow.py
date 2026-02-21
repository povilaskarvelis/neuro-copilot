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

from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.genai import types
from google.adk.tools import McpToolset, ToolContext, exit_loop
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from mcp.client.stdio import StdioServerParameters


MCP_SERVER_DIR = Path(__file__).resolve().parents[2] / "research-mcp"
DEFAULT_MODEL = os.getenv("ADK_NATIVE_MODEL", "gemini-2.5-flash")
DEFAULT_EVIDENCE_MAX_ITERS = max(
    1,
    int(os.getenv("ADK_NATIVE_EVIDENCE_MAX_ITERS", "3") or "3"),
)
HAS_BIGQUERY_RUNTIME_HINT = any(
    str(os.getenv(name, "")).strip()
    for name in ("BQ_PROJECT_ID", "BQ_DATASET_ALLOWLIST", "GOOGLE_CLOUD_PROJECT")
)
DEFAULT_PREFER_BIGQUERY = (
    str(os.getenv("ADK_NATIVE_PREFER_BIGQUERY", "1")).strip().lower() not in {"0", "false", "no"}
    and HAS_BIGQUERY_RUNTIME_HINT
)
REQUIRE_EXPLICIT_EVIDENCE_CHECKPOINT_DECISION = str(
    os.getenv("ADK_NATIVE_REQUIRE_EVIDENCE_CHECKPOINT_DECISION", "1")
).strip().lower() not in {"0", "false", "no"}
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
"""


EVIDENCE_CRITIC_INSTRUCTION = """
You are the evidence sufficiency critic in a refinement loop.
Assess whether current evidence is sufficient for a defensible synthesis.

If sufficient:
- Call `exit_loop` immediately.
- Do not emit any extra text after the tool call.

If insufficient:
- Do NOT call `exit_loop`.
- Return exactly:
NEEDS_MORE_EVIDENCE:
- <specific missing evidence item>
- <specific missing evidence item>

Be strict about citation-backed claims and unresolved contradictions.
"""


SYNTHESIZER_INSTRUCTION = """
You are the final biomedical report synthesizer.
Use the conversation evidence gathered in prior workflow steps.

Write a concise final answer:
1) Direct answer up front.
2) Key supporting evidence with citations/IDs.
3) Explicit limitations and uncertainty.
4) Practical next actions.

Do not invent unsupported claims.
"""


EVIDENCE_HITL_GATE_INSTRUCTION = """
You are the evidence continuation checkpoint in the refinement loop.

Rules:
- Inspect the latest critic output.
- If the critic indicates `NEEDS_MORE_EVIDENCE`, call `request_evidence_continuation` exactly once
  with concrete missing evidence and next actions.
- If the critic does not indicate `NEEDS_MORE_EVIDENCE`, do not call tools and return `CHECKPOINT_NOT_REQUIRED`.
- After the tool result:
  - if status is `approved`, return `CHECKPOINT_APPROVED`.
  - if status is `declined`, return `CHECKPOINT_DECLINED`.
  - if status is `pending_human_confirmation`, return `CHECKPOINT_AWAITING_CONFIRMATION`.

When calling `request_evidence_continuation`, provide:
- `missing_evidence`: short list of specific missing evidence items.
- `proposed_next_actions`: short list of concrete actions to run next.
- `rationale`: one concise sentence.
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


def _extract_user_turn_text(tool_context: ToolContext) -> str:
    user_content = getattr(tool_context, "user_content", None)
    parts = getattr(user_content, "parts", None) if user_content is not None else None
    if not parts:
        return ""
    text = " ".join(
        str(getattr(part, "text", "") or "").strip()
        for part in parts
        if str(getattr(part, "text", "") or "").strip()
    )
    return re.sub(r"\s+", " ", text).strip()


def _parse_manual_evidence_checkpoint_decision(tool_context: ToolContext) -> tuple[str, str]:
    turn_text = _extract_user_turn_text(tool_context)
    lowered = turn_text.lower().strip()
    if lowered in {"approve", "approved", "yes", "y", "/approve", "continue", "/continue"}:
        return "approve", ""
    if lowered.startswith("stop:"):
        feedback = turn_text.split(":", 1)[1].strip()
        return "stop", feedback
    if lowered.startswith("decline:"):
        feedback = turn_text.split(":", 1)[1].strip()
        return "stop", feedback
    if lowered in {"stop", "decline", "declined", "no", "n", "/stop", "/decline"}:
        return "stop", ""
    return "", ""


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


def _make_request_evidence_continuation_tool():
    def request_evidence_continuation(
        missing_evidence: list[str] | None = None,
        proposed_next_actions: list[str] | None = None,
        rationale: str | None = None,
        tool_context: ToolContext | None = None,
    ) -> dict[str, Any]:
        """Request a human checkpoint before collecting more evidence."""
        if tool_context is None:
            return {
                "status": "declined",
                "feedback": "Missing tool context for checkpoint.",
            }

        missing_items = _dedupe_str_list(missing_evidence or [], limit=8)
        next_actions = _dedupe_str_list(proposed_next_actions or [], limit=8)
        rationale_text = re.sub(r"\s+", " ", str(rationale or "").strip())

        payload = {
            "schema": "evidence_continuation_checkpoint.v1",
            "missing_evidence": missing_items,
            "proposed_next_actions": next_actions,
            "rationale": rationale_text,
            "response_contract": {
                "decision": "approve | stop",
                "feedback": "Optional short feedback.",
            },
        }

        pending_key = "evidence_checkpoint_pending"
        confirmation = tool_context.tool_confirmation
        if not confirmation:
            has_pending_checkpoint = bool(tool_context.state.get(pending_key, False))
            if has_pending_checkpoint:
                manual_decision, manual_feedback = _parse_manual_evidence_checkpoint_decision(tool_context)
                if manual_decision == "approve":
                    tool_context.state[pending_key] = False
                    return {
                        "status": "approved",
                        "feedback": manual_feedback,
                    }
                if manual_decision == "stop":
                    tool_context.state[pending_key] = False
                    exit_loop(tool_context=tool_context)
                    return {
                        "status": "declined",
                        "feedback": manual_feedback or "User chose to stop additional evidence collection.",
                    }

            tool_context.request_confirmation(
                hint=(
                    "Additional evidence collection is proposed. "
                    "Approve to continue, or stop and synthesize from current evidence. "
                    "If your UI cannot send payload choices, reply in chat with `approve` or `stop`."
                ),
                payload=payload,
            )
            tool_context.actions.skip_summarization = True
            tool_context.state[pending_key] = True
            return {"status": "pending_human_confirmation"}

        response_payload = confirmation.payload if isinstance(confirmation.payload, dict) else {}
        decision = str(response_payload.get("decision", "")).strip().lower()
        feedback = str(response_payload.get("feedback", "")).strip()
        if REQUIRE_EXPLICIT_EVIDENCE_CHECKPOINT_DECISION and not decision:
            tool_context.request_confirmation(
                hint=(
                    "Explicit decision is required: `approve` to continue evidence collection "
                    "or `stop` to synthesize from current evidence."
                ),
                payload=payload,
            )
            tool_context.actions.skip_summarization = True
            tool_context.state[pending_key] = True
            return {
                "status": "pending_human_confirmation",
                "message": "Confirmation did not include an explicit decision.",
            }
        tool_context.state[pending_key] = False
        if not decision:
            decision = "stop"

        is_approved = bool(confirmation.confirmed) and decision in {
            "approve",
            "approved",
            "continue",
            "yes",
            "y",
        }
        if is_approved:
            return {
                "status": "approved",
                "feedback": feedback,
            }

        exit_loop(tool_context=tool_context)
        return {
            "status": "declined",
            "feedback": feedback or "User chose to stop additional evidence collection.",
        }

    return request_evidence_continuation


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
    max_evidence_iterations: int | None = None,
    prefer_bigquery: bool | None = None,
) -> tuple[SequentialAgent, McpToolset | None]:
    """Create an ADK-native workflow graph and return (root_agent, mcp_toolset)."""
    del max_plan_iterations

    runtime_model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    evidence_max_iters = (
        max(1, int(max_evidence_iterations))
        if max_evidence_iterations is not None
        else DEFAULT_EVIDENCE_MAX_ITERS
    )
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

    request_evidence_continuation = _make_request_evidence_continuation_tool()
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
    evidence_critic = LlmAgent(
        name="evidence_critic",
        model=runtime_model,
        instruction=EVIDENCE_CRITIC_INSTRUCTION,
        tools=[exit_loop],
    )
    evidence_hitl_gate = LlmAgent(
        name="evidence_hitl_gate",
        model=runtime_model,
        instruction=EVIDENCE_HITL_GATE_INSTRUCTION,
        tools=[request_evidence_continuation],
        generate_content_config=types.GenerateContentConfig(
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["request_evidence_continuation"],
                )
            )
        ),
    )
    evidence_loop = LoopAgent(
        name="evidence_refinement_loop",
        description="Collect evidence and iterate until critic exits the loop.",
        sub_agents=[evidence_executor, evidence_critic, evidence_hitl_gate],
        max_iterations=evidence_max_iters,
    )
    report_synthesizer = LlmAgent(
        name="report_synthesizer",
        model=runtime_model,
        instruction=SYNTHESIZER_INSTRUCTION,
        tools=[],
    )

    root = SequentialAgent(
        name="co_scientist_workflow",
        description="ADK-native biomedical workflow: planner, evidence refinement, synthesis.",
        sub_agents=[planner, evidence_loop, report_synthesizer],
    )
    return root, mcp_toolset


__all__ = [
    "create_mcp_toolset",
    "create_workflow_agent",
]
