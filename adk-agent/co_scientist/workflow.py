"""
ADK-native orchestration graph for the Co-Scientist agent.

This module defines a workflow-agent tree (Sequential + Loop + role LLM agents)
so orchestration is handled by ADK instead of custom Python control loops.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.genai import types
from google.adk.tools import McpToolset, ToolContext, exit_loop
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from mcp.client.stdio import StdioServerParameters
from pydantic import BaseModel, Field


MCP_SERVER_DIR = Path(__file__).resolve().parents[2] / "research-mcp"
DEFAULT_MODEL = os.getenv("ADK_NATIVE_MODEL", "gemini-2.5-flash")
DEFAULT_PLAN_MAX_ITERS = max(
    1,
    int(os.getenv("ADK_NATIVE_PLAN_MAX_ITERS", "1") or "1"),
)
DEFAULT_EVIDENCE_MAX_ITERS = max(
    1,
    int(os.getenv("ADK_NATIVE_EVIDENCE_MAX_ITERS", "3") or "3"),
)
REQUIRE_EXPLICIT_CONFIRMATION_DECISION = str(
    os.getenv("ADK_NATIVE_REQUIRE_CONFIRMATION_DECISION", "1")
).strip().lower() not in {"0", "false", "no"}
REQUEST_CONFIRMATION_FUNCTION_NAME = "adk_request_confirmation"

KNOWN_MCP_TOOLS = [
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


class PlanDraft(BaseModel):
    objective: str = Field(description="Normalized research objective for execution.")
    steps: list[str] = Field(
        default_factory=list,
        description="Ordered execution steps the workflow should follow.",
    )
    planned_tools: list[str] = Field(
        default_factory=list,
        description="Exact MCP tool names expected to be used in evidence execution.",
    )
    evidence_sources: list[str] = Field(
        default_factory=list,
        description="External source families expected in evidence collection.",
    )
    stop_conditions: list[str] = Field(
        default_factory=list,
        description="Concrete conditions for when evidence collection is sufficient.",
    )


CLARIFIER_INSTRUCTION = """
You are a biomedical scope clarifier.
Goal: identify ambiguity, malformed identifiers, or missing constraints before planning.

Output 3 short bullets:
- normalized objective
- ambiguity flags (or "none")
- critical constraints (or "none")

Keep this concise and operational. Do not call tools.
"""


PLANNER_INSTRUCTION_TEMPLATE = """
You are the workflow planner for biomedical investigation.
Use the clarifier summary: {clarifier_summary}
Optional revision feedback from the human reviewer: {plan_revision_feedback?}

Available MCP tools:
__TOOL_CATALOG__

Return ONLY JSON matching the output schema.

Rules:
- Build a practical, objective-specific plan with concrete steps.
- `planned_tools` MUST use exact names from the available tool list when possible.
- Include only tools needed for the proposed steps.
- If revision feedback is present, incorporate it explicitly into updated steps.
- Keep each list concise (typically <= 8 items).
"""


PLAN_REVIEWER_INSTRUCTION = """
You are the plan approval gatekeeper.

Inputs:
- Clarifier summary: {clarifier_summary}
- Current plan JSON: {plan_draft_json}

Actions:
1) Call `request_plan_confirmation` exactly once with `plan_draft_json`.
2) Read the tool result status and return it.

Output:
- Return a short status line only (PLAN_APPROVED, PLAN_REVISION_REQUESTED, or PLAN_AWAITING_CONFIRMATION).
"""


EVIDENCE_EXECUTOR_INSTRUCTION = """
You execute evidence collection and validation.
Use this approved plan context: {plan_summary?}
Approved tools list: {approved_tools?}

Rules:
- Use MCP tools only when directly required by the approved plan.
- Prioritize high-signal evidence before broad expansion.
- Surface contradictions and unresolved gaps explicitly.
- Include source identifiers when available (PMID, DOI, NCT, OpenAlex IDs).

Return only:
1) evidence gathered (bullet list),
2) unresolved gaps (bullet list),
3) what to probe next if gaps remain (bullet list).
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
Use these internal summaries:
- Clarifier: {clarifier_summary}
- Planner: {plan_summary?}
- Critic: {evidence_critic_decision}

Write a concise final answer:
1) Direct answer up front.
2) Key supporting evidence with citations/IDs.
3) Explicit limitations and uncertainty.
4) Practical next actions.

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


def _parse_plan_draft_json(
    plan_draft_json: Any,
    *,
    allowed_tool_names: set[str],
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    if isinstance(plan_draft_json, dict):
        parsed = plan_draft_json
    elif isinstance(plan_draft_json, BaseModel):
        payload = plan_draft_json.model_dump(mode="json")
        if isinstance(payload, dict):
            parsed = payload
    else:
        raw = str(plan_draft_json or "").strip()
        if raw:
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    parsed = payload
            except json.JSONDecodeError:
                parsed = {}

    objective = str(parsed.get("objective", "")).strip() or "Investigate the user objective with traceable evidence."
    steps = _dedupe_str_list(parsed.get("steps", []) if isinstance(parsed.get("steps"), list) else [], limit=10)
    planned_tools = _dedupe_str_list(
        parsed.get("planned_tools", []) if isinstance(parsed.get("planned_tools"), list) else [],
        limit=16,
    )
    if allowed_tool_names:
        planned_tools = [tool for tool in planned_tools if tool in allowed_tool_names]
        allowed_order = [name for name in KNOWN_MCP_TOOLS if name in allowed_tool_names]
        allowed_order.extend(sorted(name for name in allowed_tool_names if name not in set(allowed_order)))
        preferred_fallback = [
            "search_pubmed",
            "search_pubmed_advanced",
            "search_clinical_trials",
            "search_disease_targets",
            "search_targets",
            "search_diseases",
        ]
        min_tools = min(2, len(allowed_tool_names))
        for candidate in preferred_fallback + allowed_order:
            if len(planned_tools) >= min_tools:
                break
            if candidate in allowed_tool_names and candidate not in planned_tools:
                planned_tools.append(candidate)
    evidence_sources = _dedupe_str_list(
        parsed.get("evidence_sources", []) if isinstance(parsed.get("evidence_sources"), list) else [],
        limit=10,
    )
    stop_conditions = _dedupe_str_list(
        parsed.get("stop_conditions", []) if isinstance(parsed.get("stop_conditions"), list) else [],
        limit=10,
    )

    if not steps:
        steps = [
            "Resolve objective scope and evidence constraints.",
            "Collect high-signal evidence with approved MCP tools.",
            "Synthesize findings with explicit uncertainty and next actions.",
        ]
    if not stop_conditions:
        stop_conditions = [
            "Core claims are grounded with citations or identifiers.",
            "Major contradictions are addressed or clearly called out.",
        ]
    return {
        "objective": objective,
        "steps": steps,
        "planned_tools": planned_tools,
        "evidence_sources": evidence_sources,
        "stop_conditions": stop_conditions,
    }


def _render_plan_summary(plan: dict[str, Any]) -> str:
    lines = [f"Objective: {plan.get('objective', '').strip()}"]
    steps = [str(item).strip() for item in plan.get("steps", []) if str(item).strip()]
    if steps:
        lines.append("Steps:")
        lines.extend([f"- {item}" for item in steps[:10]])
    tools = [str(item).strip() for item in plan.get("planned_tools", []) if str(item).strip()]
    if tools:
        lines.append("Planned tools:")
        lines.extend([f"- {name}" for name in tools[:16]])
    sources = [str(item).strip() for item in plan.get("evidence_sources", []) if str(item).strip()]
    if sources:
        lines.append("Evidence sources:")
        lines.extend([f"- {item}" for item in sources[:10]])
    stops = [str(item).strip() for item in plan.get("stop_conditions", []) if str(item).strip()]
    if stops:
        lines.append("Stop conditions:")
        lines.extend([f"- {item}" for item in stops[:10]])
    return "\n".join(lines).strip()


def _record_plan_review(tool_context: ToolContext, entry: dict[str, Any]) -> None:
    history = tool_context.state.get("plan_review_history", [])
    if not isinstance(history, list):
        history = []
    normalized_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    history.append(normalized_entry)
    tool_context.state["plan_review_history"] = history[-20:]


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


def _parse_manual_plan_decision(tool_context: ToolContext) -> tuple[str, str]:
    turn_text = _extract_user_turn_text(tool_context)
    lowered = turn_text.lower().strip()
    if lowered in {"approve", "approved", "yes", "y", "/approve"}:
        return "approve", ""
    if lowered.startswith("revise:"):
        feedback = turn_text.split(":", 1)[1].strip()
        if feedback:
            return "revise", feedback
    if lowered in {"revise", "reject", "rejected", "no", "n", "/revise"}:
        return "revise", ""
    return "", ""


def _prepare_plan_gate_for_new_query(callback_context: CallbackContext):
    content = callback_context.user_content
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return None

    is_confirmation_resume = False
    text_chunks: list[str] = []
    for part in parts:
        function_response = getattr(part, "function_response", None)
        if function_response is not None:
            name = str(getattr(function_response, "name", "")).strip()
            if name == REQUEST_CONFIRMATION_FUNCTION_NAME:
                is_confirmation_resume = True
        text = str(getattr(part, "text", "") or "").strip()
        if text:
            text_chunks.append(text)

    turn_text = re.sub(r"\s+", " ", " ".join(text_chunks)).strip()
    lowered = turn_text.lower()
    is_manual_decision = (
        lowered in {"approve", "approved", "yes", "y", "/approve", "revise", "reject", "rejected", "no", "n", "/revise"}
        or lowered.startswith("revise:")
    )
    if is_confirmation_resume or is_manual_decision:
        return None

    invocation_id = str(getattr(callback_context, "invocation_id", "") or "").strip()
    last_reset_invocation = str(callback_context.state.get("plan_gate_last_reset_invocation", "") or "").strip()
    if invocation_id and invocation_id == last_reset_invocation:
        return None

    callback_context.state["plan_approval_status"] = "pending_human_confirmation"
    callback_context.state["approved_tools"] = []
    callback_context.state["pending_plan_confirmation_payload"] = {}
    callback_context.state["plan_summary"] = ""
    callback_context.state["approved_plan_summary"] = ""
    callback_context.state["plan_revision_feedback"] = ""
    if invocation_id:
        callback_context.state["plan_gate_last_reset_invocation"] = invocation_id
    return None


def _make_request_plan_confirmation_tool(*, allowed_tool_names: list[str]):
    allowed_names = {name for name in _dedupe_str_list(allowed_tool_names, limit=120)}

    def request_plan_confirmation(plan_draft_json: Any, tool_context: ToolContext) -> dict[str, Any]:
        """Request human confirmation for the generated execution plan."""
        plan = _parse_plan_draft_json(
            plan_draft_json,
            allowed_tool_names=allowed_names,
        )
        rendered_summary = _render_plan_summary(plan)
        approval_payload = {
            "schema": "execution_plan_confirmation.v1",
            "objective": plan["objective"],
            "steps": plan["steps"],
            "proposed_tools": plan["planned_tools"],
            "evidence_sources": plan["evidence_sources"],
            "stop_conditions": plan["stop_conditions"],
            "response_contract": {
                "decision": "approve | revise",
                "feedback": "Required when decision is revise.",
            },
        }
        pending_payload_state = tool_context.state.get("pending_plan_confirmation_payload", {})
        if not tool_context.tool_confirmation:
            manual_decision, manual_feedback = _parse_manual_plan_decision(tool_context)
            if (
                manual_decision
                and isinstance(pending_payload_state, dict)
                and bool(pending_payload_state)
            ):
                pending_plan_input = {
                    "objective": pending_payload_state.get("objective", ""),
                    "steps": pending_payload_state.get("steps", []),
                    "planned_tools": pending_payload_state.get("proposed_tools", []),
                    "evidence_sources": pending_payload_state.get("evidence_sources", []),
                    "stop_conditions": pending_payload_state.get("stop_conditions", []),
                }
                plan = _parse_plan_draft_json(
                    pending_plan_input,
                    allowed_tool_names=allowed_names,
                )
                rendered_summary = _render_plan_summary(plan)
                approval_payload = {
                    "schema": "execution_plan_confirmation.v1",
                    "objective": plan["objective"],
                    "steps": plan["steps"],
                    "proposed_tools": plan["planned_tools"],
                    "evidence_sources": plan["evidence_sources"],
                    "stop_conditions": plan["stop_conditions"],
                    "response_contract": {
                        "decision": "approve | revise",
                        "feedback": "Required when decision is revise.",
                    },
                }
            if manual_decision == "approve":
                approved_tools = _dedupe_str_list(plan.get("planned_tools", []), limit=24)
                tool_context.state["plan_approval_status"] = "approved"
                tool_context.state["approved_tools"] = approved_tools
                tool_context.state["plan_revision_feedback"] = ""
                tool_context.state["plan_summary"] = rendered_summary
                tool_context.state["approved_plan_summary"] = rendered_summary
                tool_context.state["pending_plan_confirmation_payload"] = {}
                _record_plan_review(
                    tool_context,
                    {
                        "decision": "approved_manual_turn",
                        "feedback": "",
                        "approved_tools": approved_tools,
                    },
                )
                return {
                    "status": "approved",
                    "approved_tools": approved_tools,
                    "plan_summary": rendered_summary,
                }
            if manual_decision == "revise":
                revision_feedback = manual_feedback or "Revise the plan before evidence tool execution."
                tool_context.state["plan_approval_status"] = "revision_requested"
                tool_context.state["plan_revision_feedback"] = revision_feedback
                tool_context.state["approved_tools"] = []
                tool_context.state["pending_plan_confirmation_payload"] = {}
                _record_plan_review(
                    tool_context,
                    {
                        "decision": "revision_requested_manual_turn",
                        "feedback": revision_feedback,
                        "approved_tools": [],
                    },
                )
                return {
                    "status": "revision_requested",
                    "feedback": revision_feedback,
                    "plan_summary": rendered_summary,
                }

            tool_context.request_confirmation(
                hint=(
                    "Review the proposed execution plan and tool list. "
                    "Approve to continue, or revise with feedback. "
                    "Fallback: send a normal user turn with `approve` or `revise: <feedback>`."
                ),
                payload=approval_payload,
            )
            tool_context.actions.skip_summarization = True
            tool_context.state["plan_approval_status"] = "pending_human_confirmation"
            tool_context.state["plan_summary"] = rendered_summary
            tool_context.state["pending_plan_confirmation_payload"] = approval_payload
            return {
                "status": "pending_human_confirmation",
                "plan_summary": rendered_summary,
            }

        confirmation = tool_context.tool_confirmation
        payload = confirmation.payload if isinstance(confirmation.payload, dict) else {}
        decision = str(payload.get("decision", "")).strip().lower()
        feedback = str(payload.get("feedback", "")).strip()
        if REQUIRE_EXPLICIT_CONFIRMATION_DECISION and bool(confirmation.confirmed) and not decision:
            tool_context.request_confirmation(
                hint=(
                    "Explicit decision is required. Confirm with `decision=approve` "
                    "or request revision with `decision=revise` and feedback. "
                    "Fallback: send `approve` or `revise: <feedback>` as a normal user turn."
                ),
                payload=approval_payload,
            )
            tool_context.actions.skip_summarization = True
            tool_context.state["plan_approval_status"] = "pending_human_confirmation"
            tool_context.state["approved_tools"] = []
            tool_context.state["plan_summary"] = rendered_summary
            tool_context.state["pending_plan_confirmation_payload"] = approval_payload
            _record_plan_review(
                tool_context,
                {
                    "decision": "confirmation_missing_decision",
                    "feedback": "",
                    "approved_tools": [],
                },
            )
            return {
                "status": "pending_human_confirmation",
                "plan_summary": rendered_summary,
                "message": "Confirmation did not include an explicit decision.",
            }

        if not decision:
            decision = "approve" if bool(confirmation.confirmed) else "revise"
        is_approved = bool(confirmation.confirmed) and decision in {
            "approve",
            "approved",
            "accept",
            "accepted",
            "yes",
        }

        if is_approved:
            approved_tools = _dedupe_str_list(plan.get("planned_tools", []), limit=24)
            tool_context.state["plan_approval_status"] = "approved"
            tool_context.state["approved_tools"] = approved_tools
            tool_context.state["plan_revision_feedback"] = ""
            tool_context.state["plan_summary"] = rendered_summary
            tool_context.state["approved_plan_summary"] = rendered_summary
            tool_context.state["pending_plan_confirmation_payload"] = {}
            _record_plan_review(
                tool_context,
                {
                    "decision": "approved",
                    "feedback": "",
                    "approved_tools": approved_tools,
                },
            )
            return {
                "status": "approved",
                "approved_tools": approved_tools,
                "plan_summary": rendered_summary,
            }

        revision_feedback = feedback or "Revise the plan before evidence tool execution."
        tool_context.state["plan_approval_status"] = "revision_requested"
        tool_context.state["plan_revision_feedback"] = revision_feedback
        tool_context.state["approved_tools"] = []
        tool_context.state["pending_plan_confirmation_payload"] = {}
        _record_plan_review(
            tool_context,
            {
                "decision": "revision_requested",
                "feedback": revision_feedback,
                "approved_tools": [],
            },
        )
        return {
            "status": "revision_requested",
            "feedback": revision_feedback,
            "plan_summary": rendered_summary,
        }

    return request_plan_confirmation


def _build_planner_instruction(tool_hints: list[str]) -> str:
    tool_catalog = "\n".join(f"- {name}" for name in tool_hints[:80]) or "- No tools available."
    return PLANNER_INSTRUCTION_TEMPLATE.replace("__TOOL_CATALOG__", tool_catalog)


def _guard_evidence_tool_execution(tool, args: dict[str, Any], tool_context: ToolContext) -> dict | None:
    del args
    approval_status = str(tool_context.state.get("plan_approval_status", "")).strip().lower()
    if approval_status != "approved":
        tool_context.actions.skip_summarization = True
        return {
            "status": "blocked",
            "reason": "plan_not_approved",
            "message": "Evidence tools are blocked until the human approves the execution plan.",
        }

    approved_tools = _dedupe_str_list(tool_context.state.get("approved_tools", []), limit=40)
    if not approved_tools:
        tool_context.actions.skip_summarization = True
        return {
            "status": "blocked",
            "reason": "no_tools_approved",
            "message": "No tools were approved in the execution plan.",
        }

    if str(getattr(tool, "name", "")).strip() not in set(approved_tools):
        tool_context.actions.skip_summarization = True
        return {
            "status": "blocked",
            "reason": "tool_not_approved",
            "tool": str(getattr(tool, "name", "")).strip(),
            "approved_tools": approved_tools,
        }
    return None


def _render_pending_approval_message(state: dict[str, Any]) -> str:
    payload = state.get("pending_plan_confirmation_payload", {})
    if not isinstance(payload, dict):
        payload = {}
    if not payload:
        fallback_plan = _parse_plan_draft_json(
            state.get("plan_draft_json", ""),
            allowed_tool_names=set(KNOWN_MCP_TOOLS),
        )
        payload = {
            "objective": fallback_plan.get("objective", ""),
            "steps": fallback_plan.get("steps", []),
            "proposed_tools": fallback_plan.get("planned_tools", []),
        }

    lines = [
        "Plan approval is required before evidence tools can run.",
        "Review and confirm the proposed plan in your client.",
    ]
    objective = str(payload.get("objective", "")).strip()
    if objective:
        lines.append(f"\nObjective: {objective}")
    steps = [str(item).strip() for item in payload.get("steps", []) if str(item).strip()]
    if steps:
        lines.append("\nProposed steps:")
        lines.extend([f"{idx}. {step}" for idx, step in enumerate(steps[:8], start=1)])
    tools = [str(item).strip() for item in payload.get("proposed_tools", []) if str(item).strip()]
    if tools:
        lines.append("\nProposed tools:")
        lines.extend([f"- {name}" for name in tools[:16]])

    lines.append(
        "\nIf your UI does not support confirmation popups yet, reply in chat "
        "with `approve` or `revise: <feedback>`. You can also continue in CLI "
        "(`python agent.py`) for interactive confirmation prompts."
    )
    return "\n".join(lines).strip()


def _block_until_plan_approved(callback_context: CallbackContext):
    status = str(callback_context.state.get("plan_approval_status", "")).strip().lower()
    if status == "approved":
        return None

    message = _render_pending_approval_message(callback_context.state)
    return types.Content(
        role="model",
        parts=[types.Part(text=message)],
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


def create_native_workflow_agent(
    *,
    tool_filter: list[str] | None = None,
    model: str | None = None,
    max_plan_iterations: int | None = None,
    max_evidence_iterations: int | None = None,
) -> tuple[SequentialAgent, McpToolset | None]:
    """Create an ADK-native workflow graph and return (root_agent, mcp_toolset)."""
    runtime_model = str(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    plan_max_iters = (
        max(1, int(max_plan_iterations))
        if max_plan_iterations is not None
        else DEFAULT_PLAN_MAX_ITERS
    )
    evidence_max_iters = (
        max(1, int(max_evidence_iterations))
        if max_evidence_iterations is not None
        else DEFAULT_EVIDENCE_MAX_ITERS
    )

    mcp_toolset = create_mcp_toolset(tool_filter=tool_filter)
    executor_tools = [mcp_toolset] if mcp_toolset is not None else []
    planner_tool_hints = _dedupe_str_list(tool_filter if tool_filter else KNOWN_MCP_TOOLS, limit=120)
    request_plan_confirmation = _make_request_plan_confirmation_tool(
        allowed_tool_names=planner_tool_hints,
    )

    clarifier = LlmAgent(
        name="clarifier",
        model=runtime_model,
        instruction=CLARIFIER_INSTRUCTION,
        tools=[],
        include_contents="none",
        output_key="clarifier_summary",
    )
    planner = LlmAgent(
        name="planner",
        model=runtime_model,
        instruction=_build_planner_instruction(planner_tool_hints),
        tools=[],
        before_agent_callback=_prepare_plan_gate_for_new_query,
        include_contents="none",
        output_schema=PlanDraft,
        output_key="plan_draft_json",
    )
    plan_reviewer = LlmAgent(
        name="plan_reviewer",
        model=runtime_model,
        instruction=PLAN_REVIEWER_INSTRUCTION,
        tools=[request_plan_confirmation],
        generate_content_config=types.GenerateContentConfig(
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["request_plan_confirmation"],
                )
            )
        ),
        include_contents="none",
        output_key="plan_review_status",
    )
    plan_approval_loop = LoopAgent(
        name="plan_approval_loop",
        description="Iterate planning until a human approves the execution plan.",
        sub_agents=[planner, plan_reviewer],
        max_iterations=plan_max_iters,
    )
    evidence_executor = LlmAgent(
        name="evidence_executor",
        model=runtime_model,
        instruction=EVIDENCE_EXECUTOR_INSTRUCTION,
        tools=executor_tools,
        before_agent_callback=_block_until_plan_approved,
        before_tool_callback=_guard_evidence_tool_execution,
        output_key="evidence_iteration_summary",
    )
    evidence_critic = LlmAgent(
        name="evidence_critic",
        model=runtime_model,
        instruction=EVIDENCE_CRITIC_INSTRUCTION,
        tools=[exit_loop],
        include_contents="none",
        output_key="evidence_critic_decision",
    )
    evidence_loop = LoopAgent(
        name="evidence_refinement_loop",
        description="Collect evidence and iterate until critic exits the loop.",
        sub_agents=[evidence_executor, evidence_critic],
        max_iterations=evidence_max_iters,
        before_agent_callback=_block_until_plan_approved,
    )
    report_synthesizer = LlmAgent(
        name="report_synthesizer",
        model=runtime_model,
        instruction=SYNTHESIZER_INSTRUCTION,
        tools=[],
        before_agent_callback=_block_until_plan_approved,
        output_key="final_report",
    )

    root = SequentialAgent(
        name="co_scientist_workflow",
        description="ADK-native biomedical workflow: clarify, plan+approve, refine evidence, synthesize.",
        sub_agents=[clarifier, plan_approval_loop, evidence_loop, report_synthesizer],
    )
    return root, mcp_toolset


__all__ = [
    "create_mcp_toolset",
    "create_native_workflow_agent",
]
