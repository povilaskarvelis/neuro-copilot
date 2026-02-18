"""
AI Co-Scientist Agent

This ADK agent connects to the research-mcp server to provide
drug target discovery capabilities using Gemini.

Usage:
    python agent.py         # Interactive mode
    python agent.py --help  # Show help

Requirements:
    - Node.js (for the MCP server)
    - Google API key in .env file
    - pip install -r requirements.txt

Setup:
    1. Edit .env file and paste your API key
    2. Run: python agent.py
"""
import os
import asyncio
from pathlib import Path
import traceback
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Suppress noisy SDK warnings about non-text parts in tool-heavy responses.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

from google.adk import Agent, Runner
from google.adk.tools import McpToolset
from mcp.client.stdio import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from report_pdf import write_markdown_pdf
from task_state_store import TaskStateStore
from co_scientist.planning import intent as _planning_intent
from co_scientist.planning import revision as _planning_revision
from co_scientist.planning import workflow_planning as _workflow_planning
from co_scientist.presentation import cli_output as _presentation_cli
from co_scientist.presentation import hitl_summary as _presentation_hitl
from co_scientist.runtime import execution as _runtime_exec
from co_scientist.runtime import event_orchestrator as _runtime_events
from co_scientist.runtime import quality_gates as _runtime_quality
from co_scientist.runtime.tool_registry import ToolRegistry
from workflow import (
    RevisionIntent,
    WorkflowTask,
    active_plan_version,
    classify_request_type,
    create_task,
    extract_evidence_refs,
    infer_intent_tags,
    initialize_plan_version,
    replan_remaining_steps,
    render_final_report,
    render_status,
    sanitize_intent_tags,
    sanitize_request_type,
    step_prompt,
)

# Path to the MCP server
MCP_SERVER_DIR = Path(__file__).parent.parent / "research-mcp"
REPORT_ARTIFACTS_DIR = Path(__file__).resolve().parent / "reports"
STEP_TURN_TIMEOUT_SECONDS = float(os.getenv("ADK_STEP_TURN_TIMEOUT_SECONDS", "150"))
DYNAMIC_TOOL_REGISTRY_ENABLED = os.getenv("ADK_DYNAMIC_TOOL_REGISTRY", "1").strip().lower() not in {"0", "false", "no"}
DYNAMIC_PLANNER_GRAPH_ENABLED = os.getenv("ADK_DYNAMIC_PLANNER_GRAPH", "1").strip().lower() not in {"0", "false", "no"}
DYNAMIC_TOOL_RETRIEVAL_ENABLED = os.getenv("ADK_DYNAMIC_TOOL_RETRIEVAL", "1").strip().lower() not in {"0", "false", "no"}
CRITIC_LOOP_QUALITY_MODEL_ENABLED = os.getenv("ADK_CRITIC_LOOP_QUALITY_MODEL", "1").strip().lower() not in {"0", "false", "no"}


AGENT_INSTRUCTION = """You are an agentic AI research assistant operating like a high-level research intern.
Think in iterative Plan-Act-Reflect cycles.
Use tools to gather evidence, keep provenance, and state uncertainty explicitly.
After each major phase, pause and ask the user for confirmation before continuing.
Always end with a decision-ready final report grounded in captured evidence."""

PLANNER_PROMPT = """You are the planning role for an agentic scientific workflow.
Generate query-specific subgoals, dependencies, and evidence requirements.
Avoid rigid templates when a custom decomposition is better."""

PLANNER_GRAPH_SCHEMA_PROMPT = """Return strict JSON only with this schema:
{
  "subgoals": [
    {
      "subgoal_id": string,
      "title": string,
      "objective": string,
      "dependencies": [string],
      "evidence_requirements": [string],
      "done_criteria": [string],
      "max_calls": number,
      "phase": "evidence_discovery" | "researcher_scouting" | "synthesis_reporting"
    }
  ]
}

Rules:
- Build a custom plan for this specific query; do not use generic template naming.
- Keep 2-6 subgoals.
- Dependencies must reference earlier subgoal_id values only.
- Avoid including a final-report step; final synthesis is added by runtime guardrail.
- Use concise, actionable objectives with explicit evidence intent.
"""

EXECUTOR_PROMPT = """You are the execution role for an agentic scientific workflow.
Use only currently allowed tools, gather evidence efficiently, and report residual uncertainty."""

CRITIC_PROMPT = """You are the critic/verifier role for an agentic scientific workflow.
Assess evidence sufficiency, contradictions, and confidence calibration before final recommendation."""

REPORT_SYNTHESIZER_PROMPT = """You are the reporting role for an agentic scientific workflow.
Produce a decision-ready final report with explicit recommendation, rationale, limitations, and next actions."""

_TOOL_REGISTRY = ToolRegistry()

CLARIFIER_INSTRUCTION = """You are an ambiguity and typo triage assistant for biomedical queries.
Your only job is to decide if clarification is needed BEFORE any research tools run.

Return strict JSON only with this schema:
{
  "needs_clarification": boolean,
  "confidence": number,
  "questions": [string],
  "reason": string
}

Rules:
- Ask clarification for ambiguous abbreviations/acronyms (e.g., ER, AD, PD, MS, RA) when context does not disambiguate.
- Ask clarification when a key biomedical entity likely contains a typo or malformed identifier that could change interpretation.
- Do NOT ask clarification for minor spelling mistakes when intent is still clear from context.
- Do not ask clarification for harmless wording/style issues.
- Keep questions concise and actionable.
- If no clarification is needed, set questions to [].
"""

INTENT_ROUTER_INSTRUCTION = """You are an intent router for a biomedical co-scientist workflow.
Your output configures planning behavior only.

Return strict JSON only with this schema:
{
  "normalized_query": string,
  "request_type": "comparison" | "prioritization" | "validation" | "action_planning" | "exploration",
  "intent_tags": [string],
  "confidence": number,
  "reason": string
}

Rules:
- Preserve user meaning while lightly correcting obvious typos in normalized_query.
- intent_tags must come only from this set:
  researcher_discovery, evidence_landscape, variant_check, pathway_context,
  clinical_landscape, chemistry_evidence, ontology_expansion, expression_context,
  genetics_direction, safety_assessment, competitive_landscape, comparison,
  prioritization, target_comparison.
- For researcher ranking requests, prefer researcher_discovery and prioritization.
- Use target_comparison only when the request is explicitly about comparing/ranking targets.
- confidence must be between 0 and 1.
- Never include tool names or extra keys.
"""

REVISION_FEEDBACK_PARSER_INSTRUCTION = """You convert human feedback at a workflow checkpoint into structured intent updates.
Return strict JSON only with this schema:
{
  "objective_adjustments": [string],
  "constraints": [string],
  "priorities": [string],
  "exclusions": [string],
  "evidence_preferences": [string],
  "output_preferences": [string],
  "confidence": number
}

Rules:
- Do not mention tools unless the user explicitly named them.
- Keep each item concise and action-oriented.
- If a field has no signal, return [] for that field.
- confidence must be between 0 and 1.
"""

CHAT_TITLE_SUMMARIZER_INSTRUCTION = """You generate concise chat titles for biomedical research requests.
Return strict JSON only with this schema:
{
  "title": string
}

Rules:
- The title must be <= 8 words.
- Preserve key disease/target/drug entities.
- Focus on objective, not conversational filler.
- Do not include quotes, markdown, prefixes, or trailing punctuation.
"""

PROGRESS_SUMMARIZER_INSTRUCTION = """You summarize agent progress for end users.
Return strict JSON only with this schema:
{
  "headline": string,
  "summary": string,
  "completed": [string],
  "next": [string],
  "confidence": "low" | "medium" | "high"
}

Rules:
- Use only observable actions and outcomes from provided events.
- Do not include private reasoning or chain-of-thought.
- headline: <= 12 words.
- summary: 1-2 short sentences.
- completed: max 3 bullets.
- next: max 2 bullets.
"""

QUERY_TYPO_REPLACEMENTS: tuple[tuple[str, str], ...] = _planning_intent.QUERY_TYPO_REPLACEMENTS


AMBIGUOUS_ABBREVIATIONS: dict[str, dict[str, list[str]]] = _planning_intent.AMBIGUOUS_ABBREVIATIONS


def _find_ambiguous_abbreviations(query: str) -> list[tuple[str, list[str]]]:
    return _planning_intent.find_ambiguous_abbreviations(query)


def _merge_query_with_clarification(original_query: str, clarification: str) -> str:
    return _planning_intent.merge_query_with_clarification(original_query, clarification)


def _extract_revision_directives(revised_scope: str) -> list[str]:
    return _planning_revision.extract_revision_directives(revised_scope)


def _merge_objective_with_revision(original_objective: str, revised_scope: str) -> str:
    return _planning_revision.merge_objective_with_revision(original_objective, revised_scope)


def _extract_revision_directive_from_objective(objective: str) -> str | None:
    return _planning_revision.extract_revision_directive_from_objective(objective)


def _extract_timeframe_hint(text: str) -> str | None:
    return _planning_revision.extract_timeframe_hint(text)


def _extract_primary_objective_text(objective: str) -> str:
    return _planning_revision.extract_primary_objective_text(objective)


def _clean_model_text(value: str) -> str:
    return _presentation_hitl.clean_model_text(value)


def _extract_labeled_value(text: str, labels: list[str]) -> str | None:
    return _presentation_hitl.extract_labeled_value(text, labels)


def _extract_decomposition_subtasks(text: str) -> list[str]:
    return _presentation_hitl.extract_decomposition_subtasks(text)


def _default_hitl_subtasks(task: WorkflowTask) -> list[str]:
    return _presentation_hitl.default_hitl_subtasks(task)


def _extract_focus_from_step_output(step_output: str) -> str | None:
    return _presentation_hitl.extract_focus_from_step_output(step_output)


def _render_hitl_scope_summary(task: WorkflowTask, step_output: str) -> str:
    return _presentation_hitl.render_hitl_scope_summary(task, step_output)


def _extract_json_payload(text: str) -> dict | None:
    return _planning_intent.extract_json_payload(text)


def _contains_malformed_biomedical_identifier(query: str) -> bool:
    return _planning_intent.contains_malformed_biomedical_identifier(query)


def _looks_like_low_value_typo_clarification(query: str, questions: list[str], reason: str) -> bool:
    return _planning_intent.looks_like_low_value_typo_clarification(query, questions, reason)


async def _build_model_clarification_request(
    clarifier_runner,
    clarifier_session_id: str,
    user_id: str,
    query: str,
) -> str | None:
    return await _planning_intent.build_model_clarification_request(
        clarifier_runner,
        clarifier_session_id,
        user_id,
        query,
        run_runner_turn_fn=_run_runner_turn,
    )


async def _build_clarification_request(
    query: str,
    *,
    clarifier_runner=None,
    clarifier_session_id: str | None = None,
    user_id: str = "researcher",
) -> str | None:
    return await _planning_intent.build_clarification_request(
        query,
        clarifier_runner=clarifier_runner,
        clarifier_session_id=clarifier_session_id,
        user_id=user_id,
        run_runner_turn_fn=_run_runner_turn,
    )


def _dedupe_compact(items: list[str], *, limit: int = 8) -> list[str]:
    return _planning_revision.dedupe_compact(items, limit=limit)


def _coerce_str_list(value) -> list[str]:
    return _planning_revision.coerce_str_list(value)


async def _parse_revision_intent(
    feedback: str,
    *,
    feedback_parser_runner=None,
    feedback_parser_session_id: str | None = None,
    user_id: str = "researcher",
) -> RevisionIntent:
    return await _planning_revision.parse_revision_intent(
        feedback,
        feedback_parser_runner=feedback_parser_runner,
        feedback_parser_session_id=feedback_parser_session_id,
        user_id=user_id,
        run_runner_turn_fn=_run_runner_turn,
    )


def _render_revision_intent_as_text(intent: RevisionIntent) -> str:
    return _planning_revision.render_revision_intent_as_text(intent)


def _merge_objective_with_revision_intent(original_objective: str, intent: RevisionIntent) -> str:
    return _planning_revision.merge_objective_with_revision_intent(original_objective, intent)


def _merge_revision_intents(previous: RevisionIntent | None, incoming: RevisionIntent) -> RevisionIntent:
    return _planning_revision.merge_revision_intents(previous, incoming)


def _normalize_user_query(query: str) -> str:
    return _planning_intent.normalize_user_query(query)


def _coerce_model_intent_tags(value) -> list[str]:
    return _planning_intent.coerce_model_intent_tags(value)


def _default_intent_route(query: str) -> dict:
    return _planning_intent.default_intent_route(query)


async def _build_model_intent_route(
    intent_router_runner,
    intent_router_session_id: str,
    user_id: str,
    query: str,
) -> dict | None:
    return await _planning_intent.build_model_intent_route(
        intent_router_runner,
        intent_router_session_id,
        user_id,
        query,
        run_runner_turn_fn=_run_runner_turn,
    )


def _merge_intent_routes(deterministic_route: dict, model_route: dict | None) -> dict:
    return _planning_intent.merge_intent_routes(deterministic_route, model_route)


async def _route_query_intent(
    query: str,
    *,
    intent_router_runner=None,
    intent_router_session_id: str | None = None,
    user_id: str = "researcher",
) -> dict:
    return await _planning_intent.route_query_intent(
        query,
        intent_router_runner=intent_router_runner,
        intent_router_session_id=intent_router_session_id,
        user_id=user_id,
        run_runner_turn_fn=_run_runner_turn,
        build_model_intent_route_fn=_build_model_intent_route,
    )


async def _draft_model_plan_graph(
    objective: str,
    *,
    request_type: str,
    intent_tags: list[str],
    planner_runner=None,
    planner_session_id: str | None = None,
    user_id: str = "researcher",
) -> list[dict] | None:
    if planner_runner is None or not planner_session_id:
        return None
    tool_summary = _TOOL_REGISTRY.summary(max_tools=80)
    tool_lines = []
    for item in tool_summary[:50]:
        name = str(item.get("name", "")).strip()
        caps = ", ".join(str(cap).strip() for cap in item.get("capabilities", [])[:5] if str(cap).strip())
        if not name:
            continue
        tool_lines.append(f"- {name} ({caps or 'uncategorized'})")
    prompt = (
        f"{PLANNER_PROMPT}\n\n"
        f"{PLANNER_GRAPH_SCHEMA_PROMPT}\n\n"
        f"Objective: {objective}\n"
        f"Request type: {request_type}\n"
        f"Intent tags: {', '.join(intent_tags) if intent_tags else 'none'}\n"
        "Available tools (name + inferred capabilities):\n"
        f"{chr(10).join(tool_lines) if tool_lines else '- none'}\n"
    )
    try:
        raw = await _run_runner_turn(planner_runner, planner_session_id, user_id, prompt)
    except Exception:
        return None
    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        return None
    raw_subgoals = payload.get("subgoals")
    if not isinstance(raw_subgoals, list):
        return None
    normalized = _workflow_planning.normalize_plan_graph(
        raw_subgoals,
        objective=objective,
        intent_tags=intent_tags,
        request_type=request_type,
    )
    return normalized or None


def create_clarifier_agent():
    """Create a no-tool clarification agent for ambiguity/typo triage."""
    return Agent(
        name="clarifier",
        model="gemini-2.5-flash",
        instruction=CLARIFIER_INSTRUCTION,
        tools=[],
    )


def create_intent_router_agent():
    """Create a no-tool intent router for robust query classification."""
    return Agent(
        name="intent_router",
        model="gemini-2.5-flash",
        instruction=INTENT_ROUTER_INSTRUCTION,
        tools=[],
    )


def create_feedback_parser_agent():
    """Create a no-tool parser for checkpoint feedback -> structured revision intent."""
    return Agent(
        name="feedback_parser",
        model="gemini-2.5-flash",
        instruction=REVISION_FEEDBACK_PARSER_INSTRUCTION,
        tools=[],
    )


def create_title_summarizer_agent():
    """Create a no-tool model for concise chat title generation."""
    return Agent(
        name="title_summarizer",
        model="gemini-2.5-flash",
        instruction=CHAT_TITLE_SUMMARIZER_INSTRUCTION,
        tools=[],
    )


def create_progress_summarizer_agent():
    """Create a no-tool model for user-facing checkpoint summaries."""
    return Agent(
        name="progress_summarizer",
        model="gemini-2.5-flash",
        instruction=PROGRESS_SUMMARIZER_INSTRUCTION,
        tools=[],
    )


def create_planner_agent():
    """Create a no-tool planner for dynamic graph drafting."""
    return Agent(
        name="planner",
        model="gemini-2.5-flash",
        instruction=PLANNER_PROMPT,
        tools=[],
    )


def create_critic_agent():
    """Create a no-tool critic/verifier for quality reflection."""
    return Agent(
        name="critic",
        model="gemini-2.5-flash",
        instruction=CRITIC_PROMPT,
        tools=[],
    )


def create_agent(tool_filter: list[str] | None = None):
    """Create the ADK agent with MCP tools."""
    # Configure MCP server connection
    server_params = StdioServerParameters(
        command="node",
        args=["server.js"],
        cwd=str(MCP_SERVER_DIR),
    )
    
    connection_params = StdioConnectionParams(
        server_params=server_params,
        timeout=90.0,
    )

    mcp_tools = None
    agent_tools = []
    if tool_filter is None or len(tool_filter) > 0:
        # Connect to the MCP server. If tool_filter is provided, enforce it for this runner.
        mcp_tools = McpToolset(
            connection_params=connection_params,
            tool_filter=tool_filter,
        )
        agent_tools = [mcp_tools]

    # Create the agent
    agent = Agent(
        name="co_scientist",
        model="gemini-2.5-flash",
        instruction=f"{EXECUTOR_PROMPT}\n\n{AGENT_INSTRUCTION}{_runtime_tool_constraint_suffix(tool_filter)}",
        tools=agent_tools,
    )

    return agent, mcp_tools


STEP_SCOPE_TOOLS = _runtime_exec.STEP_SCOPE_TOOLS


async def _refresh_tool_registry(mcp_tools) -> int:
    if not DYNAMIC_TOOL_REGISTRY_ENABLED:
        return 0
    return await _TOOL_REGISTRY.refresh_from_mcp_toolset(mcp_tools, merge=True)


def _is_reasoning_only_step(task: WorkflowTask, step_idx: int) -> bool:
    return _runtime_exec.is_reasoning_only_step(task, step_idx)


def _build_step_allowed_tools(task: WorkflowTask, step_idx: int) -> list[str]:
    registry = _TOOL_REGISTRY if DYNAMIC_TOOL_RETRIEVAL_ENABLED else None
    return _runtime_exec.build_step_allowed_tools(task, step_idx, tool_registry=registry)


def _should_escalate_allowlist(step, trace_entries: list[dict], output: str) -> bool:
    return _runtime_exec.should_escalate_allowlist(step, trace_entries, output)


def _build_escalated_allowed_tools(task: WorkflowTask, step_idx: int) -> list[str]:
    registry = _TOOL_REGISTRY if DYNAMIC_TOOL_RETRIEVAL_ENABLED else None
    return _runtime_exec.build_escalated_allowed_tools(task, step_idx, tool_registry=registry)


def _create_step_runner(base_runner, allowed_tools: list[str]):
    return _runtime_exec.create_step_runner(base_runner, allowed_tools, create_agent, Runner)


async def _run_runner_turn(runner, session_id: str, user_id: str, prompt: str) -> str:
    return await _runtime_exec.run_runner_turn(
        runner,
        session_id,
        user_id,
        prompt,
        run_runner_turn_with_trace_fn=_run_runner_turn_with_trace,
    )


async def _run_runner_turn_with_timeout(
    runner,
    session_id: str,
    user_id: str,
    prompt: str,
    *,
    timeout_seconds: float | None = None,
) -> tuple[str, list[dict]]:
    return await _runtime_exec.run_runner_turn_with_timeout(
        runner,
        session_id,
        user_id,
        prompt,
        run_runner_turn_with_trace_fn=_run_runner_turn_with_trace,
        default_timeout_seconds=STEP_TURN_TIMEOUT_SECONDS,
        timeout_seconds=timeout_seconds,
    )


def _extract_missing_tool_name(error: Exception) -> str | None:
    return _runtime_exec.extract_missing_tool_name(error)


def _normalize_trace_detail(text: str, *, max_chars: int = 260) -> str:
    return _runtime_exec.normalize_trace_detail(text, max_chars=max_chars)


def _format_step_execution_error(error: Exception, allowed_tools: list[str]) -> str:
    return _runtime_exec.format_step_execution_error(
        error,
        allowed_tools,
        normalize_trace_detail_fn=_normalize_trace_detail,
    )


def _runtime_tool_constraint_suffix(tool_filter: list[str] | None) -> str:
    return _runtime_exec.runtime_tool_constraint_suffix(tool_filter)


def _safe_model_dump(value) -> dict:
    return _runtime_exec.safe_model_dump(value)


def _compact_json(value, *, max_chars: int = 100) -> str:
    return _runtime_exec.compact_json(value, max_chars=max_chars)


def _summarize_for_report(text: str, *, max_chars: int = 220) -> str:
    return _runtime_exec.summarize_for_report(text, max_chars=max_chars)


def _populate_step_rao_fields(step) -> None:
    _runtime_exec.populate_step_rao_fields(
        step,
        summarize_for_report_fn=_summarize_for_report,
        compact_json_fn=_compact_json,
    )


def _extract_response_excerpt(response_payload) -> str:
    return _runtime_exec.extract_response_excerpt(
        response_payload,
        normalize_trace_detail_fn=_normalize_trace_detail,
    )


def _classify_tool_response(response_payload) -> tuple[str, str]:
    return _runtime_exec.classify_tool_response(
        response_payload,
        normalize_trace_detail_fn=_normalize_trace_detail,
        extract_response_excerpt_fn=_extract_response_excerpt,
    )


async def _run_runner_turn_with_trace(
    runner,
    session_id: str,
    user_id: str,
    prompt: str,
) -> tuple[str, list[dict]]:
    return await _runtime_exec.run_runner_turn_with_trace(
        runner,
        session_id,
        user_id,
        prompt,
        safe_model_dump_fn=_safe_model_dump,
        classify_tool_response_fn=_classify_tool_response,
    )


async def _execute_step(runner, session_id: str, user_id: str, task: WorkflowTask, step_idx: int) -> str:
    return await _runtime_exec.execute_step(
        runner,
        session_id,
        user_id,
        task,
        step_idx,
        step_prompt_fn=step_prompt,
        extract_evidence_refs_fn=extract_evidence_refs,
        build_step_allowed_tools_fn=_build_step_allowed_tools,
        create_step_runner_fn=_create_step_runner,
        run_runner_turn_with_timeout_fn=_run_runner_turn_with_timeout,
        extract_missing_tool_name_fn=_extract_missing_tool_name,
        format_step_execution_error_fn=_format_step_execution_error,
        should_escalate_allowlist_fn=_should_escalate_allowlist,
        build_escalated_allowed_tools_fn=_build_escalated_allowed_tools,
        populate_step_rao_fields_fn=_populate_step_rao_fields,
    )


def _evaluate_quality_gates(task: WorkflowTask) -> dict:
    return _runtime_quality.evaluate_quality_gates(task)


def should_open_checkpoint(
    task: WorkflowTask,
    next_step,
    quality_state: dict | None = None,
    queued_feedback: list[str] | None = None,
) -> tuple[bool, str]:
    return _runtime_quality.should_open_checkpoint(
        task,
        next_step,
        quality_state,
        queued_feedback,
        active_plan_version_fn=active_plan_version,
        gate_ack_token_fn=_gate_ack_token,
    )


def _gate_ack_token(reason: str, plan_version_id: str | None) -> str | None:
    return _runtime_quality.gate_ack_token(reason, plan_version_id)


def _render_quality_gate_message(report: dict) -> str:
    return _runtime_quality.render_quality_gate_message(report)


async def _complete_remaining_steps(runner, session_id: str, user_id: str, task: WorkflowTask, state_store: TaskStateStore) -> dict:
    return await _runtime_quality.complete_remaining_steps(
        runner,
        session_id,
        user_id,
        task,
        state_store,
        execute_step_fn=_execute_step,
        evaluate_quality_gates_fn=_evaluate_quality_gates,
        render_quality_gate_message_fn=_render_quality_gate_message,
        print_fn=print,
    )


def _revision_opportunity_used(task: WorkflowTask) -> bool:
    return sum(1 for item in (task.hitl_history or []) if str(item).startswith("revise:")) >= 1


def _format_checkpoint_reason(reason: str) -> str:
    return _runtime_quality.format_checkpoint_reason(reason)


def _print_checkpoint_plan(task: WorkflowTask) -> None:
    _runtime_quality.print_checkpoint_plan(
        task,
        active_plan_version_fn=active_plan_version,
        format_checkpoint_reason_fn=_format_checkpoint_reason,
        print_fn=print,
    )


async def _apply_feedback_replan_cli(
    runner,
    session_id: str,
    user_id: str,
    state_store: TaskStateStore,
    task: WorkflowTask,
    feedback_text: str,
    *,
    intent_router_runner=None,
    intent_router_session_id: str | None = None,
    planner_runner=None,
    planner_session_id: str | None = None,
    feedback_parser_runner=None,
    feedback_parser_session_id: str | None = None,
    gate_reason: str = "feedback_replan",
) -> WorkflowTask:
    if _revision_opportunity_used(task):
        print("\nRevision already used for this task. Continuing with current plan.")
        task.hitl_history.append("revision_limit_reached")
        task.awaiting_hitl = True
        task.checkpoint_state = "open"
        task.checkpoint_reason = "feedback_replan_limit_reached"
        task.touch()
        state_store.save_task(task, note="feedback_replan_limit_reached")
        return task

    if not task.base_objective:
        task.base_objective = task.objective

    parsed_intent = await _parse_revision_intent(
        feedback_text,
        feedback_parser_runner=feedback_parser_runner,
        feedback_parser_session_id=feedback_parser_session_id,
        user_id=user_id,
    )
    prior_version = active_plan_version(task)
    merged_intent = _merge_revision_intents(
        prior_version.revision_intent if prior_version else None,
        parsed_intent,
    )
    revised_objective = _merge_objective_with_revision_intent(task.base_objective or task.objective, merged_intent)

    intent_route = await _route_query_intent(
        revised_objective,
        intent_router_runner=intent_router_runner,
        intent_router_session_id=intent_router_session_id,
        user_id=user_id,
    )
    request_type = str(intent_route.get("request_type", task.request_type) or task.request_type)
    intent_tags = list(intent_route.get("intent_tags", task.intent_tags) or task.intent_tags)
    model_plan_graph = await _draft_model_plan_graph(
        revised_objective,
        request_type=request_type,
        intent_tags=intent_tags,
        planner_runner=planner_runner,
        planner_session_id=planner_session_id,
        user_id=user_id,
    )

    replan_remaining_steps(
        task,
        revised_objective=revised_objective,
        request_type=request_type,
        intent_tags=intent_tags,
        revision_intent=merged_intent,
        gate_reason=gate_reason,
        use_dynamic_planner=DYNAMIC_PLANNER_GRAPH_ENABLED,
        tool_registry_summary=_TOOL_REGISTRY.summary(max_tools=120),
        plan_graph_override=model_plan_graph,
    )
    task.hitl_history.append(f"revise:{feedback_text}")
    _runtime_events.append_event(
        task,
        _runtime_events.EVENT_CHECKPOINT_REVISED,
        reason=gate_reason,
        feedback=feedback_text,
    )
    task.pending_feedback_queue = []
    task.awaiting_hitl = True
    task.checkpoint_state = "open"
    task.checkpoint_reason = gate_reason
    task.status = "in_progress"
    task.touch()
    state_store.save_task(task, note="feedback_replan")
    _print_checkpoint_plan(task)
    return task


async def _execute_until_next_gate_or_completion_cli(
    runner,
    session_id: str,
    user_id: str,
    state_store: TaskStateStore,
    task: WorkflowTask,
    *,
    bypass_first_gate: bool,
) -> tuple[str, dict | None]:
    quality_state: dict = {}
    first_gate_check = True

    while task.current_step_index + 1 < len(task.steps):
        next_idx = task.current_step_index + 1
        next_step = task.steps[next_idx]
        queued_feedback = [str(item).strip() for item in task.pending_feedback_queue if str(item).strip()]
        open_gate, gate_reason = should_open_checkpoint(task, next_step, quality_state, queued_feedback)
        if open_gate:
            if bypass_first_gate and first_gate_check and gate_reason == "pre_evidence_execution":
                first_gate_check = False
            else:
                task.awaiting_hitl = True
                task.checkpoint_state = "open"
                task.checkpoint_reason = gate_reason
                task.touch()
                state_store.save_task(task, note="adaptive_hitl_checkpoint_opened")
                print(f"\n[Adaptive Checkpoint] {_format_checkpoint_reason(gate_reason)}")
                _print_checkpoint_plan(task)
                return "awaiting_hitl", None

        first_gate_check = False
        step_text = await _execute_step(runner, session_id, user_id, task, next_idx)
        state_store.save_task(task, note=f"step_{next_idx + 1}_completed")
        print(step_text)

        step = task.steps[next_idx]
        tool_failures = sum(
            1
            for entry in (step.tool_trace or [])
            if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
        )
        base_quality = _evaluate_quality_gates(task)
        quality_state = {
            "unresolved_gaps": base_quality.get("unresolved_gaps", []),
            "last_step_failures": tool_failures,
            "last_step_output": step.output,
        }

    quality = _evaluate_quality_gates(task)
    print("\n" + _render_quality_gate_message(quality))
    return "completed", quality


def _persist_report_artifacts(task: WorkflowTask, report: str) -> tuple[Path, Path | None, str | None]:
    return _presentation_cli.persist_report_artifacts(
        task,
        report,
        reports_dir=REPORT_ARTIFACTS_DIR,
        write_markdown_pdf_fn=lambda markdown, pdf_path, title: write_markdown_pdf(
            markdown,
            pdf_path,
            title=title,
        ),
    )


def _print_final_report_with_artifacts(task: WorkflowTask, quality_report: dict) -> None:
    _presentation_cli.print_final_report_with_artifacts(
        task,
        quality_report,
        render_final_report_fn=render_final_report,
        reports_dir=REPORT_ARTIFACTS_DIR,
        write_markdown_pdf_fn=lambda markdown, pdf_path, title: write_markdown_pdf(
            markdown,
            pdf_path,
            title=title,
        ),
        print_fn=print,
    )


def _print_hitl_prompt() -> None:
    _presentation_cli.print_hitl_prompt(print_fn=print)


def _resolve_default_task_id(active_task: WorkflowTask | None, state_store: TaskStateStore) -> str | None:
    return _presentation_cli.resolve_default_task_id(active_task, state_store)


def _print_revision_history(state_store: TaskStateStore, task_id: str, limit: int = 12) -> None:
    _presentation_cli.print_revision_history(
        state_store,
        task_id,
        limit=limit,
        print_fn=print,
    )


def _resolve_rollback_revision_id(
    state_store: TaskStateStore,
    task_id: str,
    token: str,
) -> tuple[str | None, str | None]:
    normalized = token.strip()
    if not normalized:
        return None, "Missing revision token."
    if normalized.isdigit():
        offset = int(normalized)
        if offset < 0:
            return None, "Rollback offset cannot be negative."
        revisions = state_store.list_revisions(task_id, limit=max(20, offset + 1))
        if offset >= len(revisions):
            return None, f"Offset {offset} is out of range. Run `history {task_id}` first."
        return str(revisions[offset].get("revision_id", "")), None
    return normalized, None


async def _start_new_workflow_task(
    runner,
    session_id: str,
    user_id: str,
    state_store: TaskStateStore,
    objective: str,
    intent_route: dict | None = None,
    planner_runner=None,
    planner_session_id: str | None = None,
    task_id_override: str | None = None,
    created_at_override: str | None = None,
    hitl_history_seed: list[str] | None = None,
) -> WorkflowTask:
    route = intent_route or _default_intent_route(objective)
    routed_objective = str(route.get("normalized_query") or objective).strip() or objective
    original_revision_directive = _extract_revision_directive_from_objective(objective)
    if original_revision_directive and not _extract_revision_directive_from_objective(routed_objective):
        routed_objective = _merge_objective_with_revision(routed_objective, original_revision_directive)
    routed_request_type = sanitize_request_type(str(route.get("request_type", "")).strip())
    routed_intent_tags = sanitize_intent_tags(route.get("intent_tags"))
    model_plan_graph = await _draft_model_plan_graph(
        routed_objective,
        request_type=routed_request_type or classify_request_type(routed_objective),
        intent_tags=routed_intent_tags or infer_intent_tags(routed_objective),
        planner_runner=planner_runner,
        planner_session_id=planner_session_id,
        user_id=user_id,
    )
    revision_directive = _extract_revision_directive_from_objective(routed_objective)
    revision_timeframe = _extract_timeframe_hint(revision_directive or "")
    if revision_directive:
        print("\n[Revision Applied]")
        print(f"- Requested update: {revision_directive}")
        if revision_timeframe:
            print(f"- Interpreted timeframe update: {revision_timeframe}")
        else:
            print("- Interpreted timeframe update: not explicit; step 1 will restate assumed timeframe.")
        print("- Re-running scope/decomposition with this revision.")
    task = create_task(
        routed_objective,
        request_type_override=routed_request_type,
        intent_tags_override=routed_intent_tags,
        use_dynamic_planner=DYNAMIC_PLANNER_GRAPH_ENABLED,
        tool_registry_summary=_TOOL_REGISTRY.summary(max_tools=120),
        plan_graph_override=model_plan_graph,
    )
    task.base_objective = _extract_primary_objective_text(routed_objective) or routed_objective
    if task_id_override and task_id_override.strip():
        task.task_id = task_id_override.strip()
    if created_at_override and created_at_override.strip():
        task.created_at = created_at_override.strip()
    if hitl_history_seed is not None:
        task.hitl_history = [str(item).strip() for item in hitl_history_seed if str(item).strip()]
    state_store.save_task(task, note="task_created")
    step_text = await _execute_step(runner, session_id, user_id, task, 0)
    state_store.save_task(task, note="step_1_completed")
    print(_render_hitl_scope_summary(task, step_text))
    task.awaiting_hitl = True
    task.checkpoint_state = "open"
    task.checkpoint_reason = "pre_evidence_execution"
    _runtime_events.append_event(
        task,
        _runtime_events.EVENT_CHECKPOINT_OPENED,
        reason=task.checkpoint_reason,
        payload=task.checkpoint_payload or {},
    )
    initialize_plan_version(task, gate_reason=task.checkpoint_reason)
    task.touch()
    state_store.save_task(task, note="hitl_checkpoint_opened")
    _print_hitl_prompt()
    return task


async def run_interactive_async():
    """Run the agent in interactive mode (async version)."""
    from google.adk.sessions import InMemorySessionService
    
    print("=" * 60)
    print("AI Co-Scientist")
    print("=" * 60)
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("\n❌ GOOGLE_API_KEY not configured")
        print("\n   To fix this:")
        print("   1. Open .env file in the adk-agent folder")
        print("   2. Replace 'your-api-key-here' with your actual API key")
        print("   3. Get a free key at: https://aistudio.google.com/apikey")
        return
    
    print("\n✓ API key configured")
    print("Initializing agent with MCP server...")
    
    # Verify MCP server exists
    if not (MCP_SERVER_DIR / "server.js").exists():
        print(f"\n❌ MCP server not found at {MCP_SERVER_DIR}")
        print("\n   Make sure research-mcp/server.js exists")
        return
    
    try:
        agent, mcp_tools = create_agent()
    except Exception as e:
        print(f"\n❌ Failed to create agent: {e}")
        print("\n   Make sure:")
        print("   1. Node.js is installed")
        print("   2. Run: cd ../research-mcp && npm install")
        return
    
    # Get tool count to verify connection
    try:
        tools = await mcp_tools.get_tools()
        print(f"✓ Connected to MCP server ({len(tools)} tools available)")
        await _refresh_tool_registry(mcp_tools)
    except Exception as e:
        print(f"\n❌ MCP connection failed: {e}")
        await mcp_tools.close()
        return
    
    # Create runner with session
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="co_scientist",
        session_service=session_service,
    )
    clarifier_runner = Runner(
        agent=create_clarifier_agent(),
        app_name="co_scientist_clarifier",
        session_service=session_service,
    )
    intent_router_runner = Runner(
        agent=create_intent_router_agent(),
        app_name="co_scientist_intent_router",
        session_service=session_service,
    )
    planner_runner = Runner(
        agent=create_planner_agent(),
        app_name="co_scientist_planner",
        session_service=session_service,
    )
    planner_runner = Runner(
        agent=create_planner_agent(),
        app_name="co_scientist_planner",
        session_service=session_service,
    )
    planner_runner = Runner(
        agent=create_planner_agent(),
        app_name="co_scientist_planner",
        session_service=session_service,
    )
    feedback_parser_runner = Runner(
        agent=create_feedback_parser_agent(),
        app_name="co_scientist_feedback_parser",
        session_service=session_service,
    )
    
    # Create a session
    session = await session_service.create_session(
        app_name="co_scientist",
        user_id="researcher",
    )
    clarifier_session = await session_service.create_session(
        app_name="co_scientist_clarifier",
        user_id="researcher",
    )
    intent_router_session = await session_service.create_session(
        app_name="co_scientist_intent_router",
        user_id="researcher",
    )
    planner_session = await session_service.create_session(
        app_name="co_scientist_planner",
        user_id="researcher",
    )
    planner_session = await session_service.create_session(
        app_name="co_scientist_planner",
        user_id="researcher",
    )
    planner_session = await session_service.create_session(
        app_name="co_scientist_planner",
        user_id="researcher",
    )
    feedback_parser_session = await session_service.create_session(
        app_name="co_scientist_feedback_parser",
        user_id="researcher",
    )
    state_store = TaskStateStore(Path(__file__).parent / "state" / "workflow_tasks.json")
    active_task: WorkflowTask | None = None
    pending_clarification_query: str | None = None
    pending_clarification_prompt: str | None = None
    
    print("\n✓ Agent ready!")
    print("\nExample queries:")
    print("  - 'Find promising drug targets for Parkinson's disease'")
    print("  - 'Evaluate LRRK2 as a drug target'")
    print("  - 'What clinical trials exist for Alzheimer's gamma-secretase inhibitors?'")
    print("\nCommands: status | resume [task_id] | history [task_id] | rollback <offset|revision_id> [task_id] | help | quit")
    print("At HITL checkpoint: start | continue | <feedback text> | stop\n")
    print("-" * 60)
    
    try:
        while True:
            try:
                # Use asyncio-compatible input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\nYou: ").strip()
                )
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue

                lowered = user_input.lower().strip()
                if lowered == "help":
                    print("\nCommands: status | resume [task_id] | history [task_id] | rollback <offset|revision_id> [task_id] | help | quit")
                    print("At HITL checkpoint: start | continue | <feedback text> | stop")
                    continue

                if lowered == "status":
                    if pending_clarification_query:
                        print("\nWaiting for clarification before starting workflow.")
                        if pending_clarification_prompt:
                            print(pending_clarification_prompt)
                        print("Type your clarification, or `stop` to cancel.")
                        continue
                    task = active_task or state_store.latest_task()
                    if not task:
                        print("\nNo workflow task available.")
                    else:
                        print("\n" + render_status(task))
                    continue

                if lowered.startswith("history"):
                    parts = user_input.split(maxsplit=1)
                    task_id = parts[1].strip() if len(parts) > 1 else _resolve_default_task_id(active_task, state_store)
                    if not task_id:
                        print("\nNo workflow task available. Ask a query first.")
                        continue
                    _print_revision_history(state_store, task_id)
                    continue

                if lowered.startswith("rollback"):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 2:
                        print("\nUse: rollback <offset|revision_id> [task_id]")
                        continue
                    rollback_token = parts[1].strip()
                    task_id = parts[2].strip() if len(parts) > 2 else _resolve_default_task_id(active_task, state_store)
                    if not task_id:
                        print("\nNo workflow task available to rollback.")
                        continue
                    revision_id, error_msg = _resolve_rollback_revision_id(state_store, task_id, rollback_token)
                    if error_msg or not revision_id:
                        print(f"\n{error_msg or 'Could not resolve rollback revision.'}")
                        continue
                    rolled_back = state_store.rollback_task(task_id, revision_id)
                    if not rolled_back:
                        print(f"\nRevision {revision_id} not found for task {task_id}.")
                        continue
                    pending_clarification_query = None
                    pending_clarification_prompt = None
                    active_task = rolled_back
                    print(f"\nRolled back task {task_id} to revision {revision_id}.")
                    print(render_status(active_task))
                    if active_task.awaiting_hitl:
                        _print_hitl_prompt()
                    continue

                if pending_clarification_query:
                    if lowered in {"stop", "cancel"}:
                        pending_clarification_query = None
                        pending_clarification_prompt = None
                        print("\nClarification canceled. Ask a new query.")
                        if active_task and active_task.awaiting_hitl:
                            _print_hitl_prompt()
                        continue
                    clarified_query = _merge_query_with_clarification(
                        pending_clarification_query,
                        user_input,
                    )
                    pending_clarification_query = None
                    pending_clarification_prompt = None
                    follow_up = await _build_clarification_request(
                        clarified_query,
                        clarifier_runner=clarifier_runner,
                        clarifier_session_id=clarifier_session.id,
                        user_id="researcher",
                    )
                    if follow_up:
                        pending_clarification_query = clarified_query
                        pending_clarification_prompt = follow_up
                        print("\n[Clarification Needed]")
                        print(follow_up)
                        print("Type your clarification, or `stop` to cancel.")
                        continue
                    print("\nClarification received. Continuing workflow...")
                    intent_route = await _route_query_intent(
                        clarified_query,
                        intent_router_runner=intent_router_runner,
                        intent_router_session_id=intent_router_session.id,
                        user_id="researcher",
                    )
                    active_task = await _start_new_workflow_task(
                        runner,
                        session.id,
                        "researcher",
                        state_store,
                        clarified_query,
                        intent_route=intent_route,
                        planner_runner=planner_runner,
                        planner_session_id=planner_session.id,
                    )
                    continue

                if (
                    lowered in {"continue", "start"}
                    and active_task
                    and not active_task.awaiting_hitl
                    and active_task.status in {"pending", "in_progress"}
                    and active_task.current_step_index < len(active_task.steps) - 1
                ):
                    terminal_status, quality = await _execute_until_next_gate_or_completion_cli(
                        runner,
                        session.id,
                        "researcher",
                        state_store,
                        active_task,
                        bypass_first_gate=False,
                    )
                    if terminal_status == "awaiting_hitl":
                        active_task.status = "in_progress"
                        active_task.touch()
                        state_store.save_task(active_task, note="adaptive_checkpoint_waiting")
                        _print_hitl_prompt()
                        continue
                    active_task.status = "completed"
                    active_task.awaiting_hitl = False
                    active_task.checkpoint_state = "closed"
                    active_task.checkpoint_reason = ""
                    active_task.touch()
                    state_store.save_task(active_task, note="workflow_completed")
                    _print_final_report_with_artifacts(active_task, quality or _evaluate_quality_gates(active_task))
                    continue

                if lowered in {"continue", "start", "stop"} and not (active_task and active_task.awaiting_hitl):
                    print("\nNo pending checkpoint. Ask a new query or use `status`.")
                    continue

                if lowered.startswith("resume"):
                    parts = user_input.split(maxsplit=1)
                    task_id = parts[1].strip() if len(parts) > 1 else None
                    task = state_store.get_task(task_id) if task_id else state_store.latest_task()
                    if not task:
                        print("\nNo matching task found to resume.")
                        continue
                    active_task = task
                    print("\nResumed task:")
                    print(render_status(active_task))
                    if active_task.awaiting_hitl:
                        _print_hitl_prompt()
                    continue

                if active_task and active_task.awaiting_hitl:
                    if lowered in {"continue", "start"}:
                        active_task.hitl_history.append("continue")
                        active_task.awaiting_hitl = False
                        active_task.checkpoint_state = "closed"
                        active_task.checkpoint_reason = ""
                        active_task.touch()
                        state_store.save_task(active_task, note="hitl_start")
                        terminal_status, quality = await _execute_until_next_gate_or_completion_cli(
                            runner,
                            session.id,
                            "researcher",
                            state_store,
                            active_task,
                            bypass_first_gate=True,
                        )
                        if terminal_status == "awaiting_hitl":
                            active_task.status = "in_progress"
                            active_task.touch()
                            state_store.save_task(active_task, note="adaptive_checkpoint_waiting")
                            _print_hitl_prompt()
                            continue

                        active_task.status = "completed"
                        active_task.awaiting_hitl = False
                        active_task.checkpoint_state = "closed"
                        active_task.checkpoint_reason = ""
                        active_task.touch()
                        state_store.save_task(active_task, note="workflow_completed")
                        _print_final_report_with_artifacts(active_task, quality or _evaluate_quality_gates(active_task))
                        continue

                    if lowered == "stop":
                        active_task.hitl_history.append("stop")
                        active_task.status = "blocked"
                        active_task.awaiting_hitl = False
                        active_task.checkpoint_state = "closed"
                        active_task.checkpoint_reason = ""
                        active_task.touch()
                        state_store.save_task(active_task, note="workflow_stopped")
                        print("\nWorkflow stopped and saved.")
                        continue

                    if lowered.startswith("revise"):
                        feedback_text = user_input[6:].strip()
                    else:
                        feedback_text = user_input.strip()

                    if feedback_text:
                        await _apply_feedback_replan_cli(
                            runner=runner,
                            session_id=session.id,
                            user_id="researcher",
                            state_store=state_store,
                            task=active_task,
                            feedback_text=feedback_text,
                            intent_router_runner=intent_router_runner,
                            intent_router_session_id=intent_router_session.id,
                            planner_runner=planner_runner,
                            planner_session_id=planner_session.id,
                            feedback_parser_runner=feedback_parser_runner,
                            feedback_parser_session_id=feedback_parser_session.id,
                            gate_reason="feedback_replan",
                        )
                        _print_hitl_prompt()
                        continue

                    print("\nThis task is waiting at HITL checkpoint.")
                    _print_hitl_prompt()
                    continue

                clarification_msg = await _build_clarification_request(
                    user_input,
                    clarifier_runner=clarifier_runner,
                    clarifier_session_id=clarifier_session.id,
                    user_id="researcher",
                )
                if clarification_msg:
                    pending_clarification_query = user_input
                    pending_clarification_prompt = clarification_msg
                    print("\n[Clarification Needed]")
                    print(clarification_msg)
                    print("Type your clarification, or `stop` to cancel.")
                    continue

                # New request -> create a planned workflow and run step 1.
                intent_route = await _route_query_intent(
                    user_input,
                    intent_router_runner=intent_router_runner,
                    intent_router_session_id=intent_router_session.id,
                    user_id="researcher",
                )
                active_task = await _start_new_workflow_task(
                    runner,
                    session.id,
                    "researcher",
                    state_store,
                    user_input,
                    intent_route=intent_route,
                    planner_runner=planner_runner,
                    planner_session_id=planner_session.id,
                )
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()
                print("Please try again.")
    finally:
        # Cleanup MCP connection
        await mcp_tools.close()


def run_interactive():
    """Run the agent in interactive mode."""
    asyncio.run(run_interactive_async())


async def run_single_query_async(query: str, *, state_store_path: Path | None = None):
    """Run a single query (async version)."""
    from google.adk.sessions import InMemorySessionService

    session_service = InMemorySessionService()
    clarifier_runner = Runner(
        agent=create_clarifier_agent(),
        app_name="co_scientist_clarifier",
        session_service=session_service,
    )
    intent_router_runner = Runner(
        agent=create_intent_router_agent(),
        app_name="co_scientist_intent_router",
        session_service=session_service,
    )
    planner_runner = Runner(
        agent=create_planner_agent(),
        app_name="co_scientist_planner",
        session_service=session_service,
    )
    clarifier_session = await session_service.create_session(
        app_name="co_scientist_clarifier",
        user_id="researcher",
    )
    intent_router_session = await session_service.create_session(
        app_name="co_scientist_intent_router",
        user_id="researcher",
    )
    planner_session = await session_service.create_session(
        app_name="co_scientist_planner",
        user_id="researcher",
    )
    clarification_msg = await _build_clarification_request(
        query,
        clarifier_runner=clarifier_runner,
        clarifier_session_id=clarifier_session.id,
        user_id="researcher",
    )
    if clarification_msg:
        return "\n".join(
            [
                "## Clarification Needed",
                clarification_msg,
                "",
                "Reply with your clarification and I will continue.",
            ]
        )

    agent, mcp_tools = create_agent()
    await _refresh_tool_registry(mcp_tools)

    runner = Runner(
        agent=agent,
        app_name="co_scientist",
        session_service=session_service,
    )
    
    # Create session
    session = await session_service.create_session(
        app_name="co_scientist",
        user_id="researcher",
    )
    default_state_path = Path(__file__).parent / "state" / "workflow_tasks.json"
    state_store = TaskStateStore(state_store_path or default_state_path)
    intent_route = await _route_query_intent(
        query,
        intent_router_runner=intent_router_runner,
        intent_router_session_id=intent_router_session.id,
        user_id="researcher",
    )
    routed_query = str(intent_route.get("normalized_query") or query).strip() or query
    request_type = sanitize_request_type(str(intent_route.get("request_type", "")).strip()) or classify_request_type(routed_query)
    intent_tags = sanitize_intent_tags(intent_route.get("intent_tags")) or infer_intent_tags(routed_query)
    model_plan_graph = await _draft_model_plan_graph(
        routed_query,
        request_type=request_type,
        intent_tags=intent_tags,
        planner_runner=planner_runner,
        planner_session_id=planner_session.id,
        user_id="researcher",
    )
    task = create_task(
        routed_query,
        request_type_override=request_type,
        intent_tags_override=intent_tags,
        tool_registry_summary=_TOOL_REGISTRY.summary(max_tools=120),
        plan_graph_override=model_plan_graph,
    )
    state_store.save_task(task, note="task_created_single_query")

    for idx in range(len(task.steps)):
        await _execute_step(runner, session.id, "researcher", task, idx)
        state_store.save_task(task, note=f"step_{idx + 1}_completed_single_query")

    quality = _evaluate_quality_gates(task)

    task.status = "completed"
    task.touch()
    state_store.save_task(task, note="workflow_completed_single_query")
    report = render_final_report(task, quality_report=quality)

    await mcp_tools.close()
    return report


def run_single_query(query: str, *, state_store_path: Path | None = None):
    """Run a single query (useful for testing)."""
    return asyncio.run(run_single_query_async(query, state_store_path=state_store_path))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print(__doc__)
        print("\nUsage: python agent.py")
    else:
        run_interactive()
