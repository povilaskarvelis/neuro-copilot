"""
Runtime execution, tool tracing, and retry/escalation behavior.
"""
from __future__ import annotations

import asyncio
import json
import re


STEP_SCOPE_TOOLS = {
    "search_diseases",
    "search_targets",
    "expand_disease_context",
}

STEP_EVIDENCE_BACKSTOP_TOOLS = {
    "search_pubmed_advanced",
    "search_openalex_works",
    "search_openalex_authors",
    "search_clinical_trials",
    "search_disease_targets",
    "search_diseases",
    "search_targets",
    "expand_disease_context",
    "get_target_info",
    "get_gene_info",
    "list_local_datasets",
    "read_local_dataset",
}


def is_reasoning_only_step(task, step_idx: int) -> bool:
    return step_idx == 0 or step_idx == len(task.steps) - 1


def build_step_allowed_tools(task, step_idx: int, tool_bundle_for_intent_fn) -> list[str]:
    step = task.steps[step_idx]

    # Keep request framing and final synthesis deterministic.
    if is_reasoning_only_step(task, step_idx):
        return sorted(STEP_SCOPE_TOOLS) if step_idx == 0 else []

    preferred, fallback = tool_bundle_for_intent_fn(task.intent_tags)
    allowed = set(step.recommended_tools + step.fallback_tools + preferred + fallback)
    allowed.update(STEP_EVIDENCE_BACKSTOP_TOOLS)
    return sorted(allowed)


def should_escalate_allowlist(step, trace_entries: list[dict], output: str) -> bool:
    if not step.recommended_tools:
        return False
    if not trace_entries:
        return True
    outcomes = {str(entry.get("outcome", "unknown")) for entry in trace_entries}
    if outcomes and outcomes.issubset({"error", "not_found_or_empty", "no_response", "degraded"}):
        return True
    lower = (output or "").lower()
    if any(token in lower for token in ["cannot be completed", "insufficient data", "unable to identify"]):
        return True
    return False


def build_escalated_allowed_tools(task, step_idx: int, tool_bundle_for_intent_fn) -> list[str]:
    base = set(build_step_allowed_tools(task, step_idx, tool_bundle_for_intent_fn))
    # Escalation broadens coverage while keeping synthesis steps tool-free.
    if is_reasoning_only_step(task, step_idx):
        return sorted(base)
    base.update(
        {
            "search_pubmed",
            "get_pubmed_abstract",
            "get_pubmed_paper_details",
            "get_pubmed_author_profile",
            "get_target_drugs",
            "check_druggability",
            "search_clinvar_variants",
            "search_gwas_associations",
            "summarize_clinical_trials_landscape",
            "summarize_target_expression_context",
            "summarize_target_competitive_landscape",
            "summarize_target_safety_liabilities",
            "compare_targets_multi_axis",
        }
    )
    return sorted(base)


def create_step_runner(base_runner, allowed_tools: list[str], create_agent_fn, runner_cls):
    step_agent, step_mcp_tools = create_agent_fn(tool_filter=allowed_tools)
    step_runner = runner_cls(
        agent=step_agent,
        app_name=base_runner.app_name,
        session_service=base_runner.session_service,
        artifact_service=getattr(base_runner, "artifact_service", None),
        memory_service=getattr(base_runner, "memory_service", None),
        credential_service=getattr(base_runner, "credential_service", None),
    )
    return step_runner, step_mcp_tools


async def run_runner_turn(runner, session_id: str, user_id: str, prompt: str, run_runner_turn_with_trace_fn=None) -> str:
    """Run one model turn and return text only."""
    trace_fn = run_runner_turn_with_trace_fn or run_runner_turn_with_trace
    response_text, _ = await trace_fn(runner, session_id, user_id, prompt)
    return response_text


async def run_runner_turn_with_timeout(
    runner,
    session_id: str,
    user_id: str,
    prompt: str,
    *,
    run_runner_turn_with_trace_fn,
    default_timeout_seconds: float,
    timeout_seconds: float | None = None,
) -> tuple[str, list[dict]]:
    timeout = float(default_timeout_seconds if timeout_seconds is None else timeout_seconds)
    if timeout <= 0:
        return await run_runner_turn_with_trace_fn(runner, session_id, user_id, prompt)
    try:
        return await asyncio.wait_for(
            run_runner_turn_with_trace_fn(runner, session_id, user_id, prompt),
            timeout=timeout,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Step model/tool turn timed out after {timeout:g}s.") from exc


def extract_missing_tool_name(error: Exception) -> str | None:
    match = re.search(r"Tool '([^']+)' not found", str(error))
    return match.group(1).strip() if match else None


def runtime_tool_constraint_suffix(tool_filter: list[str] | None) -> str:
    if tool_filter is None:
        return ""
    allowed = sorted(set(tool_filter))
    if not allowed:
        return (
            "\n\n## Runtime Tool Constraint\n"
            "No tools are available for this run. Do not emit any tool calls.\n"
            "Produce reasoning-only output."
        )
    return (
        "\n\n## Runtime Tool Constraint\n"
        "You may call ONLY these tools in this run:\n"
        f"- {', '.join(allowed)}\n"
        "Do not call any tool that is not in this list."
    )


def safe_model_dump(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=True)
        except TypeError:
            dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    return {"value": str(value)}


def normalize_trace_detail(text: str, *, max_chars: int = 260) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def compact_json(value, *, max_chars: int = 100) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, sort_keys=True)
    except TypeError:
        text = str(value)
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def summarize_for_report(text: str, *, max_chars: int = 220) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)[0]
    if len(sentence) <= max_chars:
        return sentence
    return f"{sentence[: max_chars - 3].rstrip()}..."


def format_step_execution_error(
    error: Exception,
    allowed_tools: list[str],
    *,
    normalize_trace_detail_fn=normalize_trace_detail,
) -> str:
    allowed_text = ", ".join(allowed_tools) if allowed_tools else "none"
    detail = normalize_trace_detail_fn(str(error), max_chars=420)
    return (
        "Step execution issue encountered.\n"
        f"Details: {detail}\n"
        f"Allowed tools for this step: {allowed_text}\n"
        "Continuing with best-effort output under current constraints."
    )


def populate_step_rao_fields(
    step,
    *,
    summarize_for_report_fn=summarize_for_report,
    compact_json_fn=compact_json,
) -> None:
    """Populate explicit Reasoning/Actions/Observations fields for report rendering."""
    reasoning = summarize_for_report_fn(step.output) or summarize_for_report_fn(step.instruction, max_chars=180)
    actions: list[str] = []
    observations: list[str] = []

    trace_entries = step.tool_trace if isinstance(step.tool_trace, list) else []
    if trace_entries:
        for entry in trace_entries[:4]:
            tool_name = str(entry.get("tool_name", "unknown_tool"))
            outcome = str(entry.get("outcome", "unknown"))
            args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
            args_text = f" args={compact_json_fn(args, max_chars=80)}" if args else ""
            actions.append(f"Called `{tool_name}` ({outcome}).{args_text}")
        omitted = len(trace_entries) - 4
        if omitted > 0:
            actions.append(f"{omitted} additional tool call(s) omitted for brevity.")
    else:
        actions.append("No tool calls executed; step completed as reasoning/synthesis.")

    output_summary = summarize_for_report_fn(step.output, max_chars=260)
    if output_summary:
        observations.append(output_summary)
    if step.evidence_refs:
        preview = ", ".join(step.evidence_refs[:5])
        suffix = f", +{len(step.evidence_refs) - 5} more" if len(step.evidence_refs) > 5 else ""
        observations.append(f"Citation IDs captured: {preview}{suffix}.")
    issue_count = sum(
        1
        for entry in trace_entries
        if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
    )
    if issue_count:
        observations.append(f"{issue_count} tool call(s) returned errors/empty responses.")
    if not observations:
        observations.append("No additional observations captured.")

    step.reasoning_summary = reasoning or "Reasoning summary unavailable."
    step.actions = actions
    step.observations = observations


def extract_response_excerpt(response_payload, *, normalize_trace_detail_fn=normalize_trace_detail) -> str:
    if not isinstance(response_payload, dict):
        return "No structured response payload captured."

    content = response_payload.get("content")
    snippets: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    snippets.append(text)

    if not snippets:
        output_payload = response_payload.get("output")
        if isinstance(output_payload, str) and output_payload.strip():
            snippets.append(output_payload)
        elif output_payload is not None:
            snippets.append(str(output_payload))

    if not snippets:
        error_payload = response_payload.get("error")
        if error_payload:
            snippets.append(str(error_payload))

    if not snippets:
        snippets.append(str(response_payload))
    return normalize_trace_detail_fn(" | ".join(snippets))


def classify_tool_response(
    response_payload,
    *,
    normalize_trace_detail_fn=normalize_trace_detail,
    extract_response_excerpt_fn=extract_response_excerpt,
) -> tuple[str, str]:
    if response_payload is None:
        return "no_response", "Tool call was issued but no response payload was returned."
    if not isinstance(response_payload, dict):
        return "unknown", normalize_trace_detail_fn(str(response_payload))

    excerpt = extract_response_excerpt_fn(response_payload)

    explicit_error = bool(response_payload.get("error")) or response_payload.get("isError") is True
    if explicit_error:
        return "error", excerpt

    lower = excerpt.lower()
    if lower.startswith("error in ") or lower.startswith("error:") or "request failed (" in lower:
        return "error", excerpt
    not_found_markers = (
        "not found",
        "no results",
        "no matching",
        "no records",
        "no data found",
        "no target data found",
        "no clinical trials found",
        "no expression context found",
        "couldn't find",
        "unable to find",
        "did not find",
        "no evidence found",
    )
    if any(marker in lower for marker in not_found_markers):
        return "not_found_or_empty", excerpt

    degraded_markers = (
        "critical gap",
        "service unavailable",
        "underlying gwas call error",
        "fallback uses open targets genetics evidence scores",
        "risk-increasing vs protective direction cannot be inferred",
        "could not infer genetic direction-of-effect",
    )
    if any(marker in lower for marker in degraded_markers):
        return "degraded", excerpt

    return "ok", excerpt


async def run_runner_turn_with_trace(
    runner,
    session_id: str,
    user_id: str,
    prompt: str,
    *,
    safe_model_dump_fn=safe_model_dump,
    classify_tool_response_fn=classify_tool_response,
) -> tuple[str, list[dict]]:
    """Run one model turn and collect both text output and exact tool trace."""
    from google.genai.types import Content, Part

    message = Content(role="user", parts=[Part(text=prompt)])
    response_text = ""
    trace_entries: list[dict] = []
    pending_by_call_id: dict[str, int] = {}
    pending_by_tool_name: dict[str, list[int]] = {}
    sequence = 0

    async for event in runner.run_async(
        session_id=session_id,
        user_id=user_id,
        new_message=message,
    ):
        if not hasattr(event, "content") or not event.content or not hasattr(event.content, "parts"):
            continue
        if not event.content.parts:
            continue
        for part in event.content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text

            function_call = getattr(part, "function_call", None)
            if function_call:
                payload = safe_model_dump_fn(function_call)
                sequence += 1
                call_id = str(payload.get("id") or f"call-{sequence}")
                tool_name = str(payload.get("name") or "unknown_tool")
                args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
                entry = {
                    "sequence": sequence,
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "args": args,
                    "outcome": "pending",
                    "detail": "",
                    "phase": "main",
                }
                trace_entries.append(entry)
                pending_by_call_id[call_id] = len(trace_entries) - 1
                pending_by_tool_name.setdefault(tool_name, []).append(len(trace_entries) - 1)

            function_response = getattr(part, "function_response", None)
            if function_response:
                payload = safe_model_dump_fn(function_response)
                call_id = str(payload.get("id") or "")
                tool_name = str(payload.get("name") or "unknown_tool")
                response_payload = payload.get("response")
                outcome, detail = classify_tool_response_fn(response_payload)

                target_index = pending_by_call_id.get(call_id) if call_id else None
                if target_index is None:
                    for candidate_index in pending_by_tool_name.get(tool_name, []):
                        if trace_entries[candidate_index].get("outcome") == "pending":
                            target_index = candidate_index
                            break

                if target_index is None:
                    sequence += 1
                    trace_entries.append(
                        {
                            "sequence": sequence,
                            "call_id": call_id or f"response-{sequence}",
                            "tool_name": tool_name,
                            "args": {},
                            "outcome": outcome,
                            "detail": detail,
                            "phase": "main",
                        }
                    )
                    continue

                trace_entries[target_index]["outcome"] = outcome
                trace_entries[target_index]["detail"] = detail

    for entry in trace_entries:
        if entry.get("outcome") == "pending":
            entry["outcome"] = "no_response"
            entry["detail"] = "Tool call was issued but no matching function_response event was captured."

    return response_text.strip(), trace_entries


async def execute_step(
    runner,
    session_id: str,
    user_id: str,
    task,
    step_idx: int,
    *,
    step_prompt_fn,
    extract_evidence_refs_fn,
    build_step_allowed_tools_fn,
    create_step_runner_fn,
    run_runner_turn_with_timeout_fn,
    extract_missing_tool_name_fn=extract_missing_tool_name,
    format_step_execution_error_fn=format_step_execution_error,
    should_escalate_allowlist_fn=should_escalate_allowlist,
    build_escalated_allowed_tools_fn=None,
    populate_step_rao_fields_fn=populate_step_rao_fields,
) -> str:
    """Execute a single workflow step and update task status."""
    if build_escalated_allowed_tools_fn is None:
        raise ValueError("build_escalated_allowed_tools_fn is required")

    step = task.steps[step_idx]
    task.status = "in_progress"
    task.current_step_index = step_idx
    step.status = "in_progress"
    task.touch()

    step.allowed_tools = build_step_allowed_tools_fn(task, step_idx)
    prompt = step_prompt_fn(task, step)

    step_failed = False
    step_runner, step_mcp_tools = create_step_runner_fn(runner, step.allowed_tools)
    try:
        try:
            output, trace_entries = await run_runner_turn_with_timeout_fn(
                step_runner,
                session_id,
                user_id,
                prompt,
            )
        except Exception as exc:
            step_failed = True
            missing_tool = extract_missing_tool_name_fn(exc)
            if missing_tool:
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"IMPORTANT: The prior attempt called unavailable tool `{missing_tool}`.\n"
                    "Do not call unavailable tools. Use only the allowed-tools list above. "
                    "If no allowed tool is relevant, provide reasoning-only output for this step."
                )
                try:
                    output, trace_entries = await run_runner_turn_with_timeout_fn(
                        step_runner,
                        session_id,
                        user_id,
                        retry_prompt,
                    )
                    for entry in trace_entries:
                        entry["phase"] = "retry_after_missing_tool"
                    step_failed = False
                except Exception as retry_exc:
                    output = format_step_execution_error_fn(retry_exc, step.allowed_tools)
                    trace_entries = []
            else:
                output = format_step_execution_error_fn(exc, step.allowed_tools)
                trace_entries = []
    finally:
        if step_mcp_tools:
            await step_mcp_tools.close()

    if should_escalate_allowlist_fn(step, trace_entries, output):
        escalated_tools = build_escalated_allowed_tools_fn(task, step_idx)
        if set(escalated_tools) != set(step.allowed_tools):
            step.allowed_tools = escalated_tools
            escalated_runner, escalated_mcp_tools = create_step_runner_fn(runner, step.allowed_tools)
            try:
                try:
                    escalated_output, escalated_trace = await run_runner_turn_with_timeout_fn(
                        escalated_runner,
                        session_id,
                        user_id,
                        prompt,
                    )
                except Exception as exc:
                    step_failed = True
                    escalated_output = format_step_execution_error_fn(exc, step.allowed_tools)
                    escalated_trace = []
            finally:
                if escalated_mcp_tools:
                    await escalated_mcp_tools.close()
            for entry in escalated_trace:
                entry["phase"] = "step_allowlist_escalation"
            if escalated_trace:
                trace_entries.extend(escalated_trace)
            if escalated_output:
                output = escalated_output

    step.output = output if output else "(No response generated)"
    step.evidence_refs = extract_evidence_refs_fn(step.output)
    step.tool_trace = trace_entries
    populate_step_rao_fields_fn(step)
    step.status = "blocked" if step_failed else ("completed" if output else "blocked")
    task.touch()
    return step.output
