"""
Quality gates, adaptive checkpoint policy, and fallback recovery helpers.
"""
from __future__ import annotations

import re


def evaluate_quality_gates(task) -> dict:
    evidence_count = len({ref for step in task.steps for ref in step.evidence_refs})
    steps_with_output = sum(1 for step in task.steps if step.output and step.output != "(No response generated)")
    coverage_ratio = steps_with_output / len(task.steps) if task.steps else 0.0
    tool_call_count = sum(len(step.tool_trace) for step in task.steps) + len(task.fallback_tool_trace)

    unresolved_gaps: list[str] = []

    def _append_gap(message: str) -> None:
        normalized = str(message or "").strip()
        if normalized and normalized not in unresolved_gaps:
            unresolved_gaps.append(normalized)

    combined_output = "\n".join(step.output for step in task.steps if step.output).lower()
    objective_lower = task.objective.lower()
    if "researcher_discovery" in task.intent_tags:
        if "cannot be directly listed" in combined_output or "tool limitation" in combined_output:
            unresolved_gaps.append("Researcher identification appears incomplete due to tool limitations.")
        if not any(token in combined_output for token in ["author", "researcher", "investigator"]):
            unresolved_gaps.append("No explicit researcher entities were reported.")
        researcher_step = next((step for step in task.steps if "evidence" in step.title.lower()), None)
        if researcher_step:
            ranking_calls = [
                entry
                for entry in researcher_step.tool_trace
                if str(entry.get("tool_name", "")) == "rank_researchers_by_activity"
            ]
            successful_ranking = [entry for entry in ranking_calls if str(entry.get("outcome")) == "ok"]
            if not ranking_calls:
                unresolved_gaps.append(
                    "No quantitative ranking tool call (`rank_researchers_by_activity`) was executed."
                )
            elif not successful_ranking:
                unresolved_gaps.append(
                    "No successful quantitative researcher ranking call completed."
                )
            openalex_topic_calls = [
                entry
                for entry in researcher_step.tool_trace
                if str(entry.get("tool_name", "")) in {"rank_researchers_by_activity", "search_openalex_works"}
            ]
            failed_openalex_topic_calls = [
                entry
                for entry in openalex_topic_calls
                if str(entry.get("outcome", "")) in {"error", "no_response"}
            ]
            if openalex_topic_calls and len(failed_openalex_topic_calls) == len(openalex_topic_calls):
                unresolved_gaps.append(
                    "All topic-specific OpenAlex ranking/evidence calls failed; researcher ranking is unreliable."
                )
        top_query = any(
            marker in objective_lower
            for marker in [" top ", "top ", "most active", "prominent", "leading", "most prominent"]
        ) or task.request_type == "prioritization"
        if top_query and not any(
            marker in combined_output for marker in ["activity score", "score:", "ranked", "topic works"]
        ):
            unresolved_gaps.append(
                "Output lacks quantitative ranking metrics for a top/prominent researcher request."
            )
        if any(
            marker in combined_output
            for marker in [
                "request failed (429)",
                "rate limit",
                "preliminary",
                "could not perform",
                "could not be performed",
            ]
        ):
            unresolved_gaps.append(
                "Researcher ranking output still signals degraded evidence quality due to rate limits or incomplete ranking."
            )
    if any(token in objective_lower for token in ["target", "druggab", "candidate"]) or "clinical_landscape" in task.intent_tags:
        if any(
            token in combined_output
            for token in [
                "cannot be fulfilled",
                "cannot be completed",
                "insufficient data",
                "no target candidates",
                "unable to identify target",
            ]
        ):
            _append_gap("Target/trial assessment appears incomplete based on model self-reported gaps.")
        if not any(token in combined_output for token in ["ensg", "target id", "candidate target", "phase", "nct"]):
            _append_gap("No concrete target or clinical-trial entities were detected in the synthesis.")

    failed_entries = [
        entry
        for step in task.steps
        for entry in (step.tool_trace or [])
        if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
    ]
    failed_entries.extend(
        entry
        for entry in (task.fallback_tool_trace or [])
        if str(entry.get("outcome", "")) in {"error", "not_found_or_empty", "no_response", "degraded"}
    )
    if failed_entries:
        _append_gap(
            "Tool execution issues were detected "
            f"({len(failed_entries)} failed or empty tool calls)."
        )

    failed_tools = {
        str(entry.get("tool_name", "")).strip()
        for entry in failed_entries
        if str(entry.get("tool_name", "")).strip()
    }
    genetics_priority = (
        "genetics_direction" in task.intent_tags
        or any(token in objective_lower for token in ["genetic", "gwas", "variant", "direction-of-effect"])
    )
    if genetics_priority and failed_tools.intersection(
        {"infer_genetic_effect_direction", "search_gwas_associations", "search_clinvar_variants"}
    ):
        _append_gap("High-priority human genetics direction evidence is incomplete due to tool failures.")
    if "safety_assessment" in task.intent_tags and (
        "no safety liabilities" in combined_output
        or "no safety liability" in combined_output
        or "summarize_target_safety_liabilities" in failed_tools
    ):
        _append_gap("High-priority safety-liability evidence is incomplete or ambiguous.")

    critical_markers = (
        "critical gap",
        "service unavailable",
        "failed due to api error",
        "failed due to api errors",
        "could not retrieve",
        "unable to retrieve",
        "persistent failure",
    )
    if any(marker in combined_output for marker in critical_markers):
        _append_gap("Output reports critical missing evidence that affects confidence in the recommendation.")
    if any(marker in combined_output for marker in ["not directly from tool output", "historical knowledge"]):
        _append_gap("Synthesis includes claims that are not directly supported by captured tool output.")

    if evidence_count == 0:
        _append_gap("No citation evidence IDs were detected in the response.")
    if tool_call_count == 0:
        _append_gap("No tool calls were captured for the workflow.")
    missing_tool_steps = [
        step.title
        for step in task.steps
        if step.status == "completed" and step.recommended_tools and not step.tool_trace
    ]
    if missing_tool_steps:
        _append_gap(
            "Completed steps with recommended tools but no recorded tool execution: "
            + ", ".join(missing_tool_steps)
        )

    passed = evidence_count >= 2 and coverage_ratio >= 0.9 and tool_call_count >= 1 and len(unresolved_gaps) == 0
    return {
        "passed": passed,
        "evidence_count": evidence_count,
        "coverage_ratio": coverage_ratio,
        "tool_call_count": tool_call_count,
        "unresolved_gaps": unresolved_gaps,
    }


def gate_ack_token(reason: str, plan_version_id: str | None) -> str | None:
    normalized_reason = str(reason or "").strip().lower()
    if not normalized_reason:
        return None
    normalized_plan = str(plan_version_id or "none").strip() or "none"
    return f"gate_ack:{normalized_reason}:{normalized_plan}"


def should_open_checkpoint(
    task,
    next_step,
    quality_state: dict | None = None,
    queued_feedback: list[str] | None = None,
    *,
    active_plan_version_fn=None,
    gate_ack_token_fn=gate_ack_token,
) -> tuple[bool, str]:
    queued = [str(item).strip() for item in (queued_feedback or []) if str(item).strip()]
    if queued:
        return True, "queued_feedback_pending"

    if not next_step:
        return False, "none"

    quality_state = quality_state or {}
    unresolved_gap_count = len(quality_state.get("unresolved_gaps", []) or [])
    last_failures = int(quality_state.get("last_step_failures", 0) or 0)
    last_output = str(quality_state.get("last_step_output", "") or "").lower()
    plan = active_plan_version_fn(task) if active_plan_version_fn else None
    plan_id = plan.version_id if plan else str(task.active_plan_version_id or "none")
    hitl_events = set(task.hitl_history)

    def _is_gate_acknowledged(reason: str) -> bool:
        token = gate_ack_token_fn(reason, plan_id)
        return bool(token and token in hitl_events)

    if next_step.recommended_tools and not any(step.tool_trace for step in task.steps if step.status == "completed"):
        return True, "pre_evidence_execution"

    is_pre_final = bool(task.steps) and next_step.step_id == task.steps[-1].step_id
    if unresolved_gap_count >= 2:
        if _is_gate_acknowledged("quality_gap_spike"):
            return False, "none"
        return True, "quality_gap_spike"
    if is_pre_final and unresolved_gap_count >= 1:
        if _is_gate_acknowledged("quality_gap_spike"):
            return False, "none"
        return True, "quality_gap_spike"

    if last_failures >= 2:
        if _is_gate_acknowledged("repeated_tool_failures"):
            return False, "none"
        return True, "repeated_tool_failures"

    contradiction_markers = (
        "contradict",
        "inconsistent",
        "conflict",
        "uncertain",
        "critical gap",
        "service unavailable",
        "failed due to api error",
        "failed due to api errors",
        "could not retrieve",
        "unable to retrieve",
    )
    if any(marker in last_output for marker in contradiction_markers):
        if _is_gate_acknowledged("uncertainty_spike"):
            return False, "none"
        return True, "uncertainty_spike"

    if is_pre_final and any(event.lower().startswith("revise:") for event in task.hitl_history):
        if _is_gate_acknowledged("pre_final_after_intent_change"):
            return False, "none"
        return True, "pre_final_after_intent_change"

    return False, "none"


def render_quality_gate_message(report: dict) -> str:
    lines = [
        "[Quality Gate Check]",
        f"- Evidence references found: {report['evidence_count']}",
        f"- Step coverage ratio: {report['coverage_ratio']:.2f}",
        f"- Tool calls captured: {report.get('tool_call_count', 0)}",
    ]
    if report["unresolved_gaps"]:
        lines.append("- Unresolved critical gaps:")
        lines.extend([f"  - {gap}" for gap in report["unresolved_gaps"]])
    else:
        lines.append("- Unresolved critical gaps: none")
    return "\n".join(lines)


def clean_recovery_text(text: str) -> str:
    if not text:
        return text
    seen = set()
    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if normalized in {"**3. key results:**", "3. key results:"}:
            if "3-key-results" in seen:
                continue
            seen.add("3-key-results")
        if normalized and normalized in seen and normalized.startswith("**"):
            continue
        if normalized:
            seen.add(normalized)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


async def run_fallback_recovery(
    runner,
    session_id: str,
    user_id: str,
    task,
    *,
    run_runner_turn_with_trace_fn,
    format_step_execution_error_fn,
    clean_recovery_text_fn=clean_recovery_text,
) -> tuple[str, list[dict]]:
    fallback_tools: list[str] = []
    for step in task.steps:
        fallback_tools.extend(step.fallback_tools)
    fallback_tools = sorted(set(fallback_tools))
    fallback_guidance = ""
    if "researcher_discovery" in task.intent_tags:
        fallback_guidance = (
            "For researcher ranking requests, prioritize publication-centric recovery tools in this order: "
            "rank_researchers_by_activity, search_openalex_works, search_pubmed_advanced, get_pubmed_author_profile. "
            "Avoid clinical-trials-only fallback unless explicitly requested.\n"
        )
    prompt = (
        "Perform one fallback recovery pass before final synthesis.\n"
        f"Objective: {task.objective}\n"
        f"Intent tags: {', '.join(task.intent_tags)}\n"
        f"Fallback tools to prioritize: {', '.join(fallback_tools) if fallback_tools else 'N/A'}\n"
        f"{fallback_guidance}"
        "You must execute at least one relevant tool call unless no relevant tool exists.\n"
        "Required output fields: selected_tools, why_chosen, key_results, remaining_gaps.\n"
        "Use explicit citations where possible."
    )
    try:
        raw, trace_entries = await run_runner_turn_with_trace_fn(runner, session_id, user_id, prompt)
    except Exception as exc:
        return format_step_execution_error_fn(exc, fallback_tools), []
    for entry in trace_entries:
        entry["phase"] = "fallback_recovery"
    return clean_recovery_text_fn(raw), trace_entries


async def complete_remaining_steps(
    runner,
    session_id: str,
    user_id: str,
    task,
    state_store,
    *,
    execute_step_fn,
    evaluate_quality_gates_fn=evaluate_quality_gates,
    render_quality_gate_message_fn=render_quality_gate_message,
    run_fallback_recovery_fn=None,
    print_fn=print,
) -> dict:
    if run_fallback_recovery_fn is None:
        raise ValueError("run_fallback_recovery_fn is required")

    task.fallback_recovery_notes = ""
    task.fallback_tool_trace = []
    for idx in range(task.current_step_index + 1, len(task.steps)):
        step_text = await execute_step_fn(runner, session_id, user_id, task, idx)
        state_store.save_task(task, note=f"step_{idx + 1}_completed")
        print_fn(step_text)

    quality = evaluate_quality_gates_fn(task)
    print_fn("\n" + render_quality_gate_message_fn(quality))
    if not quality["passed"]:
        print_fn("\nRunning one fallback recovery pass...")
        recovery, recovery_trace = await run_fallback_recovery_fn(runner, session_id, user_id, task)
        task.fallback_tool_trace = recovery_trace
        task.fallback_recovery_notes = recovery or ""
        quality = evaluate_quality_gates_fn(task)
    return quality


def format_checkpoint_reason(reason: str) -> str:
    mapping = {
        "pre_evidence_execution": "Before bulk evidence collection",
        "quality_gap_spike": "Quality/uncertainty spike detected",
        "repeated_tool_failures": "Repeated tool failures detected",
        "uncertainty_spike": "Uncertainty spike detected",
        "pre_final_after_intent_change": "Intent changed before final synthesis",
        "feedback_replan": "Plan updated from user feedback",
        "queued_feedback_pending": "Queued feedback pending application",
    }
    key = str(reason or "").strip()
    return mapping.get(key, key.replace("_", " ") if key else "unspecified")


def print_checkpoint_plan(
    task,
    *,
    active_plan_version_fn,
    format_checkpoint_reason_fn=format_checkpoint_reason,
    print_fn=print,
) -> None:
    print_fn("\n[Checkpoint Plan]")
    if task.latest_plan_delta:
        delta = task.latest_plan_delta
        print_fn("What changed:")
        print_fn(f"- {delta.summary or 'No structural changes.'}")
        if delta.added_steps:
            print_fn(f"- Added: {', '.join(delta.added_steps)}")
        if delta.removed_steps:
            print_fn(f"- Removed: {', '.join(delta.removed_steps)}")
        if delta.modified_steps:
            print_fn(f"- Modified: {', '.join(delta.modified_steps)}")
        if delta.reordered_steps:
            print_fn(f"- Reordered: {', '.join(delta.reordered_steps)}")
        print_fn("")

    version = active_plan_version_fn(task)
    if version and version.steps:
        print_fn("Remaining plan:")
        for idx, step in enumerate(version.steps, start=1):
            print_fn(f"{idx}. {step.title}")
    else:
        print_fn("Remaining plan: none")

    if task.checkpoint_reason:
        print_fn(f"\nCheckpoint reason: {format_checkpoint_reason_fn(task.checkpoint_reason)}")
