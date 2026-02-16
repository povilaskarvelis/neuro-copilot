import asyncio

import pytest

import agent
from agent import _merge_objective_with_revision
from workflow import (
    RevisionIntent,
    WorkflowStep,
    _extract_decomposition_from_text,
    build_plan_delta,
    build_plan_steps,
    create_task,
    infer_intent_tags,
    initialize_plan_version,
    render_final_report,
    replan_remaining_steps,
    step_prompt,
)


def test_merge_objective_with_revision_preserves_original_context():
    original = "Find top researchers in schizophrenia."
    revision = "Timeframe should be 2016-2026."

    merged = _merge_objective_with_revision(original, revision)

    assert original in merged
    assert revision in merged
    assert "authoritative update" in merged


def test_merge_objective_with_revision_captures_general_directives():
    merged = _merge_objective_with_revision(
        "Find top researchers in schizophrenia.",
        "Use search_openalex_authors first; then include institution info.",
    )

    assert "Revision directives to apply" in merged
    assert "- Use search_openalex_authors first" in merged
    assert "- then include institution info" in merged
    assert "mandatory" in merged


def test_merge_objective_with_revision_handles_empty_original():
    revision = "Only include clinical-stage programs."
    assert _merge_objective_with_revision("", revision) == revision


def test_researcher_typo_maps_to_researcher_intent():
    tags = infer_intent_tags("find me top reseachers in schizophrenia")

    assert "researcher_discovery" in tags
    assert "target_comparison" not in tags


def test_researcher_typo_uses_expert_discovery_plan():
    steps = build_plan_steps("find me top reseachers in schizophrenia")

    assert steps[0].title == "Scope and decomposition"
    assert "rank_researchers_by_activity" in steps[1].recommended_tools


def test_step_prompt_includes_top_researcher_guardrails():
    task = create_task("find me top researchers in psychosis")
    prompt = step_prompt(task, task.steps[1])

    assert "Researcher ranking requirements" in prompt
    assert "rank_researchers_by_activity" in prompt
    assert "Do not switch to a clinical-trials-only fallback" in prompt


def test_step_prompt_with_revision_requires_explicit_acknowledgement():
    revised_objective = _merge_objective_with_revision(
        "find me top researchers in psychosis",
        "current year is 2026",
    )
    task = create_task(revised_objective)
    prompt = step_prompt(task, task.steps[0])

    assert "Explicitly acknowledge this revision" in prompt
    assert "Restate the updated timeframe" in prompt


def test_step_prompt_with_revision_directives_enforces_general_constraints():
    revised_objective = _merge_objective_with_revision(
        "find me top researchers in psychosis",
        "Use search_openalex_authors before search_pubmed_advanced and add institution details.",
    )
    task = create_task(revised_objective)
    step1_prompt = step_prompt(task, task.steps[0])
    step2_prompt = step_prompt(task, task.steps[1])
    step3_prompt = step_prompt(task, task.steps[2])

    assert "Revision directives (authoritative)" in step1_prompt
    assert "revision_alignment" in step1_prompt
    assert "search_openalex_authors" in step2_prompt
    assert "Carry forward revision directives as execution constraints" in step3_prompt


def test_build_plan_steps_with_revision_tool_hints_prioritizes_tools():
    revised_objective = _merge_objective_with_revision(
        "find me top researchers in psychosis",
        "Use search_openalex_authors before search_pubmed_advanced.",
    )
    steps = build_plan_steps(revised_objective)

    assert "search_openalex_authors" in steps[1].recommended_tools
    assert "search_pubmed_advanced" in steps[1].recommended_tools
    assert "Respect user tool preferences when feasible" in steps[1].instruction


def test_revision_phrase_gwas_database_maps_to_gwas_tools_in_plan():
    revised_objective = _merge_objective_with_revision(
        "Compare LRRK2 vs GBA1 as Parkinson's disease targets.",
        "Make sure you use GWAS database in the process.",
    )
    steps = build_plan_steps(revised_objective)

    assert "search_gwas_associations" in steps[1].recommended_tools
    assert "infer_genetic_effect_direction" in steps[1].recommended_tools
    assert steps[1].instruction.lower().startswith("prioritize user-requested tools:")


def test_revision_semantic_genome_wide_phrase_maps_to_gwas_tools():
    revised_objective = _merge_objective_with_revision(
        "Compare LRRK2 vs GBA1 as Parkinson's disease targets.",
        "Use genome-wide association evidence and variant directionality before final ranking.",
    )
    steps = build_plan_steps(revised_objective)

    assert "search_gwas_associations" in steps[1].recommended_tools
    assert "infer_genetic_effect_direction" in steps[1].recommended_tools
    assert "prioritize user-requested tools" in steps[1].instruction.lower()


def test_revision_semantic_clinical_trial_phrase_prioritizes_clinical_tools():
    revised_objective = _merge_objective_with_revision(
        "Compare LRRK2 vs GBA1 as Parkinson's disease targets.",
        "Focus on clinical trial safety outcomes and termination reasons.",
    )
    steps = build_plan_steps(revised_objective)

    safety_step = steps[2]
    assert "search_clinical_trials" in safety_step.recommended_tools
    assert "summarize_clinical_trials_landscape" in safety_step.recommended_tools
    assert "prioritize user-requested tools" in safety_step.instruction.lower()


def test_should_open_checkpoint_when_feedback_is_queued():
    task = create_task("find me top researchers in psychosis")
    task.current_step_index = 0

    should_open, reason = agent.should_open_checkpoint(
        task,
        task.steps[1],
        quality_state={},
        queued_feedback=["Please adjust the focus"],
    )

    assert should_open is True
    assert reason == "queued_feedback_pending"


def test_replan_remaining_steps_freezes_completed_steps():
    task = create_task("find me top researchers in psychosis")
    task.current_step_index = 0
    task.steps[0].status = "completed"
    task.steps[0].output = "Step 1 output"
    initialize_plan_version(task, gate_reason="pre_evidence_execution")
    old_completed = task.steps[0].to_dict()

    revision_intent = RevisionIntent(
        raw_feedback="Put stronger emphasis on recent evidence.",
        priorities=["Emphasize recent evidence"],
        confidence=0.7,
        parser_source="fallback",
    )
    revised_objective = _merge_objective_with_revision(
        task.objective,
        "Put stronger emphasis on recent evidence.",
    )
    version, delta = replan_remaining_steps(
        task,
        revised_objective=revised_objective,
        request_type=task.request_type,
        intent_tags=task.intent_tags,
        revision_intent=revision_intent,
        gate_reason="feedback_replan",
    )

    assert task.steps[0].to_dict() == old_completed
    assert task.active_plan_version_id == version.version_id
    assert delta.to_version_id == version.version_id
    assert all(step.status == "pending" for step in task.steps[1:])


def test_merge_revision_intents_accumulates_prior_feedback():
    previous = RevisionIntent(
        raw_feedback="Include affiliations for each researcher.",
        constraints=["Include affiliations for each researcher"],
        output_preferences=["Use a compact table"],
        confidence=0.62,
        parser_source="model",
    )
    incoming = RevisionIntent(
        raw_feedback="Also prioritize papers from the last 5 years.",
        priorities=["Prioritize papers from the last 5 years"],
        confidence=0.73,
        parser_source="fallback",
    )

    merged = agent._merge_revision_intents(previous, incoming)
    merged_objective = agent._merge_objective_with_revision_intent(
        "Find top schizophrenia researchers.",
        merged,
    )

    assert "Include affiliations for each researcher" in merged.constraints
    assert "Prioritize papers from the last 5 years" in merged.priorities
    assert merged.confidence == 0.73
    assert "affiliations" in merged_objective.lower()
    assert "last 5 years" in merged_objective.lower()


def test_build_plan_delta_detects_added_modified_and_reordered_steps():
    previous = [
        WorkflowStep(step_id="step_1", title="Scope", instruction="Define scope"),
        WorkflowStep(step_id="step_2", title="Collect evidence", instruction="Run evidence tools"),
        WorkflowStep(step_id="step_3", title="Synthesis", instruction="Draft synthesis"),
    ]
    updated = [
        WorkflowStep(step_id="step_1", title="Scope", instruction="Define scope and success criteria"),
        WorkflowStep(step_id="step_2", title="Synthesis", instruction="Draft synthesis with confidence notes"),
        WorkflowStep(step_id="step_3", title="Collect evidence", instruction="Run evidence tools"),
        WorkflowStep(step_id="step_4", title="Affiliation extraction", instruction="Extract affiliations"),
    ]

    delta = build_plan_delta(
        previous,
        updated,
        from_version_id="plan_old",
        to_version_id="plan_new",
    )

    assert delta.from_version_id == "plan_old"
    assert delta.to_version_id == "plan_new"
    assert "Affiliation extraction" in delta.added_steps
    assert "Collect evidence" in delta.reordered_steps
    assert "Scope" in delta.modified_steps


@pytest.mark.asyncio
async def test_parse_revision_intent_uses_fallback_without_parser_runner():
    parsed = await agent._parse_revision_intent(
        "Prioritize clinical evidence; do not include speculative claims."
    )

    assert parsed.parser_source == "fallback"
    assert parsed.raw_feedback
    assert parsed.priorities or parsed.constraints or parsed.exclusions


def test_classify_tool_response_marks_textual_error_payload_as_error():
    outcome, detail = agent._classify_tool_response(
        {
            "content": [
                {
                    "text": "Error in rank_researchers_by_activity: Request failed (429)",
                }
            ]
        }
    )

    assert outcome == "error"
    assert "request failed" in detail.lower()


def test_classify_tool_response_marks_critical_gap_payload_as_degraded():
    outcome, detail = agent._classify_tool_response(
        {
            "content": [
                {
                    "text": "CRITICAL GAP: GWAS API unavailable; fallback uses Open Targets genetics evidence scores.",
                }
            ]
        }
    )

    assert outcome == "degraded"
    assert "critical gap" in detail.lower()


def test_quality_gate_flags_missing_successful_top_ranking():
    task = create_task("find me top researchers in psychosis")
    for step in task.steps:
        step.status = "completed"
        step.output = "Evidence collected."

    task.steps[1].output = "Preliminary list only due to rate limit."
    task.steps[1].tool_trace = [
        {
            "tool_name": "rank_researchers_by_activity",
            "outcome": "error",
            "detail": "Error in rank_researchers_by_activity: Request failed (429)",
        }
    ]

    quality = agent._evaluate_quality_gates(task)

    assert quality["passed"] is False
    assert any(
        "quantitative researcher ranking" in gap.lower()
        or "no successful quantitative researcher ranking call" in gap.lower()
        for gap in quality["unresolved_gaps"]
    )


def test_quality_gate_flags_missing_high_weight_genetics_after_tool_failures():
    task = create_task(
        "Compare LRRK2 vs GBA1 as Parkinson's disease targets with high weight on human genetics and safety."
    )
    for step in task.steps:
        step.status = "completed"
        step.output = "Evidence collected."

    task.steps[1].output = (
        "CRITICAL GAP: human genetics direction evidence failed due to API errors; "
        "unable to retrieve direction-of-effect."
    )
    task.steps[1].tool_trace = [
        {
            "tool_name": "infer_genetic_effect_direction",
            "outcome": "error",
            "detail": "Request failed (500)",
        },
        {
            "tool_name": "search_gwas_associations",
            "outcome": "error",
            "detail": "Operation aborted",
        },
    ]

    quality = agent._evaluate_quality_gates(task)

    assert quality["passed"] is False
    assert any("human genetics" in gap.lower() for gap in quality["unresolved_gaps"])
    assert any("critical missing evidence" in gap.lower() for gap in quality["unresolved_gaps"])


def test_should_open_checkpoint_on_critical_gap_before_final_step():
    task = create_task("Compare LRRK2 vs GBA1 as Parkinson's disease targets.")
    task.current_step_index = 4

    should_open, reason = agent.should_open_checkpoint(
        task,
        task.steps[5],
        quality_state={
            "unresolved_gaps": [],
            "last_step_failures": 0,
            "last_step_output": "CRITICAL GAP: failed due to API errors; unable to retrieve genetics evidence.",
        },
    )

    assert should_open is True
    assert reason == "uncertainty_spike"


def test_should_open_checkpoint_on_service_unavailable_marker():
    task = create_task("Compare LRRK2 vs GBA1 as Parkinson's disease targets.")
    task.current_step_index = 4

    should_open, reason = agent.should_open_checkpoint(
        task,
        task.steps[5],
        quality_state={
            "unresolved_gaps": [],
            "last_step_failures": 0,
            "last_step_output": "CRITICAL GAP: GWAS service unavailable; direction-of-effect is unresolved.",
        },
    )

    assert should_open is True
    assert reason == "uncertainty_spike"


def test_should_not_reopen_pre_final_checkpoint_after_ack():
    task = create_task("Compare LRRK2 vs GBA1 as Parkinson's disease targets.")
    task.current_step_index = 4
    task.hitl_history.append("revise:use more genetics evidence")
    task.active_plan_version_id = "plan_demo_1"
    task.hitl_history.append(agent._gate_ack_token("pre_final_after_intent_change", task.active_plan_version_id))

    should_open, reason = agent.should_open_checkpoint(
        task,
        task.steps[5],
        quality_state={
            "unresolved_gaps": [],
            "last_step_failures": 0,
            "last_step_output": "",
        },
    )

    assert should_open is False
    assert reason == "none"


def test_should_not_reopen_quality_gap_checkpoint_after_ack():
    task = create_task("Compare LRRK2 vs GBA1 as Parkinson's disease targets.")
    task.current_step_index = 2
    task.active_plan_version_id = "plan_demo_quality"
    task.steps[0].status = "completed"
    task.steps[0].tool_trace = [{"tool_name": "search_targets", "outcome": "ok"}]
    task.hitl_history.append(agent._gate_ack_token("quality_gap_spike", task.active_plan_version_id))

    should_open, reason = agent.should_open_checkpoint(
        task,
        task.steps[3],
        quality_state={
            "unresolved_gaps": ["gap_a", "gap_b"],
            "last_step_failures": 0,
            "last_step_output": "",
        },
    )

    assert should_open is False
    assert reason == "none"


def test_should_not_reopen_uncertainty_checkpoint_after_ack():
    task = create_task("Compare LRRK2 vs GBA1 as Parkinson's disease targets.")
    task.current_step_index = 2
    task.active_plan_version_id = "plan_demo_uncertainty"
    task.steps[0].status = "completed"
    task.steps[0].tool_trace = [{"tool_name": "search_targets", "outcome": "ok"}]
    task.hitl_history.append(agent._gate_ack_token("uncertainty_spike", task.active_plan_version_id))

    should_open, reason = agent.should_open_checkpoint(
        task,
        task.steps[3],
        quality_state={
            "unresolved_gaps": [],
            "last_step_failures": 0,
            "last_step_output": "CRITICAL GAP: service unavailable for key evidence.",
        },
    )

    assert should_open is False
    assert reason == "none"


def test_render_final_report_includes_query_scope_and_timeframe():
    revised_objective = _merge_objective_with_revision(
        "find me top researchers in psychosis",
        "current year is 2026",
    )
    task = create_task(revised_objective)
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
    task.steps[-1].output = "Clear ranked answer."
    report = render_final_report(task, quality_report={"passed": False, "evidence_count": 0, "tool_call_count": 1, "coverage_ratio": 1.0, "unresolved_gaps": ["gap"]})

    assert report.startswith("## Answer")
    assert "## Rationale" in report
    assert "### Methodology" in report
    assert "### Limitations" in report
    assert "- gap" in report
    assert report.strip().endswith("Resolve: gap")


def test_render_final_report_renders_structured_rao_fields():
    task = create_task("find me top researchers in psychosis")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
        step.reasoning_summary = "Reasoning summary."
        step.actions = ["Called `tool_x` (ok)."]
        step.observations = ["Observed key signal."]

    report = render_final_report(task)

    assert "### Methodology" in report
    assert "This run executed the following workflow steps:" in report
    assert "Tools used in this run:" in report
    assert "Tool activity summary" in report or "No tool-call trace was recorded" in report


def test_render_final_report_uses_task_level_fallback_notes():
    task = create_task("find me top researchers in psychosis")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
    task.steps[-1].output = "Final answer only."
    task.fallback_recovery_notes = "selected_tools:\n- search_pubmed_advanced"
    task.fallback_tool_trace = [{"tool_name": "search_pubmed_advanced", "outcome": "ok"}]

    report = render_final_report(task)

    assert "### Fallback Notes" in report
    assert "### Selected Tools" in report
    assert "Final answer only." in report


def test_render_final_report_includes_explicit_decomposition_section():
    task = create_task("find me top researchers in psychosis")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
    task.status = "completed"
    report = render_final_report(task)

    assert "## Decomposition" not in report
    assert "This run executed the following workflow steps: 1. Scope and decomposition" in report


def test_render_final_report_avoids_redundant_rationale_subheadings():
    task = create_task("Compare two targets and recommend one.")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
    task.steps[-1].output = (
        "Recommendation: Prioritize A.\n"
        "Rationale Narrative: A has stronger human genetic evidence and fewer safety liabilities.\n"
        "Why this recommendation:\n"
        "A has more robust support in the available data."
    )

    report = render_final_report(task)

    assert "## Rationale" in report
    assert "### Why this recommendation" not in report
    assert "Rationale Narrative:" not in report


def test_render_final_report_decomposition_strips_markdown_wrappers():
    output = """
decomposition_subtasks:
1. **Human Genetics Evidence (High Weight):** Evaluate direction of effect.
2. **Safety Liabilities (High Weight):** Assess known and potential liabilities.
3. **Decision Report:** Provide explicit reasons to deprioritize one target.
""".strip()
    subtasks = _extract_decomposition_from_text(output)

    assert "Human Genetics Evidence (High Weight): Evaluate direction of effect." in subtasks[0]


def test_render_hitl_scope_summary_is_compact_and_clean():
    task = create_task("what are the top researchers in schizophrenia?")
    step_output = """
Request Understanding:
The user wants top schizophrenia researchers.

Findings:
* **Disease Area:** Schizophrenia (MONDO_0005090).
* **Definition of Expert:** A researcher with highly-cited recent schizophrenia publications.
* **Timeframe:** last 5 years.

decomposition_subtasks:
1. **Identify Relevant Publications:** Search schizophrenia literature.
2. **Extract Author Information:** Aggregate authors and affiliations.
3. **Rank Researchers by Activity/Impact:** Use publication and citation metrics.
""".strip()

    summary = agent._render_hitl_scope_summary(task, step_output)

    assert summary.startswith("To find the top researchers in Schizophrenia, I will:")
    assert "**" not in summary
    assert "Interpreted intent:" not in summary
    assert "Proposed plan:" not in summary
    assert "1. Search schizophrenia literature." in summary


def test_researcher_fallback_tool_bundle_avoids_clinical_trials():
    steps = build_plan_steps("find me top researchers in psychosis")
    fallback_tools = steps[1].fallback_tools

    assert "search_clinical_trials" not in fallback_tools
    assert "get_clinical_trial" not in fallback_tools
    assert "get_pubmed_author_profile" in fallback_tools


@pytest.mark.asyncio
async def test_route_query_intent_falls_back_deterministically_without_router():
    route = await agent._route_query_intent("find me top reseachers in schizophrenia")

    assert route["source"] == "deterministic"
    assert "researcher_discovery" in route["intent_tags"]
    assert route["request_type"] == "prioritization"


@pytest.mark.asyncio
async def test_route_query_intent_uses_high_confidence_model_route(monkeypatch):
    async def fake_model_route(intent_router_runner, intent_router_session_id, user_id, query):
        del intent_router_runner, intent_router_session_id, user_id, query
        return {
            "normalized_query": "find top researchers in schizophrenia",
            "request_type": "prioritization",
            "intent_tags": ["researcher_discovery", "prioritization"],
            "confidence": 0.92,
            "reason": "researcher ranking request",
            "source": "model",
        }

    monkeypatch.setattr(agent, "_build_model_intent_route", fake_model_route)

    route = await agent._route_query_intent(
        "find me top reseachers in schizophrenia",
        intent_router_runner=object(),
        intent_router_session_id="session-id",
    )

    assert route["source"] == "hybrid"
    assert route["normalized_query"] == "find top researchers in schizophrenia"
    assert route["intent_tags"] == ["prioritization", "researcher_discovery"]


@pytest.mark.asyncio
async def test_interactive_revise_skips_clarification(monkeypatch, capsys):
    initial_query = "can you find me top researchers in schizophrenia?"
    revise_input = "revise your time frame should be 2016-2026"
    objectives: list[str] = []
    clarification_queries: list[str] = []

    class DummyMcpTools:
        async def get_tools(self):
            return ["dummy_tool"]

        async def close(self):
            return None

    class DummyRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyStateStore:
        def __init__(self, _file_path):
            self._latest = None

        def save_task(self, task, note=None):
            self._latest = task

        def latest_task(self):
            return self._latest

        def get_task(self, _task_id):
            return self._latest

        def list_revisions(self, _task_id, limit=20):
            return []

        def rollback_task(self, _task_id, _revision_id):
            return None

    async def fake_build_clarification_request(query, **kwargs):
        clarification_queries.append(query)
        return None

    async def fake_route_query_intent(query, **kwargs):
        del kwargs
        return agent._default_intent_route(query)

    async def fake_start_new_workflow_task(
        runner,
        session_id,
        user_id,
        state_store,
        objective,
        intent_route=None,
        **kwargs,
    ):
        del kwargs
        del intent_route
        del runner, session_id, user_id
        objectives.append(objective)
        task = agent.create_task(objective)
        task.status = "in_progress"
        task.awaiting_hitl = True
        state_store.save_task(task, note="fake_task_started")
        return task

    inputs = iter([initial_query, revise_input, "quit"])

    def fake_input(_prompt=""):
        try:
            return next(inputs)
        except StopIteration as exc:
            raise AssertionError("Unexpected extra input prompt in interactive loop") from exc

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setattr(agent, "create_agent", lambda: (object(), DummyMcpTools()))
    monkeypatch.setattr(agent, "create_clarifier_agent", lambda: object())
    monkeypatch.setattr(agent, "Runner", DummyRunner)
    monkeypatch.setattr(agent, "TaskStateStore", DummyStateStore)
    monkeypatch.setattr(agent, "_build_clarification_request", fake_build_clarification_request)
    monkeypatch.setattr(agent, "_route_query_intent", fake_route_query_intent)
    monkeypatch.setattr(agent, "_start_new_workflow_task", fake_start_new_workflow_task)
    monkeypatch.setattr("builtins.input", fake_input)

    await agent.run_interactive_async()

    assert len(objectives) == 1
    assert objectives[0] == initial_query
    assert clarification_queries == [initial_query]
    output = capsys.readouterr().out
    assert "[Clarification Needed]" not in output
    assert "[Checkpoint Plan]" in output
    assert "Plan updated from user feedback" in output


@pytest.mark.asyncio
async def test_execute_step_retries_after_missing_tool(monkeypatch):
    call_count = 0

    class DummyMcpTools:
        async def close(self):
            return None

    async def fake_run_runner_turn_with_trace(runner, session_id, user_id, prompt):
        del runner, session_id, user_id, prompt
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ValueError("Tool 'get_pubmed_paper_details' not found.")
        return (
            "Recovered with allowed tools.",
            [
                {
                    "sequence": 1,
                    "call_id": "call-1",
                    "tool_name": "search_openalex_authors",
                    "args": {"query": "schizophrenia"},
                    "outcome": "ok",
                    "detail": "mock response",
                    "phase": "main",
                }
            ],
        )

    monkeypatch.setattr(agent, "_create_step_runner", lambda base_runner, allowed: (object(), DummyMcpTools()))
    monkeypatch.setattr(agent, "_run_runner_turn_with_trace", fake_run_runner_turn_with_trace)
    monkeypatch.setattr(agent, "_should_escalate_allowlist", lambda step, trace_entries, output: False)

    task = agent.create_task("find me top reseachers in schizophrenia")
    output = await agent._execute_step(object(), "session", "researcher", task, 1)

    assert call_count == 2
    assert "Recovered with allowed tools." in output
    assert task.steps[1].status == "completed"


@pytest.mark.asyncio
async def test_execute_step_timeout_marks_step_blocked(monkeypatch):
    class DummyMcpTools:
        async def close(self):
            return None

    async def fake_hanging_turn_with_trace(runner, session_id, user_id, prompt):
        del runner, session_id, user_id, prompt
        await asyncio.sleep(0.05)
        return ("Unexpected completion", [])

    monkeypatch.setattr(agent, "STEP_TURN_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(agent, "_create_step_runner", lambda base_runner, allowed: (object(), DummyMcpTools()))
    monkeypatch.setattr(agent, "_run_runner_turn_with_trace", fake_hanging_turn_with_trace)
    monkeypatch.setattr(agent, "_should_escalate_allowlist", lambda step, trace_entries, output: False)

    task = agent.create_task("find me top researchers in schizophrenia")
    output = await agent._execute_step(object(), "session", "researcher", task, 1)

    assert "timed out" in output.lower()
    assert task.steps[1].status == "blocked"
