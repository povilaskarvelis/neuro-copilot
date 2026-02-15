import pytest

import agent
from agent import _merge_objective_with_revision
from workflow import build_plan_steps, create_task, infer_intent_tags, render_final_report, step_prompt


def test_merge_objective_with_revision_preserves_original_context():
    original = "Find top researchers in schizophrenia."
    revision = "Timeframe should be 2016-2026."

    merged = _merge_objective_with_revision(original, revision)

    assert original in merged
    assert revision in merged
    assert "authoritative update" in merged


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

    assert "## Query" in report
    assert '> "find me top researchers in psychosis"' in report
    assert "## Scope" in report
    assert "Timeframe considered: up to 2026" in report
    assert "## Answer" in report


def test_render_final_report_renders_structured_rao_fields():
    task = create_task("find me top researchers in psychosis")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
        step.reasoning_summary = "Reasoning summary."
        step.actions = ["Called `tool_x` (ok)."]
        step.observations = ["Observed key signal."]

    report = render_final_report(task)

    assert "## Methodology" in report
    assert "### Reasoning" in report
    assert "- Reasoning summary." in report
    assert "### Actions" in report
    assert "### Observations" in report
    assert "- Observed key signal." in report


def test_render_final_report_uses_task_level_fallback_notes():
    task = create_task("find me top researchers in psychosis")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
    task.steps[-1].output = "Final answer only."
    task.fallback_recovery_notes = "selected_tools:\n- search_pubmed_advanced"
    task.fallback_tool_trace = [{"tool_name": "search_pubmed_advanced", "outcome": "ok"}]

    report = render_final_report(task)

    assert "## Fallback Recovery Notes" in report
    assert "### Selected Tools" in report
    assert "Final answer only." in report
    assert "### Tool Activity" in report


def test_render_final_report_includes_explicit_decomposition_section():
    task = create_task("find me top researchers in psychosis")
    for step in task.steps:
        step.status = "completed"
        step.output = "Step output."
    task.status = "completed"
    report = render_final_report(task)

    assert "## Decomposition" in report
    assert "Query disease/topic context and lock timeframe constraints." in report
    assert "Identify topic-matched publications from OpenAlex/PubMed." in report


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
    ):
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

    assert len(objectives) == 2
    assert objectives[0] == initial_query
    assert initial_query in objectives[1]
    assert "User revision to scope/decomposition: your time frame should be 2016-2026" in objectives[1]
    assert clarification_queries == [initial_query]
    assert "[Clarification Needed]" not in capsys.readouterr().out


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
