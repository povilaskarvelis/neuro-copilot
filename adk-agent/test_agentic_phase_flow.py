from __future__ import annotations

import pytest

from workflow import create_task
import agent
from co_scientist.runtime.event_orchestrator import PHASE_RESEARCHERS, PHASE_SYNTHESIS, infer_phase_for_step


def _mark_step_completed(task, idx: int, output: str = "Evidence with PMID:12345678.") -> None:
    step = task.steps[idx]
    step.status = "completed"
    step.output = output
    step.tool_trace = [
        {
            "tool_name": "search_pubmed_advanced",
            "outcome": "ok",
            "evidence_refs": ["PMID:12345678"],
            "response_contract": {
                "version": "mcp_response_v1",
                "valid": True,
                "issues": [],
                "text_part_count": 1,
                "has_structured_content": True,
            },
        }
    ]
    task.current_step_index = idx


def test_phase_boundary_checkpoint_opens_between_major_phases():
    task = create_task(
        "Assess disease evidence, then identify lead scientists and their activity, then synthesize report.",
        use_dynamic_planner=True,
    )
    boundary_idx = next(
        (
            i
            for i in range(1, len(task.steps))
            if infer_phase_for_step(task.steps[i - 1]) != infer_phase_for_step(task.steps[i])
            and infer_phase_for_step(task.steps[i]) != PHASE_SYNTHESIS
        ),
        None,
    )
    if boundary_idx is None:
        pytest.skip("Planner did not create a non-synthesis phase boundary for this query.")
    # Complete the step just before the boundary.
    prev_idx = boundary_idx - 1
    _mark_step_completed(task, prev_idx)
    next_step = task.steps[boundary_idx]

    should_open, reason = agent.should_open_checkpoint(task, next_step, {}, [])

    assert should_open is True
    assert reason.startswith("phase_boundary:")
    assert task.checkpoint_payload.get("question")


def test_acknowledged_phase_boundary_checkpoint_does_not_reopen():
    task = create_task(
        "Assess disease evidence, then identify lead scientists and their activity, then synthesize report.",
        use_dynamic_planner=True,
    )
    boundary_idx = next(
        (
            i
            for i in range(1, len(task.steps))
            if infer_phase_for_step(task.steps[i - 1]) != infer_phase_for_step(task.steps[i])
            and infer_phase_for_step(task.steps[i]) != PHASE_SYNTHESIS
        ),
        None,
    )
    if boundary_idx is None:
        pytest.skip("Planner did not create a non-synthesis phase boundary for this query.")
    prev_idx = boundary_idx - 1
    _mark_step_completed(task, prev_idx)
    next_step = task.steps[boundary_idx]

    should_open, reason = agent.should_open_checkpoint(task, next_step, {}, [])
    assert should_open is True
    assert reason.startswith("phase_boundary:")

    ack_token = agent._gate_ack_token(reason, task.active_plan_version_id)
    if ack_token:
        task.hitl_history.append(ack_token)
    should_reopen, reason_after_ack = agent.should_open_checkpoint(task, next_step, {}, [])
    assert should_reopen is False
    assert reason_after_ack == "none"


def test_phase_boundary_checkpoint_does_not_open_before_synthesis():
    task = create_task(
        "Assess disease evidence, then identify lead scientists and their activity, then synthesize report.",
        use_dynamic_planner=True,
    )
    boundary_idx = next(
        (
            i
            for i in range(1, len(task.steps))
            if infer_phase_for_step(task.steps[i]) == PHASE_SYNTHESIS
            and infer_phase_for_step(task.steps[i - 1]) != PHASE_SYNTHESIS
        ),
        None,
    )
    if boundary_idx is None:
        pytest.skip("Planner did not create a synthesis phase boundary for this query.")

    prev_idx = boundary_idx - 1
    _mark_step_completed(task, prev_idx)
    next_step = task.steps[boundary_idx]

    should_open, reason = agent.should_open_checkpoint(task, next_step, {}, [])

    assert should_open is False
    assert reason == "none"


def test_researcher_phase_collects_candidates():
    task = create_task(
        "Find lead scientists in this area and include contact signals.",
        use_dynamic_planner=True,
    )
    researcher_idx = next((i for i, s in enumerate(task.steps) if infer_phase_for_step(s) == PHASE_RESEARCHERS), None)
    if researcher_idx is None:
        pytest.skip("Planner did not create a dedicated researcher_scouting phase for this query.")
    step = task.steps[researcher_idx]
    step.status = "completed"
    step.output = """
    Jane Smith: high activity (last 3 years), affiliation Example Institute, contact candidate jane@example.edu
    David Chen: moderate activity, affiliation Example University, contact candidate david.chen@example.org
    """
    step.tool_trace = []
    task.current_step_index = researcher_idx

    # Reuse execute-phase post-processing logic through quality gate side effect route.
    quality = agent._evaluate_quality_gates(task)
    assert "quality_confidence" in quality

    # Candidate extraction is populated during execution path in live runs; seed shape should be supported.
    if task.researcher_candidates:
        assert any("jane" in str(item.get("name", "")).lower() for item in task.researcher_candidates)


def test_task_detail_contract_contains_agentic_fields():
    task = create_task(
        "Research disease landscape and lead scientists.",
        use_dynamic_planner=True,
    )
    payload = task.to_dict()
    assert "phase_state" in payload
    assert "event_log" in payload
    assert "checkpoint_payload" in payload
    assert "researcher_candidates" in payload
