from __future__ import annotations

import pytest

import agent
from co_scientist.domain.models import WorkflowTask
from co_scientist.planning.intent import route_query_intent
from co_scientist.planning.revision import parse_revision_intent
from workflow import create_task, render_final_report


@pytest.mark.asyncio
async def test_route_query_intent_uses_heuristic_when_router_missing():
    route = await route_query_intent("Compare LRRK2 vs GBA1 for Parkinson disease.")
    assert route["source"] == "heuristic"
    assert route["request_type"] in {"comparison", "prioritization", "validation", "action_planning", "exploration"}


@pytest.mark.asyncio
async def test_parse_revision_intent_uses_heuristic_when_parser_missing():
    parsed = await parse_revision_intent("Prioritize human genetics evidence first.")
    assert parsed.parser_source == "heuristic"
    assert parsed.objective_adjustments


def test_workflow_task_serialization_has_no_fallback_fields():
    task = create_task("Summarize evidence for target X.")
    payload = task.to_dict()
    assert "fallback_recovery_notes" not in payload
    assert "fallback_tool_trace" not in payload

    payload["fallback_recovery_notes"] = "legacy"
    payload["fallback_tool_trace"] = [{"tool_name": "search_pubmed_advanced"}]
    restored = WorkflowTask.from_dict(payload)
    assert not hasattr(restored, "fallback_recovery_notes")
    assert not hasattr(restored, "fallback_tool_trace")


def test_single_revision_opportunity_enforced_by_history():
    task = create_task("Prioritize two targets for fibrosis.")
    assert agent._revision_opportunity_used(task) is False
    task.hitl_history.append("revise:focus on safety")
    assert agent._revision_opportunity_used(task) is True


def test_final_report_does_not_include_fallback_sections():
    task = create_task("Summarize literature evidence for psychosis biomarkers.")
    for step in task.steps[:-1]:
        step.status = "completed"
        step.output = "Collected evidence with citation PMID:12345678."
    task.steps[-1].status = "completed"
    task.steps[-1].output = (
        "Recommendation: prioritize panel A based on replicated evidence (PMID:12345678).\n"
        "Confidence: medium."
    )

    report = render_final_report(task, quality_report={"unresolved_gaps": []})
    assert "Fallback recovery notes" not in report
    assert "fallback" not in report.lower()
