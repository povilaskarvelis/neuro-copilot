from __future__ import annotations

from co_scientist.runtime.quality_gates import evaluate_quality_gates
from co_scientist.runtime.tool_registry import ToolDescriptor, ToolRegistry
from workflow import create_task


def _synthetic_trace(tool_name: str, outcome: str = "ok", refs: list[str] | None = None) -> dict:
    return {
        "tool_name": tool_name,
        "outcome": outcome,
        "evidence_refs": refs or [],
        "response_contract": {
            "version": "mcp_response_v1",
            "valid": True,
            "issues": [],
            "text_part_count": 1,
            "has_structured_content": True,
        },
    }


def test_dynamic_planner_builds_non_template_graph():
    task = create_task(
        "Prioritize two fibrosis targets and stress-test contradictory evidence before recommending next experiments.",
        request_type_override="prioritization",
        intent_tags_override=["prioritization", "target_comparison", "genetics_direction"],
        use_dynamic_planner=True,
    )
    assert task.planner_mode == "dynamic"
    assert task.planner_graph
    assert len(task.steps) >= 2
    assert any(step.subgoal_id.startswith("sg_dynamic_") or step.subgoal_id.startswith("sg_") for step in task.steps[:-1])
    assert task.steps[-1].subgoal_id == "sg_final_report"


def test_tool_registry_scales_without_static_list_edits():
    registry = ToolRegistry()
    registry._tools["search_pubmed_advanced"] = ToolDescriptor(
        name="search_pubmed_advanced",
        description="Search PubMed with advanced biomedical query filters",
        capabilities={"literature"},
        source="mcp",
    )
    # Simulate newly added MCP tools discovered dynamically.
    registry._tools["search_multimodal_omics"] = ToolDescriptor(
        name="search_multimodal_omics",
        description="Search omics studies and cohorts for disease signatures",
        capabilities={"genetics", "literature"},
        source="mcp",
    )
    ranked = registry.rank_tools(
        query="find omics and genetics evidence for direction of effect",
        capability_hints={"genetics"},
        candidates=registry.names(),
        k=12,
    )
    assert "search_multimodal_omics" in ranked


def test_quality_gate_confidence_and_must_fail_conditions():
    task = create_task(
        "Validate evidence for target X and provide recommendation.",
        use_dynamic_planner=True,
    )
    # Populate synthetic execution evidence.
    for step in task.steps[:-1]:
        step.status = "completed"
        step.output = "Evidence indicates potential benefit [PMID:12345678]."
        step.tool_trace = [_synthetic_trace("search_pubmed_advanced", refs=["PMID:12345678"])]
    task.steps[-1].status = "completed"
    task.steps[-1].output = "Recommendation: Prioritize target X due to convergent evidence [PMID:12345678]."
    report = evaluate_quality_gates(task)
    assert "quality_score" in report
    assert report["quality_confidence"] in {"high", "medium", "low"}
    assert isinstance(report["passed"], bool)

    # Must-fail condition: contract violation in structured tool output.
    task.steps[0].tool_trace = [
        {
            **_synthetic_trace("search_pubmed_advanced", refs=["PMID:12345678"]),
            "response_contract": {"version": "mcp_response_v1", "valid": False, "issues": ["payload missing"]},
        }
    ]
    report_fail = evaluate_quality_gates(task)
    assert report_fail["passed"] is False
