import os

import pytest

from google.adk.sessions import InMemorySessionService

for _env_name in ("AI_CO_SCIENTIST_POSTGRES_DSN", "POSTGRES_DSN", "DATABASE_URL"):
    os.environ.pop(_env_name, None)

import co_scientist.workflow as workflow
import ui_server
from co_scientist.workflow import (
    STATE_EXECUTOR_ACTIVE_STEP_ID,
    STATE_EXECUTOR_LAST_ERROR,
    STATE_PLAN_PENDING_APPROVAL,
    STATE_PRIOR_RESEARCH,
    STATE_REACT_PARSE_RETRIES,
    STATE_WORKFLOW_TASK,
)
from state_store import JsonTaskStore


class DummyRunner:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    for env_name in ("AI_CO_SCIENTIST_POSTGRES_DSN", "POSTGRES_DSN", "DATABASE_URL"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(
        ui_server,
        "create_workflow_agent",
        lambda require_plan_approval=True: (object(), None),
    )
    monkeypatch.setattr(ui_server, "Runner", DummyRunner)

    runtime = ui_server.UiRuntime(tmp_path / "workflow_tasks.json")
    runtime.session_service = InMemorySessionService()
    runtime.ready = True
    runtime.ready_error = None
    return runtime


def _sample_graph_task_state() -> dict:
    plan = {
        "schema": workflow.PLAN_SCHEMA,
        "objective": "Assess ataxia rare-disease evidence",
        "success_criteria": ["Summarize phenotype and disease evidence"],
        "steps": [
            {
                "id": "S1",
                "goal": "Map phenotype associations",
                "tool_hint": "query_monarch_associations",
                "domains": ["genomics"],
                "completion_condition": "Return phenotype-linked disease associations",
            },
        ],
    }
    task_state = workflow._initialize_task_state_from_plan(
        plan,
        objective_text="Assess ataxia rare-disease evidence",
    )
    workflow._apply_step_execution_result_to_task_state(
        task_state,
        {
            "schema": workflow.STEP_RESULT_SCHEMA,
            "step_id": "S1",
            "status": "completed",
            "step_progress_note": "Collected phenotype-linked associations.",
            "result_summary": "Ataxia maps to established phenotype-driven disease associations.",
            "evidence_ids": ["HP:0001251", "ORPHA:100"],
            "open_gaps": [],
            "suggested_next_searches": [],
            "tools_called": ["query_monarch_associations"],
            "structured_observations": [
                {
                    "observation_type": "phenotype_association",
                    "subject": {"type": "phenotype", "label": "Ataxia", "id": "HP:0001251"},
                    "predicate": "associated_with",
                    "object": {"type": "disease", "label": "Ataxia-telangiectasia", "id": "ORPHA:100"},
                    "supporting_ids": ["HP:0001251", "ORPHA:100"],
                    "source_tool": "query_monarch_associations",
                    "confidence": "medium",
                    "qualifiers": {"mode": "phenotype_to_disease"},
                }
            ],
        },
    )
    return task_state


def test_extract_persistable_session_state_filters_transient_keys():
    payload = {
        STATE_WORKFLOW_TASK: {"objective": "Test"},
        STATE_PRIOR_RESEARCH: [{"objective": "Earlier"}],
        STATE_PLAN_PENDING_APPROVAL: True,
        "temp:executor_buffer": "ignore me",
    }

    extracted = ui_server._extract_persistable_session_state(payload)

    assert extracted == {
        STATE_WORKFLOW_TASK: {"objective": "Test"},
        STATE_PRIOR_RESEARCH: [{"objective": "Earlier"}],
        STATE_PLAN_PENDING_APPROVAL: True,
    }


def test_parse_step_event_text_uses_latest_step_block():
    text = """
### S1 · `completed`

**Goal:** Find disease IDs

**Key Findings**

Retrieved identifiers.

---

### S2 · `completed`

**Goal:** Query target evidence

**Key Findings**

+Found strong evidence.

_Progress: 2/4 steps complete. Next: S3_
""".strip()

    parsed = ui_server._parse_step_event_text(text)

    assert parsed["step_id"] == "S2"
    assert parsed["status"] == "completed"
    assert parsed["goal"] == "Query target evidence"
    assert "strong evidence" in parsed["findings"]


def test_build_step_completed_event_metrics_ignores_non_step_text():
    metrics = ui_server._build_step_completed_event_metrics(
        "_Completed 1 of 4 steps. Next: **S2**. Send `finalize` for a partial summary._"
    )

    assert metrics is None


def test_extract_executor_retry_metrics_from_session_state():
    metrics = ui_server._extract_executor_retry_metrics(
        {
            STATE_EXECUTOR_ACTIVE_STEP_ID: "S1",
            STATE_REACT_PARSE_RETRIES: 1,
            STATE_EXECUTOR_LAST_ERROR: "JSON parse error: Unexpected token",
        }
    )

    assert metrics == {
        "step_id": "S1",
        "retry_count": 1,
        "error": "JSON parse error: Unexpected token",
    }


def test_extract_tool_error_metrics_from_function_response():
    class DummyFunctionResponse:
        name = "run_bigquery_select_query"
        response = {
            "error": True,
            "error_type": "ValueError",
            "message": "Tool 'run_bigquery_select_query' failed: bad SQL",
            "suggestion": "Try a simpler query.",
        }

    metrics = ui_server._extract_tool_error_metrics(DummyFunctionResponse())

    assert metrics == {
        "tool": "run_bigquery_select_query",
        "error_type": "ValueError",
        "message": "Tool 'run_bigquery_select_query' failed: bad SQL",
        "suggestion": "Try a simpler query.",
    }


def test_derive_run_error_message_strips_markdown_noise():
    message = ui_server._derive_run_error_message(
        "## Execution Error\n\nVertex AI quota or rate limit exhausted.\n\n`429 RESOURCE_EXHAUSTED`",
        "Fallback error",
    )

    assert message == "Execution Error Vertex AI quota or rate limit exhausted. 429 RESOURCE_EXHAUSTED"


def test_fire_and_forget_threadsafe_does_not_block_on_future_result(monkeypatch):
    class FakeFuture:
        def __init__(self) -> None:
            self.result_calls = []
            self.callback_count = 0

        def add_done_callback(self, callback) -> None:
            self.callback_count += 1

        def result(self, *args, **kwargs):
            self.result_calls.append((args, kwargs))
            return None

    future = FakeFuture()

    async def sample_coro():
        return None

    def fake_run_coroutine_threadsafe(coro, loop):
        coro.close()
        return future

    monkeypatch.setattr(ui_server.asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    ui_server._fire_and_forget_threadsafe(sample_coro(), object(), label="emit_step_summary:test")

    assert future.callback_count == 1
    assert future.result_calls == []


@pytest.mark.asyncio
async def test_get_or_create_session_rehydrates_persisted_state(runtime):
    runtime.store.save_workflow_session(
        "conv_rehydrate",
        task_id="task_rehydrate",
        state={
            STATE_WORKFLOW_TASK: {"objective": "Restore me", "steps": []},
            STATE_PRIOR_RESEARCH: [{"objective": "Previous iteration"}],
            STATE_PLAN_PENDING_APPROVAL: True,
            "temp:co_scientist_executor_buffer": "discard",
        },
    )

    cs = await runtime._get_or_create_session("conv_rehydrate")
    session = await runtime.session_service.get_session(
        app_name=cs.app_name,
        user_id=runtime.user_id,
        session_id=cs.session_id,
    )

    assert session is not None
    assert session.state[STATE_WORKFLOW_TASK]["objective"] == "Restore me"
    assert session.state[STATE_PRIOR_RESEARCH] == [{"objective": "Previous iteration"}]
    assert session.state[STATE_PLAN_PENDING_APPROVAL] is True
    assert "temp:co_scientist_executor_buffer" not in session.state


@pytest.mark.asyncio
async def test_save_task_with_progress_persists_live_session_snapshot(runtime):
    async def fake_persistable_state(conversation_id: str) -> dict:
        assert conversation_id == "conv_persist"
        return {
            STATE_WORKFLOW_TASK: {"objective": "Persist me", "steps": []},
            STATE_PRIOR_RESEARCH: [{"objective": "Prior report"}],
            STATE_PLAN_PENDING_APPROVAL: False,
        }

    runtime._read_persistable_session_state = fake_persistable_state  # type: ignore[method-assign]

    task = ui_server._make_task("task_persist", "Persist me", "conv_persist")
    await runtime._save_task_with_progress(task, merge_progress=False)

    stored_task = runtime.store.get_task("task_persist")
    snapshot = runtime.store.get_workflow_session("conv_persist")

    assert stored_task is not None
    assert snapshot is not None
    assert snapshot["task_id"] == "task_persist"
    assert snapshot["state"] == {
        STATE_WORKFLOW_TASK: {"objective": "Persist me", "steps": []},
        STATE_PRIOR_RESEARCH: [{"objective": "Prior report"}],
        STATE_PLAN_PENDING_APPROVAL: False,
    }

    debug_payload = await runtime.get_task_workflow_state_debug("task_persist")
    assert debug_payload is not None
    assert debug_payload["source"] == "live"
    assert debug_payload["state"][STATE_WORKFLOW_TASK]["objective"] == "Persist me"


@pytest.mark.asyncio
async def test_get_task_evidence_graph_prefers_live_state(runtime):
    task_state = _sample_graph_task_state()

    async def fake_persistable_state(conversation_id: str) -> dict:
        assert conversation_id == "conv_graph_live"
        return {
            STATE_WORKFLOW_TASK: task_state,
            STATE_PLAN_PENDING_APPROVAL: False,
        }

    runtime._read_persistable_session_state = fake_persistable_state  # type: ignore[method-assign]
    runtime.store.save_workflow_session(
        "conv_graph_live",
        task_id="task_graph_live",
        state={STATE_WORKFLOW_TASK: {"objective": "Persisted placeholder", "steps": []}},
    )
    task = ui_server._make_task("task_graph_live", "Assess ataxia rare-disease evidence", "conv_graph_live")
    runtime.store.save_task(task)

    payload = await runtime.get_task_evidence_graph("task_graph_live")

    assert payload is not None
    assert payload["source"] == "live"
    assert payload["mode"] == "semantic"
    assert payload["summary"]["edge_count"] == 1
    assert payload["elements"]["edges"][0]["data"]["predicate"] == "associated_with"


@pytest.mark.asyncio
async def test_get_task_evidence_graph_falls_back_to_persisted_state(runtime):
    task_state = _sample_graph_task_state()
    task = ui_server._make_task("task_graph_persisted", "Assess ataxia rare-disease evidence", "conv_graph_persisted")
    runtime.store.save_task(task)
    runtime.store.save_workflow_session(
        "conv_graph_persisted",
        task_id="task_graph_persisted",
        state={STATE_WORKFLOW_TASK: task_state},
    )

    payload = await runtime.get_task_evidence_graph("task_graph_persisted")

    assert payload is not None
    assert payload["source"] == "persisted"
    assert payload["mode"] == "semantic"
    assert payload["summary"]["node_count"] == 2
    assert payload["summary"]["edge_count"] == 1
    assert payload["elements"]["nodes"][0]["data"]["type"] in {"disease", "phenotype"}


def test_json_store_persists_runs_and_interrupts_incomplete_runs(tmp_path):
    store = JsonTaskStore(tmp_path / "workflow_tasks.json")
    store.save_run(
        {
            "run_id": "run_123",
            "kind": "new_query",
            "status": "running",
            "task_id": "task_123",
            "logs": [],
            "progress_events": [],
            "progress_summaries": [],
            "created_at": "2026-03-06T00:00:00+00:00",
            "updated_at": "2026-03-06T00:00:00+00:00",
        }
    )

    assert store.get_run("run_123")["status"] == "running"

    updated = store.mark_incomplete_runs_failed("Run interrupted because the server restarted.")

    assert updated == 1
    restored = store.get_run("run_123")
    assert restored is not None
    assert restored["status"] == "failed"
    assert restored["error"] == "Run interrupted because the server restarted."
    assert any(event.get("type") == "run.interrupted" for event in restored["progress_events"])


@pytest.mark.asyncio
async def test_get_run_falls_back_to_persisted_store(runtime):
    runtime.store.save_run(
        {
            "run_id": "run_saved",
            "kind": "feedback_task",
            "status": "failed",
            "task_id": "task_saved",
            "logs": [],
            "progress_events": [],
            "progress_summaries": [],
            "error": "Persisted failure",
            "created_at": "2026-03-06T00:00:00+00:00",
            "updated_at": "2026-03-06T00:00:00+00:00",
        }
    )

    payload = await runtime.get_run("run_saved")

    assert payload is not None
    assert payload["run_id"] == "run_saved"
    assert payload["status"] == "failed"
    assert payload["error"] == "Persisted failure"
