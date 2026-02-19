"""Application service boundary for UI/runtime orchestration."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from google.adk import Runner
from google.adk.sessions import InMemorySessionService

import agent as _agent
from co_scientist.planning import intent as _planning_intent
from co_scientist.planning import revision as _planning_revision
from task_state_store import TaskStateStore
from workflow import RevisionIntent, WorkflowTask


MAIN_APP_NAME = "co_scientist_ui"
CLARIFIER_APP_NAME = "co_scientist_clarifier_ui"
FEEDBACK_PARSER_APP_NAME = "co_scientist_feedback_parser_ui"
PLANNER_APP_NAME = "co_scientist_planner_ui"
TITLE_SUMMARIZER_APP_NAME = "co_scientist_title_summarizer_ui"
PROGRESS_SUMMARIZER_APP_NAME = "co_scientist_progress_summarizer_ui"


@dataclass
class AppRuntimeComponents:
    runner: Runner
    clarifier_runner: Runner
    feedback_parser_runner: Runner
    planner_runner: Runner
    title_summarizer_runner: Runner
    progress_summarizer_runner: Runner
    session_id: str
    clarifier_session_id: str
    feedback_parser_session_id: str
    planner_session_id: str
    title_summarizer_session_id: str
    progress_summarizer_session_id: str
    mcp_tools: Any | None


async def create_runtime_components(
    session_service: InMemorySessionService,
    *,
    user_id: str,
) -> AppRuntimeComponents:
    agent, mcp_tools = _agent.create_agent()
    if mcp_tools is not None:
        await _agent._refresh_tool_registry(mcp_tools)
    runner = Runner(
        agent=agent,
        app_name=MAIN_APP_NAME,
        session_service=session_service,
    )
    clarifier_runner = Runner(
        agent=_agent.create_clarifier_agent(),
        app_name=CLARIFIER_APP_NAME,
        session_service=session_service,
    )
    feedback_parser_runner = Runner(
        agent=_agent.create_feedback_parser_agent(),
        app_name=FEEDBACK_PARSER_APP_NAME,
        session_service=session_service,
    )
    planner_runner = Runner(
        agent=_agent.create_planner_agent(),
        app_name=PLANNER_APP_NAME,
        session_service=session_service,
    )
    title_summarizer_runner = Runner(
        agent=_agent.create_title_summarizer_agent(),
        app_name=TITLE_SUMMARIZER_APP_NAME,
        session_service=session_service,
    )
    progress_summarizer_runner = Runner(
        agent=_agent.create_progress_summarizer_agent(),
        app_name=PROGRESS_SUMMARIZER_APP_NAME,
        session_service=session_service,
    )

    (
        main_session,
        clarifier_session,
        feedback_parser_session,
        planner_session,
        title_summarizer_session,
        progress_summarizer_session,
    ) = await asyncio.gather(
        session_service.create_session(app_name=MAIN_APP_NAME, user_id=user_id),
        session_service.create_session(app_name=CLARIFIER_APP_NAME, user_id=user_id),
        session_service.create_session(app_name=FEEDBACK_PARSER_APP_NAME, user_id=user_id),
        session_service.create_session(app_name=PLANNER_APP_NAME, user_id=user_id),
        session_service.create_session(app_name=TITLE_SUMMARIZER_APP_NAME, user_id=user_id),
        session_service.create_session(app_name=PROGRESS_SUMMARIZER_APP_NAME, user_id=user_id),
    )

    return AppRuntimeComponents(
        runner=runner,
        clarifier_runner=clarifier_runner,
        feedback_parser_runner=feedback_parser_runner,
        planner_runner=planner_runner,
        title_summarizer_runner=title_summarizer_runner,
        progress_summarizer_runner=progress_summarizer_runner,
        session_id=main_session.id,
        clarifier_session_id=clarifier_session.id,
        feedback_parser_session_id=feedback_parser_session.id,
        planner_session_id=planner_session.id,
        title_summarizer_session_id=title_summarizer_session.id,
        progress_summarizer_session_id=progress_summarizer_session.id,
        mcp_tools=mcp_tools,
    )


async def run_runner_turn(runner, session_id: str, user_id: str, prompt: str) -> str:
    return await _agent._run_runner_turn(runner, session_id, user_id, prompt)


async def build_clarification_request(
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
        run_runner_turn_fn=run_runner_turn,
    )


async def start_new_workflow_task(
    runner,
    session_id: str,
    user_id: str,
    state_store: TaskStateStore,
    objective: str,
    planner_runner=None,
    planner_session_id: str | None = None,
    task_id_override: str | None = None,
    created_at_override: str | None = None,
    hitl_history_seed: list[str] | None = None,
) -> WorkflowTask:
    return await _agent._start_new_workflow_task(
        runner,
        session_id,
        user_id,
        state_store,
        objective,
        planner_runner=planner_runner,
        planner_session_id=planner_session_id,
        task_id_override=task_id_override,
        created_at_override=created_at_override,
        hitl_history_seed=hitl_history_seed,
    )


async def draft_model_plan_graph(
    objective: str,
    *,
    planner_runner=None,
    planner_session_id: str | None = None,
    user_id: str = "researcher",
) -> list[dict] | None:
    return await _agent._draft_model_plan_graph(
        objective,
        planner_runner=planner_runner,
        planner_session_id=planner_session_id,
        user_id=user_id,
    )


async def parse_revision_intent(
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
        run_runner_turn_fn=run_runner_turn,
    )


def merge_revision_intents(previous: RevisionIntent | None, incoming: RevisionIntent) -> RevisionIntent:
    return _planning_revision.merge_revision_intents(previous, incoming)


def merge_objective_with_revision_intent(original_objective: str, intent: RevisionIntent) -> str:
    return _planning_revision.merge_objective_with_revision_intent(original_objective, intent)


async def execute_step(runner, session_id: str, user_id: str, task: WorkflowTask, step_idx: int) -> str:
    return await _agent._execute_step(runner, session_id, user_id, task, step_idx)


def evaluate_quality_gates(task: WorkflowTask) -> dict:
    return _agent._evaluate_quality_gates(task)


def should_open_checkpoint(
    task: WorkflowTask,
    next_step,
    quality_state: dict | None = None,
    queued_feedback: list[str] | None = None,
) -> tuple[bool, str]:
    return _agent.should_open_checkpoint(task, next_step, quality_state, queued_feedback)


def gate_ack_token(reason: str, plan_version_id: str | None) -> str | None:
    return _agent._gate_ack_token(reason, plan_version_id)


def resolve_rollback_revision_id(
    state_store: TaskStateStore,
    task_id: str,
    token: str,
) -> tuple[str | None, str | None]:
    return _agent._resolve_rollback_revision_id(state_store, task_id, token)


__all__ = [
    "AppRuntimeComponents",
    "MAIN_APP_NAME",
    "CLARIFIER_APP_NAME",
    "FEEDBACK_PARSER_APP_NAME",
    "PLANNER_APP_NAME",
    "TITLE_SUMMARIZER_APP_NAME",
    "PROGRESS_SUMMARIZER_APP_NAME",
    "build_clarification_request",
    "create_runtime_components",
    "evaluate_quality_gates",
    "execute_step",
    "draft_model_plan_graph",
    "gate_ack_token",
    "merge_objective_with_revision_intent",
    "merge_revision_intents",
    "parse_revision_intent",
    "resolve_rollback_revision_id",
    "run_runner_turn",
    "should_open_checkpoint",
    "start_new_workflow_task",
]
