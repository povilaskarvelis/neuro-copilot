"""
AI Co-Scientist ADK-native runner.

Usage:
    adk run co_scientist
    adk web .
    python agent.py
    python agent.py --query "Evaluate LRRK2 in Parkinson disease"
    python agent.py --benchmark --query "What is the Open Targets score for HTRA1 with vital capacity?"
    python agent.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Awaitable, Callable

from dotenv import load_dotenv
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, FunctionResponse, Part

from co_scientist.workflow import create_workflow_agent


load_dotenv()
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

MCP_SERVER_DIR = Path(__file__).resolve().parent.parent / "research-mcp"
REQUEST_CONFIRMATION_FUNCTION_NAME = "adk_request_confirmation"
VERTEX_ENABLE_ENV = "GOOGLE_GENAI_USE_VERTEXAI"
VERTEX_REQUIRED_ENVS = ("GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION")
TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}

ConfirmationHandler = Callable[[object], Awaitable[dict]]
BENCHMARK_OUTPUT_AUTHORS = frozenset({"benchmark_qa", "benchmark_executor", "benchmark_loop"})


def _is_truthy_env(value: str) -> bool:
    return value.strip().lower() in TRUTHY_ENV_VALUES


def _is_vertex_mode_enabled() -> bool:
    return _is_truthy_env(str(os.environ.get(VERTEX_ENABLE_ENV, "")))


def validate_runtime_configuration() -> tuple[bool, str]:
    if _is_vertex_mode_enabled():
        missing = [name for name in VERTEX_REQUIRED_ENVS if not str(os.environ.get(name, "")).strip()]
        if missing:
            return (
                False,
                (
                    "Vertex AI mode is enabled but missing required env vars: "
                    + ", ".join(missing)
                ),
            )
    else:
        api_key = str(os.environ.get("GOOGLE_API_KEY", "")).strip()
        if not api_key or api_key == "your-api-key-here":
            return (
                False,
                (
                    "Missing model auth. Configure GOOGLE_API_KEY for local mode, "
                    "or set GOOGLE_GENAI_USE_VERTEXAI=true with GOOGLE_CLOUD_PROJECT "
                    "and GOOGLE_CLOUD_LOCATION."
                ),
            )

    if not (MCP_SERVER_DIR / "server.js").exists():
        return (
            False,
            f"MCP server not found at {MCP_SERVER_DIR}. Make sure research-mcp/server.js exists.",
        )

    return True, ""


def _extract_request_confirmation_call(event) -> object | None:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None)
    if not parts:
        return None
    for part in parts:
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        if str(getattr(function_call, "name", "")).strip() == REQUEST_CONFIRMATION_FUNCTION_NAME:
            return function_call
    return None


def _extract_confirmation_payload(function_call) -> tuple[str, dict]:
    args = getattr(function_call, "args", {}) or {}
    if not isinstance(args, dict):
        return "", {}
    tool_confirmation = args.get("toolConfirmation", {})
    if not isinstance(tool_confirmation, dict):
        return "", {}
    hint = str(tool_confirmation.get("hint", "")).strip()
    payload = tool_confirmation.get("payload", {})
    if not isinstance(payload, dict):
        payload = {}
    return hint, payload


def _extract_original_tool_name(function_call) -> str:
    args = getattr(function_call, "args", {}) or {}
    if not isinstance(args, dict):
        return "unknown_tool"
    original = args.get("originalFunctionCall", {})
    if not isinstance(original, dict):
        return "unknown_tool"
    name = str(original.get("name", "")).strip()
    return name or "unknown_tool"


def _render_confirmation_payload(payload: dict) -> str:
    if not payload:
        return "No confirmation payload was provided."
    schema = str(payload.get("schema", "")).strip().lower()
    lines: list[str] = []
    if schema == "evidence_continuation_checkpoint.v1":
        missing_evidence = [str(item).strip() for item in payload.get("missing_evidence", []) if str(item).strip()]
        if missing_evidence:
            lines.append("Missing evidence:")
            lines.extend([f"- {item}" for item in missing_evidence[:12]])
        next_actions = [str(item).strip() for item in payload.get("proposed_next_actions", []) if str(item).strip()]
        if next_actions:
            lines.append("Proposed next actions:")
            lines.extend([f"- {item}" for item in next_actions[:12]])
        rationale = str(payload.get("rationale", "")).strip()
        if rationale:
            lines.append(f"Rationale: {rationale}")
        if not lines:
            return json.dumps(payload, indent=2, ensure_ascii=True)
        return "\n".join(lines)

    objective = str(payload.get("objective", "")).strip()
    if objective:
        lines.append(f"Objective: {objective}")
    steps = [str(item).strip() for item in payload.get("steps", []) if str(item).strip()]
    if steps:
        lines.append("Steps:")
        lines.extend([f"{idx}. {step}" for idx, step in enumerate(steps[:10], start=1)])
    tools = [str(item).strip() for item in payload.get("proposed_tools", []) if str(item).strip()]
    if tools:
        lines.append("Proposed tools:")
        lines.extend([f"- {name}" for name in tools[:20]])
    evidence_sources = [str(item).strip() for item in payload.get("evidence_sources", []) if str(item).strip()]
    if evidence_sources:
        lines.append("Evidence sources:")
        lines.extend([f"- {source}" for source in evidence_sources[:12]])
    stop_conditions = [str(item).strip() for item in payload.get("stop_conditions", []) if str(item).strip()]
    if stop_conditions:
        lines.append("Stop conditions:")
        lines.extend([f"- {item}" for item in stop_conditions[:12]])
    if not lines:
        return json.dumps(payload, indent=2, ensure_ascii=True)
    return "\n".join(lines)


async def _prompt_for_confirmation(function_call) -> dict:
    hint, payload = _extract_confirmation_payload(function_call)
    original_tool_name = _extract_original_tool_name(function_call)
    schema = str(payload.get("schema", "")).strip().lower()
    is_evidence_checkpoint = schema == "evidence_continuation_checkpoint.v1"

    print("\n" + "=" * 60)
    print("Human Confirmation Required")
    print("=" * 60)
    print(f"Pending tool call: {original_tool_name}")
    if hint:
        print(f"Hint: {hint}")
    print("\nConfirmation payload:")
    print(_render_confirmation_payload(payload))
    print("\nRespond with:")
    print("- approve")
    if is_evidence_checkpoint:
        print("- stop")
        print("- stop: <feedback>")
    else:
        print("- revise: <feedback>")

    while True:
        raw = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: input("\nDecision> ").strip(),
        )
        lowered = raw.lower().strip()
        if lowered in {"approve", "approved", "a", "yes", "y"}:
            return {
                "confirmed": True,
                "payload": {"decision": "approve"},
            }
        if is_evidence_checkpoint and lowered.startswith("stop:"):
            feedback = raw.split(":", 1)[1].strip()
            return {
                "confirmed": True,
                "payload": {"decision": "stop", "feedback": feedback},
            }
        if is_evidence_checkpoint and lowered in {"stop", "s", "reject", "rejected", "no", "n", "revise", "r"}:
            return {
                "confirmed": True,
                "payload": {"decision": "stop"},
            }
        if is_evidence_checkpoint and lowered.startswith("revise:"):
            feedback = raw.split(":", 1)[1].strip()
            return {
                "confirmed": True,
                "payload": {"decision": "stop", "feedback": feedback},
            }
        if lowered.startswith("revise:"):
            feedback = raw.split(":", 1)[1].strip()
            if feedback:
                return {
                    "confirmed": True,
                    "payload": {"decision": "revise", "feedback": feedback},
                }
            print("Please include revision feedback after 'revise:'.")
            continue
        if lowered in {"revise", "r", "reject", "rejected", "no", "n"}:
            feedback = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: input("Revision feedback> ").strip(),
            )
            feedback = feedback.strip()
            if not feedback:
                print("Revision feedback is required for a revise decision.")
                continue
            return {
                "confirmed": True,
                "payload": {"decision": "revise", "feedback": feedback},
            }
        if is_evidence_checkpoint:
            print("Invalid response. Use 'approve' or 'stop'.")
        else:
            print("Invalid response. Use 'approve' or 'revise: <feedback>'.")


async def _run_native_workflow_turn(
    runner: Runner,
    session_id: str,
    user_id: str,
    prompt: str,
    *,
    confirmation_handler: ConfirmationHandler | None = None,
) -> str:
    """Run one turn against the ADK workflow graph and return synthesized text."""
    current_message = Content(role="user", parts=[Part(text=prompt)])
    final_fallback = "(No response generated)"

    while True:
        partial_by_author: dict[str, str] = {}
        final_by_author: dict[str, str] = {}
        fallback_chunks: list[str] = []
        request_confirmation_call = None

        async for event in runner.run_async(
            session_id=session_id,
            user_id=user_id,
            new_message=current_message,
        ):
            if request_confirmation_call is None:
                request_confirmation_call = _extract_request_confirmation_call(event)

            content = getattr(event, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue

            text = "".join(
                part.text
                for part in parts
                if isinstance(getattr(part, "text", None), str)
            ).strip()
            if not text:
                continue

            author = str(getattr(event, "author", "") or "").strip()
            if not author:
                continue

            fallback_chunks.append(text)
            if bool(getattr(event, "partial", False)):
                partial_by_author[author] = f"{partial_by_author.get(author, '')}{text}"
                continue

            is_final = getattr(event, "is_final_response", None)
            if callable(is_final) and bool(is_final()):
                final_by_author[author] = f"{partial_by_author.pop(author, '')}{text}".strip() or text
                continue

            if author in BENCHMARK_OUTPUT_AUTHORS:
                continue
            partial_by_author[author] = f"{partial_by_author.get(author, '')}{text}"

        if request_confirmation_call is not None:
            if confirmation_handler is None:
                _, payload = _extract_confirmation_payload(request_confirmation_call)
                schema = str(payload.get("schema", "")).strip().lower() if isinstance(payload, dict) else ""
                if schema == "evidence_continuation_checkpoint.v1":
                    confirmation_response = {
                        "confirmed": True,
                        "payload": {
                            "decision": "approve",
                            "feedback": "No confirmation handler configured; continuing additional evidence collection.",
                        },
                    }
                else:
                    confirmation_response = {
                        "confirmed": True,
                        "payload": {"decision": "approve"},
                    }
            else:
                confirmation_response = await confirmation_handler(request_confirmation_call)

            current_message = Content(
                role="user",
                parts=[
                    Part(
                        function_response=FunctionResponse(
                            id=str(getattr(request_confirmation_call, "id", "")).strip(),
                            name=REQUEST_CONFIRMATION_FUNCTION_NAME,
                            response=confirmation_response,
                        )
                    )
                ],
            )
            continue

        for preferred_author in (
            "benchmark_qa",
            "benchmark_executor",
            "benchmark_loop",
            "report_synthesizer",
            "general_qa",
            "clarifier",
            "report_assistant",
            "research_workflow",
            "co_scientist_router",
        ):
            candidate = final_by_author.get(preferred_author, "").strip()
            if candidate:
                return candidate

        if final_by_author:
            for author in sorted(final_by_author.keys(), reverse=True):
                candidate = final_by_author.get(author, "").strip()
                if candidate:
                    return candidate

        merged_fallback = "\n".join(chunk for chunk in fallback_chunks if chunk).strip()
        final_fallback = merged_fallback or final_fallback
        return final_fallback


async def run_native_interactive_async() -> None:
    """Run the ADK-native workflow agent in an interactive terminal session."""
    print("=" * 60)
    print("AI Co-Scientist (ADK-native workflow mode)")
    print("=" * 60)

    is_valid, error_message = validate_runtime_configuration()
    if not is_valid:
        print(f"\nERROR: {error_message}")
        print("\nTo fix this for local mode:")
        print("1. Open .env in the adk-agent folder")
        print("2. Set GOOGLE_API_KEY to a valid key")
        print("3. Get one at: https://aistudio.google.com/apikey")
        print("\nTo fix this for Vertex mode:")
        print("1. Set GOOGLE_GENAI_USE_VERTEXAI=true")
        print("2. Set GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION")
        print("3. Authenticate with `gcloud auth application-default login`")
        return

    workflow_agent, mcp_tools = create_workflow_agent(require_plan_approval=True)
    session_service = InMemorySessionService()
    runner = Runner(
        agent=workflow_agent,
        app_name="co_scientist_native",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="co_scientist_native",
        user_id="researcher",
    )

    print("\nADK-native workflow ready")
    print("Type your biomedical research question. Commands: help | quit")

    try:
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: input("\nYou: ").strip(),
            )
            lowered = user_input.lower().strip()
            if lowered in {"quit", "exit", "q"}:
                print("\nGoodbye")
                return
            if not user_input:
                continue
            if lowered == "help":
                print("\nCommands: help | quit")
                print("This mode uses ADK workflow agents for orchestration.")
                continue

            response = await _run_native_workflow_turn(
                runner,
                session.id,
                "researcher",
                user_input,
                confirmation_handler=_prompt_for_confirmation,
            )
            print(f"\nAssistant:\n{response}")
    finally:
        if mcp_tools is not None:
            await mcp_tools.close()


def run_native_interactive() -> None:
    """Sync wrapper for ADK-native interactive mode."""
    asyncio.run(run_native_interactive_async())


async def run_single_query_native_async(query: str) -> str:
    """Run one query using ADK workflow agents only."""
    return await run_single_query_native_with_confirmation_async(
        query,
        confirmation_handler=_prompt_for_confirmation,
    )


async def run_single_query_native_with_confirmation_async(
    query: str,
    *,
    confirmation_handler: ConfirmationHandler | None,
    benchmark_mode: bool = False,
) -> str:
    """Run one query with caller-selected confirmation behavior."""
    is_valid, error_message = validate_runtime_configuration()
    if not is_valid:
        raise RuntimeError(error_message)

    session_service = InMemorySessionService()
    app_name = (
        "co_scientist_native_benchmark_query"
        if benchmark_mode
        else "co_scientist_native_single_query"
    )
    workflow_agent, mcp_tools = create_workflow_agent(
        require_plan_approval=not benchmark_mode,
        benchmark_mode=benchmark_mode,
    )
    runner = Runner(
        agent=workflow_agent,
        app_name=app_name,
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name=app_name,
        user_id="researcher",
    )

    try:
        return await _run_native_workflow_turn(
            runner,
            session.id,
            "researcher",
            query,
            confirmation_handler=confirmation_handler,
        )
    finally:
        if mcp_tools is not None:
            await mcp_tools.close()


def run_single_query_native(query: str) -> str:
    """Sync wrapper for ADK-native single-query mode."""
    return asyncio.run(run_single_query_native_async(query))


async def run_single_query_native_benchmark_async(query: str) -> str:
    """Run one query in the benchmark-optimized execution profile."""
    return await run_single_query_native_with_confirmation_async(
        query,
        confirmation_handler=None,
        benchmark_mode=True,
    )


def run_single_query_native_benchmark(query: str) -> str:
    """Sync wrapper for benchmark-optimized single-query mode."""
    return asyncio.run(run_single_query_native_benchmark_async(query))


async def run_interactive_async() -> None:
    """Compatibility alias for ADK-native interactive mode."""
    await run_native_interactive_async()


def run_interactive() -> None:
    """Compatibility alias for ADK-native interactive mode."""
    run_native_interactive()


async def run_single_query_async(query: str) -> str:
    """Compatibility alias for ADK-native single-query mode."""
    return await run_single_query_native_async(query)


def run_single_query(query: str) -> str:
    """Compatibility alias for ADK-native single-query mode."""
    return run_single_query_native(query)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Co-Scientist")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Run a single query and exit.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the query using the benchmark-optimized execution profile.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.benchmark and not args.query.strip():
        raise SystemExit("--benchmark requires --query.")
    if args.query.strip():
        if args.benchmark:
            print(run_single_query_native_benchmark(args.query.strip()))
        else:
            print(run_single_query(args.query.strip()))
        return
    run_interactive()


if __name__ == "__main__":
    main()
