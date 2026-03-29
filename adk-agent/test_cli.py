#!/usr/bin/env python3
"""Quick CLI test for the agent workflow with BigQuery."""
import asyncio
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("ADK_NATIVE_PREFER_BIGQUERY", "1")

from neuro_copilot.workflow import create_workflow_agent
from google.adk import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part


async def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What drugs target BRAF? Keep it brief."
    print(f"Query: {query}\n")

    agent, mcp = create_workflow_agent(require_plan_approval=False)
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="cli_test", session_service=session_service)
    session = await session_service.create_session(app_name="cli_test", user_id="u1")
    msg = Content(role="user", parts=[Part(text=query)])

    async for event in runner.run_async(session_id=session.id, user_id="u1", new_message=msg):
        content = getattr(event, "content", None)
        author = str(getattr(event, "author", "") or "")
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        text = "".join(p.text for p in parts if hasattr(p, "text") and p.text).strip()
        if text and author:
            print(f"\n===== [{author}] =====")
            print(text[:2000])

    if mcp:
        await mcp.close()
    print("\n--- DONE ---")


if __name__ == "__main__":
    asyncio.run(main())
