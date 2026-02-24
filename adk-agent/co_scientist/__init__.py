"""
AI Co-Scientist Agent Module

This module exports the root_agent for ADK evaluation framework.
"""
from dotenv import load_dotenv
from google.adk.apps.app import App

# Load environment variables
load_dotenv()

from .workflow import create_workflow_agent


# ADK evaluation looks for either `agent` or `root_agent`.
root_agent, _native_mcp_toolset = create_workflow_agent(require_plan_approval=True)

app = App(
    name="co_scientist",
    root_agent=root_agent,
)

# Alias for ADK evaluation framework
agent = root_agent

__all__ = ["app", "agent", "root_agent", "create_workflow_agent"]
