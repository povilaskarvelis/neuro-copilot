"""
AI Co-Scientist Agent Module

This module exports the root_agent for ADK evaluation framework.
"""
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .workflow import create_native_workflow_agent


# ADK evaluation looks for either `agent` or `root_agent`.
root_agent, _native_mcp_toolset = create_native_workflow_agent()

# Alias for ADK evaluation framework
agent = root_agent

__all__ = ["agent", "root_agent", "create_native_workflow_agent"]
