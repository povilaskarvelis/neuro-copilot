"""Repo-local helpers for loading workflow skills.

This module isolates the current ADK private skill-directory loader behind a
small wrapper so the workflow code does not depend on private ADK APIs
directly.
"""

from __future__ import annotations

from pathlib import Path

from google.adk.skills.models import Skill
from google.adk.tools.skill_toolset import SkillToolset

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
PLANNER_SKILL_DIR_NAMES = (
    "structured-data-planning",
    "archive-dataset-discovery-planning",
)
EXECUTION_SKILL_DIR_NAMES = (
    "structured-data-execution",
    "archive-dataset-discovery-execution",
)
REPORT_ASSISTANT_SKILL_DIR_NAMES = (
    "structured-data-report-followup",
    "archive-dataset-discovery-report-followup",
)


def _load_skill_from_directory(skill_dir: Path) -> Skill:
    """Load a single skill from disk via ADK's current directory loader."""
    from google.adk.skills._utils import _load_skill_from_dir

    return _load_skill_from_dir(skill_dir)


def load_skills(
    skill_dir_names: tuple[str, ...],
    *,
    skills_dir: Path | None = None,
) -> list[Skill]:
    """Load a fixed set of repo-local skills in order."""
    root = Path(skills_dir) if skills_dir is not None else SKILLS_DIR
    return [_load_skill_from_directory(root / name) for name in skill_dir_names]


def create_skill_toolset(
    skill_dir_names: tuple[str, ...],
    *,
    skills_dir: Path | None = None,
) -> tuple[list[Skill], SkillToolset]:
    """Load a fixed set of skills and wrap them in an ADK SkillToolset."""
    skills = load_skills(skill_dir_names, skills_dir=skills_dir)
    return skills, SkillToolset(skills=skills)


def load_planner_skills(*, skills_dir: Path | None = None) -> list[Skill]:
    """Load the planner's repo-local skills in a fixed order."""
    return load_skills(PLANNER_SKILL_DIR_NAMES, skills_dir=skills_dir)


def create_planner_skill_toolset(*, skills_dir: Path | None = None) -> tuple[list[Skill], SkillToolset]:
    """Return the planner skills and their ADK SkillToolset wrapper."""
    return create_skill_toolset(PLANNER_SKILL_DIR_NAMES, skills_dir=skills_dir)


def load_execution_skills(*, skills_dir: Path | None = None) -> list[Skill]:
    """Load the executor's repo-local skills in a fixed order."""
    return load_skills(EXECUTION_SKILL_DIR_NAMES, skills_dir=skills_dir)


def create_execution_skill_toolset(*, skills_dir: Path | None = None) -> tuple[list[Skill], SkillToolset]:
    """Return the executor skills and their toolset wrapper."""
    return create_skill_toolset(EXECUTION_SKILL_DIR_NAMES, skills_dir=skills_dir)


def load_report_assistant_skills(*, skills_dir: Path | None = None) -> list[Skill]:
    """Load the report assistant's repo-local skills in a fixed order."""
    return load_skills(REPORT_ASSISTANT_SKILL_DIR_NAMES, skills_dir=skills_dir)


def create_report_assistant_skill_toolset(
    *,
    skills_dir: Path | None = None,
) -> tuple[list[Skill], SkillToolset]:
    """Return the report assistant skills and their toolset wrapper."""
    return create_skill_toolset(REPORT_ASSISTANT_SKILL_DIR_NAMES, skills_dir=skills_dir)


__all__ = [
    "EXECUTION_SKILL_DIR_NAMES",
    "PLANNER_SKILL_DIR_NAMES",
    "REPORT_ASSISTANT_SKILL_DIR_NAMES",
    "SKILLS_DIR",
    "create_execution_skill_toolset",
    "create_planner_skill_toolset",
    "create_report_assistant_skill_toolset",
    "create_skill_toolset",
    "load_execution_skills",
    "load_planner_skills",
    "load_report_assistant_skills",
    "load_skills",
]
