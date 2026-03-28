"""Repo-local helpers for loading workflow skills.

This module provides a small stable loader for repo-local skills so the
workflow code does not depend on ADK private APIs that may disappear across
releases.
"""

from __future__ import annotations

from pathlib import Path
import re

import yaml
from google.adk.skills.models import Frontmatter
from google.adk.skills.models import Resources
from google.adk.skills.models import Script
from google.adk.skills.models import Skill
from google.adk.tools.skill_toolset import SkillToolset

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
PLANNER_SKILL_DIR_NAMES = (
    "structured-data-planning",
    "archive-dataset-discovery-planning",
    "clinical-trials-planning",
    "geo-dataset-discovery-planning",
    "oncology-target-validation-planning",
    "comparative-assessment-planning",
    "entity-resolution-planning",
    "safety-risk-interpretation-planning",
)
EXECUTION_SKILL_DIR_NAMES = (
    "structured-data-execution",
    "archive-dataset-discovery-execution",
    "citation-grounding-execution",
    "clinical-trials-execution",
    "variant-interpretation-execution",
    "geo-dataset-discovery-execution",
    "oncology-target-validation-execution",
    "evidence-weighting-execution",
    "comparative-assessment-execution",
    "entity-resolution-execution",
    "safety-risk-interpretation-execution",
)
REPORT_ASSISTANT_SKILL_DIR_NAMES = (
    "structured-data-report-followup",
    "archive-dataset-discovery-report-followup",
    "citation-grounding-report-followup",
    "clinical-trials-report-followup",
    "variant-interpretation-report-followup",
    "geo-dataset-discovery-report-followup",
    "oncology-target-validation-report-followup",
    "evidence-weighting-report-followup",
    "comparative-assessment-report-followup",
    "entity-resolution-report-followup",
    "safety-risk-interpretation-report-followup",
)


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<frontmatter>.*?)\n---\s*(?:\n(?P<body>.*))?\Z",
    re.DOTALL,
)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _parse_skill_markdown(skill_md_path: Path) -> tuple[Frontmatter, str]:
    content = _read_text_file(skill_md_path)
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        raise ValueError(f"{skill_md_path} is missing YAML frontmatter delimited by ---")

    frontmatter_raw = yaml.safe_load(match.group("frontmatter")) or {}
    if not isinstance(frontmatter_raw, dict):
        raise ValueError(f"{skill_md_path} frontmatter must be a YAML mapping")

    frontmatter = Frontmatter.model_validate(frontmatter_raw)
    instructions = (match.group("body") or "").strip()
    return frontmatter, instructions


def _load_text_resources(resource_dir: Path) -> dict[str, str]:
    if not resource_dir.exists():
        return {}
    if not resource_dir.is_dir():
        raise ValueError(f"Expected resource directory at {resource_dir}")

    resources: dict[str, str] = {}
    for path in sorted(resource_dir.rglob("*")):
        if not path.is_file():
            continue
        key = path.relative_to(resource_dir).as_posix()
        resources[key] = _read_text_file(path)
    return resources


def _load_script_resources(resource_dir: Path) -> dict[str, Script]:
    return {
        key: Script(src=value)
        for key, value in _load_text_resources(resource_dir).items()
    }


def _load_skill_from_directory(skill_dir: Path) -> Skill:
    """Load a single skill from disk into ADK's public Skill model."""
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.exists():
        raise FileNotFoundError(f"SKILL.md not found in {skill_dir}")

    frontmatter, instructions = _parse_skill_markdown(skill_md_path)
    if frontmatter.name != skill_dir.name:
        raise ValueError(
            f"Skill frontmatter name '{frontmatter.name}' does not match directory name "
            f"'{skill_dir.name}'"
        )

    return Skill(
        frontmatter=frontmatter,
        instructions=instructions,
        resources=Resources(
            references=_load_text_resources(skill_dir / "references"),
            assets=_load_text_resources(skill_dir / "assets"),
            scripts=_load_script_resources(skill_dir / "scripts"),
        ),
    )


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
