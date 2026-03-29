import pytest
from google.adk.tools.skill_toolset import SkillToolset

from neuro_copilot.skill_loader import (
    EXECUTION_SKILL_DIR_NAMES,
    PLANNER_SKILL_DIR_NAMES,
    REPORT_ASSISTANT_SKILL_DIR_NAMES,
    create_execution_skill_toolset,
    create_planner_skill_toolset,
    create_report_assistant_skill_toolset,
    load_execution_skills,
    load_planner_skills,
    load_report_assistant_skills,
    load_skills,
)


def _write_skill(root, name, *, frontmatter_name=None):
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_name = frontmatter_name or name
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {skill_name}\n"
        "description: test skill\n"
        "---\n\n"
        "Use this skill for tests.\n",
        encoding="utf-8",
    )
    return skill_dir


def test_load_planner_skills_from_repo():
    skills = load_planner_skills()
    assert [skill.name for skill in skills] == list(PLANNER_SKILL_DIR_NAMES)


def test_create_planner_skill_toolset_from_repo():
    skills, toolset = create_planner_skill_toolset()
    assert [skill.name for skill in skills] == list(PLANNER_SKILL_DIR_NAMES)
    assert isinstance(toolset, SkillToolset)


def test_load_execution_skills_from_repo():
    skills = load_execution_skills()
    assert [skill.name for skill in skills] == list(EXECUTION_SKILL_DIR_NAMES)


def test_create_execution_skill_toolset_from_repo():
    skills, toolset = create_execution_skill_toolset()
    assert [skill.name for skill in skills] == list(EXECUTION_SKILL_DIR_NAMES)
    assert isinstance(toolset, SkillToolset)


def test_load_report_assistant_skills_from_repo():
    skills = load_report_assistant_skills()
    assert [skill.name for skill in skills] == list(REPORT_ASSISTANT_SKILL_DIR_NAMES)


def test_create_report_assistant_skill_toolset_from_repo():
    skills, toolset = create_report_assistant_skill_toolset()
    assert [skill.name for skill in skills] == list(REPORT_ASSISTANT_SKILL_DIR_NAMES)
    assert isinstance(toolset, SkillToolset)


def test_load_planner_skills_fails_when_skill_md_missing(tmp_path):
    _write_skill(tmp_path, "structured-data-planning")
    (tmp_path / "archive-dataset-discovery-planning").mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
        load_planner_skills(skills_dir=tmp_path)


def test_load_planner_skills_fails_on_name_directory_mismatch(tmp_path):
    _write_skill(tmp_path, "structured-data-planning")
    _write_skill(
        tmp_path,
        "archive-dataset-discovery-planning",
        frontmatter_name="wrong-name",
    )

    with pytest.raises(ValueError, match="does not match directory name"):
        load_planner_skills(skills_dir=tmp_path)


def test_load_skills_loads_references_assets_and_scripts(tmp_path):
    skill_dir = _write_skill(tmp_path, "structured-data-execution")
    references_dir = skill_dir / "references"
    assets_dir = skill_dir / "assets"
    scripts_dir = skill_dir / "scripts"
    references_dir.mkdir()
    assets_dir.mkdir()
    scripts_dir.mkdir()
    (references_dir / "playbook.md").write_text("reference body", encoding="utf-8")
    (assets_dir / "template.txt").write_text("asset body", encoding="utf-8")
    (scripts_dir / "tool.py").write_text("print('ok')\n", encoding="utf-8")

    skills = load_skills(("structured-data-execution",), skills_dir=tmp_path)

    assert len(skills) == 1
    skill = skills[0]
    assert skill.resources.references["playbook.md"] == "reference body"
    assert skill.resources.assets["template.txt"] == "asset body"
    assert str(skill.resources.scripts["tool.py"]) == "print('ok')\n"


def test_load_skills_fails_when_frontmatter_missing(tmp_path):
    skill_dir = tmp_path / "structured-data-execution"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("Use this skill for tests.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing YAML frontmatter"):
        load_skills(("structured-data-execution",), skills_dir=tmp_path)
