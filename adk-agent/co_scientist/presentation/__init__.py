"""Presentation layer exports for UI/report rendering helpers."""

from .hitl_summary import (
    clean_model_text,
    default_hitl_subtasks,
    extract_decomposition_subtasks,
    extract_focus_from_step_output,
    extract_labeled_value,
    render_hitl_scope_summary,
)
from .cli_output import (
    persist_report_artifacts,
    print_final_report_with_artifacts,
    print_hitl_prompt,
    print_revision_history,
    resolve_default_task_id,
)

__all__ = [
    "clean_model_text",
    "default_hitl_subtasks",
    "extract_decomposition_subtasks",
    "extract_focus_from_step_output",
    "extract_labeled_value",
    "persist_report_artifacts",
    "print_final_report_with_artifacts",
    "print_hitl_prompt",
    "print_revision_history",
    "resolve_default_task_id",
    "render_hitl_scope_summary",
]
