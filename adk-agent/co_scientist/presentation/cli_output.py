"""CLI presentation helpers for interactive agent output."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from co_scientist.domain.models import WorkflowTask


def persist_report_artifacts(
    task: WorkflowTask,
    report: str,
    *,
    reports_dir: Path,
    write_markdown_pdf_fn: Callable[[str, Path, str], str | None],
) -> tuple[Path, Path | None, str | None]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_task_id = re.sub(r"[^a-zA-Z0-9_-]", "_", task.task_id)
    markdown_path = reports_dir / f"{safe_task_id}.md"
    pdf_path = reports_dir / f"{safe_task_id}.pdf"
    normalized_report = (report or "").rstrip() + "\n"
    markdown_path.write_text(normalized_report, encoding="utf-8")
    pdf_error = write_markdown_pdf_fn(
        normalized_report,
        pdf_path,
        f"Workflow Report ({task.task_id})",
    )
    if pdf_error:
        return markdown_path, None, pdf_error
    return markdown_path, pdf_path, None


def print_final_report_with_artifacts(
    task: WorkflowTask,
    quality_report: dict,
    *,
    render_final_report_fn: Callable[[WorkflowTask], str] | Callable[[WorkflowTask, dict], str],
    reports_dir: Path,
    write_markdown_pdf_fn: Callable[[str, Path, str], str | None],
    print_fn: Callable[[str], None] = print,
) -> None:
    report = render_final_report_fn(task, quality_report=quality_report)
    markdown_path, pdf_path, pdf_error = persist_report_artifacts(
        task,
        report,
        reports_dir=reports_dir,
        write_markdown_pdf_fn=write_markdown_pdf_fn,
    )
    print_fn("\n" + "=" * 60)
    print_fn(report)
    print_fn("=" * 60)
    print_fn(f"[report] markdown={markdown_path}")
    if pdf_path:
        print_fn(f"[report] pdf={pdf_path}")
    else:
        print_fn(f"[report] pdf=not_generated ({pdf_error})")


def print_hitl_prompt(*, print_fn: Callable[[str], None] = print) -> None:
    print_fn("\n[HITL Checkpoint]")
    print_fn("Reply with one of:")
    print_fn("  - start (or continue)")
    print_fn("  - any feedback text to revise the remaining plan")
    print_fn("  - stop")
    print_fn("You can also run: status | history | rollback")


def resolve_default_task_id(active_task: WorkflowTask | None, state_store) -> str | None:
    if active_task:
        return active_task.task_id
    latest = state_store.latest_task()
    return latest.task_id if latest else None


def print_revision_history(
    state_store,
    task_id: str,
    *,
    limit: int = 12,
    print_fn: Callable[[str], None] = print,
) -> None:
    revisions = state_store.list_revisions(task_id, limit=limit)
    if not revisions:
        print_fn(f"\nNo revision history found for task {task_id}.")
        return
    print_fn(f"\nRevision history for task {task_id} (offset 0 = latest):")
    for offset, entry in enumerate(revisions):
        note = entry.get("note", "") or "-"
        awaiting_hitl = "yes" if entry.get("awaiting_hitl") else "no"
        print_fn(
            f"  {offset}. {entry.get('revision_id')} | {entry.get('saved_at')} | "
            f"status={entry.get('status')} | step={entry.get('current_step_index')} | "
            f"hitl={awaiting_hitl} | note={note}"
        )


__all__ = [
    "persist_report_artifacts",
    "print_final_report_with_artifacts",
    "print_hitl_prompt",
    "print_revision_history",
    "resolve_default_task_id",
]
