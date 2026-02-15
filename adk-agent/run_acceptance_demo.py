#!/usr/bin/env python3
"""
Deterministic acceptance-demo harness for challenge-aligned scenarios.

Run:
    python run_acceptance_demo.py
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from report_pdf import write_markdown_pdf


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "acceptance" / "acceptance_scenarios.json"
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "acceptance" / "results"

load_dotenv(SCRIPT_DIR / ".env")

CRITERIA = {
    "report_contract": "Report contains Decomposition, Answer, Methodology, and Diagnostics sections.",
    "decomposition_visible": "Decomposition includes explicit executable sub-tasks.",
    "tool_trace_present": "Methodology includes a high-level tool activity summary.",
    "evidence_refs_present": "Report includes at least two citation identifiers.",
    "quality_gate_passed": "Diagnostics reports a passing quality gate.",
    "theme_coverage": "Scenario-specific expected themes are present in the answer.",
    "multi_source_trace": "Tool trace spans at least two distinct source families.",
}

THEME_PATTERNS = {
    "researcher": [r"\bresearcher\b", r"\bauthor\b", r"\binvestigator\b"],
    "affiliation": [r"\baffiliation\b", r"\binstitution\b", r"\buniversity\b", r"\bhospital\b"],
    "pmid": [r"\bpmid\b"],
    "recommendation": [r"\brecommend", r"\bshould\b", r"\bpursue\b"],
    "risk": [r"\brisk\b", r"\bconcern\b", r"\bliability\b"],
    "safety": [r"\bsafety\b", r"\badverse\b", r"\btox", r"\bliability\b"],
    "evidence": [r"\bevidence\b", r"\bcitation", r"\bpmid\b", r"\bnct\d+"],
}

TOOL_FAMILY_MAP = {
    "open_targets": {
        "search_diseases",
        "search_disease_targets",
        "search_targets",
        "get_target_info",
        "check_druggability",
        "get_target_drugs",
        "expand_disease_context",
        "summarize_target_expression_context",
        "summarize_target_competitive_landscape",
        "summarize_target_safety_liabilities",
        "compare_targets_multi_axis",
    },
    "clinical_trials": {
        "search_clinical_trials",
        "get_clinical_trial",
        "summarize_clinical_trials_landscape",
    },
    "pubmed": {
        "search_pubmed",
        "search_pubmed_advanced",
        "get_pubmed_abstract",
        "get_pubmed_paper_details",
        "get_pubmed_author_profile",
    },
    "openalex": {
        "search_openalex_works",
        "search_openalex_authors",
        "rank_researchers_by_activity",
        "get_researcher_contact_candidates",
    },
    "chemistry": {"search_chembl_compounds_for_target"},
    "genomics_pathways": {
        "get_gene_info",
        "search_clinvar_variants",
        "get_clinvar_variant_details",
        "search_gwas_associations",
        "search_reactome_pathways",
        "get_string_interactions",
        "infer_genetic_effect_direction",
    },
    "local_data": {"list_local_datasets", "read_local_dataset"},
}

REQUIRED_REPORT_HEADINGS = ["Decomposition", "Answer", "Methodology", "Diagnostics"]


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _write_markdown_with_pdf(
    markdown_path: Path,
    content: str,
    *,
    title: str,
    enable_pdf: bool = True,
) -> tuple[str, str | None]:
    normalized = (content or "").rstrip() + "\n"
    markdown_path.write_text(normalized, encoding="utf-8")
    if not enable_pdf:
        return "", "PDF export disabled by --no-pdf."
    pdf_path = markdown_path.with_suffix(".pdf")
    pdf_error = write_markdown_pdf(normalized, pdf_path, title=title)
    if pdf_error:
        return "", pdf_error
    return str(pdf_path), None


def _extract_section(report: str, heading: str) -> str:
    anchor = re.search(rf"^##\s+{re.escape(heading)}\s*$", report, flags=re.MULTILINE)
    if not anchor:
        return ""
    start = anchor.end()
    tail = report[start:]
    next_heading = re.search(r"^##\s+", tail, flags=re.MULTILINE)
    if next_heading:
        return tail[: next_heading.start()].strip()
    return tail.strip()


def _extract_evidence_items(report: str) -> list[str]:
    items: set[str] = set()
    evidence_block = _extract_section(report, "Evidence")
    # Backward-compatible parse for explicit evidence sections.
    for raw in evidence_block.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if not item:
            continue
        lowered = item.lower()
        if lowered.startswith("no explicit citation"):
            continue
        items.add(item)

    # New format: citations can appear inline across Answer/Methodology/Fallback sections.
    citation_patterns = [
        r"\bPMID:\d+\b",
        r"\bNCT\d{8}\b",
        r"\bDOI:[^\s,;)\]]+\b",
        r"\bOpenAlex:[A-Za-z0-9]+\b",
        r"https?://openalex\.org/[A-Za-z0-9._/-]+",
    ]
    for pattern in citation_patterns:
        for match in re.findall(pattern, report, flags=re.IGNORECASE):
            items.add(str(match).strip())

    return sorted(items)


def _extract_tool_names(report: str) -> list[str]:
    names = re.findall(r"\[[^\]]+\]\s+([a-zA-Z0-9_]+)\(call_id=", report)
    if names:
        return names
    # Concise format fallback: parse "Tools involved: tool_a, tool_b"
    fallback_names: list[str] = []
    for line in report.splitlines():
        if not line.strip().lower().startswith("- tools involved:"):
            continue
        raw = line.split(":", 1)[1].strip()
        if not raw or raw.lower() == "none":
            continue
        for part in raw.split(","):
            value = part.strip()
            if value:
                fallback_names.append(value)
    return fallback_names


def _tool_family(tool_name: str) -> str:
    for family, tool_names in TOOL_FAMILY_MAP.items():
        if tool_name in tool_names:
            return family
    return "other"


def _has_theme(report_lower: str, theme: str) -> bool:
    patterns = THEME_PATTERNS.get(theme.lower())
    if not patterns:
        return theme.lower() in report_lower
    return any(re.search(pattern, report_lower) for pattern in patterns)


def _score_report(report: str, expected_themes: list[str]) -> tuple[dict[str, bool], dict[str, Any]]:
    report_lower = report.lower()
    section_presence = {
        heading: bool(re.search(rf"^##\s+{re.escape(heading)}\s*$", report, flags=re.MULTILINE))
        for heading in REQUIRED_REPORT_HEADINGS
    }
    decomposition_block = _extract_section(report, "Decomposition")
    decomposition_count = len(re.findall(r"^\s*(?:\d+[.)]|-)\s+.+$", decomposition_block, flags=re.MULTILINE))
    tool_call_count = report.count("call_id=")
    tool_trace_visible = (
        "- Executed tool trace:" in report
        or "- Tool activity summary:" in report
        or "- Tools involved:" in report
    )
    evidence_items = _extract_evidence_items(report)
    quality_gate_match = re.search(r"- Quality gate passed:\s*(yes|no)", report, flags=re.IGNORECASE)
    quality_gate_passed = bool(quality_gate_match and quality_gate_match.group(1).lower() == "yes")
    diagnostics_tool_calls_match = re.search(r"- Tool calls captured:\s*(\d+)", report)
    diagnostics_tool_calls = int(diagnostics_tool_calls_match.group(1)) if diagnostics_tool_calls_match else None
    tool_names = _extract_tool_names(report)
    source_families = sorted({family for family in (_tool_family(name) for name in tool_names) if family != "other"})
    theme_hits = {theme: _has_theme(report_lower, theme) for theme in expected_themes}
    theme_coverage = all(theme_hits.values()) if expected_themes else True

    checks = {
        "report_contract": all(section_presence.values()),
        "decomposition_visible": decomposition_count >= 2,
        "tool_trace_present": tool_trace_visible,
        "evidence_refs_present": len(evidence_items) >= 2,
        "quality_gate_passed": quality_gate_passed,
        "theme_coverage": theme_coverage,
        "multi_source_trace": len(source_families) >= 2,
    }
    metrics = {
        "sections_present": section_presence,
        "decomposition_count": decomposition_count,
        "tool_call_count": tool_call_count,
        "diagnostics_tool_calls": diagnostics_tool_calls,
        "evidence_items": evidence_items,
        "evidence_count": len(evidence_items),
        "tool_names": sorted(set(tool_names)),
        "source_families": source_families,
        "source_family_count": len(source_families),
        "theme_hits": theme_hits,
    }
    return checks, metrics


async def _run_scenario(
    scenario: dict[str, Any],
    *,
    output_dir: Path,
    scenario_timeout_sec: int,
    enable_pdf: bool = True,
) -> dict[str, Any]:
    from agent import run_single_query_async

    scenario_id = str(scenario["id"])
    prompt = str(scenario["prompt"])
    expected_themes = [str(item).lower() for item in scenario.get("expected_themes", [])]
    report_path = output_dir / "reports" / f"{scenario_id}.md"
    state_path = output_dir / "state" / f"{scenario_id}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_title = f"Acceptance Scenario Report ({scenario_id})"
    started = time.perf_counter()

    try:
        report = await asyncio.wait_for(
            run_single_query_async(prompt, state_store_path=state_path),
            timeout=scenario_timeout_sec,
        )
    except asyncio.TimeoutError:
        duration_sec = round(time.perf_counter() - started, 2)
        timeout_message = (
            f"Scenario timed out after {scenario_timeout_sec}s. "
            "Likely causes: external API latency or MCP request timeout."
        )
        report_pdf_file, report_pdf_error = _write_markdown_with_pdf(
            report_path,
            timeout_message,
            title=pdf_title,
            enable_pdf=enable_pdf,
        )
        return {
            "id": scenario_id,
            "challenge": str(scenario.get("challenge", "")),
            "title": str(scenario.get("title", "")),
            "prompt": prompt,
            "status": "timeout",
            "passed": False,
            "duration_sec": duration_sec,
            "checks": {name: False for name in CRITERIA},
            "failed_checks": list(CRITERIA.keys()),
            "metrics": {},
            "error": timeout_message,
            "report_file": str(report_path),
            "report_pdf_file": report_pdf_file,
            "report_pdf_error": report_pdf_error,
        }
    except Exception as exc:  # pragma: no cover - defensive catch for demo runs
        duration_sec = round(time.perf_counter() - started, 2)
        error_message = f"{type(exc).__name__}: {exc}"
        report_pdf_file, report_pdf_error = _write_markdown_with_pdf(
            report_path,
            error_message,
            title=pdf_title,
            enable_pdf=enable_pdf,
        )
        return {
            "id": scenario_id,
            "challenge": str(scenario.get("challenge", "")),
            "title": str(scenario.get("title", "")),
            "prompt": prompt,
            "status": "error",
            "passed": False,
            "duration_sec": duration_sec,
            "checks": {name: False for name in CRITERIA},
            "failed_checks": list(CRITERIA.keys()),
            "metrics": {},
            "error": error_message,
            "report_file": str(report_path),
            "report_pdf_file": report_pdf_file,
            "report_pdf_error": report_pdf_error,
        }

    duration_sec = round(time.perf_counter() - started, 2)
    report_pdf_file, report_pdf_error = _write_markdown_with_pdf(
        report_path,
        report,
        title=pdf_title,
        enable_pdf=enable_pdf,
    )

    if report.lstrip().startswith("## Clarification Needed"):
        checks = {name: False for name in CRITERIA}
        return {
            "id": scenario_id,
            "challenge": str(scenario.get("challenge", "")),
            "title": str(scenario.get("title", "")),
            "prompt": prompt,
            "status": "clarification_needed",
            "passed": False,
            "duration_sec": duration_sec,
            "checks": checks,
            "failed_checks": list(CRITERIA.keys()),
            "metrics": {"clarification_needed": True},
            "error": "Scenario returned a clarification request instead of a final report.",
            "report_file": str(report_path),
            "report_pdf_file": report_pdf_file,
            "report_pdf_error": report_pdf_error,
        }

    checks, metrics = _score_report(report, expected_themes)
    failed_checks = [name for name, ok in checks.items() if not ok]
    passed = not failed_checks
    return {
        "id": scenario_id,
        "challenge": str(scenario.get("challenge", "")),
        "title": str(scenario.get("title", "")),
        "prompt": prompt,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "duration_sec": duration_sec,
        "checks": checks,
        "failed_checks": failed_checks,
        "metrics": metrics,
        "error": None,
        "report_file": str(report_path),
        "report_pdf_file": report_pdf_file,
        "report_pdf_error": report_pdf_error,
    }


async def _run_hitl_probe(
    probe_cfg: dict[str, Any],
    *,
    output_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    from google.adk import Runner
    from google.adk.sessions import InMemorySessionService

    from agent import _start_new_workflow_task, create_agent
    from task_state_store import TaskStateStore

    prompt = str(probe_cfg.get("prompt", "")).strip()
    if not prompt:
        return {
            "id": str(probe_cfg.get("id", "hitl_probe")),
            "title": str(probe_cfg.get("title", "HITL Probe")),
            "status": "skipped",
            "passed": False,
            "checks": {},
            "error": "No prompt configured for HITL probe.",
        }

    session_service = InMemorySessionService()
    try:
        agent, mcp_tools = create_agent()
    except Exception as exc:  # pragma: no cover - defensive catch for demo runs
        return {
            "id": str(probe_cfg.get("id", "hitl_probe")),
            "title": str(probe_cfg.get("title", "HITL Probe")),
            "status": "error",
            "passed": False,
            "checks": {},
            "error": f"Failed to create agent for HITL probe: {type(exc).__name__}: {exc}",
        }
    runner = Runner(
        agent=agent,
        app_name="co_scientist_acceptance_hitl",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="co_scientist_acceptance_hitl",
        user_id="researcher",
    )
    state_path = output_dir / "state" / "hitl_probe.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_store = TaskStateStore(state_path)
    stdout_path = output_dir / "hitl_probe_stdout.log"
    stdout_buffer = StringIO()
    started = time.perf_counter()

    try:
        with contextlib.redirect_stdout(stdout_buffer):
            task = await asyncio.wait_for(
                _start_new_workflow_task(
                    runner,
                    session.id,
                    "researcher",
                    state_store,
                    prompt,
                ),
                timeout=timeout_sec,
            )
    except asyncio.TimeoutError:
        duration_sec = round(time.perf_counter() - started, 2)
        stdout_path.write_text(stdout_buffer.getvalue(), encoding="utf-8")
        if mcp_tools:
            await mcp_tools.close()
        return {
            "id": str(probe_cfg.get("id", "hitl_probe")),
            "title": str(probe_cfg.get("title", "HITL Probe")),
            "status": "timeout",
            "passed": False,
            "duration_sec": duration_sec,
            "checks": {},
            "error": f"HITL probe timed out after {timeout_sec}s.",
            "stdout_log": str(stdout_path),
        }
    except Exception as exc:  # pragma: no cover - defensive catch for demo runs
        duration_sec = round(time.perf_counter() - started, 2)
        stdout_path.write_text(stdout_buffer.getvalue(), encoding="utf-8")
        if mcp_tools:
            await mcp_tools.close()
        return {
            "id": str(probe_cfg.get("id", "hitl_probe")),
            "title": str(probe_cfg.get("title", "HITL Probe")),
            "status": "error",
            "passed": False,
            "duration_sec": duration_sec,
            "checks": {},
            "error": f"{type(exc).__name__}: {exc}",
            "stdout_log": str(stdout_path),
        }

    duration_sec = round(time.perf_counter() - started, 2)
    stdout_path.write_text(stdout_buffer.getvalue(), encoding="utf-8")
    revisions = state_store.list_revisions(task.task_id, limit=6)
    notes = {str(entry.get("note", "")) for entry in revisions}
    checks = {
        "awaiting_hitl_true": bool(task.awaiting_hitl),
        "stopped_at_step_1": task.current_step_index == 0,
        "step_1_completed": bool(task.steps and task.steps[0].status == "completed"),
        "checkpoint_revision_saved": "hitl_checkpoint_opened" in notes,
    }

    if mcp_tools:
        await mcp_tools.close()
    failed_checks = [name for name, ok in checks.items() if not ok]
    return {
        "id": str(probe_cfg.get("id", "hitl_probe")),
        "title": str(probe_cfg.get("title", "HITL Probe")),
        "status": "passed" if not failed_checks else "failed",
        "passed": not failed_checks,
        "duration_sec": duration_sec,
        "checks": checks,
        "failed_checks": failed_checks,
        "task_id": task.task_id,
        "current_step_index": task.current_step_index,
        "awaiting_hitl": bool(task.awaiting_hitl),
        "error": None,
        "stdout_log": str(stdout_path),
    }


def _build_summary_markdown(scoreboard: dict[str, Any]) -> str:
    lines = [
        "# Acceptance Demo Scoreboard",
        "",
        f"- Run ID: `{scoreboard['run_id']}`",
        f"- Generated at (UTC): `{scoreboard['generated_at_utc']}`",
        f"- Config: `{scoreboard['config_path']}`",
        f"- Config SHA256: `{scoreboard['config_sha256']}`",
        "",
        "## Scenario Results",
        "",
        "| Scenario | Challenge | Status | Duration (s) | Decomposition Items | Tool Calls | Evidence IDs | Source Families | Failed Checks |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in scoreboard["scenario_results"]:
        metrics = result.get("metrics", {})
        lines.append(
            "| {id} | {challenge} | {status} | {duration:.2f} | {decomposition} | {tool_calls} | {evidence} | {families} | {failed} |".format(
                id=result.get("id", ""),
                challenge=result.get("challenge", ""),
                status=result.get("status", ""),
                duration=float(result.get("duration_sec", 0.0)),
                decomposition=int(metrics.get("decomposition_count", 0)),
                tool_calls=int(metrics.get("tool_call_count", 0)),
                evidence=int(metrics.get("evidence_count", 0)),
                families=int(metrics.get("source_family_count", 0)),
                failed=", ".join(result.get("failed_checks", [])) or "none",
            )
        )

    lines.extend(["", "## HITL Probe", ""])
    hitl = scoreboard.get("hitl_probe")
    if hitl:
        lines.append(f"- Status: `{hitl.get('status', 'n/a')}`")
        lines.append(f"- Passed: `{hitl.get('passed', False)}`")
        lines.append(f"- Duration (s): `{hitl.get('duration_sec', 0.0)}`")
        if hitl.get("failed_checks"):
            lines.append(f"- Failed checks: `{', '.join(hitl['failed_checks'])}`")
        lines.append(f"- Log: `{hitl.get('stdout_log', 'n/a')}`")
    else:
        lines.append("- Not run.")

    summary = scoreboard["summary"]
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Scenarios evaluated: `{summary['scenarios_evaluated']}`",
            f"- Scenarios passed: `{summary['scenarios_passed']}`",
            f"- Scenarios failed: `{summary['scenarios_failed']}`",
            f"- HITL passed: `{summary['hitl_passed']}`",
            f"- Overall pass: `{summary['overall_pass']}`",
        ]
    )
    return "\n".join(lines) + "\n"


async def _run_acceptance(args: argparse.Namespace) -> tuple[int, Path]:
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    scenarios = list(payload.get("scenarios", []))
    if args.only_scenario:
        allow = set(args.only_scenario)
        scenarios = [item for item in scenarios if str(item.get("id")) in allow]
    if args.limit and args.limit > 0:
        scenarios = scenarios[: args.limit]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    (run_dir / "reports").mkdir(parents=True, exist_ok=True)
    (run_dir / "state").mkdir(parents=True, exist_ok=True)

    print(f"[acceptance] run_id={run_id}")
    print(f"[acceptance] scenarios={len(scenarios)}")

    scenario_results: list[dict[str, Any]] = []
    if args.dry_run:
        for scenario in scenarios:
            scenario_results.append(
                {
                    "id": str(scenario.get("id", "")),
                    "challenge": str(scenario.get("challenge", "")),
                    "title": str(scenario.get("title", "")),
                    "prompt": str(scenario.get("prompt", "")),
                    "status": "skipped",
                    "passed": False,
                    "duration_sec": 0.0,
                    "checks": {},
                    "failed_checks": [],
                    "metrics": {},
                    "error": "Dry run: scenario execution skipped.",
                    "report_file": "",
                    "report_pdf_file": "",
                    "report_pdf_error": "PDF export disabled by --no-pdf." if args.no_pdf else None,
                }
            )
    else:
        for scenario in scenarios:
            scenario_id = str(scenario.get("id", "unknown"))
            print(f"[scenario:start] {scenario_id}")
            result = await _run_scenario(
                scenario,
                output_dir=run_dir,
                scenario_timeout_sec=args.scenario_timeout_sec,
                enable_pdf=not args.no_pdf,
            )
            scenario_results.append(result)
            print(
                f"[scenario:done] {scenario_id} status={result['status']} "
                f"duration={result['duration_sec']}s"
            )

    hitl_probe_result: dict[str, Any] | None = None
    if not args.skip_hitl_probe and not args.dry_run:
        hitl_cfg = payload.get("hitl_probe", {})
        print("[hitl:start]")
        hitl_probe_result = await _run_hitl_probe(
            hitl_cfg,
            output_dir=run_dir,
            timeout_sec=args.hitl_timeout_sec,
        )
        print(
            f"[hitl:done] status={hitl_probe_result.get('status')} "
            f"duration={hitl_probe_result.get('duration_sec', 0.0)}s"
        )

    evaluated = [item for item in scenario_results if item.get("status") != "skipped"]
    scenarios_passed = sum(1 for item in evaluated if item.get("passed"))
    scenarios_failed = sum(1 for item in evaluated if not item.get("passed"))
    hitl_passed = bool(hitl_probe_result.get("passed")) if hitl_probe_result else args.skip_hitl_probe
    overall_pass = bool(evaluated) and scenarios_failed == 0 and hitl_passed
    if args.dry_run:
        overall_pass = False

    scoreboard = {
        "run_id": run_id,
        "generated_at_utc": _now_iso_utc(),
        "execution_mode": "dry_run" if args.dry_run else "live",
        "pdf_enabled": not args.no_pdf,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256_file(args.config),
        "criteria": CRITERIA,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "scenario_results": scenario_results,
        "hitl_probe": hitl_probe_result,
        "summary": {
            "scenarios_evaluated": len(evaluated),
            "scenarios_passed": scenarios_passed,
            "scenarios_failed": scenarios_failed,
            "hitl_passed": hitl_passed,
            "overall_pass": overall_pass,
        },
    }

    scoreboard_path = run_dir / "scoreboard.json"
    summary_path = run_dir / "summary.md"
    summary_content = _build_summary_markdown(scoreboard)
    summary_pdf_path = summary_path.with_suffix(".pdf")
    summary_path.write_text(summary_content, encoding="utf-8")
    if args.no_pdf:
        summary_pdf_error = "PDF export disabled by --no-pdf."
    else:
        summary_pdf_error = write_markdown_pdf(
            summary_content,
            summary_pdf_path,
            title=f"Acceptance Summary ({run_id})",
        )
    scoreboard["summary_file"] = str(summary_path)
    scoreboard["summary_pdf_file"] = "" if summary_pdf_error else str(summary_pdf_path)
    scoreboard["summary_pdf_error"] = summary_pdf_error
    scoreboard_path.write_text(json.dumps(scoreboard, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"[acceptance] scoreboard={scoreboard_path}")
    print(f"[acceptance] summary={summary_path}")
    if args.no_pdf:
        print("[acceptance] summary_pdf=disabled (--no-pdf)")
    elif summary_pdf_error:
        print(f"[acceptance] summary_pdf=not_generated ({summary_pdf_error})")
    else:
        print(f"[acceptance] summary_pdf={summary_pdf_path}")
    print(
        f"[acceptance] overall_pass={overall_pass} "
        f"scenarios_passed={scenarios_passed}/{len(evaluated)} hitl_passed={hitl_passed}"
    )

    if args.non_strict or args.dry_run:
        return 0, run_dir
    return (0 if overall_pass else 1), run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic acceptance demo harness.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to acceptance scenarios JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory where acceptance artifacts are written.",
    )
    parser.add_argument(
        "--scenario-timeout-sec",
        type=int,
        default=420,
        help="Per-scenario timeout in seconds.",
    )
    parser.add_argument(
        "--hitl-timeout-sec",
        type=int,
        default=240,
        help="HITL probe timeout in seconds.",
    )
    parser.add_argument(
        "--only-scenario",
        action="append",
        default=[],
        help="Run only the specified scenario ID (repeat for multiple).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Run only the first N scenarios from config order.",
    )
    parser.add_argument(
        "--skip-hitl-probe",
        action="store_true",
        help="Skip the HITL checkpoint probe.",
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="Exit 0 even when acceptance checks fail.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and emit artifacts without running scenarios.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF artifact generation and write markdown artifacts only.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    try:
        exit_code, _ = asyncio.run(_run_acceptance(args))
        return exit_code
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as exc:  # pragma: no cover
        print(f"[acceptance] fatal_error={type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
