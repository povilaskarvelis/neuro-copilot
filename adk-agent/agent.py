"""
AI Co-Scientist Agent

This ADK agent connects to the research-mcp server to provide
drug target discovery capabilities using Gemini.

Usage:
    python agent.py         # Interactive mode
    python agent.py --help  # Show help

Requirements:
    - Node.js (for the MCP server)
    - Google API key in .env file
    - pip install -r requirements.txt

Setup:
    1. Edit .env file and paste your API key
    2. Run: python agent.py
"""
import os
import asyncio
from pathlib import Path
import traceback
import re
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from google.adk import Agent, Runner
from google.adk.tools import McpToolset
from mcp.client.stdio import StdioServerParameters
from google.adk.tools.mcp_tool.mcp_toolset import StdioConnectionParams
from task_state_store import TaskStateStore
from workflow import (
    WorkflowTask,
    create_task,
    extract_evidence_refs,
    render_final_report,
    render_plan,
    render_status,
    step_prompt,
    tool_bundle_for_intent,
)

# Path to the MCP server
MCP_SERVER_DIR = Path(__file__).parent.parent / "research-mcp"


AGENT_INSTRUCTION = """You are an AI co-scientist specializing in preclinical drug target discovery.
Your goal is to help researchers explore biomedical data, generate hypotheses, and evaluate drug targets.

## Available Tools
You have access to 33 tools through the MCP server:

**Disease & Target Discovery:**
- search_diseases: Find diseases by name, get their IDs
- search_disease_targets: Find drug targets associated with a disease
- get_target_info: Get detailed information about a target
- search_targets: Search for targets by gene symbol

**Druggability Assessment:**
- check_druggability: Assess if a target can be targeted by drugs
- get_target_drugs: Find existing drugs for a target

**Clinical Evidence:**
- search_clinical_trials: Search ClinicalTrials.gov
- get_clinical_trial: Get detailed trial info including results
- summarize_clinical_trials_landscape: Aggregate status/phase patterns and common termination reasons

**Chemistry Evidence:**
- search_chembl_compounds_for_target: Find target-linked compounds with potency evidence from ChEMBL

**Literature:**
- search_pubmed: Search PubMed for papers
- get_pubmed_abstract: Get full abstract for a paper
- search_pubmed_advanced: Advanced PubMed search with filters
- get_pubmed_paper_details: PubMed details including authors/affiliations
- get_pubmed_author_profile: Aggregate author publication profile

**Researcher Discovery:**
- search_openalex_works: Search literature with OpenAlex metadata
- search_openalex_authors: Find author entities and institutions
- rank_researchers_by_activity: Rank researchers with transparent activity score
- get_researcher_contact_candidates: Candidate public contact/profile signals

**Gene Information:**
- get_gene_info: Get gene details from NCBI

**Variants, Genomics, and Pathways:**
- search_clinvar_variants: Search ClinVar records
- get_clinvar_variant_details: Detailed ClinVar record metadata
- search_gwas_associations: GWAS trait/gene/rsID associations
- search_reactome_pathways: Pathway lookup in Reactome
- get_string_interactions: Protein interaction network lookup (STRING)

**Ontology Context:**
- expand_disease_context: Expand disease terms into IDs/synonyms/hierarchy for better retrieval coverage

**Expression & Cell Context:**
- summarize_target_expression_context: Summarize target tissue/cell expression context from Open Targets

**Genetic Direction-of-Effect:**
- infer_genetic_effect_direction: Infer risk-increasing vs protective genetic signals from GWAS for gene+disease

**Competitive & Safety Intelligence:**
- summarize_target_competitive_landscape: Summarize phase/disease/mechanism competition density for a target
- summarize_target_safety_liabilities: Summarize adverse liability signals, directions, and tissue contexts

**Comparative Prioritization:**
- compare_targets_multi_axis: Rank multiple targets with transparent weighted scores across evidence axes (supports user-defined custom weights and auto mode selection from goal text)

**Local Data:**
- list_local_datasets: List available local data files
- read_local_dataset: Read a local CSV/TSV file

## Workflow for Drug Target Discovery

1. **Understand the Disease**
   - Use search_diseases to find the disease and get its ID
   - Use expand_disease_context to broaden disease synonyms and ontology context
   - Note related diseases that might share targets

2. **Identify Candidate Targets**
   - Use search_disease_targets to get targets ranked by evidence
   - Note the association scores and evidence types

3. **Evaluate Top Candidates**
   For each promising target:
   - Use get_target_info for biological function and pathways
   - Use check_druggability to assess if it can be targeted by drugs
   - Use get_target_drugs to see existing drug landscape
   - Use search_chembl_compounds_for_target to assess preclinical chemical matter and potency signals
   - Use summarize_target_expression_context to assess tissue/cell specificity context
   - Use infer_genetic_effect_direction to assess risk/protective genetic directionality in disease context
   - Use summarize_target_competitive_landscape to assess how crowded the indication/mechanism space is
   - Use summarize_target_safety_liabilities to identify likely modality-specific safety risks
   - Use compare_targets_multi_axis when comparing or ranking multiple targets for prioritization

4. **Check Clinical Evidence**
   - Use search_clinical_trials to find relevant trials
   - Use summarize_clinical_trials_landscape to identify phase/status patterns and failure signals
   - Use get_clinical_trial for trials that have results
   - Pay special attention to TERMINATED trials - understand WHY they failed

5. **Gather Literature Support**
   - Use search_pubmed for recent research
   - Use get_pubmed_abstract for key papers

6. **Synthesize Recommendation**
   Provide a clear recommendation with:
   - Top target(s) ranked by potential
   - Evidence strength (cite PMIDs, NCT IDs)
   - Druggability assessment
   - Risks (failed trials, competition, safety concerns)
   - Suggested next steps

## Important Guidelines
- Always cite your sources with PMIDs or NCT IDs
- Acknowledge uncertainty when evidence is limited
- Highlight both opportunities AND risks
- Consider failed clinical trials as valuable negative evidence
- If a target has failed in trials, explain why it might still be worth pursuing (or not)

## Response Style
- Be thorough but concise
- Use structured formatting (headers, bullet points)
- Quantify when possible (e.g., "87% association score", "6 drugs in development")
- End with actionable recommendations

## Co-Investigator Workflow Requirements
- When asked to work on a complex request, think and act in explicit steps.
- Cite evidence with PMIDs and NCT IDs whenever possible.
- Be transparent about uncertainty and data gaps.
- Use this general response contract:
  1) Request Understanding
  2) Plan
  3) Execution Log
  4) Checkpoint Note
  5) Findings
  6) Evidence
  7) Limitations & Risks
  8) Next Actions
"""

CLARIFIER_INSTRUCTION = """You are an ambiguity and typo triage assistant for biomedical queries.
Your only job is to decide if clarification is needed BEFORE any research tools run.

Return strict JSON only with this schema:
{
  "needs_clarification": boolean,
  "confidence": number,
  "questions": [string],
  "reason": string
}

Rules:
- Ask clarification for ambiguous abbreviations/acronyms (e.g., ER, AD, PD, MS, RA) when context does not disambiguate.
- Ask clarification when a key biomedical entity likely contains a typo or malformed identifier that could change interpretation.
- Do NOT ask clarification for minor spelling mistakes when intent is still clear from context.
- Do not ask clarification for harmless wording/style issues.
- Keep questions concise and actionable.
- If no clarification is needed, set questions to [].
"""


AMBIGUOUS_ABBREVIATIONS: dict[str, dict[str, list[str]]] = {
    "ER": {
        "options": ["Estrogen receptor (ESR1/ESR2)", "Endoplasmic reticulum pathway"],
        "disambiguators": ["estrogen receptor", "esr1", "esr2", "endoplasmic reticulum", "er stress"],
    },
    "AD": {
        "options": ["Alzheimer disease", "Atopic dermatitis", "Autosomal dominant context"],
        "disambiguators": ["alzheimer", "atopic dermatitis", "autosomal dominant"],
    },
    "PD": {
        "options": ["Parkinson disease", "Pharmacodynamics", "PD-1/PD-L1 axis context"],
        "disambiguators": ["parkinson", "pharmacodynamic", "pd-1", "pd-l1", "programmed death"],
    },
    "MS": {
        "options": ["Multiple sclerosis", "Mass spectrometry"],
        "disambiguators": ["multiple sclerosis", "mass spectrometry", "proteomics"],
    },
    "RA": {
        "options": ["Rheumatoid arthritis", "Retinoic acid signaling"],
        "disambiguators": ["rheumatoid arthritis", "retinoic acid"],
    },
}


def _find_ambiguous_abbreviations(query: str) -> list[tuple[str, list[str]]]:
    lowered = query.lower()
    matches: list[tuple[str, list[str]]] = []
    for abbr, cfg in AMBIGUOUS_ABBREVIATIONS.items():
        if not re.search(rf"\b{abbr}\b", query):
            continue
        if any(hint in lowered for hint in cfg["disambiguators"]):
            continue
        matches.append((abbr, cfg["options"]))
    return matches


def _build_deterministic_clarification_request(query: str) -> str | None:
    matches = _find_ambiguous_abbreviations(query)
    if not matches:
        return None

    lines = ["I need a quick clarification before I run tools:"]
    for abbr, options in matches[:3]:
        option_text = " or ".join(options)
        lines.append(f"- `{abbr}` could mean {option_text}. Which one do you mean?")
    lines.append("Reply with a short clarification and I will continue.")
    return "\n".join(lines)


def _merge_query_with_clarification(original_query: str, clarification: str) -> str:
    return (
        f"{original_query}\n"
        f"User clarification: {clarification.strip()}\n"
        "Use this clarification as the intended meaning for ambiguous abbreviations."
    )


def _extract_json_payload(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = cleaned[start : end + 1]
    try:
        payload = json.loads(snippet)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def _contains_malformed_biomedical_identifier(query: str) -> bool:
    tokens = re.findall(r"\b[A-Za-z0-9_]+\b", query)
    for token in tokens:
        upper = token.upper()
        if upper.startswith("ENSG") and not re.fullmatch(r"ENSG\d{11}", upper):
            return True
        if upper.startswith("MONDO_") and not re.fullmatch(r"MONDO_\d{7}", upper):
            return True
        if upper.startswith("EFO_") and not re.fullmatch(r"EFO_\d+", upper):
            return True
        if upper.startswith("NCT") and not re.fullmatch(r"NCT\d{8}", upper):
            return True
        if upper.startswith("PMID") and not re.fullmatch(r"PMID\d{5,9}", upper):
            return True
    return False


def _looks_like_low_value_typo_clarification(query: str, questions: list[str], reason: str) -> bool:
    combined = " ".join(questions + [reason]).lower()
    typo_like = any(
        phrase in combined
        for phrase in ["did you mean", "did you intend", "possible typo", "spelling", "misspell"]
    )
    if not typo_like:
        return False
    if _contains_malformed_biomedical_identifier(query):
        return False
    return True


async def _build_model_clarification_request(
    clarifier_runner,
    clarifier_session_id: str,
    user_id: str,
    query: str,
) -> str | None:
    prompt = (
        "Analyze whether this biomedical query requires clarification before tools run.\n"
        f"Query: {query}\n"
        "Return strict JSON only."
    )
    raw = await _run_runner_turn(clarifier_runner, clarifier_session_id, user_id, prompt)
    payload = _extract_json_payload(raw)
    if not payload:
        return None

    needs_clarification = bool(payload.get("needs_clarification"))
    if not needs_clarification:
        return None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < 0.8:
        return None

    questions = [str(q).strip() for q in payload.get("questions", []) if str(q).strip()]
    if not questions:
        return None
    reason = str(payload.get("reason", "")).strip()
    if _looks_like_low_value_typo_clarification(query, questions, reason):
        return None

    lines = ["I need a quick clarification before I run tools:"]
    for question in questions[:2]:
        if question.startswith("-"):
            lines.append(question)
        else:
            lines.append(f"- {question}")
    lines.append("Reply with a short clarification and I will continue.")
    return "\n".join(lines)


async def _build_clarification_request(
    query: str,
    *,
    clarifier_runner=None,
    clarifier_session_id: str | None = None,
    user_id: str = "researcher",
) -> str | None:
    deterministic = _build_deterministic_clarification_request(query)
    if deterministic:
        return deterministic
    if clarifier_runner is None or not clarifier_session_id:
        return None
    return await _build_model_clarification_request(
        clarifier_runner,
        clarifier_session_id,
        user_id,
        query,
    )


def create_clarifier_agent():
    """Create a no-tool clarification agent for ambiguity/typo triage."""
    return Agent(
        name="clarifier",
        model="gemini-2.5-flash",
        instruction=CLARIFIER_INSTRUCTION,
        tools=[],
    )


def create_agent(tool_filter: list[str] | None = None):
    """Create the ADK agent with MCP tools."""
    # Configure MCP server connection
    server_params = StdioServerParameters(
        command="node",
        args=["server.js"],
        cwd=str(MCP_SERVER_DIR),
    )
    
    connection_params = StdioConnectionParams(
        server_params=server_params,
        timeout=90.0,
    )

    mcp_tools = None
    agent_tools = []
    if tool_filter is None or len(tool_filter) > 0:
        # Connect to the MCP server. If tool_filter is provided, enforce it for this runner.
        mcp_tools = McpToolset(
            connection_params=connection_params,
            tool_filter=tool_filter,
        )
        agent_tools = [mcp_tools]

    # Create the agent
    agent = Agent(
        name="co_scientist",
        model="gemini-2.5-flash",
        instruction=AGENT_INSTRUCTION,
        tools=agent_tools,
    )

    return agent, mcp_tools


STEP_SCOPE_TOOLS = {
    "search_diseases",
    "search_targets",
    "expand_disease_context",
}

STEP_EVIDENCE_BACKSTOP_TOOLS = {
    "search_pubmed_advanced",
    "search_openalex_works",
    "search_openalex_authors",
    "search_clinical_trials",
    "search_disease_targets",
    "search_diseases",
    "search_targets",
    "expand_disease_context",
    "get_target_info",
    "get_gene_info",
    "list_local_datasets",
    "read_local_dataset",
}


def _is_reasoning_only_step(task: WorkflowTask, step_idx: int) -> bool:
    return step_idx == 0 or step_idx == len(task.steps) - 1


def _build_step_allowed_tools(task: WorkflowTask, step_idx: int) -> list[str]:
    step = task.steps[step_idx]

    # Keep request framing and final synthesis deterministic.
    if _is_reasoning_only_step(task, step_idx):
        return sorted(STEP_SCOPE_TOOLS) if step_idx == 0 else []

    preferred, fallback = tool_bundle_for_intent(task.intent_tags)
    allowed = set(step.recommended_tools + step.fallback_tools + preferred + fallback)
    allowed.update(STEP_EVIDENCE_BACKSTOP_TOOLS)
    return sorted(allowed)


def _should_escalate_allowlist(step, trace_entries: list[dict], output: str) -> bool:
    if not step.recommended_tools:
        return False
    if not trace_entries:
        return True
    outcomes = {str(entry.get("outcome", "unknown")) for entry in trace_entries}
    if outcomes and outcomes.issubset({"error", "not_found_or_empty", "no_response"}):
        return True
    lower = (output or "").lower()
    if any(token in lower for token in ["cannot be completed", "insufficient data", "unable to identify"]):
        return True
    return False


def _build_escalated_allowed_tools(task: WorkflowTask, step_idx: int) -> list[str]:
    base = set(_build_step_allowed_tools(task, step_idx))
    # Escalation broadens coverage while keeping synthesis steps tool-free.
    if _is_reasoning_only_step(task, step_idx):
        return sorted(base)
    base.update(
        {
            "search_pubmed",
            "get_pubmed_abstract",
            "get_pubmed_paper_details",
            "get_pubmed_author_profile",
            "get_target_drugs",
            "check_druggability",
            "search_clinvar_variants",
            "search_gwas_associations",
            "summarize_clinical_trials_landscape",
            "summarize_target_expression_context",
            "summarize_target_competitive_landscape",
            "summarize_target_safety_liabilities",
            "compare_targets_multi_axis",
        }
    )
    return sorted(base)


def _create_step_runner(base_runner, allowed_tools: list[str]):
    step_agent, step_mcp_tools = create_agent(tool_filter=allowed_tools)
    step_runner = Runner(
        agent=step_agent,
        app_name=base_runner.app_name,
        session_service=base_runner.session_service,
        artifact_service=getattr(base_runner, "artifact_service", None),
        memory_service=getattr(base_runner, "memory_service", None),
        credential_service=getattr(base_runner, "credential_service", None),
    )
    return step_runner, step_mcp_tools


async def _run_runner_turn(runner, session_id: str, user_id: str, prompt: str) -> str:
    """Run one model turn and return text only."""
    response_text, _ = await _run_runner_turn_with_trace(runner, session_id, user_id, prompt)
    return response_text


def _safe_model_dump(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(exclude_none=True)
        except TypeError:
            dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {"value": dumped}
    return {"value": str(value)}


def _normalize_trace_detail(text: str, *, max_chars: int = 260) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _extract_response_excerpt(response_payload) -> str:
    if not isinstance(response_payload, dict):
        return "No structured response payload captured."

    content = response_payload.get("content")
    snippets: list[str] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    snippets.append(text)

    if not snippets:
        output_payload = response_payload.get("output")
        if isinstance(output_payload, str) and output_payload.strip():
            snippets.append(output_payload)
        elif output_payload is not None:
            snippets.append(str(output_payload))

    if not snippets:
        error_payload = response_payload.get("error")
        if error_payload:
            snippets.append(str(error_payload))

    if not snippets:
        snippets.append(str(response_payload))
    return _normalize_trace_detail(" | ".join(snippets))


def _classify_tool_response(response_payload) -> tuple[str, str]:
    if response_payload is None:
        return "no_response", "Tool call was issued but no response payload was returned."
    if not isinstance(response_payload, dict):
        return "unknown", _normalize_trace_detail(str(response_payload))

    excerpt = _extract_response_excerpt(response_payload)

    explicit_error = bool(response_payload.get("error")) or response_payload.get("isError") is True
    if explicit_error:
        return "error", excerpt

    lower = excerpt.lower()
    not_found_markers = (
        "not found",
        "no results",
        "no matching",
        "no records",
        "no data found",
        "no target data found",
        "no clinical trials found",
        "no expression context found",
        "couldn't find",
        "unable to find",
        "did not find",
        "no evidence found",
    )
    if any(marker in lower for marker in not_found_markers):
        return "not_found_or_empty", excerpt

    return "ok", excerpt


async def _run_runner_turn_with_trace(
    runner,
    session_id: str,
    user_id: str,
    prompt: str,
) -> tuple[str, list[dict]]:
    """Run one model turn and collect both text output and exact tool trace."""
    from google.genai.types import Content, Part

    message = Content(role="user", parts=[Part(text=prompt)])
    response_text = ""
    trace_entries: list[dict] = []
    pending_by_call_id: dict[str, int] = {}
    pending_by_tool_name: dict[str, list[int]] = {}
    sequence = 0

    async for event in runner.run_async(
        session_id=session_id,
        user_id=user_id,
        new_message=message,
    ):
        if not hasattr(event, "content") or not event.content or not hasattr(event.content, "parts"):
            continue
        if not event.content.parts:
            continue
        for part in event.content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text

            function_call = getattr(part, "function_call", None)
            if function_call:
                payload = _safe_model_dump(function_call)
                sequence += 1
                call_id = str(payload.get("id") or f"call-{sequence}")
                tool_name = str(payload.get("name") or "unknown_tool")
                args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
                entry = {
                    "sequence": sequence,
                    "call_id": call_id,
                    "tool_name": tool_name,
                    "args": args,
                    "outcome": "pending",
                    "detail": "",
                    "phase": "main",
                }
                trace_entries.append(entry)
                pending_by_call_id[call_id] = len(trace_entries) - 1
                pending_by_tool_name.setdefault(tool_name, []).append(len(trace_entries) - 1)

            function_response = getattr(part, "function_response", None)
            if function_response:
                payload = _safe_model_dump(function_response)
                call_id = str(payload.get("id") or "")
                tool_name = str(payload.get("name") or "unknown_tool")
                response_payload = payload.get("response")
                outcome, detail = _classify_tool_response(response_payload)

                target_index = pending_by_call_id.get(call_id) if call_id else None
                if target_index is None:
                    for candidate_index in pending_by_tool_name.get(tool_name, []):
                        if trace_entries[candidate_index].get("outcome") == "pending":
                            target_index = candidate_index
                            break

                if target_index is None:
                    sequence += 1
                    trace_entries.append(
                        {
                            "sequence": sequence,
                            "call_id": call_id or f"response-{sequence}",
                            "tool_name": tool_name,
                            "args": {},
                            "outcome": outcome,
                            "detail": detail,
                            "phase": "main",
                        }
                    )
                    continue

                trace_entries[target_index]["outcome"] = outcome
                trace_entries[target_index]["detail"] = detail

    for entry in trace_entries:
        if entry.get("outcome") == "pending":
            entry["outcome"] = "no_response"
            entry["detail"] = "Tool call was issued but no matching function_response event was captured."

    return response_text.strip(), trace_entries


async def _execute_step(runner, session_id: str, user_id: str, task: WorkflowTask, step_idx: int) -> str:
    """Execute a single workflow step and update task status."""
    step = task.steps[step_idx]
    task.status = "in_progress"
    task.current_step_index = step_idx
    step.status = "in_progress"
    task.touch()

    step.allowed_tools = _build_step_allowed_tools(task, step_idx)
    prompt = step_prompt(task, step)

    step_runner, step_mcp_tools = _create_step_runner(runner, step.allowed_tools)
    try:
        output, trace_entries = await _run_runner_turn_with_trace(step_runner, session_id, user_id, prompt)
    finally:
        if step_mcp_tools:
            await step_mcp_tools.close()

    if _should_escalate_allowlist(step, trace_entries, output):
        escalated_tools = _build_escalated_allowed_tools(task, step_idx)
        if set(escalated_tools) != set(step.allowed_tools):
            step.allowed_tools = escalated_tools
            escalated_runner, escalated_mcp_tools = _create_step_runner(runner, step.allowed_tools)
            try:
                escalated_output, escalated_trace = await _run_runner_turn_with_trace(
                    escalated_runner,
                    session_id,
                    user_id,
                    prompt,
                )
            finally:
                if escalated_mcp_tools:
                    await escalated_mcp_tools.close()
            for entry in escalated_trace:
                entry["phase"] = "step_allowlist_escalation"
            if escalated_trace:
                trace_entries.extend(escalated_trace)
            if escalated_output:
                output = escalated_output

    step.output = output if output else "(No response generated)"
    step.evidence_refs = extract_evidence_refs(step.output)
    step.tool_trace = trace_entries
    step.status = "completed" if output else "blocked"
    task.touch()
    return step.output


def _evaluate_quality_gates(task: WorkflowTask) -> dict:
    evidence_count = len({ref for step in task.steps for ref in step.evidence_refs})
    steps_with_output = sum(1 for step in task.steps if step.output and step.output != "(No response generated)")
    coverage_ratio = steps_with_output / len(task.steps) if task.steps else 0.0
    tool_call_count = sum(len(step.tool_trace) for step in task.steps)

    unresolved_gaps: list[str] = []
    combined_output = "\n".join(step.output for step in task.steps if step.output).lower()
    objective_lower = task.objective.lower()
    if "researcher_discovery" in task.intent_tags:
        if "cannot be directly listed" in combined_output or "tool limitation" in combined_output:
            unresolved_gaps.append("Researcher identification appears incomplete due to tool limitations.")
        if not any(token in combined_output for token in ["author", "researcher", "investigator"]):
            unresolved_gaps.append("No explicit researcher entities were reported.")
    if any(token in objective_lower for token in ["target", "druggab", "candidate"]) or "clinical_landscape" in task.intent_tags:
        if any(
            token in combined_output
            for token in [
                "cannot be fulfilled",
                "cannot be completed",
                "insufficient data",
                "no target candidates",
                "unable to identify target",
            ]
        ):
            unresolved_gaps.append("Target/trial assessment appears incomplete based on model self-reported gaps.")
        if not any(token in combined_output for token in ["ensg", "target id", "candidate target", "phase", "nct"]):
            unresolved_gaps.append("No concrete target or clinical-trial entities were detected in the synthesis.")
    if evidence_count == 0:
        unresolved_gaps.append("No citation evidence IDs were detected in the response.")
    if tool_call_count == 0:
        unresolved_gaps.append("No tool calls were captured for the workflow.")
    missing_tool_steps = [
        step.title
        for step in task.steps
        if step.status == "completed" and step.recommended_tools and not step.tool_trace
    ]
    if missing_tool_steps:
        unresolved_gaps.append(
            "Completed steps with recommended tools but no recorded tool execution: "
            + ", ".join(missing_tool_steps)
        )

    passed = evidence_count >= 2 and coverage_ratio >= 0.9 and tool_call_count >= 1 and len(unresolved_gaps) == 0
    return {
        "passed": passed,
        "evidence_count": evidence_count,
        "coverage_ratio": coverage_ratio,
        "tool_call_count": tool_call_count,
        "unresolved_gaps": unresolved_gaps,
    }


def _render_quality_gate_message(report: dict) -> str:
    lines = [
        "[Quality Gate Check]",
        f"- Evidence references found: {report['evidence_count']}",
        f"- Step coverage ratio: {report['coverage_ratio']:.2f}",
        f"- Tool calls captured: {report.get('tool_call_count', 0)}",
    ]
    if report["unresolved_gaps"]:
        lines.append("- Unresolved critical gaps:")
        lines.extend([f"  - {gap}" for gap in report["unresolved_gaps"]])
    else:
        lines.append("- Unresolved critical gaps: none")
    return "\n".join(lines)


def _clean_recovery_text(text: str) -> str:
    if not text:
        return text
    seen = set()
    cleaned_lines: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        normalized = re.sub(r"\s+", " ", line.strip().lower())
        if normalized in {"**3. key results:**", "3. key results:"}:
            if "3-key-results" in seen:
                continue
            seen.add("3-key-results")
        if normalized and normalized in seen and normalized.startswith("**"):
            continue
        if normalized:
            seen.add(normalized)
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


async def _run_fallback_recovery(runner, session_id: str, user_id: str, task: WorkflowTask) -> tuple[str, list[dict]]:
    fallback_tools: list[str] = []
    for step in task.steps:
        fallback_tools.extend(step.fallback_tools)
    fallback_tools = sorted(set(fallback_tools))
    prompt = (
        "Perform one fallback recovery pass before final synthesis.\n"
        f"Objective: {task.objective}\n"
        f"Intent tags: {', '.join(task.intent_tags)}\n"
        f"Fallback tools to prioritize: {', '.join(fallback_tools) if fallback_tools else 'N/A'}\n"
        "You must execute at least one relevant tool call unless no relevant tool exists.\n"
        "Required output fields: selected_tools, why_chosen, key_results, remaining_gaps.\n"
        "Use explicit citations where possible."
    )
    raw, trace_entries = await _run_runner_turn_with_trace(runner, session_id, user_id, prompt)
    for entry in trace_entries:
        entry["phase"] = "fallback_recovery"
    return _clean_recovery_text(raw), trace_entries


async def _complete_remaining_steps(runner, session_id: str, user_id: str, task: WorkflowTask, state_store: TaskStateStore) -> dict:
    print("\nContinuing remaining steps...")
    for idx in range(task.current_step_index + 1, len(task.steps)):
        print(f"\n[Executing Step {idx + 1}] {task.steps[idx].title}")
        step_text = await _execute_step(runner, session_id, user_id, task, idx)
        state_store.save_task(task, note=f"step_{idx + 1}_completed")
        print(step_text)

    quality = _evaluate_quality_gates(task)
    print("\n" + _render_quality_gate_message(quality))
    if not quality["passed"]:
        print("\nRunning one fallback recovery pass...")
        recovery, recovery_trace = await _run_fallback_recovery(runner, session_id, user_id, task)
        if recovery_trace:
            task.steps[-1].tool_trace.extend(recovery_trace)
        if recovery:
            task.steps[-1].output = f"{task.steps[-1].output}\n\nFallback recovery notes:\n{recovery}"
            task.steps[-1].evidence_refs = extract_evidence_refs(task.steps[-1].output)
        quality = _evaluate_quality_gates(task)
    return quality


def _print_hitl_prompt() -> None:
    print("\n[HITL Checkpoint]")
    print("Reply with one of:")
    print("  - continue")
    print("  - revise <new scope>")
    print("  - stop")
    print("You can also run: status | history | rollback")


def _resolve_default_task_id(active_task: WorkflowTask | None, state_store: TaskStateStore) -> str | None:
    if active_task:
        return active_task.task_id
    latest = state_store.latest_task()
    return latest.task_id if latest else None


def _print_revision_history(state_store: TaskStateStore, task_id: str, limit: int = 12) -> None:
    revisions = state_store.list_revisions(task_id, limit=limit)
    if not revisions:
        print(f"\nNo revision history found for task {task_id}.")
        return
    print(f"\nRevision history for task {task_id} (offset 0 = latest):")
    for offset, entry in enumerate(revisions):
        note = entry.get("note", "") or "-"
        awaiting_hitl = "yes" if entry.get("awaiting_hitl") else "no"
        print(
            f"  {offset}. {entry.get('revision_id')} | {entry.get('saved_at')} | "
            f"status={entry.get('status')} | step={entry.get('current_step_index')} | "
            f"hitl={awaiting_hitl} | note={note}"
        )


def _resolve_rollback_revision_id(
    state_store: TaskStateStore,
    task_id: str,
    token: str,
) -> tuple[str | None, str | None]:
    normalized = token.strip()
    if not normalized:
        return None, "Missing revision token."
    if normalized.isdigit():
        offset = int(normalized)
        if offset < 0:
            return None, "Rollback offset cannot be negative."
        revisions = state_store.list_revisions(task_id, limit=max(20, offset + 1))
        if offset >= len(revisions):
            return None, f"Offset {offset} is out of range. Run `history {task_id}` first."
        return str(revisions[offset].get("revision_id", "")), None
    return normalized, None


async def _start_new_workflow_task(
    runner,
    session_id: str,
    user_id: str,
    state_store: TaskStateStore,
    objective: str,
) -> WorkflowTask:
    task = create_task(objective)
    state_store.save_task(task, note="task_created")
    print("\n[Planner Output]")
    print(render_plan(task))
    print(f"\n[Executing Step 1] {task.steps[0].title}")
    step_text = await _execute_step(runner, session_id, user_id, task, 0)
    state_store.save_task(task, note="step_1_completed")
    print(step_text)
    task.awaiting_hitl = True
    task.touch()
    state_store.save_task(task, note="hitl_checkpoint_opened")
    _print_hitl_prompt()
    return task


async def run_interactive_async():
    """Run the agent in interactive mode (async version)."""
    from google.adk.sessions import InMemorySessionService
    
    print("=" * 60)
    print("AI Co-Scientist")
    print("=" * 60)
    
    # Check for API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("\n❌ GOOGLE_API_KEY not configured")
        print("\n   To fix this:")
        print("   1. Open .env file in the adk-agent folder")
        print("   2. Replace 'your-api-key-here' with your actual API key")
        print("   3. Get a free key at: https://aistudio.google.com/apikey")
        return
    
    print("\n✓ API key configured")
    print("Initializing agent with MCP server...")
    
    # Verify MCP server exists
    if not (MCP_SERVER_DIR / "server.js").exists():
        print(f"\n❌ MCP server not found at {MCP_SERVER_DIR}")
        print("\n   Make sure research-mcp/server.js exists")
        return
    
    try:
        agent, mcp_tools = create_agent()
    except Exception as e:
        print(f"\n❌ Failed to create agent: {e}")
        print("\n   Make sure:")
        print("   1. Node.js is installed")
        print("   2. Run: cd ../research-mcp && npm install")
        return
    
    # Get tool count to verify connection
    try:
        tools = await mcp_tools.get_tools()
        print(f"✓ Connected to MCP server ({len(tools)} tools available)")
    except Exception as e:
        print(f"\n❌ MCP connection failed: {e}")
        await mcp_tools.close()
        return
    
    # Create runner with session
    session_service = InMemorySessionService()
    runner = Runner(
        agent=agent,
        app_name="co_scientist",
        session_service=session_service,
    )
    clarifier_runner = Runner(
        agent=create_clarifier_agent(),
        app_name="co_scientist_clarifier",
        session_service=session_service,
    )
    
    # Create a session
    session = await session_service.create_session(
        app_name="co_scientist",
        user_id="researcher",
    )
    clarifier_session = await session_service.create_session(
        app_name="co_scientist_clarifier",
        user_id="researcher",
    )
    state_store = TaskStateStore(Path(__file__).parent / "state" / "workflow_tasks.json")
    active_task: WorkflowTask | None = None
    pending_clarification_query: str | None = None
    pending_clarification_prompt: str | None = None
    
    print("\n✓ Agent ready!")
    print("\nExample queries:")
    print("  - 'Find promising drug targets for Parkinson's disease'")
    print("  - 'Evaluate LRRK2 as a drug target'")
    print("  - 'What clinical trials exist for Alzheimer's gamma-secretase inhibitors?'")
    print("\nCommands: status | resume [task_id] | history [task_id] | rollback <offset|revision_id> [task_id] | help | quit")
    print("At HITL checkpoint: continue | revise <scope> | stop\n")
    print("-" * 60)
    
    try:
        while True:
            try:
                # Use asyncio-compatible input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("\nYou: ").strip()
                )
                
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue

                lowered = user_input.lower().strip()
                if lowered == "help":
                    print("\nCommands: status | resume [task_id] | history [task_id] | rollback <offset|revision_id> [task_id] | help | quit")
                    print("At HITL checkpoint: continue | revise <scope> | stop")
                    continue

                if lowered == "status":
                    if pending_clarification_query:
                        print("\nWaiting for clarification before starting workflow.")
                        if pending_clarification_prompt:
                            print(pending_clarification_prompt)
                        print("Type your clarification, or `stop` to cancel.")
                        continue
                    task = active_task or state_store.latest_task()
                    if not task:
                        print("\nNo workflow task available.")
                    else:
                        print("\n" + render_status(task))
                    continue

                if lowered.startswith("history"):
                    parts = user_input.split(maxsplit=1)
                    task_id = parts[1].strip() if len(parts) > 1 else _resolve_default_task_id(active_task, state_store)
                    if not task_id:
                        print("\nNo workflow task available. Ask a query first.")
                        continue
                    _print_revision_history(state_store, task_id)
                    continue

                if lowered.startswith("rollback"):
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 2:
                        print("\nUse: rollback <offset|revision_id> [task_id]")
                        continue
                    rollback_token = parts[1].strip()
                    task_id = parts[2].strip() if len(parts) > 2 else _resolve_default_task_id(active_task, state_store)
                    if not task_id:
                        print("\nNo workflow task available to rollback.")
                        continue
                    revision_id, error_msg = _resolve_rollback_revision_id(state_store, task_id, rollback_token)
                    if error_msg or not revision_id:
                        print(f"\n{error_msg or 'Could not resolve rollback revision.'}")
                        continue
                    rolled_back = state_store.rollback_task(task_id, revision_id)
                    if not rolled_back:
                        print(f"\nRevision {revision_id} not found for task {task_id}.")
                        continue
                    pending_clarification_query = None
                    pending_clarification_prompt = None
                    active_task = rolled_back
                    print(f"\nRolled back task {task_id} to revision {revision_id}.")
                    print(render_status(active_task))
                    if active_task.awaiting_hitl:
                        _print_hitl_prompt()
                    continue

                if pending_clarification_query:
                    if lowered in {"stop", "cancel"}:
                        pending_clarification_query = None
                        pending_clarification_prompt = None
                        print("\nClarification canceled. Ask a new query.")
                        if active_task and active_task.awaiting_hitl:
                            _print_hitl_prompt()
                        continue
                    clarified_query = _merge_query_with_clarification(
                        pending_clarification_query,
                        user_input,
                    )
                    pending_clarification_query = None
                    pending_clarification_prompt = None
                    follow_up = await _build_clarification_request(
                        clarified_query,
                        clarifier_runner=clarifier_runner,
                        clarifier_session_id=clarifier_session.id,
                        user_id="researcher",
                    )
                    if follow_up:
                        pending_clarification_query = clarified_query
                        pending_clarification_prompt = follow_up
                        print("\n[Clarification Needed]")
                        print(follow_up)
                        print("Type your clarification, or `stop` to cancel.")
                        continue
                    print("\nClarification received. Continuing workflow...")
                    active_task = await _start_new_workflow_task(
                        runner,
                        session.id,
                        "researcher",
                        state_store,
                        clarified_query,
                    )
                    continue

                if (
                    lowered == "continue"
                    and active_task
                    and not active_task.awaiting_hitl
                    and active_task.status in {"pending", "in_progress"}
                    and active_task.current_step_index < len(active_task.steps) - 1
                ):
                    quality = await _complete_remaining_steps(
                        runner, session.id, "researcher", active_task, state_store
                    )
                    active_task.status = "completed"
                    active_task.touch()
                    state_store.save_task(active_task, note="workflow_completed")
                    print("\n" + "=" * 60)
                    print(render_final_report(active_task, quality_report=quality))
                    print("=" * 60)
                    continue

                if lowered in {"continue", "stop"} and not (active_task and active_task.awaiting_hitl):
                    print("\nNo pending checkpoint. Ask a new query or use `status`.")
                    continue
                if lowered.startswith("revise") and not (active_task and active_task.awaiting_hitl):
                    print("\nNo pending checkpoint to revise. Ask a new query first.")
                    continue

                if lowered.startswith("resume"):
                    parts = user_input.split(maxsplit=1)
                    task_id = parts[1].strip() if len(parts) > 1 else None
                    task = state_store.get_task(task_id) if task_id else state_store.latest_task()
                    if not task:
                        print("\nNo matching task found to resume.")
                        continue
                    active_task = task
                    print("\nResumed task:")
                    print(render_status(active_task))
                    if active_task.awaiting_hitl:
                        _print_hitl_prompt()
                    continue

                if active_task and active_task.awaiting_hitl:
                    if lowered == "continue":
                        active_task.hitl_history.append("continue")
                        active_task.awaiting_hitl = False
                        state_store.save_task(active_task, note="hitl_continue")
                        quality = await _complete_remaining_steps(
                            runner, session.id, "researcher", active_task, state_store
                        )

                        active_task.status = "completed"
                        active_task.touch()
                        state_store.save_task(active_task, note="workflow_completed")
                        print("\n" + "=" * 60)
                        print(render_final_report(active_task, quality_report=quality))
                        print("=" * 60)
                        continue

                    if lowered.startswith("revise"):
                        revised_scope = user_input[6:].strip()
                        if not revised_scope:
                            print("\nUse: revise <new scope>")
                            continue
                        clarification_msg = await _build_clarification_request(
                            revised_scope,
                            clarifier_runner=clarifier_runner,
                            clarifier_session_id=clarifier_session.id,
                            user_id="researcher",
                        )
                        if clarification_msg:
                            pending_clarification_query = revised_scope
                            pending_clarification_prompt = clarification_msg
                            print("\n[Clarification Needed]")
                            print(clarification_msg)
                            print("Type your clarification, or `stop` to cancel.")
                            continue
                        revision_note = f"revise:{revised_scope}"
                        active_task = await _start_new_workflow_task(
                            runner,
                            session.id,
                            "researcher",
                            state_store,
                            revised_scope,
                        )
                        active_task.hitl_history.append(revision_note)
                        active_task.touch()
                        state_store.save_task(active_task, note=revision_note)
                        continue

                    if lowered == "stop":
                        active_task.hitl_history.append("stop")
                        active_task.status = "blocked"
                        active_task.awaiting_hitl = False
                        active_task.touch()
                        state_store.save_task(active_task, note="workflow_stopped")
                        print("\nWorkflow stopped and saved.")
                        continue

                    print("\nThis task is waiting at HITL checkpoint.")
                    _print_hitl_prompt()
                    continue

                clarification_msg = await _build_clarification_request(
                    user_input,
                    clarifier_runner=clarifier_runner,
                    clarifier_session_id=clarifier_session.id,
                    user_id="researcher",
                )
                if clarification_msg:
                    pending_clarification_query = user_input
                    pending_clarification_prompt = clarification_msg
                    print("\n[Clarification Needed]")
                    print(clarification_msg)
                    print("Type your clarification, or `stop` to cancel.")
                    continue

                # New request -> create a planned workflow and run step 1.
                active_task = await _start_new_workflow_task(
                    runner,
                    session.id,
                    "researcher",
                    state_store,
                    user_input,
                )
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()
                print("Please try again.")
    finally:
        # Cleanup MCP connection
        await mcp_tools.close()


def run_interactive():
    """Run the agent in interactive mode."""
    asyncio.run(run_interactive_async())


async def run_single_query_async(query: str, *, state_store_path: Path | None = None):
    """Run a single query (async version)."""
    from google.adk.sessions import InMemorySessionService

    session_service = InMemorySessionService()
    clarifier_runner = Runner(
        agent=create_clarifier_agent(),
        app_name="co_scientist_clarifier",
        session_service=session_service,
    )
    clarifier_session = await session_service.create_session(
        app_name="co_scientist_clarifier",
        user_id="researcher",
    )
    clarification_msg = await _build_clarification_request(
        query,
        clarifier_runner=clarifier_runner,
        clarifier_session_id=clarifier_session.id,
        user_id="researcher",
    )
    if clarification_msg:
        return "\n".join(
            [
                "## Clarification Needed",
                clarification_msg,
                "",
                "Reply with your clarification and I will continue.",
            ]
        )

    agent, mcp_tools = create_agent()

    runner = Runner(
        agent=agent,
        app_name="co_scientist",
        session_service=session_service,
    )
    
    # Create session
    session = await session_service.create_session(
        app_name="co_scientist",
        user_id="researcher",
    )
    default_state_path = Path(__file__).parent / "state" / "workflow_tasks.json"
    state_store = TaskStateStore(state_store_path or default_state_path)
    task = create_task(query)
    state_store.save_task(task, note="task_created_single_query")

    for idx in range(len(task.steps)):
        await _execute_step(runner, session.id, "researcher", task, idx)
        state_store.save_task(task, note=f"step_{idx + 1}_completed_single_query")

    quality = _evaluate_quality_gates(task)
    if not quality["passed"]:
        recovery, recovery_trace = await _run_fallback_recovery(runner, session.id, "researcher", task)
        if recovery_trace:
            task.steps[-1].tool_trace.extend(recovery_trace)
        if recovery:
            task.steps[-1].output = f"{task.steps[-1].output}\n\nFallback recovery notes:\n{recovery}"
            task.steps[-1].evidence_refs = extract_evidence_refs(task.steps[-1].output)
        quality = _evaluate_quality_gates(task)

    task.status = "completed"
    task.touch()
    state_store.save_task(task, note="workflow_completed_single_query")
    report = render_final_report(task, quality_report=quality)

    await mcp_tools.close()
    return report


def run_single_query(query: str, *, state_store_path: Path | None = None):
    """Run a single query (useful for testing)."""
    return asyncio.run(run_single_query_async(query, state_store_path=state_store_path))


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print(__doc__)
        print("\nUsage: python agent.py")
    else:
        run_interactive()
