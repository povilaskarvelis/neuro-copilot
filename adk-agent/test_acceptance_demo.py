import re

from run_acceptance_demo import _score_report


def test_score_report_passes_expected_contract():
    report = """
## Decomposition
1. Query disease/topic context and timeframe constraints. (completed)
2. Identify topic-matched publications from OpenAlex/PubMed. (completed)
3. Find candidate authors and affiliation signals. (completed)
4. Assess author activity and produce ranked shortlist. (completed)

## Answer
Recommendation: proceed cautiously with clear safety gating.

## Methodology
### Step 1: Scope Request
- Status: completed
- Goal: clarify scope
- Why this order: lock retrieval objective
- Planned tools: search_pubmed_advanced, search_openalex_works
- Enforced step tools: search_diseases
- Information gained: focused query terms and disease scope
- Evidence IDs from step: PMID:12345678
- Executed tool trace:
  1. [main] search_pubmed_advanced(call_id=a1, args={"query":"ipf"}) -> success
- Pivot behavior: none detected.

### Step 2: Gather Evidence
- Status: completed
- Goal: gather literature and researcher activity
- Why this order: collect primary evidence before synthesis
- Planned tools: search_pubmed_advanced, search_openalex_authors
- Enforced step tools: search_pubmed_advanced, search_openalex_authors
- Information gained: author and institution coverage
- Evidence IDs from step: PMID:12345678, PMID:23456789
- Executed tool trace:
  1. [main] search_openalex_authors(call_id=a2, args={"query":"idiopathic pulmonary fibrosis"}) -> success
  2. [main] search_pubmed_advanced(call_id=a3, args={"query":"IPF treatment"}) -> success
- Pivot behavior: none detected.

### Step 3: Synthesize
- Status: completed
- Goal: give recommendation with risk and safety context
- Why this order: synthesis after evidence collection
- Planned tools: none
- Enforced step tools: none (reasoning-only step)
- Information gained: recommendation with explicit evidence and limitations
- Evidence IDs from step: PMID:12345678, PMID:23456789
- Executed tool trace: no tool calls recorded for this step.

## Evidence
- PMID:12345678
- PMID:23456789

## Diagnostics
- Request type: target_prioritization
- Intent tags: researcher_discovery
- Steps completed: 3/3
- Quality gate passed: yes
- Evidence refs detected: 2
- Tool calls captured: 3
- Unresolved gaps: none
""".strip()

    checks, metrics = _score_report(report, ["researcher", "affiliation", "pmid", "risk", "safety", "evidence"])

    assert all(checks.values())
    assert metrics["decomposition_count"] == 4
    assert metrics["tool_call_count"] == 3
    assert metrics["evidence_count"] == 2
    assert metrics["source_family_count"] >= 2
    assert re.search(r"search_openalex_authors", " ".join(metrics["tool_names"]))


def test_score_report_fails_on_missing_contract_sections():
    report = "## Answer\nNo structure here.\n"
    checks, metrics = _score_report(report, ["evidence"])

    assert checks["report_contract"] is False
    assert checks["tool_trace_present"] is False
    assert checks["evidence_refs_present"] is False
    assert checks["quality_gate_passed"] is False
    assert checks["multi_source_trace"] is False
    assert metrics["decomposition_count"] == 0
