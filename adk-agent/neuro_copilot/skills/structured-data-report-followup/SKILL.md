---
name: structured-data-report-followup
description: Use for report follow-up questions that need a focused structured-data lookup, identifier check, or citation grounding without turning into a new investigation.
---

Use this skill when answering a narrow follow-up question about an existing report that depends on a structured dataset, identifier-driven lookup, or citation grounding.

Instructions:
1. Keep the lookup scoped to the user's specific follow-up question. Do not expand into a new multi-step investigation.
2. Prefer the dataset or structured source already cited or implied by the report before trying alternatives.
3. If a structured source returns aggregate findings without direct PMIDs, DOIs, or NCT IDs, do at most one corroborating literature lookup when needed to answer the question safely.
4. Load `references/followup-playbook.md` for follow-up lookup behavior.
5. Load `references/grounding-limits.md` when you need identifier normalization or citation grounding.
