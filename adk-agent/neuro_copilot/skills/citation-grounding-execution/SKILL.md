---
name: citation-grounding-execution
description: Use when executing steps that need PMIDs, DOIs, NCT IDs, OpenAlex IDs, or citation normalization to ground claims and avoid unsupported summaries.
---

Use this skill when the current step depends on harvesting, validating, or normalizing citable identifiers for specific claims.

Instructions:
1. Prefer the source most likely to yield the exact identifier you need before broadening to adjacent literature tools.
2. Harvest concrete identifiers first, then summarize. Do not treat uncited aggregate findings as fully grounded.
3. When a source returns incomplete citation metadata, make one focused follow-up call to fill the gap rather than starting a broad new search.
4. Keep grounding scoped to the active step's claims; do not build a full bibliography unless the step explicitly requires it.
5. In step summaries, keep identifiers attached to the specific claim, dataset, or trial they support instead of appending one pooled citation list at the end.
6. Load `references/citation-grounding-playbook.md` before concluding steps that depend on citation quality.
