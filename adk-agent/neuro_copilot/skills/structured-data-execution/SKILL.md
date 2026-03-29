---
name: structured-data-execution
description: Use when executing or following up on structured-data, dedicated-source, BigQuery-backed, or identifier-driven steps that need disciplined query selection and grounding.
---

Use this skill when the active step or follow-up lookup is primarily about structured datasets, identifier-driven lookups, or aggregate evidence that may need citation grounding.

Instructions:
1. Prefer the dataset or structured source already implied by the step before broadening to nearby alternatives.
2. When the source family is already clear, prefer the dedicated tool for that source before falling back to raw BigQuery or generic archive search.
3. Keep schema inspection lightweight and inline. Use it only to unblock the current evidence step.
4. When a structured source returns aggregate findings without direct PMIDs, DOIs, or NCT IDs, add a corroborating literature lookup before concluding.
5. Load `references/bigquery-execution-playbook.md` for structured-query behavior, including when direct source tools should replace BigQuery.
6. Load `references/grounding-and-fallbacks.md` when identifier normalization or literature grounding is needed.
