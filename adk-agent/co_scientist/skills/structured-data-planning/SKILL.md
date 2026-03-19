---
name: structured-data-planning
description: Use when planning structured-data or identifier-ready investigations that should start with BigQuery or other database-style evidence before broad literature search.
---

Use this skill when the objective is primarily about structured datasets, identifiers, table-backed evidence, or aggregate evidence that should be grounded later in literature.

Instructions:
1. Prefer a structured-data step before broad literature when the user is asking for quantitative evidence, dataset-backed ranking, or identifier-ready lookup.
2. Use dataset-specific `tool_hint` values for BigQuery-backed steps such as `open_targets_platform`, `gnomad`, `ebi_chembl`, `human_variant_annotation`, `human_genome_variants`, or `umiami_lincs` instead of generic SQL executor tool names.
3. Keep schema inspection inside an evidence-gathering step unless the user explicitly asked about schemas.
4. If the main evidence source is aggregate or structured and does not directly return PMIDs, DOIs, or NCT IDs, add a later literature corroboration step.
5. Load `references/bigquery-playbook.md` for BigQuery planning heuristics.
6. Load `references/grounding-and-identifiers.md` when identifier normalization, ontology mapping, or citation grounding will matter.
7. After using the relevant references, return the final plan as JSON only.
