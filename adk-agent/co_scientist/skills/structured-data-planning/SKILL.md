---
name: structured-data-planning
description: Use when planning structured-data or identifier-ready investigations that should start with dedicated source tools, BigQuery, or other database-style evidence before broad literature search.
---

Use this skill when the objective is primarily about structured datasets, identifiers, table-backed evidence, or aggregate evidence that should be grounded later in literature.

Instructions:
1. Prefer a structured-data step before broad literature when the user is asking for quantitative evidence, dataset-backed ranking, or identifier-ready lookup.
2. Prefer a dedicated source tool before raw BigQuery when the question is already aligned to a supported source family such as Open Targets associations or L2G, GWAS study-variant lookups, JASPAR motifs, TCGA availability, CELLxGENE marker genes, Human Protein Atlas single-cell values, Ensembl canonical transcript or TSS windows, RefSeq records, ENCODE metadata, or UniProt protein profiles.
3. Use dataset-specific `tool_hint` values for BigQuery-backed steps such as `open_targets_platform`, `gnomad`, `ebi_chembl`, `human_variant_annotation`, `human_genome_variants`, or `umiami_lincs` instead of generic SQL executor tool names, but only when a dedicated MCP lookup is not already the better fit.
4. Keep schema inspection inside an evidence-gathering step unless the user explicitly asked about schemas.
5. If the main evidence source is aggregate or structured and does not directly return PMIDs, DOIs, or NCT IDs, add a later literature corroboration step.
6. Load `references/bigquery-playbook.md` for structured-data planning heuristics, including when not to start with BigQuery.
7. Load `references/grounding-and-identifiers.md` when identifier normalization, ontology mapping, or citation grounding will matter.
8. After using the relevant references, return the final plan as JSON only.
