---
name: oncology-target-validation-planning
description: Use when planning oncology target-validation work that needs dependency, drug-response, pathway, and literature evidence combined into a tractability judgment.
---

Use this skill when the objective is about whether a cancer target is compelling, selective, tractable, biomarker-linked, or supported across dependency and pharmacology evidence.

Instructions:
1. Plan target-validation work as a sequence: target context, dependency/selectivity evidence, pharmacology or compound evidence, then literature or trial corroboration.
2. Distinguish pan-essentiality from lineage- or biomarker-selective vulnerability; they imply different target-quality conclusions.
3. Add a compound or pharmacology step when the user asks about tractability, existing programs, or druggability rather than dependency alone.
4. Add a later literature or trial step when the earlier evidence is mostly aggregate or screening-based.
5. Do not plan subtype-specific dependency or co-dependency steps unless the named tool can actually provide that slice. `get_depmap_gene_dependency` is release-level and gene-level; it does not directly provide lineage- or mutation-filtered co-dependency discovery.
6. Load `references/oncology-target-validation-playbook.md` before finalizing plans that depend on cross-tool oncology reasoning.
