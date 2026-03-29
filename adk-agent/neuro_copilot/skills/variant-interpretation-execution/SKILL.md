---
name: variant-interpretation-execution
description: Use when executing variant or gene-variant evidence steps that need careful normalization, review-status awareness, and conflict handling.
---

Use this skill when the current step depends on identifying variants from a gene, interpreting specific variant evidence, or reconciling competing annotations.

Instructions:
1. If only the gene is known, discover candidate variants first. Do not jump straight to annotation tools that require rsID or HGVS.
2. Separate functional prediction, aggregate annotation, oncology interpretation, and gene-level curation; they are related but not interchangeable.
3. Treat conflicting interpretations explicitly. Do not collapse VUS, benign, pathogenic, and oncology-specific assertions into one label.
4. Prefer the smallest variant set needed to answer the step rather than broad variant enumeration.
5. Load `references/variant-interpretation-playbook.md` before concluding variant-heavy steps.
