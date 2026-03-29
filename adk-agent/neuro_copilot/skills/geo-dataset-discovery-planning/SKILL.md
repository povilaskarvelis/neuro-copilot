---
name: geo-dataset-discovery-planning
description: Use when planning GEO-centered dataset discovery or transcriptomics evidence gathering that depends on accession hierarchy, study metadata, and follow-up record inspection.
---

Use this skill when the objective is about GEO datasets, transcriptomics studies, perturbation datasets, accession discovery, or study-level genomic evidence.

Instructions:
1. Begin with a GEO search step when the user is asking for relevant datasets rather than direct gene-level mechanistic claims.
2. Distinguish study discovery from accession inspection. Use `search_geo_datasets` first, then `get_geo_dataset` for the most relevant records.
3. Prefer disease, tissue, perturbation, assay, organism, and accession signals over broad keyword sprawl.
4. If the task needs evidence interpretation rather than simple dataset discovery, add a later literature or structured-data corroboration step.
5. Load `references/geo-discovery-playbook.md` before finalizing plans that depend on GEO results.
