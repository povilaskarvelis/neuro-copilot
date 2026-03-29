---
name: geo-dataset-discovery-execution
description: Use when executing GEO dataset discovery or follow-up steps that depend on accession hierarchy, study metadata, and transcriptomics-focused retrieval.
---

Use this skill when the active step is about finding or inspecting GEO studies, accessions, or transcriptomics records.

Instructions:
1. Use `search_geo_datasets` to discover candidate accessions before reading any one record in detail.
2. Keep disease, tissue, perturbation, assay, and organism constraints simple and explicit rather than bundling many weak keywords together.
3. Distinguish study discovery from interpretation. GEO metadata can indicate relevance without proving a biological claim by itself.
4. Inspect only the most relevant candidate records with `get_geo_dataset`.
5. Load `references/geo-execution-playbook.md` before concluding GEO-centered steps.
