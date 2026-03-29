---
name: geo-dataset-discovery-report-followup
description: Use for report follow-up questions about GEO studies, accessions, transcriptomics datasets, or metadata details.
---

Use this skill when a follow-up question asks for a GEO accession, study summary, organism, assay, perturbation context, or confirmation that a GEO dataset is relevant.

Instructions:
1. Keep the lookup scoped to the specific dataset question rather than re-running open-ended discovery.
2. Prefer `get_geo_dataset` when an accession is already known or strongly implied.
3. Use `search_geo_datasets` only to recover likely accessions or candidate studies.
4. Treat GEO metadata as study-context evidence, not full biological proof by itself.
5. Load `references/geo-followup-playbook.md` before answering GEO-centered follow-ups.
