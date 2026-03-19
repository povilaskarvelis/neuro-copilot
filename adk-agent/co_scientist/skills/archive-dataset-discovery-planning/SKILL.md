---
name: archive-dataset-discovery-planning
description: Use when planning archive or neuroscience dataset discovery where public metadata is sparse and the search strategy must be driven by archive type, modality, task, or study naming patterns.
---

Use this skill when the objective is about finding public datasets or archives, especially neuroscience datasets where repository metadata is sparse or inconsistent.

Instructions:
1. Choose one archive family per step whenever possible instead of combining multiple archives into one search step.
2. Prefer modality, task, assay, or study-name driven discovery when disease labels are likely sparse.
3. Do not treat a zero-hit disease query as enough evidence that no relevant public dataset exists.
4. Add a fallback browse or alternate-keyword step before concluding that an archive has no relevant data.
5. Load `references/archive-search-playbook.md` for archive-specific planning heuristics.
6. Load `references/modality-and-fallbacks.md` when the request is driven more by modality, assay, or archive coverage than by diagnosis labels.
7. After using the relevant references, return the final plan as JSON only.
