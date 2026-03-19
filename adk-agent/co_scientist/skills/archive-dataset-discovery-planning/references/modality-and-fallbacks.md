# Modality And Fallbacks

- If diagnosis metadata is likely sparse, plan around modality, assay, task, anatomy, study title, or cohort name instead of only diagnosis text.
- A missing disease label is weak negative evidence in public archives. Add a fallback browse or alternate-keyword step before marking the archive absent.
- Prefer one archive per step so the executor can use archive-specific query patterns and follow-ups cleanly.
- When archive discovery succeeds, plan a second step that inspects the most promising dataset or repository details before concluding suitability.
