# Fallback Patterns

- If diagnosis metadata is sparse, retry with modality, assay, task, anatomy, cohort, or study-title terms.
- Keep archive search terms simple and singular. Do not use boolean strings unless the archive explicitly supports them.
- If an archive returns no matches, move to the next archive family only after a reasonable fallback inside the current archive.
- In the result summary, distinguish “no match found in this archive” from “no public dataset exists.”
