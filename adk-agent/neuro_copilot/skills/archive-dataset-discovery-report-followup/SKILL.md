---
name: archive-dataset-discovery-report-followup
description: Use for report follow-up questions that need a focused archive or public dataset lookup where metadata may be sparse, without expanding into broad discovery.
---

Use this skill when the user asks a narrow follow-up question about archive-backed datasets mentioned in or adjacent to an existing report.

Instructions:
1. Keep searches narrow and tied to the specific cohort, modality, task, anatomy label, or dataset family relevant to the follow-up question.
2. Treat a zero-hit disease-label search as weak evidence, but use only one practical fallback before concluding the archive lookup is inconclusive.
3. Prefer checking the archive family already implied by the report before broadening elsewhere.
4. Load `references/followup-query-playbook.md` for archive-specific follow-up behavior.
5. Load `references/fallback-limits.md` when diagnosis metadata is sparse or the first query fails.
