---
name: archive-dataset-discovery-execution
description: Use when executing or following up on archive and public dataset discovery steps where metadata is sparse and search strategy must be adapted by archive, modality, or study naming.
---

Use this skill when the active step or follow-up lookup is about public archive discovery, especially neuroscience datasets with sparse diagnosis metadata.

Instructions:
1. Use simple archive-appropriate keywords, modality labels, task labels, or study names rather than compound boolean strings.
2. Treat zero-hit disease-label searches as weak negative evidence. Try a modality, task, anatomy, or study-name fallback before concluding the archive has nothing relevant.
3. Prefer one archive family at a time and inspect returned dataset IDs or repository names before broadening to another archive.
4. Load `references/archive-query-playbook.md` for archive-specific query and follow-up behavior.
5. Load `references/fallback-patterns.md` when diagnosis metadata is sparse or the first search path fails.
