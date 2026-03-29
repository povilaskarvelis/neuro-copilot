---
name: entity-resolution-report-followup
description: Use for report follow-up questions that need clarification of aliases, canonical identifiers, ontology mappings, or ambiguous entities in the report.
---

Use this skill when the follow-up question is mainly about what entity a term refers to, whether two names map to the same thing, or which canonical identifier the report means.

Instructions:
1. Keep the lookup anchored to the ambiguous entity in the report.
2. Prefer direct normalization over broad literature rediscovery.
3. If there are multiple plausible mappings, explain the ambiguity and state which mapping the report appears to rely on.
4. Return the canonical identifier or normalized name when possible.
5. Load `references/entity-resolution-followup.md` before answering normalization-focused follow-ups.
