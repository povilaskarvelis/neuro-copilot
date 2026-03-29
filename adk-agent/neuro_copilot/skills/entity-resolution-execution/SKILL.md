---
name: entity-resolution-execution
description: Use when executing steps that depend on resolving ambiguous or aliased genes, variants, diseases, phenotypes, drugs, or ontology terms before retrieval.
---

Use this skill when the current step is blocked or weakened by ambiguity about the exact entity being queried.

Instructions:
1. Resolve the entity before broad retrieval if a gene alias, disease label, phenotype term, or drug synonym could change the search results materially.
2. Keep the resolution scoped to the active step. Do not turn normalization into a broad ontology survey.
3. When several plausible entities match, state the ambiguity and use the best-supported mapping for the current step.
4. Preserve the resolved canonical identifiers in the step summary so later steps can reuse them reliably.
5. Load `references/entity-resolution-execution.md` before concluding normalization-heavy steps.
