---
name: entity-resolution-planning
description: Use when planning work that depends on resolving genes, variants, diseases, phenotypes, drugs, ontology terms, or aliases before evidence collection can be reliable.
---

Use this skill when the objective contains ambiguous names, aliases, shorthand, ontology terms, or entity types that could be confused during retrieval.

Instructions:
1. Add an early normalization step when reliable downstream retrieval depends on resolving the exact gene, disease, phenotype, variant, or drug entity.
2. Distinguish entity-resolution work from evidence-gathering work. Use normalization to unblock later steps rather than treating it as the final answer.
3. Prefer the narrowest resolution step needed for the question instead of broad ontology exploration.
4. If multiple plausible entities may match the user’s term, plan to surface that ambiguity explicitly.
5. Load `references/entity-resolution-playbook.md` before finalizing plans that depend on aliases or ontology mapping.
