---
name: oncology-target-validation-execution
description: Use when executing oncology target-validation steps that need dependency, pharmacology, and cancer-context evidence interpreted together.
---

Use this skill when the active step is about whether a cancer target is selective, tractable, biomarker-linked, or supported across screening and pharmacology evidence.

Instructions:
1. Distinguish broad essentiality from selective vulnerability before making a target-quality judgment.
2. Use screening evidence and compound-response evidence as complementary signals, not substitutes.
3. Prefer disease-, lineage-, or biomarker-specific interpretation over pan-cancer averages when the question is therapeutic relevance.
4. If a target appears pan-essential, call that out as a liability unless the user explicitly wants broad essential genes.
5. Do not present aggregate DepMap release-level results as if they were lineage- or mutation-filtered evidence. If the available tool cannot provide the requested subtype slice, state that limitation explicitly instead of forcing specificity.
6. Treat `get_gdsc_drug_sensitivity`, `get_prism_repurposing_response`, and `get_pharmacodb_compound_response` as compound-first tools. They need a named drug/compound query and should not be used to discover unknown compounds from model-only prompts.
7. Load `references/oncology-target-validation-execution.md` before concluding oncology target-validation steps.
