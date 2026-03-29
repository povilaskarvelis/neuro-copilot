---
name: evidence-weighting-execution
description: Use when executing steps that require judging source strength, handling conflicting evidence, or assigning confidence without treating all evidence equally.
---

Use this skill when the active step contains mixed evidence, multiple source types, or claims that need an explicit confidence judgment.

Instructions:
1. Distinguish evidence type before summarizing: curated clinical or regulatory evidence, direct literature evidence, structured aggregate evidence, preclinical screens, and metadata-only discovery are not interchangeable.
2. Do not let source count override source quality. A few strong direct sources can outweigh many weak indirect ones.
3. When evidence conflicts, state the conflict explicitly and explain which side currently carries more weight and why.
4. Confidence labels should reflect both source quality and agreement: high, moderate, low, or mixed.
5. Keep identifiers attached to the exact claim they support rather than pooling citations at the end.
6. Load `references/evidence-weighting-playbook.md` before concluding steps with mixed or uneven evidence.
