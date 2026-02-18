# Co-Scientist Question Taxonomy and Workflow Design

## Goal
Design workflows around capabilities, not niche request templates. The planner should compose stages from request needs across genetics, clinical, chemistry, literature, safety, pathways, and prioritization.

## Capability Buckets (Tool-Informed)
- Disease/entity context: ontology expansion, disease/target normalization, local datasets
- Literature/research ecosystem: PubMed/OpenAlex publication and researcher signals
- Human genetics/variants: GWAS, ClinVar, direction-of-effect
- Biological mechanism/context: pathways, interactions, expression
- Clinical/safety: trial outcomes, liabilities, risk signals
- Chemistry/translation: compounds, druggability, existing drugs, tractability
- Competitive landscape: density, crowding, white-space signals
- Comparison/synthesis: weighted trade-offs and decision integration

## Broad Question Set (Increasing Complexity)
### Level 1: Single-axis retrieval
1. What are the top recent papers linking LRRK2 to Parkinson disease?
2. What known compounds are reported for TYK2?
3. What clinical trials exist for EGFR inhibitors in NSCLC?
4. What GWAS associations are reported for GBA1 in Parkinson disease?
5. Which pathways include TREM2?
6. What tissue-expression context is reported for IL23R?
7. What known safety liabilities are reported for JAK1 modulation?
8. What drugs are currently linked to BACE1?

### Level 2: Context + evidence synthesis
9. Summarize genetics and pathway evidence for ATP13A2 in Parkinson disease.
10. Summarize the clinical landscape and major termination reasons for BACE1 programs.
11. Assess whether TYK2 has enough chemistry evidence for tractable small-molecule development.
12. Map disease ontology terms and related IDs for ulcerative colitis and Crohn disease.
13. Summarize literature consensus and contradictions for LRRK2 inhibition.
14. Identify key safety risks for IL23R-targeting approaches and where evidence is sparse.
15. Summarize competitive landscape for EGFR in NSCLC by phase and mechanism.
16. Build a concise evidence map for GBA1 (genetics, clinical, literature).

### Level 3: Multi-axis validation / contradiction handling
17. Validate whether BACE1 is still plausible by weighing genetics, clinical failures, and safety.
18. Determine if ATP13A2 activator strategy is supported or contradicted by current evidence.
19. Stress-test TYK2 as a target under a safety-first assumption.
20. Check whether clinical signals agree with genetics direction for LRRK2.
21. Identify conflicting evidence for EGFR resistance mechanisms and propose resolution paths.
22. Test if IL23R prioritization holds when competitive crowding is weighted higher.
23. Evaluate if GBA1 evidence remains robust when pathway evidence is down-weighted.
24. Validate if a proposed target has enough direct human evidence vs model-only evidence.

### Level 4: Comparative prioritization
25. Compare LRRK2 vs GBA1 for Parkinson disease with genetics-heavy weighting.
26. Rank TYK2, JAK1, and IL23R for ulcerative colitis under safety-first weighting.
27. Compare ATP13A2 vs SNCA for translational tractability and risk.
28. Prioritize EGFR, MET, and KRAS pathway nodes for NSCLC follow-up experiments.
29. Rank candidate targets for Alzheimer disease under clinical de-risking strategy.
30. Compare two targets where one has stronger genetics and one has stronger chemistry.
31. Build a top-3 shortlist for a disease using transparent criteria and confidence.
32. Re-rank candidates after excluding high-liability safety profiles.

### Level 5: End-to-end decision + execution planning
33. Recommend go/no-go on BACE1 and provide a 90-day validation plan.
34. Select a lead target for Parkinson disease and define milestone-based next experiments.
35. Produce a de-risked execution blueprint for TYK2 including fallback branches.
36. Generate a staged plan to resolve top evidence gaps before investment decision.
37. Recommend which target to advance now and what evidence must be collected next.
38. Define an action plan for resolving contradictory genetics vs clinical evidence.
39. Build a decision memo and ranked next actions for translational feasibility.
40. Produce an experiment and data-acquisition sequence to validate final recommendation.

### Level 6: Research ecosystem / collaboration strategy
41. Identify active researchers in a narrow mechanism area with citation-backed rationale.
42. Find likely collaborators for a target area and show supporting publication evidence.
43. Map institutions most active in a disease-target intersection over the last 5 years.
44. Shortlist experts for advisory input and state confidence/coverage limitations.
45. Compare two researcher groups by topic-specific activity and publication relevance.
46. Build a collaborator outreach shortlist tied to specific evidence gaps.

### Level 7: Mixed-source + internal-data questions
47. Integrate local biomarker CSV evidence with literature support for prioritization.
48. Re-rank targets after adding internal assay constraints from local datasets.
49. Validate whether local trend signals align with external clinical/genetics evidence.
50. Propose follow-up analyses combining internal data with public evidence sources.

## Workflow Archetypes (Composable)
- Scope and decomposition: clarify entities, constraints, success criteria
- Entity/context normalization: IDs, ontology context, local dataset framing
- Human genetics/variant evidence
- Biological mechanism/context evidence
- Clinical outcomes and safety evidence
- Chemistry/druggability/development evidence
- Literature/research-ecosystem evidence
- Cross-evidence comparison/prioritization
- Execution blueprint (for action planning requests)
- Decision report synthesis

## Planner Rule
Compose stages dynamically from detected capability needs, live plan state, and user feedback at checkpoints.
Keep only minimal deterministic guardrails: evidence traceability, explicit uncertainty handling, safety limits, and final-report delivery.
