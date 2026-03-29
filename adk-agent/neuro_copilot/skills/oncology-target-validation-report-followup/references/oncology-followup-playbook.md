Use this reference for oncology target-validation follow-up.

- Use `get_depmap_gene_dependency` when the question is about vulnerability or selectivity.
- Use `get_biogrid_orcs_gene_summary` when published screen context may clarify the report.
- Use `get_gdsc_drug_sensitivity`, `get_prism_repurposing_response`, or `get_pharmacodb_compound_response` when the follow-up is about tractability or compound-response context and a candidate compound is already named.
- Do not use those compound-response tools for model-first drug discovery; they require a named drug/compound query.
- Keep biomarker context, disease lineage, and pan-essentiality explicit in the answer.
