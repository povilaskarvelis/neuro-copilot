Use this reference when executing oncology target-validation work.

- Use `get_depmap_gene_dependency` for release-level vulnerability metrics and selectivity patterns.
- Do not overstate `get_depmap_gene_dependency` as subtype-filtered evidence; it does not directly return lineage- or mutation-specific co-dependencies.
- Use `get_biogrid_orcs_gene_summary` for published CRISPR screen context.
- Use `get_gdsc_drug_sensitivity`, `get_prism_repurposing_response`, or `get_pharmacodb_compound_response` when the question needs tractability or compound-response context and a candidate compound is already named.
- Do not use those compound-response tools for model-first drug discovery; they require a named drug/compound query.
- Use pathway or interaction tools only when they materially sharpen the mechanistic interpretation.
- Keep lineage specificity, biomarker context, and pan-essentiality separate in the final summary.
