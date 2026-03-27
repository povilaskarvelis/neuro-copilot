Use this reference when the objective is oncology target validation.

- Prefer a sequence such as: gene/target normalization -> dependency evidence -> drug-response or pharmacology context -> literature or trial corroboration.
- Use dependency tools to ask whether the target is selective, not just whether it ever scores negative.
- Pair `get_depmap_gene_dependency` with `get_biogrid_orcs_gene_summary` when you need release-level metrics plus published screen context.
- Treat `get_depmap_gene_dependency` as a release-level named-gene summary tool. Do not use it alone for lineage-, mutation-, or cell-line-subset co-dependency discovery.
- Pair screening evidence with `get_gdsc_drug_sensitivity`, `get_prism_repurposing_response`, or `get_pharmacodb_compound_response` when the question shifts from vulnerability to tractability.
