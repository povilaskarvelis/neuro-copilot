Use this reference when executing entity-resolution work.

- Use `resolve_gene_identifiers` for gene aliases and canonical gene IDs.
- When calling `query_monarch_associations` after normalization, prefer an explicit canonical `entityId`/CURIE instead of a free-text alias whenever possible.
- `query_monarch_associations` only supports `disease_to_phenotype`, `phenotype_to_gene`, `disease_to_gene_causal`, `disease_to_gene_correlated`, and `gene_to_phenotype`.
- For direct gene-disease questions, do not invent unsupported gene-to-disease Monarch modes. Normalize the disease and use a disease-to-gene mode, or use a more direct structured association source when the task is really target-disease scoring.
- Use `map_ontology_terms_oxo` when disease or ontology CURIE mapping is needed.
- Use `search_hpo_terms` when phenotype normalization matters.
- Use disease profile or association tools only when they help distinguish plausible disease mappings.
- Carry the resolved canonical identifier into the step summary for downstream reuse.
