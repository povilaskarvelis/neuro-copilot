Use this reference when the plan depends on normalization or alias resolution.

- Use `resolve_gene_identifiers` for gene symbols, aliases, Entrez IDs, and Ensembl IDs.
- Use `get_ensembl_canonical_transcript` when the task depends on a canonical transcript choice, canonical TSS, or a promoter-window sequence around the canonical TSS.
- Use `get_refseq_record` when the user already names a RefSeq accession (NM/NR/NP/…) and you need curated sequence metadata; use `search_refseq_sequences` to list candidate RefSeq accessions by gene or keyword after symbols are reasonably stable.
- Use `map_ontology_terms_oxo` for disease or ontology cross-mapping across prefixes.
- Use `search_hpo_terms` when phenotype-term normalization matters.
- Use `get_orphanet_disease_profile` or `query_monarch_associations` when disease identity and disease-gene context are tightly coupled.
- If you plan a Monarch step after normalization, prefer carrying the resolved CURIE into the step rather than leaving the executor to reuse a free-text alias.
- For direct gene-disease questions, plan around Monarch's supported disease-to-gene and gene-to-phenotype modes instead of assuming a generic gene-to-disease mode exists.
- Keep the normalization step small and only as broad as needed for the downstream evidence plan.
