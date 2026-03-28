Use this reference when the active step requires variant interpretation.

- Use `search_variants_by_gene` when you start with a gene symbol and need candidate rsIDs or HGVS forms.
- Use `get_variant_annotations` for multi-source annotation context and `annotate_variants_vep` for consequence and prediction scores.
- Use `search_civic_variants` or `search_civic_genes` when the interpretation question is oncology-specific.
- Use `get_clingen_gene_curation` when the evidence question is really about gene-disease validity rather than a single variant.
- For reference transcript or browser locus context (RefSeq accessions, UCSC coordinates), use `search_refseq_sequences`, `get_refseq_record`, or `search_ucsc_genome` alongside variant tools; they do not replace `get_variant_annotations` or VEP for pathogenicity calls.
- Be explicit when evidence is conflicting, low-confidence, or gene-level rather than variant-level.
