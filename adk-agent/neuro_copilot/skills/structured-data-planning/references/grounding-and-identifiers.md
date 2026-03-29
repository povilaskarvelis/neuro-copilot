# Grounding And Identifiers

- If the user starts with a gene symbol, alias, phenotype term, or disease name, add an early identifier-normalization step when downstream evidence depends on stable IDs.
- Use ontology mapping before structured lookups when the likely source uses MONDO, EFO, DOID, MeSH, OMIM, or related crosswalked identifiers.
- Variant-level tools typically require HGVS or rsID. If only a gene is known, plan a discovery step before annotation.
- Aggregate findings should be corroborated with sources that return PMIDs, DOIs, PMCID, OpenAlex IDs, or NCT numbers.
- Dataset or software deposits cited as `10.5281/zenodo.*` can be grounded with `get_zenodo_record` or discovered with `search_zenodo_records` when the question is about that object’s metadata or files.
- A plan is incomplete if it ends on aggregate database scores without at least one citable grounding step.
