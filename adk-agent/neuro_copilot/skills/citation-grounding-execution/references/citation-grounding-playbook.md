Use this reference when a step needs reliable identifiers and citation grounding.

- Prefer `search_pubmed` or `search_pubmed_advanced` when PMIDs matter.
- Use `search_openalex_works` when DOI discovery or broader citation graph context is needed.
- Use `get_pubmed_abstract` after you already have a PMID and need abstract-level confirmation.
- Use `get_paper_fulltext` only after you already have a PMID, DOI, or PMCID and the step truly needs paper-body detail.
- Normalize identifiers in the summary using the canonical forms already expected by the workflow.
- When multiple claims or named datasets appear in one summary, attach each identifier to the exact claim it supports rather than emitting a pooled citation block.
