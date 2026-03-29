# Grounding And Fallbacks

- When the current step starts from a symbol, alias, phenotype, or disease label, normalize identifiers before querying downstream tools that depend on stable IDs.
- Variant-level tools require HGVS or rsID; if only a gene is known, discover variants first.
- If the first structured source is insufficient, fall back to the closest overlapping source that matches the requested evidence type rather than broadening randomly.
- Structured claims should carry real identifiers whenever available and cite literature support when the primary source is aggregate.
