# BigQuery Execution Playbook

- Use `list_bigquery_tables` to inspect dataset or table schema when exact column names are uncertain.
- Use `run_bigquery_select_query` only for read-only SQL.
- Keep queries narrow and tied to the current step objective; avoid broad exploratory scans when the step is identifier-ready.
- For BigQuery-backed evidence, preserve dataset names and returned identifiers in the summary.
- If a structured result is useful but not directly citable, follow it with PubMed, OpenAlex, or ClinicalTrials.gov corroboration before concluding.
