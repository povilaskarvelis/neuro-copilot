# BigQuery Planning Playbook

- Start with BigQuery only when the task is dataset-like, identifier-ready, or clearly about structured evidence rather than exploratory literature search.
- Use the dataset name as the plan step `tool_hint` so the plan shows which structured source is being queried.
- Prefer focused evidence steps over broad schema-discovery steps. If schema inspection is needed, make it part of the evidence step's completion path.
- For Open Targets style questions, resolve identifiers early and plan around evidence, tractability, safety, pathway, or drug tables rather than generic “inspect tables” work.
- For LINCS, keep plans on metadata-sized tables unless the user explicitly wants a costly raw readout scan.
- When the structured source will likely return aggregate values rather than citable paper identifiers, plan a later corroboration step with PubMed, OpenAlex, or ClinicalTrials.gov.
