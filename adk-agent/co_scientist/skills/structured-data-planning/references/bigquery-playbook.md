# BigQuery Planning Playbook

- Start with BigQuery only when the task is dataset-like, identifier-ready, or clearly about structured evidence rather than exploratory literature search, and no dedicated source tool already covers the question more directly.
- Use the dataset name as the plan step `tool_hint` so the plan shows which structured source is being queried.
- Prefer focused evidence steps over broad schema-discovery steps. If schema inspection is needed, make it part of the evidence step's completion path.
- For Open Targets questions, prefer release-aware association or L2G tools before planning raw table inspection. Only fall back to BigQuery when the request truly needs broader Open Targets tables not exposed by a dedicated tool.
- For GWAS, JASPAR, TCGA availability, CELLxGENE marker-gene, Human Protein Atlas single-cell, Ensembl canonical transcript or TSS, RefSeq accession, ENCODE metadata, or UniProt protein-profile questions, plan around the dedicated source tool rather than generic BigQuery exploration.
- For LINCS, keep plans on metadata-sized tables unless the user explicitly wants a costly raw readout scan.
- When the structured source will likely return aggregate values rather than citable paper identifiers, plan a later corroboration step with PubMed, OpenAlex, or ClinicalTrials.gov.
