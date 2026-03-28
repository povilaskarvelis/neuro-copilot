# Archive Query Playbook

- Use `search_openneuro_datasets` for public neuroimaging discovery and pagination-aware browsing.
- Use `search_nemar_datasets` for EEG, MEG, iEEG, and related electrophysiology archive discovery.
- Use `search_dandi_datasets` for DANDI neurophysiology and NWB/BIDS discovery.
- Use `search_braincode_datasets` or `search_conp_datasets` for Brain-CODE and broader CONP repository discovery.
- Use `search_encode_metadata` / `get_encode_record` for ENCODE Portal metadata when the step is about released experiments or files there. For MPRA or related functional-characterization steps, keep the query narrow and do not assume the result type will be limited to standard Experiment records.
- Use `search_zenodo_records` for Zenodo-wide discovery; use `get_zenodo_record` to inspect a specific deposit once you have its id or DOI.
- Use `query_neurobagel_cohorts` for structured cohort filtering rather than archive keyword search.
- After finding a promising dataset or repository, inspect it with the corresponding detail tool before concluding relevance.
