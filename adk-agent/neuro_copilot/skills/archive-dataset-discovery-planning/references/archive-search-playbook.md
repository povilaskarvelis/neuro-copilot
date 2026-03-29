# Archive Search Playbook

- Use `search_openneuro_datasets` for public BIDS neuroimaging discovery with modality filters and dataset IDs.
- Use `search_nemar_datasets` for EEG, MEG, iEEG, and electrophysiology-oriented BIDS archive discovery.
- Use `search_dandi_datasets` for NWB/BIDS neurophysiology and assay-driven discovery.
- Use `search_braincode_datasets` or `search_conp_datasets` when the request targets Brain-CODE or broader CONP mirrors.
- Use `search_encode_metadata` when the goal is ENCODE functional-genomics experiments or files (assays, targets, biosamples); for MPRA or related functional-characterization requests, search with focused assay and biosample terms and be ready to inspect Functional Characterization Experiment records rather than assuming only standard Experiment hits. Follow with `get_encode_record` for a known ENCSR/ENCFF/ENCBS accession or API path.
- Use `search_zenodo_records` for general open research deposits (datasets, code, supplemental packages) identified by topic keywords, a Zenodo community, or fielded query syntax; follow with `get_zenodo_record` when you have a numeric id, `10.5281/zenodo.*` DOI, or `/records/` URL.
- Keep archive searches simple. Prefer one clean keyword, modality, task, or study name per step rather than compound boolean strings.
- Follow search steps with an inspection step when the first-stage search returns promising dataset IDs or repository names.
