# Archive Search Playbook

- Use `search_openneuro_datasets` for public BIDS neuroimaging discovery with modality filters and dataset IDs.
- Use `search_nemar_datasets` for EEG, MEG, iEEG, and electrophysiology-oriented BIDS archive discovery.
- Use `search_dandi_datasets` for NWB/BIDS neurophysiology and assay-driven discovery.
- Use `search_braincode_datasets` or `search_conp_datasets` when the request targets Brain-CODE or broader CONP mirrors.
- Keep archive searches simple. Prefer one clean keyword, modality, task, or study name per step rather than compound boolean strings.
- Follow search steps with an inspection step when the first-stage search returns promising dataset IDs or repository names.
