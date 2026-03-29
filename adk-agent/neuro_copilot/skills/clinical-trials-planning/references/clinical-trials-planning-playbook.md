Use this reference when the plan is driven by clinical-trial evidence.

- Prefer `search_clinical_trials` to discover the relevant NCT set, then `get_clinical_trial` for the 1-3 most decision-relevant records.
- If the question is about overall activity in a disease or target area, `summarize_clinical_trials_landscape` can come first, followed by direct trial inspection.
- When safety, boxed warnings, or real-world risk are part of the question, add `get_dailymed_drug_label` or `search_fda_adverse_events` after the core trial steps.
- Keep arms, status, phase, sponsor, and primary outcome interpretation separate; do not treat one field as a proxy for the rest.
