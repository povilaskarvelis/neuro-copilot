"""Central registry for tool descriptions and source-selection metadata.

This keeps tool/source knowledge out of the workflow prompt module so the
planner/executor can consume a structured registry instead of large inline
dicts scattered through ``workflow.py``.
"""

from __future__ import annotations

from typing import Any


TOOL_DOMAINS: dict[str, list[str]] = {
    "literature": [
        "search_pubmed", "search_pubmed_advanced", "get_pubmed_abstract", "get_paper_fulltext",
        "search_europe_pmc_literature",
        "search_openalex_works", "search_openalex_authors",
        "rank_researchers_by_activity", "get_researcher_contact_candidates",
    ],
    "clinical": [
        "search_clinical_trials", "get_clinical_trial",
        "summarize_clinical_trials_landscape", "search_fda_adverse_events",
        "get_dailymed_drug_label",
    ],
    "protein": [
        "get_human_protein_atlas_gene",
        "search_uniprot_proteins", "get_uniprot_protein_profile",
        "search_reactome_pathways", "get_string_interactions", "get_intact_interactions",
        "get_biogrid_interactions",
        "search_pathway_commons_top_pathways", "get_guidetopharmacology_target",
        "get_alphafold_structure", "search_protein_structures",
        "search_drug_gene_interactions",
    ],
    "genomics": [
        "resolve_gene_identifiers", "map_ontology_terms_oxo",
        "search_hpo_terms", "get_orphanet_disease_profile", "query_monarch_associations",
        "search_quickgo_terms", "get_quickgo_annotations",
        "annotate_variants_vep", "search_civic_variants", "search_civic_genes",
        "search_variants_by_gene", "get_variant_annotations", "search_gwas_associations",
        "get_gene_tissue_expression", "get_depmap_gene_dependency",
        "get_biogrid_orcs_gene_summary",
        "get_gdsc_drug_sensitivity",
        "get_prism_repurposing_response", "get_pharmacodb_compound_response",
        "search_cellxgene_datasets", "get_clingen_gene_curation", "get_alliance_genome_gene_profile",
        "get_cancer_mutation_profile",
        "search_geo_datasets", "get_geo_dataset",
    ],
    "chemistry": [
        "get_pubchem_compound", "get_chembl_bioactivities",
        "get_guidetopharmacology_target", "get_dailymed_drug_label",
        "get_gdsc_drug_sensitivity", "get_prism_repurposing_response",
        "get_pharmacodb_compound_response",
    ],
    "immunology": [
        "search_iedb_epitope_evidence",
    ],
    "neuroscience": [
        "search_aba_genes", "search_aba_structures",
        "get_aba_gene_expression", "search_aba_differential_expression",
        "search_ebrains_kg", "get_ebrains_kg_document",
        "search_conp_datasets", "get_conp_dataset_details",
        "query_neurobagel_cohorts",
        "search_openneuro_datasets", "get_openneuro_dataset",
        "search_dandi_datasets", "get_dandi_dataset",
        "search_nemar_datasets", "get_nemar_dataset_details",
        "search_braincode_datasets", "get_braincode_dataset_details",
        "search_enigma_datasets", "get_enigma_dataset_info",
    ],
    "data": [
        "list_bigquery_tables", "run_bigquery_select_query",
        "benchmark_dataset_overview", "check_gpqa_access",
    ],
}

ALWAYS_AVAILABLE_DOMAINS = {"data", "literature"}

ALL_DOMAIN_NAMES = sorted(TOOL_DOMAINS.keys())

TOOL_TO_DOMAINS: dict[str, list[str]] = {}
for _domain, _tools in TOOL_DOMAINS.items():
    for _tool in _tools:
        TOOL_TO_DOMAINS.setdefault(_tool, []).append(_domain)

TOOL_DESCRIPTIONS: dict[str, str] = {
    "list_bigquery_tables": "List tables in a BigQuery dataset or inspect a table schema",
    "run_bigquery_select_query": "Run read-only SQL against allowlisted BigQuery datasets",
    "search_clinical_trials": "Search ClinicalTrials.gov (returns NCT IDs)",
    "get_clinical_trial": "Get details of a specific clinical trial by NCT ID",
    "summarize_clinical_trials_landscape": "Aggregate trial landscape stats for a condition",
    "search_pubmed": "Search PubMed literature (returns PMIDs, titles, authors)",
    "search_pubmed_advanced": "Advanced PubMed search with field-specific queries (MeSH, author, journal)",
    "get_pubmed_abstract": "Fetch full abstract for a PMID",
    "get_paper_fulltext": "Fetch PMC full text when available using a PMID, PMCID, or DOI",
    "search_iedb_epitope_evidence": "Search IEDB for epitope, T-cell, and MHC ligand evidence using peptide, antigen, and allele filters",
    "search_geo_datasets": "Search GEO (Gene Expression Omnibus) for transcriptomics and functional genomics records by disease, gene, perturbation, tissue, or accession. Returns GSE/GSM/GPL/GDS accessions for follow-up",
    "get_geo_dataset": "Get detailed GEO metadata for a specific accession or GEO UID, including organism, study type, sample count, PubMed links, and summary text",
    "search_openalex_works": "Search OpenAlex for papers, preprints, and citations (returns DOIs)",
    "search_openalex_authors": "Find researchers and their publication profiles",
    "rank_researchers_by_activity": "Rank authors by recent publication activity",
    "get_researcher_contact_candidates": "Get contact/affiliation info for researchers",
    "search_europe_pmc_literature": "Search Europe PMC for papers, preprints, citation counts, and open-access metadata",
    "resolve_gene_identifiers": "Resolve gene symbols, aliases, Entrez IDs, and Ensembl IDs via MyGene.info for identifier normalization",
    "map_ontology_terms_oxo": "Map ontology CURIEs across MONDO/EFO/DOID/MeSH/OMIM/UMLS and other prefixes using EBI OxO cross-references",
    "search_hpo_terms": "Search Human Phenotype Ontology terms via OLS for phenotype-term normalization",
    "get_orphanet_disease_profile": "Retrieve Orphanet / ORDO rare-disease profiles with xrefs, phenotypes, and curated disease-gene links",
    "query_monarch_associations": "Query Monarch phenotype- and rare-disease-centric associations such as phenotype-to-gene, disease-to-gene, disease-to-phenotype, and gene-to-phenotype. Prefer an explicit CURIE in entityId once a gene, phenotype, or disease has been normalized; do not invent unsupported gene-to-disease modes.",
    "search_quickgo_terms": "Search Gene Ontology terms in QuickGO by text and aspect",
    "get_quickgo_annotations": "Get GO annotations for a gene product via QuickGO, resolving gene symbols to UniProtKB when needed",
    "search_uniprot_proteins": "Search UniProt for protein entries",
    "get_uniprot_protein_profile": "Detailed protein profile (isoforms, PTMs, function)",
    "search_reactome_pathways": "Search biological pathway hierarchies",
    "get_string_interactions": "Get protein-protein interaction networks from STRING",
    "get_intact_interactions": "Get curated experimental molecular interactions from IntAct with partners, interaction types, detection methods, and publication support",
    "get_biogrid_interactions": "Get broader BioGRID experimental interaction evidence with physical/genetic classes, throughput tags, partners, and PMIDs",
    "get_human_protein_atlas_gene": "Get Human Protein Atlas summaries for tissue specificity, single-cell specificity, protein class, and subcellular localization",
    "get_depmap_gene_dependency": "Summarize release-level DepMap dependency metrics for a named gene (CRISPR/RNAi dependency fractions, pan-dependency/selectivity, predictive features). Does not directly slice by lineage, mutation-defined subsets, or discover co-dependencies across other genes.",
    "get_biogrid_orcs_gene_summary": "Summarize BioGRID ORCS published CRISPR screen evidence for a gene, including hit status, phenotypes, cell lines, and representative screens",
    "get_gdsc_drug_sensitivity": "Summarize GDSC / CancerRxGene compound sensitivity profiles across cell lines and tissues using IC50/AUC pharmacogenomic screens",
    "get_prism_repurposing_response": "Summarize Broad PRISM repurposing primary-screen response using single-dose log2-fold-change viability across pooled cancer cell lines",
    "get_pharmacodb_compound_response": "Summarize PharmacoDB cross-dataset compound-response evidence across public pharmacogenomic screens such as GDSC, PRISM, and CTRPv2",
    "search_cellxgene_datasets": "Search public CELLxGENE Discover/Census-backed single-cell datasets by cell type, tissue, disease, assay, and organism",
    "search_pathway_commons_top_pathways": "Search integrated top pathways in Pathway Commons across multiple pathway providers",
    "get_guidetopharmacology_target": "Get curated target-ligand interactions and pharmacology summaries from Guide to Pharmacology",
    "get_dailymed_drug_label": "Summarize key DailyMed SPL label sections such as boxed warnings, indications, contraindications, and warnings",
    "search_variants_by_gene": "Search MyVariant.info for variants in a gene by symbol. Returns HGVS/rsIDs for downstream get_variant_annotations or annotate_variants_vep. Use when only gene is known.",
    "get_variant_annotations": "Get aggregated MyVariant.info annotations for a specific variant using rsID, genomic HGVS, gene+protein HGVS, or shorthand like KRAS G12C. Returns ClinVar, dbSNP, gnomAD, CADD, COSMIC, and related annotations.",
    "get_clingen_gene_curation": "Summarize ClinGen gene-disease validity and dosage sensitivity curations for a gene",
    "get_alliance_genome_gene_profile": "Summarize Alliance Genome Resources model-organism and translational evidence for a gene, including orthologs, disease/phenotype counts, and disease models",
    "get_chembl_bioactivities": "Get bioactivity data (IC50, Ki, Kd) for a drug from ChEMBL - selectivity profiling",
    "search_fda_adverse_events": "Search FDA FAERS for post-marketing adverse event reports by drug name",
    "search_aba_genes": "Search Allen Brain Atlas for genes by name or acronym (mouse, human, developing mouse)",
    "search_aba_structures": "Search Allen Brain Atlas structure ontology for brain regions",
    "get_aba_gene_expression": "Get quantified gene expression across brain structures from Allen Brain Atlas ISH data",
    "search_aba_differential_expression": "Find genes differentially expressed between two brain structures (Allen Mouse Brain Atlas)",
    "search_ebrains_kg": "Search EBRAINS Knowledge Graph for neuroscience datasets, models, software, and contributors",
    "get_ebrains_kg_document": "Get detailed metadata for a specific EBRAINS Knowledge Graph resource (dataset, model, etc.)",
    "search_conp_datasets": "Search CONP dataset repositories by repository metadata",
    "get_conp_dataset_details": "Get detailed metadata (README, license, topics) for a specific CONP dataset repository returned by search_conp_datasets",
    "query_neurobagel_cohorts": "Query Neurobagel public cohorts by cohort filters",
    "search_openneuro_datasets": "Search OpenNeuro public datasets by keyword and/or modality",
    "get_openneuro_dataset": "Get detailed OpenNeuro metadata by dataset ID (e.g. ds000224), including DOI, modalities, diagnosis/study fields when present, and approximate subject counts from the public summary",
    "search_dandi_datasets": "Search DANDI Archive datasets by keyword",
    "get_dandi_dataset": "Get detailed metadata (name, version, assets, size, embargo) for a DANDI dandiset by identifier (e.g. 000003)",
    "search_nemar_datasets": "Search NEMAR public datasets by keyword",
    "get_nemar_dataset_details": "Get detailed metadata for a NEMAR dataset by repo name (e.g. nm000104)",
    "search_braincode_datasets": "Search Brain-CODE public dataset mirrors",
    "get_braincode_dataset_details": "Get detailed metadata for a Brain-CODE dataset by repo name (e.g. braincode_Mouse_Image)",
    "search_enigma_datasets": "Search ENIGMA public summary-statistic datasets",
    "get_enigma_dataset_info": "List ENIGMA summary statistic files for a disorder (e.g. scz, mdd, adhd, 22q, bd, asd). Returns filenames and raw CSV URLs",
    "benchmark_dataset_overview": "Overview of available benchmark datasets",
    "check_gpqa_access": "Check access to GPQA benchmark",
}

TOOL_ROUTING_METADATA: dict[str, dict[str, Any]] = {
    "search_clinical_trials": {
        "overlap_group": "clinical_trials",
        "preferred_for": "named ClinicalTrials.gov study discovery, NCT harvesting, and intervention/status snapshots",
        "fallback_tools": ["summarize_clinical_trials_landscape", "get_clinical_trial"],
    },
    "get_clinical_trial": {
        "overlap_group": "clinical_trials",
        "preferred_for": "full details for a specific NCT study, including design, eligibility, outcomes, and posted results",
        "fallback_tools": ["search_clinical_trials", "summarize_clinical_trials_landscape"],
    },
    "summarize_clinical_trials_landscape": {
        "overlap_group": "clinical_trials",
        "preferred_for": "status, phase, intervention, and termination-pattern summaries across a ClinicalTrials.gov landscape",
        "fallback_tools": ["search_clinical_trials", "get_clinical_trial"],
    },
    "search_pubmed": {
        "overlap_group": "literature_search",
        "preferred_for": "default biomedical literature search, PMID harvesting, and MeSH-friendly follow-up",
        "fallback_tools": ["search_europe_pmc_literature", "search_openalex_works"],
    },
    "search_europe_pmc_literature": {
        "overlap_group": "literature_search",
        "preferred_for": "preprints, Europe PMC citation metadata, and open-access status",
        "fallback_tools": ["search_pubmed", "search_openalex_works"],
    },
    "search_openalex_works": {
        "overlap_group": "literature_search",
        "preferred_for": "broader citation graph context, institution/researcher discovery, and non-PubMed coverage",
        "fallback_tools": ["search_pubmed", "search_europe_pmc_literature"],
    },
    "search_iedb_epitope_evidence": {
        "preferred_for": "direct IEDB epitope, T-cell, and MHC ligand evidence when peptide sequence, antigen, or allele filters are available",
        "fallback_tools": ["get_paper_fulltext", "search_pubmed"],
    },
    "search_reactome_pathways": {
        "overlap_group": "pathway_context",
        "preferred_for": "specific curated pathway titles and canonical Reactome pathway hierarchy",
        "fallback_tools": ["search_pathway_commons_top_pathways"],
    },
    "search_pathway_commons_top_pathways": {
        "overlap_group": "pathway_context",
        "preferred_for": "integrated pathway context across multiple pathway providers",
        "fallback_tools": ["search_reactome_pathways", "get_string_interactions"],
    },
    "get_string_interactions": {
        "overlap_group": "molecular_interactions",
        "preferred_for": "broad protein-network neighborhoods and integrated interaction evidence",
        "fallback_tools": ["get_intact_interactions", "get_biogrid_interactions", "search_pathway_commons_top_pathways"],
    },
    "get_intact_interactions": {
        "overlap_group": "molecular_interactions",
        "preferred_for": "curated experimental interaction records with detection methods and PMIDs",
        "fallback_tools": ["get_biogrid_interactions", "get_string_interactions", "search_pathway_commons_top_pathways"],
    },
    "get_biogrid_interactions": {
        "overlap_group": "molecular_interactions",
        "preferred_for": "broader experimental physical/genetic interaction coverage, throughput tags, and BioGRID partner evidence",
        "fallback_tools": ["get_intact_interactions", "get_string_interactions", "search_pathway_commons_top_pathways"],
    },
    "get_guidetopharmacology_target": {
        "overlap_group": "compound_pharmacology",
        "preferred_for": "curated target-ligand summaries, mechanism/action-type evidence, and representative ligands",
        "fallback_tools": ["get_chembl_bioactivities", "search_drug_gene_interactions", "get_pubchem_compound"],
    },
    "get_chembl_bioactivities": {
        "overlap_group": "compound_pharmacology",
        "preferred_for": "quantitative potency, selectivity, and assay-level bioactivity data",
        "fallback_tools": ["get_guidetopharmacology_target", "get_pubchem_compound", "search_drug_gene_interactions"],
    },
    "search_drug_gene_interactions": {
        "overlap_group": "compound_pharmacology",
        "preferred_for": "broad druggability categories and known drug-gene interaction coverage",
        "fallback_tools": ["get_guidetopharmacology_target", "get_chembl_bioactivities"],
    },
    "get_pubchem_compound": {
        "overlap_group": "compound_pharmacology",
        "preferred_for": "compound identity, synonyms, and chemistry/property summaries",
        "fallback_tools": ["get_chembl_bioactivities", "get_guidetopharmacology_target"],
    },
    "get_dailymed_drug_label": {
        "overlap_group": "drug_safety_regulatory",
        "preferred_for": "current US label language such as boxed warnings, indications, contraindications, and precautions",
        "fallback_tools": ["search_fda_adverse_events"],
    },
    "search_fda_adverse_events": {
        "overlap_group": "drug_safety_regulatory",
        "preferred_for": "post-marketing adverse-event signals rather than label text",
        "fallback_tools": ["get_dailymed_drug_label"],
    },
    "get_variant_annotations": {
        "overlap_group": "variant_evidence",
        "preferred_for": "aggregate variant annotations across ClinVar, dbSNP, gnomAD, CADD, and COSMIC for a specific rsID, HGVS variant, gene+protein HGVS, or shorthand variant string such as KRAS G12C",
        "fallback_tools": ["annotate_variants_vep", "search_civic_variants", "get_clingen_gene_curation"],
    },
    "annotate_variants_vep": {
        "overlap_group": "variant_evidence",
        "preferred_for": "functional consequence and pathogenicity prediction scores such as SIFT, PolyPhen, and AlphaMissense",
        "fallback_tools": ["get_variant_annotations", "search_civic_variants"],
    },
    "search_civic_variants": {
        "overlap_group": "variant_evidence",
        "preferred_for": "oncology-specific clinical variant interpretations",
        "fallback_tools": ["search_civic_genes", "get_variant_annotations", "get_clingen_gene_curation"],
    },
    "search_civic_genes": {
        "overlap_group": "variant_evidence",
        "preferred_for": "oncology gene-level CIViC context when the exact variant is not yet known",
        "fallback_tools": ["search_civic_variants", "search_variants_by_gene", "get_variant_annotations"],
    },
    "search_variants_by_gene": {
        "overlap_group": "variant_evidence",
        "preferred_for": "discovering variants in a gene when only gene symbol is known (use before get_variant_annotations or annotate_variants_vep)",
        "fallback_tools": ["search_civic_variants", "search_civic_genes"],
    },
    "get_clingen_gene_curation": {
        "overlap_group": "variant_evidence",
        "preferred_for": "expert-curated gene-disease validity and dosage sensitivity at the gene level",
        "fallback_tools": ["get_variant_annotations", "search_civic_genes"],
    },
    "get_gene_tissue_expression": {
        "overlap_group": "expression_context",
        "preferred_for": "bulk tissue RNA expression across human tissues",
        "fallback_tools": ["get_human_protein_atlas_gene", "search_cellxgene_datasets"],
    },
    "get_human_protein_atlas_gene": {
        "overlap_group": "expression_context",
        "preferred_for": "protein-level tissue specificity, subcellular localization, and atlas-style single-cell summaries",
        "fallback_tools": ["get_gene_tissue_expression", "search_cellxgene_datasets"],
    },
    "search_cellxgene_datasets": {
        "overlap_group": "expression_context",
        "preferred_for": "discovering relevant single-cell datasets by tissue, disease, assay, or cell type",
        "fallback_tools": ["get_human_protein_atlas_gene", "get_gene_tissue_expression"],
    },
    "get_depmap_gene_dependency": {
        "overlap_group": "target_vulnerability",
        "preferred_for": "release-level named-gene dependency and selectivity summaries across cancer cell lines, not lineage/mutation-filtered co-dependency discovery",
        "fallback_tools": ["get_biogrid_orcs_gene_summary", "get_gdsc_drug_sensitivity"],
    },
    "get_biogrid_orcs_gene_summary": {
        "overlap_group": "target_vulnerability",
        "preferred_for": "published CRISPR screen evidence with phenotype, cell-line, and screen-level context",
        "fallback_tools": ["get_depmap_gene_dependency", "get_gdsc_drug_sensitivity"],
    },
    "get_gdsc_drug_sensitivity": {
        "overlap_group": "target_vulnerability",
        "preferred_for": "GDSC / CancerRxGene compound sensitivity and pharmacogenomic response across cancer cell lines",
        "fallback_tools": ["get_prism_repurposing_response", "get_pharmacodb_compound_response", "get_guidetopharmacology_target"],
    },
    "get_prism_repurposing_response": {
        "overlap_group": "target_vulnerability",
        "preferred_for": "Broad PRISM repurposing primary-screen single-dose log2-fold-change viability across pooled cell lines",
        "fallback_tools": ["get_gdsc_drug_sensitivity", "get_pharmacodb_compound_response", "get_guidetopharmacology_target"],
    },
    "get_pharmacodb_compound_response": {
        "overlap_group": "target_vulnerability",
        "preferred_for": "cross-dataset compound-response context across PharmacoDB datasets such as PRISM, GDSC, CTRPv2, and related public screens",
        "fallback_tools": ["get_gdsc_drug_sensitivity", "get_prism_repurposing_response", "get_guidetopharmacology_target"],
    },
    "search_hpo_terms": {
        "overlap_group": "phenotype_rare_disease",
        "preferred_for": "phenotype-term normalization and choosing the right HPO concept before disease or graph queries",
        "fallback_tools": ["get_orphanet_disease_profile", "query_monarch_associations"],
    },
    "get_orphanet_disease_profile": {
        "overlap_group": "phenotype_rare_disease",
        "preferred_for": "rare-disease profiles, disease xrefs, curated phenotype sets, and curated disease-gene links",
        "fallback_tools": ["search_hpo_terms", "query_monarch_associations"],
    },
    "query_monarch_associations": {
        "overlap_group": "phenotype_rare_disease",
        "preferred_for": "phenotype-driven or rare-disease graph reasoning after entity normalization, especially when you need disease-to-gene, phenotype-to-gene, disease-to-phenotype, or gene-to-phenotype associations",
        "fallback_tools": ["search_hpo_terms", "get_orphanet_disease_profile"],
    },
    "get_alliance_genome_gene_profile": {
        "overlap_group": "translational_model_evidence",
        "preferred_for": "model-organism evidence, ortholog context, and Alliance-wide translational summaries for a gene",
        "fallback_tools": ["get_clingen_gene_curation", "query_monarch_associations", "get_orphanet_disease_profile"],
    },
    "search_openneuro_datasets": {
        "overlap_group": "neuroscience_dataset_discovery",
        "preferred_for": "public BIDS neuroimaging discovery when modality filters, dataset metadata, and OpenNeuro dataset IDs matter",
        "fallback_tools": ["search_nemar_datasets", "search_dandi_datasets", "search_braincode_datasets", "search_conp_datasets"],
    },
    "search_nemar_datasets": {
        "overlap_group": "neuroscience_dataset_discovery",
        "preferred_for": "EEG/MEG/iEEG dataset discovery, especially BIDS electrophysiology and epilepsy-related public archive search",
        "fallback_tools": ["search_openneuro_datasets", "search_dandi_datasets", "search_braincode_datasets", "search_conp_datasets"],
    },
    "search_dandi_datasets": {
        "overlap_group": "neuroscience_dataset_discovery",
        "preferred_for": "NWB/BIDS neurophysiology discovery when the signal is more about task, assay, or neurophysiology modality than a disease label",
        "fallback_tools": ["search_openneuro_datasets", "search_nemar_datasets", "search_braincode_datasets", "search_conp_datasets"],
    },
    "search_braincode_datasets": {
        "overlap_group": "neuroscience_dataset_discovery",
        "preferred_for": "Brain-CODE public-release discovery through the CONP mirror when Ontario Brain Institute datasets or Brain-CODE-specific releases are requested",
        "fallback_tools": ["search_conp_datasets", "search_openneuro_datasets", "search_nemar_datasets", "search_dandi_datasets"],
    },
    "search_conp_datasets": {
        "overlap_group": "neuroscience_dataset_discovery",
        "preferred_for": "broader CONP repository discovery by modality, method, or study name when disease labels are sparse or archive-specific search is too narrow",
        "fallback_tools": ["search_braincode_datasets", "search_openneuro_datasets", "search_nemar_datasets", "search_dandi_datasets"],
    },
}

SOURCE_PRECEDENCE_RULES: list[dict[str, Any]] = [
    {
        "topic": "Literature search",
        "tools": ["search_pubmed", "search_europe_pmc_literature", "search_openalex_works"],
        "summary": "Use `search_pubmed` by default for biomedical papers and PMIDs; use `search_europe_pmc_literature` when preprints, citation metadata, or open-access status matter; use `search_openalex_works` for broader citation graph or researcher context.",
    },
    {
        "topic": "Pathway context",
        "tools": ["search_reactome_pathways", "search_pathway_commons_top_pathways"],
        "summary": "Use `search_reactome_pathways` for specific curated pathway titles; use `search_pathway_commons_top_pathways` when you want integrated pathway context across multiple providers.",
    },
    {
        "topic": "Interaction evidence",
        "tools": ["get_intact_interactions", "get_biogrid_interactions", "get_string_interactions", "search_pathway_commons_top_pathways"],
        "summary": "Use `get_intact_interactions` for deeply curated molecular interaction records with detection methods and PMIDs; use `get_biogrid_interactions` for broader experimental physical/genetic interaction coverage and throughput context; use `get_string_interactions` for broader network neighborhoods; use `search_pathway_commons_top_pathways` when the goal is pathway-level context rather than pairwise interactions.",
    },
    {
        "topic": "Compound pharmacology",
        "tools": ["get_guidetopharmacology_target", "get_chembl_bioactivities", "search_drug_gene_interactions", "get_pubchem_compound"],
        "summary": "Use `get_guidetopharmacology_target` for curated target-ligand summaries; use `get_chembl_bioactivities` for quantitative potency/selectivity; use `search_drug_gene_interactions` for broad druggability coverage; use `get_pubchem_compound` for compound identity and properties.",
    },
    {
        "topic": "Drug label vs safety signals",
        "tools": ["get_dailymed_drug_label", "search_fda_adverse_events"],
        "summary": "Use `get_dailymed_drug_label` for current US label language; use `search_fda_adverse_events` for post-marketing adverse-event signals rather than regulatory label text.",
    },
    {
        "topic": "Variant evidence",
        "tools": ["search_variants_by_gene", "get_variant_annotations", "annotate_variants_vep", "search_civic_variants", "search_civic_genes", "get_clingen_gene_curation"],
        "summary": "When only gene symbol is known: use `search_variants_by_gene` (or `search_civic_variants` for cancer genes) to discover variants, then `get_variant_annotations` or `annotate_variants_vep`. Variant-level tools require rsID or HGVS—not gene symbols.",
    },
    {
        "topic": "Epitope evidence discovery",
        "tools": ["search_iedb_epitope_evidence", "search_pubmed", "get_paper_fulltext"],
        "summary": "Use `search_iedb_epitope_evidence` for direct IEDB epitope, T-cell, and MHC ligand evidence when you have a peptide, antigen, or allele filter. Use `search_pubmed` and `get_paper_fulltext` first when you need to recover the exact peptide sequence, assay context, or mutation-specific details before querying IEDB.",
    },
    {
        "topic": "Expression context",
        "tools": ["get_gene_tissue_expression", "get_human_protein_atlas_gene", "search_cellxgene_datasets"],
        "summary": "Use `get_gene_tissue_expression` for bulk tissue RNA, `get_human_protein_atlas_gene` for protein localization and atlas summaries, and `search_cellxgene_datasets` when the task is finding relevant single-cell datasets rather than returning expression values directly.",
    },
    {
        "topic": "Neuroscience dataset discovery",
        "tools": ["search_openneuro_datasets", "search_nemar_datasets", "search_dandi_datasets", "search_braincode_datasets", "search_conp_datasets"],
        "summary": "Use `search_nemar_datasets` for EEG/MEG/iEEG archive discovery, `search_openneuro_datasets` for public neuroimaging with modality filters, `search_dandi_datasets` for NWB/BIDS neurophysiology, and `search_braincode_datasets` or `search_conp_datasets` for Brain-CODE or broader CONP mirrors.",
    },
    {
        "topic": "Functional screening vs drug response",
        "tools": [
            "get_depmap_gene_dependency",
            "get_biogrid_orcs_gene_summary",
            "get_gdsc_drug_sensitivity",
            "get_prism_repurposing_response",
            "get_pharmacodb_compound_response",
        ],
        "summary": "Use `get_depmap_gene_dependency` for release-level gene essentiality and vulnerability metrics; use `get_biogrid_orcs_gene_summary` for published CRISPR screens with phenotype and cell-line context; use `get_gdsc_drug_sensitivity` for GDSC / CancerRxGene response; use `get_prism_repurposing_response` for Broad PRISM single-dose repurposing response; use `get_pharmacodb_compound_response` for harmonized cross-dataset drug-response context across public screens.",
    },
    {
        "topic": "Phenotype and rare-disease reasoning",
        "tools": ["search_hpo_terms", "get_orphanet_disease_profile", "query_monarch_associations"],
        "summary": "Use `search_hpo_terms` for phenotype-term normalization, `get_orphanet_disease_profile` for rare-disease profiles and curated phenotype/gene summaries, and `query_monarch_associations` for phenotype-driven or graph-style disease/gene association reasoning once the entity has been normalized to a reliable CURIE.",
    },
    {
        "topic": "Translational model-organism evidence",
        "tools": ["get_alliance_genome_gene_profile", "get_clingen_gene_curation", "get_orphanet_disease_profile", "query_monarch_associations"],
        "summary": "Use `get_alliance_genome_gene_profile` for orthologs, disease models, and model-organism translational context; use `get_clingen_gene_curation` for expert human gene-disease validity; use `get_orphanet_disease_profile` and `query_monarch_associations` for rare-disease and phenotype-centric reasoning.",
    },
]

TOOL_SOURCE_NAMES: dict[str, str] = {
    "list_bigquery_tables": "BigQuery",
    "run_bigquery_select_query": "BigQuery",
    "open_targets_platform": "Open Targets Platform",
    "ebi_chembl": "ChEMBL",
    "gnomad": "gnomAD",
    "human_genome_variants": "Human Genome Variants",
    "human_variant_annotation": "ClinVar (BigQuery)",
    "search_iedb_epitope_evidence": "IEDB",
    "nlm_rxnorm": "RxNorm",
    "fda_drug": "FDA Drug (BigQuery)",
    "umiami_lincs": "LINCS L1000",
    "ebi_surechembl": "SureChEMBL",
    "annotate_variants_vep": "Ensembl VEP",
    "get_variant_annotations": "MyVariant.info",
    "resolve_gene_identifiers": "MyGene.info",
    "map_ontology_terms_oxo": "EBI OxO",
    "search_hpo_terms": "Human Phenotype Ontology",
    "get_orphanet_disease_profile": "Orphanet / ORDO",
    "query_monarch_associations": "Monarch Initiative",
    "get_alliance_genome_gene_profile": "Alliance Genome Resources",
    "search_quickgo_terms": "QuickGO",
    "get_quickgo_annotations": "QuickGO",
    "search_civic_variants": "CIViC",
    "search_civic_genes": "CIViC",
    "search_variants_by_gene": "MyVariant.info",
    "get_alphafold_structure": "AlphaFold API",
    "search_gwas_associations": "GWAS Catalog",
    "search_drug_gene_interactions": "DGIdb",
    "get_gene_tissue_expression": "GTEx",
    "get_human_protein_atlas_gene": "Human Protein Atlas",
    "get_depmap_gene_dependency": "DepMap",
    "get_biogrid_orcs_gene_summary": "BioGRID ORCS",
    "get_gdsc_drug_sensitivity": "GDSC / CancerRxGene",
    "get_prism_repurposing_response": "PRISM Repurposing",
    "get_pharmacodb_compound_response": "PharmacoDB",
    "search_cellxgene_datasets": "CELLxGENE Discover / Census",
    "search_protein_structures": "RCSB PDB",
    "get_cancer_mutation_profile": "cBioPortal",
    "get_chembl_bioactivities": "ChEMBL API",
    "get_pubchem_compound": "PubChem",
    "search_aba_genes": "Allen Brain Atlas",
    "search_aba_structures": "Allen Brain Atlas",
    "get_aba_gene_expression": "Allen Brain Atlas",
    "search_aba_differential_expression": "Allen Brain Atlas",
    "search_ebrains_kg": "EBRAINS Knowledge Graph",
    "get_ebrains_kg_document": "EBRAINS Knowledge Graph",
    "search_conp_datasets": "CONP Datasets",
    "get_conp_dataset_details": "CONP Datasets",
    "query_neurobagel_cohorts": "Neurobagel",
    "search_openneuro_datasets": "OpenNeuro",
    "get_openneuro_dataset": "OpenNeuro",
    "search_dandi_datasets": "DANDI Archive",
    "get_dandi_dataset": "DANDI Archive",
    "search_nemar_datasets": "NEMAR",
    "get_nemar_dataset_details": "NEMAR",
    "search_braincode_datasets": "Brain-CODE",
    "get_braincode_dataset_details": "Brain-CODE",
    "search_enigma_datasets": "ENIGMA",
    "get_enigma_dataset_info": "ENIGMA",
    "search_fda_adverse_events": "FDA FAERS (openFDA)",
    "search_pubmed": "PubMed",
    "search_pubmed_advanced": "PubMed",
    "get_pubmed_abstract": "PubMed",
    "get_paper_fulltext": "PubMed Central",
    "search_geo_datasets": "Gene Expression Omnibus",
    "get_geo_dataset": "Gene Expression Omnibus",
    "benchmark_dataset_overview": "Benchmark Datasets",
    "check_gpqa_access": "GPQA",
    "search_clinical_trials": "ClinicalTrials.gov",
    "get_clinical_trial": "ClinicalTrials.gov",
    "summarize_clinical_trials_landscape": "ClinicalTrials.gov",
    "search_openalex_works": "OpenAlex",
    "search_openalex_authors": "OpenAlex",
    "rank_researchers_by_activity": "OpenAlex",
    "get_researcher_contact_candidates": "OpenAlex",
    "search_europe_pmc_literature": "Europe PMC",
    "search_reactome_pathways": "Reactome",
    "search_pathway_commons_top_pathways": "Pathway Commons",
    "get_guidetopharmacology_target": "Guide to Pharmacology",
    "get_dailymed_drug_label": "DailyMed",
    "get_clingen_gene_curation": "ClinGen",
    "get_string_interactions": "STRING",
    "get_intact_interactions": "IntAct",
    "get_biogrid_interactions": "BioGRID",
    "search_uniprot_proteins": "UniProt",
    "get_uniprot_protein_profile": "UniProt",
}


def iter_active_source_precedence_rules(tool_hints: list[str]) -> list[dict[str, Any]]:
    """Return only the overlap rules relevant to the current tool set."""
    active_tools = set(tool_hints)
    active_rules: list[dict[str, Any]] = []
    for rule in SOURCE_PRECEDENCE_RULES:
        rule_tools = [tool for tool in rule.get("tools", []) if tool in active_tools]
        if len(rule_tools) >= 2:
            active_rules.append(rule)
    return active_rules


__all__ = [
    "ALL_DOMAIN_NAMES",
    "ALWAYS_AVAILABLE_DOMAINS",
    "SOURCE_PRECEDENCE_RULES",
    "TOOL_DESCRIPTIONS",
    "TOOL_DOMAINS",
    "TOOL_ROUTING_METADATA",
    "TOOL_SOURCE_NAMES",
    "TOOL_TO_DOMAINS",
    "iter_active_source_precedence_rules",
]
