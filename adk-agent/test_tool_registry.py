import neuro_copilot.tool_registry as tool_registry


def test_tool_to_domains_contains_expected_cross_domain_tool():
    assert "protein" in tool_registry.TOOL_TO_DOMAINS["get_guidetopharmacology_target"]
    assert "chemistry" in tool_registry.TOOL_TO_DOMAINS["get_guidetopharmacology_target"]


def test_iter_active_source_precedence_rules_filters_to_present_tools():
    active = tool_registry.iter_active_source_precedence_rules(
        ["search_pubmed", "search_europe_pmc_literature", "get_clinical_trial"]
    )
    topics = [rule["topic"] for rule in active]
    assert topics == ["Literature search"]


def test_iter_active_source_precedence_rules_ignores_single_tool_overlap_groups():
    active = tool_registry.iter_active_source_precedence_rules(["search_pubmed", "get_clinical_trial"])
    assert active == []


def test_iter_active_source_precedence_rules_includes_neuroscience_dataset_discovery():
    active = tool_registry.iter_active_source_precedence_rules(
        ["search_openneuro_datasets", "search_nemar_datasets", "get_openneuro_dataset"]
    )
    topics = [rule["topic"] for rule in active]
    assert "Neuroscience dataset discovery" in topics


def test_open_targets_archive_tool_metadata_mentions_release_specific_lookup():
    assert "data" in tool_registry.TOOL_TO_DOMAINS["get_open_targets_association"]
    desc = tool_registry.TOOL_DESCRIPTIONS["get_open_targets_association"]
    assert "release" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_open_targets_association"]["preferred_for"]
    assert "specific platform release" in preferred_for


def test_open_targets_l2g_tool_metadata_mentions_credible_sets():
    assert "data" in tool_registry.TOOL_TO_DOMAINS["get_open_targets_l2g"]
    desc = tool_registry.TOOL_DESCRIPTIONS["get_open_targets_l2g"]
    assert "credible" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_open_targets_l2g"]["preferred_for"]
    assert "locus-to-gene" in preferred_for.lower()


def test_ensembl_canonical_transcript_tool_metadata_mentions_tss():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_ensembl_canonical_transcript"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_ensembl_canonical_transcript"] == "Ensembl"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_ensembl_canonical_transcript"]
    assert "canonical transcript" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_ensembl_canonical_transcript"]["preferred_for"]
    assert "tss" in preferred_for.lower()


def test_ensembl_transcripts_by_protein_length_tool_metadata_mentions_amino_acid_range():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_ensembl_transcripts_by_protein_length"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_ensembl_transcripts_by_protein_length"] == "Ensembl"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_ensembl_transcripts_by_protein_length"]
    assert "protein length" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_ensembl_transcripts_by_protein_length"]["preferred_for"]
    assert "amino-acid range" in preferred_for


def test_iter_active_source_precedence_rules_includes_open_targets_genetics_lookup():
    active = tool_registry.iter_active_source_precedence_rules(
        ["get_open_targets_l2g", "run_bigquery_select_query"]
    )
    topics = [rule["topic"] for rule in active]
    assert "Open Targets genetics lookup" in topics


def test_iter_active_source_precedence_rules_includes_open_targets_lookup():
    active = tool_registry.iter_active_source_precedence_rules(
        ["get_open_targets_association", "run_bigquery_select_query"]
    )
    topics = [rule["topic"] for rule in active]
    assert "Open Targets association lookup" in topics


def test_gwas_study_variant_tool_metadata_mentions_raf_lookup():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_gwas_study_variant_association"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_gwas_study_variant_association"] == "GWAS Catalog"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_gwas_study_variant_association"]
    assert "raf" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_gwas_study_variant_association"]["preferred_for"]
    assert "study+variant" in preferred_for


def test_gwas_study_top_risk_allele_tool_metadata_mentions_highest_pvalue_questions():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_gwas_study_top_risk_allele"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_gwas_study_top_risk_allele"] == "GWAS Catalog"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_gwas_study_top_risk_allele"]
    assert "highest or lowest" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_gwas_study_top_risk_allele"]["preferred_for"]
    assert "highest or lowest p-value" in preferred_for


def test_jaspar_tool_metadata_mentions_consensus_and_information_content():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_jaspar_motif_profile"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_jaspar_motif_profile"] == "JASPAR"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_jaspar_motif_profile"]
    assert "consensus" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_jaspar_motif_profile"]["preferred_for"]
    assert "information content" in preferred_for


def test_gnomad_gene_constraint_tool_metadata_mentions_pli():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_gnomad_gene_constraint"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_gnomad_gene_constraint"] == "gnomAD"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_gnomad_gene_constraint"]
    assert "pli" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_gnomad_gene_constraint"]["preferred_for"]
    assert "pLI" in preferred_for


def test_gnomad_transcript_highest_af_region_tool_metadata_mentions_transcript_region():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_gnomad_transcript_highest_af_region"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_gnomad_transcript_highest_af_region"] == "gnomAD"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_gnomad_transcript_highest_af_region"]
    assert "highest-allele-frequency" in desc
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_gnomad_transcript_highest_af_region"]["preferred_for"]
    assert "transcript region" in preferred_for


def test_regulomedb_variant_summary_tool_metadata_mentions_rank_and_motif_counts():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_regulomedb_variant_summary"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_regulomedb_variant_summary"] == "RegulomeDB"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_regulomedb_variant_summary"]
    assert "motif" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_regulomedb_variant_summary"]["preferred_for"]
    assert "rank" in preferred_for.lower()


def test_dbsnp_population_frequency_tool_metadata_mentions_alfa():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_dbsnp_population_frequency"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_dbsnp_population_frequency"] == "dbSNP"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_dbsnp_population_frequency"]
    assert "alfa" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_dbsnp_population_frequency"]["preferred_for"]
    assert "population" in preferred_for.lower()


def test_screen_tool_metadata_mentions_nearest_enhancer_and_top_celltype():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_screen_nearest_ccre_assay"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_screen_nearest_ccre_assay"] == "SCREEN"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_screen_nearest_ccre_assay"]
    assert "nearest" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_screen_nearest_ccre_assay"]["preferred_for"]
    assert "enhancer" in preferred_for.lower()

    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_screen_ccre_top_celltype_assay"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_screen_ccre_top_celltype_assay"] == "SCREEN"
    desc_top = tool_registry.TOOL_DESCRIPTIONS["get_screen_ccre_top_celltype_assay"]
    assert "highest-scoring cell type" in desc_top.lower()
    preferred_for_top = tool_registry.TOOL_ROUTING_METADATA["get_screen_ccre_top_celltype_assay"]["preferred_for"]
    assert "highest assay z-score" in preferred_for_top.lower()


def test_tcga_project_data_availability_tool_metadata_mentions_case_counts():
    assert "data" in tool_registry.TOOL_TO_DOMAINS["get_tcga_project_data_availability"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_tcga_project_data_availability"] == "GDC"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_tcga_project_data_availability"]
    assert "case" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_tcga_project_data_availability"]["preferred_for"]
    assert "case counts" in preferred_for


def test_cellxgene_marker_tool_metadata_mentions_marker_rankings():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_cellxgene_marker_genes"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_cellxgene_marker_genes"] == "CELLxGENE WMG"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_cellxgene_marker_genes"]
    assert "marker" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_cellxgene_marker_genes"]["preferred_for"]
    assert "marker-gene" in preferred_for


def test_depmap_expression_subset_tool_metadata_mentions_subset_means():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_depmap_expression_subset_mean"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_depmap_expression_subset_mean"] == "DepMap"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_depmap_expression_subset_mean"]
    assert "mean log2(tpm+1)" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_depmap_expression_subset_mean"]["preferred_for"]
    assert "model subset" in preferred_for


def test_depmap_sample_top_expression_tool_metadata_mentions_named_samples():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_depmap_sample_top_expression_gene"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_depmap_sample_top_expression_gene"] == "DepMap"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_depmap_sample_top_expression_gene"]
    assert "highest-expressed gene" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_depmap_sample_top_expression_gene"]["preferred_for"]
    assert "named sample" in preferred_for


def test_direct_source_metadata_tools_are_registered():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_ena_experiment_profile"]
    assert "protein" in tool_registry.TOOL_TO_DOMAINS["get_emdb_entry_metadata"]
    assert "chemistry" in tool_registry.TOOL_TO_DOMAINS["get_gtopdb_ligand_reference"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_ena_experiment_profile"] == "ENA"
    assert tool_registry.TOOL_SOURCE_NAMES["get_emdb_entry_metadata"] == "EMDB"
    assert tool_registry.TOOL_SOURCE_NAMES["get_gtopdb_ligand_reference"] == "GtoPdb"


def test_geo_cell_type_proportions_tool_metadata_mentions_donor_filtered_composition():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_geo_cell_type_proportions"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_geo_cell_type_proportions"] == "Gene Expression Omnibus"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_geo_cell_type_proportions"]
    assert "proportion" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_geo_cell_type_proportions"]["preferred_for"]
    assert "cell-type composition" in preferred_for


def test_alphafold_domain_plddt_tool_metadata_mentions_domain_level_means():
    assert "protein" in tool_registry.TOOL_TO_DOMAINS["get_alphafold_domain_plddt"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_alphafold_domain_plddt"] == "AlphaFold DB Archive / API"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_alphafold_domain_plddt"]
    assert "domain-level" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_alphafold_domain_plddt"]["preferred_for"]
    assert "domain-level alphafold" in preferred_for.lower()


def test_iter_active_source_precedence_rules_includes_gwas_catalog_lookup():
    active = tool_registry.iter_active_source_precedence_rules(
        ["get_gwas_study_variant_association", "search_gwas_associations"]
    )
    topics = [rule["topic"] for rule in active]
    assert "GWAS Catalog lookup" in topics


def test_iter_active_source_precedence_rules_includes_variant_evidence_overlap_for_gnomad_and_regulomedb():
    active = tool_registry.iter_active_source_precedence_rules(
        ["get_gnomad_gene_constraint", "get_regulomedb_variant_summary", "get_variant_annotations"]
    )
    topics = [rule["topic"] for rule in active]
    assert "Variant evidence" in topics


def test_iter_active_source_precedence_rules_includes_expression_context_marker_tool():
    active = tool_registry.iter_active_source_precedence_rules(
        ["get_cellxgene_marker_genes", "search_cellxgene_datasets"]
    )
    topics = [rule["topic"] for rule in active]
    assert "Expression context" in topics


def test_iter_active_source_precedence_rules_include_depmap_expression_subset_tool():
    active = tool_registry.iter_active_source_precedence_rules(
        ["get_depmap_gene_dependency", "get_depmap_expression_subset_mean"]
    )
    topics = [rule["topic"] for rule in active]
    assert "Functional screening vs drug response" in topics
