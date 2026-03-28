import co_scientist.tool_registry as tool_registry


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


def test_jaspar_tool_metadata_mentions_consensus_and_information_content():
    assert "genomics" in tool_registry.TOOL_TO_DOMAINS["get_jaspar_motif_profile"]
    assert tool_registry.TOOL_SOURCE_NAMES["get_jaspar_motif_profile"] == "JASPAR"
    desc = tool_registry.TOOL_DESCRIPTIONS["get_jaspar_motif_profile"]
    assert "consensus" in desc.lower()
    preferred_for = tool_registry.TOOL_ROUTING_METADATA["get_jaspar_motif_profile"]["preferred_for"]
    assert "information content" in preferred_for


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
