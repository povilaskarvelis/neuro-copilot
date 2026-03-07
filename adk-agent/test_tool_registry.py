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
