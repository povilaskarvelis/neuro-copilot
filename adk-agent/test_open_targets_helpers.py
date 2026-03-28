from pathlib import Path
import sys

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_MCP_DIR = REPO_ROOT / "research-mcp"
if str(RESEARCH_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_MCP_DIR))

import open_targets_l2g_query as l2g_query
import open_targets_release_query as release_query


def test_lookup_association_accepts_association_score_column(monkeypatch):
    monkeypatch.setattr(release_query, "list_parquet_parts", lambda _url: ["part-00000.parquet"])
    monkeypatch.setattr(release_query, "cache_download", lambda _url, _relative: Path("/tmp/fake.parquet"))

    seen_columns = []

    def fake_read_parquet(_path, columns=None):
        seen_columns.append(tuple(columns or []))
        assert columns == ["diseaseId", "targetId", "associationScore", "evidenceCount"]
        return pd.DataFrame(
            [
                {
                    "diseaseId": "EFO_0001073",
                    "targetId": "ENSG00000166603",
                    "associationScore": 0.8171594671385342,
                    "evidenceCount": 42,
                }
            ]
        )

    monkeypatch.setattr(release_query.pd, "read_parquet", fake_read_parquet)

    result = release_query.lookup_association("ENSG00000166603", "EFO_0001073", "26.03")

    assert result["score"] == pytest.approx(0.8171594671385342)
    assert result["evidence_count"] == 42
    assert any("associationScore" in columns for columns in seen_columns)


def test_open_targets_release_run_returns_found_false_when_association_missing(monkeypatch):
    monkeypatch.setattr(release_query, "normalize_release_tag", lambda _raw: "26.03")
    monkeypatch.setattr(
        release_query,
        "resolve_target",
        lambda _query: {
            "query": "IAPP",
            "target_id": "ENSG00000121351",
            "target_symbol": "IAPP",
            "target_name": "islet amyloid polypeptide",
            "resolution_source": "mygene://IAPP",
        },
    )
    monkeypatch.setattr(
        release_query,
        "resolve_disease",
        lambda _query, _release, max_candidates=5: {
            "disease_id": "EFO_0001073",
            "disease_name": "obesity",
            "resolution_source": "ot://disease",
            "candidate_matches": [
                {"disease_id": "EFO_0001073", "disease_name": "obesity"},
                {"disease_id": "EFO_0001074", "disease_name": "morbid obesity"},
            ][:max_candidates],
        },
    )
    monkeypatch.setattr(
        release_query,
        "lookup_association",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            LookupError("No Open Targets association_overall_direct row found for target ENSG00000121351 and disease EFO_0001073 in release 26.03.")
        ),
    )

    result = release_query.run({"target": "IAPP", "disease": "obesity"})

    assert result["ok"] is True
    assert result["found"] is False
    assert result["target_id"] == "ENSG00000121351"
    assert result["disease_id"] == "EFO_0001073"
    assert "No Open Targets association_overall_direct row found" in result["message"]


def test_open_targets_l2g_run_returns_found_false_when_pair_has_no_match(monkeypatch):
    monkeypatch.setattr(l2g_query, "normalize_release_tag", lambda _raw: "26.03")
    monkeypatch.setattr(
        l2g_query,
        "resolve_target",
        lambda _query: {
            "query": "IAPP",
            "target_id": "ENSG00000121351",
            "target_symbol": "IAPP",
            "target_name": "islet amyloid polypeptide",
            "resolution_source": "mygene://IAPP",
        },
    )
    monkeypatch.setattr(
        l2g_query,
        "resolve_disease",
        lambda _query, _release, max_candidates=5: {
            "disease_id": "EFO_0001073",
            "disease_name": "obesity",
            "resolution_source": "ot://disease",
            "candidate_matches": [
                {"disease_id": "EFO_0001073", "disease_name": "obesity"},
                {"disease_id": "EFO_0001074", "disease_name": "morbid obesity"},
            ][:max_candidates],
        },
    )
    monkeypatch.setattr(l2g_query, "fetch_live_candidates", lambda **_kwargs: [])
    monkeypatch.setattr(
        l2g_query,
        "load_candidate_studies",
        lambda *_args, **_kwargs: [
            {
                "study_id": "GCST000001",
                "trait_from_source": "Obesity",
                "publication_first_author": "Doe",
                "publication_date": "2024-01-01",
                "study_source_url": "study://1",
                "match_score": 2500.0,
            }
        ],
    )
    monkeypatch.setattr(
        l2g_query,
        "load_target_l2g_rows",
        lambda *_args, **_kwargs: [
            {
                "study_locus_id": "study-locus-1",
                "target_id": "ENSG00000121351",
                "score": 0.25,
                "shap_base_value": 0.1,
                "l2g_source_url": "l2g://1",
            }
        ],
    )
    monkeypatch.setattr(
        l2g_query,
        "load_candidate_credible_sets",
        lambda *_args, **_kwargs: [
            {
                "study_locus_id": "study-locus-1",
                "study_id": "GCST000001",
                "variant_id": "1_100_A_G",
                "credible_set_source_url": "credible://1",
            }
        ],
    )
    monkeypatch.setattr(
        l2g_query,
        "choose_best_match",
        lambda **_kwargs: (_ for _ in ()).throw(
            LookupError("No Open Targets L2G rows matched target ENSG00000121351 and disease 'obesity' in release 26.03.")
        ),
    )

    result = l2g_query.run({"target": "IAPP", "disease": "obesity"})

    assert result["ok"] is True
    assert result["found"] is False
    assert result["target_id"] == "ENSG00000121351"
    assert result["disease_id"] == "EFO_0001073"
    assert result["candidate_matches"] == []
    assert "No Open Targets L2G rows matched target ENSG00000121351" in result["message"]
