#!/usr/bin/env python3
"""Resolve Open Targets L2G scores from official archived releases.

This helper is intentionally narrow: it resolves one target-disease lookup
against the official EBI Open Targets archive and prints JSON for the Node MCP
server.
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Any
import urllib.request

import pandas as pd

from open_targets_release_query import cache_download
from open_targets_release_query import list_parquet_parts
from open_targets_release_query import normalize_release_tag
from open_targets_release_query import normalize_whitespace
from open_targets_release_query import release_output_base
from open_targets_release_query import resolve_disease
from open_targets_release_query import resolve_target


VARIANT_ID_PATTERN = re.compile(r"^(?:chr)?([0-9XYM]+)_(\d+)_([ACGTN]+)_([ACGTN]+)$", re.IGNORECASE)
OPEN_TARGETS_GRAPHQL_API = "https://api.platform.opentargets.org/api/v4/graphql"
LIVE_L2G_QUERY = """
query CandidateStudies($diseaseIds: [String!], $page: Pagination) {
  studies(diseaseIds: $diseaseIds, page: $page) {
    count
    rows {
      id
      traitFromSource
      publicationFirstAuthor
      publicationDate
      credibleSets(page: { index: 0, size: 50 }) {
        rows {
          studyLocusId
          variant {
            id
            rsIds
          }
          l2GPredictions {
            rows {
              score
              target {
                id
                approvedSymbol
                approvedName
              }
            }
          }
        }
      }
    }
  }
}
"""


def _flatten_strings(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = normalize_whitespace(raw_value)
        return [text] if text else []
    if isinstance(raw_value, dict):
        values: list[str] = []
        for nested in raw_value.values():
            values.extend(_flatten_strings(nested))
        return values
    if isinstance(raw_value, (list, tuple, set)):
        values: list[str] = []
        for item in raw_value:
            values.extend(_flatten_strings(item))
        return values
    if hasattr(raw_value, "tolist"):
        return _flatten_strings(raw_value.tolist())
    text = normalize_whitespace(raw_value)
    return [text] if text else []


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed


def _post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _cache_part(dataset_name: str, release_tag: str, part_name: str) -> tuple[str, str]:
    dataset_url = f"{release_output_base(release_tag)}/{dataset_name}"
    part_url = f"{dataset_url}/{part_name}"
    part_path = cache_download(part_url, f"{release_tag}/{dataset_name}/{part_name}")
    return part_url, str(part_path)


def _read_part(
    dataset_name: str,
    release_tag: str,
    part_name: str,
    *,
    columns: list[str],
    filters: list[tuple[str, str, Any]] | None = None,
) -> tuple[str, pd.DataFrame]:
    part_url, part_path = _cache_part(dataset_name, release_tag, part_name)
    try:
        frame = pd.read_parquet(part_path, columns=columns, filters=filters)
    except Exception:
        frame = pd.read_parquet(part_path, columns=columns)
    return part_url, frame


def _normalize_variant_id(raw_value: str) -> str:
    raw = normalize_whitespace(raw_value)
    match = VARIANT_ID_PATTERN.fullmatch(raw)
    if not match:
        return ""
    chrom = match.group(1).upper().replace("CHR", "")
    return f"{chrom}_{match.group(2)}_{match.group(3).upper()}_{match.group(4).upper()}"


def _study_match_score(
    *,
    trait_from_source: str,
    disease_query: str,
    disease_name: str,
    disease_id: str,
    disease_ids: list[str],
) -> float:
    trait_lower = trait_from_source.lower()
    query_lower = disease_query.lower()
    disease_name_lower = disease_name.lower()
    score = float("-inf")

    if disease_id and disease_id in disease_ids:
        score = 900.0
    if trait_lower == query_lower:
        score = max(score, 2500.0)
    elif trait_lower == disease_name_lower:
        score = max(score, 2300.0)
    elif query_lower and query_lower in trait_lower:
        score = max(score, 1800.0 - abs(len(trait_from_source) - len(disease_query)))
    elif disease_name_lower and disease_name_lower in trait_lower:
        score = max(score, 1600.0 - abs(len(trait_from_source) - len(disease_name)))

    return score


def load_candidate_studies(
    disease_query: str,
    disease_id: str,
    disease_name: str,
    release_tag: str,
    max_candidates: int,
) -> list[dict[str, Any]]:
    part_names = list_parquet_parts(f"{release_output_base(release_tag)}/study")
    if not part_names:
        raise RuntimeError(f"No study parquet parts found for Open Targets release {release_tag}.")

    candidates: list[tuple[float, dict[str, Any]]] = []
    for part_name in part_names:
        part_url, frame = _read_part(
            "study",
            release_tag,
            part_name,
            columns=["studyId", "traitFromSource", "diseaseIds", "publicationFirstAuthor", "publicationDate"],
        )
        for row in frame.itertuples(index=False):
            trait_from_source = normalize_whitespace(getattr(row, "traitFromSource", ""))
            if not trait_from_source:
                continue
            disease_ids = _flatten_strings(getattr(row, "diseaseIds", None))
            score = _study_match_score(
                trait_from_source=trait_from_source,
                disease_query=disease_query,
                disease_name=disease_name,
                disease_id=disease_id,
                disease_ids=disease_ids,
            )
            if score == float("-inf"):
                continue
            candidates.append(
                (
                    score,
                    {
                        "study_id": normalize_whitespace(getattr(row, "studyId", "")),
                        "trait_from_source": trait_from_source,
                        "publication_first_author": normalize_whitespace(getattr(row, "publicationFirstAuthor", "")),
                        "publication_date": normalize_whitespace(getattr(row, "publicationDate", "")),
                        "study_source_url": part_url,
                        "match_score": score,
                    },
                )
            )

    if not candidates:
        raise LookupError(
            f"No Open Targets studies matched disease '{disease_query}' ({disease_id}) in release {release_tag}."
        )

    candidates.sort(key=lambda item: (item[0], item[1]["study_id"]), reverse=True)
    return [dict(candidate) for _, candidate in candidates[:max_candidates]]


def load_candidate_credible_sets(
    study_locus_ids: set[str],
    release_tag: str,
) -> list[dict[str, Any]]:
    part_names = list_parquet_parts(f"{release_output_base(release_tag)}/credible_set")
    if not part_names:
        raise RuntimeError(f"No credible_set parquet parts found for Open Targets release {release_tag}.")

    remaining_ids = set(study_locus_ids)
    rows: list[dict[str, Any]] = []
    for part_name in part_names:
        if not remaining_ids:
            break
        filters = [("studyLocusId", "in", sorted(remaining_ids))] if len(remaining_ids) <= 500 else None
        part_url, frame = _read_part(
            "credible_set",
            release_tag,
            part_name,
            columns=["studyLocusId", "studyId", "variantId"],
            filters=filters,
        )
        if frame.empty:
            continue
        frame = frame[frame["studyLocusId"].isin(remaining_ids)]
        if frame.empty:
            continue
        for record in frame.to_dict(orient="records"):
            study_locus_id = normalize_whitespace(record.get("studyLocusId"))
            rows.append(
                {
                    "study_locus_id": study_locus_id,
                    "study_id": normalize_whitespace(record.get("studyId")),
                    "variant_id": normalize_whitespace(record.get("variantId")),
                    "credible_set_source_url": part_url,
                }
            )
            remaining_ids.discard(study_locus_id)

    if not rows:
        raise LookupError(f"No Open Targets credible sets matched the candidate study-locus ids in release {release_tag}.")
    return rows


def load_target_l2g_rows(
    target_id: str,
    release_tag: str,
) -> list[dict[str, Any]]:
    part_names = list_parquet_parts(f"{release_output_base(release_tag)}/l2g_prediction")
    if not part_names:
        raise RuntimeError(f"No l2g_prediction parquet parts found for Open Targets release {release_tag}.")

    rows: list[dict[str, Any]] = []
    base_filters = [("geneId", "==", target_id)]
    for part_name in part_names:
        part_url, frame = _read_part(
            "l2g_prediction",
            release_tag,
            part_name,
            columns=["studyLocusId", "geneId", "score", "shapBaseValue"],
            filters=base_filters,
        )
        if frame.empty:
            continue
        for record in frame.to_dict(orient="records"):
            rows.append(
                {
                    "study_locus_id": normalize_whitespace(record.get("studyLocusId")),
                    "target_id": normalize_whitespace(record.get("geneId")),
                    "score": _safe_float(record.get("score")),
                    "shap_base_value": _safe_float(record.get("shapBaseValue")),
                    "l2g_source_url": part_url,
                }
            )

    if not rows:
        raise LookupError(
            f"No Open Targets L2G rows matched target {target_id} and the candidate studies in release {release_tag}."
        )
    return rows


def load_candidate_variant_aliases(
    candidate_variant_ids: set[str],
    release_tag: str,
) -> dict[str, list[str]]:
    if not candidate_variant_ids:
        return {}

    aliases: dict[str, list[str]] = {}
    part_names = list_parquet_parts(f"{release_output_base(release_tag)}/variant")
    for part_name in part_names:
        part_url, frame = _read_part(
            "variant",
            release_tag,
            part_name,
            columns=["id", "rsIds"],
        )
        if frame.empty:
            continue
        frame = frame[frame["id"].isin(candidate_variant_ids)]
        if frame.empty:
            continue
        for record in frame.to_dict(orient="records"):
            aliases[normalize_whitespace(record.get("id"))] = _flatten_strings(record.get("rsIds"))
    return aliases


def _variant_match_bonus(variant_query: str, variant_id: str, rs_ids: list[str]) -> float:
    if not variant_query:
        return 0.0

    normalized_variant_id = _normalize_variant_id(variant_query)
    if normalized_variant_id and normalized_variant_id == variant_id:
        return 3000.0

    variant_query_lower = variant_query.lower()
    if variant_id.lower() == variant_query_lower:
        return 2950.0
    if any(rs_id.lower() == variant_query_lower for rs_id in rs_ids):
        return 2900.0
    return float("-inf")


def fetch_live_candidates(
    *,
    disease_query: str,
    disease_id: str,
    disease_name: str,
    target_id: str,
    release_tag: str,
    variant_query: str,
    max_study_matches: int,
    max_matches: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    page_size = min(max(max_study_matches, 10), 25)
    page_count = max(1, (max_study_matches + page_size - 1) // page_size)
    for page_index in range(page_count):
        payload = {
            "query": LIVE_L2G_QUERY,
            "variables": {
                "diseaseIds": [disease_id],
                "page": {"index": page_index, "size": page_size},
            },
        }
        response = _post_json(OPEN_TARGETS_GRAPHQL_API, payload)
        if response.get("errors"):
            joined = " | ".join(normalize_whitespace(err.get("message")) for err in response["errors"])
            raise RuntimeError(f"Open Targets GraphQL returned errors: {joined}")
        rows = (((response.get("data") or {}).get("studies") or {}).get("rows") or [])
        if not rows:
            break
        for study in rows:
            trait_from_source = normalize_whitespace(study.get("traitFromSource"))
            study_id = normalize_whitespace(study.get("id"))
            study_score = _study_match_score(
                trait_from_source=trait_from_source,
                disease_query=disease_query,
                disease_name=disease_name,
                disease_id=disease_id,
                disease_ids=[disease_id],
            )
            if study_score == float("-inf"):
                continue
            credible_sets = (((study.get("credibleSets") or {}).get("rows")) or [])
            for credible_set in credible_sets:
                study_locus_id = normalize_whitespace(credible_set.get("studyLocusId"))
                variant = credible_set.get("variant") or {}
                variant_id = normalize_whitespace(variant.get("id"))
                rs_ids = _flatten_strings(variant.get("rsIds"))
                variant_bonus = _variant_match_bonus(variant_query, variant_id, rs_ids)
                if variant_bonus == float("-inf"):
                    continue
                l2g_predictions = (((credible_set.get("l2GPredictions") or {}).get("rows")) or [])
                for prediction in l2g_predictions:
                    target = prediction.get("target") or {}
                    if normalize_whitespace(target.get("id")) != target_id:
                        continue
                    l2g_score = _safe_float(prediction.get("score"))
                    rank = study_score + variant_bonus + l2g_score
                    candidates.append(
                        (
                            rank,
                            {
                                "release": release_tag,
                                "study_id": study_id,
                                "trait_from_source": trait_from_source,
                                "publication_first_author": normalize_whitespace(study.get("publicationFirstAuthor")),
                                "publication_date": normalize_whitespace(study.get("publicationDate")),
                                "study_locus_id": study_locus_id,
                                "variant_id": variant_id,
                                "rs_ids": rs_ids,
                                "score": l2g_score,
                                "score_rounded_3dp": round(l2g_score, 3),
                                "study_source_url": OPEN_TARGETS_GRAPHQL_API,
                                "credible_set_source_url": OPEN_TARGETS_GRAPHQL_API,
                                "l2g_source_url": OPEN_TARGETS_GRAPHQL_API,
                                "match_score": rank,
                            },
                        )
                    )
        if len(rows) < page_size:
            break
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [dict(candidate) for _, candidate in candidates[:max_matches]]


def lookup_exact_archive_l2g(study_locus_id: str, target_id: str, release_tag: str) -> dict[str, Any]:
    part_names = list_parquet_parts(f"{release_output_base(release_tag)}/l2g_prediction")
    for part_name in part_names:
        part_url, frame = _read_part(
            "l2g_prediction",
            release_tag,
            part_name,
            columns=["studyLocusId", "geneId", "score", "shapBaseValue"],
            filters=[("studyLocusId", "==", study_locus_id), ("geneId", "==", target_id)],
        )
        if frame.empty:
            continue
        frame = frame[(frame["studyLocusId"] == study_locus_id) & (frame["geneId"] == target_id)]
        if frame.empty:
            continue
        record = frame.iloc[0].to_dict()
        return {
            "study_locus_id": normalize_whitespace(record.get("studyLocusId")),
            "target_id": normalize_whitespace(record.get("geneId")),
            "score": _safe_float(record.get("score")),
            "shap_base_value": _safe_float(record.get("shapBaseValue")),
            "l2g_source_url": part_url,
        }
    raise LookupError(
        f"No Open Targets archive L2G row found for study-locus {study_locus_id} and target {target_id} in release {release_tag}."
    )


def lookup_exact_archive_credible_set(study_locus_id: str, release_tag: str) -> dict[str, Any]:
    part_names = list_parquet_parts(f"{release_output_base(release_tag)}/credible_set")
    for part_name in part_names:
        part_url, frame = _read_part(
            "credible_set",
            release_tag,
            part_name,
            columns=["studyLocusId", "studyId", "variantId"],
            filters=[("studyLocusId", "==", study_locus_id)],
        )
        if frame.empty:
            continue
        frame = frame[frame["studyLocusId"] == study_locus_id]
        if frame.empty:
            continue
        record = frame.iloc[0].to_dict()
        return {
            "study_locus_id": normalize_whitespace(record.get("studyLocusId")),
            "study_id": normalize_whitespace(record.get("studyId")),
            "variant_id": normalize_whitespace(record.get("variantId")),
            "credible_set_source_url": part_url,
        }
    raise LookupError(f"No Open Targets archive credible_set row found for study-locus {study_locus_id} in release {release_tag}.")


def choose_best_match(
    *,
    disease_query: str,
    target: dict[str, Any],
    studies: list[dict[str, Any]],
    credible_sets: list[dict[str, Any]],
    l2g_rows: list[dict[str, Any]],
    variant_query: str,
    release_tag: str,
    max_matches: int,
) -> dict[str, Any]:
    study_by_id = {study["study_id"]: study for study in studies}
    credible_set_by_study_locus_id = {
        row["study_locus_id"]: row for row in credible_sets if row.get("study_locus_id")
    }
    candidate_variant_ids = {
        row["variant_id"] for row in credible_sets if row.get("variant_id")
    }
    variant_aliases = load_candidate_variant_aliases(candidate_variant_ids, release_tag) if variant_query else {}

    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in l2g_rows:
        credible_set = credible_set_by_study_locus_id.get(row["study_locus_id"])
        if not credible_set:
            continue
        study = study_by_id.get(credible_set["study_id"])
        if not study:
            continue
        rs_ids = variant_aliases.get(credible_set["variant_id"], [])
        variant_bonus = _variant_match_bonus(variant_query, credible_set["variant_id"], rs_ids)
        if variant_bonus == float("-inf"):
            continue
        score = float(study["match_score"]) + variant_bonus + float(row["score"])
        candidates.append(
            (
                score,
                {
                    "release": release_tag,
                    "target_query": target["query"],
                    "target_id": target["target_id"],
                    "target_symbol": target["target_symbol"],
                    "target_name": target["target_name"],
                    "target_resolution_source": target["resolution_source"],
                    "disease_query": disease_query,
                    "study_id": study["study_id"],
                    "trait_from_source": study["trait_from_source"],
                    "publication_first_author": study["publication_first_author"],
                    "publication_date": study["publication_date"],
                    "study_locus_id": row["study_locus_id"],
                    "variant_id": credible_set["variant_id"],
                    "rs_ids": rs_ids,
                    "score": row["score"],
                    "score_rounded_3dp": round(float(row["score"]), 3),
                    "shap_base_value": row["shap_base_value"],
                    "study_source_url": study["study_source_url"],
                    "credible_set_source_url": credible_set["credible_set_source_url"],
                    "l2g_source_url": row["l2g_source_url"],
                    "candidate_matches": [],
                },
            )
        )

    if not candidates:
        raise LookupError(
            f"No Open Targets L2G rows matched target {target['target_id']} and disease '{disease_query}' in release {release_tag}."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = dict(candidates[0][1])
    best["candidate_matches"] = [
        {
            "study_id": candidate["study_id"],
            "trait_from_source": candidate["trait_from_source"],
            "study_locus_id": candidate["study_locus_id"],
            "variant_id": candidate["variant_id"],
            "score": candidate["score"],
            "rs_ids": candidate["rs_ids"],
        }
        for _, candidate in candidates[:max_matches]
    ]
    return best


def build_not_found_result(
    *,
    target_query: str,
    target: dict[str, Any],
    disease_query: str,
    disease: dict[str, Any],
    release_tag: str,
    variant_query: str,
    message: str,
) -> dict[str, Any]:
    return {
        "ok": True,
        "found": False,
        "message": normalize_whitespace(message),
        "release": release_tag,
        "target_query": target_query,
        "target_id": target["target_id"],
        "target_symbol": target["target_symbol"],
        "target_name": target["target_name"],
        "target_resolution_source": target["resolution_source"],
        "disease_query": disease_query,
        "disease_id": disease["disease_id"],
        "disease_name": disease["disease_name"],
        "disease_resolution_source": disease["resolution_source"],
        "candidate_diseases": disease["candidate_matches"],
        "variant_query": variant_query,
        "candidate_matches": [],
        "score": None,
        "score_rounded_3dp": None,
        "study_id": "",
        "trait_from_source": "",
        "study_locus_id": "",
        "variant_id": "",
        "rs_ids": [],
        "study_source_url": "",
        "credible_set_source_url": "",
        "l2g_source_url": "",
        "shap_base_value": None,
    }


def run(payload: dict[str, Any]) -> dict[str, Any]:
    target_query = normalize_whitespace(payload.get("target"))
    disease_query = normalize_whitespace(payload.get("disease"))
    variant_query = normalize_whitespace(payload.get("variant"))
    env_default_release = normalize_whitespace(os.getenv("OPEN_TARGETS_L2G_DEFAULT_RELEASE"))
    raw_release_query = normalize_whitespace(payload.get("release"))
    if env_default_release and raw_release_query.lower() in {"", "latest", "current", "most recent"}:
        release_query = env_default_release
    else:
        release_query = raw_release_query or "latest"
    max_disease_candidates = int(payload.get("max_disease_matches") or 5)
    max_study_candidates = int(payload.get("max_study_matches") or 10)
    max_matches = int(payload.get("max_matches") or 5)

    if not target_query:
        raise ValueError("Missing required 'target' input.")
    if not disease_query:
        raise ValueError("Missing required 'disease' input.")

    release_tag = normalize_release_tag(release_query)
    target = resolve_target(target_query)
    disease = resolve_disease(disease_query, release_tag, max_candidates=max_disease_candidates)
    live_candidates = fetch_live_candidates(
        disease_query=disease_query,
        disease_id=disease["disease_id"],
        disease_name=disease["disease_name"],
        target_id=target["target_id"],
        release_tag=release_tag,
        variant_query=variant_query,
        max_study_matches=max_study_candidates,
        max_matches=max_matches,
    )
    if live_candidates:
        best_live = dict(live_candidates[0])
        try:
            archive_l2g = lookup_exact_archive_l2g(best_live["study_locus_id"], target["target_id"], release_tag)
            archive_credible_set = lookup_exact_archive_credible_set(best_live["study_locus_id"], release_tag)
            best_live.update(
                {
                    "study_id": archive_credible_set["study_id"] or best_live["study_id"],
                    "variant_id": archive_credible_set["variant_id"] or best_live["variant_id"],
                    "score": archive_l2g["score"],
                    "score_rounded_3dp": round(float(archive_l2g["score"]), 3),
                    "shap_base_value": archive_l2g["shap_base_value"],
                    "credible_set_source_url": archive_credible_set["credible_set_source_url"],
                    "l2g_source_url": archive_l2g["l2g_source_url"],
                }
            )
            best_live.update(
                {
                    "ok": True,
                    "target_query": target_query,
                    "target_id": target["target_id"],
                    "target_symbol": target["target_symbol"],
                    "target_name": target["target_name"],
                    "target_resolution_source": target["resolution_source"],
                    "disease_query": disease_query,
                    "disease_id": disease["disease_id"],
                    "disease_name": disease["disease_name"],
                    "disease_resolution_source": disease["resolution_source"],
                    "candidate_diseases": disease["candidate_matches"],
                    "candidate_matches": [],
                    "variant_query": variant_query,
                }
            )
            for candidate in live_candidates[:max_matches]:
                candidate_score = candidate["score"]
                if candidate["study_locus_id"] == best_live["study_locus_id"]:
                    candidate_score = archive_l2g["score"]
                best_live["candidate_matches"].append(
                    {
                        "study_id": candidate["study_id"],
                        "trait_from_source": candidate["trait_from_source"],
                        "study_locus_id": candidate["study_locus_id"],
                        "variant_id": candidate["variant_id"],
                        "score": candidate_score,
                        "rs_ids": candidate["rs_ids"],
                    }
                )
            return best_live
        except LookupError:
            pass
    try:
        studies = load_candidate_studies(
            disease_query,
            disease["disease_id"],
            disease["disease_name"],
            release_tag,
            max_candidates=max_study_candidates,
        )
        l2g_rows = load_target_l2g_rows(target["target_id"], release_tag)
        study_locus_ids = {row["study_locus_id"] for row in l2g_rows}
        credible_sets = load_candidate_credible_sets(study_locus_ids, release_tag)
        best = choose_best_match(
            disease_query=disease_query,
            target=target,
            studies=studies,
            credible_sets=credible_sets,
            l2g_rows=l2g_rows,
            variant_query=variant_query,
            release_tag=release_tag,
            max_matches=max_matches,
        )
    except LookupError as exc:
        return build_not_found_result(
            target_query=target_query,
            target=target,
            disease_query=disease_query,
            disease=disease,
            release_tag=release_tag,
            variant_query=variant_query,
            message=str(exc),
        )
    best.update(
        {
            "ok": True,
            "found": True,
            "disease_id": disease["disease_id"],
            "disease_name": disease["disease_name"],
            "disease_resolution_source": disease["resolution_source"],
            "candidate_diseases": disease["candidate_matches"],
            "variant_query": variant_query,
        }
    )
    return best


def main() -> int:
    try:
        raw_payload = sys.argv[1] if len(sys.argv) > 1 else "{}"
        payload = json.loads(raw_payload)
        result = run(payload)
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        result = {
            "ok": False,
            "error": normalize_whitespace(str(exc)),
        }
    sys.stdout.write(json.dumps(result, sort_keys=True))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
