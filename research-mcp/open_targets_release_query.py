#!/usr/bin/env python3
"""Resolve Open Targets association scores from official archived releases.

This helper is intentionally narrow: it resolves one target-disease pair against
the official EBI Open Targets parquet release archive and prints JSON for the
Node MCP server.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd


MYGENE_API = "https://mygene.info/v3"
OPEN_TARGETS_ARCHIVE_ROOT = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform"
CACHE_ROOT = Path(
    os.getenv(
        "OPEN_TARGETS_RELEASE_CACHE_DIR",
        str(Path.home() / ".cache" / "neuro-copilot" / "open_targets"),
    )
).expanduser()

MONTH_TO_NUMBER = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}


def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def cache_download(url: str, relative_path: str) -> Path:
    destination = CACHE_ROOT / relative_path
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent)) as tmp:
            shutil.copyfileobj(response, tmp)
            temp_path = Path(tmp.name)
    temp_path.replace(destination)
    return destination


def normalize_release_tag(raw_value: str) -> str:
    raw = normalize_whitespace(raw_value).lower()
    if not raw or raw in {"latest", "current", "most recent"}:
        return latest_release_tag()

    direct_match = re.search(r"\b(\d{2})\.(\d{2})\b", raw)
    if direct_match:
        return f"{direct_match.group(1)}.{direct_match.group(2)}"

    year_match = re.search(r"\b(20\d{2})\b", raw)
    if year_match:
        for month_name, month_number in MONTH_TO_NUMBER.items():
            if re.search(rf"\b{re.escape(month_name)}\b", raw):
                return f"{year_match.group(1)[2:]}.{month_number}"

    raise ValueError(
        f"Could not normalize Open Targets release '{raw_value}'. Use YY.MM (for example 25.09) or a month/year label."
    )


def latest_release_tag() -> str:
    html = fetch_text(f"{OPEN_TARGETS_ARCHIVE_ROOT}/")
    tags = re.findall(r'href="((?:\d{2}\.\d{2})/)\"', html)
    normalized = sorted({tag.rstrip("/") for tag in tags})
    if not normalized:
        raise RuntimeError("Unable to discover Open Targets release tags from the official archive.")
    return normalized[-1]


def release_output_base(release_tag: str) -> str:
    return f"{OPEN_TARGETS_ARCHIVE_ROOT}/{release_tag}/output"


def list_parquet_parts(dataset_url: str) -> list[str]:
    html = fetch_text(f"{dataset_url}/")
    return re.findall(r'href="(part-[^"]+\.parquet)"', html)


def normalize_ensembl_gene_id(raw_value: str) -> str:
    raw = normalize_whitespace(raw_value).split(".")[0].upper()
    if re.fullmatch(r"ENSG\d{11}", raw):
        return raw
    return ""


def _extract_gene_id(value: Any) -> str:
    if isinstance(value, str):
        return normalize_ensembl_gene_id(value)
    if isinstance(value, list):
        for item in value:
            gene_id = _extract_gene_id(item)
            if gene_id:
                return gene_id
        return ""
    if isinstance(value, dict):
        return normalize_ensembl_gene_id(value.get("gene"))
    return ""


def resolve_target(query: str) -> dict[str, Any]:
    direct_id = normalize_ensembl_gene_id(query)
    if direct_id:
        return {
            "query": query,
            "target_id": direct_id,
            "target_symbol": "",
            "target_name": "",
            "resolution_source": "direct_input",
        }

    encoded_query = urllib.parse.quote(normalize_whitespace(query))
    url = (
        f"{MYGENE_API}/query?q={encoded_query}"
        "&species=human&size=10&fields=ensembl.gene,symbol,name,alias"
    )
    payload = fetch_json(url)
    hits = payload.get("hits") or []
    normalized_query = normalize_whitespace(query)
    normalized_upper = normalized_query.upper()
    normalized_lower = normalized_query.lower()
    best_hit: dict[str, Any] | None = None
    best_score = float("-inf")

    for hit in hits:
        gene_id = _extract_gene_id(hit.get("ensembl"))
        if not gene_id:
            continue
        symbol = normalize_whitespace(hit.get("symbol"))
        name = normalize_whitespace(hit.get("name"))
        aliases_raw = hit.get("alias") or []
        if isinstance(aliases_raw, str):
            aliases = [normalize_whitespace(aliases_raw)]
        else:
            aliases = [normalize_whitespace(item) for item in aliases_raw if normalize_whitespace(item)]

        score = float(hit.get("_score") or 0.0)
        if symbol.upper() == normalized_upper:
            score += 1000.0
        if any(alias.upper() == normalized_upper for alias in aliases):
            score += 800.0
        if name.lower() == normalized_lower:
            score += 600.0
        if normalized_lower and normalized_lower in name.lower():
            score += 100.0

        if score > best_score:
            best_score = score
            best_hit = {
                "query": query,
                "target_id": gene_id,
                "target_symbol": symbol,
                "target_name": name,
                "resolution_source": url,
            }

    if best_hit is None:
        raise ValueError(f"Could not resolve target '{query}' to an Ensembl gene ID.")
    return best_hit


def _flatten_synonyms(raw_value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(raw_value, dict):
        for nested in raw_value.values():
            values.extend(_flatten_synonyms(nested))
        return values
    if isinstance(raw_value, (list, tuple)):
        for item in raw_value:
            values.extend(_flatten_synonyms(item))
        return values
    if hasattr(raw_value, "tolist"):
        return _flatten_synonyms(raw_value.tolist())
    text = normalize_whitespace(raw_value)
    return [text] if text else []


def resolve_disease(query: str, release_tag: str, max_candidates: int = 5) -> dict[str, Any]:
    base = release_output_base(release_tag)
    disease_url = f"{base}/disease/disease.parquet"
    disease_path = cache_download(disease_url, f"{release_tag}/disease/disease.parquet")
    frame = pd.read_parquet(disease_path, columns=["id", "name", "synonyms"])

    normalized_query = normalize_whitespace(query)
    normalized_lower = normalized_query.lower()
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in frame.itertuples(index=False):
        disease_id = normalize_whitespace(getattr(row, "id", ""))
        disease_name = normalize_whitespace(getattr(row, "name", ""))
        synonyms = _flatten_synonyms(getattr(row, "synonyms", None))
        score = float("-inf")

        if disease_name.lower() == normalized_lower:
            score = 2000.0
        elif any(s.lower() == normalized_lower for s in synonyms):
            score = 1800.0
        elif normalized_lower and normalized_lower in disease_name.lower():
            score = 1200.0 - abs(len(disease_name) - len(normalized_query))
        else:
            matching_synonym = next((s for s in synonyms if normalized_lower and normalized_lower in s.lower()), "")
            if matching_synonym:
                score = 1000.0 - abs(len(matching_synonym) - len(normalized_query))

        if score == float("-inf"):
            continue

        candidates.append(
            (
                score,
                {
                    "disease_id": disease_id,
                    "disease_name": disease_name,
                    "matched_synonym": next((s for s in synonyms if s.lower() == normalized_lower), ""),
                },
            )
        )

    if not candidates:
        raise ValueError(f"Could not resolve disease or trait '{query}' in Open Targets release {release_tag}.")

    candidates.sort(key=lambda item: item[0], reverse=True)
    best = dict(candidates[0][1])
    best["resolution_source"] = disease_url
    best["candidate_matches"] = [
        {
            "disease_id": candidate["disease_id"],
            "disease_name": candidate["disease_name"],
        }
        for _, candidate in candidates[:max_candidates]
    ]
    return best


def _read_association_part(part_path: Path) -> tuple[pd.DataFrame, str, str]:
    column_sets = [
        ("associationScore", "evidenceCount"),
        ("score", "evidenceCount"),
        ("associationScore", "evidence_count"),
        ("score", "evidence_count"),
    ]
    last_error: Exception | None = None
    for score_column, evidence_column in column_sets:
        try:
            frame = pd.read_parquet(
                part_path,
                columns=["diseaseId", "targetId", score_column, evidence_column],
            )
            return frame, score_column, evidence_column
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to read Open Targets association parquet part.")


def lookup_association(target_id: str, disease_id: str, release_tag: str) -> dict[str, Any]:
    base = release_output_base(release_tag)
    dataset_url = f"{base}/association_overall_direct"
    part_names = list_parquet_parts(dataset_url)
    if not part_names:
        raise RuntimeError(f"No association parquet parts found for Open Targets release {release_tag}.")

    for part_name in part_names:
        part_url = f"{dataset_url}/{part_name}"
        part_path = cache_download(
            part_url,
            f"{release_tag}/association_overall_direct/{part_name}",
        )
        frame, score_column, evidence_column = _read_association_part(part_path)
        matches = frame[
            (frame["targetId"] == target_id)
            & (frame["diseaseId"] == disease_id)
        ]
        if matches.empty:
            continue
        record = matches.iloc[0].to_dict()
        return {
            "score": float(record[score_column]),
            "evidence_count": int(record[evidence_column]),
            "association_source_url": part_url,
        }

    raise LookupError(
        f"No Open Targets association_overall_direct row found for target {target_id} and disease {disease_id} in release {release_tag}."
    )


def run(payload: dict[str, Any]) -> dict[str, Any]:
    target_query = normalize_whitespace(payload.get("target"))
    disease_query = normalize_whitespace(payload.get("disease"))
    release_query = normalize_whitespace(payload.get("release"))
    max_candidates = int(payload.get("max_disease_matches") or 5)
    if not target_query:
        raise ValueError("Missing required 'target' input.")
    if not disease_query:
        raise ValueError("Missing required 'disease' input.")

    release_tag = normalize_release_tag(release_query or "latest")
    target = resolve_target(target_query)
    disease = resolve_disease(disease_query, release_tag, max_candidates=max_candidates)
    try:
        association = lookup_association(target["target_id"], disease["disease_id"], release_tag)
    except LookupError as exc:
        return {
            "ok": True,
            "found": False,
            "message": normalize_whitespace(str(exc)),
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
            "score": None,
            "evidence_count": None,
            "association_source_url": "",
        }
    return {
        "ok": True,
        "found": True,
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
        "score": association["score"],
        "evidence_count": association["evidence_count"],
        "association_source_url": association["association_source_url"],
    }


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
