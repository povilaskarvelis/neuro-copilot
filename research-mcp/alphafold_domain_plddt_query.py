#!/usr/bin/env python3
"""Compute AlphaFold domain-level pLDDT means from versioned models.

This helper resolves a UniProt accession to AlphaFold predictions, supports
historical archive-backed model extraction for versioned releases such as v4,
and combines those models with UniProt feature annotations to compute
domain-level pLDDT summaries.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from typing import Any


ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api"
ALPHAFOLD_FTP_ROOT = "https://ftp.ebi.ac.uk/pub/databases/alphafold"
UNIPROT_API = "https://rest.uniprot.org"
CACHE_ROOT = Path(
    os.getenv(
        "ALPHAFOLD_DOMAIN_CACHE_DIR",
        str(Path.home() / ".cache" / "neuro-copilot" / "alphafold"),
    )
).expanduser()
ZERO_BLOCK = b"\0" * 512


def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8", "ignore")


def fetch_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def cache_write_text(relative_path: str, text: str) -> Path:
    destination = CACHE_ROOT / relative_path
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(destination.parent), mode="w", encoding="utf-8") as tmp:
        tmp.write(text)
        temp_path = Path(tmp.name)
    temp_path.replace(destination)
    return destination


def cache_read_text(relative_path: str) -> str | None:
    path = CACHE_ROOT / relative_path
    if path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8")
    return None


def normalize_version(raw_value: str, latest_version: int) -> int:
    raw = normalize_whitespace(raw_value).lower()
    if not raw or raw in {"latest", "current", "most recent"}:
        return latest_version
    match = re.search(r"v?(\d+)", raw)
    if not match:
        raise ValueError(f"Could not normalize AlphaFold version from '{raw_value}'.")
    version = int(match.group(1))
    if version <= 0:
        raise ValueError(f"AlphaFold version must be positive, got '{raw_value}'.")
    return version


def fetch_uniprot_entry(uniprot_id: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(uniprot_id)
    return fetch_json(f"{UNIPROT_API}/uniprotkb/{encoded}.json")


def resolve_proteome_id(uniprot_entry: dict[str, Any]) -> str:
    for row in uniprot_entry.get("uniProtKBCrossReferences") or []:
        if normalize_whitespace(row.get("database")) != "Proteomes":
            continue
        proteome_id = normalize_whitespace(row.get("id"))
        if proteome_id:
            return proteome_id
    return ""


def fetch_download_metadata() -> list[dict[str, Any]]:
    cache_name = "download_metadata.json"
    cached = cache_read_text(cache_name)
    if cached:
        return json.loads(cached)
    data = fetch_json(f"{ALPHAFOLD_FTP_ROOT}/download_metadata.json")
    cache_write_text(cache_name, json.dumps(data))
    return data


def resolve_archive_name(proteome_id: str, version: int) -> str:
    wanted_suffix = f"_v{version}.tar"
    candidate_name = ""
    for row in fetch_download_metadata():
        if normalize_whitespace(row.get("reference_proteome")) != proteome_id:
            continue
        archive_name = normalize_whitespace(row.get("archive_name"))
        if archive_name.endswith(wanted_suffix):
            return archive_name
        if archive_name:
            candidate_name = re.sub(r"_v\d+\.tar$", wanted_suffix, archive_name)
            if candidate_name != archive_name:
                return candidate_name
    raise LookupError(
        f"Could not find AlphaFold archive metadata for proteome {proteome_id} version v{version}."
    )


def stream_tar_member_bytes(url: str, member_name: str) -> bytes:
    request = urllib.request.Request(url)
    with urllib.request.urlopen(request, timeout=180) as response:
        while True:
            header = response.read(512)
            if not header or len(header) < 512:
                break
            if header == ZERO_BLOCK:
                break
            name = header[:100].split(b"\0", 1)[0].decode("utf-8", "ignore").strip()
            size_field = header[124:136].split(b"\0", 1)[0].strip() or b"0"
            size = int(size_field, 8)
            padded_size = ((size + 511) // 512) * 512
            if name == member_name:
                data = response.read(size)
                padding = padded_size - size
                if padding:
                    response.read(padding)
                return data
            remaining = padded_size
            while remaining > 0:
                chunk = response.read(min(1 << 20, remaining))
                if not chunk:
                    raise RuntimeError(f"Unexpected EOF while scanning AlphaFold archive for {member_name}.")
                remaining -= len(chunk)
    raise LookupError(f"AlphaFold archive member {member_name} was not found in {url}.")


def cache_alphafold_archive_member_text(uniprot_id: str, version: int, member_name: str) -> str:
    relative = f"v{version}/{member_name.removesuffix('.gz')}"
    cached = cache_read_text(relative)
    if cached is not None:
        return cached

    uniprot_entry = fetch_uniprot_entry(uniprot_id)
    proteome_id = resolve_proteome_id(uniprot_entry)
    if not proteome_id:
        raise LookupError(f"Could not resolve a UniProt proteome reference for {uniprot_id}.")
    archive_name = resolve_archive_name(proteome_id, version)
    archive_url = f"{ALPHAFOLD_FTP_ROOT}/v{version}/{archive_name}"
    compressed = stream_tar_member_bytes(archive_url, member_name)
    text = gzip.decompress(compressed).decode("utf-8", "ignore")
    cache_write_text(relative, text)
    return text


def fetch_latest_prediction_entry(uniprot_id: str) -> dict[str, Any]:
    payload = fetch_json(f"{ALPHAFOLD_API}/prediction/{urllib.parse.quote(uniprot_id)}")
    entries = payload if isinstance(payload, list) else [payload]
    entry = entries[0] if entries else {}
    if not entry or not entry.get("entryId"):
        raise LookupError(f"No AlphaFold prediction was found for {uniprot_id}.")
    return entry


def fetch_alphafold_pdb_text(uniprot_id: str, version: int) -> tuple[str, str, int]:
    latest_entry = fetch_latest_prediction_entry(uniprot_id)
    latest_version = int(latest_entry.get("latestVersion") or latest_entry.get("allVersions", [version])[-1] or version)
    entry_id = normalize_whitespace(latest_entry.get("entryId") or f"AF-{uniprot_id.upper()}-F1")
    if version == latest_version:
        pdb_url = normalize_whitespace(latest_entry.get("pdbUrl"))
        if not pdb_url:
            raise LookupError(f"AlphaFold did not return a PDB URL for {entry_id}.")
        pdb_text = fetch_text(pdb_url)
        return pdb_text, pdb_url, latest_version

    member_name = f"{entry_id}-model_v{version}.pdb.gz"
    pdb_text = cache_alphafold_archive_member_text(uniprot_id, version, member_name)
    archive_name = resolve_archive_name(resolve_proteome_id(fetch_uniprot_entry(uniprot_id)), version)
    archive_url = f"{ALPHAFOLD_FTP_ROOT}/v{version}/{archive_name}"
    return pdb_text, archive_url, latest_version


def parse_pdb_plddt(pdb_text: str) -> tuple[dict[int, list[float]], dict[int, float]]:
    atom_scores: dict[int, list[float]] = {}
    ca_scores: dict[int, float] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            residue_number = int(line[22:26].strip())
            score = float(line[60:66].strip())
        except ValueError:
            continue
        atom_scores.setdefault(residue_number, []).append(score)
        if line[12:16].strip() == "CA":
          ca_scores[residue_number] = score
    if not atom_scores:
        raise RuntimeError("The AlphaFold PDB did not contain any ATOM rows with pLDDT values.")
    return atom_scores, ca_scores


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return math.nan
    return sum(values) / len(values)


def quantize(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return round(value, 1)


def feature_range(feature: dict[str, Any]) -> tuple[int, int] | None:
    location = feature.get("location") or {}
    start = location.get("start", {}).get("value")
    end = location.get("end", {}).get("value")
    try:
        start_i = int(start)
        end_i = int(end)
    except (TypeError, ValueError):
        return None
    if start_i <= 0 or end_i < start_i:
        return None
    return start_i, end_i


def clean_label(value: str) -> str:
    label = normalize_whitespace(value)
    if not label:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def build_domain_rows(uniprot_entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    has_signal_peptide = False
    transmembrane_rows: list[dict[str, Any]] = []
    chain_rows: list[dict[str, Any]] = []

    for feature in uniprot_entry.get("features") or []:
        feature_type = normalize_whitespace(feature.get("type"))
        description = normalize_whitespace(feature.get("description"))
        residue_range = feature_range(feature)
        if residue_range is None:
            continue
        start, end = residue_range
        lowered_type = feature_type.lower()
        lowered_desc = description.lower()

        if lowered_type in {"signal", "signal peptide"}:
            has_signal_peptide = True
            rows.append(
                {
                    "label": "signal peptide",
                    "feature_type": feature_type,
                    "description": description or "Signal peptide",
                    "start": start,
                    "end": end,
                    "aliases": ["signal peptide", clean_label(description)],
                    "derived": False,
                }
            )
            continue

        if lowered_type == "topological domain":
            label = clean_label(description) or "topological domain"
            aliases = [label]
            if "cytoplasmic" in lowered_desc:
                aliases.extend(["cytoplasmic", "cytoplasm"])
            if "extracellular" in lowered_desc:
                aliases.extend(["extracellular", "extracellular domain"])
            if "luminal" in lowered_desc:
                aliases.extend(["luminal", "lumen"])
            rows.append(
                {
                    "label": label,
                    "feature_type": feature_type,
                    "description": description or "Topological domain",
                    "start": start,
                    "end": end,
                    "aliases": aliases,
                    "derived": False,
                }
            )
            continue

        if lowered_type == "transmembrane":
            aliases = ["transmembrane", "transmembrane domain", clean_label(description)]
            row = {
                "label": "transmembrane",
                "feature_type": feature_type,
                "description": description or "Transmembrane region",
                "start": start,
                "end": end,
                "aliases": aliases,
                "derived": False,
            }
            rows.append(row)
            transmembrane_rows.append(row)
            continue

        if lowered_type == "chain":
            chain_rows.append(
                {
                    "description": description,
                    "start": start,
                    "end": end,
                }
            )

    if not has_signal_peptide:
        for row in transmembrane_rows:
            if "signal-anchor" not in row["description"].lower():
                continue
            rows.append(
                {
                    "label": "signal anchor region",
                    "feature_type": "Derived",
                    "description": f"Inferred N-terminal signal-anchor region ending at residue {row['end']} from UniProt transmembrane annotation.",
                    "start": 1,
                    "end": row["end"],
                    "aliases": ["signal anchor", "signal anchor region", "signal peptide"],
                    "derived": True,
                }
            )
            break

    for chain in chain_rows:
        if chain["start"] <= 1:
            continue
        lowered_desc = chain["description"].lower()
        if "serum form" not in lowered_desc and "soluble" not in lowered_desc and "mature" not in lowered_desc:
            continue
        rows.append(
            {
                "label": "cleaved n terminal segment",
                "feature_type": "Derived",
                "description": f"Inferred cleaved N-terminal segment before the alternate chain starting at residue {chain['start']}.",
                "start": 1,
                "end": chain["start"] - 1,
                "aliases": ["cleaved n terminal segment", "n terminal segment", "signal peptide"],
                "derived": True,
            }
        )
        break

    deduped: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        key = (row["label"], row["start"], row["end"])
        if key in seen:
            continue
        seen.add(key)
        row["aliases"] = [alias for alias in {clean_label(alias) for alias in row["aliases"]} if alias]
        deduped.append(row)
    return deduped


def annotate_domain_means(domain_rows: list[dict[str, Any]], atom_scores: dict[int, list[float]], ca_scores: dict[int, float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in domain_rows:
        start = row["start"]
        end = row["end"]
        residue_values = [ca_scores[r] for r in range(start, end + 1) if r in ca_scores]
        atom_values: list[float] = []
        for residue in range(start, end + 1):
            atom_values.extend(atom_scores.get(residue, []))
        out.append(
            {
                **row,
                "mean_plddt": quantize(mean_or_nan(residue_values)),
                "per_residue_mean_plddt": quantize(mean_or_nan(residue_values)),
                "all_atom_mean_plddt": quantize(mean_or_nan(atom_values)),
                "n_residues": len(residue_values),
            }
        )
    return out


def pick_requested_domains(domain_rows: list[dict[str, Any]], requested_domains: list[str]) -> list[dict[str, Any]]:
    if not requested_domains:
        return domain_rows
    picked: list[dict[str, Any]] = []
    used = set()
    normalized_requests = [clean_label(value) for value in requested_domains if clean_label(value)]
    for requested in normalized_requests:
        for row in domain_rows:
            if requested not in row.get("aliases", []):
                continue
            key = (row["label"], row["start"], row["end"])
            if key in used:
                continue
            used.add(key)
            picked.append(row)
            break
    return picked


def main(argv: list[str]) -> int:
    payload = json.loads(argv[1]) if len(argv) > 1 else {}
    uniprot_id = normalize_whitespace(payload.get("uniprotId") or payload.get("uniprot_id")).upper()
    requested_version = payload.get("version") or "latest"
    requested_domains = payload.get("domains") or []
    if not uniprot_id:
        print(json.dumps({"status": "error", "error": "Provide a UniProt accession."}))
        return 0

    try:
        latest_entry = fetch_latest_prediction_entry(uniprot_id)
        latest_version = int(latest_entry.get("latestVersion") or latest_entry.get("allVersions", [1])[-1] or 1)
        version = normalize_version(str(requested_version), latest_version)
        uniprot_entry = fetch_uniprot_entry(uniprot_id)
        pdb_text, model_source, resolved_latest_version = fetch_alphafold_pdb_text(uniprot_id, version)
        atom_scores, ca_scores = parse_pdb_plddt(pdb_text)
        domain_rows = annotate_domain_means(build_domain_rows(uniprot_entry), atom_scores, ca_scores)
        requested_rows = pick_requested_domains(domain_rows, requested_domains if isinstance(requested_domains, list) else [])
        result = {
            "status": "ok",
            "uniprot_id": uniprot_id,
            "entry_id": normalize_whitespace(latest_entry.get("entryId") or f"AF-{uniprot_id}-F1"),
            "version": version,
            "latest_version": resolved_latest_version,
            "model_source": model_source,
            "global_plddt": round(float(latest_entry.get("globalMetricValue") or 0.0), 1) if latest_entry.get("globalMetricValue") is not None else None,
            "domain_means": domain_rows,
            "requested_domain_means": requested_rows,
            "notes": [
                "Domain boundaries come from UniProt feature annotations plus explicitly labeled derived regions when a type II signal-anchor annotation implies an N-terminal signal-anchor region or a cleaved serum-form chain implies an N-terminal cleaved segment.",
                "mean_plddt and per_residue_mean_plddt are calculated from one pLDDT value per residue (the CA atom); all_atom_mean_plddt is also provided because AlphaFold stores the same pLDDT value on all atoms in the PDB B-factor field.",
            ],
        }
        print(json.dumps(result))
    except Exception as exc:  # pragma: no cover - surfaced to Node
        print(json.dumps({"status": "error", "error": normalize_whitespace(str(exc))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
