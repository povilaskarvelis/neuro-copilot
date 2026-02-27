#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="shaquille-oneal-1771992308"
DATASET="hackathon_data"
BUCKET="gs://benchspark-data-1771447466-datasets"

echo "=== Creating BigQuery dataset ==="
bq mk --project_id="${PROJECT_ID}" --location=US "${DATASET}" 2>/dev/null || echo "Dataset already exists"

load_tsv() {
  local table="$1" path="$2"
  echo "Loading ${table} from ${path}..."
  bq load --autodetect --source_format=CSV --field_delimiter='\t' \
    --allow_quoted_newlines --allow_jagged_rows \
    "${PROJECT_ID}:${DATASET}.${table}" "${path}" || echo "FAILED: ${table}"
}

load_csv() {
  local table="$1" path="$2" skip="${3:-0}"
  echo "Loading ${table} from ${path}..."
  bq load --autodetect --source_format=CSV \
    --allow_quoted_newlines --allow_jagged_rows \
    --skip_leading_rows="${skip}" \
    "${PROJECT_ID}:${DATASET}.${table}" "${path}" || echo "FAILED: ${table}"
}

# ── CIViC (TSV) ─────────────────────────────────────────
echo ""
echo "=== CIViC ==="
load_tsv civic_assertion_summaries      "${BUCKET}/civic/nightly-AcceptedAssertionSummaries.tsv"
load_tsv civic_clinical_evidence        "${BUCKET}/civic/nightly-AcceptedClinicalEvidenceSummaries.tsv"
load_tsv civic_features                 "${BUCKET}/civic/nightly-FeatureSummaries.tsv"
load_tsv civic_molecular_profiles       "${BUCKET}/civic/nightly-MolecularProfileSummaries.tsv"
load_tsv civic_variant_groups           "${BUCKET}/civic/nightly-VariantGroupSummaries.tsv"
load_tsv civic_variants                 "${BUCKET}/civic/nightly-VariantSummaries.tsv"

# ── ClinGen (CSV with 6-line header) ────────────────────
echo ""
echo "=== ClinGen ==="
load_csv clingen_gene_disease_validity    "${BUCKET}/clingen/gene-disease-validity.csv" 6
load_csv clingen_variant_pathogenicity    "${BUCKET}/clingen/variant-pathogenicity.csv" 6
load_csv clingen_dosage_sensitivity       "${BUCKET}/clingen/dosage-sensitivity-all.csv" 6
load_csv clingen_dosage_genes             "${BUCKET}/clingen/dosage-sensitivity-genes.csv" 6
load_csv clingen_curation_summary         "${BUCKET}/clingen/curation-activity-summary.csv" 6
load_tsv clingen_actionability_adult      "${BUCKET}/clingen/actionability-adult.tsv"
load_tsv clingen_actionability_pediatric  "${BUCKET}/clingen/actionability-pediatric.tsv"

# ── GTEx (GCT format — skip 2-line header, then tab-delimited) ──
echo ""
echo "=== GTEx ==="
load_tsv gtex_sample_attributes "${BUCKET}/gtex/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
load_tsv gtex_subject_phenotypes "${BUCKET}/gtex/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"

echo ""
echo "=== Done ==="
echo "Loaded tables into ${PROJECT_ID}:${DATASET}"
echo "Verify with: bq ls ${PROJECT_ID}:${DATASET}"
