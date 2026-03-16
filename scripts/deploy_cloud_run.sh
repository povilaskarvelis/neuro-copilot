#!/usr/bin/env bash
set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
# PROJECT_ID  – GCP project (pass via env or edit default below)
# GOOGLE_API_KEY – Google AI Studio key (pass via env; stored in Secret Manager)
# ── Optional overrides ───────────────────────────────────────────────────────
# REGION, SERVICE_NAME, REPO_NAME, IMAGE_NAME, USE_VERTEX_AI, GA4_MEASUREMENT_ID
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID="gen-lang-client-0943167408"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-ai-co-scientist}"
REPO_NAME="${REPO_NAME:-co-scientist-images}"
IMAGE_NAME="${IMAGE_NAME:-ai-co-scientist}"
USE_VERTEX_AI="${USE_VERTEX_AI:-}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
GA4_MEASUREMENT_ID="${GA4_MEASUREMENT_ID:-G-NTCXHW3B2G}"

# Source local .env if API key not already in environment
if [[ -z "${GOOGLE_API_KEY}" && -f "adk-agent/.env" ]]; then
  set -a; source adk-agent/.env; set +a
fi

if [[ -z "${PROJECT_ID}" ]]; then
  echo "Error: PROJECT_ID is required."
  echo "Usage:"
  echo "  PROJECT_ID=my-project GOOGLE_API_KEY=AIza... bash scripts/deploy_cloud_run.sh"
  echo ""
  echo "Options:"
  echo "  USE_VERTEX_AI=true   – use Vertex AI instead of AI Studio API key"
  echo "  REGION=us-central1   – GCP region (default: us-central1)"
  exit 1
fi

# Default to the API-key backend when one is already configured. This keeps
# Cloud deploys aligned with local behavior instead of silently switching to
# Vertex AI quotas unless the caller explicitly opts in.
if [[ -z "${USE_VERTEX_AI}" ]]; then
  if [[ -n "${GOOGLE_API_KEY}" ]]; then
    USE_VERTEX_AI="false"
  else
    USE_VERTEX_AI="true"
  fi
fi

if [[ "${USE_VERTEX_AI}" != "true" && -z "${GOOGLE_API_KEY}" ]]; then
  echo "Error: GOOGLE_API_KEY is required when USE_VERTEX_AI is not true."
  echo "Either set GOOGLE_API_KEY or set USE_VERTEX_AI=true to use Vertex AI."
  exit 1
fi

echo "Using project=${PROJECT_ID} region=${REGION} service=${SERVICE_NAME}"
echo "LLM backend: $([ "${USE_VERTEX_AI}" = "true" ] && echo "Vertex AI" || echo "AI Studio API key")"

# ── Artifact Registry ────────────────────────────────────────────────────────

if ! gcloud artifacts repositories describe "${REPO_NAME}" \
  --location "${REGION}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location "${REGION}" \
    --description="Container images for AI Co-Scientist" \
    --project "${PROJECT_ID}"
fi

# ── Build ────────────────────────────────────────────────────────────────────

IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE_URI}" \
  --suppress-logs \
  . || true

# ── Store API key in Secret Manager (if using AI Studio) ─────────────────────

SECRET_NAME="ai-co-scientist-api-key"
if [[ "${USE_VERTEX_AI}" != "true" ]]; then
  if ! gcloud secrets describe "${SECRET_NAME}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1; then
    echo "${GOOGLE_API_KEY}" | gcloud secrets create "${SECRET_NAME}" \
      --project "${PROJECT_ID}" \
      --data-file=-
  else
    echo "${GOOGLE_API_KEY}" | gcloud secrets versions add "${SECRET_NAME}" \
      --project "${PROJECT_ID}" \
      --data-file=-
  fi

  PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
  COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

  gcloud secrets add-iam-policy-binding "${SECRET_NAME}" \
    --project "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null 2>&1 || true
fi

# ── Deploy ───────────────────────────────────────────────────────────────────

BQ_ALLOWLIST="bigquery-public-data.open_targets_platform,bigquery-public-data.ebi_chembl,bigquery-public-data.gnomAD,bigquery-public-data.fda_drug,bigquery-public-data.human_variant_annotation,bigquery-public-data.human_genome_variants,bigquery-public-data.immune_epitope_db,bigquery-public-data.umiami_lincs,bigquery-public-data.nlm_rxnorm,bigquery-public-data.ebi_surechembl"

DEPLOY_FLAGS=(
  --project "${PROJECT_ID}"
  --region "${REGION}"
  --image "${IMAGE_URI}"
  --platform managed
  --allow-unauthenticated
  --port 8080
  --cpu 4
  --memory 4Gi
  --min-instances 1
  --max-instances 3
  --concurrency 160
  --no-cpu-throttling
  --timeout 900
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}"
  --set-env-vars "GOOGLE_CLOUD_LOCATION=${REGION}"
  --set-env-vars "BQ_PROJECT_ID=${PROJECT_ID}"
  --set-env-vars "BQ_LOCATION=US"
  --set-env-vars "ADK_NATIVE_PREFER_BIGQUERY=1"
  --set-env-vars "^||^BQ_DATASET_ALLOWLIST=${BQ_ALLOWLIST}"
)

if [[ -n "${GA4_MEASUREMENT_ID}" ]]; then
  DEPLOY_FLAGS+=(--set-env-vars "GA4_MEASUREMENT_ID=${GA4_MEASUREMENT_ID}")
fi

if [[ "${USE_VERTEX_AI}" == "true" ]]; then
  DEPLOY_FLAGS+=(--set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=true")
else
  DEPLOY_FLAGS+=(
    --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=false"
    --set-secrets "GOOGLE_API_KEY=${SECRET_NAME}:latest"
  )
fi

gcloud run deploy "${SERVICE_NAME}" "${DEPLOY_FLAGS[@]}"

# ── Output ───────────────────────────────────────────────────────────────────

SERVICE_URL="$(
  gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format='value(status.url)'
)"

echo ""
echo "Deployment complete."
echo "Service URL: ${SERVICE_URL}"
echo "Health check: ${SERVICE_URL}/healthz"
echo "Query endpoint: ${SERVICE_URL}/query"
