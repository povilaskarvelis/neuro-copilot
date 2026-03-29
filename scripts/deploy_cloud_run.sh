#!/usr/bin/env bash
set -euo pipefail

# ── Required ─────────────────────────────────────────────────────────────────
# PROJECT_ID  – GCP project (pass via env or edit default below)
# GOOGLE_API_KEY – Google AI Studio key (required only when USE_VERTEX_AI=false; stored in Secret Manager)
# ── Optional overrides ───────────────────────────────────────────────────────
# REGION, SERVICE_NAME, REPO_NAME, IMAGE_NAME, USE_VERTEX_AI, GA4_MEASUREMENT_ID, CONCURRENCY
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ID="gen-lang-client-0943167408"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-neuro-copilot}"
REPO_NAME="${REPO_NAME:-neuro-copilot-images}"
IMAGE_NAME="${IMAGE_NAME:-neuro-copilot}"
USE_VERTEX_AI="${USE_VERTEX_AI:-}"
GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"
BIOGRID_ACCESS_KEY="${BIOGRID_ACCESS_KEY:-}"
BIOGRID_ORCS_ACCESS_KEY="${BIOGRID_ORCS_ACCESS_KEY:-}"
GA4_MEASUREMENT_ID="${GA4_MEASUREMENT_ID:-G-NTCXHW3B2G}"
CONCURRENCY="${CONCURRENCY:-16}"

ENV_FILE="adk-agent/.env"

load_env_var_from_file() {
  local var_name="$1"
  local env_file="$2"
  local line=""
  local value=""

  if [[ -n "${!var_name:-}" || ! -f "${env_file}" ]]; then
    return 0
  fi

  line="$(grep -E "^${var_name}=" "${env_file}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    return 0
  fi

  value="${line#*=}"
  if [[ ${#value} -ge 2 ]]; then
    if [[ "${value:0:1}" == '"' && "${value: -1}" == '"' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value:0:1}" == "'" && "${value: -1}" == "'" ]]; then
      value="${value:1:${#value}-2}"
    fi
  fi

  export "${var_name}=${value}"
}

load_env_var_from_file "GOOGLE_API_KEY" "${ENV_FILE}"
load_env_var_from_file "BIOGRID_ACCESS_KEY" "${ENV_FILE}"
load_env_var_from_file "BIOGRID_ORCS_ACCESS_KEY" "${ENV_FILE}"

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

# Default to Vertex AI for Cloud Run deploys. Vertex AI uses project-level
# quotas that are separate from the AI Studio API key, preventing deployed
# services from competing with local development for TPM quota.
# Pass USE_VERTEX_AI=false to explicitly opt into AI Studio API key backend.
if [[ -z "${USE_VERTEX_AI}" ]]; then
  USE_VERTEX_AI="true"
fi

if [[ "${USE_VERTEX_AI}" != "true" && -z "${GOOGLE_API_KEY}" ]]; then
  echo "Error: GOOGLE_API_KEY is required when USE_VERTEX_AI is not true."
  echo "Either set GOOGLE_API_KEY or set USE_VERTEX_AI=true to use Vertex AI."
  exit 1
fi

echo "Using project=${PROJECT_ID} region=${REGION} service=${SERVICE_NAME}"
echo "LLM backend: $([ "${USE_VERTEX_AI}" = "true" ] && echo "Vertex AI" || echo "AI Studio API key")"
echo "Cloud Run concurrency: ${CONCURRENCY}"

# ── Artifact Registry ────────────────────────────────────────────────────────

if ! gcloud artifacts repositories describe "${REPO_NAME}" \
  --location "${REGION}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location "${REGION}" \
    --description="Container images for Neuro Copilot" \
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

# ── Store secrets in Secret Manager ───────────────────────────────────────────

GOOGLE_SECRET_NAME="neuro-copilot-api-key"
BIOGRID_SECRET_NAME="neuro-copilot-biogrid-access-key"
BIOGRID_ORCS_SECRET_NAME="neuro-copilot-biogrid-orcs-access-key"
PROJECT_NUMBER=""
COMPUTE_SA=""

ensure_compute_service_account() {
  if [[ -n "${COMPUTE_SA}" ]]; then
    return 0
  fi
  PROJECT_NUMBER="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')"
  COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
}

upsert_secret_value() {
  local secret_name="$1"
  local secret_value="$2"

  if [[ -z "${secret_value}" ]]; then
    return 0
  fi

  if ! gcloud secrets describe "${secret_name}" \
    --project "${PROJECT_ID}" >/dev/null 2>&1; then
    printf '%s' "${secret_value}" | gcloud secrets create "${secret_name}" \
      --project "${PROJECT_ID}" \
      --data-file=-
  else
    printf '%s' "${secret_value}" | gcloud secrets versions add "${secret_name}" \
      --project "${PROJECT_ID}" \
      --data-file=-
  fi

  ensure_compute_service_account
  gcloud secrets add-iam-policy-binding "${secret_name}" \
    --project "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null 2>&1 || true
}

if [[ "${USE_VERTEX_AI}" != "true" ]]; then
  upsert_secret_value "${GOOGLE_SECRET_NAME}" "${GOOGLE_API_KEY}"
fi
upsert_secret_value "${BIOGRID_SECRET_NAME}" "${BIOGRID_ACCESS_KEY}"
upsert_secret_value "${BIOGRID_ORCS_SECRET_NAME}" "${BIOGRID_ORCS_ACCESS_KEY}"

# ── Deploy ───────────────────────────────────────────────────────────────────

BQ_ALLOWLIST="bigquery-public-data.open_targets_platform,bigquery-public-data.ebi_chembl,bigquery-public-data.gnomAD,bigquery-public-data.fda_drug,bigquery-public-data.human_variant_annotation,bigquery-public-data.human_genome_variants,bigquery-public-data.umiami_lincs,bigquery-public-data.nlm_rxnorm,bigquery-public-data.ebi_surechembl"

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
  --concurrency "${CONCURRENCY}"
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
    --set-secrets "GOOGLE_API_KEY=${GOOGLE_SECRET_NAME}:latest"
  )
fi

if [[ -n "${BIOGRID_ACCESS_KEY}" ]]; then
  DEPLOY_FLAGS+=(--set-secrets "BIOGRID_ACCESS_KEY=${BIOGRID_SECRET_NAME}:latest")
fi

if [[ -n "${BIOGRID_ORCS_ACCESS_KEY}" ]]; then
  DEPLOY_FLAGS+=(--set-secrets "BIOGRID_ORCS_ACCESS_KEY=${BIOGRID_ORCS_SECRET_NAME}:latest")
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
echo "Web UI: ${SERVICE_URL}/"
echo "Health endpoint: ${SERVICE_URL}/api/health"
echo "Query endpoint: ${SERVICE_URL}/api/query"
