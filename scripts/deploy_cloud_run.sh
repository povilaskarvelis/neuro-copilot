#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="shaquille-oneal-1771992308"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-ai-co-scientist}"
REPO_NAME="${REPO_NAME:-hackathon-images}"
IMAGE_NAME="${IMAGE_NAME:-ai-co-scientist}"

if [[ -z "${PROJECT_ID}" ]]; then
  echo "PROJECT_ID is required. Example:"
  echo "  PROJECT_ID=my-project-id REGION=us-central1 bash scripts/deploy_cloud_run.sh"
  exit 1
fi

echo "Using project=${PROJECT_ID} region=${REGION} service=${SERVICE_NAME}"

if ! gcloud artifacts repositories describe "${REPO_NAME}" \
  --location "${REGION}" \
  --project "${PROJECT_ID}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location "${REGION}" \
    --description="Container images for AI Co-Scientist" \
    --project "${PROJECT_ID}"
fi

IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

gcloud builds submit \
  --project "${PROJECT_ID}" \
  --tag "${IMAGE_URI}" \
  --suppress-logs \
  . || true

gcloud run deploy "${SERVICE_NAME}" \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --image "${IMAGE_URI}" \
  --platform managed \
  --no-allow-unauthenticated \
  --port 8080 \
  --cpu 4 \
  --memory 4Gi \
  --min-instances 1 \
  --timeout 900 \
  --set-env-vars "GOOGLE_GENAI_USE_VERTEXAI=true" \
  --set-env-vars "GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
  --set-env-vars "GOOGLE_CLOUD_LOCATION=${REGION}" \
  --set-env-vars "BQ_PROJECT_ID=${PROJECT_ID}" \
  --set-env-vars "BQ_LOCATION=US" \
  --set-env-vars "ADK_NATIVE_PREFER_BIGQUERY=1" \
  --set-env-vars "^||^BQ_DATASET_ALLOWLIST=${PROJECT_ID}.hackathon_data,bigquery-public-data.open_targets_platform,bigquery-public-data.ebi_chembl,bigquery-public-data.gnomAD,bigquery-public-data.deepmind_alphafold,bigquery-public-data.fda_drug,bigquery-public-data.human_variant_annotation,bigquery-public-data.human_genome_variants,bigquery-public-data.immune_epitope_db,bigquery-public-data.umiami_lincs,bigquery-public-data.nlm_rxnorm,bigquery-public-data.ebi_surechembl"

SERVICE_URL="$(
  gcloud run services describe "${SERVICE_NAME}" \
    --project "${PROJECT_ID}" \
    --region "${REGION}" \
    --format='value(status.url)'
)"

echo "Deployment complete."
echo "Service URL: ${SERVICE_URL}"
echo "Health check: ${SERVICE_URL}/healthz"
echo "Query endpoint: ${SERVICE_URL}/query"
