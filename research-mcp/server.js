import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { BigQuery } from "@google-cloud/bigquery";
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const server = new McpServer({
  name: "research-mcp",
  version: "0.1.0",
});

const STRUCTURED_CONTENT_ENVELOPE_VERSION = "research_mcp_response_v1";
const VALID_RESULT_STATUSES = new Set(["ok", "error", "not_found_or_empty", "degraded"]);

const UNIPROT_API = "https://rest.uniprot.org";
const CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2";
const OPENALEX_API = "https://api.openalex.org";
const REACTOME_API = "https://reactome.org/ContentService";
const STRING_API = "https://string-db.org/api";
const ENSEMBL_REST_API = "https://rest.ensembl.org";
const CIVIC_GRAPHQL_API = "https://civicdb.org/api/graphql";
const MYVARIANT_API = "https://myvariant.info/v1";
const ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api";
const GWAS_CATALOG_API = "https://www.ebi.ac.uk/gwas/rest/api/v2";
const DGIDB_GRAPHQL_API = "https://dgidb.org/api/graphql";
const GTEX_API = "https://gtexportal.org/api/v2";
const RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query";
const RCSB_DATA_API = "https://data.rcsb.org/rest/v1";
const CBIOPORTAL_API = "https://www.cbioportal.org/api";
const PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug";
const CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data";
const ABA_API = "https://api.brain-map.org/api/v2";
const EBRAINS_KG_SEARCH_API = "https://search.kg.ebrains.eu/api";
const HF_DATASETS_SERVER_API = "https://datasets-server.huggingface.co";
const GITHUB_API = "https://api.github.com";
const CONP_GITHUB_ORG = "conpdatasets";
const NEMAR_GITHUB_ORG = "nemarDatasets";
const BRAINCODE_CONP_QUERY = "braincode";
const NEUROBAGEL_API = "https://api.neurobagel.org";
const OPENNEURO_GRAPHQL = "https://openneuro.org/crn/graphql";
const DANDI_API = "https://api.dandiarchive.org/api";
const ENIGMA_TOOLBOX_REPO = "MICA-MNI/ENIGMA";
const ENIGMA_SUMMARY_STATS_PATH = "enigmatoolbox/datasets/summary_statistics";
const OPENALEX_WORKS_SELECT = [
  "id",
  "display_name",
  "publication_year",
  "publication_date",
  "cited_by_count",
  "doi",
  "ids",
  "primary_location",
  "authorships",
  "type",
].join(",");
const OPENALEX_AUTHORS_SELECT = [
  "id",
  "display_name",
  "works_count",
  "cited_by_count",
  "last_known_institution",
  "orcid",
].join(",");
const HF_DATASET_PUBMEDQA = String(process.env.HF_DATASET_PUBMEDQA || "qiaojin/PubMedQA").trim();
const HF_DATASET_BIOASQ = String(process.env.HF_DATASET_BIOASQ || "kroshan/BioASQ").trim();
const HF_DATASET_GPQA = String(process.env.HF_DATASET_GPQA || "Idavidrein/gpqa").trim();
const HF_TOKEN = String(process.env.HF_TOKEN || process.env.HUGGINGFACE_TOKEN || "").trim();
const OPENALEX_MAILTO = process.env.OPENALEX_MAILTO || process.env.CONTACT_EMAIL || "";
const BQ_PROJECT_ID = String(process.env.BQ_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT || "").trim();
const BQ_LOCATION = String(process.env.BQ_LOCATION || process.env.GOOGLE_CLOUD_LOCATION || "US").trim() || "US";
const BQ_DATASET_ALLOWLIST = String(process.env.BQ_DATASET_ALLOWLIST || "").trim();
const BQ_DEFAULT_MAX_ROWS = 200;
const BQ_HARD_MAX_ROWS = 1000;
const BQ_DEFAULT_MAX_BYTES_BILLED = 5_000_000_000;
const BQ_QUERY_TIMEOUT_MS = 30_000;
const FORBIDDEN_SQL_KEYWORDS = /\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|call|export|load|copy)\b/i;
let bigQueryClient = null;

function normalizeWhitespace(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function normalizeTextPart(part) {
  if (typeof part === "string") {
    const normalized = part.trim();
    return normalized ? { type: "text", text: normalized } : null;
  }
  if (!part || typeof part !== "object") return null;
  const text = typeof part.text === "string" ? part.text.trim() : "";
  if (!text) return null;
  return {
    type: "text",
    text,
  };
}

function inferEnvelopeStatus(payload, combinedText) {
  const text = normalizeWhitespace(combinedText).toLowerCase();
  if (payload?.isError === true || text.startsWith("error in ") || text.startsWith("error:")) {
    return "error";
  }
  if (text.includes("critical gap") || text.includes("service unavailable")) {
    return "degraded";
  }
  const notFoundMarkers = [
    "no results",
    "no records",
    "no data found",
    "not found",
    "no target data found",
    "no clinical trials found",
    "no diseases found",
    "no targets found",
  ];
  if (notFoundMarkers.some((marker) => text.includes(marker))) {
    return "not_found_or_empty";
  }
  return "ok";
}

function extractSummaryLine(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  return lines[0] || "No summary available.";
}

function normalizeResultStatus(value, fallback = "ok") {
  const normalized = String(value || "")
    .trim()
    .toLowerCase();
  if (VALID_RESULT_STATUSES.has(normalized)) {
    return normalized;
  }
  return fallback;
}

function buildGenericToolPayload(toolName, { status, summary, combinedText, contentPartCount, rawPayload }) {
  const normalizedStatus = normalizeResultStatus(status, "ok");
  const excerpt = String(combinedText || "").slice(0, 900).trim();
  const payload = {
    schema: `${toolName}.generic.v1`,
    result_status: normalizedStatus,
    tool_name: toolName,
    summary: String(summary || "No summary available."),
    content_part_count: Math.max(0, Math.trunc(Number(contentPartCount || 0))),
    text_excerpt: excerpt || String(summary || "No summary available."),
  };
  const notes = [];
  if (rawPayload?.isError === true) {
    notes.push("Tool handler returned isError=true.");
  }
  if (notes.length > 0) {
    payload.notes = notes;
  }
  if (normalizedStatus === "error") {
    payload.error = compactErrorMessage(excerpt || summary || `Error in ${toolName}.`);
  }
  return payload;
}

function normalizeStructuredPayload(toolName, { status, summary, combinedText, contentPartCount, rawPayload, originalStructured }) {
  if (!originalStructured || typeof originalStructured !== "object") {
    return buildGenericToolPayload(toolName, {
      status,
      summary,
      combinedText,
      contentPartCount,
      rawPayload,
    });
  }

  const schemaValue = normalizeWhitespace(originalStructured.schema || "");
  if (!schemaValue) {
    return {
      ...buildGenericToolPayload(toolName, {
        status,
        summary,
        combinedText,
        contentPartCount,
        rawPayload,
      }),
      details: originalStructured,
    };
  }

  const normalizedPayload = { ...originalStructured };
  const payloadStatus = normalizeResultStatus(normalizedPayload.result_status, "");
  if (!payloadStatus) {
    normalizedPayload.result_status = normalizeResultStatus(status, "ok");
  }
  return normalizedPayload;
}

function normalizeToolResponseEnvelope(toolName, rawResult) {
  const payload = rawResult && typeof rawResult === "object" ? { ...rawResult } : {};
  const originalContent = Array.isArray(payload.content) ? payload.content : [];
  const normalizedParts = originalContent.map(normalizeTextPart).filter(Boolean);
  const fallbackText = typeof rawResult === "string" ? rawResult.trim() : "";
  const fallbackParts = fallbackText ? [{ type: "text", text: fallbackText }] : [];
  const content = normalizedParts.length > 0 ? normalizedParts : fallbackParts;
  const safeContent =
    content.length > 0
      ? content
      : [{ type: "text", text: `No response content was produced by ${toolName}.` }];
  const combinedText = safeContent.map((part) => part.text).join("\n");
  const status = inferEnvelopeStatus(payload, combinedText);
  const summary = extractSummaryLine(combinedText);
  const originalStructured =
    payload.structuredContent && typeof payload.structuredContent === "object"
      ? payload.structuredContent
      : null;
  const normalizedPayload = normalizeStructuredPayload(toolName, {
    status,
    summary,
    combinedText,
    contentPartCount: safeContent.length,
    rawPayload: payload,
    originalStructured,
  });

  const structuredContent = {
    envelope_version: STRUCTURED_CONTENT_ENVELOPE_VERSION,
    tool_name: toolName,
    status,
    summary,
    text: combinedText,
    content_part_count: safeContent.length,
    emitted_at_utc: new Date().toISOString(),
    payload: normalizedPayload,
  };

  const normalizedResponse = {
    ...payload,
    content: safeContent,
    structuredContent,
  };
  return normalizedResponse;
}

function wrapToolHandler(toolName, handler) {
  return async (...args) => {
    try {
      const rawResult = await handler(...args);
      return normalizeToolResponseEnvelope(toolName, rawResult);
    } catch (error) {
      const message = normalizeWhitespace(error?.message || String(error));
      return normalizeToolResponseEnvelope(toolName, {
        isError: true,
        content: [{ type: "text", text: `Error in ${toolName}: ${message}` }],
      });
    }
  };
}

const registerTool = server.registerTool.bind(server);
server.registerTool = (toolName, config, handler) =>
  registerTool(toolName, config, wrapToolHandler(toolName, handler));

function sanitizeXmlText(value) {
  if (!value) return "";
  return value.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseRetryAfterMs(value) {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) {
    return seconds * 1000;
  }
  const retryDate = Date.parse(value);
  if (Number.isFinite(retryDate)) {
    const delta = retryDate - Date.now();
    return delta > 0 ? delta : 0;
  }
  return null;
}

function buildOpenAlexUrl(pathname, params = new URLSearchParams()) {
  const nextParams = new URLSearchParams(params);
  if (OPENALEX_MAILTO && !nextParams.has("mailto")) {
    nextParams.set("mailto", OPENALEX_MAILTO);
  }
  const query = nextParams.toString();
  return query ? `${OPENALEX_API}${pathname}?${query}` : `${OPENALEX_API}${pathname}`;
}

function buildUniProtUrl(pathname, params = new URLSearchParams()) {
  const query = params.toString();
  return query ? `${UNIPROT_API}${pathname}?${query}` : `${UNIPROT_API}${pathname}`;
}
function buildHfDatasetsServerUrl(pathname, params = new URLSearchParams()) {
  const query = params.toString();
  return query ? `${HF_DATASETS_SERVER_API}${pathname}?${query}` : `${HF_DATASETS_SERVER_API}${pathname}`;
}

function buildHfHubDatasetUrl(dataset) {
  const normalized = normalizeWhitespace(dataset || "");
  if (!normalized) return "https://huggingface.co/datasets";
  const safePath = normalized
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `https://huggingface.co/datasets/${safePath}`;
}

function getHfAuthHeaders() {
  if (!HF_TOKEN) return {};
  return { Authorization: `Bearer ${HF_TOKEN}` };
}

async function fetchHfDatasetSplits(dataset) {
  const url = buildHfDatasetsServerUrl(
    "/splits",
    new URLSearchParams({
      dataset,
    })
  );
  return fetchJsonWithRetry(url, {
    headers: getHfAuthHeaders(),
    retries: 1,
    timeoutMs: 12000,
  });
}
function pickHfDefaultSplit(splitsResponse, fallbackConfig = "default", fallbackSplit = "train") {
  const splits = Array.isArray(splitsResponse?.splits) ? splitsResponse.splits : [];
  if (splits.length === 0) {
    return {
      config: fallbackConfig,
      split: fallbackSplit,
    };
  }
  const preferredTrain = splits.find(
    (row) => normalizeWhitespace(row?.split || "").toLowerCase() === "train"
  );
  const selected = preferredTrain || splits[0];
  return {
    config: normalizeWhitespace(selected?.config || fallbackConfig) || fallbackConfig,
    split: normalizeWhitespace(selected?.split || fallbackSplit) || fallbackSplit,
  };
}

function extractPubMedQaContextSnippet(contextPayload, maxContexts = 2) {
  const contexts = Array.isArray(contextPayload?.contexts) ? contextPayload.contexts : [];
  if (contexts.length === 0) return "";
  return contexts
    .slice(0, maxContexts)
    .map((value) => compactErrorMessage(normalizeWhitespace(value), 320))
    .filter(Boolean)
    .join(" | ");
}

function parseBioAsqPackedText(text) {
  const raw = String(text || "");
  const answerMatch = raw.match(/<answer>\s*([\s\S]*?)\s*<context>/i);
  const contextMatch = raw.match(/<context>\s*([\s\S]*)/i);
  const answer = normalizeWhitespace(answerMatch?.[1] || "");
  const context = normalizeWhitespace(contextMatch?.[1] || raw);
  return {
    answer,
    context,
  };
}

function extractUniProtProteinName(entry) {
  const recommended = normalizeWhitespace(entry?.proteinDescription?.recommendedName?.fullName?.value || "");
  if (recommended) return recommended;
  const submitted = normalizeWhitespace(entry?.proteinDescription?.submissionNames?.[0]?.fullName?.value || "");
  if (submitted) return submitted;
  const alternative = normalizeWhitespace(entry?.proteinDescription?.alternativeNames?.[0]?.fullName?.value || "");
  if (alternative) return alternative;
  return "Unknown protein";
}

function extractUniProtGeneSymbols(entry, limit = 6) {
  const genes = Array.isArray(entry?.genes) ? entry.genes : [];
  const symbols = [];
  for (const gene of genes) {
    const primary = normalizeWhitespace(gene?.geneName?.value || "");
    if (primary) symbols.push(primary);
    const synonyms = Array.isArray(gene?.synonyms) ? gene.synonyms : [];
    for (const synonym of synonyms) {
      const value = normalizeWhitespace(synonym?.value || "");
      if (value) symbols.push(value);
    }
  }
  return dedupeArray(symbols).slice(0, limit);
}

function extractUniProtEnsemblGeneIds(entry, limit = 6) {
  const refs = Array.isArray(entry?.uniProtKBCrossReferences) ? entry.uniProtKBCrossReferences : [];
  const geneIds = [];
  for (const ref of refs) {
    if (normalizeWhitespace(ref?.database || "").toLowerCase() !== "ensembl") continue;
    const properties = Array.isArray(ref?.properties) ? ref.properties : [];
    for (const prop of properties) {
      if (normalizeWhitespace(prop?.key || "").toLowerCase() !== "geneid") continue;
      const value = normalizeWhitespace(prop?.value || "");
      if (!value) continue;
      geneIds.push(value.split(".")[0]);
    }
  }
  return dedupeArray(geneIds).slice(0, limit);
}

function extractUniProtCommentTexts(entry, commentType, limit = 3, maxChars = 280) {
  const wantedType = normalizeWhitespace(commentType || "").toUpperCase();
  const comments = Array.isArray(entry?.comments) ? entry.comments : [];
  const output = [];
  for (const comment of comments) {
    if (normalizeWhitespace(comment?.commentType || "").toUpperCase() !== wantedType) continue;
    const texts = Array.isArray(comment?.texts) ? comment.texts : [];
    for (const textEntry of texts) {
      const value = normalizeWhitespace(textEntry?.value || "");
      if (!value) continue;
      output.push(compactErrorMessage(value, maxChars));
      if (output.length >= limit) return output;
    }
  }
  return output;
}

function extractUniProtSubcellularLocations(entry, limit = 8) {
  const comments = Array.isArray(entry?.comments) ? entry.comments : [];
  const locations = [];
  for (const comment of comments) {
    if (normalizeWhitespace(comment?.commentType || "").toUpperCase() !== "SUBCELLULAR LOCATION") continue;
    const slots = Array.isArray(comment?.subcellularLocations) ? comment.subcellularLocations : [];
    for (const slot of slots) {
      const location = normalizeWhitespace(slot?.location?.value || "");
      if (location) locations.push(location);
    }
  }
  return dedupeArray(locations).slice(0, limit);
}

function extractUniProtCrossRefs(entry, databaseName, limit = 8) {
  const database = normalizeWhitespace(databaseName || "").toLowerCase();
  const refs = Array.isArray(entry?.uniProtKBCrossReferences) ? entry.uniProtKBCrossReferences : [];
  const ids = refs
    .filter((ref) => normalizeWhitespace(ref?.database || "").toLowerCase() === database)
    .map((ref) => normalizeWhitespace(ref?.id || ""))
    .filter(Boolean);
  return dedupeArray(ids).slice(0, limit);
}

function summarizeUniProtFeatureTypes(entry, limit = 8) {
  const features = Array.isArray(entry?.features) ? entry.features : [];
  const counts = new Map();
  for (const feature of features) {
    const type = normalizeWhitespace(feature?.type || "");
    if (!type) continue;
    incrementCount(counts, type);
  }
  return summarizeTopCounts(counts, limit);
}

async function fetchWithRetry(url, options = {}) {
  const retries = options.retries ?? 2;
  const timeoutMs = options.timeoutMs ?? 12000;
  const maxBackoffMs = options.maxBackoffMs ?? 4000;
  const fetchOptions = { ...options };
  delete fetchOptions.retries;
  delete fetchOptions.timeoutMs;
  delete fetchOptions.maxBackoffMs;

  let lastError;
  for (let attempt = 0; attempt <= retries; attempt++) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(url, { ...fetchOptions, signal: controller.signal });
      clearTimeout(timer);
      if (response.ok) {
        return response;
      }

      const retryable = response.status === 429 || response.status >= 500;
      const responseBody = await response.text().catch(() => "");
      lastError = new Error(
        `Request failed (${response.status}): ${url}${
          responseBody ? ` | ${responseBody.slice(0, 220).replace(/\s+/g, " ").trim()}` : ""
        }`
      );
      if (!retryable || attempt >= retries) {
        throw lastError;
      }
      const retryAfterMs = parseRetryAfterMs(response.headers.get("retry-after"));
      const backoffMs = Math.min(maxBackoffMs, retryAfterMs ?? Math.min(maxBackoffMs, 600 * 2 ** attempt));
      await sleep(backoffMs + Math.floor(Math.random() * 200));
    } catch (error) {
      clearTimeout(timer);
      lastError = error;
      if (attempt >= retries) {
        throw lastError;
      }
      const backoffMs = Math.min(maxBackoffMs, 600 * 2 ** attempt);
      await sleep(backoffMs + Math.floor(Math.random() * 200));
    }
  }
  throw lastError;
}

async function fetchJsonWithRetry(url, options = {}) {
  const response = await fetchWithRetry(url, options);
  return response.json();
}
function formatGraphQLErrors(responsePayload) {
  const errors = Array.isArray(responsePayload?.errors) ? responsePayload.errors : [];
  if (errors.length === 0) return "";
  return errors
    .map((err) => normalizeWhitespace(err?.message || String(err)))
    .filter(Boolean)
    .slice(0, 4)
    .join(" | ");
}
function requireGraphQLData(responsePayload, providerName = "GraphQL") {
  const errText = formatGraphQLErrors(responsePayload);
  if (errText) {
    throw new Error(`${providerName} returned errors: ${errText}`);
  }
  return responsePayload?.data ?? null;
}

function renderStructuredResponse({ summary, keyFields = [], sources = [], limitations = [] }) {
  const fields =
    keyFields.length > 0
      ? keyFields.map((line) => `- ${line}`).join("\n")
      : "- No key fields available.";
  const sourceLines =
    sources.length > 0
      ? sources.map((line) => `- ${line}`).join("\n")
      : "- No sources provided.";
  const limitationLines =
    limitations.length > 0
      ? limitations.map((line) => `- ${line}`).join("\n")
      : "- No explicit limitations noted.";

  return `Summary:
${summary}

Key Fields:
${fields}

Sources:
${sourceLines}

Limitations:
${limitationLines}`;
}

function normalizeDoiValue(rawDoi) {
  const value = normalizeWhitespace(rawDoi || "");
  if (!value) return "";
  const stripped = value.replace(/^https?:\/\/(?:dx\.)?doi\.org\//i, "").trim();
  return stripped.replace(/^doi:\s*/i, "").trim();
}
function buildDoiMarkdown(rawDoi) {
  const doi = normalizeDoiValue(rawDoi);
  if (!doi) return "";
  return `[DOI:${doi}](https://doi.org/${doi})`;
}

function buildOpenAlexWorkCitation(work) {
  const title = normalizeWhitespace(work?.display_name || "Untitled");
  const year = toNonNegativeInt(work?.publication_year, 0);
  const venue = normalizeWhitespace(work?.primary_location?.source?.display_name || "");
  const firstAuthor = normalizeWhitespace(work?.authorships?.[0]?.author?.display_name || "");
  const authorLabel = firstAuthor ? `${firstAuthor}${(work?.authorships?.length || 0) > 1 ? " et al." : ""}` : "Unknown author";
  const doiMarkdown = buildDoiMarkdown(work?.doi || work?.ids?.doi || "");
  const openAlexId = normalizeWhitespace(work?.id || "");
  const openAlexLabel = openAlexId ? `[OpenAlex](${openAlexId})` : "";
  const venuePart = venue ? ` ${venue}.` : "";
  const yearPart = year > 0 ? year : "n.d.";
  const links = [doiMarkdown, openAlexLabel].filter(Boolean).join(" ");
  return `${authorLabel} (${yearPart}). ${title}.${venuePart}${links ? ` ${links}` : ""}`.trim();
}

function parseBigQueryDatasetAllowlist(rawValue, defaultProjectId = "") {
  const values = String(rawValue || "")
    .split(",")
    .map((value) => normalizeWhitespace(value))
    .filter(Boolean);
  const allowlist = [];
  const seen = new Set();

  for (const value of values) {
    const stripped = value.replace(/`/g, "");
    const parts = stripped
      .split(".")
      .map((part) => normalizeWhitespace(part))
      .filter(Boolean);
    let projectId = "";
    let datasetId = "";
    if (parts.length === 1) {
      projectId = defaultProjectId;
      datasetId = parts[0];
    } else if (parts.length >= 2) {
      projectId = parts[0];
      datasetId = parts[1];
    }
    if (!projectId || !datasetId) continue;
    const key = `${projectId}.${datasetId}`;
    const normalizedKey = key.toLowerCase();
    if (seen.has(normalizedKey)) continue;
    seen.add(normalizedKey);
    allowlist.push(key);
  }
  return allowlist;
}

const BQ_ALLOWED_DATASETS = parseBigQueryDatasetAllowlist(BQ_DATASET_ALLOWLIST, BQ_PROJECT_ID);
const BQ_ALLOWED_DATASET_SET = new Set(BQ_ALLOWED_DATASETS.map((value) => value.toLowerCase()));

function getBigQueryClient() {
  if (bigQueryClient) return bigQueryClient;
  const options = {};
  if (BQ_PROJECT_ID) {
    options.projectId = BQ_PROJECT_ID;
  }
  bigQueryClient = new BigQuery(options);
  return bigQueryClient;
}

function getBigQueryMaxBytesBilled() {
  const parsed = Number(String(process.env.BQ_MAX_BYTES_BILLED || "").trim());
  if (Number.isFinite(parsed) && parsed > 0) {
    return Math.max(10_000_000, Math.trunc(parsed));
  }
  return BQ_DEFAULT_MAX_BYTES_BILLED;
}

function normalizeBigQuerySql(sql) {
  let remaining = String(sql || "");
  while (true) {
    remaining = remaining.replace(/^\s+/, "");
    if (remaining.startsWith("--")) {
      const newline = remaining.indexOf("\n");
      if (newline === -1) return "";
      remaining = remaining.slice(newline + 1);
      continue;
    }
    if (remaining.startsWith("/*")) {
      const closing = remaining.indexOf("*/");
      if (closing === -1) return "";
      remaining = remaining.slice(closing + 2);
      continue;
    }
    break;
  }
  return remaining.replace(/;+\s*$/, "").trim();
}

function validateBigQuerySelectSql(sql) {
  const normalizedSql = normalizeBigQuerySql(sql);
  if (!normalizedSql) {
    return { ok: false, error: "Query is empty after normalization." };
  }
  const lowered = normalizedSql.toLowerCase();
  if (!(lowered.startsWith("select") || lowered.startsWith("with"))) {
    return { ok: false, error: "Only SELECT or WITH queries are allowed." };
  }
  if (FORBIDDEN_SQL_KEYWORDS.test(lowered)) {
    return { ok: false, error: "Query includes a forbidden SQL keyword." };
  }
  return { ok: true, normalizedSql };
}

function resolveDatasetKeyAgainstAllowlist(shortName) {
  const lower = shortName.toLowerCase();
  const match = BQ_ALLOWED_DATASETS.find(
    (entry) => entry.toLowerCase().endsWith(`.${lower}`)
  );
  return match || "";
}

function extractReferencedBigQueryDatasets(sql) {
  const identifiers = [...String(sql || "").matchAll(/`([^`]+)`/g)]
    .map((match) => normalizeWhitespace(match?.[1] || ""))
    .filter(Boolean);
  const datasets = [];
  const seen = new Set();
  for (const identifier of identifiers) {
    const parts = identifier
      .split(".")
      .map((part) => normalizeWhitespace(part))
      .filter(Boolean);
    let key = "";
    if (parts.length >= 3) {
      key = `${parts[0]}.${parts[1]}`;
    } else if (parts.length === 2) {
      key = resolveDatasetKeyAgainstAllowlist(parts[0]) || (BQ_PROJECT_ID ? `${BQ_PROJECT_ID}.${parts[0]}` : "");
    } else if (parts.length === 1) {
      key = resolveDatasetKeyAgainstAllowlist(parts[0]) || (BQ_PROJECT_ID ? `${BQ_PROJECT_ID}.${parts[0]}` : "");
    }
    if (!key) continue;
    const normalizedKey = key.toLowerCase();
    if (seen.has(normalizedKey)) continue;
    seen.add(normalizedKey);
    datasets.push(key);
  }
  return datasets;
}

function expandBigQueryTableReferences(sql) {
  return String(sql || "").replace(/`([^`]+)`/g, (original, identifier) => {
    const parts = identifier.split(".").map((p) => p.trim()).filter(Boolean);
    if (parts.length === 2) {
      const resolved = resolveDatasetKeyAgainstAllowlist(parts[0]);
      if (resolved) return `\`${resolved}.${parts[1]}\``;
    }
    return original;
  });
}

function isAllowedBigQueryDataset(datasetKey) {
  if (BQ_ALLOWED_DATASET_SET.size === 0) return true;
  return BQ_ALLOWED_DATASET_SET.has(String(datasetKey || "").toLowerCase());
}

function normalizeBigQueryDatasetKey(rawValue) {
  const value = normalizeWhitespace(rawValue || "").replace(/`/g, "");
  if (!value) return "";
  const parts = value
    .split(".")
    .map((part) => normalizeWhitespace(part))
    .filter(Boolean);
  if (parts.length >= 2) {
    return `${parts[0]}.${parts[1]}`;
  }
  if (parts.length === 1) {
    const shortName = parts[0].toLowerCase();
    const match = BQ_ALLOWED_DATASETS.find(
      (entry) => entry.toLowerCase().endsWith(`.${shortName}`)
    );
    if (match) return match;
    return BQ_PROJECT_ID ? `${BQ_PROJECT_ID}.${parts[0]}` : "";
  }
  return "";
}

function formatBigQueryBytes(value) {
  const bytes = Number(value);
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB", "PB"];
  let unitIndex = 0;
  let size = bytes;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const decimals = size >= 10 || unitIndex === 0 ? 0 : 1;
  return `${size.toFixed(decimals)} ${units[unitIndex]}`;
}

function incrementCount(counter, key) {
  const label = key && typeof key === "string" ? key.trim() : "";
  const normalized = label || "Unknown";
  counter.set(normalized, (counter.get(normalized) || 0) + 1);
}

function summarizeTopCounts(counter, limit = 5) {
  return Array.from(counter.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([label, count]) => `${label} (${count})`);
}

function safeNumber(value, fallback = Number.NaN) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatClinicalPhaseLabel(phaseValue) {
  const numeric = safeNumber(phaseValue, Number.NaN);
  if (!Number.isFinite(numeric)) return "Unknown";
  if (numeric >= 4) return "Approved";
  if (numeric >= 3) return "Phase III";
  if (numeric >= 2) return "Phase II";
  if (numeric >= 1) return "Phase I";
  if (numeric >= 0.5) return "Early Phase I";
  if (numeric >= 0) return "Preclinical";
  return "Unknown";
}
function clamp01(value) {
  const numeric = safeNumber(value, 0);
  if (numeric <= 0) return 0;
  if (numeric >= 1) return 1;
  return numeric;
}

function normalizeWeightProfile(profile) {
  const entries = Object.entries(profile || {}).filter(([, value]) => safeNumber(value, 0) > 0);
  const total = entries.reduce((acc, [, value]) => acc + safeNumber(value, 0), 0);
  if (total <= 0) {
    return {
      disease_association: 0.3,
      druggability: 0.2,
      clinical_maturity: 0.2,
      competitive_whitespace: 0.15,
      safety: 0.15,
    };
  }
  const normalized = {};
  for (const [key, value] of entries) {
    normalized[key] = safeNumber(value, 0) / total;
  }
  return normalized;
}

function strategyWeightProfile(strategy) {
  const normalizedStrategy = String(strategy || "balanced").trim().toLowerCase();
  if (normalizedStrategy === "first_in_class") {
    return normalizeWeightProfile({
      disease_association: 0.3,
      druggability: 0.15,
      clinical_maturity: 0.1,
      competitive_whitespace: 0.3,
      safety: 0.15,
    });
  }
  if (normalizedStrategy === "de_risked") {
    return normalizeWeightProfile({
      disease_association: 0.25,
      druggability: 0.2,
      clinical_maturity: 0.3,
      competitive_whitespace: 0.1,
      safety: 0.15,
    });
  }
  if (normalizedStrategy === "safety_first") {
    return normalizeWeightProfile({
      disease_association: 0.25,
      druggability: 0.15,
      clinical_maturity: 0.15,
      competitive_whitespace: 0.1,
      safety: 0.35,
    });
  }
  return normalizeWeightProfile({
    disease_association: 0.3,
    druggability: 0.2,
    clinical_maturity: 0.2,
    competitive_whitespace: 0.15,
    safety: 0.15,
  });
}

function customWeightProfile(customWeights) {
  if (!customWeights || typeof customWeights !== "object") return null;
  const profile = {
    disease_association: Math.max(0, safeNumber(customWeights?.diseaseAssociation, 0)),
    druggability: Math.max(0, safeNumber(customWeights?.druggability, 0)),
    clinical_maturity: Math.max(0, safeNumber(customWeights?.clinicalMaturity, 0)),
    competitive_whitespace: Math.max(0, safeNumber(customWeights?.competitiveWhitespace, 0)),
    safety: Math.max(0, safeNumber(customWeights?.safety, 0)),
  };
  const hasPositive = Object.values(profile).some((value) => value > 0);
  if (!hasPositive) return null;
  return normalizeWeightProfile(profile);
}

function formatWeightInputSummary(customWeights) {
  if (!customWeights || typeof customWeights !== "object") return "none";
  const fields = [
    `diseaseAssociation=${safeNumber(customWeights?.diseaseAssociation, 0)}`,
    `druggability=${safeNumber(customWeights?.druggability, 0)}`,
    `clinicalMaturity=${safeNumber(customWeights?.clinicalMaturity, 0)}`,
    `competitiveWhitespace=${safeNumber(customWeights?.competitiveWhitespace, 0)}`,
    `safety=${safeNumber(customWeights?.safety, 0)}`,
  ];
  return fields.join(", ");
}

function inferStrategyFromGoalText(goalText) {
  const text = String(goalText || "").toLowerCase().trim();
  if (!text) {
    return {
      mode: "balanced",
      reason: "No explicit prioritization phrase was provided; defaulting to balanced strategy.",
      needsClarification: false,
    };
  }

  const keywordConfig = {
    safety_first: [
      "safest",
      "safety",
      "least risky",
      "lowest risk",
      "toxicity",
      "tolerability",
      "adverse",
      "side effect",
      "side-effect",
    ],
    first_in_class: [
      "novel",
      "novelty",
      "first-in-class",
      "first in class",
      "white space",
      "whitespace",
      "less crowded",
      "uncrowded",
      "differentiated",
      "differentiate",
    ],
    de_risked: [
      "de-risked",
      "derisked",
      "de risked",
      "de-risk",
      "near-term success",
      "highest chance",
      "high probability",
      "mature",
      "late-stage",
      "clinic-ready",
      "fastest path",
    ],
    balanced: ["balanced", "overall"],
  };

  const isNegatedKeyword = (fullText, keyword) => {
    const patterns = [
      `not ${keyword}`,
      `no ${keyword}`,
      `without ${keyword}`,
      `do not ${keyword}`,
      `avoid ${keyword}`,
      `deprioritize ${keyword}`,
    ];
    return patterns.some((pattern) => fullText.includes(pattern));
  };

  const scores = new Map();
  for (const [mode, keywords] of Object.entries(keywordConfig)) {
    let score = 0;
    for (const keyword of keywords) {
      if (!text.includes(keyword)) continue;
      if (isNegatedKeyword(text, keyword)) continue;
      score += 1;
    }
    scores.set(mode, score);
  }

  const ranked = Array.from(scores.entries()).sort((a, b) => b[1] - a[1]);
  const [bestMode, bestScore] = ranked[0] || ["balanced", 0];
  const topOperationalModes = ranked.filter(
    ([mode, score]) => mode !== "balanced" && score > 0
  );
  const explicitMultiObjectiveLanguage =
    text.includes("both") ||
    text.includes("at the same time") ||
    text.includes("simultaneously") ||
    text.includes("equally") ||
    text.includes("primary goals");
  if (explicitMultiObjectiveLanguage && topOperationalModes.length >= 2) {
    return {
      mode: "balanced",
      reason: "Goal text explicitly requested multiple primary objectives simultaneously.",
      needsClarification: true,
      clarificationQuestion:
        "You asked for multiple primary goals at once. Which one should be primary for ranking: safety-first, first-in-class novelty, or de-risked?",
    };
  }

  const nearTieModes = ranked.filter((entry) => entry[1] >= bestScore - 1 && entry[1] > 0).map((entry) => entry[0]);
  if (nearTieModes.length >= 2 && bestScore > 0) {
    return {
      mode: "balanced",
      reason: "Goal text matched multiple near-equal optimization intents.",
      needsClarification: true,
      clarificationQuestion:
        "I detected competing optimization intents. Which should be primary: safety-first, first-in-class novelty, de-risked, or balanced?",
    };
  }

  const sameBest = ranked.filter((entry) => entry[1] === bestScore && bestScore > 0).map((entry) => entry[0]);
  if (sameBest.length >= 2) {
    const labels = sameBest
      .map((mode) =>
        mode === "safety_first"
          ? "safety-first"
          : mode === "first_in_class"
            ? "first-in-class novelty"
            : mode === "de_risked"
              ? "de-risked"
              : "balanced"
      )
      .join(" vs ");
    return {
      mode: "balanced",
      reason: `Goal text matched multiple conflicting priorities (${labels}).`,
      needsClarification: true,
      clarificationQuestion:
        "Your request mixes multiple optimization goals (e.g., safest and most novel). Which should be primary: safety-first, first-in-class novelty, de-risked, or balanced?",
    };
  }

  if (bestScore <= 0) {
    return {
      mode: "balanced",
      reason: "No recognizable optimization keywords were detected; defaulting to balanced strategy.",
      needsClarification: false,
    };
  }

  return {
    mode: bestMode,
    reason: `Auto-selected strategy "${bestMode}" from goal text keywords.`,
    needsClarification: false,
  };
}

function formatPct(value) {
  return `${(clamp01(value) * 100).toFixed(1)}%`;
}

function sanitizeReasonText(value) {
  return (value || "Reason not reported").replace(/\s+/g, " ").trim();
}

function tokenizeQuery(text) {
  if (!text || typeof text !== "string") return [];
  const stopwords = new Set([
    "and",
    "or",
    "the",
    "of",
    "for",
    "with",
    "disease",
    "syndrome",
    "disorder",
    "type",
    "stage",
    "human",
  ]);
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length >= 3 && !stopwords.has(token));
}

function mapProteinLevel(level) {
  const numeric = safeNumber(level, -1);
  if (numeric >= 3) return "high";
  if (numeric === 2) return "medium";
  if (numeric === 1) return "low";
  if (numeric === 0) return "not_detected";
  return "unknown";
}

function dedupeArray(values) {
  return [...new Set(values.filter(Boolean))];
}

function toFiniteNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function toNonNegativeInt(value, fallback = 0) {
  return Math.max(0, Math.trunc(toFiniteNumber(value, fallback)));
}

function toNullableNumber(value) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function compactErrorMessage(value, maxChars = 220) {
  const text = normalizeWhitespace(value || "");
  if (!text) return "";
  if (text.length <= maxChars) return text;
  return `${text.slice(0, maxChars - 3).trim()}...`;
}

function safeBuildTypedPayload(schema, payload, schemaName) {
  const parsed = schema.safeParse(payload);
  if (parsed.success) {
    return parsed.data;
  }
  const issues = parsed.error.issues.slice(0, 6).map((issue) => {
    const path = issue.path.length ? issue.path.join(".") : "(root)";
    return `${path}: ${issue.message}`;
  });
  return {
    schema: `${schemaName}.invalid`,
    result_status: "error",
    validation_errors: issues,
  };
}

const RankResearcherExampleWorkPayloadSchema = z.object({
  title: z.string(),
  year: z.number().int().nullable(),
  cited_by: z.number().int().nonnegative(),
});

const RankResearcherPayloadSchema = z.object({
  rank: z.number().int().positive(),
  author_id: z.string(),
  name: z.string(),
  institution: z.string(),
  activity_score: z.number(),
  topic_works: z.number().int().nonnegative(),
  topic_citations: z.number().int().nonnegative(),
  recent_topic_works: z.number().int().nonnegative(),
  leadership_works: z.number().int().nonnegative(),
  active_years: z.number().int().nonnegative(),
  example_works: z.array(RankResearcherExampleWorkPayloadSchema),
});

const RankResearchersPayloadSchema = z.object({
  schema: z.literal("rank_researchers_by_activity.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  query: z.string(),
  from_year: z.number().int().nonnegative(),
  limit: z.number().int().positive(),
  scanned_works: z.number().int().nonnegative(),
  researcher_count: z.number().int().nonnegative(),
  researchers: z.array(RankResearcherPayloadSchema),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildRankResearchersPayload({
  resultStatus = "ok",
  query = "",
  fromYear = 0,
  limit = 10,
  scannedWorks = 0,
  researchers = [],
  notes = [],
  errorMessage = "",
}) {
  return safeBuildTypedPayload(
    RankResearchersPayloadSchema,
    {
      schema: "rank_researchers_by_activity.v1",
      result_status: resultStatus,
      query: String(query || ""),
      from_year: toNonNegativeInt(fromYear),
      limit: Math.max(1, toNonNegativeInt(limit, 10)),
      scanned_works: toNonNegativeInt(scannedWorks),
      researcher_count: researchers.length,
      researchers,
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "rank_researchers_by_activity.v1"
  );
}

const ClinicalTrialsLandscapePayloadSchema = z.object({
  schema: z.literal("summarize_clinical_trials_landscape.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  query: z.string(),
  status_filter: z.string(),
  studies_analyzed: z.number().int().nonnegative(),
  total_reported: z.number().int().nonnegative().nullable(),
  max_studies: z.number().int().positive(),
  max_pages: z.number().int().positive(),
  has_more_pages: z.boolean(),
  trials_with_posted_results: z.number().int().nonnegative(),
  status_breakdown: z.array(z.string()),
  phase_breakdown: z.array(z.string()),
  top_interventions: z.array(z.string()),
  top_conditions: z.array(z.string()),
  top_termination_reasons: z.array(z.string()),
  example_terminated_nct_ids: z.array(z.string()),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildClinicalTrialsLandscapePayload({
  resultStatus = "ok",
  query = "",
  statusFilter = "",
  studiesAnalyzed = 0,
  totalReported = null,
  maxStudies = 60,
  maxPages = 4,
  hasMorePages = false,
  trialsWithPostedResults = 0,
  statusBreakdown = [],
  phaseBreakdown = [],
  topInterventions = [],
  topConditions = [],
  topTerminationReasons = [],
  exampleTerminatedNctIds = [],
  notes = [],
  errorMessage = "",
}) {
  return safeBuildTypedPayload(
    ClinicalTrialsLandscapePayloadSchema,
    {
      schema: "summarize_clinical_trials_landscape.v1",
      result_status: resultStatus,
      query: String(query || ""),
      status_filter: String(statusFilter || ""),
      studies_analyzed: toNonNegativeInt(studiesAnalyzed),
      total_reported: toNullableNumber(totalReported),
      max_studies: Math.max(1, toNonNegativeInt(maxStudies, 60)),
      max_pages: Math.max(1, toNonNegativeInt(maxPages, 4)),
      has_more_pages: Boolean(hasMorePages),
      trials_with_posted_results: toNonNegativeInt(trialsWithPostedResults),
      status_breakdown: dedupeArray((statusBreakdown || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      phase_breakdown: dedupeArray((phaseBreakdown || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      top_interventions: dedupeArray((topInterventions || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      top_conditions: dedupeArray((topConditions || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      top_termination_reasons: dedupeArray((topTerminationReasons || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      example_terminated_nct_ids: dedupeArray((exampleTerminatedNctIds || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "summarize_clinical_trials_landscape.v1"
  );
}


// ============================================
// TOOL 5: List BigQuery tables (read-only metadata) + schema discovery
// ============================================
server.registerTool(
  "list_bigquery_tables",
  {
    description:
      "Lists tables for a BigQuery dataset with column schemas. When `table` is provided, returns the full column schema (name, type, description) for that table — use this to discover column names before writing queries. When `table` is omitted, lists all tables in the dataset.",
    inputSchema: {
      dataset: z
        .string()
        .optional()
        .describe("Dataset as `project.dataset` or just `dataset` (e.g. 'open_targets_platform', 'ebi_chembl'). Short names are resolved against the allowlist. If omitted, uses first allowlisted dataset."),
      table: z
        .string()
        .optional()
        .describe("Table name to inspect (e.g. 'metadata', 'evidence'). When provided, returns the full column schema for that table instead of the table list."),
      limit: z.number().optional().default(200).describe("Maximum number of tables to return (1-200). Ignored when `table` is set."),
    },
  },
  async ({ dataset = "", table = "", limit = 200 }) => {
    const boundedLimit = Math.max(1, Math.min(200, Math.round(Number(limit) || 200)));
    const normalizedDataset = normalizeBigQueryDatasetKey(dataset);
    const selectedDataset = normalizedDataset || BQ_ALLOWED_DATASETS[0] || "";

    if (!selectedDataset) {
      return {
        content: [
          {
            type: "text",
            text:
              "No dataset specified and no allowlist configured. Set BQ_DATASET_ALLOWLIST (for example, `project.dataset`) or pass `dataset`.",
          },
        ],
      };
    }
    if (!isAllowedBigQueryDataset(selectedDataset)) {
      return {
        content: [
          {
            type: "text",
            text: `Dataset ${selectedDataset} is not in the BQ_DATASET_ALLOWLIST.`,
          },
        ],
      };
    }

    const [projectId, datasetId] = selectedDataset.split(".");
    if (!projectId || !datasetId) {
      return {
        content: [{ type: "text", text: `Invalid dataset reference: ${selectedDataset}` }],
      };
    }

    const client = getBigQueryClient();
    const datasetRef = client.dataset(datasetId, { projectId });

    const requestedTable = normalizeWhitespace(table || "").trim();
    if (requestedTable) {
      try {
        const tableRef = datasetRef.table(requestedTable);
        const [tableMetadata] = await tableRef.getMetadata();
        const schemaFields = tableMetadata?.schema?.fields || [];
        if (!schemaFields.length) {
          return {
            content: [
              {
                type: "text",
                text: renderStructuredResponse({
                  summary: `Table \`${selectedDataset}.${requestedTable}\` exists but has no schema fields.`,
                  keyFields: [`Dataset: ${selectedDataset}`, `Table: ${requestedTable}`],
                  sources: [`bigquery://${selectedDataset}.${requestedTable}`],
                  limitations: ["The table may be empty or use a non-standard schema."],
                }),
              },
            ],
          };
        }

        function formatFields(fields, indent = 0) {
          const lines = [];
          for (const field of fields) {
            const name = field.name || "unknown";
            const type = field.type || "UNKNOWN";
            const mode = field.mode && field.mode !== "NULLABLE" ? ` (${field.mode})` : "";
            const desc = field.description ? ` — ${field.description}` : "";
            const prefix = "  ".repeat(indent);
            lines.push(`${prefix}- ${name}: ${type}${mode}${desc}`);
            if (Array.isArray(field.fields) && field.fields.length > 0) {
              lines.push(...formatFields(field.fields, indent + 1));
            }
          }
          return lines;
        }

        const schemaLines = formatFields(schemaFields);
        const numRows = tableMetadata?.numRows ? `Rows: ~${Number(tableMetadata.numRows).toLocaleString()}` : null;
        const numBytes = tableMetadata?.numBytes ? `Size: ${formatBigQueryBytes(Number(tableMetadata.numBytes))}` : null;
        const keyFields = [
          `Dataset: ${selectedDataset}`,
          `Table: ${requestedTable}`,
          `Columns: ${schemaFields.length}`,
        ];
        if (numRows) keyFields.push(numRows);
        if (numBytes) keyFields.push(numBytes);

        return {
          content: [
            {
              type: "text",
              text: `${renderStructuredResponse({
                summary: `Schema for \`${selectedDataset}.${requestedTable}\` (${schemaFields.length} columns).`,
                keyFields,
                sources: [`bigquery://${selectedDataset}.${requestedTable}`],
                limitations: [
                  "Use these column names in your SQL queries. Wrap table references in backticks.",
                ],
              })}\n\nColumns:\n${schemaLines.join("\n")}`,
            },
          ],
        };
      } catch (error) {
        return {
          content: [{ type: "text", text: `Error fetching schema for ${selectedDataset}.${requestedTable}: ${error.message}` }],
        };
      }
    }

    try {
      const [tables] = await datasetRef.getTables({ maxResults: boundedLimit, autoPaginate: false });
      if (!tables || tables.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `No tables found for ${selectedDataset}.`,
                keyFields: [`Dataset: ${selectedDataset}`],
                sources: [`bigquery://${selectedDataset}`],
                limitations: [
                  "Dataset may be empty, inaccessible, or filtered by IAM.",
                ],
              }),
            },
          ],
        };
      }

      const tableList = tables
        .slice(0, boundedLimit)
        .map((table, idx) => `${idx + 1}. ${table.id}`)
        .join("\n");
      return {
        content: [
          {
            type: "text",
            text: `${renderStructuredResponse({
              summary: `Retrieved ${Math.min(tables.length, boundedLimit)} table(s) from ${selectedDataset}.`,
              keyFields: [
                `Dataset: ${selectedDataset}`,
                `Requested limit: ${boundedLimit}`,
                `Allowlist enforced: ${BQ_ALLOWED_DATASET_SET.size > 0 ? "yes" : "no"}`,
              ],
              sources: [`bigquery://${selectedDataset}`],
              limitations: [
                "To see column names for a table, call this tool again with the `table` parameter.",
              ],
            })}\n\nTables:\n${tableList}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in list_bigquery_tables: ${error.message}` }],
      };
    }
  }
);

// ============================================
// TOOL 6: Run guarded BigQuery SELECT query
// ============================================
server.registerTool(
  "run_bigquery_select_query",
  {
    description:
      "Runs a read-only BigQuery SELECT/WITH query with row and bytes guardrails.",
    inputSchema: {
      query: z
        .string()
        .describe("BigQuery Standard SQL query. Must be read-only (SELECT/WITH). Table refs like `dataset.table` are auto-expanded to `project.dataset.table` using the allowlist."),
      maxRows: z.number().optional().default(100).describe("Max rows to return (1-1000)."),
      dryRun: z.boolean().optional().default(false).describe("If true, validates and estimates bytes without executing."),
    },
  },
  async ({ query, maxRows = 100, dryRun = false }) => {
    const expandedQuery = expandBigQueryTableReferences(query);
    const validation = validateBigQuerySelectSql(expandedQuery);
    if (!validation.ok) {
      return {
        content: [{ type: "text", text: `Invalid query: ${validation.error}` }],
      };
    }
    const normalizedSql = validation.normalizedSql;
    const referencedDatasets = extractReferencedBigQueryDatasets(normalizedSql);
    if (referencedDatasets.length === 0) {
      return {
        content: [
          {
            type: "text",
            text:
              "Query must reference at least one table with backticks (for example `project.dataset.table`) so dataset guardrails can be enforced.",
          },
        ],
      };
    }

    const disallowedDatasets = referencedDatasets.filter((datasetKey) => !isAllowedBigQueryDataset(datasetKey));
    if (disallowedDatasets.length > 0) {
      return {
        content: [
          {
            type: "text",
            text: `Query references dataset(s) outside BQ_DATASET_ALLOWLIST: ${disallowedDatasets.join(", ")}`,
          },
        ],
      };
    }

    const boundedRows = Math.max(1, Math.min(BQ_HARD_MAX_ROWS, Math.round(Number(maxRows) || 100)));
    const maximumBytesBilled = getBigQueryMaxBytesBilled();
    const jobOptions = {
      query: normalizedSql,
      useLegacySql: false,
      location: BQ_LOCATION,
      maximumBytesBilled: String(maximumBytesBilled),
      jobTimeoutMs: BQ_QUERY_TIMEOUT_MS,
      dryRun: Boolean(dryRun),
    };

    try {
      const client = getBigQueryClient();
      const [job] = await client.createQueryJob(jobOptions);
      const [metadata] = await job.getMetadata();
      const stats = metadata?.statistics?.query || {};
      const totalBytesProcessed = Number(stats?.totalBytesProcessed || 0);
      const totalBytesBilled = Number(stats?.totalBytesBilled || 0);
      const cacheHit = Boolean(stats?.cacheHit);

      if (Boolean(dryRun)) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "BigQuery dry run completed successfully.",
                keyFields: [
                  `Datasets referenced: ${referencedDatasets.join(", ")}`,
                  `Bytes processed estimate: ${formatBigQueryBytes(totalBytesProcessed)} (${totalBytesProcessed})`,
                  `Max bytes billed guardrail: ${formatBigQueryBytes(maximumBytesBilled)} (${maximumBytesBilled})`,
                  `Location: ${BQ_LOCATION}`,
                ],
                sources: referencedDatasets.map((datasetKey) => `bigquery://${datasetKey}`),
                limitations: [
                  "Dry run validates syntax and estimates bytes, but does not return rows.",
                ],
              }),
            },
          ],
        };
      }

      const [rows] = await job.getQueryResults({ maxResults: boundedRows, autoPaginate: false });
      const previewLines = (rows || [])
        .slice(0, boundedRows)
        .map((row, idx) => `${idx + 1}. ${JSON.stringify(row)}`);
      const previewTextRaw = previewLines.join("\n");
      const previewText =
        previewTextRaw.length > 7000 ? `${previewTextRaw.slice(0, 7000)}\n... (truncated)` : previewTextRaw;

      return {
        content: [
          {
            type: "text",
            text: `${renderStructuredResponse({
              summary: `BigQuery query completed with ${rows?.length || 0} row(s) returned.`,
              keyFields: [
                `Datasets referenced: ${referencedDatasets.join(", ")}`,
                `Rows returned: ${rows?.length || 0}`,
                `Row cap: ${boundedRows}`,
                `Bytes processed: ${formatBigQueryBytes(totalBytesProcessed)} (${totalBytesProcessed})`,
                `Bytes billed: ${formatBigQueryBytes(totalBytesBilled)} (${totalBytesBilled})`,
                `Cache hit: ${cacheHit ? "yes" : "no"}`,
                `Location: ${BQ_LOCATION}`,
                `Job ID: ${job?.id || "unknown"}`,
              ],
              sources: referencedDatasets.map((datasetKey) => `bigquery://${datasetKey}`),
              limitations: [
                "Only the first page of results is returned.",
                `Rows are capped at ${boundedRows} to control costs and context size.`,
              ],
            })}\n\nRow Preview:\n${previewText || "(no rows returned)"}`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in run_bigquery_select_query: ${error.message}` }],
      };
    }
  }
);

// ============================================
// TOOL 7: Benchmark dataset overview (GPQA, PubMedQA, BioASQ)
// ============================================
server.registerTool(
  "benchmark_dataset_overview",
  {
    description:
      "Provides a practical overview of GPQA, PubMedQA, and BioASQ benchmark datasets, with optional live accessibility checks.",
    inputSchema: {
      checkAccess: z.boolean().optional().default(true).describe("If true, checks dataset split access via Hugging Face datasets-server."),
    },
  },
  async ({ checkAccess = true }) => {
    const catalog = [
      {
        name: "GPQA",
        dataset: HF_DATASET_GPQA,
        bestFor: "Hard graduate-level scientific reasoning and multi-step tool planning stress tests.",
      },
      {
        name: "PubMedQA",
        dataset: HF_DATASET_PUBMEDQA,
        bestFor: "Biomedical yes/no/maybe QA with evidence-grounded abstract context.",
      },
      {
        name: "BioASQ",
        dataset: HF_DATASET_BIOASQ,
        bestFor: "Biomedical retrieval + synthesis evaluation with realistic QA-style prompts.",
      },
    ];

    const keyFields = catalog.map(
      (entry) => `${entry.name}: ${entry.dataset} (${entry.bestFor})`
    );
    const limitations = [
      "Dataset schemas can vary by source/config and may change over time.",
      "Some datasets (for example GPQA) may be gated and require Hugging Face terms acceptance and/or token auth.",
    ];

    if (checkAccess) {
      for (const entry of catalog) {
        try {
          const splitPayload = await fetchHfDatasetSplits(entry.dataset);
          const splits = Array.isArray(splitPayload?.splits) ? splitPayload.splits : [];
          if (splits.length === 0) {
            keyFields.push(`${entry.name} access: reachable but no splits reported.`);
          } else {
            const splitPreview = splits
              .slice(0, 4)
              .map((row) => `${row.config}/${row.split}`)
              .join(", ");
            keyFields.push(`${entry.name} access: OK (${splits.length} split entries, e.g. ${splitPreview}).`);
          }
        } catch (error) {
          keyFields.push(`${entry.name} access: unavailable (${compactErrorMessage(error?.message || "unknown error", 180)}).`);
        }
      }
    }

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: "Benchmark dataset catalog for local non-BigQuery exploration is ready.",
            keyFields,
            sources: catalog.map((entry) => buildHfHubDatasetUrl(entry.dataset)),
            limitations,
          }),
        },
      ],
    };
  }
);
// ============================================
// TOOL 10: Check GPQA accessibility
// ============================================
server.registerTool(
  "check_gpqa_access",
  {
    description:
      "Checks GPQA dataset accessibility and reports next steps for gated access (without exposing benchmark question content).",
  },
  async () => {
    try {
      const splitPayload = await fetchHfDatasetSplits(HF_DATASET_GPQA);
      const splits = Array.isArray(splitPayload?.splits) ? splitPayload.splits : [];
      const splitSummary = splits.length > 0
        ? splits.slice(0, 6).map((row) => `${row.config}/${row.split}`).join(", ")
        : "No splits reported";
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: "GPQA endpoint is reachable.",
              keyFields: [
                `Dataset: ${HF_DATASET_GPQA}`,
                `Split summary: ${splitSummary}`,
                "Question-content display is intentionally restricted in this tool to reduce benchmark leakage risk.",
              ],
              sources: [buildHfHubDatasetUrl(HF_DATASET_GPQA)],
              limitations: [
                "Even with access, avoid publishing raw benchmark questions to preserve evaluation integrity.",
              ],
            }),
          },
        ],
      };
    } catch (error) {
      const message = compactErrorMessage(error?.message || "unknown error", 220);
      const authHint = HF_TOKEN
        ? "HF token is present, but dataset may still require explicit gating approval on the Hugging Face dataset page."
        : "No HF token detected. Set HF_TOKEN (or HUGGINGFACE_TOKEN) after accepting dataset terms on Hugging Face.";
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: "GPQA dataset access is currently unavailable.",
              keyFields: [
                `Dataset: ${HF_DATASET_GPQA}`,
                `Error: ${message}`,
                authHint,
              ],
              sources: [buildHfHubDatasetUrl(HF_DATASET_GPQA)],
              limitations: [
                "GPQA is commonly gated; local tool access depends on Hugging Face permissions.",
              ],
            }),
          },
        ],
      };
    }
  }
);
// ============================================
// TOOL 12: Search clinical trials
// ============================================
server.registerTool(
  "search_clinical_trials",
  {
    description:
      "Searches ClinicalTrials.gov for clinical trials. Find trials by disease, drug, target/gene, or sponsor. Returns trial status, phase, and key details. Supports pagination up to 200 results.",
    inputSchema: {
      query: z
        .string()
        .describe("Search terms (e.g., 'LRRK2 Parkinson', 'pembrolizumab lung cancer', 'Alzheimer Phase 3')"),
      status: z
        .string()
        .optional()
        .describe("Filter by status: 'RECRUITING', 'COMPLETED', 'ACTIVE_NOT_RECRUITING', 'TERMINATED', or leave empty for all"),
      limit: z.number().optional().describe("Max results to return. Omit to let the system choose a reasonable default based on the query. Hard cap: 200."),
    },
  },
  async ({ query, status, limit }) => {
    const boundedLimit = limit ? Math.max(1, Math.min(200, Math.round(limit))) : 50;
    const boundedPages = Math.ceil(boundedLimit / 100);
    const studies = [];
    let resultCount = 0;
    let hasMorePages = false;
    let nextPageToken = "";

    try {
      for (let page = 0; page < boundedPages && studies.length < boundedLimit; page++) {
        const pageSize = Math.min(100, boundedLimit - studies.length);
        const params = new URLSearchParams({
          "query.term": query,
          pageSize: String(pageSize),
          format: "json",
        });

        if (status) {
          params.append("filter.overallStatus", status);
        }
        if (nextPageToken) {
          params.append("pageToken", nextPageToken);
        }

        const url = `${CLINICAL_TRIALS_API}/studies?${params.toString()}`;
        const data = await fetchJsonWithRetry(url, { retries: 2, timeoutMs: 15000, maxBackoffMs: 3500 });
        const pageStudies = data?.studies ?? [];
        if (Number.isFinite(data?.totalCount)) {
          resultCount = data.totalCount;
        }
        studies.push(...pageStudies);
        nextPageToken = data?.nextPageToken || "";
        if (!nextPageToken || pageStudies.length === 0) break;
      }
      hasMorePages = Boolean(nextPageToken);
      if (!resultCount) resultCount = studies.length;
    } catch (error) {
      if (studies.length === 0) {
        return {
          content: [{ type: "text", text: `Error searching clinical trials: ${error.message}. Try again or use different search terms.` }],
        };
      }
    }

    if (studies.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: `No clinical trials found for: "${query}"${status ? ` with status ${status}` : ""}`,
          },
        ],
      };
    }

    const formatted = studies.map((study, i) => {
      const protocol = study.protocolSection;
      const id = protocol?.identificationModule;
      const status = protocol?.statusModule;
      const design = protocol?.designModule;
      const conditions = protocol?.conditionsModule?.conditions?.slice(0, 3).join(", ") || "Not specified";
      const interventions = protocol?.armsInterventionsModule?.interventions
        ?.slice(0, 2)
        .map((int) => `${int.name} (${int.type})`)
        .join(", ") || "Not specified";
      const sponsor = protocol?.sponsorCollaboratorsModule?.leadSponsor?.name || "Unknown";

      const phase = design?.phases?.join(", ") || "Not specified";
      const enrollment = design?.enrollmentInfo?.count || "Unknown";

      return `${i + 1}. ${id?.briefTitle || "Untitled"}
   NCT ID: ${id?.nctId || "Unknown"}
   Status: ${status?.overallStatus || "Unknown"}
   Phase: ${phase}
   Conditions: ${conditions}
   Interventions: ${interventions}
   Enrollment: ${enrollment} participants
   Sponsor: ${sponsor}`;
    }).join("\n\n");

    return {
      content: [
        {
          type: "text",
          text: `Clinical trials for "${query}":\nShowing ${studies.length} of ${resultCount} total trials${hasMorePages ? " (more available — increase limit or maxPages)" : ""}${status ? `\nStatus filter: ${status}` : ""}\n\n${formatted}\n\nUse get_clinical_trial with the NCT ID for full details including results.`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 13: Get clinical trial details
// ============================================
server.registerTool(
  "get_clinical_trial",
  {
    description:
      "Gets detailed information about a specific clinical trial including design, outcomes, and results if available.",
    inputSchema: {
      nctId: z.string().describe("ClinicalTrials.gov ID (e.g., 'NCT04665245')"),
    },
  },
  async ({ nctId }) => {
    const url = `${CLINICAL_TRIALS_API}/studies/${nctId}?format=json`;
    
    let study;
    try {
      study = await fetchJsonWithRetry(url, { retries: 2, timeoutMs: 15000, maxBackoffMs: 3500 });
    } catch (error) {
      const message = String(error?.message || "");
      if (message.includes("404")) {
        return {
          content: [{ type: "text", text: `Clinical trial not found: ${nctId}. Check the NCT ID format (e.g., NCT04665245).` }],
        };
      }
      return {
        content: [{ type: "text", text: `Error fetching trial ${nctId}: ${error.message}` }],
      };
    }
    const protocol = study.protocolSection;
    const results = study.resultsSection;

    // Basic info
    const id = protocol?.identificationModule;
    const status = protocol?.statusModule;
    const description = protocol?.descriptionModule;
    const design = protocol?.designModule;
    const eligibility = protocol?.eligibilityModule;
    const outcomes = protocol?.outcomesModule;
    const conditions = protocol?.conditionsModule?.conditions?.join(", ") || "Not specified";
    
    // Interventions
    const interventions = protocol?.armsInterventionsModule?.interventions
      ?.map((int) => `- ${int.name} (${int.type}): ${int.description || "No description"}`)
      .join("\n") || "Not specified";

    // Primary outcomes
    const primaryOutcomes = outcomes?.primaryOutcomes
      ?.map((o) => `- ${o.measure} (${o.timeFrame})`)
      .join("\n") || "Not specified";

    // Results summary (if available)
    let resultsText = "No results posted yet.";
    if (results) {
      const participants = results.participantFlowModule;
      const baseline = results.baselineCharacteristicsModule;
      const outcomeResults = results.outcomeMeasuresModule?.outcomeMeasures;
      
      resultsText = "RESULTS AVAILABLE:\n";
      
      if (participants?.preAssignmentDetails) {
        resultsText += `   Pre-assignment: ${participants.preAssignmentDetails}\n`;
      }
      
      if (outcomeResults && outcomeResults.length > 0) {
        resultsText += "   Outcome measures:\n";
        outcomeResults.slice(0, 3).forEach((outcome) => {
          resultsText += `   - ${outcome.title}: ${outcome.description || "See full results"}\n`;
        });
      }
      
      if (results.adverseEventsModule) {
        const ae = results.adverseEventsModule;
        resultsText += `   Serious adverse events: ${ae.seriousNumAffected || "Not reported"} participants\n`;
      }
    }

    // Determine if trial succeeded/failed (heuristic based on status and results)
    let outcomeAssessment = "";
    const overallStatus = status?.overallStatus;
    if (overallStatus === "TERMINATED") {
      const whyStopped = status?.whyStopped || status?.whyStoppedText || "Reason not provided";
      outcomeAssessment = `⚠️ TERMINATED: ${whyStopped}`;
    } else if (overallStatus === "COMPLETED" && results) {
      outcomeAssessment = "✓ COMPLETED with results posted";
    } else if (overallStatus === "COMPLETED") {
      outcomeAssessment = "✓ COMPLETED (results not yet posted)";
    } else if (overallStatus === "RECRUITING") {
      outcomeAssessment = "🔄 Currently RECRUITING";
    } else {
      outcomeAssessment = `Status: ${overallStatus}`;
    }

    return {
      content: [
        {
          type: "text",
          text: `Clinical Trial: ${id?.briefTitle || "Untitled"}

NCT ID: ${nctId}
${outcomeAssessment}

Official Title: ${id?.officialTitle || "Not provided"}

Phase: ${design?.phases?.join(", ") || "Not specified"}
Study Type: ${design?.studyType || "Not specified"}
Enrollment: ${design?.enrollmentInfo?.count || "Unknown"} participants

Conditions: ${conditions}

Brief Summary:
${description?.briefSummary || "No summary available"}

Interventions:
${interventions}

Primary Outcome Measures:
${primaryOutcomes}

Eligibility:
   Age: ${eligibility?.minimumAge || "Not specified"} to ${eligibility?.maximumAge || "Not specified"}
   Sex: ${eligibility?.sex || "All"}
   Healthy Volunteers: ${eligibility?.healthyVolunteers || "No"}

Dates:
   Start: ${status?.startDateStruct?.date || "Unknown"}
   Completion: ${status?.completionDateStruct?.date || "Unknown"}

${resultsText}

Link: https://clinicaltrials.gov/study/${nctId}`,
        },
      ],
    };
  }
);
function parsePubmedAuthors(xml) {
  const authorBlocks = [...xml.matchAll(/<Author(?:\s+ValidYN="[^"]+")?>([\s\S]*?)<\/Author>/g)];
  return authorBlocks.map((match) => {
    const block = match[1];
    const foreName = block.match(/<ForeName>([\s\S]*?)<\/ForeName>/)?.[1]?.trim() || "";
    const lastName = block.match(/<LastName>([\s\S]*?)<\/LastName>/)?.[1]?.trim() || "";
    const collectiveName = block.match(/<CollectiveName>([\s\S]*?)<\/CollectiveName>/)?.[1]?.trim() || "";
    const affiliations = [...block.matchAll(/<Affiliation>([\s\S]*?)<\/Affiliation>/g)].map((a) =>
      sanitizeXmlText(a[1])
    );
    const name = sanitizeXmlText(`${foreName} ${lastName}`.trim()) || sanitizeXmlText(collectiveName) || "Unknown";
    return { name, affiliations };
  });
}
// ============================================
// TOOL: Search PubMed
// ============================================
const PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
const PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";
const PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi";

function parsePubmedArticleSummary(xml) {
  const articles = [];
  const docSums = [...xml.matchAll(/<DocSum>([\s\S]*?)<\/DocSum>/g)];
  for (const [, block] of docSums) {
    const pmid = block.match(/<Id>(\d+)<\/Id>/)?.[1] || "";
    const getItem = (name) => {
      const re = new RegExp(`<Item Name="${name}"[^>]*>([\\s\\S]*?)<\\/Item>`);
      return sanitizeXmlText(block.match(re)?.[1] || "");
    };
    const title = getItem("Title");
    const source = getItem("Source");
    const pubDate = getItem("PubDate");
    const authorList = [...block.matchAll(/<Item Name="Author"[^>]*>([\s\S]*?)<\/Item>/g)]
      .map((m) => sanitizeXmlText(m[1]))
      .slice(0, 5);
    const doi = getItem("DOI") || getItem("ELocationID");
    articles.push({ pmid, title, source, pubDate, authors: authorList, doi });
  }
  return articles;
}

server.registerTool(
  "search_pubmed",
  {
    description:
      "Searches PubMed for biomedical literature. Returns PMIDs, titles, authors, journal, and publication dates. Use for finding specific papers, systematic reviews, or evidence for claims.",
    inputSchema: {
      query: z.string().describe("PubMed search query. Supports MeSH terms and boolean operators (AND, OR, NOT)."),
      maxResults: z.number().optional().describe("Max results to return (default 20, max 100)."),
      minDate: z.string().optional().describe("Minimum publication date (YYYY/MM/DD or YYYY)."),
      maxDate: z.string().optional().describe("Maximum publication date (YYYY/MM/DD or YYYY)."),
      sort: z.string().optional().describe("Sort order: 'relevance' (default) or 'date'."),
    },
  },
  async ({ query, maxResults, minDate, maxDate, sort }) => {
    try {
      const retMax = Math.max(1, Math.min(100, Math.round(maxResults || 20)));
      const params = new URLSearchParams({
        db: "pubmed",
        term: query,
        retmax: String(retMax),
        retmode: "json",
        sort: sort === "date" ? "pub+date" : "relevance",
      });
      if (minDate) params.set("mindate", minDate);
      if (maxDate) params.set("maxdate", maxDate);
      if (minDate || maxDate) params.set("datetype", "pdat");

      const searchUrl = `${PUBMED_ESEARCH}?${params.toString()}`;
      const searchData = await fetchJsonWithRetry(searchUrl, { retries: 2, timeoutMs: 10000 });
      const idList = searchData?.esearchresult?.idlist ?? [];
      const totalCount = parseInt(searchData?.esearchresult?.count || "0", 10);

      if (idList.length === 0) {
        return {
          content: [{ type: "text", text: `No PubMed results found for: "${query}"${minDate ? ` (from ${minDate})` : ""}${maxDate ? ` (to ${maxDate})` : ""}. Try broader terms or remove date filters.` }],
        };
      }

      const summaryParams = new URLSearchParams({
        db: "pubmed",
        id: idList.join(","),
        retmode: "xml",
      });
      const summaryUrl = `${PUBMED_ESUMMARY}?${summaryParams.toString()}`;
      const summaryResp = await fetch(summaryUrl);
      const summaryXml = await summaryResp.text();
      const articles = parsePubmedArticleSummary(summaryXml);

      const formatted = articles.map((a, i) => {
        const authorStr = a.authors.length > 0
          ? `${a.authors[0]}${a.authors.length > 1 ? " et al." : ""}`
          : "Unknown";
        const doiStr = a.doi ? ` DOI: ${a.doi}` : "";
        return `${i + 1}. ${a.title}\n   PMID: ${a.pmid} | ${authorStr} | ${a.source} (${a.pubDate})${doiStr}`;
      }).join("\n\n");

      return {
        content: [{
          type: "text",
          text: `PubMed search for "${query}":\nShowing ${articles.length} of ${totalCount} results${minDate ? ` | From: ${minDate}` : ""}${maxDate ? ` | To: ${maxDate}` : ""}\n\n${formatted}\n\nUse get_pubmed_abstract with a PMID for the full abstract.`,
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error searching PubMed: ${error.message}. Try different search terms.` }] };
    }
  }
);

// ============================================
// TOOL: Get PubMed abstract
// ============================================
server.registerTool(
  "get_pubmed_abstract",
  {
    description:
      "Fetches the full abstract and metadata for a PubMed article by PMID. Returns title, authors, journal, abstract text, MeSH terms, and publication type.",
    inputSchema: {
      pmid: z.string().describe("PubMed ID (e.g., '12345678')"),
    },
  },
  async ({ pmid }) => {
    try {
      const cleanPmid = pmid.replace(/^PMID:\s*/i, "").trim();
      const params = new URLSearchParams({
        db: "pubmed",
        id: cleanPmid,
        retmode: "xml",
        rettype: "abstract",
      });
      const url = `${PUBMED_EFETCH}?${params.toString()}`;
      const resp = await fetch(url);
      const xml = await resp.text();

      if (!xml || xml.includes("<ERROR>") || !xml.includes("<PubmedArticle>")) {
        return { content: [{ type: "text", text: `PubMed article not found for PMID: ${cleanPmid}` }] };
      }

      const title = sanitizeXmlText(xml.match(/<ArticleTitle>([\s\S]*?)<\/ArticleTitle>/)?.[1] || "Untitled");
      const abstractParts = [...xml.matchAll(/<AbstractText[^>]*>([\s\S]*?)<\/AbstractText>/g)]
        .map((m) => sanitizeXmlText(m[1]));
      const abstractText = abstractParts.join("\n\n") || "No abstract available.";
      const journal = sanitizeXmlText(xml.match(/<Title>([\s\S]*?)<\/Title>/)?.[1] || "");
      const year = xml.match(/<PubDate>[\s\S]*?<Year>(\d{4})<\/Year>/)?.[1] || "";
      const doi = xml.match(/<ArticleId IdType="doi">([\s\S]*?)<\/ArticleId>/)?.[1]?.trim() || "";
      const pmcId = xml.match(/<ArticleId IdType="pmc">([\s\S]*?)<\/ArticleId>/)?.[1]?.trim() || "";
      const authors = parsePubmedAuthors(xml);
      const authorStr = authors.slice(0, 5).map((a) => a.name).join(", ") + (authors.length > 5 ? " et al." : "");
      const meshTerms = [...xml.matchAll(/<DescriptorName[^>]*>([\s\S]*?)<\/DescriptorName>/g)]
        .map((m) => sanitizeXmlText(m[1]))
        .slice(0, 15);
      const pubTypes = [...xml.matchAll(/<PublicationType[^>]*>([\s\S]*?)<\/PublicationType>/g)]
        .map((m) => sanitizeXmlText(m[1]));

      let text = `Title: ${title}\nPMID: ${cleanPmid}`;
      if (doi) text += ` | DOI: ${doi}`;
      if (pmcId) text += ` | ${pmcId}`;
      text += `\nAuthors: ${authorStr}\nJournal: ${journal} (${year})`;
      if (pubTypes.length) text += `\nType: ${pubTypes.join(", ")}`;
      text += `\n\nAbstract:\n${abstractText}`;
      if (meshTerms.length) text += `\n\nMeSH Terms: ${meshTerms.join("; ")}`;
      text += `\n\nLink: https://pubmed.ncbi.nlm.nih.gov/${cleanPmid}/`;

      return { content: [{ type: "text", text }] };
    } catch (error) {
      return { content: [{ type: "text", text: `Error fetching PubMed abstract: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Advanced PubMed search
// ============================================
server.registerTool(
  "search_pubmed_advanced",
  {
    description:
      "Advanced PubMed search with field-specific queries. Use for precise searches targeting specific fields like author, journal, MeSH terms, or publication type.",
    inputSchema: {
      query: z.string().describe("Full PubMed query with field tags, e.g., '\"BRCA1\"[Title] AND \"breast cancer\"[MeSH] AND review[pt]'"),
      maxResults: z.number().optional().describe("Max results (default 20, max 100)."),
      sort: z.string().optional().describe("Sort: 'relevance' (default) or 'date'."),
    },
  },
  async ({ query, maxResults, sort }) => {
    try {
      const retMax = Math.max(1, Math.min(100, Math.round(maxResults || 20)));
      const params = new URLSearchParams({
        db: "pubmed",
        term: query,
        retmax: String(retMax),
        retmode: "json",
        sort: sort === "date" ? "pub+date" : "relevance",
      });

      const searchUrl = `${PUBMED_ESEARCH}?${params.toString()}`;
      const searchData = await fetchJsonWithRetry(searchUrl, { retries: 2, timeoutMs: 10000 });
      const idList = searchData?.esearchresult?.idlist ?? [];
      const totalCount = parseInt(searchData?.esearchresult?.count || "0", 10);

      if (idList.length === 0) {
        return {
          content: [{ type: "text", text: `No PubMed results for advanced query: ${query}. Check field tags and try broader terms.` }],
        };
      }

      const summaryParams = new URLSearchParams({
        db: "pubmed",
        id: idList.join(","),
        retmode: "xml",
      });
      const summaryUrl = `${PUBMED_ESUMMARY}?${summaryParams.toString()}`;
      const summaryResp = await fetch(summaryUrl);
      const summaryXml = await summaryResp.text();
      const articles = parsePubmedArticleSummary(summaryXml);

      const formatted = articles.map((a, i) => {
        const authorStr = a.authors.length > 0
          ? `${a.authors[0]}${a.authors.length > 1 ? " et al." : ""}`
          : "Unknown";
        const doiStr = a.doi ? ` DOI: ${a.doi}` : "";
        return `${i + 1}. ${a.title}\n   PMID: ${a.pmid} | ${authorStr} | ${a.source} (${a.pubDate})${doiStr}`;
      }).join("\n\n");

      return {
        content: [{
          type: "text",
          text: `PubMed advanced search:\nShowing ${articles.length} of ${totalCount} results\nQuery: ${query}\n\n${formatted}\n\nUse get_pubmed_abstract with a PMID for the full abstract.`,
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in advanced PubMed search: ${error.message}` }] };
    }
  }
);

// TOOL 17: OpenAlex works search
// ============================================
server.registerTool(
  "search_openalex_works",
  {
    description: "Search OpenAlex works for broader literature coverage and metadata.",
    inputSchema: {
      query: z.string(),
      fromYear: z.number().optional(),
      toYear: z.number().optional(),
      limit: z.number().optional().default(10),
    },
  },
  async ({ query, fromYear, toYear, limit = 10 }) => {
    try {
      const filters = [];
      if (fromYear) filters.push(`from_publication_date:${fromYear}-01-01`);
      if (toYear) filters.push(`to_publication_date:${toYear}-12-31`);
      const boundedLimit = Math.max(1, Math.min(50, Math.round(limit)));
      const params = new URLSearchParams({
        search: query,
        per_page: String(boundedLimit),
        select: OPENALEX_WORKS_SELECT,
      });
      if (filters.length) params.set("filter", filters.join(","));
      const url = buildOpenAlexUrl("/works", params);
      const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
      const results = data?.results ?? [];
      const keyFields = results.map((w, idx) => {
        const citation = buildOpenAlexWorkCitation(w);
        return `${idx + 1}. ${citation} | Cited by: ${w.cited_by_count ?? 0}`;
      });
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Retrieved ${results.length} OpenAlex works.`,
              keyFields,
              sources: [url],
              limitations: ["OpenAlex coverage may differ from PubMed indexing for biomedical niche topics."],
            }),
          },
        ],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_openalex_works: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 18: OpenAlex author search
// ============================================
server.registerTool(
  "search_openalex_authors",
  {
    description: "Search OpenAlex authors with affiliation and impact metadata.",
    inputSchema: {
      query: z.string(),
      limit: z.number().optional().default(10),
    },
  },
  async ({ query, limit = 10 }) => {
    try {
      const boundedLimit = Math.max(1, Math.min(50, Math.round(limit)));
      const url = buildOpenAlexUrl(
        "/authors",
        new URLSearchParams({
          search: query,
          per_page: String(boundedLimit),
          select: OPENALEX_AUTHORS_SELECT,
        })
      );
      const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
      const results = data?.results ?? [];
      const keyFields = results.map((a, idx) => {
        const inst = a.last_known_institution?.display_name || "Unknown institution";
        return `${idx + 1}. ${a.display_name || "Unknown"} | Works: ${a.works_count ?? 0} | Cited by: ${a.cited_by_count ?? 0} | Institution: ${inst} | ID: ${a.id}`;
      });
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Retrieved ${results.length} OpenAlex authors.`,
              keyFields,
              sources: [url],
              limitations: ["Author identity may still require manual validation in edge cases."],
            }),
          },
        ],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_openalex_authors: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 19: Rank researchers by activity
// ============================================
server.registerTool(
  "rank_researchers_by_activity",
  {
    description: "Rank researchers by transparent activity score using OpenAlex metadata.",
    inputSchema: {
      query: z.string().describe("Topic query for finding candidate authors"),
      limit: z.number().optional().default(10),
      fromYear: z.number().optional().default(new Date().getUTCFullYear() - 3),
    },
  },
  async ({ query, limit = 10, fromYear }) => {
    try {
      const boundedLimit = Math.max(1, Math.min(25, Math.round(limit)));
      const nowYear = new Date().getUTCFullYear();
      const normalizedFromYear = Number.isFinite(Number(fromYear))
        ? Math.max(1990, Math.min(nowYear, Math.round(Number(fromYear))))
        : nowYear - 6;
      const worksPerPage = 100;
      const maxPages = 2;

      const sources = [];
      const byAuthor = new Map();
      let scannedWorks = 0;

      for (let page = 1; page <= maxPages; page++) {
        const filters = [`from_publication_date:${normalizedFromYear}-01-01`, "type:article|review"];
        const params = new URLSearchParams({
          search: query,
          sort: "cited_by_count:desc",
          per_page: String(worksPerPage),
          page: String(page),
          filter: filters.join(","),
          select: OPENALEX_WORKS_SELECT,
        });
        const worksUrl = buildOpenAlexUrl("/works", params);
        const data = await fetchJsonWithRetry(worksUrl, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
        const works = data?.results ?? [];
        if (works.length === 0) {
          break;
        }
        sources.push(worksUrl);
        scannedWorks += works.length;

        for (const work of works) {
          const title = work?.display_name || "Untitled";
          const citedBy = Number(work?.cited_by_count || 0);
          const year = Number(work?.publication_year || 0);
          const authorships = Array.isArray(work?.authorships) ? work.authorships : [];

          for (let idx = 0; idx < authorships.length; idx++) {
            const authorship = authorships[idx];
            const authorId = authorship?.author?.id;
            if (!authorId) continue;
            const authorName = authorship?.author?.display_name || "Unknown";
            const authorRecord = byAuthor.get(authorId) || {
              id: authorId,
              name: authorName,
              topicWorks: 0,
              topicCitations: 0,
              recentTopicWorks: 0,
              firstAuthorWorks: 0,
              lastAuthorWorks: 0,
              activeYears: new Set(),
              institutionCounts: new Map(),
              exampleWorks: [],
            };

            authorRecord.topicWorks += 1;
            authorRecord.topicCitations += citedBy;
            if (year >= nowYear - 3) {
              authorRecord.recentTopicWorks += 1;
            }
            if (idx === 0) {
              authorRecord.firstAuthorWorks += 1;
            }
            if (idx === authorships.length - 1) {
              authorRecord.lastAuthorWorks += 1;
            }
            if (year > 0) {
              authorRecord.activeYears.add(year);
            }
            const institutions = Array.isArray(authorship?.institutions) ? authorship.institutions : [];
            for (const inst of institutions) {
              const instName = inst?.display_name;
              if (!instName) continue;
              authorRecord.institutionCounts.set(instName, (authorRecord.institutionCounts.get(instName) || 0) + 1);
            }
            if (authorRecord.exampleWorks.length < 3) {
              authorRecord.exampleWorks.push({
                title,
                year,
                citedBy,
              });
            }
            byAuthor.set(authorId, authorRecord);
          }
        }

        if (works.length < worksPerPage) {
          break;
        }
      }

      if (byAuthor.size === 0) {
        const rankPayload = buildRankResearchersPayload({
          resultStatus: "not_found_or_empty",
          query,
          fromYear: normalizedFromYear,
          limit: boundedLimit,
          scannedWorks,
          researchers: [],
          notes: ["No topic-matched OpenAlex works were available for ranking."],
        });
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `No OpenAlex works matched query "${query}" for topic-based researcher ranking.`,
                keyFields: [`From year: ${normalizedFromYear}`],
                sources: sources.length > 0 ? sources : [buildOpenAlexUrl("/works", new URLSearchParams({ search: query }))],
                limitations: ["Try broader disease/topic terms or an earlier fromYear."],
              }),
            },
          ],
          structuredContent: rankPayload,
        };
      }

      const scored = Array.from(byAuthor.values())
        .map((author) => {
          const leadership = author.firstAuthorWorks + author.lastAuthorWorks;
          const activeYears = author.activeYears.size;
          const score =
            0.45 * Math.log1p(author.topicWorks) +
            0.35 * Math.log1p(author.topicCitations) +
            0.12 * Math.log1p(author.recentTopicWorks) +
            0.08 * Math.log1p(leadership) +
            0.05 * Math.log1p(activeYears);
          const topInstitution =
            Array.from(author.institutionCounts.entries()).sort((a, b) => b[1] - a[1])[0]?.[0] || "Unknown institution";
          return {
            id: author.id,
            name: author.name,
            score,
            topicWorks: author.topicWorks,
            topicCitations: author.topicCitations,
            recentTopicWorks: author.recentTopicWorks,
            leadership,
            activeYears,
            institution: topInstitution,
            exampleWorks: author.exampleWorks,
          };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, boundedLimit);

      const keyFields = scored.map((r, idx) => {
        const examples = r.exampleWorks
          .slice(0, 2)
          .map((work) => `${work.year || "n/a"}:${work.citedBy}c "${String(work.title).slice(0, 70)}"`)
          .join(" | ");
        return `${idx + 1}. ${r.name} | Activity score: ${r.score.toFixed(2)} | Topic works since ${normalizedFromYear}: ${r.topicWorks} | Topic citations: ${r.topicCitations} | Recent works (last 3y): ${r.recentTopicWorks} | Leadership (first/last): ${r.leadership} | Active years: ${r.activeYears} | Institution: ${r.institution} | ID: ${r.id}${examples ? ` | Example works: ${examples}` : ""}`;
      });
      const researchersPayload = scored.map((r, idx) => ({
        rank: idx + 1,
        author_id: String(r.id || ""),
        name: String(r.name || "Unknown"),
        institution: String(r.institution || "Unknown institution"),
        activity_score: toFiniteNumber(r.score, 0),
        topic_works: toNonNegativeInt(r.topicWorks),
        topic_citations: toNonNegativeInt(r.topicCitations),
        recent_topic_works: toNonNegativeInt(r.recentTopicWorks),
        leadership_works: toNonNegativeInt(r.leadership),
        active_years: toNonNegativeInt(r.activeYears),
        example_works: (r.exampleWorks || []).slice(0, 3).map((work) => ({
          title: String(work?.title || "Untitled"),
          year: Number.isFinite(Number(work?.year)) && Number(work?.year) > 0 ? Math.trunc(Number(work.year)) : null,
          cited_by: toNonNegativeInt(work?.citedBy),
        })),
      }));
      const rankPayload = buildRankResearchersPayload({
        resultStatus: "ok",
        query,
        fromYear: normalizedFromYear,
        limit: boundedLimit,
        scannedWorks,
        researchers: researchersPayload,
        notes: [`Scanned up to ${maxPages} OpenAlex works pages.`],
      });
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Ranked ${scored.length} researchers by topic-specific activity for query "${query}" using OpenAlex works.`,
              keyFields,
              sources,
              limitations: [
                "Score is heuristic and based on topic-matched OpenAlex works; it does not directly model field-wide esteem.",
                "Lexical topic matching can include adjacent subdomains for broad terms.",
                `Scanned up to ${maxPages} OpenAlex works pages (${scannedWorks} works).`,
              ],
            }),
          },
        ],
        structuredContent: rankPayload,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in rank_researchers_by_activity: ${error.message}` }],
        structuredContent: buildRankResearchersPayload({
          resultStatus: "error",
          query,
          fromYear,
          limit,
          scannedWorks: 0,
          researchers: [],
          notes: ["Ranking execution failed before completion."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 20: Researcher contact candidates
// ============================================
server.registerTool(
  "get_researcher_contact_candidates",
  {
    description: "Return contact candidates for a researcher (institution pages, ORCID/profile links when available).",
    inputSchema: {
      authorName: z.string(),
      authorId: z.string().optional().describe("Optional OpenAlex author ID (e.g., https://openalex.org/A123...)"),
      limit: z.number().optional().default(5),
    },
  },
  async ({ authorName, authorId, limit = 5 }) => {
    try {
      let url = "";
      let results = [];
      if (authorId) {
        url = buildOpenAlexUrl(
          `/authors/${encodeURIComponent(authorId.replace("https://openalex.org/", ""))}`,
          new URLSearchParams({ select: OPENALEX_AUTHORS_SELECT })
        );
        const author = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
        results = author ? [author] : [];
      } else {
        const boundedLimit = Math.max(1, Math.min(20, Math.round(limit)));
        url = buildOpenAlexUrl(
          "/authors",
          new URLSearchParams({
            search: authorName,
            per_page: String(boundedLimit),
            select: OPENALEX_AUTHORS_SELECT,
          })
        );
        const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
        results = data?.results ?? [];
      }
      const keyFields = [];
      for (let idx = 0; idx < results.length; idx++) {
        const a = results[idx];
        let inst = a.last_known_institution?.display_name || "";
        let homepage = a.last_known_institution?.homepage_url || "N/A";
        let instSource = "author.last_known_institution";

        if (!inst && a.id) {
          // Fallback: derive institution from the author's latest work authorship record.
          const worksUrl = buildOpenAlexUrl(
            "/works",
            new URLSearchParams({
              filter: `author.id:${a.id}`,
              sort: "publication_date:desc",
              per_page: "1",
              select: OPENALEX_WORKS_SELECT,
            })
          );
          try {
            const workData = await fetchJsonWithRetry(worksUrl, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
            const latestWork = workData?.results?.[0];
            const authorship = (latestWork?.authorships || []).find((au) => au?.author?.id === a.id);
            const institutions = (authorship?.institutions || []).map((i) => i.display_name).filter(Boolean);
            const homepages = (authorship?.institutions || []).map((i) => i.homepage_url).filter(Boolean);
            if (institutions.length > 0) {
              inst = institutions.join("; ");
              instSource = "latest_work.authorships.institutions";
            }
            if (homepages.length > 0) {
              homepage = homepages[0];
            }
          } catch (error) {
            // Ignore fallback failure and continue with available metadata.
          }
        }

        if (!inst) {
          inst = "Unknown institution";
          instSource = "not_available";
        }
        const orcid = a.orcid || "N/A";
        const profile = a.id || "N/A";
        const confidence = orcid !== "N/A" ? "high" : inst !== "Unknown institution" ? "medium" : "low";
        keyFields.push(
          `${idx + 1}. ${a.display_name || "Unknown"} | Institution: ${inst} | Institution source: ${instSource} | ORCID: ${orcid} | Institution homepage: ${homepage} | OpenAlex profile: ${profile} | Contact confidence: ${confidence}`
        );
      }
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Contact candidates for "${authorName}".`,
              keyFields,
              sources: [url],
              limitations: [
                "Direct email is often unavailable; contact signals are inferred from public profiles.",
                "Institution fallback from latest work may not reflect current affiliation in all cases.",
              ],
            }),
          },
        ],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_researcher_contact_candidates: ${error.message}` }] };
    }
  }
);
// ============================================
// TOOL 24: Search Reactome pathways
// ============================================
server.registerTool(
  "search_reactome_pathways",
  {
    description: "Search Reactome pathways by topic or gene keyword.",
    inputSchema: {
      query: z.string(),
      species: z.string().optional().default("Homo sapiens"),
      limit: z.number().optional().default(10),
    },
  },
  async ({ query, species = "Homo sapiens", limit = 10 }) => {
    try {
      const url = `${REACTOME_API}/search/query?${new URLSearchParams({
        query,
        species,
        types: "Pathway",
        cluster: "true",
      }).toString()}`;
      const data = await fetchJsonWithRetry(url);
      const entries = data?.results?.[0]?.entries || data?.entries || [];
      const sliced = entries.slice(0, limit);
      const keyFields = sliced.map((e, idx) => `${idx + 1}. ${e.name || "Unnamed pathway"} | Stable ID: ${e.stId || "N/A"} | Species: ${e.species?.[0]?.displayName || species}`);
      return { content: [{ type: "text", text: renderStructuredResponse({ summary: `Retrieved ${sliced.length} Reactome pathway hits.`, keyFields, sources: [url], limitations: ["Reactome search ranking may include broad pathways; validate specificity downstream."] }) }] };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_reactome_pathways: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 25: STRING interactions
// ============================================
server.registerTool(
  "get_string_interactions",
  {
    description: "Get STRING protein-protein interactions for a gene/protein identifier.",
    inputSchema: {
      identifier: z.string().describe("Gene/protein name"),
      species: z.number().optional().default(9606),
      requiredScore: z.number().optional().default(700),
      limit: z.number().optional().default(20),
    },
  },
  async ({ identifier, species = 9606, requiredScore = 700, limit = 20 }) => {
    try {
      const url = `${STRING_API}/json/network?${new URLSearchParams({
        identifiers: identifier,
        species: String(species),
        required_score: String(requiredScore),
      }).toString()}`;
      const rows = await fetchJsonWithRetry(url);
      const sliced = (rows || []).slice(0, limit);
      const keyFields = sliced.map((r, idx) => {
        const score = r.score ?? r.combined_score ?? "N/A";
        return `${idx + 1}. ${r.preferredName_A || r.stringId_A || "A"} -> ${r.preferredName_B || r.stringId_B || "B"} | Score: ${score}`;
      });
      return { content: [{ type: "text", text: renderStructuredResponse({ summary: `Retrieved ${sliced.length} STRING interactions for ${identifier}.`, keyFields, sources: [url], limitations: ["STRING interactions combine evidence channels; high score does not guarantee direct causal interaction."] }) }] };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_string_interactions: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 26: Summarize clinical-trial landscape
// ============================================
server.registerTool(
  "summarize_clinical_trials_landscape",
  {
    description:
      "Summarizes the clinical-trial landscape for a query with status/phase distributions, top interventions, and termination reasons.",
    inputSchema: {
      query: z.string().describe("Disease/target/drug query (e.g., 'LRRK2 Parkinson')"),
      status: z
        .string()
        .optional()
        .describe("Optional status filter (e.g., RECRUITING, COMPLETED, TERMINATED)."),
      maxStudies: z.number().optional().default(60).describe("Max studies to aggregate (10-200)."),
      maxPages: z.number().optional().default(4).describe("Max pages to scan from ClinicalTrials.gov (1-8)."),
    },
  },
  async ({ query, status, maxStudies = 60, maxPages = 4 }) => {
    try {
      const boundedStudies = Math.max(10, Math.min(200, Math.round(maxStudies)));
      const boundedPages = Math.max(1, Math.min(8, Math.round(maxPages)));
      const studies = [];
      const sources = [];
      let totalCount = null;
      let nextPageToken = "";

      for (let page = 0; page < boundedPages && studies.length < boundedStudies; page++) {
        const pageSize = Math.min(100, boundedStudies - studies.length);
        const params = new URLSearchParams({
          "query.term": query,
          pageSize: String(pageSize),
          format: "json",
        });
        if (status) {
          params.append("filter.overallStatus", status);
        }
        if (nextPageToken) {
          params.append("pageToken", nextPageToken);
        }

        const url = `${CLINICAL_TRIALS_API}/studies?${params.toString()}`;
        sources.push(url);
        const data = await fetchJsonWithRetry(url);
        const pageStudies = data?.studies ?? [];
        if (Number.isFinite(data?.totalCount)) {
          totalCount = data.totalCount;
        }
        studies.push(...pageStudies);
        nextPageToken = data?.nextPageToken || "";
        if (!nextPageToken || pageStudies.length === 0) {
          break;
        }
      }

      if (studies.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "No clinical trials found for landscape summary.",
                keyFields: [`Query: ${query}`, `Status filter: ${status || "none"}`],
                sources: sources.length > 0 ? sources : [`${CLINICAL_TRIALS_API}/studies`],
                limitations: ["Try broader terms or remove status filters."],
              }),
            },
          ],
          structuredContent: buildClinicalTrialsLandscapePayload({
            resultStatus: "not_found_or_empty",
            query,
            statusFilter: status || "",
            studiesAnalyzed: 0,
            totalReported: totalCount,
            maxStudies: boundedStudies,
            maxPages: boundedPages,
            hasMorePages: false,
            trialsWithPostedResults: 0,
            statusBreakdown: [],
            phaseBreakdown: [],
            topInterventions: [],
            topConditions: [],
            topTerminationReasons: [],
            exampleTerminatedNctIds: [],
            notes: ["No studies were returned for the query/filter combination."],
          }),
        };
      }

      const statusCounts = new Map();
      const phaseCounts = new Map();
      const interventionCounts = new Map();
      const conditionCounts = new Map();
      const terminationReasonCounts = new Map();
      const terminatedNctIds = [];
      const terminatedLike = new Set(["TERMINATED", "SUSPENDED", "WITHDRAWN"]);
      let withPostedResults = 0;

      for (const study of studies) {
        const protocol = study?.protocolSection || {};
        const statusModule = protocol?.statusModule || {};
        const designModule = protocol?.designModule || {};
        const interventions = protocol?.armsInterventionsModule?.interventions || [];
        const conditions = protocol?.conditionsModule?.conditions || [];
        const nctId = protocol?.identificationModule?.nctId || "Unknown";
        const trialStatus = statusModule?.overallStatus || "Unknown";

        incrementCount(statusCounts, trialStatus);
        if (study?.resultsSection) {
          withPostedResults += 1;
        }

        const phases = designModule?.phases || [];
        if (phases.length === 0) {
          incrementCount(phaseCounts, "Not specified");
        } else {
          for (const phase of phases) {
            incrementCount(phaseCounts, phase);
          }
        }

        for (const intervention of interventions.slice(0, 3)) {
          incrementCount(interventionCounts, intervention?.name || "Unnamed intervention");
        }
        for (const condition of conditions.slice(0, 3)) {
          incrementCount(conditionCounts, condition);
        }

        if (terminatedLike.has(trialStatus)) {
          if (terminatedNctIds.length < 5 && nctId !== "Unknown") {
            terminatedNctIds.push(nctId);
          }
          incrementCount(terminationReasonCounts, sanitizeReasonText(statusModule?.whyStopped));
        }
      }

      const statusSummary = summarizeTopCounts(statusCounts, 8);
      const phaseSummary = summarizeTopCounts(phaseCounts, 6);
      const interventionSummary = summarizeTopCounts(interventionCounts, 6);
      const conditionSummary = summarizeTopCounts(conditionCounts, 6);
      const reasonSummary = summarizeTopCounts(terminationReasonCounts, 5);
      const analyzed = studies.length;
      const hasMore = Boolean(nextPageToken);

      const keyFields = [
        `Query: ${query}`,
        `Studies analyzed: ${analyzed}${Number.isFinite(totalCount) ? ` (reported total: ${totalCount})` : ""}`,
        `Status breakdown: ${statusSummary.join(", ") || "N/A"}`,
        `Phase breakdown: ${phaseSummary.join(", ") || "N/A"}`,
        `Trials with posted results: ${withPostedResults}/${analyzed}`,
        `Top interventions: ${interventionSummary.join(", ") || "N/A"}`,
        `Top conditions: ${conditionSummary.join(", ") || "N/A"}`,
      ];

      if (reasonSummary.length > 0) {
        keyFields.push(`Common termination/suspension reasons: ${reasonSummary.join(" | ")}`);
      }
      if (terminatedNctIds.length > 0) {
        keyFields.push(`Example terminated/suspended/withdrawn trials: ${terminatedNctIds.join(", ")}`);
      }
      if (hasMore) {
        keyFields.push("Additional studies remain beyond current page/study limits.");
      }
      const clinicalLandscapePayload = buildClinicalTrialsLandscapePayload({
        resultStatus: "ok",
        query,
        statusFilter: status || "",
        studiesAnalyzed: analyzed,
        totalReported: totalCount,
        maxStudies: boundedStudies,
        maxPages: boundedPages,
        hasMorePages: hasMore,
        trialsWithPostedResults: withPostedResults,
        statusBreakdown: statusSummary,
        phaseBreakdown: phaseSummary,
        topInterventions: interventionSummary,
        topConditions: conditionSummary,
        topTerminationReasons: reasonSummary,
        exampleTerminatedNctIds: terminatedNctIds,
        notes: [
          "Counts are aggregated from scanned ClinicalTrials.gov studies.",
          hasMore ? "Additional studies remain outside scanned pages." : "All fetched pages were processed within limits.",
        ],
      });

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Clinical-trial landscape summary generated for "${query}".`,
              keyFields,
              sources,
              limitations: [
                "Landscape is limited to scanned pages/studies and may not include the full registry.",
                "Termination reasons are free text and may need manual normalization.",
              ],
            }),
          },
        ],
        structuredContent: clinicalLandscapePayload,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_clinical_trials_landscape: ${error.message}` }],
        structuredContent: buildClinicalTrialsLandscapePayload({
          resultStatus: "error",
          query,
          statusFilter: status || "",
          studiesAnalyzed: 0,
          totalReported: null,
          maxStudies,
          maxPages,
          hasMorePages: false,
          trialsWithPostedResults: 0,
          statusBreakdown: [],
          phaseBreakdown: [],
          topInterventions: [],
          topConditions: [],
          topTerminationReasons: [],
          exampleTerminatedNctIds: [],
          notes: ["Unexpected error during clinical-trial landscape aggregation."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);
// ============================================
// TOOL 34: UniProt protein search
// ============================================
server.registerTool(
  "search_uniprot_proteins",
  {
    description:
      "Search UniProtKB for protein entries by gene/protein text with optional species and reviewed-only filters.",
    inputSchema: {
      query: z.string().describe("Gene/protein query text (e.g., EGFR, TYK2, interleukin receptor)"),
      organismTaxId: z.number().optional().default(9606).describe("NCBI taxonomy ID filter (default: 9606 for human)"),
      reviewedOnly: z.boolean().optional().default(true).describe("If true, return Swiss-Prot reviewed entries only"),
      limit: z.number().optional().default(10).describe("Max number of UniProt entries to return"),
    },
  },
  async ({ query, organismTaxId = 9606, reviewedOnly = true, limit = 10 }) => {
    try {
      const normalizedQuery = normalizeWhitespace(query || "");
      if (!normalizedQuery) {
        return { content: [{ type: "text", text: "Provide a non-empty UniProt query string." }] };
      }
      const boundedLimit = Math.max(1, Math.min(25, Math.round(limit)));
      const boundedTaxId = Math.max(0, toNonNegativeInt(organismTaxId, 9606));
      const clauses = [`(${normalizedQuery})`];
      if (boundedTaxId > 0) clauses.push(`organism_id:${boundedTaxId}`);
      if (reviewedOnly) clauses.push("reviewed:true");
      const term = clauses.join(" AND ");

      const params = new URLSearchParams({
        query: term,
        format: "json",
        size: String(boundedLimit),
        fields: "accession,id,protein_name,gene_names,organism_name,length,reviewed,xref_ensembl",
      });
      const url = buildUniProtUrl("/uniprotkb/search", params);
      const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
      const results = Array.isArray(data?.results) ? data.results : [];

      if (results.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "No UniProt records matched this query.",
                keyFields: [
                  `Query: ${normalizedQuery}`,
                  `Taxonomy filter: ${boundedTaxId > 0 ? boundedTaxId : "none"}`,
                  `Reviewed-only: ${reviewedOnly ? "yes" : "no"}`,
                ],
                sources: [url],
                limitations: ["Try broader query terms or disable reviewed-only mode for exploratory searches."],
              }),
            },
          ],
        };
      }

      const keyFields = results.map((entry, idx) => {
        const accession = normalizeWhitespace(entry?.primaryAccession || "");
        const accessionLink = accession ? `[${accession}](https://www.uniprot.org/uniprotkb/${accession})` : "N/A";
        const proteinName = extractUniProtProteinName(entry);
        const symbols = extractUniProtGeneSymbols(entry, 4);
        const geneLabel = symbols.length > 0 ? symbols.join(", ") : "No gene symbol";
        const organism =
          normalizeWhitespace(entry?.organism?.scientificName || entry?.organism?.commonName || "") || "Unknown organism";
        const length = toNonNegativeInt(entry?.sequence?.length);
        const ensemblGenes = extractUniProtEnsemblGeneIds(entry, 3);
        const reviewedLabel = normalizeWhitespace(entry?.entryType || "").toLowerCase().includes("reviewed")
          ? "reviewed"
          : "unreviewed";
        return `${idx + 1}. ${geneLabel} | ${proteinName} | Accession: ${accessionLink} (${reviewedLabel}) | Length: ${
          length > 0 ? `${length} aa` : "unknown"
        } | Organism: ${organism}${ensemblGenes.length > 0 ? ` | Ensembl: ${ensemblGenes.join(", ")}` : ""}`;
      });

      const sourceLinks = [
        url,
        ...results
          .slice(0, 4)
          .map((entry) => normalizeWhitespace(entry?.primaryAccession || ""))
          .filter(Boolean)
          .map((accession) => `https://www.uniprot.org/uniprotkb/${accession}`),
      ];

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Retrieved ${results.length} UniProt protein record${results.length === 1 ? "" : "s"}.`,
              keyFields,
              sources: dedupeArray(sourceLinks),
              limitations: [
                "Search matches can include close homologs and alias terms; validate accession and organism before downstream use.",
                "UniProt entry coverage varies by species and curation level (reviewed vs unreviewed).",
              ],
            }),
          },
        ],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_uniprot_proteins: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 35: UniProt protein profile
// ============================================
server.registerTool(
  "get_uniprot_protein_profile",
  {
    description:
      "Get a concise UniProt protein profile by accession, including function, localization, sequence facts, and major cross-references.",
    inputSchema: {
      accession: z.string().describe("UniProt accession (e.g., P00533 for human EGFR)"),
    },
  },
  async ({ accession }) => {
    const normalizedAccession = normalizeWhitespace(accession || "").toUpperCase();
    if (!normalizedAccession) {
      return { content: [{ type: "text", text: "Provide a UniProt accession (for example, P00533)." }] };
    }

    const url = buildUniProtUrl(`/uniprotkb/${encodeURIComponent(normalizedAccession)}.json`);
    try {
      const record = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
      const primaryAccession = normalizeWhitespace(record?.primaryAccession || normalizedAccession);
      const profileUrl = `https://www.uniprot.org/uniprotkb/${primaryAccession}`;
      const entryId = normalizeWhitespace(record?.uniProtkbId || "");
      const reviewedLabel = normalizeWhitespace(record?.entryType || "").toLowerCase().includes("reviewed")
        ? "reviewed (Swiss-Prot)"
        : "unreviewed (TrEMBL)";
      const proteinName = extractUniProtProteinName(record);
      const genes = extractUniProtGeneSymbols(record, 10);
      const organism =
        normalizeWhitespace(record?.organism?.scientificName || record?.organism?.commonName || "") || "Unknown organism";
      const taxonId = toNonNegativeInt(record?.organism?.taxonId);
      const sequenceLength = toNonNegativeInt(record?.sequence?.length);
      const sequenceMassDa = toNonNegativeInt(record?.sequence?.molWeight);
      const proteinExistence = normalizeWhitespace(record?.proteinExistence || "Unknown");
      const annotationScore = toFiniteNumber(record?.annotationScore, Number.NaN);
      const functionHighlights = extractUniProtCommentTexts(record, "FUNCTION", 2, 320);
      const subcellularLocations = extractUniProtSubcellularLocations(record, 8);
      const diseaseNotes = extractUniProtCommentTexts(record, "DISEASE", 3, 220);
      const ensemblGeneIds = extractUniProtEnsemblGeneIds(record, 8);
      const pdbIds = extractUniProtCrossRefs(record, "PDB", 8);
      const reactomeIds = extractUniProtCrossRefs(record, "Reactome", 8);
      const featureSummary = summarizeUniProtFeatureTypes(record, 10);
      const lastUpdate = normalizeWhitespace(record?.entryAudit?.lastAnnotationUpdateDate || "");

      const keyFields = [
        `Accession: [${primaryAccession}](${profileUrl})`,
        ...(entryId ? [`UniProt ID: ${entryId}`] : []),
        `Review status: ${reviewedLabel}`,
        `Protein: ${proteinName}`,
        `Genes: ${genes.length > 0 ? genes.join(", ") : "N/A"}`,
        `Organism: ${organism}${taxonId > 0 ? ` (taxon ${taxonId})` : ""}`,
        `Protein existence evidence: ${proteinExistence}`,
        `Sequence: ${sequenceLength > 0 ? `${sequenceLength} aa` : "unknown length"}${sequenceMassDa > 0 ? `, ${sequenceMassDa} Da` : ""}`,
        `Annotation score: ${Number.isFinite(annotationScore) ? annotationScore.toFixed(1) : "N/A"}`,
        `Subcellular locations: ${subcellularLocations.length > 0 ? subcellularLocations.join(", ") : "N/A"}`,
        `Ensembl gene IDs: ${ensemblGeneIds.length > 0 ? ensemblGeneIds.join(", ") : "N/A"}`,
        `PDB cross-references: ${pdbIds.length > 0 ? pdbIds.join(", ") : "N/A"}`,
        `Reactome cross-references: ${reactomeIds.length > 0 ? reactomeIds.join(", ") : "N/A"}`,
        `Feature type summary: ${featureSummary.length > 0 ? featureSummary.join("; ") : "N/A"}`,
        ...(functionHighlights.length > 0 ? [`Function highlights: ${functionHighlights.join(" | ")}`] : []),
        ...(diseaseNotes.length > 0 ? [`Disease notes: ${diseaseNotes.join(" | ")}`] : []),
        ...(lastUpdate ? [`Last annotation update: ${lastUpdate}`] : []),
      ];

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Retrieved UniProt profile for ${primaryAccession}${entryId ? ` (${entryId})` : ""}.`,
              keyFields,
              sources: [url, profileUrl],
              limitations: [
                "UniProt captures protein-centric evidence; disease-level interpretation still needs clinical/genetic triangulation.",
                "Some sections (for example disease notes) are absent for many proteins and species.",
              ],
            }),
          },
        ],
      };
    } catch (error) {
      const message = String(error?.message || "unknown error");
      if (message.includes("Request failed (404)")) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `UniProt accession ${normalizedAccession} was not found.`,
                keyFields: [`Requested accession: ${normalizedAccession}`],
                sources: [url],
                limitations: [
                  "Check for typos and ensure you are using a UniProt accession (for example P00533), not a gene symbol.",
                ],
              }),
            },
          ],
        };
      }
      return { content: [{ type: "text", text: `Error in get_uniprot_protein_profile: ${message}` }] };
    }
  }
);

// ---------------------------------------------------------------------------
// Ensembl VEP — variant effect predictions (SIFT, PolyPhen, AlphaMissense)
// ---------------------------------------------------------------------------

server.registerTool(
  "annotate_variants_vep",
  {
    description:
      "Predict functional effects of human variants using Ensembl VEP. Returns consequence type, SIFT, PolyPhen, " +
      "and AlphaMissense predictions. Use for assessing pathogenicity of specific mutations.",
    inputSchema: {
      variants: z
        .array(z.string())
        .min(1)
        .max(10)
        .describe('HGVS notation variants, e.g. ["BRAF:p.Val600Glu", "chr7:g.140753336A>T"]. Max 10.'),
      includeAlphaMissense: z.boolean().optional().default(true).describe("Include AlphaMissense predictions"),
    },
  },
  async ({ variants, includeAlphaMissense = true }) => {
    const cleaned = (variants || []).map((v) => normalizeWhitespace(v)).filter(Boolean).slice(0, 10);
    if (cleaned.length === 0) {
      return { content: [{ type: "text", text: "Provide at least one variant in HGVS notation." }] };
    }

    const body = { hgvs_notations: cleaned };
    if (includeAlphaMissense) body.AlphaMissense = 1;

    const url = `${ENSEMBL_REST_API}/vep/human/hgvs`;
    const data = await fetchJsonWithRetry(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      retries: 1,
      timeoutMs: 20000,
      maxBackoffMs: 3000,
    });

    const results = Array.isArray(data) ? data : [];
    if (results.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: "No VEP results returned for the submitted variants.",
              keyFields: cleaned.map((v) => `Input: ${v}`),
              sources: [url],
              limitations: ["Check HGVS notation format. Accepted examples: BRAF:p.Val600Glu, chr7:g.140753336A>T, 9:g.22125504G>C"],
            }),
          },
        ],
      };
    }

    const keyFields = results.map((entry) => {
      const input = normalizeWhitespace(entry?.input || entry?.id || "");
      const consequences = (entry?.most_severe_consequence || "").replace(/_/g, " ");
      const transcripts = Array.isArray(entry?.transcript_consequences) ? entry.transcript_consequences : [];
      const top = transcripts[0] || {};

      const gene = normalizeWhitespace(top.gene_symbol || "");
      const siftPred = normalizeWhitespace(top.sift_prediction || "");
      const siftScore = top.sift_score != null ? Number(top.sift_score).toFixed(3) : "";
      const sift = siftPred ? `${siftPred}${siftScore ? ` (${siftScore})` : ""}` : "N/A";

      const ppPred = normalizeWhitespace(top.polyphen_prediction || "");
      const ppScore = top.polyphen_score != null ? Number(top.polyphen_score).toFixed(3) : "";
      const polyphen = ppPred ? `${ppPred}${ppScore ? ` (${ppScore})` : ""}` : "N/A";

      let alphaMissense = "N/A";
      if (top.am_class) {
        const amScore = top.am_pathogenicity != null ? Number(top.am_pathogenicity).toFixed(3) : "";
        alphaMissense = `${normalizeWhitespace(top.am_class)}${amScore ? ` (${amScore})` : ""}`;
      }

      const parts = [`Input: ${input}`];
      if (gene) parts.push(`Gene: ${gene}`);
      parts.push(`Consequence: ${consequences}`);
      parts.push(`SIFT: ${sift}`);
      parts.push(`PolyPhen: ${polyphen}`);
      if (includeAlphaMissense) parts.push(`AlphaMissense: ${alphaMissense}`);
      return parts.join(" | ");
    });

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: `VEP returned predictions for ${results.length} variant${results.length === 1 ? "" : "s"}.`,
            keyFields,
            sources: [url, "https://www.ensembl.org/info/docs/tools/vep/index.html"],
            limitations: [
              "SIFT/PolyPhen scores are only available for missense variants in protein-coding transcripts.",
              "AlphaMissense coverage may not include all missense variants.",
            ],
          }),
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// CIViC — clinical variant interpretations
// ---------------------------------------------------------------------------

async function fetchCivicGraphQL(query, variables = {}) {
  const payload = await fetchJsonWithRetry(CIVIC_GRAPHQL_API, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ query, variables }),
    retries: 1,
    timeoutMs: 15000,
    maxBackoffMs: 3000,
  });
  return requireGraphQLData(payload, "CIViC");
}

server.registerTool(
  "search_civic_variants",
  {
    description:
      "Search CIViC for clinical variant interpretations by gene name with optional variant filter. " +
      "Returns evidence levels, clinical significance, associated therapies, and diseases. " +
      "Use for understanding clinical actionability of cancer variants.",
    inputSchema: {
      gene: z.string().describe("Gene symbol (e.g. BRAF, EGFR, KRAS)"),
      variantName: z.string().optional().describe("Optional variant name filter (e.g. V600E)"),
      limit: z.number().optional().default(10).describe("Max variants to return (default 10, max 25)"),
    },
  },
  async ({ gene, variantName, limit = 10 }) => {
    const normalizedGene = normalizeWhitespace(gene || "").toUpperCase();
    if (!normalizedGene) {
      return { content: [{ type: "text", text: "Provide a gene symbol (e.g. BRAF)." }] };
    }
    const boundedLimit = Math.max(1, Math.min(25, Math.round(limit)));

    const query = `
      query SearchVariants($geneName: String!, $first: Int) {
        variants(
          geneNames: [$geneName]
          first: $first
          sortBy: { column: evidenceItemCount, direction: DESC }
        ) {
          totalCount
          nodes {
            id
            name
            feature {
              name
            }
            molecularProfiles {
              nodes {
                id
                name
                evidenceCountByClinicalSignificance
                evidenceItems(first: 5) {
                  nodes {
                    id
                    status
                    evidenceLevel
                    evidenceDirection
                    significance
                    therapies { name }
                    disease { name }
                    source { citation }
                  }
                }
              }
            }
          }
        }
      }
    `;

    const data = await fetchCivicGraphQL(query, { geneName: normalizedGene, first: boundedLimit });
    const variants = data?.variants?.nodes || [];
    const totalCount = data?.variants?.totalCount || 0;

    const filtered = variantName
      ? variants.filter((v) => normalizeWhitespace(v.name || "").toLowerCase().includes(normalizeWhitespace(variantName).toLowerCase()))
      : variants;

    if (filtered.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: variantName
                ? `No CIViC entries for ${normalizedGene} ${variantName}.`
                : `No CIViC entries for gene ${normalizedGene}.`,
              keyFields: [`Gene: ${normalizedGene}`, `Total CIViC variants: ${totalCount}`],
              sources: [`https://civicdb.org/genes/${normalizedGene}`],
              limitations: ["CIViC focuses on cancer variants with clinical evidence; not all known variants are curated."],
            }),
          },
        ],
      };
    }

    const keyFields = filtered.map((v) => {
      const vName = normalizeWhitespace(v.name || "");
      const profiles = v.molecularProfiles?.nodes || [];
      const evidenceItems = profiles.flatMap((p) => p.evidenceItems?.nodes || []);
      const accepted = evidenceItems.filter((e) => normalizeWhitespace(e.status || "").toLowerCase() === "accepted");

      const levels = accepted.map((e) => normalizeWhitespace(e.evidenceLevel || "")).filter(Boolean);
      const levelSummary = levels.length > 0 ? [...new Set(levels)].join(", ") : "none";
      const significances = [...new Set(accepted.map((e) => normalizeWhitespace(e.significance || "")).filter(Boolean))];
      const diseases = [...new Set(accepted.map((e) => normalizeWhitespace(e.disease?.name || "")).filter(Boolean))].slice(0, 3);
      const therapies = [...new Set(accepted.flatMap((e) => (e.therapies || []).map((t) => normalizeWhitespace(t.name || ""))).filter(Boolean))].slice(0, 3);

      const parts = [`${normalizedGene} ${vName}`];
      parts.push(`Evidence levels: ${levelSummary}`);
      if (significances.length > 0) parts.push(`Significance: ${significances.join(", ")}`);
      if (diseases.length > 0) parts.push(`Diseases: ${diseases.join(", ")}`);
      if (therapies.length > 0) parts.push(`Therapies: ${therapies.join(", ")}`);
      parts.push(`Evidence items: ${accepted.length} accepted`);
      return parts.join(" | ");
    });

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: `Found ${filtered.length} CIViC variant${filtered.length === 1 ? "" : "s"} for ${normalizedGene}${variantName ? ` (filtered: ${variantName})` : ""} (${totalCount} total in CIViC).`,
            keyFields,
            sources: [`https://civicdb.org/genes/${normalizedGene}`, CIVIC_GRAPHQL_API],
            limitations: [
              "CIViC is community-curated and focuses on cancer-relevant variants with clinical or therapeutic evidence.",
              "Evidence levels: A (validated), B (clinical), C (case study), D (preclinical), E (inferential).",
            ],
          }),
        },
      ],
    };
  }
);

server.registerTool(
  "search_civic_genes",
  {
    description:
      "Get a CIViC gene summary including description, associated variant count, diseases, and therapies. " +
      "Use to understand the clinical significance landscape for a gene in oncology.",
    inputSchema: {
      name: z.string().describe("Gene symbol (e.g. BRAF, TP53, EGFR)"),
    },
  },
  async ({ name }) => {
    const normalizedName = normalizeWhitespace(name || "").toUpperCase();
    if (!normalizedName) {
      return { content: [{ type: "text", text: "Provide a gene symbol (e.g. BRAF)." }] };
    }

    const query = `
      query GeneDetail($name: String!) {
        genes(name: $name) {
          nodes {
            id
            name
            description
            officialName
            variants {
              totalCount
            }
            therapies: variants(first: 50) {
              nodes {
                molecularProfiles {
                  nodes {
                    evidenceItems(first: 10) {
                      nodes {
                        status
                        therapies { name }
                        disease { name }
                        evidenceLevel
                        significance
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    `;

    const data = await fetchCivicGraphQL(query, { name: normalizedName });
    const genes = data?.genes?.nodes || [];

    if (genes.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Gene ${normalizedName} not found in CIViC.`,
              keyFields: [`Queried: ${normalizedName}`],
              sources: [CIVIC_GRAPHQL_API],
              limitations: ["CIViC covers genes with clinical/therapeutic evidence in oncology; not all genes are represented."],
            }),
          },
        ],
      };
    }

    const gene = genes[0];
    const variantCount = gene.variants?.totalCount || 0;
    const description = normalizeWhitespace(gene.description || "No description available.");

    const allEvidence = (gene.therapies?.nodes || []).flatMap((v) =>
      (v.molecularProfiles?.nodes || []).flatMap((p) =>
        (p.evidenceItems?.nodes || []).filter((e) => normalizeWhitespace(e.status || "").toLowerCase() === "accepted")
      )
    );

    const diseases = [...new Set(allEvidence.map((e) => normalizeWhitespace(e.disease?.name || "")).filter(Boolean))].slice(0, 8);
    const therapies = [...new Set(allEvidence.flatMap((e) => (e.therapies || []).map((t) => normalizeWhitespace(t.name || ""))).filter(Boolean))].slice(0, 8);
    const levels = allEvidence.map((e) => normalizeWhitespace(e.evidenceLevel || "")).filter(Boolean);
    const levelCounts = {};
    levels.forEach((l) => { levelCounts[l] = (levelCounts[l] || 0) + 1; });
    const levelSummary = Object.entries(levelCounts).map(([k, v]) => `${k}:${v}`).join(", ") || "none";

    const keyFields = [
      `Gene: ${normalizedName} (${normalizeWhitespace(gene.officialName || "")})`,
      `Description: ${description.slice(0, 300)}${description.length > 300 ? "..." : ""}`,
      `CIViC variants: ${variantCount}`,
      `Evidence items (accepted): ${allEvidence.length} (levels: ${levelSummary})`,
      `Associated diseases: ${diseases.length > 0 ? diseases.join(", ") : "none listed"}`,
      `Associated therapies: ${therapies.length > 0 ? therapies.join(", ") : "none listed"}`,
    ];

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: `CIViC gene profile for ${normalizedName} with ${variantCount} curated variants.`,
            keyFields,
            sources: [`https://civicdb.org/genes/${normalizedName}`, CIVIC_GRAPHQL_API],
            limitations: [
              "CIViC is oncology-focused; gene descriptions reflect cancer-relevance, not general function.",
              "Therapy associations reflect curated clinical evidence, not approved indications.",
            ],
          }),
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// MyVariant.info — aggregated variant annotations (ClinVar, dbSNP, CADD, etc.)
// ---------------------------------------------------------------------------

server.registerTool(
  "get_variant_annotations",
  {
    description:
      "Get aggregated variant annotations from MyVariant.info, which combines 20+ sources including ClinVar, " +
      "dbSNP, CADD, gnomAD, COSMIC, and more. Accepts HGVS notation or rsID. " +
      "Use for comprehensive variant characterization in a single call.",
    inputSchema: {
      variantId: z
        .string()
        .describe('Variant identifier: rsID (e.g. "rs113488022") or HGVS genomic notation (e.g. "chr7:g.140753336A>T")'),
      fields: z
        .string()
        .optional()
        .describe("Comma-separated fields to return (e.g. clinvar,cadd,dbsnp,gnomad_exome). Default returns key fields."),
    },
  },
  async ({ variantId, fields }) => {
    const normalizedId = normalizeWhitespace(variantId || "");
    if (!normalizedId) {
      return { content: [{ type: "text", text: 'Provide a variant ID (rsID or HGVS notation, e.g. "rs113488022").' }] };
    }

    const defaultFields = "clinvar,cadd,dbsnp,gnomad_exome,gnomad_genome,cosmic,dbnsfp,snpeff";
    const requestFields = normalizeWhitespace(fields || "") || defaultFields;

    const isRsId = /^rs\d+$/i.test(normalizedId);
    let url;
    if (isRsId) {
      const params = new URLSearchParams({ q: `dbsnp.rsid:${normalizedId}`, fields: requestFields, size: "1" });
      url = `${MYVARIANT_API}/query?${params}`;
    } else {
      const params = new URLSearchParams({ fields: requestFields });
      url = `${MYVARIANT_API}/variant/${encodeURIComponent(normalizedId)}?${params}`;
    }

    const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });

    const hit = isRsId ? (Array.isArray(data?.hits) ? data.hits[0] : null) : data;
    if (!hit || hit.notfound) {
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Variant ${normalizedId} not found in MyVariant.info.`,
              keyFields: [`Queried: ${normalizedId}`],
              sources: [url],
              limitations: [
                'Check variant format. Examples: "rs113488022", "chr7:g.140753336A>T".',
                "MyVariant.info uses hg19 coordinates by default.",
              ],
            }),
          },
        ],
      };
    }

    const keyFields = [];
    const hgvsId = normalizeWhitespace(hit._id || "");
    if (hgvsId) keyFields.push(`HGVS ID: ${hgvsId}`);

    const rsid = normalizeWhitespace(hit.dbsnp?.rsid || "");
    if (rsid) keyFields.push(`dbSNP: ${rsid}`);

    const gene = normalizeWhitespace(hit.dbsnp?.gene?.symbol || hit.cadd?.gene?.[0]?.genename || hit.dbnsfp?.genename || "");
    if (gene) keyFields.push(`Gene: ${gene}`);

    if (hit.clinvar) {
      const cv = hit.clinvar;
      const sig = normalizeWhitespace(
        Array.isArray(cv.rcv) ? cv.rcv.map((r) => normalizeWhitespace(r.clinical_significance || "")).filter(Boolean).join("; ")
        : cv.rcv?.clinical_significance || ""
      );
      if (sig) keyFields.push(`ClinVar significance: ${sig}`);
      const conditions = Array.isArray(cv.rcv)
        ? [...new Set(cv.rcv.map((r) => normalizeWhitespace(r.conditions?.identifiers?.medgen || r.conditions?.name || "")).filter(Boolean))].slice(0, 3)
        : [];
      if (conditions.length > 0) keyFields.push(`ClinVar conditions: ${conditions.join(", ")}`);
    }

    if (hit.cadd) {
      const phred = hit.cadd.phred != null ? Number(hit.cadd.phred).toFixed(2) : "";
      const rawScore = hit.cadd.rawscore != null ? Number(hit.cadd.rawscore).toFixed(3) : "";
      if (phred) keyFields.push(`CADD phred: ${phred}${rawScore ? ` (raw: ${rawScore})` : ""}`);
      const consequence = normalizeWhitespace(hit.cadd.consequence || "");
      if (consequence) keyFields.push(`CADD consequence: ${consequence}`);
    }

    if (hit.dbnsfp) {
      const sift = normalizeWhitespace(hit.dbnsfp.sift?.pred || "");
      const pp2 = normalizeWhitespace(hit.dbnsfp.polyphen2?.hdiv?.pred || "");
      if (sift) keyFields.push(`SIFT (via dbNSFP): ${sift}`);
      if (pp2) keyFields.push(`PolyPhen-2 (via dbNSFP): ${pp2}`);
    }

    if (hit.gnomad_exome) {
      const af = hit.gnomad_exome.af?.af != null ? hit.gnomad_exome.af.af : hit.gnomad_exome.af;
      if (af != null) keyFields.push(`gnomAD exome AF: ${Number(af).toExponential(3)}`);
    }
    if (hit.gnomad_genome) {
      const af = hit.gnomad_genome.af?.af != null ? hit.gnomad_genome.af.af : hit.gnomad_genome.af;
      if (af != null) keyFields.push(`gnomAD genome AF: ${Number(af).toExponential(3)}`);
    }

    if (hit.cosmic) {
      const cosmicId = normalizeWhitespace(hit.cosmic.cosmic_id || hit.cosmic._license || "");
      if (cosmicId) keyFields.push(`COSMIC: ${cosmicId}`);
    }

    if (keyFields.length <= 1) {
      keyFields.push("Limited annotation data available for this variant.");
    }

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: `Aggregated annotations for variant ${normalizedId} from MyVariant.info.`,
            keyFields,
            sources: [url, `https://myvariant.info/v1/variant/${encodeURIComponent(hgvsId || normalizedId)}`],
            limitations: [
              "MyVariant.info uses hg19 coordinates by default. For hg38, append &assembly=hg38 or convert coordinates.",
              "Data freshness depends on source update frequency; ClinVar updates monthly, gnomAD less frequently.",
            ],
          }),
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// AlphaFold — predicted protein structure lookup
// ---------------------------------------------------------------------------

server.registerTool(
  "get_alphafold_structure",
  {
    description:
      "Look up AlphaFold predicted protein structure by UniProt accession. Returns confidence scores (pLDDT), " +
      "model quality, and download URLs for PDB/CIF structure files. Use for structural biology questions.",
    inputSchema: {
      uniprotId: z.string().describe("UniProt accession (e.g. P00533 for EGFR, P04637 for TP53)"),
    },
  },
  async ({ uniprotId }) => {
    const normalizedId = normalizeWhitespace(uniprotId || "").toUpperCase();
    if (!normalizedId) {
      return { content: [{ type: "text", text: "Provide a UniProt accession (e.g. P00533)." }] };
    }

    const url = `${ALPHAFOLD_API}/prediction/${normalizedId}`;
    const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 10000, maxBackoffMs: 3000 });

    const entries = Array.isArray(data) ? data : [data];
    const entry = entries[0];
    if (!entry || !entry.entryId) {
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `No AlphaFold prediction found for UniProt accession ${normalizedId}.`,
              keyFields: [`Queried: ${normalizedId}`],
              sources: [url],
              limitations: [
                "AlphaFold DB covers most of UniProt but not all proteins. Check the accession is correct.",
              ],
            }),
          },
        ],
      };
    }

    const entryId = normalizeWhitespace(entry.entryId || "");
    const gene = normalizeWhitespace(entry.gene || "");
    const organism = normalizeWhitespace(entry.organismScientificName || "");
    const globalPlddt = entry.globalMetricValue != null ? Number(entry.globalMetricValue).toFixed(1) : "N/A";
    const seqLength = entry.uniprotEnd || entry.sequenceLength || "unknown";

    const confidenceLabel =
      globalPlddt === "N/A" ? "N/A"
      : Number(globalPlddt) >= 90 ? `${globalPlddt} (very high)`
      : Number(globalPlddt) >= 70 ? `${globalPlddt} (confident)`
      : Number(globalPlddt) >= 50 ? `${globalPlddt} (low)`
      : `${globalPlddt} (very low)`;

    const pdbUrl = normalizeWhitespace(entry.pdbUrl || "");
    const cifUrl = normalizeWhitespace(entry.cifUrl || "");
    const paeUrl = normalizeWhitespace(entry.paeImageUrl || "");

    const keyFields = [
      `Entry: ${entryId}`,
      `UniProt: [${normalizedId}](https://www.uniprot.org/uniprotkb/${normalizedId})`,
    ];
    if (gene) keyFields.push(`Gene: ${gene}`);
    if (organism) keyFields.push(`Organism: ${organism}`);
    keyFields.push(`Sequence length: ${seqLength} residues`);
    keyFields.push(`Global pLDDT: ${confidenceLabel}`);
    if (pdbUrl) keyFields.push(`PDB file: ${pdbUrl}`);
    if (cifUrl) keyFields.push(`CIF file: ${cifUrl}`);
    if (paeUrl) keyFields.push(`PAE image: ${paeUrl}`);

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: `AlphaFold structure prediction for ${gene || normalizedId} (pLDDT: ${confidenceLabel}).`,
            keyFields,
            sources: [
              `https://alphafold.ebi.ac.uk/entry/${entryId}`,
              url,
            ],
            limitations: [
              "pLDDT > 90: high confidence; 70-90: confident backbone; 50-70: low confidence; < 50: likely disordered.",
              "AlphaFold predictions are computational models, not experimentally determined structures.",
            ],
          }),
        },
      ],
    };
  }
);

// ---------------------------------------------------------------------------
// GWAS Catalog — trait-variant associations from genome-wide association studies
// ---------------------------------------------------------------------------

server.registerTool(
  "search_gwas_associations",
  {
    description:
      "Search the GWAS Catalog for trait-variant associations from genome-wide association studies. " +
      "Query by disease/trait name (e.g. 'breast carcinoma', 'Parkinson disease') or variant rsID (e.g. 'rs1234'). " +
      "Returns SNPs, mapped genes, p-values, odds ratios, and study metadata.",
    inputSchema: {
      trait: z.string().optional().describe("Disease or trait name (e.g. 'breast carcinoma', 'type 2 diabetes')"),
      variantId: z.string().optional().describe("Variant rsID (e.g. 'rs7903146'). Use instead of trait for variant-specific queries."),
      limit: z.number().int().min(1).max(50).optional().default(20).describe("Max results to return (default 20, max 50)"),
    },
  },
  async ({ trait, variantId, limit = 20 }) => {
    const cleanTrait = normalizeWhitespace(trait || "");
    const cleanVariant = normalizeWhitespace(variantId || "");

    if (!cleanTrait && !cleanVariant) {
      return { content: [{ type: "text", text: "Provide either a trait/disease name or a variant rsID." }] };
    }

    const params = new URLSearchParams({ size: String(limit), show_child_traits: "false" });
    if (cleanVariant) {
      params.set("variant_id", cleanVariant);
    } else {
      params.set("efo_trait", cleanTrait);
    }

    const url = `${GWAS_CATALOG_API}/associations?${params}`;
    let data;
    try {
      data = await fetchJsonWithRetry(url, { retries: 2, timeoutMs: 15000 });
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `GWAS Catalog query failed: ${err.message}`,
          keyFields: [`Query: ${cleanVariant || cleanTrait}`],
          sources: [url],
          limitations: ["The GWAS Catalog API may be temporarily unavailable. Rate limit: 15 req/s."],
        }) }],
      };
    }

    const associations = data?._embedded?.associations || [];
    const totalElements = data?.page?.totalElements || 0;

    if (associations.length === 0) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No GWAS associations found for "${cleanVariant || cleanTrait}".`,
          keyFields: [`Query: ${cleanVariant || cleanTrait}`, `Total results: 0`],
          sources: [url],
          limitations: ["Try alternative trait names or check EFO ontology terms at www.ebi.ac.uk/gwas."],
        }) }],
      };
    }

    const rows = associations.map((a) => {
      const snps = (a.snp_allele || []).map((s) => `${s.rs_id}(${s.effect_allele})`).join(", ");
      const genes = (a.mapped_genes || []).join(", ");
      const traits = (a.efo_traits || []).map((t) => t.efo_trait).join("; ");
      const pval = a.p_value != null ? Number(a.p_value).toExponential(1) : "N/A";
      const or_val = a.or_value || "-";
      const beta = a.beta || "-";
      const ci = (a.ci_lower != null && a.ci_upper != null) ? `[${a.ci_lower}-${a.ci_upper}]` : "";
      const location = (a.locations || []).join(", ");
      return `- **${snps || "N/A"}** | genes: ${genes || "intergenic"} | trait: ${traits} | p=${pval} | OR=${or_val} beta=${beta} ${ci} | loc: ${location} | study: ${a.accession_id || ""} (PMID:${a.pubmed_id || "?"})`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Found ${totalElements.toLocaleString()} GWAS associations for "${cleanVariant || cleanTrait}" (showing top ${associations.length}).`,
          keyFields: [
            `Query: ${cleanVariant || cleanTrait}`,
            `Total associations: ${totalElements.toLocaleString()}`,
            `Showing: ${associations.length}`,
            ...rows,
          ],
          sources: [
            url,
            cleanVariant
              ? `https://www.ebi.ac.uk/gwas/variants/${cleanVariant}`
              : `https://www.ebi.ac.uk/gwas/search?query=${encodeURIComponent(cleanTrait)}`,
          ],
          limitations: [
            "Results sorted by default API ordering. For strongest associations, filter by p-value.",
            "GWAS Catalog contains curated top associations from published GWAS only.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// DGIdb — drug-gene interactions and druggability categories
// ---------------------------------------------------------------------------

server.registerTool(
  "search_drug_gene_interactions",
  {
    description:
      "Query DGIdb for drug-gene interactions and druggability categories. Returns approved/experimental drugs " +
      "targeting the gene, interaction types (inhibitor, agonist, etc.), evidence scores, and gene categories " +
      "(kinase, druggable genome, clinically actionable, etc.). Use for druggability assessment.",
    inputSchema: {
      genes: z.array(z.string()).min(1).max(5).describe('Gene symbols (e.g. ["BRAF", "EGFR"]). Max 5.'),
    },
  },
  async ({ genes }) => {
    const cleanGenes = (genes || []).map((g) => normalizeWhitespace(g).toUpperCase()).filter(Boolean);
    if (cleanGenes.length === 0) {
      return { content: [{ type: "text", text: "Provide at least one gene symbol." }] };
    }

    const query = `
      query DgidbGeneInteractions($genes: [String!]!) {
      genes(names: $genes) {
        nodes {
          name
          longName
          geneCategories { name }
          interactions {
            drug { name conceptId approved }
            interactionScore
            interactionTypes { type directionality }
            publications { pmid }
            sources { fullName }
          }
        }
      }
    }`;

    let data;
    try {
      const resp = await fetchJsonWithRetry(DGIDB_GRAPHQL_API, {
        retries: 2,
        timeoutMs: 15000,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, variables: { genes: cleanGenes } }),
      });
      const gqlData = requireGraphQLData(resp, "DGIdb");
      data = gqlData?.genes?.nodes || [];
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `DGIdb query failed: ${err.message}`,
          keyFields: [`Genes: ${cleanGenes.join(", ")}`],
          sources: [DGIDB_GRAPHQL_API],
          limitations: ["DGIdb API may be temporarily unavailable."],
        }) }],
      };
    }

    if (data.length === 0) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No genes found in DGIdb for: ${cleanGenes.join(", ")}`,
          keyFields: [`Queried: ${cleanGenes.join(", ")}`],
          sources: [`https://dgidb.org/genes/${cleanGenes[0]}`],
          limitations: ["Check gene symbol spelling. DGIdb uses HUGO gene symbols."],
        }) }],
      };
    }

    const sections = data.map((gene) => {
      const categories = (gene.geneCategories || []).map((c) => c.name).join(", ") || "None";
      const interactions = gene.interactions || [];
      const approved = interactions.filter((i) => i.drug?.approved);
      const experimental = interactions.filter((i) => !i.drug?.approved);

      const topInteractions = [...interactions]
        .sort((a, b) => (b.interactionScore || 0) - (a.interactionScore || 0))
        .slice(0, 15);

      const drugLines = topInteractions.map((i) => {
        const name = i.drug?.name || "Unknown";
        const status = i.drug?.approved ? "approved" : "experimental";
        const types = (i.interactionTypes || []).map((t) => t.type).join(", ") || "unspecified";
        const score = i.interactionScore != null ? i.interactionScore.toFixed(3) : "N/A";
        const pmids = (i.publications || []).slice(0, 3).map((p) => `PMID:${p.pmid}`).join(", ");
        return `  - ${name} (${status}) | type: ${types} | score: ${score} | ${pmids}`;
      });

      return [
        `**${gene.name}** — ${gene.longName || ""}`,
        `Categories: ${categories}`,
        `Total interactions: ${interactions.length} (${approved.length} approved, ${experimental.length} experimental)`,
        `Top interactions:`,
        ...drugLines,
      ].join("\n");
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `DGIdb results for ${cleanGenes.join(", ")}: ${data.reduce((n, g) => n + (g.interactions || []).length, 0)} drug-gene interactions found.`,
          keyFields: sections,
          sources: cleanGenes.map((g) => `https://dgidb.org/genes/${g}`),
          limitations: [
            "Interaction scores reflect aggregated evidence; higher is stronger.",
            "DGIdb aggregates from 40+ sources; individual source quality varies.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// GTEx — tissue-level gene expression
// ---------------------------------------------------------------------------

server.registerTool(
  "get_gene_tissue_expression",
  {
    description:
      "Get median gene expression across human tissues from GTEx (v8). Returns median TPM values for ~54 tissues. " +
      "Use for target safety assessment ('where is this gene expressed?') and tissue specificity analysis.",
    inputSchema: {
      geneSymbol: z.string().describe("Gene symbol (e.g. 'BRCA1', 'EGFR', 'TP53')"),
    },
  },
  async ({ geneSymbol }) => {
    const symbol = normalizeWhitespace(geneSymbol || "").toUpperCase();
    if (!symbol) {
      return { content: [{ type: "text", text: "Provide a gene symbol (e.g. BRCA1)." }] };
    }

    let gencodeId;
    try {
      const geneData = await fetchJsonWithRetry(
        `${GTEX_API}/reference/gene?geneId=${encodeURIComponent(symbol)}&datasetId=gtex_v8`,
        { retries: 1, timeoutMs: 10000 }
      );
      const genes = geneData?.data || [];
      if (genes.length === 0) {
        return {
          content: [{ type: "text", text: renderStructuredResponse({
            summary: `Gene "${symbol}" not found in GTEx.`,
            keyFields: [`Queried: ${symbol}`],
            sources: [`${GTEX_API}/reference/gene?geneId=${symbol}`],
            limitations: ["Check gene symbol spelling. GTEx uses HUGO gene symbols."],
          }) }],
        };
      }
      gencodeId = genes[0].gencodeId;
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `GTEx gene lookup failed: ${err.message}`,
          keyFields: [`Queried: ${symbol}`],
          sources: [`${GTEX_API}/reference/gene?geneId=${symbol}`],
          limitations: ["GTEx API may be temporarily unavailable."],
        }) }],
      };
    }

    let expressionData;
    try {
      const exprResp = await fetchJsonWithRetry(
        `${GTEX_API}/expression/medianGeneExpression?gencodeId=${encodeURIComponent(gencodeId)}&datasetId=gtex_v8&itemsPerPage=100`,
        { retries: 1, timeoutMs: 10000 }
      );
      expressionData = exprResp?.data || [];
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `GTEx expression query failed: ${err.message}`,
          keyFields: [`Gene: ${symbol}`, `GENCODE ID: ${gencodeId}`],
          sources: [`${GTEX_API}/expression/medianGeneExpression?gencodeId=${gencodeId}`],
          limitations: ["GTEx API may be temporarily unavailable."],
        }) }],
      };
    }

    if (expressionData.length === 0) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No expression data found for ${symbol} in GTEx.`,
          keyFields: [`Gene: ${symbol}`, `GENCODE ID: ${gencodeId}`],
          sources: [`https://gtexportal.org/home/gene/${symbol}`],
          limitations: ["Gene may not have sufficient expression data in GTEx v8."],
        }) }],
      };
    }

    const sorted = [...expressionData].sort((a, b) => (b.median || 0) - (a.median || 0));
    const formatTissue = (id) => id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

    const tissueLines = sorted.map((t) => {
      const tpm = (t.median || 0).toFixed(2);
      const bar = "█".repeat(Math.min(30, Math.round((t.median || 0) / (sorted[0]?.median || 1) * 20)));
      return `  ${formatTissue(t.tissueSiteDetailId).padEnd(45)} ${tpm.padStart(8)} TPM ${bar}`;
    });

    const maxTissue = sorted[0];
    const expressed = sorted.filter((t) => (t.median || 0) > 1);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `GTEx expression profile for ${symbol}: expressed (>1 TPM) in ${expressed.length}/${sorted.length} tissues. Highest in ${formatTissue(maxTissue.tissueSiteDetailId)} (${maxTissue.median.toFixed(1)} TPM).`,
          keyFields: [
            `Gene: ${symbol} (${gencodeId})`,
            `Tissues with data: ${sorted.length}`,
            `Tissues >1 TPM: ${expressed.length}`,
            `Highest expression: ${formatTissue(maxTissue.tissueSiteDetailId)} (${maxTissue.median.toFixed(1)} TPM)`,
            `\nExpression by tissue (median TPM, GTEx v8):`,
            ...tissueLines,
          ],
          sources: [`https://gtexportal.org/home/gene/${symbol}`],
          limitations: [
            "Values are median TPM from GTEx v8 (948 donors, 54 tissues).",
            "Expression in cell lines may not reflect in vivo tissue expression.",
            "GTEx samples are from post-mortem donors; disease-state expression may differ.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// RCSB PDB — experimental protein structures
// ---------------------------------------------------------------------------

server.registerTool(
  "search_protein_structures",
  {
    description:
      "Search RCSB PDB for experimentally determined protein structures. Query by UniProt accession (e.g. 'P15056') " +
      "or gene name. Returns PDB IDs, resolution, experimental method, title, deposition date, and ligands. " +
      "Complements AlphaFold predictions with actual experimental structures.",
    inputSchema: {
      query: z.string().describe("UniProt accession (e.g. 'P15056' for BRAF) or gene name (e.g. 'BRAF')"),
      limit: z.number().int().min(1).max(20).optional().default(10).describe("Max structures to return (default 10)"),
    },
  },
  async ({ query: rawQuery, limit = 10 }) => {
    const q = normalizeWhitespace(rawQuery || "");
    if (!q) {
      return { content: [{ type: "text", text: "Provide a UniProt accession or gene name." }] };
    }

    const isUniprot = /^[A-Z][0-9][A-Z0-9]{3}[0-9]$/i.test(q) || /^[A-Z][0-9][A-Z0-9]{3}[0-9]-\d+$/i.test(q);

    const searchBody = isUniprot
      ? {
          query: {
            type: "terminal",
            service: "text",
            parameters: {
              attribute: "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
              operator: "in",
              value: [q.toUpperCase()],
              negation: false,
            },
          },
          return_type: "entry",
          request_options: {
            paginate: { start: 0, rows: limit },
            results_content_type: ["experimental"],
            sort: [{ sort_by: "rcsb_accession_info.initial_release_date", direction: "desc" }],
          },
        }
      : {
          query: {
            type: "terminal",
            service: "full_text",
            parameters: { value: q },
          },
          return_type: "entry",
          request_options: {
            paginate: { start: 0, rows: limit },
            results_content_type: ["experimental"],
            sort: [{ sort_by: "rcsb_accession_info.initial_release_date", direction: "desc" }],
          },
        };

    let searchResult;
    try {
      searchResult = await fetchJsonWithRetry(RCSB_SEARCH_API, {
        retries: 2,
        timeoutMs: 15000,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(searchBody),
      });
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `RCSB PDB search failed: ${err.message}`,
          keyFields: [`Query: ${q}`],
          sources: ["https://www.rcsb.org/"],
          limitations: ["RCSB API may be temporarily unavailable."],
        }) }],
      };
    }

    const totalCount = searchResult?.total_count || 0;
    const entries = searchResult?.result_set || [];

    if (entries.length === 0) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No PDB structures found for "${q}".`,
          keyFields: [`Query: ${q}`, `Total: 0`],
          sources: [`https://www.rcsb.org/search?request=${encodeURIComponent(q)}`],
          limitations: ["Try a UniProt accession for more precise results."],
        }) }],
      };
    }

    const pdbIds = entries.map((e) => e.identifier);
    const details = await Promise.allSettled(
      pdbIds.map((id) =>
        fetchJsonWithRetry(`${RCSB_DATA_API}/core/entry/${id}`, { retries: 1, timeoutMs: 8000 })
      )
    );

    const structureLines = pdbIds.map((id, i) => {
      const d = details[i].status === "fulfilled" ? details[i].value : null;
      if (!d) return `- **${id}** — details unavailable`;
      const title = normalizeWhitespace(d?.struct?.title || "N/A");
      const method = d?.exptl?.[0]?.method || "N/A";
      const resolution = d?.rcsb_entry_info?.resolution_combined?.[0];
      const resStr = resolution != null ? `${resolution.toFixed(2)} Å` : "N/A";
      const deposited = d?.rcsb_accession_info?.deposit_date?.slice(0, 10) || "N/A";
      const ligands = (d?.rcsb_entry_info?.nonpolymer_bound_components || []).join(", ") || "none";
      return `- **[${id}](https://www.rcsb.org/structure/${id})** | ${method} ${resStr} | deposited: ${deposited} | ligands: ${ligands}\n  ${title}`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Found ${totalCount} PDB structures for "${q}" (showing ${pdbIds.length}, newest first).`,
          keyFields: [
            `Query: ${q}`,
            `Total structures: ${totalCount}`,
            `Showing: ${pdbIds.length}`,
            ...structureLines,
          ],
          sources: [
            isUniprot
              ? `https://www.rcsb.org/search?request=${q}`
              : `https://www.rcsb.org/search?request=${encodeURIComponent(q)}`,
          ],
          limitations: [
            "Results sorted by deposition date (newest first).",
            "Resolution values are in Ångströms; lower is better (< 2.0 Å = high quality).",
            "For predicted structures, use the AlphaFold tool instead.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// cBioPortal — cancer mutation profiles across TCGA studies
// ---------------------------------------------------------------------------

const TCGA_PAN_CANCER_STUDIES = [
  "acc_tcga_pan_can_atlas_2018", "blca_tcga_pan_can_atlas_2018", "brca_tcga_pan_can_atlas_2018",
  "cesc_tcga_pan_can_atlas_2018", "chol_tcga_pan_can_atlas_2018", "coadread_tcga_pan_can_atlas_2018",
  "dlbc_tcga_pan_can_atlas_2018", "esca_tcga_pan_can_atlas_2018", "gbm_tcga_pan_can_atlas_2018",
  "hnsc_tcga_pan_can_atlas_2018", "kich_tcga_pan_can_atlas_2018", "kirc_tcga_pan_can_atlas_2018",
  "kirp_tcga_pan_can_atlas_2018", "laml_tcga_pan_can_atlas_2018", "lgg_tcga_pan_can_atlas_2018",
  "lihc_tcga_pan_can_atlas_2018", "luad_tcga_pan_can_atlas_2018", "lusc_tcga_pan_can_atlas_2018",
  "meso_tcga_pan_can_atlas_2018", "ov_tcga_pan_can_atlas_2018", "paad_tcga_pan_can_atlas_2018",
  "pcpg_tcga_pan_can_atlas_2018", "prad_tcga_pan_can_atlas_2018", "sarc_tcga_pan_can_atlas_2018",
  "skcm_tcga_pan_can_atlas_2018", "stad_tcga_pan_can_atlas_2018", "tgct_tcga_pan_can_atlas_2018",
  "thca_tcga_pan_can_atlas_2018", "thym_tcga_pan_can_atlas_2018", "ucec_tcga_pan_can_atlas_2018",
  "ucs_tcga_pan_can_atlas_2018", "uvm_tcga_pan_can_atlas_2018",
];

server.registerTool(
  "get_cancer_mutation_profile",
  {
    description:
      "Get mutation profile for a gene across cancer types from cBioPortal (TCGA Pan-Cancer Atlas). " +
      "Returns mutation counts by cancer type, most frequent protein changes, and mutation types. " +
      "Use for oncology questions like 'how often is BRAF mutated in melanoma?'.",
    inputSchema: {
      geneSymbol: z.string().describe("Gene symbol (e.g. 'BRAF', 'TP53', 'KRAS')"),
    },
  },
  async ({ geneSymbol }) => {
    const symbol = normalizeWhitespace(geneSymbol || "").toUpperCase();
    if (!symbol) {
      return { content: [{ type: "text", text: "Provide a gene symbol (e.g. BRAF)." }] };
    }

    let entrezGeneId;
    try {
      const geneInfo = await fetchJsonWithRetry(
        `${CBIOPORTAL_API}/genes/${encodeURIComponent(symbol)}`,
        { retries: 1, timeoutMs: 8000 }
      );
      entrezGeneId = geneInfo?.entrezGeneId;
      if (!entrezGeneId) throw new Error("Gene not found");
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `Gene "${symbol}" not found in cBioPortal.`,
          keyFields: [`Queried: ${symbol}`],
          sources: [`${CBIOPORTAL_API}/genes/${symbol}`],
          limitations: ["Check gene symbol spelling. cBioPortal uses HUGO gene symbols."],
        }) }],
      };
    }

    const mutationProfileIds = TCGA_PAN_CANCER_STUDIES.map((s) => `${s}_mutations`);

    let mutations;
    try {
      mutations = await fetchJsonWithRetry(
        `${CBIOPORTAL_API}/mutations/fetch?projection=SUMMARY`,
        {
          retries: 2,
          timeoutMs: 30000,
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify({ entrezGeneIds: [entrezGeneId], molecularProfileIds: mutationProfileIds }),
        }
      );
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `cBioPortal mutation query failed: ${err.message}`,
          keyFields: [`Gene: ${symbol} (Entrez: ${entrezGeneId})`],
          sources: [`https://www.cbioportal.org/results/mutations?gene_list=${symbol}`],
          limitations: ["cBioPortal API may be temporarily slow for pan-cancer queries."],
        }) }],
      };
    }

    if (!mutations || mutations.length === 0) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No mutations found for ${symbol} across TCGA Pan-Cancer Atlas studies.`,
          keyFields: [`Gene: ${symbol}`, `Studies queried: ${TCGA_PAN_CANCER_STUDIES.length}`],
          sources: [`https://www.cbioportal.org/results/mutations?gene_list=${symbol}`],
          limitations: ["Gene may not be frequently mutated in TCGA cohorts."],
        }) }],
      };
    }

    const byCancerType = {};
    const mutationTypeCounts = {};
    const proteinChangeCounts = {};

    for (const m of mutations) {
      const study = (m.studyId || "unknown").replace(/_tcga_pan_can_atlas_2018$/, "").toUpperCase();
      byCancerType[study] = (byCancerType[study] || 0) + 1;
      const mtype = m.mutationType || "Unknown";
      mutationTypeCounts[mtype] = (mutationTypeCounts[mtype] || 0) + 1;
      const pc = m.proteinChange;
      if (pc) proteinChangeCounts[pc] = (proteinChangeCounts[pc] || 0) + 1;
    }

    const sortedCancerTypes = Object.entries(byCancerType).sort((a, b) => b[1] - a[1]);
    const sortedProteinChanges = Object.entries(proteinChangeCounts).sort((a, b) => b[1] - a[1]).slice(0, 15);
    const sortedMutTypes = Object.entries(mutationTypeCounts).sort((a, b) => b[1] - a[1]);

    const cancerLines = sortedCancerTypes.map(
      ([type, count]) => `  ${type.padEnd(12)} ${String(count).padStart(5)} mutations`
    );

    const hotspotLines = sortedProteinChanges.map(
      ([change, count]) => `  ${change.padEnd(15)} ${String(count).padStart(4)}x`
    );

    const typeSummary = sortedMutTypes.map(([t, c]) => `${t}: ${c}`).join(", ");

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `${symbol}: ${mutations.length} mutations across ${sortedCancerTypes.length} cancer types (TCGA Pan-Cancer Atlas). Most mutated in ${sortedCancerTypes[0][0]} (${sortedCancerTypes[0][1]}).`,
          keyFields: [
            `Gene: ${symbol} (Entrez: ${entrezGeneId})`,
            `Total mutations: ${mutations.length}`,
            `Cancer types with mutations: ${sortedCancerTypes.length}/${TCGA_PAN_CANCER_STUDIES.length}`,
            `Mutation types: ${typeSummary}`,
            `\nMutations by cancer type:`,
            ...cancerLines,
            `\nTop protein changes (hotspots):`,
            ...hotspotLines,
          ],
          sources: [
            `https://www.cbioportal.org/results/mutations?gene_list=${symbol}&cancer_study_list=`,
          ],
          limitations: [
            "Data from TCGA Pan-Cancer Atlas (32 cancer types, ~10,000 samples).",
            "Mutation counts are absolute; frequency depends on cohort size per cancer type.",
            "Does not include non-TCGA studies. Query cBioPortal directly for broader coverage.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// PubChem — chemical compound data
// ---------------------------------------------------------------------------

const PUBCHEM_PROPERTIES = [
  "MolecularFormula", "MolecularWeight", "IUPACName", "CanonicalSMILES",
  "InChIKey", "XLogP", "ExactMass", "HBondDonorCount", "HBondAcceptorCount",
  "RotatableBondCount", "TPSA", "Complexity",
].join(",");

server.registerTool(
  "get_pubchem_compound",
  {
    description:
      "Look up a chemical compound in PubChem by name, CID, or SMILES. Returns molecular properties " +
      "(formula, weight, SMILES, InChIKey, XLogP, H-bond donors/acceptors, polar surface area), " +
      "description, and synonyms. Use for chemistry and pharmacology questions.",
    inputSchema: {
      query: z.string().describe("Compound name (e.g. 'metformin'), PubChem CID (e.g. '2244'), or SMILES string"),
      queryType: z.enum(["name", "cid", "smiles"]).optional().default("name")
        .describe("Type of query: 'name' (default), 'cid' (PubChem compound ID), or 'smiles'"),
    },
  },
  async ({ query: rawQuery, queryType = "name" }) => {
    const q = normalizeWhitespace(rawQuery || "");
    if (!q) {
      return { content: [{ type: "text", text: "Provide a compound name, CID, or SMILES string." }] };
    }

    const namespace = queryType === "cid" ? "cid" : queryType === "smiles" ? "smiles" : "name";
    const encodedQ = encodeURIComponent(q);
    const baseUrl = `${PUBCHEM_API}/compound/${namespace}/${encodedQ}`;

    const [propsResult, descResult, synResult] = await Promise.allSettled([
      fetchJsonWithRetry(`${baseUrl}/property/${PUBCHEM_PROPERTIES}/JSON`, { retries: 1, timeoutMs: 10000 }),
      fetchJsonWithRetry(`${baseUrl}/description/JSON`, { retries: 1, timeoutMs: 10000 }),
      fetchJsonWithRetry(`${baseUrl}/synonyms/JSON`, { retries: 1, timeoutMs: 10000 }),
    ]);

    const props = propsResult.status === "fulfilled"
      ? propsResult.value?.PropertyTable?.Properties?.[0]
      : null;

    if (!props) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `Compound "${q}" not found in PubChem.`,
          keyFields: [`Query: ${q} (${namespace})`],
          sources: [`https://pubchem.ncbi.nlm.nih.gov/#query=${encodeURIComponent(q)}`],
          limitations: ["Check compound name spelling. Try alternative names or SMILES."],
        }) }],
      };
    }

    const cid = props.CID;

    const descriptions = descResult.status === "fulfilled"
      ? (descResult.value?.InformationList?.Information || [])
          .filter((i) => i.Description)
          .map((i) => normalizeWhitespace(i.Description))
          .slice(0, 2)
      : [];

    const synonyms = synResult.status === "fulfilled"
      ? (synResult.value?.InformationList?.Information?.[0]?.Synonym || []).slice(0, 10)
      : [];

    const keyFields = [
      `PubChem CID: [${cid}](https://pubchem.ncbi.nlm.nih.gov/compound/${cid})`,
      `IUPAC Name: ${props.IUPACName || "N/A"}`,
      `Formula: ${props.MolecularFormula || "N/A"}`,
      `Molecular Weight: ${props.MolecularWeight || "N/A"} g/mol`,
      `Exact Mass: ${props.ExactMass || "N/A"}`,
      `SMILES: ${props.ConnectivitySMILES || props.CanonicalSMILES || "N/A"}`,
      `InChIKey: ${props.InChIKey || "N/A"}`,
      `XLogP: ${props.XLogP != null ? props.XLogP : "N/A"}`,
      `H-bond Donors: ${props.HBondDonorCount != null ? props.HBondDonorCount : "N/A"}`,
      `H-bond Acceptors: ${props.HBondAcceptorCount != null ? props.HBondAcceptorCount : "N/A"}`,
      `Rotatable Bonds: ${props.RotatableBondCount != null ? props.RotatableBondCount : "N/A"}`,
      `Topological Polar Surface Area: ${props.TPSA != null ? props.TPSA + " Å²" : "N/A"}`,
      `Complexity: ${props.Complexity != null ? props.Complexity : "N/A"}`,
    ];

    if (synonyms.length > 0) keyFields.push(`\nSynonyms: ${synonyms.join(", ")}`);
    if (descriptions.length > 0) keyFields.push(`\nDescription: ${descriptions.join(" ")}`);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `PubChem compound: ${props.IUPACName || q} (CID ${cid}, MW ${props.MolecularWeight} g/mol, XLogP ${props.XLogP != null ? props.XLogP : "N/A"}).`,
          keyFields,
          sources: [`https://pubchem.ncbi.nlm.nih.gov/compound/${cid}`],
          limitations: [
            "XLogP is a computed partition coefficient; negative values indicate hydrophilicity.",
            "Lipinski's Rule of 5: MW <= 500, XLogP <= 5, HBD <= 5, HBA <= 10 for oral drug-likeness.",
            "Rate limit: 5 requests/second to PubChem PUG-REST.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// ChEMBL REST API — Bioactivity & Selectivity
// ---------------------------------------------------------------------------

server.registerTool(
  "get_chembl_bioactivities",
  {
    description:
      "Get bioactivity data (IC50, Ki, Kd, EC50) for a drug from ChEMBL REST API. " +
      "Returns activity values grouped by target with min/median values — ideal for kinase selectivity " +
      "profiling and off-target assessment. Prefer this over BigQuery ebi_chembl for bioactivity lookups.",
    inputSchema: {
      drugName: z.string().describe("Drug name (e.g. 'tofacitinib') or ChEMBL ID (e.g. 'CHEMBL221959')"),
      activityTypes: z.string().optional().default("IC50,Ki,Kd")
        .describe("Comma-separated activity types to retrieve (default: 'IC50,Ki,Kd')"),
      targetFilter: z.string().optional()
        .describe("Optional substring to filter target names (e.g. 'JAK', 'kinase'). Case-insensitive."),
    },
  },
  async ({ drugName, activityTypes = "IC50,Ki,Kd", targetFilter }) => {
    const name = normalizeWhitespace(drugName || "");
    if (!name) {
      return { content: [{ type: "text", text: "Provide a drug name or ChEMBL ID." }] };
    }

    let chemblId = name.toUpperCase().startsWith("CHEMBL") ? name.toUpperCase() : null;

    if (!chemblId) {
      try {
        const searchResult = await fetchJsonWithRetry(
          `${CHEMBL_API}/molecule/search.json?q=${encodeURIComponent(name)}&limit=5`,
          { retries: 1, timeoutMs: 10000 }
        );
        const molecules = searchResult?.molecules || [];
        const exact = molecules.find(
          (m) => (m.pref_name || "").toLowerCase() === name.toLowerCase()
        ) || molecules[0];
        if (exact) chemblId = exact.molecule_chembl_id;
      } catch (_) { /* fall through */ }
    }

    if (!chemblId) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `Drug "${name}" not found in ChEMBL.`,
          keyFields: [`Query: ${name}`],
          sources: [`${CHEMBL_API}/molecule/search.json?q=${encodeURIComponent(name)}`],
          limitations: ["Try the generic drug name or a ChEMBL ID (e.g. CHEMBL221959)."],
        }) }],
      };
    }

    const types = normalizeWhitespace(activityTypes).split(",").map((t) => t.trim()).filter(Boolean).join(",");
    const activityUrl =
      `${CHEMBL_API}/activity.json?molecule_chembl_id=${chemblId}` +
      `&standard_type__in=${encodeURIComponent(types)}` +
      `&target_organism=Homo%20sapiens&limit=1000`;

    let activities;
    try {
      const result = await fetchJsonWithRetry(activityUrl, { retries: 2, timeoutMs: 20000 });
      activities = result?.activities || [];
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `ChEMBL activity query failed: ${err.message}`,
          keyFields: [`Drug: ${chemblId}`],
          sources: [activityUrl],
          limitations: ["ChEMBL API may be temporarily slow. Try again."],
        }) }],
      };
    }

    if (activities.length === 0) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No bioactivity data (${types}) found for ${chemblId} against human targets.`,
          keyFields: [`Drug: ${chemblId}`, `Activity types: ${types}`],
          sources: [`https://www.ebi.ac.uk/chembl/compound_report_card/${chemblId}/`],
          limitations: ["Try broader activity types (e.g. IC50,Ki,Kd,EC50,Potency)."],
        }) }],
      };
    }

    const byTarget = {};
    for (const a of activities) {
      const tgt = a.target_pref_name || "Unknown";
      const stype = a.standard_type || "?";
      const val = parseFloat(a.standard_value);
      const units = a.standard_units || "nM";
      if (isNaN(val)) continue;

      const key = `${tgt} | ${stype}`;
      if (!byTarget[key]) byTarget[key] = { target: tgt, type: stype, units, values: [] };
      byTarget[key].values.push(val);
    }

    let entries = Object.values(byTarget);

    if (targetFilter) {
      const filter = targetFilter.toLowerCase();
      entries = entries.filter((e) => e.target.toLowerCase().includes(filter));
    }

    entries.sort((a, b) => Math.min(...a.values) - Math.min(...b.values));

    const median = (arr) => {
      const s = [...arr].sort((a, b) => a - b);
      const mid = Math.floor(s.length / 2);
      return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
    };

    const lines = entries.slice(0, 40).map((e) => {
      const min = Math.min(...e.values);
      const med = median(e.values);
      return `  ${e.target.substring(0, 50).padEnd(50)} ${e.type.padEnd(6)} ` +
        `n=${String(e.values.length).padStart(3)}  min=${String(min.toFixed(1)).padStart(10)} ${e.units}` +
        (e.values.length > 1 ? `  median=${med.toFixed(1)}` : "");
    });

    const keyFields = [
      `Drug: ${chemblId}`,
      `Activity types: ${types}`,
      `Total data points: ${activities.length}`,
      `Unique targets: ${entries.length}${targetFilter ? ` (filtered by "${targetFilter}")` : ""}`,
    ];

    if (lines.length > 0) {
      keyFields.push(`\nBioactivity by target (sorted by potency):`);
      keyFields.push(...lines);
    }

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `ChEMBL: ${activities.length} bioactivity records for ${chemblId} across ${entries.length} targets. ` +
            `Most potent: ${entries[0]?.target} (${entries[0]?.type} min ${Math.min(...(entries[0]?.values || [0])).toFixed(1)} ${entries[0]?.units || "nM"}).`,
          keyFields,
          sources: [`https://www.ebi.ac.uk/chembl/compound_report_card/${chemblId}/`],
          limitations: [
            "Values from different assays are not directly comparable (different assay conditions).",
            "Showing min and median across all deposited measurements per target.",
            `Limited to first 1000 activity records from ChEMBL (${types}).`,
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// openFDA FAERS — Adverse Event Reports
// ---------------------------------------------------------------------------

const OPENFDA_API = "https://api.fda.gov/drug/event.json";

server.registerTool(
  "search_fda_adverse_events",
  {
    description:
      "Search FDA FAERS (post-marketing adverse event reports) for a drug. " +
      "Returns top adverse reactions by frequency, total report count, serious vs non-serious breakdown, " +
      "and top reported indications. Use for drug safety questions like 'what are the side effects of sotorasib?'.",
    inputSchema: {
      drugName: z.string().describe("Drug generic name, brand name, or active ingredient (e.g. 'sotorasib', 'lumakras', 'ibuprofen')"),
      limit: z.number().optional().default(20).describe("Max adverse reactions to return (default 20)"),
    },
  },
  async ({ drugName, limit = 20 }) => {
    const name = normalizeWhitespace(drugName || "");
    if (!name) {
      return { content: [{ type: "text", text: "Provide a drug name (e.g. sotorasib)." }] };
    }

    const boundedLimit = Math.min(Math.max(limit, 5), 50);
    const encoded = encodeURIComponent(name);
    const searchFields = [
      `patient.drug.openfda.generic_name:"${encoded}"`,
      `patient.drug.openfda.brand_name:"${encoded}"`,
      `patient.drug.openfda.substance_name:"${encoded}"`,
    ].join("+");

    const [reactionsResult, seriousResult, indicationsResult, totalResult] = await Promise.allSettled([
      fetchJsonWithRetry(
        `${OPENFDA_API}?search=${searchFields}&count=patient.reaction.reactionmeddrapt.exact&limit=${boundedLimit}`,
        { retries: 1, timeoutMs: 12000 }
      ),
      fetchJsonWithRetry(
        `${OPENFDA_API}?search=${searchFields}&count=serious`,
        { retries: 1, timeoutMs: 10000 }
      ),
      fetchJsonWithRetry(
        `${OPENFDA_API}?search=${searchFields}&count=patient.drug.drugindication.exact&limit=10`,
        { retries: 1, timeoutMs: 10000 }
      ),
      fetchJsonWithRetry(
        `${OPENFDA_API}?search=${searchFields}&limit=1`,
        { retries: 1, timeoutMs: 10000 }
      ),
    ]);

    const reactions = reactionsResult.status === "fulfilled"
      ? (reactionsResult.value?.results || [])
      : [];
    const seriousCounts = seriousResult.status === "fulfilled"
      ? (seriousResult.value?.results || [])
      : [];
    const indications = indicationsResult.status === "fulfilled"
      ? (indicationsResult.value?.results || [])
      : [];
    const totalReports = totalResult.status === "fulfilled"
      ? (totalResult.value?.meta?.results?.total || 0)
      : 0;

    if (reactions.length === 0 && totalReports === 0) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No FDA FAERS adverse event reports found for "${name}".`,
          keyFields: [`Drug: ${name}`],
          sources: [`https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:"${encoded}"`],
          limitations: [
            "Try alternative drug names (generic, brand, or active ingredient).",
            "Very new drugs may have limited FAERS data.",
          ],
        }) }],
      };
    }

    const seriousMap = {};
    for (const s of seriousCounts) {
      if (s.term === 1) seriousMap.serious = s.count;
      else if (s.term === 2) seriousMap.nonSerious = s.count;
    }

    const reactionLines = reactions.map(
      (r, i) => `  ${String(i + 1).padStart(2)}. ${r.term.padEnd(45)} ${String(r.count).padStart(5)} reports`
    );

    const indicationLines = indications.slice(0, 8).map(
      (ind) => `  ${ind.term}: ${ind.count}`
    );

    const keyFields = [
      `Drug: ${name}`,
      `Total FAERS reports: ${totalReports.toLocaleString()}`,
    ];

    if (seriousMap.serious != null || seriousMap.nonSerious != null) {
      keyFields.push(`Serious: ${(seriousMap.serious || 0).toLocaleString()} | Non-serious: ${(seriousMap.nonSerious || 0).toLocaleString()}`);
    }

    if (reactionLines.length > 0) {
      keyFields.push(`\nTop adverse reactions (by report count):`);
      keyFields.push(...reactionLines);
    }

    if (indicationLines.length > 0) {
      keyFields.push(`\nTop reported indications:`);
      keyFields.push(...indicationLines);
    }

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `FDA FAERS: ${totalReports.toLocaleString()} adverse event reports for ${name}. ` +
            `${seriousMap.serious ? seriousMap.serious.toLocaleString() + " serious. " : ""}` +
            `Most reported: ${reactions.slice(0, 3).map((r) => r.term.toLowerCase()).join(", ")}.`,
          keyFields,
          sources: [
            `https://api.fda.gov/drug/event.json?search=patient.drug.openfda.generic_name:"${encoded}"`,
            "https://open.fda.gov/apis/drug/event/",
          ],
          limitations: [
            "FAERS is a spontaneous reporting system — counts reflect reports, not incidence rates.",
            "Reporting bias: serious events and events with new drugs are over-represented.",
            "A report does not prove causation between the drug and the adverse event.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// Allen Brain Atlas (ABA) tools
// ---------------------------------------------------------------------------

const ABA_PRODUCT_IDS = {
  mouse: 1,
  human: 3,
  developing_mouse: 5,
  mouse_connectivity: 7,
};

function abaProductId(organism) {
  const key = String(organism || "mouse").trim().toLowerCase().replace(/[\s-]+/g, "_");
  return ABA_PRODUCT_IDS[key] ?? ABA_PRODUCT_IDS.mouse;
}

server.registerTool(
  "search_aba_genes",
  {
    description:
      "Searches the Allen Brain Atlas for genes by name, acronym, or keyword. Returns gene metadata including Entrez ID, acronym, and full name. Covers the Mouse Brain Atlas (default), Human Brain Atlas, Developing Mouse Brain Atlas, and Mouse Connectivity Atlas.",
    inputSchema: {
      query: z.string().describe("Gene name, acronym, or keyword (e.g. 'Pdyn', 'prodynorphin', 'dopamine')."),
      organism: z
        .enum(["mouse", "human", "developing_mouse", "mouse_connectivity"])
        .optional()
        .describe("Atlas product to search. Default 'mouse'."),
      maxResults: z.number().optional().describe("Max results (default 20, max 100)."),
    },
  },
  async ({ query, organism, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 100);
    const productId = abaProductId(organism);

    const nameUrl =
      `${ABA_API}/data/Gene/query.json` +
      `?criteria=products[id$eq${productId}],[name$il'*${encodeURIComponent(query)}*']` +
      `&num_rows=${limit}`;
    const acronymUrl =
      `${ABA_API}/data/Gene/query.json` +
      `?criteria=products[id$eq${productId}],[acronym$il'*${encodeURIComponent(query)}*']` +
      `&num_rows=${limit}`;

    const [byName, byAcronym] = await Promise.all([
      fetchJsonWithRetry(nameUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] })),
      fetchJsonWithRetry(acronymUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] })),
    ]);

    const seen = new Set();
    const genes = [];
    for (const g of [...(Array.isArray(byName.msg) ? byName.msg : []), ...(Array.isArray(byAcronym.msg) ? byAcronym.msg : [])]) {
      if (!g?.id || seen.has(g.id)) continue;
      seen.add(g.id);
      genes.push(g);
      if (genes.length >= limit) break;
    }

    if (genes.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No genes found in the Allen Brain Atlas for "${query}" (product: ${organism || "mouse"}).`,
            keyFields: [`Query: ${query}`, `Product: ${organism || "mouse"}`],
            sources: [`${ABA_API}/data/Gene/query.json`],
            limitations: ["Search uses case-insensitive LIKE matching on name and acronym fields."],
          }),
        }],
      };
    }

    const lines = genes.map((g, i) => {
      const parts = [`${String(i + 1).padStart(3)}. ${g.acronym} — ${g.name}`];
      if (g.entrez_id) parts.push(`Entrez: ${g.entrez_id}`);
      if (g.homologene_id) parts.push(`HomoloGene: ${g.homologene_id}`);
      parts.push(`ABA ID: ${g.id}`);
      return parts.join(" | ");
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Allen Brain Atlas: ${genes.length} gene(s) matching "${query}" (${organism || "mouse"} atlas).`,
          keyFields: [`Query: ${query}`, `Product: ${organism || "mouse"}`, `Results: ${genes.length}`, ...lines],
          sources: [
            `http://api.brain-map.org/api/v2/data/Gene/query.json`,
            "https://mouse.brain-map.org/",
          ],
          limitations: [
            "Only genes with ISH experiments in the selected atlas product are returned.",
            "Search is by name/acronym substring match — use the gene acronym for precise results.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "search_aba_structures",
  {
    description:
      "Searches the Allen Brain Atlas structure ontology for brain regions by name or acronym. Returns structure hierarchy, depth, acronym, and ID. Useful for identifying structure IDs needed by other ABA tools.",
    inputSchema: {
      query: z.string().describe("Brain structure name or acronym (e.g. 'hippocampus', 'CA1', 'thalamus')."),
      ontologyId: z
        .number()
        .optional()
        .describe("Ontology ID. Default 1 (adult mouse). Use 12 for developing mouse."),
      maxResults: z.number().optional().describe("Max results (default 20, max 100)."),
    },
  },
  async ({ query, ontologyId, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 100);
    const ontId = ontologyId || 1;

    async function searchStructures(term) {
      const encoded = encodeURIComponent(term);
      const nameUrl =
        `${ABA_API}/data/Structure/query.json` +
        `?criteria=[ontology_id$eq${ontId}],[name$il'*${encoded}*']` +
        `&num_rows=${limit}&order=depth`;
      const acronymUrl =
        `${ABA_API}/data/Structure/query.json` +
        `?criteria=[ontology_id$eq${ontId}],[acronym$il'*${encoded}*']` +
        `&num_rows=${limit}&order=depth`;
      const [byName, byAcronym] = await Promise.all([
        fetchJsonWithRetry(nameUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] })),
        fetchJsonWithRetry(acronymUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] })),
      ]);
      return [...(Array.isArray(byName.msg) ? byName.msg : []), ...(Array.isArray(byAcronym.msg) ? byAcronym.msg : [])];
    }

    let raw = await searchStructures(query);

    if (raw.length === 0 && query.length > 4) {
      const stem = query.replace(/(us|al|um|ar|ic|is|ine|eum|ular|ial)$/i, "");
      if (stem.length >= 4 && stem !== query) {
        raw = await searchStructures(stem);
      }
    }

    const seen = new Set();
    const structures = [];
    for (const s of raw) {
      if (!s?.id || seen.has(s.id)) continue;
      seen.add(s.id);
      structures.push(s);
      if (structures.length >= limit) break;
    }

    if (structures.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No brain structures found in the Allen ontology for "${query}" (ontology ${ontId}).`,
            keyFields: [`Query: ${query}`, `Ontology ID: ${ontId}`],
            sources: [`${ABA_API}/data/Structure/query.json`],
            limitations: ["Search uses case-insensitive LIKE matching on name and acronym."],
          }),
        }],
      };
    }

    const lines = structures.map((s, i) => {
      const path = s.structure_id_path || "";
      return `${String(i + 1).padStart(3)}. ${s.acronym} — ${s.name} (ID: ${s.id}, depth: ${s.depth}, color: #${s.color_hex_triplet || "000000"}, path: ${path})`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Allen Brain Atlas ontology: ${structures.length} structure(s) matching "${query}".`,
          keyFields: [`Query: ${query}`, `Ontology ID: ${ontId}`, `Results: ${structures.length}`, ...lines],
          sources: [
            `http://api.brain-map.org/api/v2/data/Structure/query.json`,
            "https://mouse.brain-map.org/",
          ],
          limitations: [
            "Structures are part of a hierarchical ontology; a parent structure's expression includes all descendants.",
            "Use the structure ID in get_aba_gene_expression for quantified expression queries.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_aba_gene_expression",
  {
    description:
      "Retrieves quantified gene expression data from the Allen Brain Atlas. Given a gene acronym, returns expression energy, density, and intensity across brain structures (StructureUnionize data). Optionally filter to a specific structure or sort by expression metric.",
    inputSchema: {
      geneAcronym: z.string().describe("Gene acronym (e.g. 'Pdyn', 'Gad1', 'Slc17a7')."),
      organism: z
        .enum(["mouse", "human", "developing_mouse", "mouse_connectivity"])
        .optional()
        .describe("Atlas product. Default 'mouse'."),
      structureId: z
        .number()
        .optional()
        .describe("Filter to a specific structure ID and its descendants. Omit for whole-brain."),
      planeOfSection: z
        .enum(["sagittal", "coronal"])
        .optional()
        .describe("Plane of section. Default: sagittal (delegate experiment)."),
      maxStructures: z.number().optional().describe("Max structures to return (default 25, max 200)."),
      sortBy: z
        .enum(["expression_energy", "expression_density"])
        .optional()
        .describe("Sort metric. Default 'expression_energy'."),
    },
  },
  async ({ geneAcronym, organism, structureId, planeOfSection, maxStructures, sortBy }) => {
    const productId = abaProductId(organism);
    const limit = Math.min(Math.max(1, maxStructures || 25), 200);
    const sortField = sortBy || "expression_energy";

    const planeFilter = planeOfSection
      ? `,plane_of_section[name$eq'${planeOfSection}']`
      : "";
    const datasetUrl =
      `${ABA_API}/data/SectionDataSet/query.json` +
      `?criteria=products[id$eq${productId}],genes[acronym$eq'${encodeURIComponent(geneAcronym)}']${planeFilter}` +
      `&include=genes` +
      `&num_rows=1`;

    const datasetRes = await fetchJsonWithRetry(datasetUrl, { timeoutMs: 15000 });
    const datasets = datasetRes.msg || [];

    if (datasets.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No ISH experiments found for gene "${geneAcronym}" in the Allen Brain Atlas (${organism || "mouse"}).`,
            keyFields: [`Gene: ${geneAcronym}`, `Product: ${organism || "mouse"}`],
            sources: [`${ABA_API}/data/SectionDataSet/query.json`],
            limitations: ["Ensure the gene acronym is valid and exists in the selected atlas product."],
          }),
        }],
      };
    }

    const dataset = datasets[0];
    const datasetId = dataset.id;

    let unionizeCriteria = `[section_data_set_id$eq${datasetId}]`;
    if (structureId) {
      unionizeCriteria += `,[structure_id$eq${structureId}]`;
    }

    const unionizeUrl =
      `${ABA_API}/data/StructureUnionize/query.json` +
      `?criteria=${unionizeCriteria}` +
      `&include=structure` +
      `&order=${sortField}$desc` +
      `&num_rows=${limit}`;

    const unionizeRes = await fetchJsonWithRetry(unionizeUrl, { timeoutMs: 20000 });
    const records = unionizeRes.msg || [];

    if (records.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No expression data returned for gene "${geneAcronym}" (dataset ${datasetId})${structureId ? ` in structure ${structureId}` : ""}.`,
            keyFields: [`Gene: ${geneAcronym}`, `Dataset ID: ${datasetId}`],
            sources: [`${ABA_API}/data/StructureUnionize/query.json`],
            limitations: ["The dataset may lack unionize data for the specified structure."],
          }),
        }],
      };
    }

    const plane = planeOfSection || (dataset.plane_of_section_id === 1 ? "coronal" : "sagittal");
    const geneName = dataset.genes?.[0]?.name || geneAcronym;

    const lines = records.map((r, i) => {
      const s = r.structure || {};
      const energy = r.expression_energy != null ? r.expression_energy.toFixed(4) : "N/A";
      const density = r.expression_density != null ? r.expression_density.toFixed(6) : "N/A";
      return `${String(i + 1).padStart(3)}. ${(s.acronym || "?").padEnd(12)} ${(s.name || "unknown").padEnd(45)} energy=${energy}  density=${density}`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Allen Brain Atlas expression for ${geneAcronym} (${geneName}): top ${records.length} structures by ${sortField} (${plane}, dataset ${datasetId}).`,
          keyFields: [
            `Gene: ${geneAcronym} (${geneName})`,
            `Dataset ID: ${datasetId}`,
            `Plane: ${plane}`,
            `Sorted by: ${sortField}`,
            `Structures returned: ${records.length}`,
            `\nExpression by structure:`,
            ...lines,
          ],
          sources: [
            `http://mouse.brain-map.org/experiment/show/${datasetId}`,
            `http://api.brain-map.org/api/v2/data/StructureUnionize/query.json`,
            "https://mouse.brain-map.org/",
          ],
          limitations: [
            "Expression values are derived from automated ISH image analysis — false positives/negatives can occur.",
            "Expression energy = intensity × density; compare within the same experiment for consistency.",
            "Only the delegate (highest quality) experiment is queried unless a specific plane is requested.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "search_aba_differential_expression",
  {
    description:
      "Finds genes differentially expressed between two brain structures in the Allen Mouse Brain Atlas. Uses the mouse_differential connected service to compute fold-change of expression energy in a target structure vs. a contrast structure. Returns genes ranked by enrichment in the target region.",
    inputSchema: {
      targetStructure: z.string().describe("Target structure name (e.g. 'thalamus'). Genes enriched here will be returned."),
      contrastStructure: z.string().describe("Contrast structure name (e.g. 'isocortex'). Used as the comparison baseline."),
      startRow: z.number().optional().describe("Pagination offset (default 0)."),
      numRows: z.number().optional().describe("Number of results (default 25, max 100)."),
    },
  },
  async ({ targetStructure, contrastStructure, startRow, numRows }) => {
    const limit = Math.min(Math.max(1, numRows || 25), 100);
    const offset = Math.max(0, startRow || 0);

    async function lookupStructure(name) {
      const encoded = encodeURIComponent(name);
      const normalizedQuery = normalizeWhitespace(name).toLowerCase();
      const queryStem = normalizedQuery.replace(/(us|al|um|ar|ic|is|ine|eum|ular|ial|s)$/i, "");
      const coverageCache = new Map();

      function structureScore(s) {
        const nameValue = normalizeWhitespace(s?.name || "").toLowerCase();
        const acronymValue = normalizeWhitespace(s?.acronym || "").toLowerCase();
        const safeNameValue = normalizeWhitespace(s?.safe_name || "").toLowerCase();
        const depth = Number.isFinite(Number(s?.depth)) ? Number(s.depth) : 0;
        const stLevel = Number.isFinite(Number(s?.st_level)) ? Number(s.st_level) : 0;

        let score = 0;
        if (nameValue === normalizedQuery) score += 1000;
        if (safeNameValue === normalizedQuery) score += 900;
        if (acronymValue === normalizedQuery) score += 850;
        if (nameValue.startsWith(normalizedQuery)) score += 700;
        if (nameValue.includes(` ${normalizedQuery}`) || nameValue.includes(`${normalizedQuery} `)) score += 600;
        if (nameValue.includes(normalizedQuery)) score += 500;
        if (queryStem && nameValue.includes(queryStem)) score += 420;
        if (queryStem && safeNameValue.includes(queryStem)) score += 380;
        if (queryStem && acronymValue.includes(queryStem)) score += 300;

        // Prefer canonical anatomical nodes over very specific tracts/surfaces for broad queries.
        if (/region|formation|nucleus|area|complex|cortex/.test(nameValue)) score += 110;
        if (/fissure|commissure|tract|bundle|ventricle|fiber|sulcus/.test(nameValue)) score -= 220;
        if (/transition|retro|septo/.test(nameValue)) score -= 120;

        // Slight preference for mid-level ontology nodes.
        score += Math.min(60, Math.max(0, stLevel));
        score -= Math.max(0, depth - 9) * 5;
        return score;
      }

      async function getCoverageScore(structureId) {
        if (coverageCache.has(structureId)) return coverageCache.get(structureId);
        const url =
          `${ABA_API}/data/StructureUnionize/query.json` +
          `?criteria=[structure_id$eq${structureId}]&num_rows=1`;
        const res = await fetchJsonWithRetry(url, { timeoutMs: 15000 }).catch(() => null);
        const total = Number.isFinite(Number(res?.total_rows)) ? Number(res.total_rows) : 0;
        coverageCache.set(structureId, total);
        return total;
      }

      async function pickBestCandidate(candidates) {
        if (!Array.isArray(candidates) || candidates.length === 0) return null;
        if (candidates.length === 1) return candidates[0];

        const ranked = candidates
          .map((s) => ({ structure: s, lexical: structureScore(s) }))
          .sort((a, b) => b.lexical - a.lexical);

        const shortlist = ranked.slice(0, Math.min(6, ranked.length));
        const withCoverage = await Promise.all(
          shortlist.map(async (entry) => ({
            ...entry,
            coverage: await getCoverageScore(entry.structure.id),
          }))
        );

        withCoverage.sort((a, b) => {
          if (b.lexical !== a.lexical) return b.lexical - a.lexical;
          if (b.coverage !== a.coverage) return b.coverage - a.coverage;
          return 0;
        });
        return withCoverage[0]?.structure || ranked[0]?.structure || null;
      }

      const exactUrl =
        `${ABA_API}/data/Structure/query.json` +
        `?criteria=[ontology_id$eq1],[name$il'${encoded}']` +
        `&num_rows=25`;
      const exactRes = await fetchJsonWithRetry(exactUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] }));
      const exactRows = Array.isArray(exactRes.msg) ? exactRes.msg : [];
      if (exactRows.length > 0) {
        return pickBestCandidate(exactRows);
      }

      const wildcardUrl =
        `${ABA_API}/data/Structure/query.json` +
        `?criteria=[ontology_id$eq1],[name$il'*${encoded}*']` +
        `&num_rows=50`;
      const wildcardRes = await fetchJsonWithRetry(wildcardUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] }));
      const wildcardRows = Array.isArray(wildcardRes.msg) ? wildcardRes.msg : [];
      if (wildcardRows.length > 0) {
        return pickBestCandidate(wildcardRows);
      }

      if (name.length > 4) {
        const stem = name.replace(/(us|al|um|ar|ic|is|ine|eum|ular|ial|s)$/i, "");
        if (stem.length >= 4 && stem !== name) {
          const stemUrl =
            `${ABA_API}/data/Structure/query.json` +
            `?criteria=[ontology_id$eq1],[name$il'*${encodeURIComponent(stem)}*']` +
            `&num_rows=50`;
          const stemRes = await fetchJsonWithRetry(stemUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] }));
          const stemRows = Array.isArray(stemRes.msg) ? stemRes.msg : [];
          if (stemRows.length > 0) {
            return pickBestCandidate(stemRows);
          }
        }
      }

      const acronymUrl =
        `${ABA_API}/data/Structure/query.json` +
        `?criteria=[ontology_id$eq1],[acronym$il'*${encoded}*']` +
        `&num_rows=25`;
      const acronymRes = await fetchJsonWithRetry(acronymUrl, { timeoutMs: 15000 }).catch(() => ({ msg: [] }));
      const acronymRows = Array.isArray(acronymRes.msg) ? acronymRes.msg : [];
      if (acronymRows.length > 0) {
        return pickBestCandidate(acronymRows);
      }

      return null;
    }

    const [targetStruct, contrastStruct] = await Promise.all([
      lookupStructure(targetStructure),
      lookupStructure(contrastStructure),
    ]);

    if (!targetStruct) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Could not find target structure "${targetStructure}" in the Allen Brain Atlas ontology.`,
            keyFields: [`Query: ${targetStructure}`],
            sources: [`${ABA_API}/data/Structure/query.json`],
            limitations: ["Use search_aba_structures to find valid structure names. Allen uses terms like 'Hippocampal region' rather than 'hippocampus'."],
          }),
        }],
      };
    }
    if (!contrastStruct) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Could not find contrast structure "${contrastStructure}" in the Allen Brain Atlas ontology.`,
            keyFields: [`Query: ${contrastStructure}`],
            sources: [`${ABA_API}/data/Structure/query.json`],
            limitations: ["Use search_aba_structures to find valid structure names. Allen uses terms like 'Hippocampal region' rather than 'hippocampus'."],
          }),
        }],
      };
    }

    function normalizeServiceRows(rows) {
      return rows.map((r) => ({
        geneSymbol: r["gene-symbol"] || "",
        geneName: r["gene-name"] || "",
        foldChange: Number.parseFloat(r["fold-change"]),
        targetValue: Number.parseFloat(r["target-sum"]),
        contrastValue: Number.parseFloat(r["contrast-sum"]),
        plane: r["plane-of-section"] || "",
        source: "mouse_differential_service",
      }));
    }

    async function fetchServiceRowsWithRetries(maxAttempts = 4) {
      const diffUrl =
        `${ABA_API}/data/query.json` +
        `?criteria=service::mouse_differential` +
        `[set$eqmouse]` +
        `[structures1$eq${contrastStruct.id}][threshold1$eq0,50]` +
        `[structures2$eq${targetStruct.id}][threshold2$eq1,50]`;
      let lastErr = "";
      for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        const res = await fetchJsonWithRetry(diffUrl, { timeoutMs: 45000 }).catch((err) => ({
          msg: `Request error: ${err?.message || String(err)}`,
        }));
        if (Array.isArray(res?.msg)) {
          return { rows: normalizeServiceRows(res.msg), error: "" };
        }
        lastErr = typeof res?.msg === "string" ? res.msg : "Unknown differential service error.";
        if (attempt < maxAttempts) {
          await sleep(Math.min(5000, 700 * 2 ** (attempt - 1)) + Math.floor(Math.random() * 250));
        }
      }
      return { rows: [], error: lastErr || "Differential service unavailable after retries." };
    }

    const serviceAttempt = await fetchServiceRowsWithRetries();
    const allRows = serviceAttempt.rows;
    if (allRows.length === 0) {
      const errMsg = serviceAttempt.error || "Unknown Allen Brain Atlas differential service error.";
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Allen Brain Atlas differential service error: ${errMsg.slice(0, 200)}`,
            keyFields: [
              `Target: ${targetStruct.name} (${targetStruct.acronym}, ID: ${targetStruct.id})`,
              `Contrast: ${contrastStruct.name} (${contrastStruct.acronym}, ID: ${contrastStruct.id})`,
            ],
            sources: [`${ABA_API}/data/query.json?criteria=service::mouse_differential`],
            limitations: [
              "The differential service may be temporarily unavailable.",
              "No local fallback approximation is enabled.",
            ],
          }),
        }],
      };
    }

    const results = allRows.slice(offset, offset + limit);

    if (results.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No differentially expressed genes found between ${targetStruct.name} (target) and ${contrastStruct.name} (contrast)${offset > 0 ? ` at offset ${offset}` : ""}.`,
            keyFields: [
              `Target: ${targetStruct.name} (${targetStruct.acronym}, ID: ${targetStruct.id})`,
              `Contrast: ${contrastStruct.name} (${contrastStruct.acronym}, ID: ${contrastStruct.id})`,
              `Total rows available: ${allRows.length}`,
            ],
            sources: [`${ABA_API}/data/query.json?criteria=service::mouse_differential`],
            limitations: ["The differential service compares expression energy averaged over voxels."],
          }),
        }],
      };
    }

    const lines = results.map((r, i) => {
      const fc = Number.isFinite(r.foldChange) ? r.foldChange.toFixed(2) : "N/A";
      const targetValue = Number.isFinite(r.targetValue) ? r.targetValue.toFixed(2) : "N/A";
      const contrastValue = Number.isFinite(r.contrastValue) ? r.contrastValue.toFixed(2) : "N/A";
      const gene = (r.geneSymbol || `ID:${i + offset + 1}`).slice(0, 14);
      const geneName = (r.geneName || "").slice(0, 38);
      return `${String(i + 1 + offset).padStart(3)}. ${gene.padEnd(14)} ${geneName.padEnd(40)} FC=${fc}  target=${targetValue}  contrast=${contrastValue}  (${r.plane || ""})`;
    });
    const limitations = [
      "Fold-change is based on automated ISH expression energy; visual inspection of images is recommended for confirmation.",
      "The differential service uses a subset of voxels loaded in memory (~26,000) and may miss low-abundance transcripts.",
      "This service is available only for the Mouse Brain Atlas (adult ISH, ~25,000 datasets).",
    ];

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Allen Brain Atlas differential expression: ${allRows.length} total genes enriched in ${targetStruct.name} (${targetStruct.acronym}) vs ${contrastStruct.name} (${contrastStruct.acronym}). Showing ${offset + 1}–${offset + results.length}.`,
          keyFields: [
            `Target: ${targetStruct.name} (${targetStruct.acronym}, ID: ${targetStruct.id})`,
            `Contrast: ${contrastStruct.name} (${contrastStruct.acronym}, ID: ${contrastStruct.id})`,
            `Total genes: ${allRows.length}`,
            `Showing: ${offset + 1}–${offset + results.length}`,
            `\nDifferentially expressed genes (by fold-change):`,
            ...lines,
          ],
          sources: [
            `http://mouse.brain-map.org/search/show?search_type=differential&domain1=${contrastStruct.id}&domain2=${targetStruct.id}`,
            `http://api.brain-map.org/api/v2/data/query.json?criteria=service::mouse_differential`,
            "https://mouse.brain-map.org/",
          ],
          limitations,
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// EBRAINS Knowledge Graph tools
// ---------------------------------------------------------------------------

const EBRAINS_KG_VALID_TYPES = new Set([
  "Dataset", "Model", "Software", "Contributor", "Project",
  "Learning Resource", "Web service", "Workflow", "(Meta)Data Model",
]);

server.registerTool(
  "search_ebrains_kg",
  {
    description:
      "Searches the EBRAINS Knowledge Graph for neuroscience datasets, computational models, software tools, contributors, and projects. The Knowledge Graph aggregates curated, FAIR-compliant neuroscience resources from the Human Brain Project and EBRAINS ecosystem.",
    inputSchema: {
      query: z.string().describe("Search query (e.g. 'hippocampus', 'Parkinson', 'EEG resting state')."),
      type: z
        .enum(["Dataset", "Model", "Software", "Contributor", "Project", "Learning Resource", "Web service", "Workflow", "(Meta)Data Model"])
        .optional()
        .describe("Filter by resource type. Omit to search across all types."),
      from: z.number().optional().describe("Pagination offset (default 0)."),
      size: z.number().optional().describe("Number of results (default 20, max 50)."),
    },
  },
  async ({ query, type, from: offset, size }) => {
    const limit = Math.min(Math.max(1, size || 20), 50);
    const pageFrom = Math.max(0, offset || 0);
    const typeParam = type && EBRAINS_KG_VALID_TYPES.has(type) ? `&type=${encodeURIComponent(type)}` : "";

    const url =
      `${EBRAINS_KG_SEARCH_API}/groups/public/search` +
      `?q=${encodeURIComponent(query)}${typeParam}&from=${pageFrom}&size=${limit}`;

    const res = await fetchWithRetry(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
      timeoutMs: 20000,
    });
    const data = await res.json();

    const hits = data.hits || [];
    const total = data.total || 0;
    const types = data.types || {};

    if (hits.length === 0) {
      const typeCounts = Object.entries(types)
        .map(([t, v]) => `${t}: ${v.count}`)
        .join(", ");
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No EBRAINS Knowledge Graph results for "${query}"${type ? ` (type: ${type})` : ""}.${typeCounts ? ` Available counts across types: ${typeCounts}` : ""}`,
            keyFields: [`Query: ${query}`, type ? `Type filter: ${type}` : "Type: all"],
            sources: ["https://search.kg.ebrains.eu/"],
            limitations: ["Only publicly released resources are searched."],
          }),
        }],
      };
    }

    const lines = hits.map((hit, i) => {
      const title = hit.title || "Untitled";
      const hitType = hit.type || hit.category || "Unknown";
      const desc = hit.fields?.description?.value || "";
      const snippet = desc.length > 150 ? desc.slice(0, 147) + "..." : desc;
      const tags = (hit.tags?.data || []).slice(0, 5).join(", ");
      const doi = hit.fields?.citation?.value ? `DOI: ${hit.fields.citation.value}` : "";
      const access = hit.fields?.dataAccessibility?.value || "";
      const parts = [
        `${String(i + 1 + pageFrom).padStart(3)}. [${hitType}] ${title}`,
        `     ID: ${hit.id}`,
      ];
      if (doi) parts.push(`     ${doi}`);
      if (access) parts.push(`     Access: ${access}`);
      if (tags) parts.push(`     Tags: ${tags}`);
      if (snippet) parts.push(`     ${snippet}`);
      return parts.join("\n");
    });

    const typeSummary = Object.entries(types)
      .sort((a, b) => b[1].count - a[1].count)
      .map(([t, v]) => `${t}: ${v.count}`)
      .join(", ");

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `EBRAINS Knowledge Graph: ${total} result(s) for "${query}"${type ? ` (type: ${type})` : ""}. Showing ${pageFrom + 1}–${pageFrom + hits.length}.`,
          keyFields: [
            `Query: ${query}`,
            type ? `Type filter: ${type}` : "Type: all",
            `Total: ${total}`,
            `Type breakdown: ${typeSummary}`,
            `\nResults:`,
            ...lines,
          ],
          sources: [
            `https://search.kg.ebrains.eu/?q=${encodeURIComponent(query)}${type ? `&category=${encodeURIComponent(type)}` : ""}`,
            "https://search.kg.ebrains.eu/",
          ],
          limitations: [
            "Only publicly released (curated) resources are included in search results.",
            "Use get_ebrains_kg_document with the ID and type to retrieve full metadata, DOIs, authors, and file information.",
            "EBRAINS primarily hosts data from the Human Brain Project and European neuroscience initiatives.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_ebrains_kg_document",
  {
    description:
      "Retrieves detailed metadata for a specific resource in the EBRAINS Knowledge Graph by its type and ID. Returns full description, authors, DOI/citation, data accessibility, techniques, brain regions, species, and linked resources.",
    inputSchema: {
      type: z
        .enum(["Dataset", "Model", "Software", "Contributor", "Project", "Learning Resource", "Web service", "Workflow", "(Meta)Data Model"])
        .describe("Resource type (e.g. 'Dataset', 'Model', 'Software')."),
      id: z.string().describe("EBRAINS KG document UUID (e.g. '885b4936-9345-43bd-880e-eebc19898ded')."),
    },
  },
  async ({ type, id }) => {
    const url = `${EBRAINS_KG_SEARCH_API}/groups/public/documents/${encodeURIComponent(type)}/${encodeURIComponent(id)}`;
    const res = await fetchWithRetry(url, { timeoutMs: 15000 });

    if (!res.ok) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `EBRAINS KG document not found: ${type}/${id} (HTTP ${res.status}).`,
            keyFields: [`Type: ${type}`, `ID: ${id}`],
            sources: [`https://search.kg.ebrains.eu/instances/${id}`],
            limitations: ["The document may not be publicly released or the ID may be incorrect."],
          }),
        }],
      };
    }

    const doc = await res.json();
    const title = doc.title || doc.meta?.name || "Untitled";
    const description = doc.fields?.description?.value || doc.meta?.description || "";
    const fields = doc.fields || {};

    const keyFields = [`Title: ${title}`, `Type: ${doc.type || type}`, `ID: ${doc.id || id}`];

    const identifiers = doc.meta?.identifier || [];
    for (const ident of identifiers) {
      if (typeof ident === "string" && ident.startsWith("https://doi.org/")) {
        keyFields.push(`DOI: ${ident}`);
      }
    }

    if (fields.citation?.value) keyFields.push(`Citation DOI: ${fields.citation.value}`);
    if (fields.releasedAt?.value) keyFields.push(`Released: ${fields.releasedAt.value}`);
    if (fields.dataAccessibility?.value) keyFields.push(`Access: ${fields.dataAccessibility.value}`);

    const custodians = (fields.custodians || []).map((c) => c.value).filter(Boolean);
    if (custodians.length > 0) keyFields.push(`Custodians: ${custodians.join(", ")}`);

    const authors = (doc.meta?.creator || []).map((a) => a.name).filter(Boolean);
    if (authors.length > 0) keyFields.push(`Authors: ${authors.join(", ")}`);

    const techniques = (fields.technique || []).map((t) => t.value || t.children?.map((c) => c.value).join(", ")).filter(Boolean);
    if (techniques.length > 0) keyFields.push(`Techniques: ${techniques.join(", ")}`);

    const keywords = (fields.keywords || []).map((k) => k.value).filter(Boolean);
    if (keywords.length > 0) keyFields.push(`Keywords: ${keywords.join(", ")}`);

    const tags = (doc.tags?.data || []);
    if (tags.length > 0) keyFields.push(`Tags: ${tags.join(", ")}`);

    const versions = (fields.versions || []).map((v) => `${v.value || ""}${v.reference ? ` (${v.reference})` : ""}`).filter(Boolean);
    if (versions.length > 0) keyFields.push(`Versions: ${versions.join("; ")}`);

    if (description) {
      const descSnippet = description.length > 800 ? description.slice(0, 797) + "..." : description;
      keyFields.push(`\nDescription:\n${descSnippet}`);
    }

    const sources = [`https://search.kg.ebrains.eu/instances/${id}`];
    const doiIdent = identifiers.find((x) => typeof x === "string" && x.startsWith("https://doi.org/"));
    if (doiIdent) sources.push(doiIdent);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `EBRAINS KG: ${title} (${doc.type || type}).`,
          keyFields,
          sources,
          limitations: [
            "Only publicly released metadata is available. Some linked resources may require EBRAINS account access.",
            "File downloads may require separate authentication via the EBRAINS data proxy.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// CONP (Canadian Open Neuroscience Platform) dataset tools
// ---------------------------------------------------------------------------

server.registerTool(
  "search_conp_datasets",
  {
    description:
      "Searches public CONP (Canadian Open Neuroscience Platform) dataset repositories. " +
      "CONP datasets are GitHub repositories in the conpdatasets organization, indexed by repository name, description, and topics — NOT by disease or diagnosis terms. " +
      "Use short neuroscience keywords like 'EEG', 'fMRI', 'MRI', 'PET', 'MEG', 'resting state', 'brain', 'PREVENT-AD', 'POND', 'phantom', or study/cohort names. " +
      "Disease-specific queries (e.g. 'Parkinson disease') rarely match because repos are named after projects, not conditions. " +
      "If a disease-specific search returns no results, retry with broader modality or method terms, or browse all repos with a single-word query like 'brain' or 'neuro'.",
    inputSchema: {
      query: z.string().describe("Short keyword for repo name/description/topics. Use modality or method terms (e.g. 'EEG', 'fMRI', 'MRI', 'PET'), study names (e.g. 'PREVENT-AD', 'POND'), or broad terms ('brain', 'neuro'). Avoid full disease names."),
      sortBy: z.enum(["updated", "stars", "name"]).optional().describe("Result ordering. Default 'updated'."),
      maxResults: z.number().optional().describe("Maximum results to return (default 20, max 50)."),
    },
  },
  async ({ query, sortBy, maxResults }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a CONP dataset search query." }] };
    }

    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const mode = String(sortBy || "updated").toLowerCase();
    const q = `${normalizedQuery} org:${CONP_GITHUB_ORG}`;
    const params = new URLSearchParams({
      q,
      per_page: String(limit),
      page: "1",
    });
    if (mode === "updated" || mode === "stars") {
      params.set("sort", mode);
      params.set("order", "desc");
    }

    const url = `${GITHUB_API}/search/repositories?${params.toString()}`;
    let data;
    try {
      data = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 12000,
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "research-mcp",
        },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `CONP dataset search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [`Query: ${normalizedQuery}`, `Organization: ${CONP_GITHUB_ORG}`],
            sources: ["https://github.com/conpdatasets"],
            limitations: [
              "GitHub API rate limits may apply to unauthenticated requests.",
            ],
          }),
        }],
      };
    }

    const items = Array.isArray(data?.items) ? data.items : [];
    const total = Number(data?.total_count || 0);
    if (items.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No CONP datasets found for "${normalizedQuery}".`,
            keyFields: [`Query: ${normalizedQuery}`, `Organization: ${CONP_GITHUB_ORG}`],
            sources: ["https://github.com/conpdatasets"],
            limitations: [
              "Search is performed against GitHub repository metadata (name, description, topics).",
            ],
          }),
        }],
      };
    }

    const lines = items.map((repo, idx) => {
      const name = repo?.name || "unknown";
      const desc = normalizeWhitespace(repo?.description || "");
      const descPreview = desc ? (desc.length > 140 ? `${desc.slice(0, 137)}...` : desc) : "No description.";
      const stars = Number(repo?.stargazers_count || 0);
      const updatedAt = repo?.updated_at ? String(repo.updated_at).slice(0, 10) : "unknown";
      const language = repo?.language || "n/a";
      const topics = Array.isArray(repo?.topics) ? repo.topics.slice(0, 4).join(", ") : "";
      const parts = [
        `${String(idx + 1).padStart(3)}. ${name}`,
        `     Stars: ${stars} | Language: ${language} | Updated: ${updatedAt}`,
        `     URL: ${repo?.html_url || "n/a"}`,
        `     ${descPreview}`,
      ];
      if (topics) parts.push(`     Topics: ${topics}`);
      return parts.join("\n");
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `CONP datasets: ${total} repository match(es) for "${normalizedQuery}" in ${CONP_GITHUB_ORG}. Showing top ${items.length}.`,
          keyFields: [
            `Query: ${normalizedQuery}`,
            `Organization: ${CONP_GITHUB_ORG}`,
            `Sort: ${mode}`,
            `Total matches: ${total}`,
            "\nResults:",
            ...lines,
          ],
          sources: [
            `https://github.com/search?q=${encodeURIComponent(q)}&type=repositories`,
            "https://github.com/conpdatasets",
            "https://conp.ca/",
          ],
          limitations: [
            "This tool indexes CONP datasets represented as GitHub repositories in the conpdatasets organization.",
            "Repository metadata is not a substitute for full dataset documentation and usage terms.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_conp_dataset_details",
  {
    description:
      "Fetches detailed metadata for a specific CONP dataset repository, including README preview, license, stars, topics, and links. " +
      "Use this after search_conp_datasets returns repository names you want to inspect further.",
    inputSchema: {
      repo: z.string().describe("Repository name from search_conp_datasets results (e.g. 'preventad-open', 'SIMON-dataset'). Can also be full path like 'conpdatasets/preventad-open'."),
    },
  },
  async ({ repo }) => {
    const rawRepo = normalizeWhitespace(repo || "");
    if (!rawRepo) {
      return { content: [{ type: "text", text: "Provide a CONP repository name (e.g. preventad-open)." }] };
    }
    const repoName = rawRepo.includes("/") ? rawRepo.split("/").pop() : rawRepo;
    if (!repoName) {
      return { content: [{ type: "text", text: "Unable to parse repository name." }] };
    }

    const repoUrl = `${GITHUB_API}/repos/${CONP_GITHUB_ORG}/${encodeURIComponent(repoName)}`;
    let repoData;
    try {
      repoData = await fetchJsonWithRetry(repoUrl, {
        retries: 1,
        timeoutMs: 12000,
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "research-mcp",
        },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `CONP dataset repository not found: ${CONP_GITHUB_ORG}/${repoName}.`,
            keyFields: [
              `Repository: ${CONP_GITHUB_ORG}/${repoName}`,
              `Error: ${compactErrorMessage(error?.message || "unknown error", 220)}`,
            ],
            sources: [
              `https://github.com/${CONP_GITHUB_ORG}/${repoName}`,
            ],
            limitations: [
              "The repository may be private, renamed, or unavailable.",
            ],
          }),
        }],
      };
    }

    let readmePreview = "";
    try {
      const readmeUrl = `${GITHUB_API}/repos/${CONP_GITHUB_ORG}/${encodeURIComponent(repoName)}/readme`;
      const readmeResponse = await fetchWithRetry(readmeUrl, {
        retries: 1,
        timeoutMs: 10000,
        headers: {
          Accept: "application/vnd.github.raw",
          "User-Agent": "research-mcp",
        },
      });
      const readmeText = await readmeResponse.text();
      if (readmeText) {
        const cleaned = normalizeWhitespace(readmeText.replace(/[#*_`>-]/g, " "));
        readmePreview = cleaned.length > 420 ? `${cleaned.slice(0, 417)}...` : cleaned;
      }
    } catch (_) {
      // README preview is best-effort only.
    }

    const topics = Array.isArray(repoData?.topics) ? repoData.topics : [];
    const keyFields = [
      `Repository: ${repoData?.full_name || `${CONP_GITHUB_ORG}/${repoName}`}`,
      `Description: ${normalizeWhitespace(repoData?.description || "No description.")}`,
      `Stars: ${Number(repoData?.stargazers_count || 0)}`,
      `Watchers: ${Number(repoData?.subscribers_count || 0)}`,
      `Open issues: ${Number(repoData?.open_issues_count || 0)}`,
      `Default branch: ${repoData?.default_branch || "unknown"}`,
      `Updated: ${repoData?.updated_at ? String(repoData.updated_at).slice(0, 10) : "unknown"}`,
    ];
    if (repoData?.license?.name) keyFields.push(`License: ${repoData.license.name}`);
    if (topics.length > 0) keyFields.push(`Topics: ${topics.slice(0, 8).join(", ")}`);
    if (repoData?.homepage) keyFields.push(`Homepage: ${repoData.homepage}`);
    if (readmePreview) keyFields.push(`\nREADME preview:\n${readmePreview}`);

    const sources = [
      repoData?.html_url || `https://github.com/${CONP_GITHUB_ORG}/${repoName}`,
      "https://github.com/conpdatasets",
      "https://conp.ca/",
    ];

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `CONP dataset details retrieved for ${repoData?.full_name || `${CONP_GITHUB_ORG}/${repoName}`}.`,
          keyFields,
          sources,
          limitations: [
            "Metadata is sourced from GitHub repository records and may not include complete participant-level access constraints.",
            "Many CONP datasets require reviewing dataset-specific usage terms before downloading or analysis.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// NEMAR (NeuroElectroMagnetic data Archive) dataset tools
// ---------------------------------------------------------------------------

server.registerTool(
  "search_nemar_datasets",
  {
    description:
      "Searches NEMAR (NeuroElectroMagnetic data Archive) for EEG, MEG, and iEEG datasets. " +
      "NEMAR hosts OpenNeuro neuroelectromagnetic data at SDSC with BIDS format, HED event descriptions, and NSG compute integration. " +
      "Datasets are GitHub repos in the nemarDatasets organization. Use keywords like 'EEG', 'MEG', 'iEEG', 'resting state', 'visual', 'auditory', or study names. " +
      "Omit query or use 'EEG' to browse. Use get_nemar_dataset_details with a repo name (e.g. nm000104) for full metadata.",
    inputSchema: {
      query: z.string().optional().describe("Search term for repo name/description/topics. Use 'EEG', 'MEG', 'iEEG', 'resting', 'visual', or study keywords. Omit to list recent datasets."),
      sortBy: z.enum(["updated", "stars", "name"]).optional().describe("Result ordering. Default 'updated'."),
      maxResults: z.number().optional().describe("Maximum results (default 20, max 50)."),
    },
  },
  async ({ query, sortBy, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const mode = String(sortBy || "updated").toLowerCase();
    const q = normalizeWhitespace(query || "");
    const searchQ = q ? `${q} org:${NEMAR_GITHUB_ORG}` : `org:${NEMAR_GITHUB_ORG}`;
    const params = new URLSearchParams({
      q: searchQ,
      per_page: String(limit),
      page: "1",
    });
    if (mode === "updated" || mode === "stars") {
      params.set("sort", mode);
      params.set("order", "desc");
    }

    const url = `${GITHUB_API}/search/repositories?${params.toString()}`;
    let data;
    try {
      data = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 12000,
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "research-mcp",
        },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `NEMAR dataset search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [q ? `Query: ${q}` : "Browse: all", `Organization: ${NEMAR_GITHUB_ORG}`],
            sources: ["https://nemar.org/", "https://github.com/nemarDatasets"],
            limitations: ["GitHub API rate limits may apply to unauthenticated requests."],
          }),
        }],
      };
    }

    const items = Array.isArray(data?.items) ? data.items : [];
    const total = Number(data?.total_count || 0);
    if (items.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No NEMAR datasets found${q ? ` for "${q}"` : ""}.`,
            keyFields: [q ? `Query: ${q}` : "Browse: all", `Organization: ${NEMAR_GITHUB_ORG}`],
            sources: ["https://nemar.org/discover", "https://github.com/nemarDatasets"],
            limitations: ["Try 'EEG', 'MEG', or 'iEEG' to find modality-specific datasets."],
          }),
        }],
      };
    }

    const lines = items.map((repo, idx) => {
      const name = repo?.name || "unknown";
      const desc = normalizeWhitespace(repo?.description || "");
      const descPreview = desc ? (desc.length > 100 ? `${desc.slice(0, 97)}...` : desc) : "No description.";
      const stars = Number(repo?.stargazers_count || 0);
      const updatedAt = repo?.updated_at ? String(repo.updated_at).slice(0, 10) : "unknown";
      return `  ${String(idx + 1).padStart(3)}. ${name} — ${descPreview} | stars: ${stars} | updated: ${updatedAt}`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `NEMAR: ${items.length} dataset(s)${q ? ` matching "${q}"` : ""} in ${NEMAR_GITHUB_ORG}.`,
          keyFields: [
            q ? `Query: ${q}` : "Browse: all",
            `Total matches: ${total}`,
            "\nResults:",
            ...lines,
          ],
          sources: [
            "https://nemar.org/discover",
            "https://nemar.org/dataexplorer",
            "https://github.com/nemarDatasets",
          ],
          limitations: [
            "NEMAR datasets are EEG/MEG/iEEG from OpenNeuro, hosted at SDSC. Use get_nemar_dataset_details with repo name for full metadata.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_nemar_dataset_details",
  {
    description:
      "Fetches detailed metadata for a NEMAR dataset repository (e.g. nm000104). " +
      "Returns description, stars, topics, README preview, and links. Use after search_nemar_datasets.",
    inputSchema: {
      repo: z.string().describe("Repository name from search_nemar_datasets (e.g. 'nm000104'). Can be full path 'nemarDatasets/nm000104'."),
    },
  },
  async ({ repo }) => {
    const rawRepo = normalizeWhitespace(repo || "");
    if (!rawRepo) {
      return { content: [{ type: "text", text: "Provide a NEMAR dataset repo name (e.g. nm000104)." }] };
    }
    const repoName = rawRepo.includes("/") ? rawRepo.split("/").pop() : rawRepo;
    if (!repoName) {
      return { content: [{ type: "text", text: "Unable to parse repository name." }] };
    }

    const repoUrl = `${GITHUB_API}/repos/${NEMAR_GITHUB_ORG}/${encodeURIComponent(repoName)}`;
    let repoData;
    try {
      repoData = await fetchJsonWithRetry(repoUrl, {
        retries: 1,
        timeoutMs: 12000,
        headers: {
          Accept: "application/vnd.github+json",
          "User-Agent": "research-mcp",
        },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `NEMAR dataset not found: ${NEMAR_GITHUB_ORG}/${repoName}.`,
            keyFields: [`Repository: ${NEMAR_GITHUB_ORG}/${repoName}`, `Error: ${compactErrorMessage(error?.message || "unknown error", 220)}`],
            sources: [`https://github.com/${NEMAR_GITHUB_ORG}/${repoName}`, "https://nemar.org/"],
            limitations: ["Repository may be private or the name incorrect."],
          }),
        }],
      };
    }

    let readmePreview = "";
    try {
      const readmeUrl = `${GITHUB_API}/repos/${NEMAR_GITHUB_ORG}/${encodeURIComponent(repoName)}/readme`;
      const readmeResponse = await fetchWithRetry(readmeUrl, {
        retries: 1,
        timeoutMs: 10000,
        headers: { Accept: "application/vnd.github.raw", "User-Agent": "research-mcp" },
      });
      const readmeText = await readmeResponse.text();
      if (readmeText) {
        const cleaned = normalizeWhitespace(readmeText.replace(/[#*_`>-]/g, " "));
        readmePreview = cleaned.length > 400 ? `${cleaned.slice(0, 397)}...` : cleaned;
      }
    } catch (_) {}

    const topics = Array.isArray(repoData?.topics) ? repoData.topics : [];
    const keyFields = [
      `Repository: ${repoData?.full_name || `${NEMAR_GITHUB_ORG}/${repoName}`}`,
      `Description: ${normalizeWhitespace(repoData?.description || "No description.")}`,
      `Stars: ${Number(repoData?.stargazers_count || 0)}`,
      `Updated: ${repoData?.updated_at ? String(repoData.updated_at).slice(0, 10) : "unknown"}`,
    ];
    if (topics.length > 0) keyFields.push(`Topics: ${topics.slice(0, 8).join(", ")}`);
    if (readmePreview) keyFields.push(`\nREADME preview:\n${readmePreview}`);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `NEMAR dataset: ${repoData?.full_name || `${NEMAR_GITHUB_ORG}/${repoName}`}.`,
          keyFields,
          sources: [
            repoData?.html_url || `https://github.com/${NEMAR_GITHUB_ORG}/${repoName}`,
            "https://nemar.org/discover",
            "https://nemar.org/",
          ],
          limitations: [
            "Data download and NSG compute access require the NEMAR website or DataLad. Dataset IDs map to OpenNeuro equivalents.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// Brain-CODE (Ontario Brain Institute) dataset tools
// ---------------------------------------------------------------------------

server.registerTool(
  "search_braincode_datasets",
  {
    description:
      "Searches Brain-CODE (Ontario Brain Institute) datasets available through the CONP archive. " +
      "Brain-CODE hosts clinical, MRI, EEG, genomic and other neuroscience data from Canadian brain disorder research (epilepsy, depression, neurodegenerative disease, cerebral palsy, concussion). " +
      "Datasets are mirrored in conpdatasets under braincode_* repos. Use keywords like 'mouse', 'fBIRN', 'NDD', 'epilepsy', 'POND', or omit to list all Brain-CODE datasets in CONP. " +
      "Use get_braincode_dataset_details with a repo name (e.g. braincode_Mouse_Image) for full metadata. Full catalog and controlled releases: braincode.ca.",
    inputSchema: {
      query: z.string().optional().describe("Keyword to narrow results (e.g. 'mouse', 'fBIRN', 'NDD'). Omit to list all Brain-CODE datasets in CONP."),
      maxResults: z.number().optional().describe("Maximum results (default 20, max 50)."),
    },
  },
  async ({ query, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const extra = normalizeWhitespace(query || "");
    const q = extra ? `${BRAINCODE_CONP_QUERY} ${extra} org:${CONP_GITHUB_ORG}` : `${BRAINCODE_CONP_QUERY} org:${CONP_GITHUB_ORG}`;
    const params = new URLSearchParams({ q, per_page: String(limit), page: "1", sort: "updated", order: "desc" });

    const url = `${GITHUB_API}/search/repositories?${params.toString()}`;
    let data;
    try {
      data = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 12000,
        headers: { Accept: "application/vnd.github+json", "User-Agent": "research-mcp" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Brain-CODE search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [extra ? `Query: ${extra}` : "Browse: all Brain-CODE"],
            sources: ["https://www.braincode.ca/", "https://github.com/conpdatasets"],
            limitations: ["GitHub API rate limits may apply."],
          }),
        }],
      };
    }

    const items = Array.isArray(data?.items) ? data.items : [];
    const total = Number(data?.total_count || 0);
    if (items.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No Brain-CODE datasets found${extra ? ` for "${extra}"` : ""}.`,
            keyFields: [extra ? `Query: ${extra}` : "Browse: all"],
            sources: ["https://www.braincode.ca/content/public-data-releases", "https://github.com/conpdatasets"],
            limitations: ["Brain-CODE datasets in CONP use braincode_* naming. Full catalog at braincode.ca."],
          }),
        }],
      };
    }

    const lines = items.map((r, i) => {
      const name = r?.name || "?";
      const desc = normalizeWhitespace(r?.description || "").slice(0, 90);
      const stars = Number(r?.stargazers_count || 0);
      const updated = r?.updated_at ? String(r.updated_at).slice(0, 10) : "—";
      return `  ${String(i + 1).padStart(3)}. ${name} — ${desc || "No description"} | stars: ${stars} | updated: ${updated}`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Brain-CODE: ${items.length} dataset(s)${extra ? ` matching "${extra}"` : ""} in CONP. Total: ${total}.`,
          keyFields: [extra ? `Query: ${extra}` : "Browse: all Brain-CODE", "\nResults:", ...lines],
          sources: [
            "https://www.braincode.ca/content/public-data-releases",
            "https://www.braincode.ca/",
            "https://github.com/conpdatasets",
          ],
          limitations: [
            "This tool indexes Brain-CODE datasets mirrored in CONP. For controlled releases and full catalog, see braincode.ca.",
            "Use get_braincode_dataset_details with repo name for full metadata.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_braincode_dataset_details",
  {
    description:
      "Fetches detailed metadata for a Brain-CODE dataset repository (e.g. braincode_Mouse_Image, braincode_fBIRN). " +
      "Brain-CODE datasets in CONP use braincode_* naming. Returns description, README preview, topics, and links to braincode.ca.",
    inputSchema: {
      repo: z.string().describe("Repository name from search_braincode_datasets (e.g. 'braincode_Mouse_Image', 'braincode_fBIRN')."),
    },
  },
  async ({ repo }) => {
    const rawRepo = normalizeWhitespace(repo || "");
    if (!rawRepo) {
      return { content: [{ type: "text", text: "Provide a Brain-CODE repo name (e.g. braincode_Mouse_Image)." }] };
    }
    const repoName = rawRepo.includes("/") ? rawRepo.split("/").pop() : rawRepo;
    if (!repoName) {
      return { content: [{ type: "text", text: "Unable to parse repository name." }] };
    }

    const repoUrl = `${GITHUB_API}/repos/${CONP_GITHUB_ORG}/${encodeURIComponent(repoName)}`;
    let repoData;
    try {
      repoData = await fetchJsonWithRetry(repoUrl, {
        retries: 1,
        timeoutMs: 12000,
        headers: { Accept: "application/vnd.github+json", "User-Agent": "research-mcp" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Brain-CODE dataset not found: ${CONP_GITHUB_ORG}/${repoName}.`,
            keyFields: [`Repository: ${repoName}`, `Error: ${compactErrorMessage(error?.message || "unknown error", 220)}`],
            sources: ["https://www.braincode.ca/", `https://github.com/${CONP_GITHUB_ORG}/${repoName}`],
            limitations: ["Brain-CODE repos use braincode_* prefix. Verify name from search_braincode_datasets."],
          }),
        }],
      };
    }

    let readmePreview = "";
    try {
      const readmeResponse = await fetchWithRetry(
        `${GITHUB_API}/repos/${CONP_GITHUB_ORG}/${encodeURIComponent(repoName)}/readme`,
        { retries: 1, timeoutMs: 10000, headers: { Accept: "application/vnd.github.raw", "User-Agent": "research-mcp" } }
      );
      const readmeText = await readmeResponse.text();
      if (readmeText) {
        const cleaned = normalizeWhitespace(readmeText.replace(/[#*_`>-]/g, " "));
        readmePreview = cleaned.length > 400 ? `${cleaned.slice(0, 397)}...` : cleaned;
      }
    } catch (_) {}

    const topics = Array.isArray(repoData?.topics) ? repoData.topics : [];
    const keyFields = [
      `Repository: ${repoData?.full_name || `${CONP_GITHUB_ORG}/${repoName}`}`,
      `Description: ${normalizeWhitespace(repoData?.description || "No description.")}`,
      `Stars: ${Number(repoData?.stargazers_count || 0)}`,
      `Updated: ${repoData?.updated_at ? String(repoData.updated_at).slice(0, 10) : "unknown"}`,
    ];
    if (topics.length > 0) keyFields.push(`Topics: ${topics.slice(0, 8).join(", ")}`);
    if (readmePreview) keyFields.push(`\nREADME preview:\n${readmePreview}`);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Brain-CODE dataset: ${repoData?.full_name || `${CONP_GITHUB_ORG}/${repoName}`}.`,
          keyFields,
          sources: [
            repoData?.html_url || `https://github.com/${CONP_GITHUB_ORG}/${repoName}`,
            "https://www.braincode.ca/content/public-data-releases",
            "https://www.braincode.ca/",
          ],
          limitations: [
            "Controlled releases and full metadata require registration at braincode.ca.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// ENIGMA Consortium (imaging genetics meta-analysis) tools
// ---------------------------------------------------------------------------

const ENIGMA_DISORDER_PREFIXES = {
  scz: "Schizophrenia",
  schizophrenia: "Schizophrenia",
  mdd: "mdd",
  depression: "mdd",
  adhd: "adhd",
  bd: "bd",
  bipolar: "bd",
  asd: "asd",
  autism: "asd",
  ocd: "ocd",
  "22q": "22q",
  allepi: "allepi",
  epilepsy: "allepi",
  anorexia: "anorexia",
  parkinsons: "parkinsons",
  parkinson: "parkinsons",
  antisocial: "Antisocial",
  schizotypy: "schizotypy",
};

function enigmaFileMatchesQuery(filename, query) {
  const q = normalizeWhitespace(query || "").toLowerCase();
  if (!q) return true;
  const f = filename.replace(".csv", "");
  const fLower = f.toLowerCase();
  if (fLower.includes(q)) return true;
  const prefix = ENIGMA_DISORDER_PREFIXES[q] || q;
  if (fLower.startsWith(prefix.toLowerCase())) return true;
  for (const [key, val] of Object.entries(ENIGMA_DISORDER_PREFIXES)) {
    if (q.includes(key) && fLower.startsWith(val.toLowerCase())) return true;
  }
  return false;
}

function enigmaDisorderPrefix(code) {
  const c = normalizeWhitespace(code || "").toLowerCase();
  return ENIGMA_DISORDER_PREFIXES[c] || c;
}

server.registerTool(
  "search_enigma_datasets",
  {
    description:
      "Searches ENIGMA Consortium case-control summary statistics from the ENIGMA Toolbox. " +
      "ENIGMA provides 100+ meta-analytical neuroimaging datasets (cortical thickness, subcortical volume, surface area) for disorders including schizophrenia, depression, ADHD, bipolar, OCD, autism, epilepsy, Parkinson's, 22q, anorexia. " +
      "Use disorder keywords: 'schizophrenia', 'depression', 'ADHD', 'bipolar', 'OCD', 'autism', 'epilepsy', 'Parkinson', '22q', 'anorexia'. Omit query to list all. " +
      "Returns dataset names, metrics (CortThick, SubVol, CortSurf), and links. Use get_enigma_dataset_info for a specific disorder.",
    inputSchema: {
      query: z.string().optional().describe("Disorder or keyword (e.g. 'schizophrenia', 'depression', 'ADHD', 'epilepsy'). Omit to list all."),
      maxResults: z.number().optional().describe("Maximum results (default 30, max 80)."),
    },
  },
  async ({ query, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 30), 80);
    const url = `${GITHUB_API}/repos/${ENIGMA_TOOLBOX_REPO}/contents/${ENIGMA_SUMMARY_STATS_PATH}?ref=master`;

    let fileList;
    try {
      fileList = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 12000,
        headers: { Accept: "application/vnd.github+json", "User-Agent": "research-mcp" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `ENIGMA catalog fetch failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [],
            sources: ["https://enigma-toolbox.readthedocs.io/", "https://github.com/MICA-MNI/ENIGMA"],
            limitations: ["GitHub API rate limits may apply."],
          }),
        }],
      };
    }

    const files = Array.isArray(fileList) ? fileList.filter((f) => (f.name || "").endsWith(".csv")) : [];
    const filtered = normalizeWhitespace(query)
      ? files.filter((f) => enigmaFileMatchesQuery(f.name, query))
      : files;

    if (filtered.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No ENIGMA datasets found${query ? ` for "${query}"` : ""}.`,
            keyFields: [query ? `Query: ${query}` : "Query: (none)"],
            sources: ["https://enigma-toolbox.readthedocs.io/", "https://enigma.ini.usc.edu/"],
            limitations: ["Try: schizophrenia, depression, ADHD, bipolar, OCD, autism, epilepsy, Parkinson, 22q, anorexia."],
          }),
        }],
      };
    }

    const shown = filtered.slice(0, limit);
    const byDisorder = {};
    for (const f of shown) {
      const base = f.name.replace(".csv", "");
      const disorder = base.split("_")[0] || "unknown";
      const metric = base.includes("CortThick") ? "CortThick" : base.includes("SubVol") ? "SubVol" : base.includes("CortSurf") ? "CortSurf" : "other";
      if (!byDisorder[disorder]) byDisorder[disorder] = [];
      byDisorder[disorder].push(metric);
    }
    const lines = Object.entries(byDisorder).map(([d, mets]) => {
      const unique = [...new Set(mets)];
      return `  ${d}: ${unique.join(", ")} (${mets.length} file(s))`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `ENIGMA: ${filtered.length} summary statistic dataset(s)${query ? ` matching "${query}"` : ""}. Disorders: ${Object.keys(byDisorder).length}.`,
          keyFields: [
            query ? `Query: ${query}` : "Browse: all",
            `Total matches: ${filtered.length}`,
            "\nBy disorder (metric types):",
            ...lines,
          ],
          sources: [
            "https://enigma-toolbox.readthedocs.io/",
            "https://enigma.ini.usc.edu/",
            `https://github.com/${ENIGMA_TOOLBOX_REPO}/tree/master/${ENIGMA_SUMMARY_STATS_PATH}`,
          ],
          limitations: [
            "Data are case-control meta-analysis summary statistics. Raw imaging requires joining ENIGMA Working Groups.",
            "Use get_enigma_dataset_info with a disorder code (e.g. scz, mdd) for file list and access.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_enigma_dataset_info",
  {
    description:
      "Lists ENIGMA summary statistic files for a specific disorder. " +
      "Use disorder codes: 22q, scz, mdd, adhd (or adhdadult, adhdpediatric), bd, asd, ocd, allepi, parkinsons, anorexia, antisocial, schizotypy. " +
      "Returns filenames, metric types, and raw CSV URLs from the ENIGMA Toolbox.",
    inputSchema: {
      disorder: z.string().describe("ENIGMA disorder code (e.g. scz, mdd, adhd, 22q, bd, asd, allepi)."),
    },
  },
  async ({ disorder }) => {
    const code = normalizeWhitespace(disorder || "").toLowerCase();
    if (!code) {
      return { content: [{ type: "text", text: "Provide an ENIGMA disorder code (e.g. scz, mdd, adhd)." }] };
    }

    const url = `${GITHUB_API}/repos/${ENIGMA_TOOLBOX_REPO}/contents/${ENIGMA_SUMMARY_STATS_PATH}?ref=master`;
    let fileList;
    try {
      fileList = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 12000,
        headers: { Accept: "application/vnd.github+json", "User-Agent": "research-mcp" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `ENIGMA catalog fetch failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [`Disorder: ${code}`],
            sources: ["https://github.com/MICA-MNI/ENIGMA"],
            limitations: [],
          }),
        }],
      };
    }

    const files = Array.isArray(fileList) ? fileList.filter((f) => (f.name || "").endsWith(".csv")) : [];
    const prefix = enigmaDisorderPrefix(code);
    const prefixLower = prefix.toLowerCase();
    const matching = files.filter((f) => {
      const base = f.name.replace(".csv", "");
      const filePrefix = base.split("_")[0] || "";
      return base.toLowerCase().startsWith(prefixLower) || filePrefix.toLowerCase().startsWith(code) || code.startsWith(filePrefix.toLowerCase());
    });

    if (matching.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No ENIGMA files found for disorder "${code}".`,
            keyFields: [`Disorder: ${code}`],
            sources: ["https://enigma-toolbox.readthedocs.io/"],
            limitations: ["Try: scz, mdd, adhd, bd, asd, ocd, 22q, allepi, parkinsons, anorexia."],
          }),
        }],
      };
    }

    const lines = matching.map((f) => {
      const rawUrl = f.download_url || `https://raw.githubusercontent.com/${ENIGMA_TOOLBOX_REPO}/master/${ENIGMA_SUMMARY_STATS_PATH}/${f.name}`;
      return `  ${f.name} | ${rawUrl}`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `ENIGMA: ${matching.length} file(s) for disorder "${code}".`,
          keyFields: [`Disorder: ${code}`, "\nFiles (with raw URLs):", ...lines],
          sources: [
            "https://enigma-toolbox.readthedocs.io/",
            "https://enigma.ini.usc.edu/",
            `https://github.com/${ENIGMA_TOOLBOX_REPO}`,
          ],
          limitations: [
            "Load via Python: from enigmatoolbox.datasets import load_summary_stats; load_summary_stats('" + code + "').",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// Neurobagel cohort discovery tools
// ---------------------------------------------------------------------------

function neurobagelFormatTerm(value) {
  const raw = normalizeWhitespace(value || "");
  if (!raw) return "";
  const cleaned = raw
    .replace(/^https?:\/\/snomed\.info\/id\//i, "snomed:")
    .replace(/^https?:\/\/purl\.org\/nidash\/nidm#/i, "nidm:")
    .replace(/^https?:\/\/raw\.githubusercontent\.com\/neurobagel\/.*?\/vocab\/terms\//i, "")
    .trim();
  if (cleaned.includes("#")) return cleaned.split("#").pop() || cleaned;
  if (cleaned.includes("/")) return cleaned.split("/").pop() || cleaned;
  return cleaned;
}

server.registerTool(
  "query_neurobagel_cohorts",
  {
    description:
      "Queries public Neurobagel cohorts using structured demographic and imaging filters. " +
      "The public Neurobagel node indexes harmonized OpenNeuro datasets — primarily healthy-participant neuroimaging studies. " +
      "IMPORTANT query strategy: " +
      "(1) Start with broad demographic/imaging filters (age range, sex, image_modal) rather than diagnosis codes. " +
      "Most indexed datasets do NOT have diagnosis annotations, so diagnosis filters often return zero results. " +
      "(2) Use image_modal to find datasets by modality: 'http://purl.org/nidash/nidm#T1Weighted', 'http://purl.org/nidash/nidm#T2Weighted', 'http://purl.org/nidash/nidm#FlowWeighted', 'http://purl.org/nidash/nidm#DiffusionWeighted'. " +
      "(3) Calling with NO filters returns all indexed cohorts — useful for browsing available datasets. " +
      "(4) Only use diagnosis if you know the specific SNOMED code is present in the graph (rare for the public node).",
    inputSchema: {
      minAge: z.number().optional().describe("Minimum participant age in years."),
      maxAge: z.number().optional().describe("Maximum participant age in years."),
      sex: z.string().optional().describe("SNOMED sex term: 'snomed:248152002' (female) or 'snomed:248153007' (male)."),
      diagnosis: z.string().optional().describe("SNOMED diagnosis term. WARNING: most public-node datasets lack diagnosis annotations — this filter often returns empty results. Only use if you know the term is indexed."),
      minImagingSessions: z.number().optional().describe("Minimum number of imaging sessions per participant."),
      minPhenotypicSessions: z.number().optional().describe("Minimum number of phenotypic sessions per participant."),
      assessment: z.string().optional().describe("Assessment/tool term used in Neurobagel harmonization."),
      imageModal: z.string().optional().describe("Imaging modality NIDM URI, e.g. 'http://purl.org/nidash/nidm#T1Weighted', 'http://purl.org/nidash/nidm#DiffusionWeighted'."),
      pipelineName: z.string().optional().describe("Pipeline name URI from Neurobagel pipeline catalog."),
      pipelineVersion: z.string().optional().describe("Pipeline version string."),
      maxResults: z.number().optional().describe("Maximum records to display (default 25, max 100)."),
    },
  },
  async ({
    minAge,
    maxAge,
    sex,
    diagnosis,
    minImagingSessions,
    minPhenotypicSessions,
    assessment,
    imageModal,
    pipelineName,
    pipelineVersion,
    maxResults,
  }) => {
    const params = new URLSearchParams();
    if (Number.isFinite(minAge)) params.set("min_age", String(minAge));
    if (Number.isFinite(maxAge)) params.set("max_age", String(maxAge));
    if (normalizeWhitespace(sex)) params.set("sex", normalizeWhitespace(sex));
    if (normalizeWhitespace(diagnosis)) params.set("diagnosis", normalizeWhitespace(diagnosis));
    if (Number.isFinite(minImagingSessions)) params.set("min_num_imaging_sessions", String(minImagingSessions));
    if (Number.isFinite(minPhenotypicSessions)) params.set("min_num_phenotypic_sessions", String(minPhenotypicSessions));
    if (normalizeWhitespace(assessment)) params.set("assessment", normalizeWhitespace(assessment));
    if (normalizeWhitespace(imageModal)) params.set("image_modal", normalizeWhitespace(imageModal));
    if (normalizeWhitespace(pipelineName)) params.set("pipeline_name", normalizeWhitespace(pipelineName));
    if (normalizeWhitespace(pipelineVersion)) params.set("pipeline_version", normalizeWhitespace(pipelineVersion));

    const limit = Math.min(Math.max(1, maxResults || 25), 100);
    const url = `${NEUROBAGEL_API}/query${params.toString() ? `?${params.toString()}` : ""}`;

    let rows;
    try {
      rows = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 20000,
        headers: { Accept: "application/json" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Neurobagel query failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [`Query URL: ${url}`],
            sources: [
              "https://api.neurobagel.org/docs",
              "https://neurobagel.org/user_guide/api/",
            ],
            limitations: [
              "Neurobagel API may reject invalid controlled terms or temporary network interruptions.",
            ],
          }),
        }],
      };
    }

    const matches = Array.isArray(rows) ? rows : [];
    if (matches.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: "No Neurobagel cohorts matched the provided filters.",
            keyFields: [
              `Filters: ${params.toString() || "(none)"}`,
            ],
            sources: ["https://api.neurobagel.org/docs"],
            limitations: [
              "Public node results reflect currently indexed harmonized datasets and may change over time.",
            ],
          }),
        }],
      };
    }

    const shown = matches.slice(0, limit);
    const lines = shown.map((row, i) => {
      const dataset = row?.dataset_name || "Unnamed dataset";
      const uuid = neurobagelFormatTerm(row?.dataset_uuid || "");
      const totalSubjects = Number(row?.dataset_total_subjects || 0);
      const matchedSubjects = Number(row?.num_matching_subjects || 0);
      const modalities = Array.isArray(row?.image_modals)
        ? row.image_modals.slice(0, 4).map((m) => neurobagelFormatTerm(m)).join(", ")
        : "";
      const protectedFlag = row?.records_protected ? "yes" : "no";
      const parts = [
        `${String(i + 1).padStart(3)}. ${dataset}`,
        `     Dataset ID: ${uuid || "unknown"} | Matched subjects: ${matchedSubjects} | Total subjects: ${totalSubjects}`,
        `     Records protected: ${protectedFlag}`,
      ];
      if (modalities) parts.push(`     Modalities: ${modalities}`);
      if (row?.dataset_portal_uri) parts.push(`     Portal: ${row.dataset_portal_uri}`);
      return parts.join("\n");
    });

    const uniqueDatasets = new Set(matches.map((r) => r?.dataset_uuid).filter(Boolean)).size;
    const totalMatched = matches.reduce((acc, r) => acc + Number(r?.num_matching_subjects || 0), 0);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `Neurobagel returned ${matches.length} cohort record(s) across ${uniqueDatasets} dataset(s). Showing top ${shown.length}.`,
          keyFields: [
            `Filters: ${params.toString() || "(none)"}`,
            `Unique datasets: ${uniqueDatasets}`,
            `Total matching subjects (sum across records): ${totalMatched}`,
            "\nResults:",
            ...lines,
          ],
          sources: [
            url,
            "https://api.neurobagel.org/docs",
            "https://neurobagel.org/user_guide/public_nodes/",
          ],
          limitations: [
            "Public Neurobagel node currently reflects harmonized OpenNeuro-linked cohorts and may not include all private/institutional nodes.",
            "Subject-level records may be redacted or aggregated depending on dataset protection rules.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// OpenNeuro neuroimaging dataset tools
// ---------------------------------------------------------------------------

const OPENNEURO_MODALITIES = new Set(["MRI", "MEG", "EEG", "PET", "iEEG", "behavioral"]);

server.registerTool(
  "search_openneuro_datasets",
  {
    description:
      "Searches public OpenNeuro neuroimaging datasets by imaging modality. " +
      "OpenNeuro is the primary open platform for fMRI, MRI, MEG, EEG, and other neuroimaging data (BIDS format). " +
      "Use modality to filter: 'MRI', 'MEG', 'EEG', 'PET', 'iEEG', or 'behavioral'. Omit modality to browse all public datasets. " +
      "Returns dataset IDs (e.g. ds000224), names, modalities, and latest snapshot tags. Use get_openneuro_dataset with an ID for full metadata.",
    inputSchema: {
      modality: z.string().optional().describe("Imaging modality: MRI, MEG, EEG, PET, iEEG, or behavioral. Omit to list all datasets."),
      maxResults: z.number().optional().describe("Maximum results (default 20, max 50)."),
    },
  },
  async ({ modality, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const modArg = modality && OPENNEURO_MODALITIES.has(String(modality).trim().toUpperCase())
      ? String(modality).trim().toUpperCase()
      : null;

    const query = modArg
      ? `{ datasets(first: ${limit}, modality: "${modArg}") { edges { node { id name metadata { modalities } latestSnapshot { tag } } } pageInfo { hasNextPage } } }`
      : `{ datasets(first: ${limit}) { edges { node { id name metadata { modalities } latestSnapshot { tag } } } pageInfo { hasNextPage } } }`;

    let data;
    try {
      const res = await fetchWithRetry(OPENNEURO_GRAPHQL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        timeoutMs: 15000,
        retries: 1,
      });
      data = await res.json();
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OpenNeuro search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [modArg ? `Modality: ${modArg}` : "Modality: all"],
            sources: ["https://openneuro.org/", "https://docs.openneuro.org/api.html"],
            limitations: ["GraphQL API may be temporarily unavailable."],
          }),
        }],
      };
    }

    const errs = data?.errors;
    if (errs && errs.length > 0) {
      const msg = errs.map((e) => e?.message || "").filter(Boolean).join("; ");
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OpenNeuro GraphQL error: ${msg.slice(0, 300)}`,
            keyFields: [modArg ? `Modality: ${modArg}` : "Modality: all"],
            sources: ["https://openneuro.org/"],
            limitations: ["Check modality spelling: MRI, MEG, EEG, PET, iEEG, behavioral."],
          }),
        }],
      };
    }

    const edges = data?.data?.datasets?.edges || [];
    if (edges.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No OpenNeuro datasets found${modArg ? ` for modality ${modArg}` : ""}.`,
            keyFields: [modArg ? `Modality: ${modArg}` : "Modality: all"],
            sources: ["https://openneuro.org/datasets"],
            limitations: ["Valid modalities: MRI, MEG, EEG, PET, iEEG, behavioral."],
          }),
        }],
      };
    }

    const lines = edges.map((e, i) => {
      const node = e?.node || {};
      const id = node.id || "?";
      const name = node.name || "Unnamed";
      const mods = Array.isArray(node.metadata?.modalities) ? node.metadata.modalities.join(", ") : "unknown";
      const tag = node.latestSnapshot?.tag || "—";
      return `  ${String(i + 1).padStart(3)}. ${id} — ${name} (${mods}) snapshot: ${tag}`;
    });

    const hasMore = data?.data?.datasets?.pageInfo?.hasNextPage;

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `OpenNeuro: ${edges.length} dataset(s)${modArg ? ` (modality: ${modArg})` : ""}.${hasMore ? " More available via pagination." : ""}`,
          keyFields: [
            modArg ? `Modality filter: ${modArg}` : "Modality: all",
            `Showing: ${edges.length}`,
            "\nDatasets:",
            ...lines,
          ],
          sources: [
            "https://openneuro.org/datasets",
            `https://openneuro.org/datasets${modArg ? `?modality=${modArg.toLowerCase()}` : ""}`,
          ],
          limitations: [
            "Use get_openneuro_dataset with a dataset ID (e.g. ds000224) for full metadata, DOI, and description.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_openneuro_dataset",
  {
    description:
      "Retrieves detailed metadata for a specific OpenNeuro dataset by ID (e.g. ds000224). " +
      "Returns dataset name, modalities, DOI, latest snapshot tag, and description. Use after search_openneuro_datasets to inspect promising datasets.",
    inputSchema: {
      datasetId: z.string().describe("OpenNeuro dataset ID, e.g. ds000224 or ds001."),
    },
  },
  async ({ datasetId }) => {
    const id = normalizeWhitespace(datasetId || "").toLowerCase();
    if (!id) {
      return { content: [{ type: "text", text: "Provide an OpenNeuro dataset ID (e.g. ds000224)." }] };
    }
    const normalizedId = id.startsWith("ds") ? id : `ds${id}`;

    const query = `{ dataset(id: "${normalizedId}") { id name metadata { modalities } latestSnapshot { tag description { Name DatasetDOI } } } }`;

    let data;
    try {
      const res = await fetchWithRetry(OPENNEURO_GRAPHQL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
        timeoutMs: 15000,
        retries: 1,
      });
      data = await res.json();
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OpenNeuro dataset fetch failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [`Dataset ID: ${normalizedId}`],
            sources: [`https://openneuro.org/datasets/${normalizedId}`],
            limitations: ["Dataset may not exist or may be private."],
          }),
        }],
      };
    }

    const errs = data?.errors;
    if (errs && errs.length > 0) {
      const msg = errs.map((e) => e?.message || "").filter(Boolean).join("; ");
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OpenNeuro error: ${msg.slice(0, 300)}`,
            keyFields: [`Dataset ID: ${normalizedId}`],
            sources: [`https://openneuro.org/datasets/${normalizedId}`],
            limitations: ["Verify the dataset ID is correct and the dataset is public."],
          }),
        }],
      };
    }

    const ds = data?.data?.dataset;
    if (!ds) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OpenNeuro dataset not found: ${normalizedId}.`,
            keyFields: [`Dataset ID: ${normalizedId}`],
            sources: ["https://openneuro.org/datasets"],
            limitations: ["Dataset may be private, deleted, or the ID may be incorrect."],
          }),
        }],
      };
    }

    const mods = Array.isArray(ds.metadata?.modalities) ? ds.metadata.modalities.join(", ") : "unknown";
    const desc = ds.latestSnapshot?.description || {};
    const name = desc.Name || ds.name || "Unnamed";
    const doi = desc.DatasetDOI || "";
    const tag = ds.latestSnapshot?.tag || "—";

    const keyFields = [
      `Dataset: ${name}`,
      `ID: ${ds.id}`,
      `Modalities: ${mods}`,
      `Latest snapshot: ${tag}`,
    ];
    if (doi) keyFields.push(`DOI: ${doi}`);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `OpenNeuro: ${name} (${ds.id}).`,
          keyFields,
          sources: [
            `https://openneuro.org/datasets/${ds.id}`,
            doi ? `https://doi.org/${doi.replace(/^doi:/i, "").trim()}` : null,
          ].filter(Boolean),
          limitations: [
            "File listings and subject-level metadata require the OpenNeuro CLI or direct S3/git access.",
          ],
        }),
      }],
    };
  }
);

// ---------------------------------------------------------------------------
// DANDI Archive neurophysiology dataset tools
// ---------------------------------------------------------------------------

server.registerTool(
  "search_dandi_datasets",
  {
    description:
      "Searches the DANDI Archive for neurophysiology datasets (electrophysiology, optophysiology, behavioral, immunostaining). " +
      "DANDI hosts NWB/BIDS-format data from the BRAIN Initiative. Use search terms like 'electrophysiology', 'hippocampus', 'calcium imaging', 'spike', or topic keywords. " +
      "Returns dandiset identifiers, names, asset counts, sizes, and embargo status. Use get_dandi_dataset with an identifier for full metadata.",
    inputSchema: {
      query: z.string().optional().describe("Search term (e.g. 'electrophysiology', 'hippocampus', 'calcium'). Omit to list recent dandisets."),
      maxResults: z.number().optional().describe("Maximum results (default 20, max 50)."),
    },
  },
  async ({ query, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const params = new URLSearchParams({ page_size: String(limit), page: "1" });
    if (normalizeWhitespace(query)) params.set("search", normalizeWhitespace(query));

    const url = `${DANDI_API}/dandisets/?${params.toString()}`;

    let data;
    try {
      data = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 15000,
        headers: { Accept: "application/json" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `DANDI search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [query ? `Query: ${query}` : "Query: (none)"],
            sources: ["https://dandiarchive.org/", "https://api.dandiarchive.org/"],
            limitations: ["DANDI API may be temporarily unavailable."],
          }),
        }],
      };
    }

    const results = data?.results || [];
    const total = Number(data?.count ?? 0);

    if (results.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No DANDI dandisets found${query ? ` for "${query}"` : ""}.`,
            keyFields: [query ? `Query: ${query}` : "Query: (none)"],
            sources: ["https://dandiarchive.org/dandisets/"],
            limitations: ["Try broader search terms or omit the query to browse all dandisets."],
          }),
        }],
      };
    }

    const formatBytes = (n) => {
      if (!Number.isFinite(n)) return "—";
      const gb = n / 1e9;
      return gb >= 1 ? `${gb.toFixed(1)} GB` : `${(n / 1e6).toFixed(1)} MB`;
    };

    const lines = results.map((r, i) => {
      const ver = r.most_recent_published_version || r.draft_version || {};
      const name = ver.name || "Unnamed";
      const ident = r.identifier || "?";
      const assets = Number(ver.asset_count ?? 0);
      const size = formatBytes(Number(ver.size ?? 0));
      const embargo = r.embargo_status || "?";
      return `  ${String(i + 1).padStart(3)}. ${ident} — ${name.slice(0, 60)}${name.length > 60 ? "..." : ""} | ${assets} assets, ${size} | ${embargo}`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `DANDI: ${results.length} dandiset(s)${query ? ` matching "${query}"` : ""}. Total in archive: ${total}.`,
          keyFields: [
            query ? `Search: ${query}` : "Browse: recent dandisets",
            `Total: ${total}`,
            "\nResults:",
            ...lines,
          ],
          sources: [
            query ? `https://dandiarchive.org/dandisets/?search=${encodeURIComponent(query)}` : "https://dandiarchive.org/dandisets/",
            "https://dandiarchive.org/",
          ],
          limitations: [
            "Use get_dandi_dataset with a dandiset identifier (e.g. 000003) for full metadata and asset details.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_dandi_dataset",
  {
    description:
      "Retrieves detailed metadata for a DANDI dandiset by identifier (e.g. 000003). " +
      "Returns name, version, asset count, size, embargo status, and contact. Use after search_dandi_datasets to inspect promising datasets.",
    inputSchema: {
      dandisetId: z.string().describe("DANDI dandiset identifier, e.g. 000003 or dandi:000003."),
    },
  },
  async ({ dandisetId }) => {
    let id = normalizeWhitespace(dandisetId || "").replace(/^dandi:/i, "").replace(/\D/g, "");
    if (!id) {
      return { content: [{ type: "text", text: "Provide a DANDI dandiset identifier (e.g. 000003)." }] };
    }
    id = id.padStart(6, "0");

    const url = `${DANDI_API}/dandisets/${id}/`;

    let data;
    try {
      data = await fetchJsonWithRetry(url, {
        retries: 1,
        timeoutMs: 15000,
        headers: { Accept: "application/json" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `DANDI dandiset fetch failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [`Identifier: ${id}`],
            sources: [`https://dandiarchive.org/dandiset/${id}`],
            limitations: ["Dandiset may not exist or may be embargoed."],
          }),
        }],
      };
    }

    const ver = data.most_recent_published_version || data.draft_version || {};
    const name = ver.name || "Unnamed";
    const version = ver.version || "—";
    const assets = Number(ver.asset_count ?? 0);
    const size = Number(ver.size ?? 0);
    const sizeStr = size >= 1e9 ? `${(size / 1e9).toFixed(1)} GB` : `${(size / 1e6).toFixed(1)} MB`;

    const keyFields = [
      `Name: ${name}`,
      `Identifier: ${data.identifier || id}`,
      `Version: ${version}`,
      `Assets: ${assets}`,
      `Size: ${sizeStr}`,
      `Embargo: ${data.embargo_status || "?"}`,
    ];
    if (data.contact_person) keyFields.push(`Contact: ${data.contact_person}`);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `DANDI: ${name} (${data.identifier || id}).`,
          keyFields,
          sources: [
            `https://dandiarchive.org/dandiset/${data.identifier || id}`,
          ],
          limitations: [
            "Asset-level metadata and downloads require the DANDI CLI or direct API asset endpoints.",
          ],
        }),
      }],
    };
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
