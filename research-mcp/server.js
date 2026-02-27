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
const HF_DATASETS_SERVER_API = "https://datasets-server.huggingface.co";
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
// TOOL 5: List BigQuery tables (read-only metadata)
// ============================================
server.registerTool(
  "list_bigquery_tables",
  {
    description:
      "Lists tables for a BigQuery dataset. Dataset allowlist can be enforced via BQ_DATASET_ALLOWLIST.",
    inputSchema: {
      dataset: z
        .string()
        .optional()
        .describe("Dataset as `project.dataset` or just `dataset` (e.g. 'open_targets_platform'). Short names are resolved against the allowlist. If omitted, uses first allowlisted dataset."),
      limit: z.number().optional().default(200).describe("Maximum number of tables to return (1-200)."),
    },
  },
  async ({ dataset = "", limit = 200 }) => {
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

    try {
      const client = getBigQueryClient();
      const datasetRef = client.dataset(datasetId, { projectId });
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
                "Only metadata is returned. Query table contents via run_bigquery_select_query.",
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
        const response = await fetch(url);

        if (!response.ok) {
          if (studies.length > 0) break;
          return {
            content: [{ type: "text", text: `ClinicalTrials.gov API error (${response.status}). Try a different search term.` }],
          };
        }

        const text = await response.text();
        if (!text || text.trim() === '') {
          if (studies.length > 0) break;
          return {
            content: [{ type: "text", text: `ClinicalTrials.gov returned empty response for: "${query}". Try broader search terms.` }],
          };
        }

        const data = JSON.parse(text);
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
      const response = await fetch(url);
      
      if (!response.ok) {
        if (response.status === 404) {
          return {
            content: [{ type: "text", text: `Clinical trial not found: ${nctId}. Check the NCT ID format (e.g., NCT04665245).` }],
          };
        }
        return {
          content: [{ type: "text", text: `ClinicalTrials.gov API error (${response.status}) for ${nctId}.` }],
        };
      }
      
      const text = await response.text();
      if (!text || text.trim() === '') {
        return {
          content: [{ type: "text", text: `Empty response for ${nctId}. The trial may not exist.` }],
        };
      }
      
      study = JSON.parse(text);
    } catch (error) {
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
      const params = new URLSearchParams({ search: query, per_page: String(limit) });
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
      const url = buildOpenAlexUrl("/authors", new URLSearchParams({ search: query, per_page: String(limit) }));
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
        url = buildOpenAlexUrl(`/authors/${encodeURIComponent(authorId.replace("https://openalex.org/", ""))}`);
        const author = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
        results = author ? [author] : [];
      } else {
        url = buildOpenAlexUrl("/authors", new URLSearchParams({ search: authorName, per_page: String(limit) }));
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

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
