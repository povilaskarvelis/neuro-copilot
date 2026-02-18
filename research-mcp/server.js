import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
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

const NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";
const OPEN_TARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql";
const UNIPROT_API = "https://rest.uniprot.org";
const CLINICAL_TRIALS_API = "https://clinicaltrials.gov/api/v2";
const OPENALEX_API = "https://api.openalex.org";
const REACTOME_API = "https://reactome.org/ContentService";
const STRING_API = "https://string-db.org/api";
const GWAS_API = "https://www.ebi.ac.uk/gwas/rest/api";
const CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data";
const OLS_API = "https://www.ebi.ac.uk/ols4";
const DATA_DIR = path.resolve(__dirname, "data");
const OPENALEX_MAILTO = process.env.OPENALEX_MAILTO || process.env.CONTACT_EMAIL || "";
let gwasCooldownUntilMs = 0;

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

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status}): ${url}`);
  }
  return response.json();
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

async function fetchText(url) {
  const response = await fetchWithRetry(url);
  return response.text();
}

function isLikelyTransientUpstreamError(error) {
  const message = String(error?.message || "").toLowerCase();
  return (
    message.includes("request failed (5") ||
    message.includes("request failed (429)") ||
    message.includes("operation was aborted") ||
    message.includes("aborted") ||
    message.includes("fetch failed") ||
    message.includes("econnreset") ||
    message.includes("etimedout") ||
    message.includes("gateway timeout")
  );
}

function isGwasEndpointError(error) {
  const message = String(error?.message || "").toLowerCase();
  return message.includes("/gwas/rest/api") || message.includes("gwas");
}

function isGwasCooldownActive() {
  return Date.now() < gwasCooldownUntilMs;
}

function activateGwasCooldown(ms = 90_000) {
  gwasCooldownUntilMs = Math.max(gwasCooldownUntilMs, Date.now() + ms);
}

async function fetchGwasJson(url, options = {}) {
  if (isGwasCooldownActive()) {
    throw new Error("GWAS service temporarily unavailable (cooldown active).");
  }
  const timeoutMs = options.timeoutMs ?? 9000;
  const retries = options.retries ?? 1;
  try {
    return await fetchJsonWithRetry(url, { ...options, timeoutMs, retries });
  } catch (error) {
    if (isLikelyTransientUpstreamError(error) || isGwasEndpointError(error)) {
      activateGwasCooldown();
    }
    throw error;
  }
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
  return `[DOI:${doi}](https://doi.org/${encodeURIComponent(doi)})`;
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

function extractPubmedSummaryDoi(item) {
  const articleIds = Array.isArray(item?.articleids) ? item.articleids : [];
  const doiRecord = articleIds.find((entry) => normalizeWhitespace(entry?.idtype || "").toLowerCase() === "doi");
  return normalizeDoiValue(doiRecord?.value || "");
}

async function queryOpenTargets(query, variables = {}) {
  try {
    const response = await fetch(OPEN_TARGETS_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, variables }),
    });
    if (!response.ok) {
      console.error(`Open Targets API error: ${response.status}`);
      return { data: null, error: `API returned ${response.status}` };
    }
    return response.json();
  } catch (error) {
    console.error(`Open Targets network error: ${error.message}`);
    return { data: null, error: `Network error: ${error.message}` };
  }
}

async function listDataFiles() {
  try {
    const entries = await fs.readdir(DATA_DIR, { withFileTypes: true });
    return entries.filter((entry) => entry.isFile()).map((entry) => entry.name);
  } catch (error) {
    if (error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

function resolveDataPath(filename) {
  const safeName = path.basename(filename);
  const resolved = path.resolve(DATA_DIR, safeName);
  if (!resolved.startsWith(DATA_DIR)) {
    throw new Error("Invalid filename. Only files in ./data are allowed.");
  }
  return resolved;
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

async function resolveTargetIdFromInput({ targetId, geneSymbol }) {
  let resolvedTargetId = (targetId || "").trim();
  const sourceHints = [];
  let searchHint = "";

  if (!resolvedTargetId || !/^ENSG\d{11}$/i.test(resolvedTargetId)) {
    const symbolQuery =
      (geneSymbol || "").trim() ||
      (resolvedTargetId && !/^ENSG\d{11}$/i.test(resolvedTargetId) ? resolvedTargetId : "");
    if (!symbolQuery) {
      return { error: "Provide either `targetId` (ENSG...) or `geneSymbol`." };
    }

    const searchQuery = `
      query SearchTargets($queryString: String!, $size: Int!) {
        search(queryString: $queryString, entityNames: ["target"], page: { size: $size, index: 0 }) {
          hits {
            id
            name
            description
          }
        }
      }
    `;
    const searchResult = await queryOpenTargets(searchQuery, { queryString: symbolQuery, size: 12 });
    const hits = searchResult?.data?.search?.hits || [];
    const exact = hits.find(
      (hit) =>
        String(hit?.name || "").toUpperCase() === symbolQuery.toUpperCase() ||
        String(hit?.id || "").toUpperCase() === symbolQuery.toUpperCase()
    );
    const selected = exact || hits[0];
    if (!selected?.id) {
      return { error: `No target match found for "${symbolQuery}".` };
    }
    resolvedTargetId = selected.id;
    searchHint = String(selected?.name || "").trim();
    sourceHints.push(`${OPEN_TARGETS_API}#search:${encodeURIComponent(symbolQuery)}`);
  }

  return {
    targetId: resolvedTargetId,
    searchHint,
    sourceHints,
  };
}

async function resolveDiseaseFromInput({ diseaseId, diseaseQuery }) {
  const sourceHints = [];
  const normalizedDiseaseId = String(diseaseId || "").trim();
  if (normalizedDiseaseId) {
    const byIdQuery = `
      query DiseaseById($diseaseId: String!) {
        disease(efoId: $diseaseId) {
          id
          name
        }
      }
    `;
    const byIdResult = await queryOpenTargets(byIdQuery, { diseaseId: normalizedDiseaseId });
    const disease = byIdResult?.data?.disease;
    sourceHints.push(`${OPEN_TARGETS_API}#disease:${encodeURIComponent(normalizedDiseaseId)}`);
    if (!disease?.id) {
      return { error: `Disease not found for ID "${normalizedDiseaseId}".` };
    }
    return {
      diseaseId: disease.id,
      diseaseName: disease.name || normalizedDiseaseId,
      sourceHints,
    };
  }

  const normalizedDiseaseQuery = String(diseaseQuery || "").trim();
  if (!normalizedDiseaseQuery) {
    return { error: "Provide either `diseaseId` or `diseaseQuery`." };
  }

  const searchQuery = `
    query SearchDiseases($queryString: String!, $size: Int!) {
      search(queryString: $queryString, entityNames: ["disease"], page: { size: $size, index: 0 }) {
        hits {
          id
          name
          description
        }
      }
    }
  `;
  const searchResult = await queryOpenTargets(searchQuery, {
    queryString: normalizedDiseaseQuery,
    size: 12,
  });
  const hits = searchResult?.data?.search?.hits || [];
  if (hits.length === 0) {
    return { error: `No disease match found for "${normalizedDiseaseQuery}".` };
  }

  const exact = hits.find(
    (hit) =>
      String(hit?.name || "").toLowerCase() === normalizedDiseaseQuery.toLowerCase() ||
      String(hit?.id || "").toUpperCase() === normalizedDiseaseQuery.toUpperCase()
  );
  const preferredOntology = hits.find((hit) => {
    const id = String(hit?.id || "");
    return id.startsWith("MONDO_") || id.startsWith("EFO_");
  });
  const selected = exact || preferredOntology || hits[0];
  sourceHints.push(`${OPEN_TARGETS_API}#search:${encodeURIComponent(normalizedDiseaseQuery)}`);

  return {
    diseaseId: selected.id,
    diseaseName: selected.name || normalizedDiseaseQuery,
    sourceHints,
  };
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

const InferDirectionCountsPayloadSchema = z.object({
  risk_increasing: z.number().int().nonnegative(),
  protective: z.number().int().nonnegative(),
  neutral: z.number().int().nonnegative(),
  unknown: z.number().int().nonnegative(),
});

const InferAssociationPayloadSchema = z.object({
  rank: z.number().int().positive(),
  association_id: z.string(),
  rs_id: z.string(),
  direction: z.string(),
  pvalue: z.number().nullable(),
  effect_text: z.string(),
  risk_allele: z.string(),
  traits: z.array(z.string()),
});

const InferGeneticPayloadSchema = z.object({
  schema: z.literal("infer_genetic_effect_direction.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  fallback_mode: z.enum(["none", "open_targets_proxy", "critical_gap"]),
  gene_symbol: z.string(),
  disease_query: z.string(),
  pvalue_threshold: z.number().nonnegative(),
  max_snps: z.number().int().positive(),
  max_associations: z.number().int().positive(),
  time_budget_sec: z.number().int().positive(),
  snps_scanned: z.number().int().nonnegative(),
  associations_scanned: z.number().int().nonnegative(),
  matched_count: z.number().int().nonnegative(),
  timed_out_early: z.boolean(),
  has_mixed_signals: z.boolean(),
  direction_counts: InferDirectionCountsPayloadSchema,
  matched_associations: z.array(InferAssociationPayloadSchema),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildInferGeneticPayload({
  resultStatus = "ok",
  fallbackMode = "none",
  geneSymbol = "",
  diseaseQuery = "",
  pvalueThreshold = 5e-8,
  maxSnps = 8,
  maxAssociations = 40,
  timeBudgetSec = 25,
  snpsScanned = 0,
  associationsScanned = 0,
  matchedAssociations = [],
  timedOutEarly = false,
  hasMixedSignals = false,
  directionCounts = null,
  notes = [],
  errorMessage = "",
}) {
  const normalizedDirectionCounts = {
    risk_increasing: toNonNegativeInt(directionCounts?.risk_increasing),
    protective: toNonNegativeInt(directionCounts?.protective),
    neutral: toNonNegativeInt(directionCounts?.neutral),
    unknown: toNonNegativeInt(directionCounts?.unknown),
  };
  return safeBuildTypedPayload(
    InferGeneticPayloadSchema,
    {
      schema: "infer_genetic_effect_direction.v1",
      result_status: resultStatus,
      fallback_mode: fallbackMode,
      gene_symbol: String(geneSymbol || ""),
      disease_query: String(diseaseQuery || ""),
      pvalue_threshold: Math.max(0, toFiniteNumber(pvalueThreshold, 5e-8)),
      max_snps: Math.max(1, toNonNegativeInt(maxSnps, 8)),
      max_associations: Math.max(1, toNonNegativeInt(maxAssociations, 40)),
      time_budget_sec: Math.max(1, toNonNegativeInt(timeBudgetSec, 25)),
      snps_scanned: toNonNegativeInt(snpsScanned),
      associations_scanned: toNonNegativeInt(associationsScanned),
      matched_count: matchedAssociations.length,
      timed_out_early: Boolean(timedOutEarly),
      has_mixed_signals: Boolean(hasMixedSignals),
      direction_counts: normalizedDirectionCounts,
      matched_associations: matchedAssociations,
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "infer_genetic_effect_direction.v1"
  );
}

const CompareWeightsPayloadSchema = z.object({
  disease_association: z.number().nullable(),
  druggability: z.number().nullable(),
  clinical_maturity: z.number().nullable(),
  competitive_whitespace: z.number().nullable(),
  safety: z.number().nullable(),
});

const CompareLeadPayloadSchema = z.object({
  target_id: z.string(),
  symbol: z.string(),
  composite_score: z.number(),
  lead_margin: z.number(),
});

const CompareScoresPayloadSchema = z.object({
  composite: z.number(),
  disease_association: z.number(),
  druggability: z.number(),
  clinical_maturity: z.number(),
  competitive_whitespace: z.number(),
  safety: z.number(),
});

const CompareRankingPayloadSchema = z.object({
  rank: z.number().int().positive(),
  target_id: z.string(),
  symbol: z.string(),
  approved_name: z.string(),
  scores: CompareScoresPayloadSchema,
  max_phase: z.number(),
  known_unique_drugs: z.number().nonnegative(),
  withdrawn_drug_rows: z.number().int().nonnegative(),
  positive_tractability_count: z.number().int().nonnegative(),
  safety_rows: z.number().int().nonnegative(),
  clinical_safety_rows: z.number().int().nonnegative(),
});

const CompareTargetsPayloadSchema = z.object({
  schema: z.literal("compare_targets_multi_axis.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  targets_requested: z.array(z.string()),
  targets_resolved: z.number().int().nonnegative(),
  targets_compared: z.number().int().nonnegative(),
  unresolved_targets: z.array(z.string()),
  disease_id: z.string(),
  disease_name: z.string(),
  strategy_requested: z.string(),
  strategy_effective: z.string(),
  weight_mode: z.string(),
  goal_text: z.string(),
  weights: CompareWeightsPayloadSchema,
  lead_target: CompareLeadPayloadSchema.nullable(),
  rankings: z.array(CompareRankingPayloadSchema),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildCompareTargetsPayload({
  resultStatus = "ok",
  targetsRequested = [],
  targetsResolved = 0,
  targetsCompared = 0,
  unresolvedTargets = [],
  diseaseId = "unknown",
  diseaseName = "unknown",
  strategyRequested = "balanced",
  strategyEffective = "balanced",
  weightMode = "preset",
  goalText = "",
  weights = null,
  leadTarget = null,
  rankings = [],
  notes = [],
  errorMessage = "",
}) {
  const normalizedWeights = {
    disease_association: Number.isFinite(Number(weights?.disease_association))
      ? Number(weights.disease_association)
      : null,
    druggability: Number.isFinite(Number(weights?.druggability)) ? Number(weights.druggability) : null,
    clinical_maturity: Number.isFinite(Number(weights?.clinical_maturity))
      ? Number(weights.clinical_maturity)
      : null,
    competitive_whitespace: Number.isFinite(Number(weights?.competitive_whitespace))
      ? Number(weights.competitive_whitespace)
      : null,
    safety: Number.isFinite(Number(weights?.safety)) ? Number(weights.safety) : null,
  };
  return safeBuildTypedPayload(
    CompareTargetsPayloadSchema,
    {
      schema: "compare_targets_multi_axis.v1",
      result_status: resultStatus,
      targets_requested: (targetsRequested || []).map((item) => String(item || "").trim()).filter(Boolean),
      targets_resolved: toNonNegativeInt(targetsResolved),
      targets_compared: toNonNegativeInt(targetsCompared),
      unresolved_targets: (unresolvedTargets || []).map((item) => String(item || "").trim()).filter(Boolean),
      disease_id: String(diseaseId || "unknown"),
      disease_name: String(diseaseName || "unknown"),
      strategy_requested: String(strategyRequested || "balanced"),
      strategy_effective: String(strategyEffective || "balanced"),
      weight_mode: String(weightMode || "preset"),
      goal_text: String(goalText || ""),
      weights: normalizedWeights,
      lead_target: leadTarget
        ? {
            target_id: String(leadTarget.target_id || ""),
            symbol: String(leadTarget.symbol || ""),
            composite_score: toFiniteNumber(leadTarget.composite_score, 0),
            lead_margin: toFiniteNumber(leadTarget.lead_margin, 0),
          }
        : null,
      rankings,
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "compare_targets_multi_axis.v1"
  );
}

const DiseaseTargetEvidencePayloadSchema = z.object({
  id: z.string(),
  score_pct: z.number().nonnegative(),
});

const DiseaseTargetRowPayloadSchema = z.object({
  rank: z.number().int().positive(),
  target_id: z.string(),
  symbol: z.string(),
  approved_name: z.string(),
  biotype: z.string(),
  overall_score_pct: z.number().nonnegative(),
  evidence: z.array(DiseaseTargetEvidencePayloadSchema),
});

const SearchDiseaseTargetsPayloadSchema = z.object({
  schema: z.literal("search_disease_targets.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  disease_id: z.string(),
  disease_name: z.string(),
  requested_limit: z.number().int().positive(),
  total_associated_targets: z.number().int().nonnegative(),
  targets_returned: z.number().int().nonnegative(),
  targets: z.array(DiseaseTargetRowPayloadSchema),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildSearchDiseaseTargetsPayload({
  resultStatus = "ok",
  diseaseId = "",
  diseaseName = "",
  requestedLimit = 10,
  totalAssociatedTargets = 0,
  targets = [],
  notes = [],
  errorMessage = "",
}) {
  const normalizedTargets = (targets || []).map((target, idx) => ({
    rank: Math.max(1, toNonNegativeInt(target?.rank, idx + 1)),
    target_id: String(target?.target_id || ""),
    symbol: String(target?.symbol || "Unknown"),
    approved_name: String(target?.approved_name || "Unknown"),
    biotype: String(target?.biotype || "unknown"),
    overall_score_pct: Math.max(0, toFiniteNumber(target?.overall_score_pct, 0)),
    evidence: (target?.evidence || []).slice(0, 8).map((item) => ({
      id: String(item?.id || "unknown"),
      score_pct: Math.max(0, toFiniteNumber(item?.score_pct, 0)),
    })),
  }));
  return safeBuildTypedPayload(
    SearchDiseaseTargetsPayloadSchema,
    {
      schema: "search_disease_targets.v1",
      result_status: resultStatus,
      disease_id: String(diseaseId || ""),
      disease_name: String(diseaseName || ""),
      requested_limit: Math.max(1, toNonNegativeInt(requestedLimit, 10)),
      total_associated_targets: toNonNegativeInt(totalAssociatedTargets),
      targets_returned: normalizedTargets.length,
      targets: normalizedTargets,
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "search_disease_targets.v1"
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

const ExpandedDiseaseCandidatePayloadSchema = z.object({
  rank: z.number().int().positive(),
  label: z.string(),
  obo_id: z.string(),
  short_form: z.string(),
  description: z.string(),
  synonyms: z.array(z.string()),
  iri: z.string().optional(),
});

const ExpandedDiseaseTopTermPayloadSchema = z.object({
  label: z.string(),
  obo_id: z.string(),
  short_form: z.string(),
});

const ExpandDiseaseContextPayloadSchema = z.object({
  schema: z.literal("expand_disease_context.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  query: z.string(),
  ontology: z.string(),
  include_hierarchy: z.boolean(),
  candidate_count: z.number().int().nonnegative(),
  top_term: ExpandedDiseaseTopTermPayloadSchema.nullable(),
  candidates: z.array(ExpandedDiseaseCandidatePayloadSchema),
  parent_concepts: z.array(z.string()),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildExpandDiseaseContextPayload({
  resultStatus = "ok",
  query = "",
  ontology = "efo",
  includeHierarchy = true,
  topTerm = null,
  candidates = [],
  parentConcepts = [],
  notes = [],
  errorMessage = "",
}) {
  const normalizedCandidates = (candidates || []).map((candidate, idx) => ({
    rank: Math.max(1, toNonNegativeInt(candidate?.rank, idx + 1)),
    label: String(candidate?.label || "Unknown term"),
    obo_id: String(candidate?.obo_id || "N/A"),
    short_form: String(candidate?.short_form || "N/A"),
    description: String(candidate?.description || "No description"),
    synonyms: dedupeArray((candidate?.synonyms || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
    ...(candidate?.iri ? { iri: String(candidate.iri) } : {}),
  }));
  return safeBuildTypedPayload(
    ExpandDiseaseContextPayloadSchema,
    {
      schema: "expand_disease_context.v1",
      result_status: resultStatus,
      query: String(query || ""),
      ontology: String(ontology || "efo"),
      include_hierarchy: Boolean(includeHierarchy),
      candidate_count: normalizedCandidates.length,
      top_term: topTerm
        ? {
            label: String(topTerm?.label || "Unknown term"),
            obo_id: String(topTerm?.obo_id || "N/A"),
            short_form: String(topTerm?.short_form || "N/A"),
          }
        : null,
      candidates: normalizedCandidates,
      parent_concepts: dedupeArray((parentConcepts || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "expand_disease_context.v1"
  );
}

const ChemblSelectedTargetPayloadSchema = z.object({
  target_chembl_id: z.string(),
  pref_name: z.string(),
  target_type: z.string(),
  organism: z.string(),
});

const ChemblCompoundRowPayloadSchema = z.object({
  rank: z.number().int().positive(),
  molecule_chembl_id: z.string(),
  name: z.string(),
  standard_type: z.string(),
  relation: z.string(),
  standard_value: z.string(),
  standard_units: z.string(),
  standard_value_nm: z.number().nullable(),
  pchembl: z.number().nullable(),
  assay_chembl_id: z.string(),
  document_chembl_id: z.string(),
  document_year: z.string(),
});

const SearchChemblCompoundsPayloadSchema = z.object({
  schema: z.literal("search_chembl_compounds_for_target.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  query: z.string(),
  organism: z.string(),
  activity_type: z.string(),
  min_pchembl: z.number(),
  max_nanomolar: z.number().nullable(),
  selected_target: ChemblSelectedTargetPayloadSchema.nullable(),
  candidate_targets_considered: z.number().int().nonnegative(),
  compounds_returned: z.number().int().nonnegative(),
  compounds: z.array(ChemblCompoundRowPayloadSchema),
  top_candidate_target_matches: z.array(z.string()),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildSearchChemblCompoundsPayload({
  resultStatus = "ok",
  query = "",
  organism = "",
  activityType = "IC50",
  minPchembl = 6.0,
  maxNanomolar = null,
  selectedTarget = null,
  candidateTargetsConsidered = 0,
  compounds = [],
  topCandidateTargetMatches = [],
  notes = [],
  errorMessage = "",
}) {
  const normalizedCompounds = (compounds || []).map((compound, idx) => ({
    rank: Math.max(1, toNonNegativeInt(compound?.rank, idx + 1)),
    molecule_chembl_id: String(compound?.molecule_chembl_id || ""),
    name: String(compound?.name || "Unknown"),
    standard_type: String(compound?.standard_type || ""),
    relation: String(compound?.relation || "="),
    standard_value: String(compound?.standard_value || "N/A"),
    standard_units: String(compound?.standard_units || "N/A"),
    standard_value_nm: toNullableNumber(compound?.standard_value_nm),
    pchembl: toNullableNumber(compound?.pchembl),
    assay_chembl_id: String(compound?.assay_chembl_id || "N/A"),
    document_chembl_id: String(compound?.document_chembl_id || "N/A"),
    document_year: String(compound?.document_year || "N/A"),
  }));
  return safeBuildTypedPayload(
    SearchChemblCompoundsPayloadSchema,
    {
      schema: "search_chembl_compounds_for_target.v1",
      result_status: resultStatus,
      query: String(query || ""),
      organism: String(organism || ""),
      activity_type: String(activityType || "IC50"),
      min_pchembl: toFiniteNumber(minPchembl, 6.0),
      max_nanomolar: toNullableNumber(maxNanomolar),
      selected_target: selectedTarget
        ? {
            target_chembl_id: String(selectedTarget?.target_chembl_id || ""),
            pref_name: String(selectedTarget?.pref_name || "Unknown"),
            target_type: String(selectedTarget?.target_type || "Unknown"),
            organism: String(selectedTarget?.organism || "Unknown"),
          }
        : null,
      candidate_targets_considered: toNonNegativeInt(candidateTargetsConsidered),
      compounds_returned: normalizedCompounds.length,
      compounds: normalizedCompounds,
      top_candidate_target_matches: dedupeArray((topCandidateTargetMatches || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 8),
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "search_chembl_compounds_for_target.v1"
  );
}

const ExpressionRowPayloadSchema = z.object({
  rank: z.number().int().positive(),
  tissue_label: z.string(),
  anatomical_systems: z.array(z.string()),
  organs: z.array(z.string()),
  rna_value: z.number().nullable(),
  rna_unit: z.string(),
  rna_level: z.number().nullable(),
  rna_zscore: z.number().nullable(),
  protein_level: z.string(),
  protein_reliable: z.boolean(),
  top_cell_types: z.array(z.string()),
});

const TargetExpressionContextPayloadSchema = z.object({
  schema: z.literal("summarize_target_expression_context.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  target_id: z.string(),
  target_symbol: z.string(),
  target_name: z.string(),
  anatomical_system_filter: z.string(),
  include_cell_types: z.boolean(),
  rows_considered: z.number().int().nonnegative(),
  rows_returned: z.number().int().nonnegative(),
  dominant_systems: z.array(z.string()),
  dominant_organs: z.array(z.string()),
  dominant_cell_types: z.array(z.string()),
  expression_rows: z.array(ExpressionRowPayloadSchema),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildTargetExpressionContextPayload({
  resultStatus = "ok",
  targetId = "",
  targetSymbol = "",
  targetName = "",
  anatomicalSystemFilter = "",
  includeCellTypes = true,
  rowsConsidered = 0,
  expressionRows = [],
  dominantSystems = [],
  dominantOrgans = [],
  dominantCellTypes = [],
  notes = [],
  errorMessage = "",
}) {
  const normalizedRows = (expressionRows || []).map((row, idx) => ({
    rank: Math.max(1, toNonNegativeInt(row?.rank, idx + 1)),
    tissue_label: String(row?.tissue_label || "Unknown tissue"),
    anatomical_systems: dedupeArray((row?.anatomical_systems || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 6),
    organs: dedupeArray((row?.organs || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 6),
    rna_value: toNullableNumber(row?.rna_value),
    rna_unit: String(row?.rna_unit || ""),
    rna_level: toNullableNumber(row?.rna_level),
    rna_zscore: toNullableNumber(row?.rna_zscore),
    protein_level: String(row?.protein_level || "unknown"),
    protein_reliable: Boolean(row?.protein_reliable),
    top_cell_types: dedupeArray((row?.top_cell_types || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 8),
  }));
  return safeBuildTypedPayload(
    TargetExpressionContextPayloadSchema,
    {
      schema: "summarize_target_expression_context.v1",
      result_status: resultStatus,
      target_id: String(targetId || ""),
      target_symbol: String(targetSymbol || ""),
      target_name: String(targetName || ""),
      anatomical_system_filter: String(anatomicalSystemFilter || ""),
      include_cell_types: Boolean(includeCellTypes),
      rows_considered: toNonNegativeInt(rowsConsidered),
      rows_returned: normalizedRows.length,
      dominant_systems: dedupeArray((dominantSystems || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      dominant_organs: dedupeArray((dominantOrgans || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      dominant_cell_types: dedupeArray((dominantCellTypes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      expression_rows: normalizedRows,
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "summarize_target_expression_context.v1"
  );
}

const CompetitiveLeadAssetPayloadSchema = z.object({
  rank: z.number().int().positive(),
  drug_name: z.string(),
  drug_id: z.string(),
  phase_label: z.string(),
  phase_numeric: z.number().nullable(),
  withdrawn: z.boolean(),
  disease_name: z.string(),
  mechanism: z.string(),
});

const TargetCompetitiveLandscapePayloadSchema = z.object({
  schema: z.literal("summarize_target_competitive_landscape.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  target_id: z.string(),
  target_symbol: z.string(),
  target_name: z.string(),
  disease_filter: z.string(),
  rows_analyzed: z.number().int().nonnegative(),
  catalog_unique_drugs: z.number().int().nonnegative(),
  catalog_interactions: z.number().int().nonnegative(),
  catalog_unique_diseases: z.number().int().nonnegative(),
  unique_drugs_in_rows: z.number().int().nonnegative(),
  withdrawn_interactions: z.number().int().nonnegative(),
  phase_distribution: z.array(z.string()),
  top_diseases: z.array(z.string()),
  top_mechanisms: z.array(z.string()),
  modality_mix: z.array(z.string()),
  lead_assets: z.array(CompetitiveLeadAssetPayloadSchema),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildTargetCompetitiveLandscapePayload({
  resultStatus = "ok",
  targetId = "",
  targetSymbol = "",
  targetName = "",
  diseaseFilter = "",
  rowsAnalyzed = 0,
  catalogUniqueDrugs = 0,
  catalogInteractions = 0,
  catalogUniqueDiseases = 0,
  uniqueDrugsInRows = 0,
  withdrawnInteractions = 0,
  phaseDistribution = [],
  topDiseases = [],
  topMechanisms = [],
  modalityMix = [],
  leadAssets = [],
  notes = [],
  errorMessage = "",
}) {
  const normalizedLeadAssets = (leadAssets || []).map((asset, idx) => ({
    rank: Math.max(1, toNonNegativeInt(asset?.rank, idx + 1)),
    drug_name: String(asset?.drug_name || "Unknown"),
    drug_id: String(asset?.drug_id || ""),
    phase_label: String(asset?.phase_label || "Unknown"),
    phase_numeric: toNullableNumber(asset?.phase_numeric),
    withdrawn: Boolean(asset?.withdrawn),
    disease_name: String(asset?.disease_name || "Unspecified disease"),
    mechanism: String(asset?.mechanism || "Unknown mechanism"),
  }));
  return safeBuildTypedPayload(
    TargetCompetitiveLandscapePayloadSchema,
    {
      schema: "summarize_target_competitive_landscape.v1",
      result_status: resultStatus,
      target_id: String(targetId || ""),
      target_symbol: String(targetSymbol || ""),
      target_name: String(targetName || ""),
      disease_filter: String(diseaseFilter || ""),
      rows_analyzed: toNonNegativeInt(rowsAnalyzed),
      catalog_unique_drugs: toNonNegativeInt(catalogUniqueDrugs),
      catalog_interactions: toNonNegativeInt(catalogInteractions),
      catalog_unique_diseases: toNonNegativeInt(catalogUniqueDiseases),
      unique_drugs_in_rows: toNonNegativeInt(uniqueDrugsInRows),
      withdrawn_interactions: toNonNegativeInt(withdrawnInteractions),
      phase_distribution: dedupeArray((phaseDistribution || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      top_diseases: dedupeArray((topDiseases || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      top_mechanisms: dedupeArray((topMechanisms || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      modality_mix: dedupeArray((modalityMix || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      lead_assets: normalizedLeadAssets,
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "summarize_target_competitive_landscape.v1"
  );
}

const SafetyEventPayloadSchema = z.object({
  rank: z.number().int().positive(),
  event_name: z.string(),
  count: z.number().int().nonnegative(),
  directions: z.array(z.string()),
  study_types: z.array(z.string()),
  tissues: z.array(z.string()),
  dosing_signals: z.array(z.string()),
  datasources: z.array(z.string()),
});

const TargetSafetyLiabilitiesPayloadSchema = z.object({
  schema: z.literal("summarize_target_safety_liabilities.v1"),
  result_status: z.enum(["ok", "not_found_or_empty", "degraded", "error"]),
  target_id: z.string(),
  target_symbol: z.string(),
  target_name: z.string(),
  include_clinical_only: z.boolean(),
  event_filter: z.string(),
  liabilities_analyzed: z.number().int().nonnegative(),
  unique_events: z.number().int().nonnegative(),
  direction_pattern: z.array(z.string()),
  study_type_mix: z.array(z.string()),
  tissue_contexts: z.array(z.string()),
  datasource_mix: z.array(z.string()),
  events: z.array(SafetyEventPayloadSchema),
  notes: z.array(z.string()),
  error: z.string().optional(),
});

function buildTargetSafetyLiabilitiesPayload({
  resultStatus = "ok",
  targetId = "",
  targetSymbol = "",
  targetName = "",
  includeClinicalOnly = false,
  eventFilter = "",
  liabilitiesAnalyzed = 0,
  uniqueEvents = 0,
  directionPattern = [],
  studyTypeMix = [],
  tissueContexts = [],
  datasourceMix = [],
  events = [],
  notes = [],
  errorMessage = "",
}) {
  const normalizedEvents = (events || []).map((event, idx) => ({
    rank: Math.max(1, toNonNegativeInt(event?.rank, idx + 1)),
    event_name: String(event?.event_name || "Unspecified event"),
    count: toNonNegativeInt(event?.count),
    directions: dedupeArray((event?.directions || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 6),
    study_types: dedupeArray((event?.study_types || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 6),
    tissues: dedupeArray((event?.tissues || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 6),
    dosing_signals: dedupeArray((event?.dosing_signals || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 6),
    datasources: dedupeArray((event?.datasources || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 6),
  }));
  return safeBuildTypedPayload(
    TargetSafetyLiabilitiesPayloadSchema,
    {
      schema: "summarize_target_safety_liabilities.v1",
      result_status: resultStatus,
      target_id: String(targetId || ""),
      target_symbol: String(targetSymbol || ""),
      target_name: String(targetName || ""),
      include_clinical_only: Boolean(includeClinicalOnly),
      event_filter: String(eventFilter || ""),
      liabilities_analyzed: toNonNegativeInt(liabilitiesAnalyzed),
      unique_events: toNonNegativeInt(uniqueEvents),
      direction_pattern: dedupeArray((directionPattern || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      study_type_mix: dedupeArray((studyTypeMix || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      tissue_contexts: dedupeArray((tissueContexts || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      datasource_mix: dedupeArray((datasourceMix || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 10),
      events: normalizedEvents,
      notes: dedupeArray((notes || []).map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 12),
      ...(errorMessage ? { error: compactErrorMessage(errorMessage) } : {}),
    },
    "summarize_target_safety_liabilities.v1"
  );
}

function inferDirectionLabel(association) {
  const oddsRatio = safeNumber(association?.orPerCopyNum);
  if (Number.isFinite(oddsRatio)) {
    if (oddsRatio > 1) return "risk_increasing";
    if (oddsRatio < 1) return "protective";
    return "neutral";
  }
  const beta = safeNumber(association?.betaNum);
  if (Number.isFinite(beta)) {
    if (beta > 0) return "risk_increasing";
    if (beta < 0) return "protective";
    return "neutral";
  }
  const betaDirection = String(association?.betaDirection || "").toLowerCase();
  if (betaDirection.includes("increase") || betaDirection.includes("positive")) {
    return "risk_increasing";
  }
  if (betaDirection.includes("decrease") || betaDirection.includes("negative")) {
    return "protective";
  }
  return "unknown";
}

function associationRiskAllele(association) {
  if (association?.riskAllele) return association.riskAllele;
  return association?.loci?.[0]?.strongestRiskAlleles?.[0]?.riskAlleleName || "N/A";
}

function getDatatypeScore(row, datatypeId) {
  const scores = row?.datatypeScores || [];
  const match = scores.find((item) => String(item?.id || "") === datatypeId);
  return clamp01(match?.score ?? 0);
}

function normalizeTopRows(rows, limit = 8) {
  return rows
    .slice()
    .sort((a, b) => {
      const aScore = safeNumber(a?.score, 0);
      const bScore = safeNumber(b?.score, 0);
      return bScore - aScore;
    })
    .slice(0, limit);
}

async function buildOpenTargetsGeneGeneticsFallback(geneSymbol, diseaseQuery = "", limit = 8) {
  const resolved = await resolveTargetIdFromInput({ geneSymbol });
  if (resolved?.error || !resolved?.targetId) {
    return null;
  }

  const associatedDiseasesQuery = `
    query TargetAssociatedDiseases($targetId: String!, $size: Int!) {
      target(ensemblId: $targetId) {
        id
        approvedSymbol
        associatedDiseases(page: { size: $size, index: 0 }) {
          rows {
            disease {
              id
              name
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
  `;
  const result = await queryOpenTargets(associatedDiseasesQuery, {
    targetId: resolved.targetId,
    size: 120,
  });
  const target = result?.data?.target;
  if (!target) {
    return null;
  }

  const diseaseTokens = tokenizeQuery(diseaseQuery || "");
  let rows = target?.associatedDiseases?.rows || [];
  if (diseaseTokens.length > 0) {
    rows = rows.filter((row) => {
      const diseaseName = String(row?.disease?.name || "").toLowerCase();
      return diseaseTokens.some((token) => diseaseName.includes(token));
    });
  }

  const rowsWithGenetics = rows.filter(
    (row) =>
      getDatatypeScore(row, "genetic_association") > 0 ||
      getDatatypeScore(row, "genetic_literature") > 0
  );
  const selected = normalizeTopRows(rowsWithGenetics.length ? rowsWithGenetics : rows, limit);
  if (selected.length === 0) {
    return null;
  }

  const keyFields = selected.map((row, idx) => {
    const diseaseName = row?.disease?.name || "Unknown disease";
    const overall = clamp01(row?.score ?? 0);
    const ga = getDatatypeScore(row, "genetic_association");
    const gl = getDatatypeScore(row, "genetic_literature");
    return `${idx + 1}. ${diseaseName} | Overall score: ${formatPct(overall)} | Genetic association: ${formatPct(
      ga
    )} | Genetic literature: ${formatPct(gl)}`;
  });

  return {
    summary: `CRITICAL GAP: GWAS API unavailable; returning Open Targets genetics proxy for ${target.approvedSymbol}.`,
    keyFields: [
      `Target: ${target.approvedSymbol} (${target.id})`,
      ...(diseaseQuery ? [`Disease context filter: ${diseaseQuery}`] : []),
      ...keyFields,
    ],
    sources: [...(resolved.sourceHints || []), `${OPEN_TARGETS_API}#target.associatedDiseases:${target.id}`],
    limitations: [
      "Fallback uses Open Targets genetics evidence scores and does not provide SNP-level effect direction.",
      "Risk-increasing vs protective direction cannot be inferred from this fallback alone.",
    ],
  };
}

async function buildOpenTargetsDiseaseGeneticsFallback(diseaseQuery, limit = 8) {
  const resolvedDisease = await resolveDiseaseFromInput({ diseaseQuery });
  if (resolvedDisease?.error || !resolvedDisease?.diseaseId) {
    return null;
  }

  const diseaseTargetsQuery = `
    query DiseaseTargets($diseaseId: String!, $size: Int!) {
      disease(efoId: $diseaseId) {
        id
        name
        associatedTargets(page: { size: $size, index: 0 }) {
          rows {
            target {
              id
              approvedSymbol
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
  `;
  const result = await queryOpenTargets(diseaseTargetsQuery, {
    diseaseId: resolvedDisease.diseaseId,
    size: 120,
  });
  const disease = result?.data?.disease;
  if (!disease) {
    return null;
  }

  const rows = disease?.associatedTargets?.rows || [];
  const withGenetics = rows.filter(
    (row) =>
      getDatatypeScore(row, "genetic_association") > 0 ||
      getDatatypeScore(row, "genetic_literature") > 0
  );
  const selected = normalizeTopRows(withGenetics.length ? withGenetics : rows, limit);
  if (selected.length === 0) {
    return null;
  }

  const keyFields = selected.map((row, idx) => {
    const symbol = row?.target?.approvedSymbol || "Unknown";
    const targetId = row?.target?.id || "N/A";
    const overall = clamp01(row?.score ?? 0);
    const ga = getDatatypeScore(row, "genetic_association");
    const gl = getDatatypeScore(row, "genetic_literature");
    return `${idx + 1}. ${symbol} (${targetId}) | Overall score: ${formatPct(overall)} | Genetic association: ${formatPct(
      ga
    )} | Genetic literature: ${formatPct(gl)}`;
  });

  return {
    summary: `CRITICAL GAP: GWAS API unavailable; returning Open Targets genetics proxy for ${disease.name}.`,
    keyFields: [`Disease: ${disease.name} (${disease.id})`, ...keyFields],
    sources: [...(resolvedDisease.sourceHints || []), `${OPEN_TARGETS_API}#disease.associatedTargets:${disease.id}`],
    limitations: [
      "Fallback uses Open Targets genetics evidence scores and is not variant-level GWAS output.",
      "Risk-allele level statistics (OR/beta per rsID) are unavailable in this fallback.",
    ],
  };
}

// ============================================
// TOOL 1: Search PubMed for relevant papers
// ============================================
server.registerTool(
  "search_pubmed",
  {
    description: "Searches PubMed and returns recent papers with IDs and titles",
    inputSchema: {
      query: z.string().describe("PubMed search query (e.g., 'Alzheimer microglia single cell')"),
      retmax: z.number().optional().default(5).describe("Max number of results"),
      sort: z.string().optional().default("relevance").describe("Sort order: 'relevance' or 'date'"),
    },
  },
  async ({ query, retmax = 5, sort = "relevance" }) => {
    try {
      const searchUrl = `${NCBI_BASE}/esearch.fcgi?db=pubmed&term=${encodeURIComponent(
        query
      )}&retmax=${retmax}&sort=${encodeURIComponent(sort)}&retmode=json`;
      const search = await fetchJson(searchUrl);
      const ids = search?.esearchresult?.idlist ?? [];
      if (ids.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `No PubMed results found for query: "${query}". Try different keywords.`,
            },
          ],
        };
      }

      const summaryUrl = `${NCBI_BASE}/esummary.fcgi?db=pubmed&id=${ids.join(
        ","
      )}&retmode=json`;
      const summary = await fetchJson(summaryUrl);
      const results = ids.map((id) => ({
        pmid: id,
        title: summary?.result?.[id]?.title ?? "Untitled",
        pubdate: summary?.result?.[id]?.pubdate ?? "Unknown date",
        journal: summary?.result?.[id]?.fulljournalname ?? "Unknown journal",
      }));

      return {
        content: [
          {
            type: "text",
            text: `PubMed results for "${query}":\n\n` +
              results
                .map(
                  (item, index) =>
                    `${index + 1}. PMID ${item.pmid} — ${item.title} (${item.journal}, ${item.pubdate})`
                )
                .join("\n"),
          },
        ],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error searching PubMed: ${error.message}. Try again.` }],
      };
    }
  }
);

// ============================================
// TOOL 2: Fetch PubMed abstract by PMID
// ============================================
server.registerTool(
  "get_pubmed_abstract",
  {
    description: "Fetches a PubMed abstract and title for a given PMID",
    inputSchema: {
      pmid: z.string().describe("PubMed ID (PMID) to fetch"),
    },
  },
  async ({ pmid }) => {
    const fetchUrl = `${NCBI_BASE}/efetch.fcgi?db=pubmed&id=${encodeURIComponent(
      pmid
    )}&retmode=xml`;
    const xml = await fetchText(fetchUrl);
    const titleMatch = xml.match(/<ArticleTitle>([\s\S]*?)<\/ArticleTitle>/);
    const abstractMatches = [...xml.matchAll(/<AbstractText[^>]*>([\s\S]*?)<\/AbstractText>/g)];
    const title = sanitizeXmlText(titleMatch?.[1]);
    const abstract = sanitizeXmlText(
      abstractMatches.map((match) => match[1]).join(" ")
    );

    if (!title && !abstract) {
      return {
        content: [
          {
            type: "text",
            text: `No abstract found for PMID ${pmid}.`,
          },
        ],
      };
    }

    return {
      content: [
        {
          type: "text",
          text: `PMID ${pmid}\nTitle: ${title || "Untitled"}\nAbstract: ${
            abstract || "No abstract available."
          }`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 3: List local datasets (./data)
// ============================================
server.registerTool(
  "list_local_datasets",
  {
    description: "Lists files in the local ./data directory",
  },
  async () => {
    const files = await listDataFiles();
    if (files.length === 0) {
      return {
        content: [
          {
            type: "text",
            text:
              "No local datasets found. Add files to ./data (e.g., CSV, TSV, JSON).",
          },
        ],
      };
    }
    return {
      content: [
        {
          type: "text",
          text: `Local datasets:\n${files.map((file) => `- ${file}`).join("\n")}`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 4: Read a local dataset file (safe path)
// ============================================
server.registerTool(
  "read_local_dataset",
  {
    description: "Reads the first N lines from a local dataset in ./data",
    inputSchema: {
      filename: z.string().describe("Filename inside ./data"),
      maxLines: z.number().optional().default(200).describe("Max number of lines to return"),
    },
  },
  async ({ filename, maxLines = 200 }) => {
    const resolved = resolveDataPath(filename);
    const contents = await fs.readFile(resolved, "utf-8");
    const lines = contents.split(/\r?\n/).slice(0, maxLines);

    return {
      content: [
        {
          type: "text",
          text: lines.join("\n"),
        },
      ],
    };
  }
);

// ============================================
// TOOL 5: Search for diseases in Open Targets
// ============================================
server.registerTool(
  "search_diseases",
  {
    description:
      "Searches Open Targets for diseases matching a query. Returns disease IDs needed for target lookups.",
    inputSchema: {
      query: z.string().describe("Disease name to search (e.g., 'Alzheimer', 'Parkinson', 'breast cancer')"),
      limit: z.number().optional().default(5).describe("Max number of results"),
    },
  },
  async ({ query, limit = 5 }) => {
    const graphqlQuery = `
      query SearchDiseases($queryString: String!, $size: Int!) {
        search(queryString: $queryString, entityNames: ["disease"], page: { size: $size, index: 0 }) {
          hits {
            id
            name
            description
            entity
          }
        }
      }
    `;
    const result = await queryOpenTargets(graphqlQuery, { queryString: query, size: limit });
    const hits = result?.data?.search?.hits ?? [];

    if (hits.length === 0) {
      return {
        content: [{ type: "text", text: `No diseases found for query: "${query}"` }],
      };
    }

    const formatted = hits
      .map(
        (hit, i) =>
          `${i + 1}. ${hit.name}\n   ID: ${hit.id}\n   ${hit.description || "No description"}`
      )
      .join("\n\n");

    return {
      content: [
        {
          type: "text",
          text: `Diseases matching "${query}":\n\n${formatted}\n\nUse the disease ID (e.g., "${hits[0].id}") with search_disease_targets to find associated drug targets.`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 6: Get drug targets for a disease
// ============================================
server.registerTool(
  "search_disease_targets",
  {
    description:
      "Finds drug targets (genes/proteins) associated with a disease. Returns targets ranked by association score with evidence.",
    inputSchema: {
      diseaseId: z
        .string()
        .describe("Open Targets disease ID (e.g., 'EFO_0000249' for Alzheimer's). Use search_diseases first to find IDs."),
      limit: z.number().optional().default(10).describe("Max number of targets to return"),
    },
  },
  async ({ diseaseId, limit = 10 }) => {
    try {
      const boundedLimit = Math.max(1, Math.min(25, Math.round(limit)));
      const graphqlQuery = `
        query DiseaseTargets($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(page: { size: $size, index: 0 }) {
              count
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                  biotype
                }
                score
                datatypeScores {
                  id
                  score
                }
              }
            }
          }
        }
      `;
      const result = await queryOpenTargets(graphqlQuery, { diseaseId, size: boundedLimit });
      const disease = result?.data?.disease;

      if (!disease) {
        const message = `Disease not found: "${diseaseId}". Use search_diseases to find valid disease IDs.`;
        return {
          content: [
            {
              type: "text",
              text: message,
            },
          ],
          structuredContent: buildSearchDiseaseTargetsPayload({
            resultStatus: "not_found_or_empty",
            diseaseId,
            diseaseName: "",
            requestedLimit: boundedLimit,
            totalAssociatedTargets: 0,
            targets: [],
            notes: ["Disease ID did not resolve in Open Targets."],
          }),
        };
      }

      const rows = disease.associatedTargets?.rows ?? [];
      if (rows.length === 0) {
        return {
          content: [{ type: "text", text: `No targets found for disease: ${disease.name}` }],
          structuredContent: buildSearchDiseaseTargetsPayload({
            resultStatus: "not_found_or_empty",
            diseaseId: disease.id,
            diseaseName: disease.name,
            requestedLimit: boundedLimit,
            totalAssociatedTargets: toNonNegativeInt(disease?.associatedTargets?.count),
            targets: [],
            notes: ["No associated targets were returned for this disease."],
          }),
        };
      }

      const targetPayloadRows = rows.map((row, idx) => {
        const datatypeScores = Array.isArray(row?.datatypeScores) ? row.datatypeScores : [];
        return {
          rank: idx + 1,
          target_id: String(row?.target?.id || ""),
          symbol: String(row?.target?.approvedSymbol || "Unknown"),
          approved_name: String(row?.target?.approvedName || "Unknown"),
          biotype: String(row?.target?.biotype || "unknown"),
          overall_score_pct: Math.max(0, toFiniteNumber(row?.score, 0) * 100),
          evidence: datatypeScores
            .filter((item) => toFiniteNumber(item?.score, 0) > 0)
            .map((item) => ({
              id: String(item?.id || "unknown"),
              score_pct: Math.max(0, toFiniteNumber(item?.score, 0) * 100),
            })),
        };
      });

      const formatted = rows
        .map((row, i) => {
          const t = row.target;
          const evidenceTypes = (Array.isArray(row?.datatypeScores) ? row.datatypeScores : [])
            .filter((d) => toFiniteNumber(d?.score, 0) > 0)
            .map((d) => `${d.id}: ${(toFiniteNumber(d?.score, 0) * 100).toFixed(0)}%`)
            .join(", ");
          return `${i + 1}. ${t.approvedSymbol} (${t.approvedName})
   Target ID: ${t.id}
   Overall Score: ${(toFiniteNumber(row?.score, 0) * 100).toFixed(1)}%
   Evidence: ${evidenceTypes || "N/A"}
   Type: ${t.biotype}`;
        })
        .join("\n\n");

      return {
        content: [
          {
            type: "text",
            text: `Top ${rows.length} drug targets for ${disease.name} (${disease.id}):\nTotal associated targets: ${disease.associatedTargets.count}\n\n${formatted}\n\nUse get_target_info or check_druggability with the Target ID for more details.`,
          },
        ],
        structuredContent: buildSearchDiseaseTargetsPayload({
          resultStatus: "ok",
          diseaseId: disease.id,
          diseaseName: disease.name,
          requestedLimit: boundedLimit,
          totalAssociatedTargets: toNonNegativeInt(disease?.associatedTargets?.count),
          targets: targetPayloadRows,
          notes: ["Overall score and datatype evidence percentages are from Open Targets associatedTargets rows."],
        }),
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in search_disease_targets: ${error.message}` }],
        structuredContent: buildSearchDiseaseTargetsPayload({
          resultStatus: "error",
          diseaseId,
          diseaseName: "",
          requestedLimit: limit,
          totalAssociatedTargets: 0,
          targets: [],
          notes: ["Unexpected error during disease target lookup."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 7: Get detailed target information
// ============================================
server.registerTool(
  "get_target_info",
  {
    description:
      "Gets detailed information about a drug target (gene/protein) including function, pathways, and tractability.",
    inputSchema: {
      targetId: z
        .string()
        .describe("Ensembl gene ID (e.g., 'ENSG00000142192' for EGFR). Get this from search_disease_targets."),
    },
  },
  async ({ targetId }) => {
    const graphqlQuery = `
      query TargetInfo($targetId: String!) {
        target(ensemblId: $targetId) {
          id
          approvedSymbol
          approvedName
          biotype
          functionDescriptions
          subcellularLocations {
            location
          }
          pathways {
            pathway
            pathwayId
          }
          tractability {
            label
            modality
            value
          }
          synonyms {
            label
          }
        }
      }
    `;
    const result = await queryOpenTargets(graphqlQuery, { targetId });
    const target = result?.data?.target;

    if (!target) {
      return {
        content: [{ type: "text", text: `Target not found: "${targetId}"` }],
      };
    }

    const functions = target.functionDescriptions?.slice(0, 3).join("\n   ") || "No function data";
    const locations =
      target.subcellularLocations?.map((l) => l.location).join(", ") || "Unknown";
    const pathways =
      target.pathways?.slice(0, 5).map((p) => p.pathway).join(", ") || "None listed";
    const tractability =
      target.tractability
        ?.filter((t) => t.value === true)
        .map((t) => `${t.modality}: ${t.label}`)
        .join(", ") || "No tractability data";
    const synonyms = target.synonyms?.slice(0, 5).map((s) => s.label).join(", ") || "None";

    return {
      content: [
        {
          type: "text",
          text: `Target: ${target.approvedSymbol} (${target.approvedName})
ID: ${target.id}
Type: ${target.biotype}
Synonyms: ${synonyms}

Function:
   ${functions}

Subcellular Location: ${locations}

Key Pathways: ${pathways}

Tractability (druggability indicators):
   ${tractability}

Use check_druggability for detailed druggability assessment, or get_target_drugs to see known drugs.`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 8: Check target druggability
// ============================================
server.registerTool(
  "check_druggability",
  {
    description:
      "Assesses whether a target is druggable - can it be modulated by small molecules, antibodies, or other modalities?",
    inputSchema: {
      targetId: z.string().describe("Ensembl gene ID (e.g., 'ENSG00000142192')"),
    },
  },
  async ({ targetId }) => {
    const graphqlQuery = `
      query Druggability($targetId: String!) {
        target(ensemblId: $targetId) {
          id
          approvedSymbol
          approvedName
          tractability {
            label
            modality
            value
          }
          knownDrugs {
            uniqueDrugs
            count
          }
        }
      }
    `;
    const result = await queryOpenTargets(graphqlQuery, { targetId });
    const target = result?.data?.target;

    if (!target) {
      return {
        content: [{ type: "text", text: `Target not found: "${targetId}"` }],
      };
    }

    // Group tractability by modality
    const tractByModality = {};
    for (const t of target.tractability || []) {
      if (!tractByModality[t.modality]) tractByModality[t.modality] = [];
      if (t.value === true) tractByModality[t.modality].push(t.label);
    }

    const modalityLines = Object.entries(tractByModality)
      .map(([modality, labels]) => {
        if (labels.length === 0) return `   ${modality}: No positive indicators`;
        return `   ${modality}: ${labels.join(", ")}`;
      })
      .join("\n");

    const drugInfo = target.knownDrugs;
    const drugSummary =
      drugInfo && drugInfo.uniqueDrugs > 0
        ? `Yes - ${drugInfo.uniqueDrugs} unique drugs (${drugInfo.count} drug-target interactions)`
        : "No known drugs targeting this protein";

    // Calculate overall druggability assessment
    const hasSmallMolecule = tractByModality["SM"]?.length > 0;
    const hasAntibody = tractByModality["AB"]?.length > 0;
    const hasOther = tractByModality["PR"]?.length > 0 || tractByModality["OC"]?.length > 0;
    
    let assessment = "LOW";
    if (hasSmallMolecule || (drugInfo?.uniqueDrugs > 0)) assessment = "HIGH";
    else if (hasAntibody || hasOther) assessment = "MEDIUM";

    return {
      content: [
        {
          type: "text",
          text: `Druggability Assessment for ${target.approvedSymbol} (${target.approvedName})

Overall Druggability: ${assessment}

Known Drugs: ${drugSummary}

Tractability by Modality:
${modalityLines || "   No tractability data available"}

Legend:
- SM = Small Molecule
- AB = Antibody
- PR = PROTAC
- OC = Other Clinical

${assessment === "HIGH" ? "This target has strong evidence of being druggable." : assessment === "MEDIUM" ? "This target may be druggable with certain modalities." : "This target may be challenging to drug with current approaches."}`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 9: Get known drugs for a target
// ============================================
server.registerTool(
  "get_target_drugs",
  {
    description: "Gets known drugs that target a specific gene/protein, including clinical trial status.",
    inputSchema: {
      targetId: z.string().describe("Ensembl gene ID (e.g., 'ENSG00000146648' for EGFR)"),
      limit: z.number().optional().default(10).describe("Max number of drugs to return"),
    },
  },
  async ({ targetId, limit = 10 }) => {
    const graphqlQuery = `
      query TargetDrugs($targetId: String!) {
        target(ensemblId: $targetId) {
          id
          approvedSymbol
          approvedName
          knownDrugs {
            uniqueDrugs
            count
            rows {
              drug {
                id
                name
                drugType
                maximumClinicalTrialPhase
                hasBeenWithdrawn
                description
              }
              mechanismOfAction
              disease {
                id
                name
              }
            }
          }
        }
      }
    `;
    const result = await queryOpenTargets(graphqlQuery, { targetId });
    const target = result?.data?.target;

    if (!target) {
      return {
        content: [{ type: "text", text: `Target not found: "${targetId}"` }],
      };
    }

    const drugs = target.knownDrugs;
    if (!drugs || drugs.uniqueDrugs === 0) {
      return {
        content: [
          {
            type: "text",
            text: `No known drugs for ${target.approvedSymbol} (${target.approvedName}).\n\nThis could be an opportunity for novel drug development, or the target may be challenging to drug.`,
          },
        ],
      };
    }

    const phaseLabels = {
      4: "Approved",
      3: "Phase III",
      2: "Phase II",
      1: "Phase I",
      0.5: "Early Phase I",
      0: "Preclinical",
    };

    const formatted = drugs.rows
      .slice(0, limit)
      .map((row, i) => {
        const d = row.drug;
        const phase = phaseLabels[d.maximumClinicalTrialPhase] || `Phase ${d.maximumClinicalTrialPhase}`;
        const withdrawn = d.hasBeenWithdrawn ? " [WITHDRAWN]" : "";
        return `${i + 1}. ${d.name}${withdrawn}
   Type: ${d.drugType}
   Status: ${phase}
   Mechanism: ${row.mechanismOfAction || "Unknown"}
   Indication: ${row.disease?.name || "Various"}`;
      })
      .join("\n\n");

    return {
      content: [
        {
          type: "text",
          text: `Known drugs for ${target.approvedSymbol} (${target.approvedName}):
Total: ${drugs.uniqueDrugs} unique drugs, ${drugs.count} interactions

${formatted}

Note: A target with approved drugs validates it as druggable. Multiple drugs may indicate competitive landscape.`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 10: Get gene info from NCBI
// ============================================
server.registerTool(
  "get_gene_info",
  {
    description:
      "Gets gene information from NCBI Gene database including summary, aliases, and genomic location.",
    inputSchema: {
      geneSymbol: z.string().describe("Gene symbol (e.g., 'BRCA1', 'EGFR', 'TP53')"),
    },
  },
  async ({ geneSymbol }) => {
    // First search for the gene to get its ID
    const searchUrl = `${NCBI_BASE}/esearch.fcgi?db=gene&term=${encodeURIComponent(
      geneSymbol
    )}[sym]+AND+human[orgn]&retmode=json`;
    const search = await fetchJson(searchUrl);
    const ids = search?.esearchresult?.idlist ?? [];

    if (ids.length === 0) {
      return {
        content: [{ type: "text", text: `No gene found for symbol: "${geneSymbol}"` }],
      };
    }

    // Fetch gene summary
    const summaryUrl = `${NCBI_BASE}/esummary.fcgi?db=gene&id=${ids[0]}&retmode=json`;
    const summary = await fetchJson(summaryUrl);
    const gene = summary?.result?.[ids[0]];

    if (!gene) {
      return {
        content: [{ type: "text", text: `Could not retrieve info for gene: "${geneSymbol}"` }],
      };
    }

    const aliases = gene.otheraliases || "None";
    const otherDesignations = gene.otherdesignations || "None";

    return {
      content: [
        {
          type: "text",
          text: `Gene: ${gene.name} (${gene.description})
NCBI Gene ID: ${ids[0]}
Organism: ${gene.organism?.scientificname || "Homo sapiens"}

Summary:
${gene.summary || "No summary available"}

Aliases: ${aliases}
Other Names: ${otherDesignations}

Genomic Location:
   Chromosome: ${gene.chromosome || "Unknown"}
   Map Location: ${gene.maplocation || "Unknown"}

Links:
   NCBI Gene: https://www.ncbi.nlm.nih.gov/gene/${ids[0]}`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 11: Search targets by gene symbol
// ============================================
server.registerTool(
  "search_targets",
  {
    description:
      "Searches Open Targets for genes/proteins by symbol or name. Returns target IDs for use with other tools.",
    inputSchema: {
      query: z.string().describe("Gene symbol or name (e.g., 'BRCA1', 'epidermal growth factor')"),
      limit: z.number().optional().default(5).describe("Max number of results"),
    },
  },
  async ({ query, limit = 5 }) => {
    const graphqlQuery = `
      query SearchTargets($queryString: String!, $size: Int!) {
        search(queryString: $queryString, entityNames: ["target"], page: { size: $size, index: 0 }) {
          hits {
            id
            name
            description
            entity
          }
        }
      }
    `;
    const result = await queryOpenTargets(graphqlQuery, { queryString: query, size: limit });
    const hits = result?.data?.search?.hits ?? [];

    if (hits.length === 0) {
      return {
        content: [{ type: "text", text: `No targets found for query: "${query}"` }],
      };
    }

    const formatted = hits
      .map(
        (hit, i) =>
          `${i + 1}. ${hit.name}\n   Target ID: ${hit.id}\n   ${hit.description?.slice(0, 150) || "No description"}...`
      )
      .join("\n\n");

    return {
      content: [
        {
          type: "text",
          text: `Targets matching "${query}":\n\n${formatted}\n\nUse the Target ID with get_target_info, check_druggability, or get_target_drugs.`,
        },
      ],
    };
  }
);

// ============================================
// TOOL 12: Search clinical trials
// ============================================
server.registerTool(
  "search_clinical_trials",
  {
    description:
      "Searches ClinicalTrials.gov for clinical trials. Find trials by disease, drug, target/gene, or sponsor. Returns trial status, phase, and key details.",
    inputSchema: {
      query: z
        .string()
        .describe("Search terms (e.g., 'LRRK2 Parkinson', 'pembrolizumab lung cancer', 'Alzheimer Phase 3')"),
      status: z
        .string()
        .optional()
        .describe("Filter by status: 'RECRUITING', 'COMPLETED', 'ACTIVE_NOT_RECRUITING', 'TERMINATED', or leave empty for all"),
      limit: z.number().optional().default(10).describe("Max number of results"),
    },
  },
  async ({ query, status, limit = 10 }) => {
    const params = new URLSearchParams({
      "query.term": query,
      pageSize: String(limit),
      format: "json",
    });

    if (status) {
      params.append("filter.overallStatus", status);
    }

    const url = `${CLINICAL_TRIALS_API}/studies?${params.toString()}`;
    
    let studies = [];
    let resultCount = 0;
    let hasMorePages = false;
    try {
      const response = await fetch(url);
      
      if (!response.ok) {
        return {
          content: [{ type: "text", text: `ClinicalTrials.gov API error (${response.status}). Try a different search term.` }],
        };
      }
      
      const text = await response.text();
      if (!text || text.trim() === '') {
        return {
          content: [{ type: "text", text: `ClinicalTrials.gov returned empty response for: "${query}". Try broader search terms.` }],
        };
      }
      
      const data = JSON.parse(text);
      studies = data?.studies ?? [];
      resultCount = Number.isFinite(data?.totalCount) ? data.totalCount : studies.length;
      hasMorePages = Boolean(data?.nextPageToken);
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error searching clinical trials: ${error.message}. Try again or use different search terms.` }],
      };
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
          text: `Clinical trials for "${query}":\nFound ${resultCount} trials${hasMorePages ? "\nNote: Additional pages of results are available from ClinicalTrials.gov." : ""}\n\n${formatted}\n\nUse get_clinical_trial with the NCT ID for full details including results.`,
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
// TOOL 14: Advanced PubMed search
// ============================================
server.registerTool(
  "search_pubmed_advanced",
  {
    description: "Advanced PubMed search with date/type/journal filters and basic author metadata.",
    inputSchema: {
      query: z.string().describe("Search query"),
      startDate: z.string().optional().describe("Start date YYYY/MM/DD"),
      endDate: z.string().optional().describe("End date YYYY/MM/DD"),
      articleType: z.string().optional().describe("Article type, e.g. Clinical Trial, Review"),
      journal: z.string().optional().describe("Journal name filter"),
      retmax: z.number().optional().default(10),
      sort: z.string().optional().default("relevance"),
    },
  },
  async ({ query, startDate, endDate, articleType, journal, retmax = 10, sort = "relevance" }) => {
    try {
      const clauses = [query];
      if (journal) clauses.push(`${journal}[Journal]`);
      if (articleType) clauses.push(`${articleType}[Publication Type]`);
      if (startDate && endDate) clauses.push(`("${startDate}"[Date - Publication] : "${endDate}"[Date - Publication])`);
      const term = clauses.join(" AND ");

      const searchUrl = `${NCBI_BASE}/esearch.fcgi?db=pubmed&term=${encodeURIComponent(term)}&retmax=${retmax}&sort=${encodeURIComponent(sort)}&retmode=json`;
      const search = await fetchJsonWithRetry(searchUrl);
      const ids = search?.esearchresult?.idlist ?? [];
      if (ids.length === 0) {
        return { content: [{ type: "text", text: renderStructuredResponse({ summary: "No PubMed records matched the advanced query.", keyFields: [`Query: ${term}`], sources: [searchUrl], limitations: ["Try broader terms or remove filters."] }) }] };
      }

      const summaryUrl = `${NCBI_BASE}/esummary.fcgi?db=pubmed&id=${ids.join(",")}&retmode=json`;
      const summary = await fetchJsonWithRetry(summaryUrl);
      const keyFields = ids.map((id, idx) => {
        const item = summary?.result?.[id] || {};
        const firstAuthor = item.authors?.[0]?.name || "Unknown";
        const pubdate = normalizeWhitespace(item.pubdate || "Unknown date");
        const title = normalizeWhitespace(item.title || "Untitled");
        const doiMarkdown = buildDoiMarkdown(extractPubmedSummaryDoi(item));
        const pubmedLink = `[PMID:${id}](https://pubmed.ncbi.nlm.nih.gov/${id}/)`;
        return `${idx + 1}. ${firstAuthor}${item.authors?.length > 1 ? " et al." : ""} (${pubdate}). ${title}. ${pubmedLink}${doiMarkdown ? ` ${doiMarkdown}` : ""}`;
      });
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Retrieved ${ids.length} PubMed records.`,
              keyFields,
              sources: [searchUrl, summaryUrl],
              limitations: ["Author metadata here is summary-level; use get_pubmed_paper_details for richer author/affiliation details."],
            }),
          },
        ],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_pubmed_advanced: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 15: PubMed paper details including authors
// ============================================
server.registerTool(
  "get_pubmed_paper_details",
  {
    description: "Get PubMed paper details including title, abstract, authors, and affiliations.",
    inputSchema: {
      pmid: z.string().describe("PubMed ID"),
    },
  },
  async ({ pmid }) => {
    try {
      const fetchUrl = `${NCBI_BASE}/efetch.fcgi?db=pubmed&id=${encodeURIComponent(pmid)}&retmode=xml`;
      const xml = await fetchText(fetchUrl);
      const title = sanitizeXmlText(xml.match(/<ArticleTitle>([\s\S]*?)<\/ArticleTitle>/)?.[1] || "");
      const abstract = sanitizeXmlText(
        [...xml.matchAll(/<AbstractText[^>]*>([\s\S]*?)<\/AbstractText>/g)].map((m) => m[1]).join(" ")
      );
      const journal = sanitizeXmlText(xml.match(/<Title>([\s\S]*?)<\/Title>/)?.[1] || "");
      const year = sanitizeXmlText(xml.match(/<PubDate>[\s\S]*?<Year>(\d{4})<\/Year>[\s\S]*?<\/PubDate>/)?.[1] || "");
      const doi = normalizeDoiValue(
        xml.match(/<ELocationID[^>]*EIdType="doi"[^>]*>([\s\S]*?)<\/ELocationID>/i)?.[1] || ""
      );
      const authors = parsePubmedAuthors(xml);
      const firstAuthor = normalizeWhitespace(authors?.[0]?.name || "Unknown author");
      const doiMarkdown = buildDoiMarkdown(doi);
      const pubmedMarkdown = `[PMID:${pmid}](https://pubmed.ncbi.nlm.nih.gov/${pmid}/)`;
      const citationLine = `${firstAuthor}${authors.length > 1 ? " et al." : ""} (${year || "n.d."}). ${title || "Untitled"}${journal ? `. ${journal}.` : "."} ${pubmedMarkdown}${doiMarkdown ? ` ${doiMarkdown}` : ""}`;
      const keyFields = [
        `Citation: ${citationLine}`,
        `PMID: ${pmid}`,
        `Title: ${title || "Untitled"}`,
        `Author count: ${authors.length}`,
        ...authors.slice(0, 10).map((a, idx) => `${idx + 1}. ${a.name}${a.affiliations[0] ? ` | ${a.affiliations[0]}` : ""}`),
      ];
      if (abstract) keyFields.push(`Abstract: ${abstract.slice(0, 600)}${abstract.length > 600 ? "..." : ""}`);
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Detailed paper metadata for PMID ${pmid}.`,
              keyFields,
              sources: [fetchUrl, `https://pubmed.ncbi.nlm.nih.gov/${pmid}/`],
              limitations: ["Corresponding author email is not consistently available in PubMed XML."],
            }),
          },
        ],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_pubmed_paper_details: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 16: PubMed author profile summary
// ============================================
server.registerTool(
  "get_pubmed_author_profile",
  {
    description: "Summarize a researcher profile from PubMed by author name.",
    inputSchema: {
      authorName: z.string().describe("Author name, e.g. John Smith"),
      topicQuery: z.string().optional().describe("Optional topic constraint"),
      retmax: z.number().optional().default(25),
    },
  },
  async ({ authorName, topicQuery, retmax = 25 }) => {
    try {
      const term = `${authorName}[Author]${topicQuery ? ` AND (${topicQuery})` : ""}`;
      const searchUrl = `${NCBI_BASE}/esearch.fcgi?db=pubmed&term=${encodeURIComponent(term)}&retmax=${retmax}&sort=date&retmode=json`;
      const search = await fetchJsonWithRetry(searchUrl);
      const ids = search?.esearchresult?.idlist ?? [];
      if (ids.length === 0) {
        return { content: [{ type: "text", text: renderStructuredResponse({ summary: "No PubMed publications found for author profile query.", keyFields: [`Query: ${term}`], sources: [searchUrl], limitations: ["Try alternate author spelling or initials."] }) }] };
      }
      const summaryUrl = `${NCBI_BASE}/esummary.fcgi?db=pubmed&id=${ids.join(",")}&retmode=json`;
      const summary = await fetchJsonWithRetry(summaryUrl);
      const journals = {};
      const years = {};
      ids.forEach((id) => {
        const item = summary?.result?.[id] || {};
        const journal = item.fulljournalname || "Unknown";
        journals[journal] = (journals[journal] || 0) + 1;
        const year = (item.pubdate || "").slice(0, 4) || "Unknown";
        years[year] = (years[year] || 0) + 1;
      });
      const topJournals = Object.entries(journals).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([j, c]) => `${j} (${c})`);
      const recentYears = Object.entries(years).sort((a, b) => a[0].localeCompare(b[0])).map(([y, c]) => `${y}:${c}`);
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `PubMed author profile for ${authorName}.`,
              keyFields: [
                `Matched publications: ${ids.length}`,
                `Topic constraint: ${topicQuery || "none"}`,
                `Top journals: ${topJournals.join(", ") || "N/A"}`,
                `Publication years: ${recentYears.join(", ") || "N/A"}`,
              ],
              sources: [searchUrl, summaryUrl],
              limitations: ["Author disambiguation can be ambiguous for common names."],
            }),
          },
        ],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_pubmed_author_profile: ${error.message}` }] };
    }
  }
);

// ============================================
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
// TOOL 21: Search ClinVar variants
// ============================================
server.registerTool(
  "search_clinvar_variants",
  {
    description: "Search ClinVar variants by gene/condition/query term.",
    inputSchema: {
      query: z.string(),
      retmax: z.number().optional().default(10),
    },
  },
  async ({ query, retmax = 10 }) => {
    try {
      const searchUrl = `${NCBI_BASE}/esearch.fcgi?db=clinvar&term=${encodeURIComponent(query)}&retmax=${retmax}&retmode=json`;
      const search = await fetchJsonWithRetry(searchUrl);
      const ids = search?.esearchresult?.idlist ?? [];
      if (ids.length === 0) {
        return { content: [{ type: "text", text: renderStructuredResponse({ summary: "No ClinVar variants found.", keyFields: [`Query: ${query}`], sources: [searchUrl], limitations: ["Try gene symbol + condition or variant notation."] }) }] };
      }
      const summaryUrl = `${NCBI_BASE}/esummary.fcgi?db=clinvar&id=${ids.join(",")}&retmode=json`;
      const summary = await fetchJsonWithRetry(summaryUrl);
      const keyFields = ids.map((id, idx) => {
        const item = summary?.result?.[id] || {};
        return `${idx + 1}. ClinVar ID ${id} | ${item.title || "Untitled"} | Clinical significance: ${item.clinical_significance?.description || "N/A"}`;
      });
      return { content: [{ type: "text", text: renderStructuredResponse({ summary: `Retrieved ${ids.length} ClinVar records.`, keyFields, sources: [searchUrl, summaryUrl], limitations: ["ClinVar assertions may be conflicting across submitters."] }) }] };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_clinvar_variants: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 22: ClinVar variant details
// ============================================
server.registerTool(
  "get_clinvar_variant_details",
  {
    description: "Get detailed ClinVar metadata for a specific ClinVar record ID.",
    inputSchema: {
      clinvarId: z.string().describe("ClinVar numeric record ID"),
    },
  },
  async ({ clinvarId }) => {
    try {
      const summaryUrl = `${NCBI_BASE}/esummary.fcgi?db=clinvar&id=${encodeURIComponent(clinvarId)}&retmode=json`;
      const summary = await fetchJsonWithRetry(summaryUrl);
      const item = summary?.result?.[clinvarId];
      if (!item) {
        return { content: [{ type: "text", text: `No ClinVar details found for ID ${clinvarId}.` }] };
      }
      const keyFields = [
        `ClinVar ID: ${clinvarId}`,
        `Title: ${item.title || "Untitled"}`,
        `Clinical significance: ${item.clinical_significance?.description || "N/A"}`,
        `Review status: ${item.clinical_significance?.review_status || "N/A"}`,
        `Last evaluated: ${item.clinical_significance?.last_evaluated || "N/A"}`,
      ];
      return { content: [{ type: "text", text: renderStructuredResponse({ summary: `ClinVar details for record ${clinvarId}.`, keyFields, sources: [summaryUrl], limitations: ["ClinVar data can contain conflicting submissions; inspect submitter-level evidence when critical."] }) }] };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_clinvar_variant_details: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 23: Search GWAS associations
// ============================================
server.registerTool(
  "search_gwas_associations",
  {
    description: "Search GWAS Catalog associations by trait, gene symbol, or rsID.",
    inputSchema: {
      query: z.string(),
      limit: z.number().optional().default(10),
      timeBudgetSec: z.number().optional().default(20).describe("Soft runtime budget in seconds."),
    },
  },
  async ({ query, limit = 10, timeBudgetSec = 20 }) => {
    try {
      const isRsId = /^rs\d+$/i.test(query.trim());
      let associations = [];
      const sourceUrls = [];
      const boundedBudgetSec = Math.max(8, Math.min(40, Math.round(timeBudgetSec)));
      const deadlineMs = Date.now() + boundedBudgetSec * 1000;
      let timedOutEarly = false;

      if (isRsId) {
        const sourceUrl = `${GWAS_API}/associations/search/findByRsId?rsId=${encodeURIComponent(query)}`;
        sourceUrls.push(sourceUrl);
        const data = await fetchGwasJson(sourceUrl, { timeoutMs: 9000, retries: 1 });
        associations = data?._embedded?.associations || [];
      } else if (/^[A-Z0-9-]{2,}$/.test(query.trim())) {
        const snpSearchUrl = `${GWAS_API}/singleNucleotidePolymorphisms/search/findByGene?geneName=${encodeURIComponent(query.trim())}&page=0&size=${Math.min(
          12,
          Math.max(limit, 8)
        )}`;
        sourceUrls.push(snpSearchUrl);
        const snpData = await fetchGwasJson(snpSearchUrl, { timeoutMs: 9000, retries: 1 });
        const snps = snpData?._embedded?.singleNucleotidePolymorphisms || [];
        const seen = new Set();

        for (const snp of snps.slice(0, Math.min(6, Math.max(Math.min(limit, 6), 4)))) {
          if (Date.now() >= deadlineMs) {
            timedOutEarly = true;
            break;
          }
          const assocLink = snp?._links?.associations?.href;
          if (!assocLink) continue;
          sourceUrls.push(assocLink);
          const assocData = await fetchGwasJson(assocLink, { timeoutMs: 8000, retries: 1 });
          for (const association of assocData?._embedded?.associations || []) {
            const assocId = association.associationId || association.id || association?._links?.self?.href;
            if (assocId && seen.has(assocId)) continue;
            if (assocId) seen.add(assocId);
            associations.push(association);
            if (associations.length >= limit) break;
          }
          if (associations.length >= limit) break;
        }
      } else {
        // Modern GWAS API supports direct trait lookup via findByEfoTrait.
        const directTraitUrl = `${GWAS_API}/associations/search/findByEfoTrait?efoTrait=${encodeURIComponent(query)}&page=0&size=${Math.max(limit, 10)}`;
        sourceUrls.push(directTraitUrl);
        try {
          const traitAssocData = await fetchGwasJson(directTraitUrl, { timeoutMs: 9000, retries: 1 });
          associations = traitAssocData?._embedded?.associations || [];
        } catch (error) {
          // Fallback for older deployments where trait lookup is routed through efoTraits search.
          const sourceUrl = `${GWAS_API}/efoTraits/search/findByTrait?trait=${encodeURIComponent(query)}`;
          sourceUrls.push(sourceUrl);
          const traitData = await fetchGwasJson(sourceUrl, { timeoutMs: 8000, retries: 1 });
          const traits = traitData?._embedded?.efoTraits || [];
          for (const trait of traits.slice(0, 3)) {
            if (Date.now() >= deadlineMs) {
              timedOutEarly = true;
              break;
            }
            const assocLink = trait?._links?.associations?.href;
            if (!assocLink) continue;
            sourceUrls.push(assocLink);
            const assocData = await fetchGwasJson(assocLink, { timeoutMs: 8000, retries: 1 });
            associations.push(...(assocData?._embedded?.associations || []));
            if (associations.length >= limit) break;
          }
        }
      }

      associations = associations.slice(0, limit);
      const sources = sourceUrls.slice(0, 10);
      if (associations.length === 0) {
        return { content: [{ type: "text", text: renderStructuredResponse({ summary: "No GWAS associations found.", keyFields: [`Query: ${query}`], sources: sources.length ? sources : [GWAS_API], limitations: ["GWAS endpoint behavior varies by query type; try rsID, exact trait phrase, or gene symbol."] }) }] };
      }

      const keyFields = associations.map((a, idx) => {
        const riskAllele = a.riskAllele || a?.loci?.[0]?.strongestRiskAlleles?.[0]?.riskAlleleName || "N/A";
        return `${idx + 1}. Association ID: ${a.associationId || a.id || "N/A"} | p-value: ${a.pvalue || "N/A"} | OR/Beta: ${a.orPerCopyNum || a.betaNum || "N/A"} | Risk allele: ${riskAllele}`;
      });
      if (timedOutEarly) {
        keyFields.push(`Search stopped early due to runtime budget (${boundedBudgetSec}s); results are partial.`);
      }
      return { content: [{ type: "text", text: renderStructuredResponse({ summary: `Retrieved ${associations.length} GWAS associations.`, keyFields, sources: sources.length ? sources : [GWAS_API], limitations: ["Trait/variant harmonization may require downstream normalization."] }) }] };
    } catch (error) {
      const trimmedQuery = String(query || "").trim();
      const isRsIdQuery = /^rs\d+$/i.test(trimmedQuery);
      const isGeneLike = /^[A-Z0-9-]{2,}$/.test(trimmedQuery);
      if (isLikelyTransientUpstreamError(error) || isGwasEndpointError(error) || isGwasCooldownActive()) {
        const fallback = isRsIdQuery
          ? null
          : isGeneLike
            ? await buildOpenTargetsGeneGeneticsFallback(trimmedQuery, "", Math.min(limit, 8))
            : await buildOpenTargetsDiseaseGeneticsFallback(trimmedQuery, Math.min(limit, 8));
        if (fallback) {
          const limitations = [
            ...fallback.limitations,
            `Underlying GWAS call error: ${String(error?.message || "unknown error").slice(0, 220)}`,
          ];
          return {
            content: [
              {
                type: "text",
                text: renderStructuredResponse({
                  summary: fallback.summary,
                  keyFields: fallback.keyFields,
                  sources: fallback.sources,
                  limitations,
                }),
              },
            ],
          };
        }
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "CRITICAL GAP: GWAS service unavailable and no proxy fallback could be resolved.",
                keyFields: [`Query: ${trimmedQuery}`],
                sources: [GWAS_API],
                limitations: [
                  `Underlying GWAS call error: ${String(error?.message || "unknown error").slice(0, 220)}`,
                  "Retry later or use gene/disease-level genetics proxy evidence from Open Targets.",
                ],
              }),
            },
          ],
        };
      }
      return { content: [{ type: "text", text: `Error in search_gwas_associations: ${error.message}` }] };
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
// TOOL 27: Expand disease context (ontology)
// ============================================
server.registerTool(
  "expand_disease_context",
  {
    description:
      "Expand a disease query into ontology-aligned terms, IDs, synonyms, and parent concepts using OLS.",
    inputSchema: {
      query: z.string().describe("Disease term to expand (e.g., 'Alzheimer disease', 'IPF')"),
      ontology: z.string().optional().default("efo").describe("Ontology short name for OLS (default: efo)"),
      limit: z.number().optional().default(5).describe("Max ontology candidates to return (1-10)."),
      includeHierarchy: z.boolean().optional().default(true).describe("Whether to fetch parent concepts for top hit."),
    },
  },
  async ({ query, ontology = "efo", limit = 5, includeHierarchy = true }) => {
    try {
      const boundedLimit = Math.max(1, Math.min(10, Math.round(limit)));
      const normalizedOntology = (ontology || "efo").trim().toLowerCase();
      const searchUrl = `${OLS_API}/api/search?${new URLSearchParams({
        q: query,
        ontology: normalizedOntology,
        rows: String(Math.max(8, boundedLimit * 2)),
      }).toString()}`;

      const searchData = await fetchJsonWithRetry(searchUrl);
      const docs = (searchData?.response?.docs || [])
        .filter((doc) => doc?.type === "class")
        .slice(0, boundedLimit);

      if (docs.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "No ontology terms found for disease context expansion.",
                keyFields: [`Query: ${query}`, `Ontology: ${normalizedOntology}`],
                sources: [searchUrl],
                limitations: ["Try a broader disease phrase or a different ontology short name."],
              }),
            },
          ],
          structuredContent: buildExpandDiseaseContextPayload({
            resultStatus: "not_found_or_empty",
            query,
            ontology: normalizedOntology,
            includeHierarchy,
            topTerm: null,
            candidates: [],
            parentConcepts: [],
            notes: ["No ontology class terms matched the query."],
          }),
        };
      }

      const sources = [searchUrl];
      const candidatePayloadRows = docs.map((doc, idx) => {
        const synonyms = [
          ...(doc?.exact_synonyms || []),
          ...(doc?.narrow_synonyms || []),
          ...(doc?.related_synonyms || []),
        ];
        return {
          rank: idx + 1,
          label: String(doc?.label || "Unknown term"),
          obo_id: String(doc?.obo_id || doc?.short_form || "N/A"),
          short_form: String(doc?.short_form || "N/A"),
          description: String((doc?.description?.[0] || "No description").replace(/\s+/g, " ").slice(0, 220)),
          synonyms: dedupeArray(synonyms.map((item) => String(item || "").trim()).filter(Boolean)).slice(0, 8),
          iri: doc?.iri ? String(doc.iri) : undefined,
        };
      });
      const keyFields = docs.map((doc, idx) => {
        const synonyms = [
          ...(doc?.exact_synonyms || []),
          ...(doc?.narrow_synonyms || []),
          ...(doc?.related_synonyms || []),
        ];
        const synonymText = [...new Set(synonyms)].slice(0, 6).join("; ") || "N/A";
        const desc = (doc?.description?.[0] || "No description").replace(/\s+/g, " ").slice(0, 220);
        return `${idx + 1}. ${doc?.label || "Unknown term"} | OBO ID: ${doc?.obo_id || doc?.short_form || "N/A"} | Synonyms: ${synonymText} | Description: ${desc}`;
      });
      const parentConcepts = [];

      if (includeHierarchy && docs[0]?.iri) {
        const detailsUrl = `${OLS_API}/api/ontologies/${encodeURIComponent(normalizedOntology)}/terms?iri=${encodeURIComponent(
          docs[0].iri
        )}`;
        sources.push(detailsUrl);
        const details = await fetchJsonWithRetry(detailsUrl);
        const term = details?._embedded?.terms?.[0];

        const parentHref = term?._links?.hierarchicalParents?.href || term?._links?.parents?.href;
        if (parentHref) {
          const normalizedParentUrl = parentHref.replace(/^http:\/\//i, "https://");
          sources.push(normalizedParentUrl);
          const parentsData = await fetchJsonWithRetry(normalizedParentUrl);
          const parents = (parentsData?._embedded?.terms || [])
            .slice(0, 6)
            .map((p) => `${p?.label || "Unknown"} (${p?.obo_id || p?.short_form || "N/A"})`);
          parentConcepts.push(...parents);
          if (parents.length > 0) {
            keyFields.push(`Top parent concepts for ${docs[0].label}: ${parents.join(", ")}`);
          }
        }
      }
      const topTerm = docs[0]
        ? {
            label: String(docs[0]?.label || "Unknown term"),
            obo_id: String(docs[0]?.obo_id || docs[0]?.short_form || "N/A"),
            short_form: String(docs[0]?.short_form || "N/A"),
          }
        : null;

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Expanded disease context for "${query}" using ontology "${normalizedOntology}".`,
              keyFields,
              sources,
              limitations: [
                "Ontology synonym coverage is strong but not exhaustive across all biomedical vocabularies.",
                "Parent concepts reflect ontology hierarchy, not causal biological relationships.",
              ],
            }),
          },
        ],
        structuredContent: buildExpandDiseaseContextPayload({
          resultStatus: "ok",
          query,
          ontology: normalizedOntology,
          includeHierarchy,
          topTerm,
          candidates: candidatePayloadRows,
          parentConcepts,
          notes: [
            "Candidate terms are sourced from OLS ontology search.",
            includeHierarchy ? "Hierarchy expansion used the top hit parent links when available." : "Hierarchy expansion was disabled by request.",
          ],
        }),
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in expand_disease_context: ${error.message}` }],
        structuredContent: buildExpandDiseaseContextPayload({
          resultStatus: "error",
          query,
          ontology,
          includeHierarchy,
          topTerm: null,
          candidates: [],
          parentConcepts: [],
          notes: ["Unexpected error during ontology context expansion."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 28: ChEMBL compound evidence for target
// ============================================
server.registerTool(
  "search_chembl_compounds_for_target",
  {
    description:
      "Find chemical matter for a target from ChEMBL with potency-centric ranking and assay references.",
    inputSchema: {
      query: z.string().describe("Target query (e.g., LRRK2, EGFR, PCSK9)"),
      organism: z.string().optional().default("Homo sapiens").describe("Preferred organism filter."),
      activityType: z.string().optional().default("IC50").describe("Activity type (e.g., IC50, Ki, EC50)."),
      minPchembl: z.number().optional().default(6.0).describe("Minimum pChEMBL cutoff (default 6.0)."),
      maxNanomolar: z.number().optional().describe("Optional upper cutoff for standard value in nM."),
      limit: z.number().optional().default(10).describe("Max compounds to return (1-20)."),
    },
  },
  async ({
    query,
    organism = "Homo sapiens",
    activityType = "IC50",
    minPchembl = 6.0,
    maxNanomolar,
    limit = 10,
  }) => {
    try {
      const boundedLimit = Math.max(1, Math.min(20, Math.round(limit)));
      const normalizedOrganism = (organism || "").trim().toLowerCase();
      const normalizedQuery = query.trim();
      const normalizedMaxNanomolar = toNullableNumber(maxNanomolar);

      const targetSearchUrl = `${CHEMBL_API}/target/search?${new URLSearchParams({
        q: normalizedQuery,
        format: "json",
      }).toString()}`;
      const targetSearch = await fetchJsonWithRetry(targetSearchUrl);
      const targetCandidates = targetSearch?.targets || [];

      if (targetCandidates.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "No ChEMBL targets matched the query.",
                keyFields: [`Query: ${normalizedQuery}`, `Organism preference: ${organism}`],
                sources: [targetSearchUrl],
                limitations: ["Try alternate gene symbols or target names."],
              }),
            },
          ],
          structuredContent: buildSearchChemblCompoundsPayload({
            resultStatus: "not_found_or_empty",
            query: normalizedQuery,
            organism,
            activityType,
            minPchembl,
            maxNanomolar: normalizedMaxNanomolar,
            selectedTarget: null,
            candidateTargetsConsidered: 0,
            compounds: [],
            topCandidateTargetMatches: [],
            notes: ["No ChEMBL target candidates matched the query."],
          }),
        };
      }

      const rankedCandidates = targetCandidates
        .map((target) => {
          let rankScore = safeNumber(target?.score, 0);
          if ((target?.organism || "").toLowerCase() === normalizedOrganism) {
            rankScore += 100;
          }
          if ((target?.target_type || "").toUpperCase() === "SINGLE PROTEIN") {
            rankScore += 25;
          }
          const symbols = (target?.target_components || [])
            .flatMap((component) => component?.target_component_synonyms || [])
            .map((syn) => String(syn?.component_synonym || "").toUpperCase());
          if (symbols.includes(normalizedQuery.toUpperCase())) {
            rankScore += 15;
          }
          return { target, rankScore };
        })
        .sort((a, b) => b.rankScore - a.rankScore);

      const selected = rankedCandidates[0]?.target;
      const targetChemblId = selected?.target_chembl_id;
      if (!targetChemblId) {
        const message = `ChEMBL target resolution failed for "${query}" (no target_chembl_id found).`;
        return {
          content: [
            {
              type: "text",
              text: message,
            },
          ],
          structuredContent: buildSearchChemblCompoundsPayload({
            resultStatus: "degraded",
            query: normalizedQuery,
            organism,
            activityType,
            minPchembl,
            maxNanomolar: normalizedMaxNanomolar,
            selectedTarget: null,
            candidateTargetsConsidered: targetCandidates.length,
            compounds: [],
            topCandidateTargetMatches: [],
            notes: ["Top candidate lacked target_chembl_id and could not be queried for activity."],
            errorMessage: message,
          }),
        };
      }

      const activityParams = new URLSearchParams({
        target_chembl_id: targetChemblId,
        standard_type: activityType,
        standard_relation: "=",
        pchembl_value__gte: String(minPchembl),
        limit: "200",
        format: "json",
      });
      if (Number.isFinite(safeNumber(maxNanomolar))) {
        activityParams.set("standard_units", "nM");
        activityParams.set("standard_value__lte", String(maxNanomolar));
      }
      const activityUrl = `${CHEMBL_API}/activity.json?${activityParams.toString()}`;
      const activityData = await fetchJsonWithRetry(activityUrl);
      const activities = activityData?.activities || [];

      if (activities.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `Resolved target ${targetChemblId} but no activities matched current filters.`,
                keyFields: [
                  `Selected target: ${selected?.pref_name || "Unknown"} (${targetChemblId})`,
                  `Organism: ${selected?.organism || "Unknown"}`,
                  `Filter: ${activityType}, pChEMBL >= ${minPchembl}${normalizedMaxNanomolar !== null ? `, <= ${normalizedMaxNanomolar} nM` : ""}`,
                ],
                sources: [targetSearchUrl, activityUrl],
                limitations: ["Filters may be too strict; lower pChEMBL threshold or remove nM cutoff."],
              }),
            },
          ],
          structuredContent: buildSearchChemblCompoundsPayload({
            resultStatus: "not_found_or_empty",
            query: normalizedQuery,
            organism,
            activityType,
            minPchembl,
            maxNanomolar: normalizedMaxNanomolar,
            selectedTarget: {
              target_chembl_id: targetChemblId,
              pref_name: String(selected?.pref_name || "Unknown"),
              target_type: String(selected?.target_type || "Unknown"),
              organism: String(selected?.organism || "Unknown"),
            },
            candidateTargetsConsidered: targetCandidates.length,
            compounds: [],
            topCandidateTargetMatches: rankedCandidates
              .slice(0, 3)
              .map((entry) => `${entry.target?.target_chembl_id || "N/A"} (${entry.target?.organism || "Unknown"})`),
            notes: ["Target resolved but no activity rows matched requested potency filters."],
          }),
        };
      }

      const bestByMolecule = new Map();
      const isBetterRecord = (candidate, incumbent) => {
        const candidateP = safeNumber(candidate?.pchembl, Number.NEGATIVE_INFINITY);
        const incumbentP = safeNumber(incumbent?.pchembl, Number.NEGATIVE_INFINITY);
        if (candidateP !== incumbentP) {
          return candidateP > incumbentP;
        }
        const candidateNm = safeNumber(candidate?.standardValueNm, Number.POSITIVE_INFINITY);
        const incumbentNm = safeNumber(incumbent?.standardValueNm, Number.POSITIVE_INFINITY);
        return candidateNm < incumbentNm;
      };

      for (const activity of activities) {
        const moleculeId = activity?.molecule_chembl_id || activity?.parent_molecule_chembl_id;
        if (!moleculeId) continue;
        const record = {
          moleculeId,
          name: activity?.molecule_pref_name || moleculeId,
          standardType: activity?.standard_type || activityType,
          standardValue: activity?.standard_value || "N/A",
          standardUnits: activity?.standard_units || "N/A",
          standardValueNm: safeNumber(activity?.standard_value),
          pchembl: safeNumber(activity?.pchembl_value),
          relation: activity?.standard_relation || "=",
          assayId: activity?.assay_chembl_id || "N/A",
          documentId: activity?.document_chembl_id || "N/A",
          documentYear: activity?.document_year || "N/A",
        };

        const incumbent = bestByMolecule.get(moleculeId);
        if (!incumbent || isBetterRecord(record, incumbent)) {
          bestByMolecule.set(moleculeId, record);
        }
      }

      const rankedCompounds = Array.from(bestByMolecule.values())
        .sort((a, b) => {
          const pDiff = safeNumber(b.pchembl, Number.NEGATIVE_INFINITY) - safeNumber(a.pchembl, Number.NEGATIVE_INFINITY);
          if (pDiff !== 0) return pDiff;
          return safeNumber(a.standardValueNm, Number.POSITIVE_INFINITY) - safeNumber(b.standardValueNm, Number.POSITIVE_INFINITY);
        })
        .slice(0, boundedLimit);

      const keyFields = [
        `Selected target: ${selected?.pref_name || "Unknown"} (${targetChemblId})`,
        `Target type: ${selected?.target_type || "N/A"} | Organism: ${selected?.organism || "Unknown"}`,
        `Candidate targets considered: ${targetCandidates.length}`,
        `Activity filter: ${activityType}, pChEMBL >= ${minPchembl}${normalizedMaxNanomolar !== null ? `, <= ${normalizedMaxNanomolar} nM` : ""}`,
        ...rankedCompounds.map(
          (compound, idx) =>
            `${idx + 1}. ${compound.name} (${compound.moleculeId}) | ${compound.standardType} ${compound.relation} ${compound.standardValue} ${compound.standardUnits} | pChEMBL ${Number.isFinite(compound.pchembl) ? compound.pchembl.toFixed(2) : "N/A"} | Assay ${compound.assayId} | Doc ${compound.documentId} (${compound.documentYear})`
        ),
      ];

      const topCandidateIds = rankedCandidates
        .slice(0, 3)
        .map((entry) => `${entry.target?.target_chembl_id || "N/A"} (${entry.target?.organism || "Unknown"})`);
      keyFields.push(`Top candidate target matches: ${topCandidateIds.join(", ")}`);
      const compoundsPayload = rankedCompounds.map((compound, idx) => ({
        rank: idx + 1,
        molecule_chembl_id: String(compound?.moleculeId || ""),
        name: String(compound?.name || "Unknown"),
        standard_type: String(compound?.standardType || ""),
        relation: String(compound?.relation || "="),
        standard_value: String(compound?.standardValue || "N/A"),
        standard_units: String(compound?.standardUnits || "N/A"),
        standard_value_nm: toNullableNumber(compound?.standardValueNm),
        pchembl: toNullableNumber(compound?.pchembl),
        assay_chembl_id: String(compound?.assayId || "N/A"),
        document_chembl_id: String(compound?.documentId || "N/A"),
        document_year: String(compound?.documentYear || "N/A"),
      }));
      const chemblPayload = buildSearchChemblCompoundsPayload({
        resultStatus: "ok",
        query: normalizedQuery,
        organism,
        activityType,
        minPchembl,
        maxNanomolar: normalizedMaxNanomolar,
        selectedTarget: {
          target_chembl_id: targetChemblId,
          pref_name: String(selected?.pref_name || "Unknown"),
          target_type: String(selected?.target_type || "Unknown"),
          organism: String(selected?.organism || "Unknown"),
        },
        candidateTargetsConsidered: targetCandidates.length,
        compounds: compoundsPayload,
        topCandidateTargetMatches: topCandidateIds,
        notes: ["Compounds are ranked by pChEMBL, with tie-break by lower standard value (nM)."],
      });

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Retrieved ${rankedCompounds.length} ChEMBL compounds for target query "${query}".`,
              keyFields,
              sources: [targetSearchUrl, activityUrl],
              limitations: [
                "Assays vary by format/conditions, so potency values are not directly comparable across all records.",
                "Returned compounds are evidence-backed hits, not a full medicinal chemistry triage.",
              ],
            }),
          },
        ],
        structuredContent: chemblPayload,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in search_chembl_compounds_for_target: ${error.message}` }],
        structuredContent: buildSearchChemblCompoundsPayload({
          resultStatus: "error",
          query,
          organism,
          activityType,
          minPchembl,
          maxNanomolar,
          selectedTarget: null,
          candidateTargetsConsidered: 0,
          compounds: [],
          topCandidateTargetMatches: [],
          notes: ["Unexpected error during ChEMBL compound search."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 29: Target expression and cell context
// ============================================
server.registerTool(
  "summarize_target_expression_context",
  {
    description:
      "Summarize baseline target expression across tissues and cell types using Open Targets expression data.",
    inputSchema: {
      targetId: z.string().optional().describe("Optional Ensembl target ID (e.g., ENSG00000130203)."),
      geneSymbol: z.string().optional().describe("Optional gene symbol fallback (e.g., APOE)."),
      topTissues: z.number().optional().default(12).describe("Max tissues/cell contexts to return (1-25)."),
      anatomicalSystem: z.string().optional().describe("Optional anatomical system filter (e.g., immune system)."),
      includeCellTypes: z.boolean().optional().default(true).describe("Include protein cell-type-level context."),
    },
  },
  async ({ targetId, geneSymbol, topTissues = 12, anatomicalSystem, includeCellTypes = true }) => {
    try {
      const boundedTop = Math.max(1, Math.min(25, Math.round(topTissues)));
      const sources = [];

      let resolvedTargetId = (targetId || "").trim();
      const normalizedGeneSymbol = (geneSymbol || "").trim();
      const symbolFromTargetField =
        resolvedTargetId && !/^ENSG\d{11}$/i.test(resolvedTargetId) ? resolvedTargetId : "";

      if (!resolvedTargetId || !/^ENSG\d{11}$/i.test(resolvedTargetId)) {
        const symbolQuery = normalizedGeneSymbol || symbolFromTargetField;
        if (!symbolQuery) {
          const message = "Provide either `targetId` (ENSG...) or `geneSymbol` to summarize expression context.";
          return {
            content: [
              {
                type: "text",
                text: message,
              },
            ],
            structuredContent: buildTargetExpressionContextPayload({
              resultStatus: "error",
              targetId: "",
              targetSymbol: "",
              targetName: "",
              anatomicalSystemFilter: anatomicalSystem || "",
              includeCellTypes,
              rowsConsidered: 0,
              expressionRows: [],
              dominantSystems: [],
              dominantOrgans: [],
              dominantCellTypes: [],
              notes: ["Either targetId or geneSymbol is required."],
              errorMessage: message,
            }),
          };
        }
        const searchQuery = `
          query SearchTargets($queryString: String!, $size: Int!) {
            search(queryString: $queryString, entityNames: ["target"], page: { size: $size, index: 0 }) {
              hits {
                id
                name
                description
              }
            }
          }
        `;
        const searchResult = await queryOpenTargets(searchQuery, { queryString: symbolQuery, size: 10 });
        const hits = searchResult?.data?.search?.hits || [];
        const exact = hits.find((hit) => String(hit?.name || "").toUpperCase() === symbolQuery.toUpperCase());
        resolvedTargetId = exact?.id || hits[0]?.id || "";
        sources.push(`${OPEN_TARGETS_API}#search:${encodeURIComponent(symbolQuery)}`);
        if (!resolvedTargetId) {
          return {
            content: [
              {
                type: "text",
                text: renderStructuredResponse({
                  summary: `No target match found for symbol "${symbolQuery}".`,
                  keyFields: ["Try Ensembl target ID (ENSG...) or alternate gene symbol."],
                  sources,
                  limitations: ["Open Targets search may require canonical symbol naming."],
                }),
              },
            ],
            structuredContent: buildTargetExpressionContextPayload({
              resultStatus: "not_found_or_empty",
              targetId: "",
              targetSymbol: symbolQuery,
              targetName: "",
              anatomicalSystemFilter: anatomicalSystem || "",
              includeCellTypes,
              rowsConsidered: 0,
              expressionRows: [],
              dominantSystems: [],
              dominantOrgans: [],
              dominantCellTypes: [],
              notes: ["Target symbol could not be resolved in Open Targets search."],
            }),
          };
        }
      }

      const expressionQuery = `
        query TargetExpression($targetId: String!) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            approvedName
            expressions {
              tissue {
                id
                label
                anatomicalSystems
                organs
              }
              rna {
                zscore
                value
                unit
                level
              }
              protein {
                level
                reliability
                cellType {
                  name
                  level
                  reliability
                }
              }
            }
          }
        }
      `;

      const result = await queryOpenTargets(expressionQuery, { targetId: resolvedTargetId });
      const target = result?.data?.target;
      if (!target) {
        return {
          content: [
            {
              type: "text",
              text: `No expression context found for target ID ${resolvedTargetId}.`,
            },
          ],
          structuredContent: buildTargetExpressionContextPayload({
            resultStatus: "not_found_or_empty",
            targetId: resolvedTargetId,
            targetSymbol: normalizedGeneSymbol || symbolFromTargetField,
            targetName: "",
            anatomicalSystemFilter: anatomicalSystem || "",
            includeCellTypes,
            rowsConsidered: 0,
            expressionRows: [],
            dominantSystems: [],
            dominantOrgans: [],
            dominantCellTypes: [],
            notes: ["No expression payload was returned for the resolved target."],
          }),
        };
      }

      const systemFilter = (anatomicalSystem || "").trim().toLowerCase();
      let expressions = target?.expressions || [];
      if (systemFilter) {
        expressions = expressions.filter((entry) =>
          (entry?.tissue?.anatomicalSystems || []).some((system) =>
            String(system || "")
              .toLowerCase()
              .includes(systemFilter)
          )
        );
      }

      if (expressions.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `No expression rows matched filter for ${target.approvedSymbol}.`,
                keyFields: [
                  `Target: ${target.approvedSymbol} (${target.id})`,
                  `Anatomical system filter: ${anatomicalSystem || "none"}`,
                ],
                sources: [...sources, `${OPEN_TARGETS_API}#target:${target.id}`],
                limitations: ["Filter may be too narrow; remove or broaden anatomicalSystem."],
              }),
            },
          ],
          structuredContent: buildTargetExpressionContextPayload({
            resultStatus: "not_found_or_empty",
            targetId: target.id,
            targetSymbol: target.approvedSymbol,
            targetName: target.approvedName,
            anatomicalSystemFilter: anatomicalSystem || "",
            includeCellTypes,
            rowsConsidered: 0,
            expressionRows: [],
            dominantSystems: [],
            dominantOrgans: [],
            dominantCellTypes: [],
            notes: ["No expression rows matched the anatomical system filter."],
          }),
        };
      }

      const scored = expressions
        .map((entry) => {
          const rnaValue = safeNumber(entry?.rna?.value, -1);
          const proteinLevel = safeNumber(entry?.protein?.level, -1);
          const score = (Number.isFinite(rnaValue) ? rnaValue : -1) + (proteinLevel >= 0 ? proteinLevel * 250 : 0);
          return { entry, score, rnaValue, proteinLevel };
        })
        .sort((a, b) => b.score - a.score)
        .slice(0, boundedTop);

      const systemCounts = new Map();
      const organCounts = new Map();
      const cellTypeCounts = new Map();
      const expressionRowsPayload = [];
      const keyFields = [
        `Target: ${target.approvedSymbol} (${target.approvedName})`,
        `Target ID: ${target.id}`,
        `Rows considered: ${expressions.length}`,
      ];
      if (systemFilter) {
        keyFields.push(`Applied anatomical system filter: ${anatomicalSystem}`);
      }

      for (const { entry } of scored) {
        const tissue = entry?.tissue || {};
        const rna = entry?.rna || {};
        const protein = entry?.protein || {};

        for (const system of tissue?.anatomicalSystems || []) {
          incrementCount(systemCounts, system);
        }
        for (const organ of tissue?.organs || []) {
          incrementCount(organCounts, organ);
        }

        const proteinText = `${mapProteinLevel(protein?.level)}${protein?.reliability ? " (reliable)" : ""}`;
        let cellTypeText = "";
        let topCellTypeLabels = [];
        if (includeCellTypes) {
          topCellTypeLabels = (protein?.cellType || []).slice(0, 4).map((cell) => {
            const label = `${cell?.name || "unknown"}:${mapProteinLevel(cell?.level)}`;
            incrementCount(cellTypeCounts, cell?.name || "unknown");
            return label;
          });
          if (topCellTypeLabels.length > 0) {
            cellTypeText = ` | Cell types: ${topCellTypeLabels.join(", ")}`;
          }
        }
        expressionRowsPayload.push({
          rank: expressionRowsPayload.length + 1,
          tissue_label: String(tissue?.label || "Unknown tissue"),
          anatomical_systems: (tissue?.anatomicalSystems || []).map((item) => String(item || "")),
          organs: (tissue?.organs || []).map((item) => String(item || "")),
          rna_value: toNullableNumber(rna?.value),
          rna_unit: String(rna?.unit || ""),
          rna_level: toNullableNumber(rna?.level),
          rna_zscore: toNullableNumber(rna?.zscore),
          protein_level: mapProteinLevel(protein?.level),
          protein_reliable: Boolean(protein?.reliability),
          top_cell_types: topCellTypeLabels,
        });

        keyFields.push(
          `${keyFields.length - (systemFilter ? 3 : 2)}. ${tissue?.label || "Unknown tissue"} | RNA: ${
            Number.isFinite(safeNumber(rna?.value)) ? rna.value : "N/A"
          } ${rna?.unit || ""} (level ${rna?.level ?? "N/A"}, z=${rna?.zscore ?? "N/A"}) | Protein: ${proteinText}${cellTypeText}`
        );
      }

      const topSystems = summarizeTopCounts(systemCounts, 6);
      const topOrgans = summarizeTopCounts(organCounts, 6);
      const topCellTypes = summarizeTopCounts(cellTypeCounts, 6);
      if (topSystems.length > 0) keyFields.push(`Dominant anatomical systems: ${topSystems.join(", ")}`);
      if (topOrgans.length > 0) keyFields.push(`Dominant organs: ${topOrgans.join(", ")}`);
      if (includeCellTypes && topCellTypes.length > 0) {
        keyFields.push(`Dominant cell-type context: ${topCellTypes.join(", ")}`);
      }
      const expressionPayload = buildTargetExpressionContextPayload({
        resultStatus: "ok",
        targetId: target.id,
        targetSymbol: target.approvedSymbol,
        targetName: target.approvedName,
        anatomicalSystemFilter: anatomicalSystem || "",
        includeCellTypes,
        rowsConsidered: expressions.length,
        expressionRows: expressionRowsPayload,
        dominantSystems: topSystems,
        dominantOrgans: topOrgans,
        dominantCellTypes: topCellTypes,
        notes: [
          "Rows are ranked by combined RNA abundance and protein-level heuristic.",
          includeCellTypes ? "Cell-type annotations were included when available." : "Cell-type annotations were excluded by request.",
        ],
      });

      const targetUrl = `https://platform.opentargets.org/target/${target.id}`;
      sources.push(`${OPEN_TARGETS_API}#target.expressions:${target.id}`);
      sources.push(targetUrl);

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Expression and cell-context summary for ${target.approvedSymbol}.`,
              keyFields,
              sources,
              limitations: [
                "Expression rows are baseline context and not necessarily disease-state differential expression.",
                "RNA/protein abundance does not alone establish target efficacy or safety directionality.",
              ],
            }),
          },
        ],
        structuredContent: expressionPayload,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_target_expression_context: ${error.message}` }],
        structuredContent: buildTargetExpressionContextPayload({
          resultStatus: "error",
          targetId: targetId || "",
          targetSymbol: geneSymbol || "",
          targetName: "",
          anatomicalSystemFilter: anatomicalSystem || "",
          includeCellTypes,
          rowsConsidered: 0,
          expressionRows: [],
          dominantSystems: [],
          dominantOrgans: [],
          dominantCellTypes: [],
          notes: ["Unexpected error during target expression-context summarization."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 30: Genetic direction-of-effect summary
// ============================================
server.registerTool(
  "infer_genetic_effect_direction",
  {
    description:
      "Infer direction-of-effect signals (risk-increasing vs protective) for a gene in a disease context using GWAS associations.",
    inputSchema: {
      geneSymbol: z.string().describe("Gene symbol (e.g., TYK2, IL23R, LRRK2)."),
      diseaseQuery: z.string().describe("Disease context phrase used to match GWAS traits (e.g., ulcerative colitis)."),
      pvalueThreshold: z.number().optional().default(5e-8).describe("Maximum p-value to include."),
      maxSnps: z.number().optional().default(8).describe("Max gene-linked SNPs to scan (1-40)."),
      maxAssociations: z.number().optional().default(40).describe("Max matched associations to retain (1-200)."),
      timeBudgetSec: z.number().optional().default(25).describe("Soft runtime budget in seconds for this tool call."),
    },
  },
  async ({ geneSymbol, diseaseQuery, pvalueThreshold = 5e-8, maxSnps = 8, maxAssociations = 40, timeBudgetSec = 25 }) => {
    try {
      const boundedSnps = Math.max(1, Math.min(40, Math.round(maxSnps)));
      const boundedAssociations = Math.max(1, Math.min(200, Math.round(maxAssociations)));
      const boundedThreshold = Math.max(0, safeNumber(pvalueThreshold, 5e-8));
      const boundedBudgetSec = Math.max(8, Math.min(50, Math.round(timeBudgetSec)));
      const deadlineMs = Date.now() + boundedBudgetSec * 1000;
      const normalizedDisease = diseaseQuery.trim().toLowerCase();
      const diseaseTokens = tokenizeQuery(diseaseQuery);
      let timedOutEarly = false;

      const snpSearchUrl = `${GWAS_API}/singleNucleotidePolymorphisms/search/findByGene?geneName=${encodeURIComponent(
        geneSymbol.trim()
      )}&page=0&size=${boundedSnps}`;
      const snpData = await fetchGwasJson(snpSearchUrl, { timeoutMs: 10000, retries: 1 });
      const snps = dedupeArray((snpData?._embedded?.singleNucleotidePolymorphisms || []).map((snp) => snp?.rsId)).slice(
        0,
        boundedSnps
      );
      const sources = [snpSearchUrl];

      if (snps.length === 0) {
        const geneticPayload = buildInferGeneticPayload({
          resultStatus: "not_found_or_empty",
          fallbackMode: "none",
          geneSymbol,
          diseaseQuery,
          pvalueThreshold: boundedThreshold,
          maxSnps: boundedSnps,
          maxAssociations: boundedAssociations,
          timeBudgetSec: boundedBudgetSec,
          snpsScanned: 0,
          associationsScanned: 0,
          matchedAssociations: [],
          timedOutEarly,
          hasMixedSignals: false,
          directionCounts: null,
          notes: ["No GWAS SNP mappings found for the requested gene."],
        });
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `No GWAS SNPs were found for gene ${geneSymbol}.`,
                keyFields: [`Gene: ${geneSymbol}`, `Disease context: ${diseaseQuery}`],
                sources,
                limitations: ["GWAS catalog mappings may miss some loci or use alternate gene mappings."],
              }),
            },
          ],
          structuredContent: geneticPayload,
        };
      }

      const traitCache = new Map();
      const directionCounts = new Map();
      const matched = [];
      let scannedAssociations = 0;
      const maxScanned = Math.max(60, boundedAssociations * 6);

      for (const rsId of snps) {
        if (Date.now() >= deadlineMs) {
          timedOutEarly = true;
          break;
        }
        if (matched.length >= boundedAssociations || scannedAssociations >= maxScanned) break;
        const assocUrl = `${GWAS_API}/associations/search/findByRsId?rsId=${encodeURIComponent(rsId)}`;
        if (sources.length < 12) sources.push(assocUrl);
        const assocData = await fetchGwasJson(assocUrl, { timeoutMs: 9000, retries: 1 });
        const associations = assocData?._embedded?.associations || [];

        for (const association of associations) {
          if (Date.now() >= deadlineMs) {
            timedOutEarly = true;
            break;
          }
          scannedAssociations += 1;
          if (scannedAssociations > maxScanned) break;
          const pvalue = safeNumber(association?.pvalue);
          if (Number.isFinite(pvalue) && pvalue > boundedThreshold) continue;

          let traits = [];
          const traitHrefRaw = association?._links?.efoTraits?.href;
          if (traitHrefRaw) {
            const traitHref = traitHrefRaw.replace(/^http:\/\//i, "https://");
            if (traitCache.has(traitHref)) {
              traits = traitCache.get(traitHref);
            } else {
              const traitData = await fetchGwasJson(traitHref, { timeoutMs: 7000, retries: 0 });
              traits = (traitData?._embedded?.efoTraits || []).map((trait) => ({
                name: String(trait?.trait || "").trim(),
                shortForm: String(trait?.shortForm || "").trim(),
              }));
              traitCache.set(traitHref, traits);
              if (sources.length < 18) sources.push(traitHref);
            }
          }

          const traitNames = dedupeArray(traits.map((trait) => trait.name).filter(Boolean));
          const traitText = traitNames.join(" | ").toLowerCase();
          let diseaseMatch = false;
          if (traitText) {
            diseaseMatch =
              traitText.includes(normalizedDisease) ||
              (diseaseTokens.length > 0 && diseaseTokens.some((token) => traitText.includes(token)));
          }
          if (!diseaseMatch) continue;

          const direction = inferDirectionLabel(association);
          incrementCount(directionCounts, direction);

          const oddsRatio = safeNumber(association?.orPerCopyNum);
          const beta = safeNumber(association?.betaNum);
          const effectText = Number.isFinite(oddsRatio)
            ? `OR=${oddsRatio.toFixed(3)}`
            : Number.isFinite(beta)
              ? `beta=${beta.toFixed(4)}`
              : association?.betaDirection
                ? `betaDirection=${association.betaDirection}`
                : "effect=N/A";
          const associationId =
            association?.associationId ||
            association?.id ||
            String(association?._links?.self?.href || "").split("/").pop() ||
            "N/A";

          matched.push({
            associationId,
            rsId,
            direction,
            pvalue: Number.isFinite(pvalue) ? pvalue : null,
            effectText,
            riskAllele: associationRiskAllele(association),
            traits: traitNames.slice(0, 3),
          });
          if (matched.length >= boundedAssociations) break;
        }
      }

      if (matched.length === 0) {
        const geneticPayload = buildInferGeneticPayload({
          resultStatus: "not_found_or_empty",
          fallbackMode: "none",
          geneSymbol,
          diseaseQuery,
          pvalueThreshold: boundedThreshold,
          maxSnps: boundedSnps,
          maxAssociations: boundedAssociations,
          timeBudgetSec: boundedBudgetSec,
          snpsScanned: snps.length,
          associationsScanned: scannedAssociations,
          matchedAssociations: [],
          timedOutEarly,
          hasMixedSignals: false,
          directionCounts: {
            risk_increasing: directionCounts.get("risk_increasing") || 0,
            protective: directionCounts.get("protective") || 0,
            neutral: directionCounts.get("neutral") || 0,
            unknown: directionCounts.get("unknown") || 0,
          },
          notes: ["No disease-matched GWAS associations passed filtering criteria."],
        });
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `No matched GWAS associations met criteria for ${geneSymbol} in "${diseaseQuery}".`,
                keyFields: [
                  `Gene: ${geneSymbol}`,
                  `Disease context: ${diseaseQuery}`,
                  `SNPs scanned: ${snps.length}`,
                  `Associations scanned: ${scannedAssociations}`,
                  `P-value threshold: ${boundedThreshold}`,
                  `Time budget (sec): ${boundedBudgetSec}`,
                ],
                sources,
                limitations: [
                  "Trait matching is text-based and may miss semantically related trait labels.",
                  timedOutEarly
                    ? "Tool stopped early due to runtime budget; increase timeBudgetSec for deeper scan."
                    : "Try a broader disease query or relax p-value threshold.",
                ],
              }),
            },
          ],
          structuredContent: geneticPayload,
        };
      }

      matched.sort((a, b) => {
        const ap = Number.isFinite(a.pvalue) ? a.pvalue : Number.POSITIVE_INFINITY;
        const bp = Number.isFinite(b.pvalue) ? b.pvalue : Number.POSITIVE_INFINITY;
        return ap - bp;
      });

      const directionSummary = summarizeTopCounts(directionCounts, 5);
      const keyFields = [
        `Gene: ${geneSymbol}`,
        `Disease context: ${diseaseQuery}`,
        `SNPs scanned: ${snps.length}`,
        `Associations scanned: ${scannedAssociations}`,
        `Matched associations (p <= ${boundedThreshold}): ${matched.length}`,
        `Time budget (sec): ${boundedBudgetSec}`,
        `Direction summary: ${directionSummary.join(", ")}`,
        ...matched.slice(0, 12).map((row, idx) => {
          const pLabel = row.pvalue !== null ? row.pvalue.toExponential(2) : "N/A";
          return `${idx + 1}. Assoc ${row.associationId} | ${row.direction} | ${row.effectText} | p=${pLabel} | Risk allele: ${row.riskAllele} | Traits: ${row.traits.join("; ") || "N/A"} | SNP: ${row.rsId}`;
        }),
      ];

      const hasMixedSignals =
        (directionCounts.get("risk_increasing") || 0) > 0 && (directionCounts.get("protective") || 0) > 0;
      if (hasMixedSignals) {
        keyFields.push("Mixed protective and risk-increasing signals detected; context-specific interpretation is required.");
      }
      const matchedAssociationsPayload = matched.slice(0, 40).map((row, idx) => ({
        rank: idx + 1,
        association_id: String(row.associationId || "N/A"),
        rs_id: String(row.rsId || ""),
        direction: String(row.direction || "unknown"),
        pvalue: Number.isFinite(Number(row.pvalue)) ? Number(row.pvalue) : null,
        effect_text: String(row.effectText || "effect=N/A"),
        risk_allele: String(row.riskAllele || "N/A"),
        traits: (row.traits || []).map((trait) => String(trait || "").trim()).filter(Boolean),
      }));
      const geneticPayload = buildInferGeneticPayload({
        resultStatus: "ok",
        fallbackMode: "none",
        geneSymbol,
        diseaseQuery,
        pvalueThreshold: boundedThreshold,
        maxSnps: boundedSnps,
        maxAssociations: boundedAssociations,
        timeBudgetSec: boundedBudgetSec,
        snpsScanned: snps.length,
        associationsScanned: scannedAssociations,
        matchedAssociations: matchedAssociationsPayload,
        timedOutEarly,
        hasMixedSignals,
        directionCounts: {
          risk_increasing: directionCounts.get("risk_increasing") || 0,
          protective: directionCounts.get("protective") || 0,
          neutral: directionCounts.get("neutral") || 0,
          unknown: directionCounts.get("unknown") || 0,
        },
        notes: ["Direction inference is association-sign based and non-causal."],
      });

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Genetic direction-of-effect summary for ${geneSymbol} in "${diseaseQuery}".`,
              keyFields,
              sources,
              limitations: [
                "Direction inference uses OR/beta signs from GWAS associations and is not equivalent to causal perturbation direction.",
                "Trait matching is query-text based and may include nearby phenotypes.",
                "LD, colocalization, and mediation effects are not resolved by this summary.",
                ...(timedOutEarly
                  ? ["Tool stopped early due to runtime budget; increase timeBudgetSec for deeper scan."]
                  : []),
              ],
            }),
          },
        ],
        structuredContent: geneticPayload,
      };
    } catch (error) {
      if (isLikelyTransientUpstreamError(error) || isGwasEndpointError(error) || isGwasCooldownActive()) {
        const fallback = await buildOpenTargetsGeneGeneticsFallback(
          String(geneSymbol || "").trim(),
          String(diseaseQuery || "").trim(),
          8
        );
        if (fallback) {
          const fallbackPayload = buildInferGeneticPayload({
            resultStatus: "degraded",
            fallbackMode: "open_targets_proxy",
            geneSymbol,
            diseaseQuery,
            pvalueThreshold,
            maxSnps,
            maxAssociations,
            timeBudgetSec,
            snpsScanned: 0,
            associationsScanned: 0,
            matchedAssociations: [],
            timedOutEarly: false,
            hasMixedSignals: false,
            directionCounts: null,
            notes: ["Returned Open Targets genetics proxy because GWAS endpoint was unavailable."],
            errorMessage: String(error?.message || "unknown error"),
          });
          return {
            content: [
              {
                type: "text",
                text: renderStructuredResponse({
                  summary: fallback.summary,
                  keyFields: fallback.keyFields,
                  sources: fallback.sources,
                  limitations: [
                    ...fallback.limitations,
                    `Underlying GWAS call error: ${String(error?.message || "unknown error").slice(0, 220)}`,
                  ],
                }),
              },
            ],
            structuredContent: fallbackPayload,
          };
        }
        const degradedPayload = buildInferGeneticPayload({
          resultStatus: "degraded",
          fallbackMode: "critical_gap",
          geneSymbol,
          diseaseQuery,
          pvalueThreshold,
          maxSnps,
          maxAssociations,
          timeBudgetSec,
          snpsScanned: 0,
          associationsScanned: 0,
          matchedAssociations: [],
          timedOutEarly: false,
          hasMixedSignals: false,
          directionCounts: null,
          notes: ["GWAS endpoint unavailable and no Open Targets fallback was available."],
          errorMessage: String(error?.message || "unknown error"),
        });
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `CRITICAL GAP: could not infer genetic direction-of-effect for ${geneSymbol} in "${diseaseQuery}" because GWAS service is unavailable.`,
                keyFields: [`Gene: ${geneSymbol}`, `Disease context: ${diseaseQuery}`],
                sources: [GWAS_API],
                limitations: [
                  `Underlying GWAS call error: ${String(error?.message || "unknown error").slice(0, 220)}`,
                  "Retry later or use Open Targets genetics evidence as a non-directional proxy.",
                ],
              }),
            },
          ],
          structuredContent: degradedPayload,
        };
      }
      return {
        content: [{ type: "text", text: `Error in infer_genetic_effect_direction: ${error.message}` }],
        structuredContent: buildInferGeneticPayload({
          resultStatus: "error",
          fallbackMode: "none",
          geneSymbol,
          diseaseQuery,
          pvalueThreshold,
          maxSnps,
          maxAssociations,
          timeBudgetSec,
          snpsScanned: 0,
          associationsScanned: 0,
          matchedAssociations: [],
          timedOutEarly: false,
          hasMixedSignals: false,
          directionCounts: null,
          notes: ["Unexpected error during genetic direction inference."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 31: Competitive landscape summary
// ============================================
server.registerTool(
  "summarize_target_competitive_landscape",
  {
    description:
      "Summarize target competition from known drug programs: phase distribution, crowded indications, modality mix, and lead assets.",
    inputSchema: {
      targetId: z.string().optional().describe("Optional Ensembl target ID (e.g., ENSG00000146648)."),
      geneSymbol: z.string().optional().describe("Optional gene symbol fallback (e.g., EGFR)."),
      diseaseFilter: z.string().optional().describe("Optional indication filter (e.g., non-small cell lung cancer)."),
      maxRows: z.number().optional().default(120).describe("Maximum known-drug rows to summarize (20-200)."),
      topDiseases: z.number().optional().default(6).describe("Maximum disease buckets to return (1-12)."),
      topMechanisms: z.number().optional().default(6).describe("Maximum mechanism buckets to return (1-12)."),
    },
  },
  async ({ targetId, geneSymbol, diseaseFilter, maxRows = 120, topDiseases = 6, topMechanisms = 6 }) => {
    try {
      const boundedRows = Math.max(20, Math.min(200, Math.round(maxRows)));
      const boundedDiseases = Math.max(1, Math.min(12, Math.round(topDiseases)));
      const boundedMechanisms = Math.max(1, Math.min(12, Math.round(topMechanisms)));
      const resolved = await resolveTargetIdFromInput({ targetId, geneSymbol });
      if (resolved?.error) {
        return {
          content: [{ type: "text", text: resolved.error }],
          structuredContent: buildTargetCompetitiveLandscapePayload({
            resultStatus: "error",
            targetId: String(targetId || ""),
            targetSymbol: String(geneSymbol || ""),
            targetName: "",
            diseaseFilter: String(diseaseFilter || ""),
            rowsAnalyzed: 0,
            catalogUniqueDrugs: 0,
            catalogInteractions: 0,
            catalogUniqueDiseases: 0,
            uniqueDrugsInRows: 0,
            withdrawnInteractions: 0,
            phaseDistribution: [],
            topDiseases: [],
            topMechanisms: [],
            modalityMix: [],
            leadAssets: [],
            notes: ["Target resolution failed before competitive landscape summarization."],
            errorMessage: String(resolved.error || "target resolution failed"),
          }),
        };
      }

      const normalizedFilter = String(diseaseFilter || "").trim().toLowerCase();
      const filterTokens = tokenizeQuery(normalizedFilter);
      const knownDrugsQuery = `
        query TargetCompetition($targetId: String!, $size: Int!, $freeTextQuery: String) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            approvedName
            knownDrugs(size: $size, freeTextQuery: $freeTextQuery) {
              uniqueDrugs
              uniqueDiseases
              count
              rows {
                drug {
                  id
                  name
                  drugType
                  maximumClinicalTrialPhase
                  hasBeenWithdrawn
                }
                disease {
                  id
                  name
                }
                mechanismOfAction
                phase
                status
              }
            }
          }
        }
      `;

      const result = await queryOpenTargets(knownDrugsQuery, {
        targetId: resolved.targetId,
        size: boundedRows,
        freeTextQuery: normalizedFilter || null,
      });
      const target = result?.data?.target;
      if (!target) {
        return {
          content: [{ type: "text", text: `No target data found for ID ${resolved.targetId}.` }],
          structuredContent: buildTargetCompetitiveLandscapePayload({
            resultStatus: "not_found_or_empty",
            targetId: resolved.targetId,
            targetSymbol: resolved.searchHint || "",
            targetName: "",
            diseaseFilter: String(diseaseFilter || ""),
            rowsAnalyzed: 0,
            catalogUniqueDrugs: 0,
            catalogInteractions: 0,
            catalogUniqueDiseases: 0,
            uniqueDrugsInRows: 0,
            withdrawnInteractions: 0,
            phaseDistribution: [],
            topDiseases: [],
            topMechanisms: [],
            modalityMix: [],
            leadAssets: [],
            notes: ["Target snapshot returned no known-drugs payload."],
          }),
        };
      }

      let rows = target?.knownDrugs?.rows || [];
      if (normalizedFilter) {
        rows = rows.filter((row) => {
          const diseaseName = String(row?.disease?.name || "").toLowerCase();
          const moa = String(row?.mechanismOfAction || "").toLowerCase();
          if (diseaseName.includes(normalizedFilter) || moa.includes(normalizedFilter)) return true;
          if (filterTokens.length === 0) return false;
          return filterTokens.every((token) => diseaseName.includes(token) || moa.includes(token));
        });
      }

      const baseSources = dedupeArray([
        ...(resolved.sourceHints || []),
        `${OPEN_TARGETS_API}#target.knownDrugs:${target.id}`,
        `https://platform.opentargets.org/target/${target.id}/known-drugs`,
      ]);

      if (rows.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `No known-drug records matched for ${target.approvedSymbol}${normalizedFilter ? ` with filter "${diseaseFilter}"` : ""}.`,
                keyFields: [
                  `Target: ${target.approvedSymbol} (${target.id})`,
                  `Catalog totals (unfiltered): ${target?.knownDrugs?.uniqueDrugs || 0} unique drugs across ${
                    target?.knownDrugs?.count || 0
                  } interactions`,
                  `Rows inspected: ${boundedRows}`,
                ],
                sources: baseSources,
                limitations: [
                  "Filter may be too specific; broaden diseaseFilter or remove it.",
                  "Known-drug coverage depends on Open Targets curation and may not include all private/preclinical assets.",
                ],
              }),
            },
          ],
          structuredContent: buildTargetCompetitiveLandscapePayload({
            resultStatus: "not_found_or_empty",
            targetId: target.id,
            targetSymbol: target.approvedSymbol,
            targetName: target.approvedName,
            diseaseFilter: String(diseaseFilter || ""),
            rowsAnalyzed: 0,
            catalogUniqueDrugs: toNonNegativeInt(target?.knownDrugs?.uniqueDrugs),
            catalogInteractions: toNonNegativeInt(target?.knownDrugs?.count),
            catalogUniqueDiseases: toNonNegativeInt(target?.knownDrugs?.uniqueDiseases),
            uniqueDrugsInRows: 0,
            withdrawnInteractions: 0,
            phaseDistribution: [],
            topDiseases: [],
            topMechanisms: [],
            modalityMix: [],
            leadAssets: [],
            notes: ["No known-drug rows matched the disease filter."],
          }),
        };
      }

      const phaseCounts = new Map();
      const diseaseCounts = new Map();
      const mechanismCounts = new Map();
      const modalityCounts = new Map();
      let withdrawnInteractions = 0;

      const bestByDrug = new Map();
      for (const row of rows) {
        const diseaseName = row?.disease?.name || "Unspecified disease";
        const mechanism = row?.mechanismOfAction || "Unknown mechanism";
        const drugType = row?.drug?.drugType || "Unknown";
        const phaseNumeric = safeNumber(row?.phase, safeNumber(row?.drug?.maximumClinicalTrialPhase, Number.NaN));
        const phaseLabel = formatClinicalPhaseLabel(phaseNumeric);

        incrementCount(diseaseCounts, diseaseName);
        incrementCount(mechanismCounts, mechanism);
        incrementCount(modalityCounts, drugType);
        incrementCount(phaseCounts, phaseLabel);
        if (row?.drug?.hasBeenWithdrawn) withdrawnInteractions += 1;

        const drugId = row?.drug?.id || row?.drug?.name;
        if (!drugId) continue;

        const candidate = {
          drugName: row?.drug?.name || drugId,
          drugId,
          diseaseName,
          mechanism,
          phaseNumeric: Number.isFinite(phaseNumeric) ? phaseNumeric : -1,
          phaseLabel,
          withdrawn: Boolean(row?.drug?.hasBeenWithdrawn),
        };
        const incumbent = bestByDrug.get(drugId);
        if (!incumbent || candidate.phaseNumeric > incumbent.phaseNumeric) {
          bestByDrug.set(drugId, candidate);
        }
      }

      const leadAssets = Array.from(bestByDrug.values())
        .sort((a, b) => b.phaseNumeric - a.phaseNumeric)
        .slice(0, 8);

      const phaseSummary = summarizeTopCounts(phaseCounts, 6);
      const topDiseaseSummary = summarizeTopCounts(diseaseCounts, boundedDiseases);
      const topMechanismSummary = summarizeTopCounts(mechanismCounts, boundedMechanisms);
      const modalitySummary = summarizeTopCounts(modalityCounts, 5);
      const uniqueDrugsInRows = bestByDrug.size;
      const approvedOrLatePhase = leadAssets.filter((asset) => asset.phaseNumeric >= 3).length;
      const leadAssetsPayload = leadAssets.map((asset, idx) => ({
        rank: idx + 1,
        drug_name: String(asset?.drugName || "Unknown"),
        drug_id: String(asset?.drugId || ""),
        phase_label: String(asset?.phaseLabel || "Unknown"),
        phase_numeric: toNullableNumber(asset?.phaseNumeric),
        withdrawn: Boolean(asset?.withdrawn),
        disease_name: String(asset?.diseaseName || "Unspecified disease"),
        mechanism: String(asset?.mechanism || "Unknown mechanism"),
      }));
      const competitivePayload = buildTargetCompetitiveLandscapePayload({
        resultStatus: "ok",
        targetId: target.id,
        targetSymbol: target.approvedSymbol,
        targetName: target.approvedName,
        diseaseFilter: String(diseaseFilter || ""),
        rowsAnalyzed: rows.length,
        catalogUniqueDrugs: toNonNegativeInt(target?.knownDrugs?.uniqueDrugs),
        catalogInteractions: toNonNegativeInt(target?.knownDrugs?.count),
        catalogUniqueDiseases: toNonNegativeInt(target?.knownDrugs?.uniqueDiseases),
        uniqueDrugsInRows,
        withdrawnInteractions,
        phaseDistribution: phaseSummary,
        topDiseases: topDiseaseSummary,
        topMechanisms: topMechanismSummary,
        modalityMix: modalitySummary,
        leadAssets: leadAssetsPayload,
        notes: [
          "Known-drug rows are aggregated from Open Targets target.knownDrugs links.",
          normalizedFilter ? "Disease filter was applied to disease and mechanism text." : "No disease filter was applied.",
        ],
      });

      const keyFields = [
        `Target: ${target.approvedSymbol} (${target.approvedName})`,
        `Target ID: ${target.id}`,
        `Rows analyzed: ${rows.length}`,
        `Known-drug totals (Open Targets): ${target?.knownDrugs?.uniqueDrugs || 0} unique drugs, ${
          target?.knownDrugs?.count || 0
        } interactions, ${target?.knownDrugs?.uniqueDiseases || 0} diseases`,
        `Unique drugs in analyzed rows: ${uniqueDrugsInRows}`,
        normalizedFilter ? `Applied disease filter: ${diseaseFilter}` : "Applied disease filter: none",
        `Phase distribution: ${phaseSummary.join(", ") || "N/A"}`,
        `Late-stage density (phase >= III among top assets): ${approvedOrLatePhase}/${leadAssets.length}`,
        `Withdrawn interactions in analyzed rows: ${withdrawnInteractions}`,
        `Dominant indications: ${topDiseaseSummary.join(", ") || "N/A"}`,
        `Dominant mechanisms: ${topMechanismSummary.join(", ") || "N/A"}`,
        `Modality mix: ${modalitySummary.join(", ") || "N/A"}`,
        ...leadAssets.map(
          (asset, idx) =>
            `${idx + 1}. ${asset.drugName} (${asset.drugId}) | ${asset.phaseLabel}${
              asset.withdrawn ? " [WITHDRAWN]" : ""
            } | ${asset.diseaseName} | MOA: ${asset.mechanism}`
        ),
      ];

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Competitive landscape summary for ${target.approvedSymbol}.`,
              keyFields,
              sources: baseSources,
              limitations: [
                "Known-drug rows are evidence links and do not represent full sponsor-level pipeline intelligence.",
                "A crowded landscape can still contain white-space opportunities by indication, modality, biomarker segment, or dosing strategy.",
              ],
            }),
          },
        ],
        structuredContent: competitivePayload,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_target_competitive_landscape: ${error.message}` }],
        structuredContent: buildTargetCompetitiveLandscapePayload({
          resultStatus: "error",
          targetId: String(targetId || ""),
          targetSymbol: String(geneSymbol || ""),
          targetName: "",
          diseaseFilter: String(diseaseFilter || ""),
          rowsAnalyzed: 0,
          catalogUniqueDrugs: 0,
          catalogInteractions: 0,
          catalogUniqueDiseases: 0,
          uniqueDrugsInRows: 0,
          withdrawnInteractions: 0,
          phaseDistribution: [],
          topDiseases: [],
          topMechanisms: [],
          modalityMix: [],
          leadAssets: [],
          notes: ["Unexpected error during competitive landscape summarization."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 32: Target safety liabilities summary
// ============================================
server.registerTool(
  "summarize_target_safety_liabilities",
  {
    description:
      "Summarize target-linked safety liabilities from Open Targets with event, direction, tissue context, and study-type patterns.",
    inputSchema: {
      targetId: z.string().optional().describe("Optional Ensembl target ID (e.g., ENSG00000146648)."),
      geneSymbol: z.string().optional().describe("Optional gene symbol fallback (e.g., EGFR)."),
      topEvents: z.number().optional().default(8).describe("Maximum event groups to return (1-15)."),
      includeClinicalOnly: z.boolean().optional().default(false).describe("Keep only liabilities with at least one clinical study."),
      eventFilter: z.string().optional().describe("Optional event keyword filter (e.g., rash, cardiotoxicity)."),
    },
  },
  async ({ targetId, geneSymbol, topEvents = 8, includeClinicalOnly = false, eventFilter }) => {
    try {
      const boundedTopEvents = Math.max(1, Math.min(15, Math.round(topEvents)));
      const resolved = await resolveTargetIdFromInput({ targetId, geneSymbol });
      if (resolved?.error) {
        return {
          content: [{ type: "text", text: resolved.error }],
          structuredContent: buildTargetSafetyLiabilitiesPayload({
            resultStatus: "error",
            targetId: String(targetId || ""),
            targetSymbol: String(geneSymbol || ""),
            targetName: "",
            includeClinicalOnly,
            eventFilter: String(eventFilter || ""),
            liabilitiesAnalyzed: 0,
            uniqueEvents: 0,
            directionPattern: [],
            studyTypeMix: [],
            tissueContexts: [],
            datasourceMix: [],
            events: [],
            notes: ["Target resolution failed before safety-liability summarization."],
            errorMessage: String(resolved.error || "target resolution failed"),
          }),
        };
      }

      const normalizedEventFilter = String(eventFilter || "").trim().toLowerCase();
      const eventTokens = tokenizeQuery(normalizedEventFilter);
      const safetyQuery = `
        query TargetSafety($targetId: String!) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            approvedName
            safetyLiabilities {
              event
              eventId
              datasource
              url
              literature
              effects {
                direction
                dosing
              }
              biosamples {
                tissueLabel
                cellLabel
              }
              studies {
                name
                type
                description
              }
            }
          }
        }
      `;

      const result = await queryOpenTargets(safetyQuery, { targetId: resolved.targetId });
      const target = result?.data?.target;
      if (!target) {
        return {
          content: [{ type: "text", text: `No target data found for ID ${resolved.targetId}.` }],
          structuredContent: buildTargetSafetyLiabilitiesPayload({
            resultStatus: "not_found_or_empty",
            targetId: resolved.targetId,
            targetSymbol: resolved.searchHint || "",
            targetName: "",
            includeClinicalOnly,
            eventFilter: String(eventFilter || ""),
            liabilitiesAnalyzed: 0,
            uniqueEvents: 0,
            directionPattern: [],
            studyTypeMix: [],
            tissueContexts: [],
            datasourceMix: [],
            events: [],
            notes: ["Target snapshot returned no safety-liabilities payload."],
          }),
        };
      }

      let liabilities = target?.safetyLiabilities || [];
      if (includeClinicalOnly) {
        liabilities = liabilities.filter((liability) =>
          (liability?.studies || []).some((study) => String(study?.type || "").toLowerCase() === "clinical")
        );
      }
      if (normalizedEventFilter) {
        liabilities = liabilities.filter((liability) => {
          const combined = `${liability?.event || ""} ${liability?.datasource || ""}`.toLowerCase();
          if (combined.includes(normalizedEventFilter)) return true;
          if (eventTokens.length === 0) return false;
          return eventTokens.every((token) => combined.includes(token));
        });
      }

      const baseSources = dedupeArray([
        ...(resolved.sourceHints || []),
        `${OPEN_TARGETS_API}#target.safetyLiabilities:${target.id}`,
        `https://platform.opentargets.org/target/${target.id}/safety`,
      ]);

      if (liabilities.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `No safety liability rows matched for ${target.approvedSymbol}.`,
                keyFields: [
                  `Target: ${target.approvedSymbol} (${target.id})`,
                  `Applied clinical-only filter: ${includeClinicalOnly ? "yes" : "no"}`,
                  `Applied event filter: ${eventFilter || "none"}`,
                ],
                sources: baseSources,
                limitations: [
                  "Absence of listed liabilities is not evidence of safety; it can reflect sparse evidence or curation lag.",
                ],
              }),
            },
          ],
          structuredContent: buildTargetSafetyLiabilitiesPayload({
            resultStatus: "not_found_or_empty",
            targetId: target.id,
            targetSymbol: target.approvedSymbol,
            targetName: target.approvedName,
            includeClinicalOnly,
            eventFilter: String(eventFilter || ""),
            liabilitiesAnalyzed: 0,
            uniqueEvents: 0,
            directionPattern: [],
            studyTypeMix: [],
            tissueContexts: [],
            datasourceMix: [],
            events: [],
            notes: ["No safety liability rows matched current filters."],
          }),
        };
      }

      const directionCounts = new Map();
      const tissueCounts = new Map();
      const studyTypeCounts = new Map();
      const datasourceCounts = new Map();
      const eventMap = new Map();
      const eventSourceLinks = [];

      for (const liability of liabilities) {
        const eventName = String(liability?.event || "Unspecified event").trim();
        if (!eventMap.has(eventName)) {
          eventMap.set(eventName, {
            count: 0,
            directions: new Set(),
            tissues: new Set(),
            studyTypes: new Set(),
            datasources: new Set(),
            dosingSignals: new Set(),
          });
        }
        const eventRecord = eventMap.get(eventName);
        eventRecord.count += 1;

        const datasource = String(liability?.datasource || "Unknown datasource").trim();
        eventRecord.datasources.add(datasource);
        incrementCount(datasourceCounts, datasource);

        if (liability?.url) eventSourceLinks.push(String(liability.url).trim());
        if (liability?.literature) eventSourceLinks.push(String(liability.literature).trim());

        for (const effect of liability?.effects || []) {
          const direction = String(effect?.direction || "unknown").trim();
          const dosing = String(effect?.dosing || "").trim();
          eventRecord.directions.add(direction || "unknown");
          incrementCount(directionCounts, direction || "unknown");
          if (dosing) eventRecord.dosingSignals.add(dosing);
        }

        for (const biosample of liability?.biosamples || []) {
          const tissueLabel = String(biosample?.tissueLabel || "").trim();
          const cellLabel = String(biosample?.cellLabel || "").trim();
          const context = tissueLabel && cellLabel ? `${tissueLabel} / ${cellLabel}` : tissueLabel || cellLabel || "Unspecified tissue";
          eventRecord.tissues.add(context);
          incrementCount(tissueCounts, context);
        }

        for (const study of liability?.studies || []) {
          const type = String(study?.type || "unknown").trim();
          eventRecord.studyTypes.add(type || "unknown");
          incrementCount(studyTypeCounts, type || "unknown");
        }
      }

      const rankedEvents = Array.from(eventMap.entries())
        .sort((a, b) => b[1].count - a[1].count)
        .slice(0, boundedTopEvents);
      const directionSummary = summarizeTopCounts(directionCounts, 6);
      const tissueSummary = summarizeTopCounts(tissueCounts, 6);
      const studyTypeSummary = summarizeTopCounts(studyTypeCounts, 6);
      const datasourceSummary = summarizeTopCounts(datasourceCounts, 6);
      const dedupedEventSources = dedupeArray(eventSourceLinks).slice(0, 12);
      const rankedEventsPayload = rankedEvents.map(([eventName, data], idx) => ({
        rank: idx + 1,
        event_name: String(eventName || "Unspecified event"),
        count: toNonNegativeInt(data?.count),
        directions: Array.from(data?.directions || []).map((item) => String(item || "").trim()),
        study_types: Array.from(data?.studyTypes || []).map((item) => String(item || "").trim()),
        tissues: Array.from(data?.tissues || []).map((item) => String(item || "").trim()),
        dosing_signals: Array.from(data?.dosingSignals || []).map((item) => String(item || "").trim()),
        datasources: Array.from(data?.datasources || []).map((item) => String(item || "").trim()),
      }));
      const safetyPayload = buildTargetSafetyLiabilitiesPayload({
        resultStatus: "ok",
        targetId: target.id,
        targetSymbol: target.approvedSymbol,
        targetName: target.approvedName,
        includeClinicalOnly,
        eventFilter: String(eventFilter || ""),
        liabilitiesAnalyzed: liabilities.length,
        uniqueEvents: eventMap.size,
        directionPattern: directionSummary,
        studyTypeMix: studyTypeSummary,
        tissueContexts: tissueSummary,
        datasourceMix: datasourceSummary,
        events: rankedEventsPayload,
        notes: [
          "Safety entries are heterogeneous literature and study-derived liabilities.",
          includeClinicalOnly ? "Liabilities were restricted to rows with at least one clinical study." : "Clinical and preclinical rows were included.",
        ],
      });

      const keyFields = [
        `Target: ${target.approvedSymbol} (${target.approvedName})`,
        `Target ID: ${target.id}`,
        `Liability rows analyzed: ${liabilities.length}`,
        `Unique events: ${eventMap.size}`,
        `Applied clinical-only filter: ${includeClinicalOnly ? "yes" : "no"}`,
        `Applied event filter: ${eventFilter || "none"}`,
        `Direction pattern: ${directionSummary.join(", ") || "N/A"}`,
        `Study-type mix: ${studyTypeSummary.join(", ") || "N/A"}`,
        `Frequent tissue/cell contexts: ${tissueSummary.join(", ") || "N/A"}`,
        `Dominant data sources: ${datasourceSummary.join(", ") || "N/A"}`,
        ...rankedEvents.map(([eventName, data], idx) => {
          const directions = Array.from(data.directions).slice(0, 3).join("; ") || "N/A";
          const tissues = Array.from(data.tissues).slice(0, 2).join("; ") || "N/A";
          const studyTypes = Array.from(data.studyTypes).slice(0, 3).join("; ") || "N/A";
          const dosingSignals = Array.from(data.dosingSignals).slice(0, 2).join("; ") || "N/A";
          return `${idx + 1}. ${eventName} (${data.count}) | Directions: ${directions} | Study types: ${studyTypes} | Tissues: ${tissues} | Dosing: ${dosingSignals}`;
        }),
      ];

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Safety-liability summary for ${target.approvedSymbol}.`,
              keyFields,
              sources: dedupeArray([...baseSources, ...dedupedEventSources]),
              limitations: [
                "Safety-liability entries are heterogeneous literature and study signals, not quantitative incidence estimates.",
                "Event direction and translational relevance depend on modality, dose, tissue exposure, and patient context.",
              ],
            }),
          },
        ],
        structuredContent: safetyPayload,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_target_safety_liabilities: ${error.message}` }],
        structuredContent: buildTargetSafetyLiabilitiesPayload({
          resultStatus: "error",
          targetId: String(targetId || ""),
          targetSymbol: String(geneSymbol || ""),
          targetName: "",
          includeClinicalOnly,
          eventFilter: String(eventFilter || ""),
          liabilitiesAnalyzed: 0,
          uniqueEvents: 0,
          directionPattern: [],
          studyTypeMix: [],
          tissueContexts: [],
          datasourceMix: [],
          events: [],
          notes: ["Unexpected error during safety-liability summarization."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

// ============================================
// TOOL 33: Multi-axis target comparison
// ============================================
server.registerTool(
  "compare_targets_multi_axis",
  {
    description:
      "Compare and rank multiple targets in one disease context using transparent weighted axes: disease association, druggability, clinical maturity, competitive whitespace, and safety.",
    inputSchema: {
      targets: z
        .array(z.string())
        .min(2)
        .max(6)
        .describe("Target identifiers (gene symbols or ENSG IDs), e.g., ['TYK2', 'JAK1', 'IL23R']."),
      diseaseQuery: z.string().optional().describe("Disease phrase to resolve context (e.g., ulcerative colitis)."),
      diseaseId: z.string().optional().describe("Optional Open Targets disease ID (e.g., EFO_0000729)."),
      goal: z
        .string()
        .optional()
        .describe("Optional plain-language optimization goal (e.g., 'safest path', 'most novel', 'de-risked')."),
      strategy: z
        .enum(["balanced", "first_in_class", "de_risked", "safety_first"])
        .optional()
        .default("balanced")
        .describe("Weighting strategy for the composite score."),
      weightDiseaseAssociation: z.number().optional().describe("Optional raw weight for disease-association axis."),
      weightDruggability: z.number().optional().describe("Optional raw weight for druggability axis."),
      weightClinicalMaturity: z.number().optional().describe("Optional raw weight for clinical-maturity axis."),
      weightCompetitiveWhitespace: z.number().optional().describe("Optional raw weight for competitive-white-space axis."),
      weightSafety: z.number().optional().describe("Optional raw weight for safety axis."),
      autoSelectStrategy: z
        .boolean()
        .optional()
        .default(true)
        .describe("If true, infer strategy from `goal` text when explicit strategy is absent or balanced."),
      maxKnownDrugRows: z.number().optional().default(80).describe("Known-drug rows to inspect per target (20-200)."),
      includeAxisBreakdown: z.boolean().optional().default(true).describe("Include per-target axis breakdown details."),
    },
  },
  async ({
    targets,
    diseaseQuery,
    diseaseId,
    goal,
    strategy = "balanced",
    weightDiseaseAssociation,
    weightDruggability,
    weightClinicalMaturity,
    weightCompetitiveWhitespace,
    weightSafety,
    autoSelectStrategy = true,
    maxKnownDrugRows = 80,
    includeAxisBreakdown = true,
  }) => {
    try {
      const boundedRows = Math.max(20, Math.min(200, Math.round(maxKnownDrugRows)));
      const boundedTargets = dedupeArray((targets || []).map((entry) => String(entry || "").trim())).slice(0, 6);
      const includeBreakdown = Boolean(includeAxisBreakdown);
      if (boundedTargets.length < 2) {
        return {
          content: [{ type: "text", text: "Provide at least two distinct targets to compare." }],
          structuredContent: buildCompareTargetsPayload({
            resultStatus: "error",
            targetsRequested: boundedTargets,
            targetsResolved: 0,
            targetsCompared: 0,
            unresolvedTargets: [],
            diseaseId: String(diseaseId || "unknown"),
            diseaseName: String(diseaseQuery || "unknown"),
            strategyRequested: String(strategy || "balanced").trim().toLowerCase(),
            strategyEffective: String(strategy || "balanced").trim().toLowerCase(),
            weightMode: "preset",
            goalText: String(goal || ""),
            notes: ["At least two distinct targets are required for comparison."],
            errorMessage: "Provide at least two distinct targets to compare.",
          }),
        };
      }

      const resolvedDisease = await resolveDiseaseFromInput({ diseaseId, diseaseQuery });
      if (resolvedDisease?.error) {
        return {
          content: [{ type: "text", text: resolvedDisease.error }],
          structuredContent: buildCompareTargetsPayload({
            resultStatus: "error",
            targetsRequested: boundedTargets,
            targetsResolved: 0,
            targetsCompared: 0,
            unresolvedTargets: [String(resolvedDisease.error || "Disease resolution failed.")],
            diseaseId: String(diseaseId || "unknown"),
            diseaseName: String(diseaseQuery || "unknown"),
            strategyRequested: String(strategy || "balanced").trim().toLowerCase(),
            strategyEffective: String(strategy || "balanced").trim().toLowerCase(),
            weightMode: "preset",
            goalText: String(goal || ""),
            notes: ["Disease context resolution failed before ranking."],
            errorMessage: String(resolvedDisease.error || "Disease resolution failed."),
          }),
        };
      }

      const resolvedTargetEntries = [];
      const unresolvedTargets = [];
      const sources = [...(resolvedDisease.sourceHints || [])];
      for (const targetInput of boundedTargets) {
        const resolvedTarget = await resolveTargetIdFromInput({ targetId: targetInput, geneSymbol: targetInput });
        if (resolvedTarget?.error) {
          unresolvedTargets.push(`${targetInput}: ${resolvedTarget.error}`);
          continue;
        }
        resolvedTargetEntries.push({
          input: targetInput,
          targetId: resolvedTarget.targetId,
          searchHint: resolvedTarget.searchHint || targetInput,
        });
        sources.push(...(resolvedTarget.sourceHints || []));
      }

      const byTargetId = new Map();
      for (const entry of resolvedTargetEntries) {
        if (!byTargetId.has(entry.targetId)) byTargetId.set(entry.targetId, entry);
      }
      const uniqueTargets = Array.from(byTargetId.values());
      if (uniqueTargets.length < 2) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "Unable to resolve enough distinct targets for comparison.",
                keyFields: [
                  `Targets requested: ${boundedTargets.join(", ")}`,
                  `Distinct targets resolved: ${uniqueTargets.length}`,
                  `Unresolved targets: ${unresolvedTargets.length > 0 ? unresolvedTargets.join(" | ") : "none"}`,
                ],
                sources: dedupeArray(sources),
                limitations: ["At least two resolvable targets are required for ranking."],
              }),
            },
          ],
          structuredContent: buildCompareTargetsPayload({
            resultStatus: "not_found_or_empty",
            targetsRequested: boundedTargets,
            targetsResolved: uniqueTargets.length,
            targetsCompared: 0,
            unresolvedTargets,
            diseaseId: String(resolvedDisease?.diseaseId || diseaseId || "unknown"),
            diseaseName: String(resolvedDisease?.diseaseName || diseaseQuery || "unknown"),
            strategyRequested: String(strategy || "balanced").trim().toLowerCase(),
            strategyEffective: String(strategy || "balanced").trim().toLowerCase(),
            weightMode: "preset",
            goalText: String(goal || ""),
            notes: ["Insufficient distinct target resolution for ranking."],
          }),
        };
      }

      const diseaseAssociationQuery = `
        query DiseaseAssociations($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            id
            name
            associatedTargets(page: { size: $size, index: 0 }) {
              count
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
      `;
      const diseaseAssociationResult = await queryOpenTargets(diseaseAssociationQuery, {
        diseaseId: resolvedDisease.diseaseId,
        size: 600,
      });
      const disease = diseaseAssociationResult?.data?.disease;
      if (!disease?.id) {
        return {
          content: [{ type: "text", text: `Could not load disease context for ${resolvedDisease.diseaseId}.` }],
          structuredContent: buildCompareTargetsPayload({
            resultStatus: "degraded",
            targetsRequested: boundedTargets,
            targetsResolved: uniqueTargets.length,
            targetsCompared: 0,
            unresolvedTargets,
            diseaseId: String(resolvedDisease?.diseaseId || "unknown"),
            diseaseName: String(resolvedDisease?.diseaseName || diseaseQuery || "unknown"),
            strategyRequested: String(strategy || "balanced").trim().toLowerCase(),
            strategyEffective: String(strategy || "balanced").trim().toLowerCase(),
            weightMode: "preset",
            goalText: String(goal || ""),
            notes: ["Disease context load failed in Open Targets query."],
            errorMessage: `Could not load disease context for ${resolvedDisease.diseaseId}.`,
          }),
        };
      }
      sources.push(`${OPEN_TARGETS_API}#disease.associatedTargets:${disease.id}`);

      const associationByTarget = new Map();
      for (const row of disease?.associatedTargets?.rows || []) {
        const targetIdValue = row?.target?.id;
        if (!targetIdValue) continue;
        associationByTarget.set(targetIdValue, clamp01(row?.score));
      }

      const targetSnapshotQuery = `
        query TargetSnapshot($targetId: String!, $size: Int!, $freeTextQuery: String) {
          target(ensemblId: $targetId) {
            id
            approvedSymbol
            approvedName
            biotype
            tractability {
              modality
              value
              label
            }
            knownDrugs(size: $size, freeTextQuery: $freeTextQuery) {
              uniqueDrugs
              uniqueDiseases
              count
              rows {
                drug {
                  id
                  name
                  drugType
                  maximumClinicalTrialPhase
                  hasBeenWithdrawn
                }
                disease {
                  id
                  name
                }
                mechanismOfAction
                phase
              }
            }
            safetyLiabilities {
              event
              studies {
                type
              }
            }
          }
        }
      `;

      const diseaseFilterText = String(diseaseQuery || disease?.name || "").trim();
      const targetProfiles = [];
      for (const entry of uniqueTargets) {
        const snapshotResult = await queryOpenTargets(targetSnapshotQuery, {
          targetId: entry.targetId,
          size: boundedRows,
          freeTextQuery: diseaseFilterText || null,
        });
        const target = snapshotResult?.data?.target;
        if (!target?.id) {
          unresolvedTargets.push(`${entry.input}: no target snapshot available.`);
          continue;
        }

        const knownDrugs = target?.knownDrugs || {};
        const knownRows = knownDrugs?.rows || [];
        const positiveTractabilityCount = (target?.tractability || []).filter((item) => item?.value === true).length;
        const uniqueDrugs = Math.max(0, safeNumber(knownDrugs?.uniqueDrugs, 0));
        const mechanismCounts = new Map();
        let withdrawnDrugRows = 0;
        let maxPhase = 0;
        for (const row of knownRows) {
          const phase = safeNumber(row?.phase, safeNumber(row?.drug?.maximumClinicalTrialPhase, 0));
          if (Number.isFinite(phase)) maxPhase = Math.max(maxPhase, phase);
          const mechanism = String(row?.mechanismOfAction || "").trim() || "Unknown mechanism";
          incrementCount(mechanismCounts, mechanism);
          if (row?.drug?.hasBeenWithdrawn) withdrawnDrugRows += 1;
        }
        const topMechanisms = summarizeTopCounts(mechanismCounts, 3);

        const safetyRows = target?.safetyLiabilities || [];
        const clinicalSafetyRows = safetyRows.filter((liability) =>
          (liability?.studies || []).some((study) => String(study?.type || "").toLowerCase() === "clinical")
        ).length;

        const diseaseAssociationScore = clamp01(associationByTarget.get(target.id) || 0);
        const druggabilityScore = clamp01(
          (Math.min(positiveTractabilityCount, 6) / 6) * 0.65 + (Math.min(uniqueDrugs, 25) / 25) * 0.35
        );
        const clinicalMaturityScore = clamp01(maxPhase / 4);
        const competitiveWhitespaceScore = clamp01(1 - Math.min(uniqueDrugs, 60) / 60);
        const safetyPressure = clinicalSafetyRows * 1.5 + Math.max(0, safetyRows.length - clinicalSafetyRows) * 0.5;
        const safetyScore = clamp01(1 - Math.min(safetyPressure, 25) / 25);

        targetProfiles.push({
          targetId: target.id,
          approvedSymbol: target.approvedSymbol || entry.searchHint || entry.input,
          approvedName: target.approvedName || "Unknown target",
          diseaseAssociationScore,
          druggabilityScore,
          clinicalMaturityScore,
          competitiveWhitespaceScore,
          safetyScore,
          maxPhase,
          knownUniqueDrugs: uniqueDrugs,
          knownDrugRows: knownRows.length,
          withdrawnDrugRows,
          positiveTractabilityCount,
          safetyRows: safetyRows.length,
          clinicalSafetyRows,
          topMechanisms,
        });
        sources.push(`${OPEN_TARGETS_API}#target.snapshot:${target.id}`);
        sources.push(`https://platform.opentargets.org/target/${target.id}`);
      }

      if (targetProfiles.length < 2) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "Unable to compute multi-target ranking because fewer than two target profiles were available.",
                keyFields: [
                  `Targets requested: ${boundedTargets.join(", ")}`,
                  `Target profiles computed: ${targetProfiles.length}`,
                  `Unresolved targets: ${unresolvedTargets.length > 0 ? unresolvedTargets.join(" | ") : "none"}`,
                ],
                sources: dedupeArray(sources),
                limitations: ["Comparison requires at least two targets with retrievable snapshot data."],
              }),
            },
          ],
          structuredContent: buildCompareTargetsPayload({
            resultStatus: "not_found_or_empty",
            targetsRequested: boundedTargets,
            targetsResolved: uniqueTargets.length,
            targetsCompared: targetProfiles.length,
            unresolvedTargets,
            diseaseId: String(disease?.id || resolvedDisease?.diseaseId || "unknown"),
            diseaseName: String(disease?.name || resolvedDisease?.diseaseName || diseaseQuery || "unknown"),
            strategyRequested: String(strategy || "balanced").trim().toLowerCase(),
            strategyEffective: String(strategy || "balanced").trim().toLowerCase(),
            weightMode: "preset",
            goalText: String(goal || ""),
            notes: ["Insufficient target snapshots were available for scoring."],
          }),
        };
      }

      const requestedStrategy = String(strategy || "balanced").trim().toLowerCase();
      const goalText = String(goal || "").trim();
      const inferredFromGoal = inferStrategyFromGoalText(goalText);
      const rawCustomWeights = {
        diseaseAssociation: weightDiseaseAssociation,
        druggability: weightDruggability,
        clinicalMaturity: weightClinicalMaturity,
        competitiveWhitespace: weightCompetitiveWhitespace,
        safety: weightSafety,
      };
      const customWeightsProfile = customWeightProfile(rawCustomWeights);
      if (
        autoSelectStrategy &&
        !customWeightsProfile &&
        requestedStrategy === "balanced" &&
        inferredFromGoal.needsClarification
      ) {
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: "Ranking requires one primary optimization goal before scoring.",
                keyFields: [
                  `Goal text: ${goalText || "none"}`,
                  `Inference note: ${inferredFromGoal.reason}`,
                  `Clarification needed: ${inferredFromGoal.clarificationQuestion}`,
                ],
                sources: dedupeArray(sources),
                limitations: [
                  "Conflicting optimization goals can produce unstable ranking behavior unless one objective is primary.",
                ],
              }),
            },
          ],
          structuredContent: buildCompareTargetsPayload({
            resultStatus: "degraded",
            targetsRequested: boundedTargets,
            targetsResolved: uniqueTargets.length,
            targetsCompared: targetProfiles.length,
            unresolvedTargets,
            diseaseId: String(disease?.id || resolvedDisease?.diseaseId || "unknown"),
            diseaseName: String(disease?.name || resolvedDisease?.diseaseName || diseaseQuery || "unknown"),
            strategyRequested: requestedStrategy,
            strategyEffective: requestedStrategy,
            weightMode: "clarification_required",
            goalText,
            notes: [String(inferredFromGoal.reason || "Clarification required for goal prioritization.")],
            errorMessage: String(inferredFromGoal.clarificationQuestion || "Clarification required for ranking strategy."),
          }),
        };
      }

      const effectiveStrategy =
        autoSelectStrategy && requestedStrategy === "balanced" && !inferredFromGoal.needsClarification
          ? inferredFromGoal.mode
          : requestedStrategy;
      const presetWeights = strategyWeightProfile(effectiveStrategy);
      const weights = customWeightsProfile || presetWeights;
      const weightedProfiles = targetProfiles.map((profile) => {
        const compositeScore = clamp01(
          profile.diseaseAssociationScore * weights.disease_association +
            profile.druggabilityScore * weights.druggability +
            profile.clinicalMaturityScore * weights.clinical_maturity +
            profile.competitiveWhitespaceScore * weights.competitive_whitespace +
            profile.safetyScore * weights.safety
        );
        return { ...profile, compositeScore };
      });

      weightedProfiles.sort((a, b) => b.compositeScore - a.compositeScore);
      const lead = weightedProfiles[0];
      const runnerUp = weightedProfiles[1];
      const leadMargin = runnerUp ? clamp01(lead.compositeScore - runnerUp.compositeScore) : 0;
      const strategyLabel = String(effectiveStrategy || "balanced");
      const weightMode = customWeightsProfile
        ? "custom_override"
        : strategyLabel !== requestedStrategy
          ? "auto_inferred_from_goal"
          : "preset";
      const weightsText = [
        `disease_association=${formatPct(weights.disease_association)}`,
        `druggability=${formatPct(weights.druggability)}`,
        `clinical_maturity=${formatPct(weights.clinical_maturity)}`,
        `competitive_whitespace=${formatPct(weights.competitive_whitespace)}`,
        `safety=${formatPct(weights.safety)}`,
      ].join(", ");

      const keyFields = [
        `Disease context: ${disease.name} (${disease.id})`,
        `Targets requested: ${boundedTargets.join(", ")}`,
        `Targets compared: ${weightedProfiles.length}`,
        `Strategy: ${strategyLabel} (${weightMode})`,
        `Goal text: ${goalText || "none"}`,
        `Strategy inference: ${inferredFromGoal.reason}`,
        `Custom weights input: ${formatWeightInputSummary(rawCustomWeights)}`,
        `Weight profile: ${weightsText}`,
        `Lead target: ${lead.approvedSymbol} (${lead.targetId}) with composite ${formatPct(lead.compositeScore)}`,
        `Lead margin vs #2: ${formatPct(leadMargin)}`,
        ...weightedProfiles.map((profile, idx) => {
          return `${idx + 1}. ${profile.approvedSymbol} (${profile.targetId}) | Composite ${formatPct(
            profile.compositeScore
          )} | Disease assoc ${formatPct(profile.diseaseAssociationScore)} | Druggability ${formatPct(
            profile.druggabilityScore
          )} | Clinical maturity ${formatPct(profile.clinicalMaturityScore)} | White-space ${formatPct(
            profile.competitiveWhitespaceScore
          )} | Safety ${formatPct(profile.safetyScore)}`;
        }),
      ];

      if (includeBreakdown) {
        for (const profile of weightedProfiles.slice(0, Math.min(4, weightedProfiles.length))) {
          keyFields.push(
            `${profile.approvedSymbol} detail: max phase ${formatClinicalPhaseLabel(
              profile.maxPhase
            )}, unique drugs ${profile.knownUniqueDrugs}, withdrawn rows ${profile.withdrawnDrugRows}, tractability hits ${
              profile.positiveTractabilityCount
            }, safety rows ${profile.safetyRows} (clinical ${profile.clinicalSafetyRows}), top mechanisms: ${
              profile.topMechanisms.join("; ") || "N/A"
            }`
          );
        }
      }

      if (unresolvedTargets.length > 0) {
        keyFields.push(`Unresolved targets: ${unresolvedTargets.join(" | ")}`);
      }
      const rankingsPayload = weightedProfiles.map((profile, idx) => ({
        rank: idx + 1,
        target_id: String(profile.targetId || ""),
        symbol: String(profile.approvedSymbol || "Unknown"),
        approved_name: String(profile.approvedName || "Unknown target"),
        scores: {
          composite: toFiniteNumber(profile.compositeScore, 0),
          disease_association: toFiniteNumber(profile.diseaseAssociationScore, 0),
          druggability: toFiniteNumber(profile.druggabilityScore, 0),
          clinical_maturity: toFiniteNumber(profile.clinicalMaturityScore, 0),
          competitive_whitespace: toFiniteNumber(profile.competitiveWhitespaceScore, 0),
          safety: toFiniteNumber(profile.safetyScore, 0),
        },
        max_phase: toFiniteNumber(profile.maxPhase, 0),
        known_unique_drugs: Math.max(0, toFiniteNumber(profile.knownUniqueDrugs, 0)),
        withdrawn_drug_rows: toNonNegativeInt(profile.withdrawnDrugRows),
        positive_tractability_count: toNonNegativeInt(profile.positiveTractabilityCount),
        safety_rows: toNonNegativeInt(profile.safetyRows),
        clinical_safety_rows: toNonNegativeInt(profile.clinicalSafetyRows),
      }));
      const comparePayload = buildCompareTargetsPayload({
        resultStatus: "ok",
        targetsRequested: boundedTargets,
        targetsResolved: uniqueTargets.length,
        targetsCompared: weightedProfiles.length,
        unresolvedTargets,
        diseaseId: String(disease?.id || resolvedDisease?.diseaseId || "unknown"),
        diseaseName: String(disease?.name || resolvedDisease?.diseaseName || diseaseQuery || "unknown"),
        strategyRequested: requestedStrategy,
        strategyEffective: strategyLabel,
        weightMode,
        goalText,
        weights,
        leadTarget: {
          target_id: String(lead?.targetId || ""),
          symbol: String(lead?.approvedSymbol || "Unknown"),
          composite_score: toFiniteNumber(lead?.compositeScore, 0),
          lead_margin: toFiniteNumber(leadMargin, 0),
        },
        rankings: rankingsPayload,
        notes: [
          "Composite scores are heuristic and intended for decision support.",
          includeBreakdown ? "Axis breakdown details were included in text output." : "Axis breakdown was suppressed.",
        ],
      });

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Compared ${weightedProfiles.length} targets for ${disease.name}; top-ranked target is ${lead.approvedSymbol} under ${strategyLabel}${customWeightsProfile ? " + custom weight override" : ""} weighting.`,
              keyFields,
              sources: dedupeArray(sources).slice(0, 24),
              limitations: [
                "Composite ranking uses heuristic axis formulas and should be treated as decision support, not definitive truth.",
                "Scores depend on Open Targets coverage and current curation; missing data can shift rank positions.",
                "Competitive white-space and safety scores are proxies and do not replace trial-level due diligence or mechanistic toxicology review.",
              ],
            }),
          },
        ],
        structuredContent: comparePayload,
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in compare_targets_multi_axis: ${error.message}` }],
        structuredContent: buildCompareTargetsPayload({
          resultStatus: "error",
          targetsRequested: (targets || []).map((item) => String(item || "").trim()).filter(Boolean).slice(0, 6),
          targetsResolved: 0,
          targetsCompared: 0,
          unresolvedTargets: [],
          diseaseId: String(diseaseId || "unknown"),
          diseaseName: String(diseaseQuery || "unknown"),
          strategyRequested: String(strategy || "balanced").trim().toLowerCase(),
          strategyEffective: String(strategy || "balanced").trim().toLowerCase(),
          weightMode: "preset",
          goalText: String(goal || ""),
          notes: ["Unexpected error during multi-axis comparison."],
          errorMessage: String(error?.message || "unknown error"),
        }),
      };
    }
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
