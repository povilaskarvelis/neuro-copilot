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
    const result = await queryOpenTargets(graphqlQuery, { diseaseId, size: limit });
    const disease = result?.data?.disease;

    if (!disease) {
      return {
        content: [
          {
            type: "text",
            text: `Disease not found: "${diseaseId}". Use search_diseases to find valid disease IDs.`,
          },
        ],
      };
    }

    const rows = disease.associatedTargets?.rows ?? [];
    if (rows.length === 0) {
      return {
        content: [{ type: "text", text: `No targets found for disease: ${disease.name}` }],
      };
    }

    const formatted = rows
      .map((row, i) => {
        const t = row.target;
        const evidenceTypes = row.datatypeScores
          .filter((d) => d.score > 0)
          .map((d) => `${d.id}: ${(d.score * 100).toFixed(0)}%`)
          .join(", ");
        return `${i + 1}. ${t.approvedSymbol} (${t.approvedName})
   Target ID: ${t.id}
   Overall Score: ${(row.score * 100).toFixed(1)}%
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
    };
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
        return `${idx + 1}. PMID ${id} | ${item.title || "Untitled"} | ${item.pubdate || "Unknown date"} | First author: ${firstAuthor}`;
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
      const authors = parsePubmedAuthors(xml);
      const keyFields = [
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
        const firstAuthor = w.authorships?.[0]?.author?.display_name || "Unknown";
        return `${idx + 1}. ${w.display_name || "Untitled"} | ${w.publication_year || "Unknown year"} | First author: ${firstAuthor} | Cited by: ${w.cited_by_count ?? 0} | ID: ${w.id}`;
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
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in rank_researchers_by_activity: ${error.message}` }] };
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
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_clinical_trials_landscape: ${error.message}` }],
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
        };
      }

      const sources = [searchUrl];
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
          if (parents.length > 0) {
            keyFields.push(`Top parent concepts for ${docs[0].label}: ${parents.join(", ")}`);
          }
        }
      }

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
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in expand_disease_context: ${error.message}` }],
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
        return {
          content: [
            {
              type: "text",
              text: `ChEMBL target resolution failed for "${query}" (no target_chembl_id found).`,
            },
          ],
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
                  `Filter: ${activityType}, pChEMBL >= ${minPchembl}${Number.isFinite(safeNumber(maxNanomolar)) ? `, <= ${maxNanomolar} nM` : ""}`,
                ],
                sources: [targetSearchUrl, activityUrl],
                limitations: ["Filters may be too strict; lower pChEMBL threshold or remove nM cutoff."],
              }),
            },
          ],
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
        `Activity filter: ${activityType}, pChEMBL >= ${minPchembl}${Number.isFinite(safeNumber(maxNanomolar)) ? `, <= ${maxNanomolar} nM` : ""}`,
        ...rankedCompounds.map(
          (compound, idx) =>
            `${idx + 1}. ${compound.name} (${compound.moleculeId}) | ${compound.standardType} ${compound.relation} ${compound.standardValue} ${compound.standardUnits} | pChEMBL ${Number.isFinite(compound.pchembl) ? compound.pchembl.toFixed(2) : "N/A"} | Assay ${compound.assayId} | Doc ${compound.documentId} (${compound.documentYear})`
        ),
      ];

      const topCandidateIds = rankedCandidates
        .slice(0, 3)
        .map((entry) => `${entry.target?.target_chembl_id || "N/A"} (${entry.target?.organism || "Unknown"})`);
      keyFields.push(`Top candidate target matches: ${topCandidateIds.join(", ")}`);

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
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in search_chembl_compounds_for_target: ${error.message}` }],
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
          return {
            content: [
              {
                type: "text",
                text: "Provide either `targetId` (ENSG...) or `geneSymbol` to summarize expression context.",
              },
            ],
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
        if (includeCellTypes) {
          const topCellTypes = (protein?.cellType || []).slice(0, 4).map((cell) => {
            const label = `${cell?.name || "unknown"}:${mapProteinLevel(cell?.level)}`;
            incrementCount(cellTypeCounts, cell?.name || "unknown");
            return label;
          });
          if (topCellTypes.length > 0) {
            cellTypeText = ` | Cell types: ${topCellTypes.join(", ")}`;
          }
        }

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
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_target_expression_context: ${error.message}` }],
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
      };
    } catch (error) {
      if (isLikelyTransientUpstreamError(error) || isGwasEndpointError(error) || isGwasCooldownActive()) {
        const fallback = await buildOpenTargetsGeneGeneticsFallback(
          String(geneSymbol || "").trim(),
          String(diseaseQuery || "").trim(),
          8
        );
        if (fallback) {
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
          };
        }
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
        };
      }
      return {
        content: [{ type: "text", text: `Error in infer_genetic_effect_direction: ${error.message}` }],
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
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_target_competitive_landscape: ${error.message}` }],
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
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in summarize_target_safety_liabilities: ${error.message}` }],
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
        };
      }

      const resolvedDisease = await resolveDiseaseFromInput({ diseaseId, diseaseQuery });
      if (resolvedDisease?.error) {
        return {
          content: [{ type: "text", text: resolvedDisease.error }],
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
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in compare_targets_multi_axis: ${error.message}` }],
      };
    }
  }
);

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch(console.error);
