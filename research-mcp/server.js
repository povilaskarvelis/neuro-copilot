import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { BigQuery } from "@google-cloud/bigquery";
import { XMLParser } from "fast-xml-parser";
import { z } from "zod";
import { execFile, execFileSync } from "node:child_process";
import fs from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const execFileAsync = promisify(execFile);

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
const INTACT_API = "https://www.ebi.ac.uk/intact/ws";
const BIOGRID_API = "https://webservice.thebiogrid.org";
const BIOGRID_ORCS_API = "https://orcsws.thebiogrid.org";
const ENSEMBL_REST_API = "https://rest.ensembl.org";
const CIVIC_GRAPHQL_API = "https://civicdb.org/api/graphql";
const MYGENE_API = "https://mygene.info/v3";
const MYVARIANT_API = "https://myvariant.info/v1";
const OXO_API = "https://www.ebi.ac.uk/spot/oxo/api";
const OLS_API = "https://www.ebi.ac.uk/ols4/api";
const QUICKGO_API = "https://www.ebi.ac.uk/QuickGO/services";
const MONARCH_API = "https://monarchinitiative.org/v3/api";
const ALLIANCE_GENOME_API = "https://www.alliancegenome.org/api";
const GTOPDB_API = "https://www.guidetopharmacology.org/services";
const ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api";
const ALPHAFOLD_FTP_ROOT = "https://ftp.ebi.ac.uk/pub/databases/alphafold";
const GWAS_CATALOG_API = "https://www.ebi.ac.uk/gwas/rest/api/v2";
const GWAS_CATALOG_LEGACY_API = "https://www.ebi.ac.uk/gwas/rest/api";
const JASPAR_API = "https://jaspar.elixir.no/api/v1";
const DGIDB_GRAPHQL_API = "https://dgidb.org/api/graphql";
const GTEX_API = "https://gtexportal.org/api/v2";
const HPA_API = "https://www.proteinatlas.org";
const DAILYMED_API = "https://dailymed.nlm.nih.gov/dailymed/services/v2";
const PATHWAY_COMMONS_API = "https://www.pathwaycommons.org/pc2";
const EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest";
const CLINGEN_GENE_VALIDITY_DOWNLOAD = "https://search.clinicalgenome.org/kb/gene-validity/download";
const CLINGEN_GENE_DOSAGE_DOWNLOAD = "https://search.clinicalgenome.org/kb/gene-dosage/download";
const ORPHADATA_PRODUCT1_XML = "https://www.orphadata.com/data/xml/en_product1.xml";
const ORPHADATA_PRODUCT4_XML = "https://www.orphadata.com/data/xml/en_product4.xml";
const ORPHADATA_PRODUCT6_XML = "https://www.orphadata.com/data/xml/en_product6.xml";
const CANCERRXGENE_API = "https://www.cancerrxgene.org/api";
const PHARMACODB_GRAPHQL_API = "https://pharmacodb.ca/graphql";
const PRISM_24Q2_FIGSHARE_ARTICLE = "https://figshare.com/articles/dataset/Repurposing_Public_24Q2/25917643";
const PRISM_24Q2_COMPOUND_LIST_URL = "https://ndownloader.figshare.com/files/46630981";
const PRISM_24Q2_CELL_LINE_METADATA_URL = "https://ndownloader.figshare.com/files/46630978";
const PRISM_24Q2_PRIMARY_MATRIX_URL = "https://ndownloader.figshare.com/files/46630984";
const RCSB_SEARCH_API = "https://search.rcsb.org/rcsbsearch/v2/query";
const RCSB_DATA_API = "https://data.rcsb.org/rest/v1";
const CBIOPORTAL_API = "https://www.cbioportal.org/api";
const DEPMAP_PORTAL_API = "https://depmap.org/portal";
const GDC_API = "https://api.gdc.cancer.gov";
const PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug";
const CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data";
const IEDB_QUERY_API = "https://query-api.iedb.org";
const ABA_API = "https://api.brain-map.org/api/v2";
const EBRAINS_KG_SEARCH_API = "https://search.kg.ebrains.eu/api";
const HF_DATASETS_SERVER_API = "https://datasets-server.huggingface.co";
const GITHUB_API = "https://api.github.com";
const CONP_GITHUB_ORG = "conpdatasets";
const NEMAR_GITHUB_ORG = "nemarDatasets";
const NEMAR_DATAEXPLORER_API = "https://nemar.org/dataexplorer";
const NEMAR_DATAEXPLORER_VIEW_API = `${NEMAR_DATAEXPLORER_API}/viewapi`;
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
const RESEARCH_MCP_PYTHON = String(process.env.RESEARCH_MCP_PYTHON || "python3").trim() || "python3";
const OPEN_TARGETS_RELEASE_QUERY_SCRIPT = path.join(__dirname, "open_targets_release_query.py");
const OPEN_TARGETS_L2G_QUERY_SCRIPT = path.join(__dirname, "open_targets_l2g_query.py");
const ALPHAFOLD_DOMAIN_QUERY_SCRIPT = path.join(__dirname, "alphafold_domain_plddt_query.py");
const BIOGRID_ACCESS_KEY = String(process.env.BIOGRID_ACCESS_KEY || "").trim();
const BIOGRID_ORCS_ACCESS_KEY = String(process.env.BIOGRID_ORCS_ACCESS_KEY || process.env.BIOGRID_ACCESS_KEY || "").trim();
const OPENALEX_MAILTO = process.env.OPENALEX_MAILTO || process.env.CONTACT_EMAIL || "";
const NCBI_API_KEY = String(process.env.NCBI_API_KEY || process.env.PUBMED_API_KEY || "").trim();
const NCBI_EMAIL = String(process.env.NCBI_EMAIL || process.env.CONTACT_EMAIL || OPENALEX_MAILTO || "").trim();
const BQ_PROJECT_ID = String(process.env.BQ_PROJECT_ID || process.env.GOOGLE_CLOUD_PROJECT || "").trim();
const BQ_LOCATION = String(process.env.BQ_LOCATION || process.env.GOOGLE_CLOUD_LOCATION || "US").trim() || "US";
const BQ_DATASET_ALLOWLIST = String(process.env.BQ_DATASET_ALLOWLIST || "").trim();
const BQ_DEFAULT_MAX_ROWS = 200;
const BQ_HARD_MAX_ROWS = 1000;
const BQ_DEFAULT_MAX_BYTES_BILLED = 5_000_000_000;
const BQ_QUERY_TIMEOUT_MS = 30_000;
const CELLXGENE_CURATION_API = "https://api.cellxgene.cziscience.com/curation/v1";
const CELLXGENE_DISCOVER_API = "https://api.cellxgene.cziscience.com/dp/v1";
const CELLXGENE_DE_API = "https://api.cellxgene.cziscience.com/de/v1";
const CELLXGENE_WMG_API = "https://api.cellxgene.cziscience.com/wmg/v2";
const FORBIDDEN_SQL_KEYWORDS = /\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|call|export|load|copy)\b/i;
const RESEARCH_MCP_VENV_PYTHON = path.resolve(__dirname, "../.venv/bin/python");
const RESEARCH_MCP_PYTHON_FALLBACKS = [
  RESEARCH_MCP_PYTHON,
  process.env.PYTHON,
  RESEARCH_MCP_VENV_PYTHON,
  "python3",
  "python",
  "/opt/homebrew/Caskroom/miniconda/base/bin/python3",
].filter(Boolean);
const RESEARCH_MCP_PYTHON_CACHE = new Map();
const MONARCH_ASSOCIATION_MODES = {
  disease_to_phenotype: {
    category: "biolink:DiseaseToPhenotypicFeatureAssociation",
    inputCategories: ["biolink:Disease"],
    counterpartSide: "object",
    contextSide: "subject",
    description: "Phenotypic features associated with a disease query.",
  },
  phenotype_to_gene: {
    category: "biolink:GeneToPhenotypicFeatureAssociation",
    inputCategories: ["biolink:PhenotypicFeature"],
    counterpartSide: "subject",
    contextSide: "object",
    description: "Genes associated with a phenotype query.",
  },
  disease_to_gene_causal: {
    category: "biolink:CausalGeneToDiseaseAssociation",
    inputCategories: ["biolink:Disease"],
    counterpartSide: "subject",
    contextSide: "object",
    description: "Causal gene associations for a disease query.",
  },
  disease_to_gene_correlated: {
    category: "biolink:CorrelatedGeneToDiseaseAssociation",
    inputCategories: ["biolink:Disease"],
    counterpartSide: "subject",
    contextSide: "object",
    description: "Correlated gene associations for a disease query.",
  },
  gene_to_phenotype: {
    category: "biolink:GeneToPhenotypicFeatureAssociation",
    inputCategories: ["biolink:Gene"],
    counterpartSide: "object",
    contextSide: "subject",
    description: "Phenotypic features associated with a gene query.",
  },
};
const ALLIANCE_MODEL_SPECIES_RANK = {
  "NCBITaxon:10090": 1,
  "NCBITaxon:10116": 2,
  "NCBITaxon:7955": 3,
  "NCBITaxon:7227": 4,
  "NCBITaxon:6239": 5,
  "NCBITaxon:559292": 6,
  "NCBITaxon:8364": 7,
  "NCBITaxon:8355": 8,
  "NCBITaxon:9606": 99,
};
let bigQueryClient = null;
let depMapSummaryCache = null;
let depMapDownloadCatalogCache = null;
const depMapSubtypeTreeCache = new Map();
const depMapSubtypeMatrixCache = new Map();
const depMapExpressionSubsetMeanCache = new Map();
const geoSupplementaryArchiveCache = new Map();
const geoSampleQuickMetadataCache = new Map();
let cellxgeneDatasetCache = null;
let cellxgeneWmgPrimaryDimensionsCache = null;
const cellxgeneWmgFiltersCache = new Map();
const cellxgeneDeFiltersCache = new Map();
let clinGenGeneValidityCache = null;
let clinGenDosageCache = null;
let gdscCompoundCache = null;
let prismCompoundCatalogCache = null;
let prismCellLineCatalogCache = null;
let prismMatrixCache = null;
let orphanetDisorderCatalogCache = null;
let orphanetPhenotypeCatalogCache = null;
let orphanetGeneCatalogCache = null;

function normalizeWhitespace(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}

function splitArchiveSearchTerms(rawQuery) {
  const raw = normalizeWhitespace(rawQuery || "");
  if (!raw) return [];

  const looksCompound = /\b(?:or|and|not)\b/i.test(raw) || /[,;|/]/.test(raw);
  if (!looksCompound) return [raw];

  const parts = raw
    .replace(/[()]/g, " ")
    .replace(/\s*\/\s*/g, " / ")
    .split(/\b(?:or|and|not)\b|[,;|/]/i)
    .map((part) => normalizeWhitespace(part))
    .filter((part) => part && !/^(?:or|and|not)$/i.test(part));

  if (parts.length === 0) return [raw];
  return [...new Set(parts)];
}

function buildArchiveSearchKeyFields(rawQuery, searchTerms, { browseLabel = "Browse: all", queryLabel = "Query" } = {}) {
  const raw = normalizeWhitespace(rawQuery || "");
  if (!raw) return [browseLabel];
  const fields = [`${queryLabel}: ${raw}`];
  if (searchTerms.length > 1) {
    fields.push(`Parsed search terms: ${searchTerms.join("; ")}`);
  }
  return fields;
}

function decodeHtmlEntities(value) {
  return String(value || "")
    .replace(/&nbsp;|&#160;/gi, " ")
    .replace(/&quot;/gi, "\"")
    .replace(/&#39;|&apos;/gi, "'")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

function stripHtmlToText(value, { preserveParagraphs = false } = {}) {
  let text = String(value || "");
  if (!text) return "";
  text = text
    .replace(/<br\s*\/?>/gi, "\n")
    .replace(/<\/p>/gi, preserveParagraphs ? "\n\n" : "\n")
    .replace(/<\/h[1-6]>/gi, preserveParagraphs ? "\n\n" : "\n")
    .replace(/<li[^>]*>/gi, preserveParagraphs ? "\n- " : " ")
    .replace(/<\/li>/gi, preserveParagraphs ? "\n" : " ")
    .replace(/<[^>]+>/g, " ");
  text = decodeHtmlEntities(text);
  if (!preserveParagraphs) {
    return normalizeWhitespace(text);
  }
  return text
    .split(/\r?\n/)
    .map((line) => normalizeWhitespace(line))
    .filter((line, idx, lines) => line || (idx > 0 && lines[idx - 1]))
    .join("\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

const CLINICAL_TRIAL_STATUS_VALUES = new Set([
  "RECRUITING",
  "COMPLETED",
  "ACTIVE_NOT_RECRUITING",
  "TERMINATED",
]);

function normalizeClinicalTrialStatus(value) {
  const raw = normalizeWhitespace(value).toUpperCase();
  if (!raw) return "";
  if (CLINICAL_TRIAL_STATUS_VALUES.has(raw)) {
    return raw;
  }
  const matches = raw.match(/\b(?:RECRUITING|COMPLETED|ACTIVE_NOT_RECRUITING|TERMINATED)\b/g) || [];
  const uniqueMatches = [...new Set(matches)];
  if (uniqueMatches.length === 1) {
    return uniqueMatches[0];
  }
  return "";
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

function asFiniteCount(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }
  const numeric = Number(String(value).replace(/,/g, ""));
  if (!Number.isFinite(numeric) || numeric < 0) {
    return null;
  }
  return Math.trunc(numeric);
}

function firstFiniteCount(...values) {
  for (const value of values) {
    const numeric = asFiniteCount(value);
    if (numeric !== null) {
      return numeric;
    }
  }
  return null;
}

function extractArrayLengthFromKeys(obj, keys = []) {
  if (!obj || typeof obj !== "object") {
    return null;
  }
  for (const key of keys) {
    if (Array.isArray(obj[key])) {
      return obj[key].length;
    }
  }
  return null;
}

function inferResultMode(toolName) {
  if (toolName.startsWith("search_")) {
    return "search";
  }
  if (toolName.startsWith("list_")) {
    return "list";
  }
  if (toolName.startsWith("summarize_") || toolName.startsWith("rank_")) {
    return "summary";
  }
  if (toolName.startsWith("get_")) {
    return "detail";
  }
  return "other";
}

function inferResultItemLabel(toolName) {
  const explicit = {
    search_clinical_trials: "clinical trials",
    summarize_clinical_trials_landscape: "studies",
    get_clinical_trial: "clinical trials",
    search_pubmed: "articles",
    search_pubmed_advanced: "articles",
    search_openalex_works: "articles",
    search_europe_pmc_literature: "articles",
    search_geo_datasets: "records",
    search_refseq_sequences: "RefSeq records",
    search_ucsc_genome: "UCSC genome hits",
    search_encode_metadata: "ENCODE records",
    search_zenodo_records: "Zenodo records",
    search_openneuro_datasets: "datasets",
    search_dandi_datasets: "datasets",
    search_nemar_datasets: "datasets",
    search_conp_datasets: "datasets",
    search_braincode_datasets: "datasets",
    search_enigma_datasets: "datasets",
    search_cellxgene_datasets: "datasets",
    search_drug_gene_interactions: "drug-gene interaction records",
    search_gwas_associations: "associations",
    search_hpo_terms: "phenotype terms",
    search_quickgo_terms: "terms",
    map_ontology_terms_oxo: "mappings",
    search_variants_by_gene: "variants",
    search_civic_variants: "variants",
    search_protein_structures: "structures",
  };
  return explicit[toolName] || "records";
}

function buildResultQueryScope(requestArgs) {
  if (!requestArgs || typeof requestArgs !== "object" || Array.isArray(requestArgs)) {
    return null;
  }
  const scope = {};
  for (const key of [
    "search",
    "query",
    "gene",
    "genes",
    "disease",
    "compound",
    "drugName",
    "variantId",
    "status",
    "entryType",
    "sort",
    "minDate",
    "maxDate",
    "maxMatches",
    "maxItems",
    "maxResults",
    "maxBases",
    "organism",
    "moleculeType",
    "identifier",
    "refseqOnly",
    "genome",
    "chrom",
    "start",
    "end",
    "track",
    "revComp",
    "hubUrl",
    "categories",
    "objectType",
    "searchTerm",
    "assayTitle",
    "accessionOrPath",
    "frame",
    "searchQuery",
    "recordType",
    "community",
    "allVersions",
    "recordId",
    "page",
  ]) {
    if (!(key in requestArgs)) {
      continue;
    }
    const rawValue = requestArgs[key];
    if (rawValue === null || rawValue === undefined || rawValue === "") {
      continue;
    }
    if (Array.isArray(rawValue)) {
      const items = rawValue.slice(0, 8).map((item) => normalizeWhitespace(String(item || ""))).filter(Boolean);
      if (items.length > 0) {
        scope[key] = items;
      }
      continue;
    }
    const scalar = normalizeWhitespace(String(rawValue));
    if (scalar) {
      scope[key] = scalar;
    }
  }
  return Object.keys(scope).length > 0 ? scope : null;
}

function inferResultMetaFromText(text) {
  const cleaned = normalizeWhitespace(text || "");
  if (!cleaned) {
    return {};
  }

  const meta = {};

  let match = cleaned.match(/\bShowing\s+([\d,]+)\s+of\s+([\d,]+)\b/i);
  if (match) {
    meta.returned_count = asFiniteCount(match[1]);
    meta.reported_total = asFiniteCount(match[2]);
    if (meta.returned_count !== null && meta.reported_total !== null) {
      meta.has_more = meta.reported_total > meta.returned_count;
    }
    return meta;
  }

  match = cleaned.match(/\bFound\s+([\d,]+)\b[^.]*\(\s*showing\s+top\s+([\d,]+)\s*\)/i)
    || cleaned.match(/\bFound\s+([\d,]+)\b[^.]*\(\s*showing\s+([\d,]+)(?:[^)]*)\)/i)
    || cleaned.match(/\b([\d,]+)\s+record\(s\)\b[^.]*\bShowing\s+top\s+([\d,]+)/i);
  if (match) {
    meta.reported_total = asFiniteCount(match[1]);
    meta.returned_count = asFiniteCount(match[2]);
    if (meta.returned_count !== null && meta.reported_total !== null) {
      meta.has_more = meta.reported_total > meta.returned_count;
    }
    return meta;
  }

  match = cleaned.match(/\b(?:Retrieved|Returned)\s+([\d,]+)\b/i);
  if (match) {
    meta.returned_count = asFiniteCount(match[1]);
  }

  if (meta.returned_count === undefined) {
    match = cleaned.match(/\bFound\s+([\d,]+)\b/i);
    if (match) {
      meta.returned_count = asFiniteCount(match[1]);
    }
  }

  match = cleaned.match(/\bTotal\s+(?:matches|interactions|records|results|studies|articles|works|associations|terms):\s*([\d,]+)/i);
  if (match) {
    meta.reported_total = asFiniteCount(match[1]);
  }

  return meta;
}

function normalizeResultMeta(toolName, { originalStructured, summary, combinedText, requestArgs }) {
  const existing = originalStructured?.result_meta && typeof originalStructured.result_meta === "object"
    ? { ...originalStructured.result_meta }
    : {};
  const inferred = inferResultMetaFromText(`${summary || ""}\n${combinedText || ""}`);
  const mode = normalizeWhitespace(existing.mode || inferResultMode(toolName)) || "other";
  const itemLabel = normalizeWhitespace(existing.item_label || inferResultItemLabel(toolName)) || "records";
  const returnedCount = firstFiniteCount(
    existing.returned_count,
    originalStructured?.returned_count,
    extractArrayLengthFromKeys(originalStructured, [
      "records",
      "results",
      "articles",
      "studies",
      "trials",
      "associations",
      "terms",
      "mappings",
      "works",
      "authors",
      "datasets",
      "items",
      "hits",
    ]),
    inferred.returned_count,
  );
  const reportedTotal = firstFiniteCount(
    existing.reported_total,
    originalStructured?.reported_total,
    originalStructured?.total_count,
    originalStructured?.totalCount,
    originalStructured?.total,
    originalStructured?.count,
    originalStructured?.totalReported,
    inferred.reported_total,
  );
  const limitApplied = firstFiniteCount(
    existing.limit_applied,
    requestArgs?.limit,
    requestArgs?.maxResults,
    requestArgs?.maxStudies,
    requestArgs?.size,
    requestArgs?.numRows,
    requestArgs?.maxStructures,
    requestArgs?.maxRecords,
    requestArgs?.mappingsPerId,
  );

  let hasMore = null;
  if (typeof existing.has_more === "boolean") {
    hasMore = existing.has_more;
  } else if (typeof originalStructured?.has_more === "boolean") {
    hasMore = originalStructured.has_more;
  } else if (typeof originalStructured?.hasMorePages === "boolean") {
    hasMore = originalStructured.hasMorePages;
  } else if (typeof originalStructured?.has_more_pages === "boolean") {
    hasMore = originalStructured.has_more_pages;
  } else if (normalizeWhitespace(originalStructured?.nextPageToken || "")) {
    hasMore = true;
  } else if (normalizeWhitespace(originalStructured?.next_page_token || "")) {
    hasMore = true;
  } else if (typeof inferred.has_more === "boolean") {
    hasMore = inferred.has_more;
  } else if (reportedTotal !== null && returnedCount !== null) {
    hasMore = reportedTotal > returnedCount;
  }

  let totalRelation = normalizeWhitespace(existing.total_relation || "").toLowerCase();
  if (!["exact", "lower_bound", "unknown"].includes(totalRelation)) {
    if (reportedTotal !== null) {
      totalRelation = "exact";
    } else if (hasMore === true) {
      totalRelation = "lower_bound";
    } else if (limitApplied !== null && returnedCount !== null && returnedCount >= limitApplied && mode !== "detail") {
      totalRelation = "lower_bound";
    } else {
      totalRelation = "unknown";
    }
  }

  const resultMeta = {
    mode,
    item_label: itemLabel,
    total_relation: totalRelation,
  };
  if (returnedCount !== null) {
    resultMeta.returned_count = returnedCount;
  }
  if (reportedTotal !== null) {
    resultMeta.reported_total = reportedTotal;
  }
  if (hasMore !== null) {
    resultMeta.has_more = hasMore;
  }
  if (limitApplied !== null) {
    resultMeta.limit_applied = limitApplied;
  }
  const queryScope = buildResultQueryScope(requestArgs);
  if (queryScope) {
    resultMeta.query_scope = queryScope;
  }
  return resultMeta;
}

function normalizeToolResponseEnvelope(toolName, rawResult, requestArgs) {
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
  const resultMeta = normalizeResultMeta(toolName, {
    originalStructured: normalizedPayload,
    summary,
    combinedText,
    requestArgs,
  });
  normalizedPayload.result_meta = resultMeta;

  const structuredContent = {
    envelope_version: STRUCTURED_CONTENT_ENVELOPE_VERSION,
    tool_name: toolName,
    status,
    summary,
    text: combinedText,
    content_part_count: safeContent.length,
    emitted_at_utc: new Date().toISOString(),
    result_meta: resultMeta,
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
      return normalizeToolResponseEnvelope(toolName, rawResult, args[0]);
    } catch (error) {
      const message = normalizeWhitespace(error?.message || String(error));
      return normalizeToolResponseEnvelope(toolName, {
        isError: true,
        content: [{ type: "text", text: `Error in ${toolName}: ${message}` }],
      }, args[0]);
    }
  };
}

const registerTool = server.registerTool.bind(server);
server.registerTool = (toolName, config, handler) =>
  registerTool(toolName, config, wrapToolHandler(toolName, handler));

function decodeXmlEntities(value) {
  return String(value || "")
    .replace(/&#x([0-9a-f]+);/gi, (_, hex) => {
      const codePoint = Number.parseInt(hex, 16);
      return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : _;
    })
    .replace(/&#(\d+);/g, (_, dec) => {
      const codePoint = Number.parseInt(dec, 10);
      return Number.isFinite(codePoint) ? String.fromCodePoint(codePoint) : _;
    })
    .replace(/&nbsp;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">")
    .replace(/&quot;/gi, "\"")
    .replace(/&apos;/gi, "'");
}
function sanitizeXmlText(value) {
  if (!value) return "";
  return decodeXmlEntities(value.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
}
function xmlToReadableText(value) {
  if (!value) return "";
  let text = String(value);
  const blockBreaks = [
    /<\s*br\s*\/?>/gi,
    /<\s*\/\s*(?:p|sec|title|abstract|list-item|li|caption|boxed-text|disp-quote|ack|ref)\s*>/gi,
  ];
  for (const re of blockBreaks) {
    text = text.replace(re, "\n\n");
  }
  text = text
    .replace(/<label[^>]*>([\s\S]*?)<\/label>/gi, "$1 ")
    .replace(/<xref[^>]*>([\s\S]*?)<\/xref>/gi, "$1")
    .replace(/<sup[^>]*>([\s\S]*?)<\/sup>/gi, "^$1")
    .replace(/<sub[^>]*>([\s\S]*?)<\/sub>/gi, "_$1")
    .replace(/<[^>]+>/g, " ");
  return decodeXmlEntities(text)
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
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

function buildIedbQueryUrl(pathname, params = new URLSearchParams()) {
  const query = params.toString();
  return query ? `${IEDB_QUERY_API}${pathname}?${query}` : `${IEDB_QUERY_API}${pathname}`;
}

function buildNcbiEutilsUrl(baseUrl, params = new URLSearchParams()) {
  const nextParams = new URLSearchParams(params);
  if (NCBI_API_KEY && !nextParams.has("api_key")) {
    nextParams.set("api_key", NCBI_API_KEY);
  }
  if (NCBI_EMAIL && !nextParams.has("email")) {
    nextParams.set("email", NCBI_EMAIL);
  }
  if (!nextParams.has("tool")) {
    nextParams.set("tool", "research-mcp");
  }
  const query = nextParams.toString();
  return query ? `${baseUrl}?${query}` : baseUrl;
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

function parseContentRangeTotal(value) {
  const normalized = normalizeWhitespace(value || "");
  if (!normalized) return null;
  const match = normalized.match(/\/(\d+|\*)$/);
  if (!match || match[1] === "*") return null;
  const total = Number(match[1]);
  return Number.isFinite(total) ? total : null;
}

function normalizePeptideSequence(value) {
  return normalizeWhitespace(value || "").toUpperCase().replace(/[^A-Z]/g, "");
}

function iedbArrayFilterContains(value) {
  const normalized = normalizeWhitespace(value || "");
  if (!normalized) return "";
  return `cs.{${JSON.stringify(normalized)}}`;
}

function toTextArray(value) {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeWhitespace(item)).filter(Boolean);
  }
  const normalized = normalizeWhitespace(value || "");
  return normalized ? [normalized] : [];
}

function extractIedbAntigenNames(row) {
  const direct = [
    ...toTextArray(row?.parent_source_antigen_names),
    ...toTextArray(row?.parent_source_antigen_name),
    ...toTextArray(row?.curated_source_antigen),
  ];
  const curated = Array.isArray(row?.curated_source_antigens)
    ? row.curated_source_antigens.map((item) => normalizeWhitespace(item?.name || "")).filter(Boolean)
    : [];
  return dedupeArray([...direct, ...curated]);
}

function extractIedbAntigenIris(row) {
  const direct = [
    ...toTextArray(row?.parent_source_antigen_iris),
    ...toTextArray(row?.parent_source_antigen_iri),
  ];
  const curated = Array.isArray(row?.curated_source_antigens)
    ? row.curated_source_antigens.map((item) => normalizeWhitespace(item?.iri || "")).filter(Boolean)
    : [];
  return dedupeArray([...direct, ...curated]);
}

function extractIedbAlleleNames(row) {
  return dedupeArray([
    ...toTextArray(row?.mhc_allele_names),
    ...toTextArray(row?.mhc_allele_name),
  ]);
}

function extractIedbQualitativeMeasures(row) {
  return dedupeArray([
    ...toTextArray(row?.qualitative_measures),
    ...toTextArray(row?.qualitative_measure),
  ]);
}

function extractIedbPubmedIds(row) {
  return dedupeArray([
    ...toTextArray(row?.pubmed_ids),
    ...toTextArray(row?.pubmed_id),
  ]);
}

function extractIedbReferenceTitles(row) {
  return dedupeArray([
    ...toTextArray(row?.reference_titles),
    ...toTextArray(row?.reference_title),
    ...toTextArray(row?.reference_summary),
  ]);
}

function extractIedbDiseaseNames(row) {
  return dedupeArray([
    ...toTextArray(row?.disease_names),
    ...toTextArray(row?.disease_name),
  ]);
}

function extractIedbHostNames(row) {
  return dedupeArray([
    ...toTextArray(row?.host_organism_names),
    ...toTextArray(row?.host_organism_name),
  ]);
}

function extractIedbAssayNames(row) {
  return dedupeArray([
    ...toTextArray(row?.assay_names),
    ...toTextArray(row?.assay_name),
  ]);
}

function matchesIedbText(values, query) {
  const normalizedQuery = normalizeWhitespace(query || "").toLowerCase();
  if (!normalizedQuery) return true;
  return (Array.isArray(values) ? values : [])
    .some((value) => normalizeWhitespace(value).toLowerCase().includes(normalizedQuery));
}

function looksLikeMutationToken(value) {
  const normalized = normalizeWhitespace(value || "");
  return /\b(?:[A-Z]\d+[A-Z]|p\.[A-Za-z]{3}\d+[A-Za-z]{3})\b/.test(normalized);
}

function deriveIedbAntigenCandidates(value) {
  const normalized = normalizeWhitespace(value || "");
  if (!normalized) return [];
  const candidates = [normalized];
  const firstToken = normalized.split(/\s+/)[0] || "";
  if (looksLikeBareGeneSymbol(firstToken)) {
    candidates.push(firstToken);
  }
  const strippedMutation = normalizeWhitespace(
    normalized
      .replace(/\bp\.[A-Za-z]{3}\d+[A-Za-z]{3}\b/gi, " ")
      .replace(/\b[A-Z]\d+[A-Z]\b/g, " ")
  );
  if (strippedMutation) {
    candidates.push(strippedMutation);
    const strippedFirst = strippedMutation.split(/\s+/)[0] || "";
    if (looksLikeBareGeneSymbol(strippedFirst)) {
      candidates.push(strippedFirst);
    }
  }
  return dedupeArray(candidates).slice(0, 4);
}

async function resolveIedbAntigenTarget(query) {
  const candidates = deriveIedbAntigenCandidates(query);
  for (const candidate of candidates) {
    const params = new URLSearchParams({
      query: `(${candidate}) AND reviewed:true`,
      format: "json",
      size: "5",
      fields: "accession,id,protein_name,gene_names,organism_name,reviewed",
    });
    const url = buildUniProtUrl("/uniprotkb/search", params);
    try {
      const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 9000, maxBackoffMs: 2500 });
      const rawResults = Array.isArray(data?.results) ? data.results : [];
      const ranked = rankUniProtResults(rawResults, candidate);
      const selected = ranked[0];
      if (!selected) continue;
      const accession = normalizeWhitespace(selected?.primaryAccession || "");
      if (!accession) continue;
      return {
        query: candidate,
        accession,
        iri: `UNIPROT:${accession}`,
        label: extractUniProtProteinName(selected),
        gene_symbols: extractUniProtGeneSymbols(selected, 4),
        source_url: url,
      };
    } catch (_) {
      // Keep trying simplified antigen candidates.
    }
  }
  return null;
}

const IEDB_ENDPOINT_CONFIG = {
  epitope: {
    path: "/epitope_search",
    select: [
      "structure_id",
      "structure_iri",
      "linear_sequence",
      "parent_source_antigen_names",
      "parent_source_antigen_iris",
      "mhc_allele_names",
      "qualitative_measures",
      "pubmed_ids",
      "reference_titles",
      "disease_names",
      "host_organism_names",
      "assay_names",
    ].join(","),
  },
  tcell: {
    path: "/tcell_search",
    select: [
      "tcell_id",
      "tcell_iri",
      "structure_id",
      "structure_iri",
      "linear_sequence",
      "parent_source_antigen_name",
      "parent_source_antigen_iri",
      "mhc_allele_name",
      "qualitative_measure",
      "pubmed_id",
      "reference_titles",
      "reference_summary",
      "disease_names",
      "host_organism_name",
      "assay_names",
    ].join(","),
  },
  mhc: {
    path: "/mhc_search",
    select: [
      "elution_id",
      "elution_iri",
      "structure_id",
      "structure_iri",
      "linear_sequence",
      "parent_source_antigen_name",
      "parent_source_antigen_iri",
      "mhc_allele_name",
      "qualitative_measure",
      "quantitative_measure",
      "pubmed_id",
      "reference_titles",
      "reference_summary",
      "disease_names",
      "host_organism_name",
      "assay_names",
    ].join(","),
  },
};

async function fetchIedbEndpoint(endpoint, { peptide, antigenTarget, allele, positiveOnly, limit }) {
  const config = IEDB_ENDPOINT_CONFIG[endpoint];
  if (!config) throw new Error(`Unknown IEDB endpoint: ${endpoint}`);
  const params = new URLSearchParams({
    select: config.select,
    limit: String(limit),
  });
  if (peptide) {
    params.set("linear_sequence", `ilike.*${peptide}*`);
  }
  if (antigenTarget?.iri) {
    if (endpoint === "epitope") {
      params.set("parent_source_antigen_iris", `cs.{${antigenTarget.iri}}`);
    } else {
      params.set("parent_source_antigen_iri", `eq.${antigenTarget.iri}`);
    }
  }
  if (allele) {
    if (endpoint === "epitope") {
      params.set("mhc_allele_names", iedbArrayFilterContains(allele));
    } else {
      params.set("mhc_allele_name", `eq.${allele}`);
    }
  }
  if (positiveOnly) {
    if (endpoint === "epitope") {
      params.set("qualitative_measures", iedbArrayFilterContains("Positive"));
    } else {
      params.set("qualitative_measure", "eq.Positive");
    }
  }

  const url = buildIedbQueryUrl(config.path, params);
  const response = await fetchWithRetry(url, {
    retries: 1,
    timeoutMs: 12000,
    maxBackoffMs: 2500,
    headers: { Prefer: "count=exact" },
  });
  const totalCount = parseContentRangeTotal(response.headers.get("content-range"));
  const rows = await response.json();
  if (!Array.isArray(rows)) {
    throw new Error(`IEDB returned a non-array response for ${endpoint}.`);
  }
  return { rows, totalCount, url };
}

function normalizeIedbRecord(endpoint, row) {
  return {
    endpoint,
    record_id:
      normalizeWhitespace(row?.tcell_iri || row?.elution_iri || row?.structure_iri || "")
      || normalizeWhitespace(String(row?.tcell_id || row?.elution_id || row?.structure_id || "")),
    structure_id: normalizeWhitespace(row?.structure_iri || "") || null,
    sequence: normalizePeptideSequence(row?.linear_sequence || "") || null,
    antigen_names: extractIedbAntigenNames(row),
    antigen_iris: extractIedbAntigenIris(row),
    allele_names: extractIedbAlleleNames(row),
    qualitative_measures: extractIedbQualitativeMeasures(row),
    quantitative_measure: normalizeWhitespace(row?.quantitative_measure || "") || null,
    pubmed_ids: extractIedbPubmedIds(row),
    reference_titles: extractIedbReferenceTitles(row),
    disease_names: extractIedbDiseaseNames(row),
    host_organism_names: extractIedbHostNames(row),
    assay_names: extractIedbAssayNames(row),
  };
}

function scoreIedbRecord(record, { peptide, allele, antigenQuery, disease, hostOrganism }) {
  let score = 0;
  if (record.endpoint === "tcell") score += 18;
  else if (record.endpoint === "mhc") score += 12;
  else score += 6;

  if (peptide) {
    if (record.sequence === peptide) score += 80;
    else if ((record.sequence || "").includes(peptide)) score += 45;
  }
  if (allele && record.allele_names.includes(allele)) score += 35;
  if (antigenQuery && matchesIedbText(record.antigen_names, antigenQuery)) score += 24;
  if (disease && matchesIedbText(record.disease_names, disease)) score += 12;
  if (hostOrganism && matchesIedbText(record.host_organism_names, hostOrganism)) score += 10;
  if (record.qualitative_measures.includes("Positive")) score += 8;
  if (record.pubmed_ids.length > 0) score += Math.min(8, record.pubmed_ids.length);
  if (record.allele_names.length > 0) score += 2;
  return score;
}

function formatIedbRecordLine(record, idx) {
  const antigen = compactErrorMessage(record.antigen_names[0] || "unknown antigen", 90);
  const allele = compactErrorMessage(record.allele_names.slice(0, 3).join(", ") || "unspecified", 70);
  const measureBits = [];
  if (record.qualitative_measures.length > 0) {
    measureBits.push(record.qualitative_measures.join(", "));
  }
  if (record.quantitative_measure) {
    measureBits.push(record.quantitative_measure);
  }
  const measure = compactErrorMessage(measureBits.join(" | ") || "not reported", 70);
  const pmidText = record.pubmed_ids.length > 0 ? record.pubmed_ids.slice(0, 3).join(", ") : "n/a";
  return `${idx + 1}. [${record.endpoint}] ${record.sequence || "sequence unavailable"} | antigen: ${antigen} | allele: ${allele} | measure: ${measure} | PMID: ${pmidText}`;
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

function extractUniProtDiseaseVariantAnnotations(entry, limitDiseases = 6, limitVariantsPerDisease = 8) {
  const comments = Array.isArray(entry?.comments) ? entry.comments : [];
  const features = Array.isArray(entry?.features) ? entry.features : [];

  const diseaseEntries = [];
  for (const comment of comments) {
    if (normalizeWhitespace(comment?.commentType || "").toUpperCase() !== "DISEASE") continue;
    const disease = comment?.disease || {};
    const acronym = normalizeWhitespace(disease?.acronym || "");
    const diseaseId = normalizeWhitespace(disease?.diseaseId || "");
    const aliases = [];
    if (acronym) aliases.push(acronym);
    if (acronym) {
      const withoutTrailingDigits = acronym.replace(/\d+$/, "");
      if (withoutTrailingDigits && withoutTrailingDigits !== acronym) aliases.push(withoutTrailingDigits);
    }
    if (diseaseId) aliases.push(diseaseId);
    const normalizedAliases = dedupeArray(aliases.map((value) => normalizeWhitespace(value)).filter(Boolean));
    if (normalizedAliases.length === 0) continue;
    const primaryAlias = normalizedAliases[0];
    const baseAlias = primaryAlias.replace(/\d+$/, "");
    const label =
      baseAlias && baseAlias !== primaryAlias
        ? `${baseAlias}/${primaryAlias}`
        : primaryAlias;
    diseaseEntries.push({
      label,
      aliases: normalizedAliases,
      diseaseId,
      acronym,
    });
  }

  const annotations = [];
  for (const disease of diseaseEntries) {
    const matchedVariants = [];
    for (const feature of features) {
      if (normalizeWhitespace(feature?.type || "").toLowerCase() !== "natural variant") continue;
      const description = normalizeWhitespace(feature?.description || "");
      if (!description) continue;
      const loweredDescription = description.toLowerCase();
      const matchedAlias = disease.aliases.find((alias) => loweredDescription.includes(alias.toLowerCase()));
      if (!matchedAlias) continue;

      const start = toNonNegativeInt(feature?.location?.start?.value);
      const end = toNonNegativeInt(feature?.location?.end?.value);
      let position = "";
      if (start > 0 && end > 0) {
        position = start === end ? `${start}` : `${start}-${end}`;
      } else {
        position = normalizeWhitespace(feature?.location?.start?.value || feature?.location?.end?.value || "");
      }
      if (!position) continue;

      const originalResidue = normalizeWhitespace(feature?.alternativeSequence?.originalSequence || "");
      const alternativeResidues = Array.isArray(feature?.alternativeSequence?.alternativeSequences)
        ? feature.alternativeSequence.alternativeSequences.map((value) => normalizeWhitespace(value)).filter(Boolean)
        : [];
      const dbsnpRef = (Array.isArray(feature?.featureCrossReferences) ? feature.featureCrossReferences : []).find(
        (ref) => normalizeWhitespace(ref?.database || "").toLowerCase() === "dbsnp"
      );
      const dbsnpId = normalizeWhitespace(dbsnpRef?.id || "");
      const variantNotation =
        originalResidue && alternativeResidues.length > 0
          ? `${originalResidue}${position}${alternativeResidues[0]}`
          : "";
      matchedVariants.push({
        position,
        positionSort: toNonNegativeInt(start || end || Number.NaN),
        description,
        variantNotation,
        dbsnpId,
        featureId: normalizeWhitespace(feature?.featureId || ""),
      });
    }

    if (matchedVariants.length === 0) continue;
    const deduped = [];
    const seen = new Set();
    for (const variant of matchedVariants.sort((left, right) => {
      const leftPos = Number.isFinite(left.positionSort) ? left.positionSort : Number.POSITIVE_INFINITY;
      const rightPos = Number.isFinite(right.positionSort) ? right.positionSort : Number.POSITIVE_INFINITY;
      return leftPos - rightPos || left.position.localeCompare(right.position);
    })) {
      const key = `${variant.position}|${variant.variantNotation}|${variant.dbsnpId}`;
      if (seen.has(key)) continue;
      seen.add(key);
      deduped.push(variant);
      if (deduped.length >= limitVariantsPerDisease) break;
    }

    const uniquePositions = dedupeArray(deduped.map((variant) => variant.position)).slice(0, limitVariantsPerDisease);
    annotations.push({
      label: disease.label,
      acronym: disease.acronym,
      diseaseId: disease.diseaseId,
      aliases: disease.aliases,
      positions: uniquePositions,
      variants: deduped.map((variant) => ({
        position: variant.position,
        notation: variant.variantNotation || null,
        dbsnpId: variant.dbsnpId || null,
        description: variant.description,
        featureId: variant.featureId || null,
      })),
    });
    if (annotations.length >= limitDiseases) break;
  }

  return annotations;
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

async function fetchBufferWithRetry(url, options = {}) {
  const response = await fetchWithRetry(url, options);
  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

function helperScriptRequiresPandas(scriptPath) {
  const base = path.basename(String(scriptPath || ""));
  return base === "open_targets_release_query.py" || base === "open_targets_l2g_query.py";
}

function dedupeStringArray(values) {
  const seen = new Set();
  const out = [];
  for (const raw of values || []) {
    const value = String(raw || "").trim();
    if (!value || seen.has(value)) {
      continue;
    }
    seen.add(value);
    out.push(value);
  }
  return out;
}

function pythonSupportsModule(executable, moduleName) {
  try {
    execFileSync(executable, ["-c", `import ${moduleName}`], {
      cwd: __dirname,
      stdio: "ignore",
      timeout: 10000,
    });
    return true;
  } catch {
    return false;
  }
}

function resolveResearchMcpPython(scriptPath) {
  const requirePandas = helperScriptRequiresPandas(scriptPath);
  const cacheKey = requirePandas ? "requires_pandas" : "generic";
  if (RESEARCH_MCP_PYTHON_CACHE.has(cacheKey)) {
    return RESEARCH_MCP_PYTHON_CACHE.get(cacheKey);
  }

  const candidates = dedupeStringArray(RESEARCH_MCP_PYTHON_FALLBACKS);
  if (!requirePandas) {
    const selected = candidates[0] || RESEARCH_MCP_PYTHON;
    RESEARCH_MCP_PYTHON_CACHE.set(cacheKey, selected);
    return selected;
  }

  for (const candidate of candidates) {
    if (pythonSupportsModule(candidate, "pandas")) {
      RESEARCH_MCP_PYTHON_CACHE.set(cacheKey, candidate);
      return candidate;
    }
  }

  const selected = candidates[0] || RESEARCH_MCP_PYTHON;
  RESEARCH_MCP_PYTHON_CACHE.set(cacheKey, selected);
  return selected;
}

async function runPythonJsonHelper(scriptPath, payload, { timeoutMs = 120000 } = {}) {
  const pythonExecutable = resolveResearchMcpPython(scriptPath);
  const { stdout } = await execFileAsync(
    pythonExecutable,
    [scriptPath, JSON.stringify(payload || {})],
    {
      cwd: __dirname,
      timeout: timeoutMs,
      maxBuffer: 25 * 1024 * 1024,
    }
  );
  const raw = String(stdout || "").trim();
  if (!raw) {
    throw new Error(`Helper script ${path.basename(scriptPath)} returned no output.`);
  }
  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`Helper script ${path.basename(scriptPath)} returned invalid JSON: ${raw.slice(0, 400)}`);
  }
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

function buildBigQuerySummary(rows, referencedDatasets, rowCount) {
  if (!rowCount || !rows || rows.length === 0) {
    return `Query returned 0 rows from ${referencedDatasets.join(", ")}.`;
  }
  const first = rows[0];
  const keys = Object.keys(first || {});
  // Extract the most informative fields from the first row
  const highlights = [];
  const nameFields = ["name", "approvedSymbol", "symbol", "label", "title", "approvedName"];
  const scoreFields = ["overallAssociationScore", "score", "associationScore", "datatypeId"];
  const idFields = ["id", "targetId", "diseaseId", "ensemblId"];
  for (const f of nameFields) {
    const val = first[f];
    if (val && typeof val === "string") { highlights.push(val); break; }
  }
  for (const f of scoreFields) {
    const val = first[f];
    if (val !== undefined && val !== null) {
      const num = typeof val === "number" ? val : parseFloat(val);
      if (!isNaN(num)) { highlights.push(`score: ${num}`); break; }
    }
  }
  for (const f of idFields) {
    const val = first[f];
    if (val && typeof val === "string" && !highlights.includes(val)) {
      highlights.push(val); break;
    }
  }
  const detail = highlights.length > 0 ? ` — ${highlights.join(", ")}` : "";
  if (rowCount === 1) {
    return `Found 1 result${detail}.`;
  }
  return `Found ${rowCount} results${detail}.`;
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

function buildBigQueryErrorHint(query, errorMessage) {
  const normalizedMessage = normalizeWhitespace(errorMessage || "");
  const normalizedQuery = String(query || "");
  const hints = [];

  if (
    /concatenated string literals/i.test(normalizedMessage)
    || /Expected end of input but got identifier/i.test(normalizedMessage)
  ) {
    if (normalizedQuery.includes("'")) {
      hints.push("If a SQL string contains an apostrophe, escape it as two single quotes, e.g. 'Alzheimer''s disease'.");
    } else {
      hints.push("Check string quoting and unmatched apostrophes in SQL literals.");
    }
  }

  if (
    /does not exist in STRUCT<element STRUCT/i.test(normalizedMessage)
    || /Cannot access field .* on a value with type ARRAY<STRUCT/i.test(normalizedMessage)
  ) {
    hints.push("A nested ARRAY<STRUCT> usually needs UNNEST with an alias before accessing inner fields.");
    hints.push("Re-inspect the schema and preview one row with SELECT TO_JSON_STRING(t) FROM `dataset.table` AS t LIMIT 1.");
  }

  if (/Unrecognized name/i.test(normalizedMessage)) {
    hints.push("Re-check exact column names with list_bigquery_tables(dataset=..., table=...).");
  }

  if (/Not found: Job/i.test(normalizedMessage)) {
    hints.push("Dry-run jobs do not support follow-up metadata/result lookups; use the stats returned by createQueryJob directly. For executed queries, also re-check the configured BigQuery location.");
  }

  if (/Query exceeded limit for bytes billed/i.test(normalizedMessage)) {
    if (/`?bigquery-public-data\.umiami_lincs\.readout`?/i.test(normalizedQuery) || /`?umiami_lincs\.readout`?/i.test(normalizedQuery)) {
      hints.push(
        "umiami_lincs.readout is extremely large; broad gene-list filters still tend to scan roughly the full table."
      );
      hints.push(
        "Use umiami_lincs metadata tables first (signature, perturbagen, small_molecule, model_system, cell_line), switch to PRISM/GDSC/PharmacoDB for compound prioritization, or raise BQ_MAX_BYTES_BILLED only if you intentionally want a costly raw LINCS scan."
      );
    } else {
      hints.push("Run the same query with dryRun=true to estimate bytes, then narrow joins/filters or raise BQ_MAX_BYTES_BILLED if the scan is intentionally large.");
    }
  }

  return hints.join(" ");
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
function clamp01(value) {
  const numeric = safeNumber(value, 0);
  if (numeric <= 0) return 0;
  if (numeric >= 1) return 1;
  return numeric;
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

function dedupeArray(values) {
  return [...new Set(values.filter(Boolean))];
}

function asArray(value) {
  if (Array.isArray(value)) return value;
  if (value === null || value === undefined) return [];
  return [value];
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

function parseXmlDocument(text) {
  const parser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "",
    trimValues: true,
    parseTagValue: false,
    removeNSPrefix: true,
  });
  return parser.parse(String(text || ""));
}

function getFreshCacheValue(cacheEntry, ttlMs) {
  if (!cacheEntry || typeof cacheEntry !== "object") return null;
  const fetchedAt = toNonNegativeInt(cacheEntry.fetchedAt, 0);
  if (!fetchedAt) return null;
  if (Date.now() - fetchedAt > ttlMs) return null;
  return cacheEntry.value ?? null;
}

function storeCacheValue(cacheTarget, value) {
  return {
    fetchedAt: Date.now(),
    value,
  };
}

function parseDelimitedLine(line, delimiter = ",") {
  const out = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === "\"") {
      if (inQuotes && line[i + 1] === "\"") {
        current += "\"";
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (char === delimiter && !inQuotes) {
      out.push(current);
      current = "";
      continue;
    }
    current += char;
  }
  out.push(current);
  return out;
}

function parseDelimitedObjects(text, delimiter = ",") {
  const lines = String(text || "")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
  if (lines.length === 0) return [];
  const headers = parseDelimitedLine(lines[0], delimiter).map((value) => value.trim());
  const rows = [];
  for (const line of lines.slice(1)) {
    const cols = parseDelimitedLine(line, delimiter);
    const row = {};
    for (let i = 0; i < headers.length; i += 1) {
      row[headers[i]] = cols[i] ?? "";
    }
    rows.push(row);
  }
  return rows;
}

function parseCsvLine(line) {
  return parseDelimitedLine(line, ",");
}

function parseCsvObjects(text) {
  return parseDelimitedObjects(text, ",");
}

function findCsvRowByValue(text, columnName, targetValue) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
  if (lines.length === 0) return null;
  const headers = parseCsvLine(lines[0]).map((value) => value.trim());
  const targetIndex = headers.indexOf(columnName);
  if (targetIndex < 0) return null;
  const wanted = String(targetValue || "").trim().toUpperCase();
  if (!wanted) return null;
  for (const line of lines.slice(1)) {
    const cols = parseCsvLine(line);
    if (String(cols[targetIndex] || "").trim().toUpperCase() !== wanted) continue;
    const row = {};
    for (let i = 0; i < headers.length; i += 1) {
      row[headers[i]] = cols[i] ?? "";
    }
    return row;
  }
  return null;
}

function median(values) {
  const numbers = (Array.isArray(values) ? values : [])
    .map((value) => Number(value))
    .filter(Number.isFinite)
    .sort((a, b) => a - b);
  if (numbers.length === 0) return Number.NaN;
  const mid = Math.floor(numbers.length / 2);
  return numbers.length % 2 === 1 ? numbers[mid] : (numbers[mid - 1] + numbers[mid]) / 2;
}

function flattenOntologyLabels(items, limit = 5) {
  const values = Array.isArray(items) ? items : [];
  return values
    .map((item) => normalizeWhitespace(item?.label || item?.ontology_term_id || item?.id || ""))
    .filter(Boolean)
    .slice(0, limit);
}

function normalizeOntologyPrefix(value) {
  const raw = normalizeWhitespace(value || "");
  if (!raw) return "";
  const upper = raw.toUpperCase();
  if (upper === "MESH") return "MeSH";
  if (upper === "NCBITAXON") return "NCBITaxon";
  return upper;
}

function normalizeOntologyCurie(value) {
  const raw = normalizeWhitespace(value || "");
  const match = raw.match(/^([^:]+):(.*)$/);
  if (!match) return raw;
  return `${normalizeOntologyPrefix(match[1])}:${match[2]}`;
}

function extractCuriePrefix(curie) {
  const match = normalizeWhitespace(curie || "").match(/^([^:]+):/);
  return match ? normalizeOntologyPrefix(match[1]) : "";
}

function getOlsOntologyNameForPrefix(prefix) {
  const normalized = normalizeOntologyPrefix(prefix);
  const mapping = {
    DOID: "doid",
    EFO: "efo",
    MeSH: "mesh",
    MONDO: "mondo",
    NCIT: "ncit",
    OMIM: "omim",
    UBERON: "uberon",
  };
  return mapping[normalized] || "";
}

function buildCellxgeneDatasetText(dataset) {
  const parts = [
    normalizeWhitespace(dataset?.title || ""),
    normalizeWhitespace(dataset?.collection_name || ""),
    normalizeWhitespace(dataset?.description || ""),
    normalizeWhitespace(dataset?.summary_citation || ""),
    flattenOntologyLabels(dataset?.organism, 5).join(" "),
    flattenOntologyLabels(dataset?.disease, 8).join(" "),
    flattenOntologyLabels(dataset?.tissue, 8).join(" "),
    flattenOntologyLabels(dataset?.cell_type, 12).join(" "),
    flattenOntologyLabels(dataset?.assay, 5).join(" "),
  ];
  return normalizeWhitespace(parts.join(" ").toLowerCase());
}

function matchesAllTokens(text, tokens) {
  const haystack = String(text || "").toLowerCase();
  return tokens.every((token) => haystack.includes(token));
}

function myGeneHitScore(hit, normalizedQuery) {
  if (!hit || typeof hit !== "object") return Number.NEGATIVE_INFINITY;
  let score = safeNumber(hit?._score, 0);
  const symbol = normalizeWhitespace(hit?.symbol || "").toLowerCase();
  const name = normalizeWhitespace(hit?.name || "").toLowerCase();
  const aliases = Array.isArray(hit?.alias)
    ? hit.alias.map((value) => normalizeWhitespace(value).toLowerCase())
    : normalizeWhitespace(hit?.alias || "")
      ? [normalizeWhitespace(hit.alias).toLowerCase()]
      : [];
  if (symbol === normalizedQuery) score += 100;
  if (aliases.includes(normalizedQuery)) score += 60;
  if (name === normalizedQuery) score += 25;
  if (safeNumber(hit?.taxid, 0) === 9606) score += 10;
  return score;
}

function selectBestMyGeneHit(hits, query) {
  const rows = Array.isArray(hits) ? hits : [];
  const normalizedQuery = normalizeWhitespace(query || "").toLowerCase();
  return rows
    .slice()
    .sort((a, b) => myGeneHitScore(b, normalizedQuery) - myGeneHitScore(a, normalizedQuery))[0] || null;
}

function normalizeMyGeneIds(hit) {
  const ensemblGenes = [];
  const ensembl = hit?.ensembl;
  if (Array.isArray(ensembl)) {
    for (const entry of ensembl) {
      const gene = normalizeWhitespace(entry?.gene || "");
      if (gene) ensemblGenes.push(gene);
    }
  } else {
    const gene = normalizeWhitespace(ensembl?.gene || "");
    if (gene) ensemblGenes.push(gene);
  }
  const swissProt = hit?.uniprot?.["Swiss-Prot"];
  const swissProtIds = Array.isArray(swissProt) ? swissProt : swissProt ? [swissProt] : [];
  const trembl = hit?.uniprot?.TrEMBL;
  const tremblIds = Array.isArray(trembl) ? trembl : trembl ? [trembl] : [];
  const aliases = Array.isArray(hit?.alias) ? hit.alias : hit?.alias ? [hit.alias] : [];
  return {
    symbol: normalizeWhitespace(hit?.symbol || ""),
    name: normalizeWhitespace(hit?.name || ""),
    taxid: safeNumber(hit?.taxid, 0),
    aliases: dedupeArray(aliases.map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 20),
    entrezgene: normalizeWhitespace(hit?.entrezgene || ""),
    ensemblGenes: dedupeArray(ensemblGenes),
    swissProtIds: dedupeArray(swissProtIds.map((value) => normalizeWhitespace(value)).filter(Boolean)),
    tremblIds: dedupeArray(tremblIds.map((value) => normalizeWhitespace(value)).filter(Boolean)),
  };
}

async function queryMyGene(query, { species = "human", size = 5, fields = "" } = {}) {
  const params = new URLSearchParams({
    q: query,
    species,
    size: String(Math.max(1, Math.min(10, Math.round(size || 5)))),
  });
  if (fields) params.set("fields", fields);
  const url = `${MYGENE_API}/query?${params.toString()}`;
  return fetchJsonWithRetry(url, { retries: 1, timeoutMs: 12000, maxBackoffMs: 2500 });
}

async function resolveGeneWithMyGene(query, species = "human") {
  const data = await queryMyGene(query, {
    species,
    size: 8,
    fields: "symbol,name,alias,entrezgene,ensembl.gene,uniprot.Swiss-Prot,uniprot.TrEMBL,taxid",
  });
  const hits = Array.isArray(data?.hits) ? data.hits : [];
  return {
    hits,
    bestHit: selectBestMyGeneHit(hits, query),
  };
}

function normalizeQuickGoAspect(value) {
  const raw = String(value || "").trim().toLowerCase();
  if (!raw) return "";
  if (["biological_process", "process", "bp", "p"].includes(raw)) return "biological_process";
  if (["molecular_function", "function", "mf", "f"].includes(raw)) return "molecular_function";
  if (["cellular_component", "component", "cc", "c"].includes(raw)) return "cellular_component";
  return "";
}

function normalizeQuickGoGeneProductId(rawValue) {
  const value = normalizeWhitespace(rawValue || "");
  if (!value) return "";
  if (/^[A-Z0-9_.-]+:[A-Z0-9_.-]+$/i.test(value)) return value;
  if (/^[OPQ][0-9][A-Z0-9]{3}[0-9](-\d+)?$/i.test(value) || /^[A-NR-Z][0-9][A-Z0-9]{3}[0-9](-\d+)?$/i.test(value)) {
    return `UniProtKB:${value}`;
  }
  return value;
}

async function fetchQuickGoTerms(goIds) {
  const ids = dedupeArray((Array.isArray(goIds) ? goIds : []).map((value) => normalizeWhitespace(value)).filter(Boolean));
  if (ids.length === 0) return new Map();
  const url = `${QUICKGO_API}/ontology/go/terms/${encodeURIComponent(ids.join(","))}`;
  const data = await fetchJsonWithRetry(url, {
    headers: { Accept: "application/json" },
    retries: 1,
    timeoutMs: 15000,
    maxBackoffMs: 2500,
  });
  const rows = Array.isArray(data?.results) ? data.results : [];
  return new Map(
    rows.map((row) => [
      normalizeWhitespace(row?.id || ""),
      {
        name: normalizeWhitespace(row?.name || ""),
        aspect: normalizeWhitespace(row?.aspect || ""),
        definition: normalizeWhitespace(row?.definition?.text || ""),
      },
    ]).filter(([id]) => Boolean(id))
  );
}

async function fetchDepMapDownloadCatalog() {
  const cached = getFreshCacheValue(depMapDownloadCatalogCache, 6 * 60 * 60 * 1000);
  if (cached) return cached;
  const response = await fetchWithRetry(`${DEPMAP_PORTAL_API}/api/download/files`, {
    retries: 1,
    timeoutMs: 20000,
    maxBackoffMs: 3000,
  });
  const text = await response.text();
  const rows = parseCsvObjects(text);
  depMapDownloadCatalogCache = storeCacheValue(depMapDownloadCatalogCache, rows);
  return rows;
}

async function fetchDepMapSummaryTable() {
  const cached = getFreshCacheValue(depMapSummaryCache, 6 * 60 * 60 * 1000);
  if (cached) return cached;
  const response = await fetchWithRetry(`${DEPMAP_PORTAL_API}/tda/table_download`, {
    retries: 1,
    timeoutMs: 25000,
    maxBackoffMs: 3000,
  });
  const text = await response.text();
  depMapSummaryCache = storeCacheValue(depMapSummaryCache, text);
  return text;
}

const DEPMAP_EXPRESSION_DATASET_FILES = {
  protein_coding_stranded: "OmicsExpressionTPMLogp1HumanProteinCodingGenesStranded.csv",
  protein_coding: "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv",
  all_genes_stranded: "OmicsExpressionTPMLogp1HumanAllGenesStranded.csv",
  all_genes: "OmicsExpressionTPMLogp1HumanAllGenes.csv",
};

function normalizeDepMapReleaseQuery(value) {
  const text = normalizeWhitespace(value || "").toUpperCase();
  if (!text) return "";
  const quarterMatch = text.match(/\b(\d{2}Q[1-4])\b/);
  if (quarterMatch) return quarterMatch[1];
  return text;
}

function depMapCatalogRowMatchesRelease(row, releaseQuery = "") {
  const normalizedQuery = normalizeDepMapReleaseQuery(releaseQuery);
  if (!normalizedQuery) return true;
  return normalizeWhitespace(row?.release || "").toUpperCase().includes(normalizedQuery);
}

function findDepMapCatalogRow(catalogRows, filename, releaseQuery = "") {
  const wantedFilename = normalizeWhitespace(filename || "");
  if (!wantedFilename) return null;
  const rows = (Array.isArray(catalogRows) ? catalogRows : []).filter(
    (row) => normalizeWhitespace(row?.filename || "") === wantedFilename
  );
  if (rows.length === 0) return null;
  const releaseMatches = rows.filter((row) => depMapCatalogRowMatchesRelease(row, releaseQuery));
  return (releaseMatches[0] || rows[0] || null);
}

async function fetchDepMapSubtypeTree(releaseQuery = "") {
  const cacheKey = normalizeDepMapReleaseQuery(releaseQuery) || "LATEST";
  const cached = getFreshCacheValue(depMapSubtypeTreeCache.get(cacheKey), 6 * 60 * 60 * 1000);
  if (cached) return cached;

  const catalogRows = await fetchDepMapDownloadCatalog();
  const catalogRow = findDepMapCatalogRow(catalogRows, "SubtypeTree.csv", releaseQuery);
  if (!catalogRow?.url) {
    throw new Error(`No DepMap SubtypeTree.csv file was found for release "${releaseQuery || "latest"}".`);
  }

  const response = await fetchWithRetry(catalogRow.url, {
    retries: 1,
    timeoutMs: 45000,
    maxBackoffMs: 5000,
    headers: {
      Accept: "text/csv",
      "User-Agent": "Mozilla/5.0 research-mcp",
    },
  });
  const rows = parseCsvObjects(await response.text());
  const payload = {
    rows,
    release: normalizeWhitespace(catalogRow.release || cacheKey || "latest"),
    sourceUrl: catalogRow.url,
  };
  depMapSubtypeTreeCache.set(cacheKey, storeCacheValue(null, payload));
  return payload;
}

async function fetchDepMapSubtypeMatrix(releaseQuery = "") {
  const cacheKey = normalizeDepMapReleaseQuery(releaseQuery) || "LATEST";
  const cached = getFreshCacheValue(depMapSubtypeMatrixCache.get(cacheKey), 6 * 60 * 60 * 1000);
  if (cached) return cached;

  const catalogRows = await fetchDepMapDownloadCatalog();
  const catalogRow = findDepMapCatalogRow(catalogRows, "SubtypeMatrix.csv", releaseQuery);
  if (!catalogRow?.url) {
    throw new Error(`No DepMap SubtypeMatrix.csv file was found for release "${releaseQuery || "latest"}".`);
  }

  const response = await fetchWithRetry(catalogRow.url, {
    retries: 1,
    timeoutMs: 45000,
    maxBackoffMs: 5000,
    headers: {
      Accept: "text/csv",
      "User-Agent": "Mozilla/5.0 research-mcp",
    },
  });
  const text = await response.text();
  const lines = String(text || "").split(/\r?\n/).filter((line) => line.trim().length > 0);
  const header = lines.length > 0 ? parseCsvLine(lines[0]).map((value) => normalizeWhitespace(value)) : [];
  const payload = {
    text,
    lines,
    header,
    release: normalizeWhitespace(catalogRow.release || cacheKey || "latest"),
    sourceUrl: catalogRow.url,
  };
  depMapSubtypeMatrixCache.set(cacheKey, storeCacheValue(null, payload));
  return payload;
}

function normalizeDepMapSubtypeLookupToken(value) {
  return normalizeWhitespace(value || "")
    .toLowerCase()
    .replace(/loss of function/g, "loss")
    .replace(/\blof\b/g, "loss")
    .replace(/[^a-z0-9]+/g, "");
}

function buildDepMapSubtypeAliasLookup(treeRows) {
  const lookup = new Map();
  const addAlias = (alias, code) => {
    const normalizedAlias = normalizeDepMapSubtypeLookupToken(alias);
    const normalizedCode = normalizeWhitespace(code || "");
    if (!normalizedAlias || !normalizedCode || lookup.has(normalizedAlias)) {
      return;
    }
    lookup.set(normalizedAlias, normalizedCode);
  };

  for (const row of Array.isArray(treeRows) ? treeRows : []) {
    const canonicalCode = normalizeWhitespace(
      row?.MolecularSubtypeCode
      || row?.DepmapModelType
      || row?.NodeName
      || row?.DisplayName
      || ""
    );
    if (!canonicalCode) continue;
    addAlias(canonicalCode, canonicalCode);
    addAlias(row?.NodeName, canonicalCode);
    addAlias(row?.DisplayName, canonicalCode);
    addAlias(row?.DepmapModelType, canonicalCode);
    addAlias(row?.MolecularSubtypeCode, canonicalCode);
    if (/_LoF$/i.test(canonicalCode)) {
      addAlias(canonicalCode.replace(/_LoF$/i, "Loss"), canonicalCode);
      addAlias(canonicalCode.replace(/_LoF$/i, " Loss"), canonicalCode);
      addAlias(canonicalCode.replace(/_LoF$/i, " loss"), canonicalCode);
    }
  }

  return lookup;
}

function resolveDepMapSubtypeCode(subtypeQuery, treeRows, headerColumns = []) {
  const rawQuery = normalizeWhitespace(subtypeQuery || "");
  if (!rawQuery) {
    return { code: "", matchedBy: "", query: rawQuery };
  }

  const queryToken = normalizeDepMapSubtypeLookupToken(rawQuery);
  const headerLookup = new Map(
    (Array.isArray(headerColumns) ? headerColumns : [])
      .map((value) => [normalizeDepMapSubtypeLookupToken(value), normalizeWhitespace(value)])
      .filter(([token, value]) => token && value)
  );
  const aliasLookup = buildDepMapSubtypeAliasLookup(treeRows);

  let code = headerLookup.get(queryToken) || aliasLookup.get(queryToken) || "";
  if (!code && queryToken.endsWith("loss")) {
    const lofToken = `${queryToken.slice(0, -4)}lof`;
    code = headerLookup.get(lofToken) || aliasLookup.get(lofToken) || "";
  }
  if (!code && queryToken.endsWith("lof")) {
    const lossToken = `${queryToken.slice(0, -3)}loss`;
    code = headerLookup.get(lossToken) || aliasLookup.get(lossToken) || "";
  }

  return {
    code: normalizeWhitespace(code || ""),
    matchedBy: code ? "alias" : "",
    query: rawQuery,
  };
}

function resolveDepMapExpressionGeneColumn(headers, geneSymbol) {
  const query = normalizeWhitespace(geneSymbol || "").toUpperCase();
  if (!query) return null;
  const candidates = [];

  (Array.isArray(headers) ? headers : []).forEach((header, index) => {
    const label = normalizeWhitespace(header || "");
    if (!label) return;
    const bare = label.replace(/\s*\([^)]*\)\s*$/, "").trim().toUpperCase();
    const upper = label.toUpperCase();
    let score = -1;
    if (upper === query) score = 120;
    else if (bare === query) score = 110;
    else if (upper.startsWith(`${query} (`)) score = 100;
    if (score >= 0) {
      candidates.push({ index, label, score });
    }
  });

  candidates.sort((a, b) => b.score - a.score || a.index - b.index);
  return candidates[0] || null;
}

async function computeDepMapExpressionSubsetMeanFromResponse(response, {
  modelIds,
  geneSymbol,
  defaultProfileOnly = true,
}) {
  const reader = response?.body?.getReader?.();
  if (!reader) {
    throw new Error("DepMap expression response body is not stream-readable.");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let header = null;
  let modelIndex = -1;
  let defaultModelIndex = -1;
  let geneColumn = null;
  let valueCount = 0;
  let valueSum = 0;
  const matchedModels = new Set();

  const processLine = (rawLine) => {
    let line = String(rawLine || "");
    if (!line) return;
    if (line.endsWith("\r")) line = line.slice(0, -1);
    if (!header) {
      header = parseCsvLine(line).map((value) => normalizeWhitespace(value));
      modelIndex = header.indexOf("ModelID");
      defaultModelIndex = header.indexOf("IsDefaultEntryForModel");
      geneColumn = resolveDepMapExpressionGeneColumn(header, geneSymbol);
      if (modelIndex < 0 || !geneColumn) {
        throw new Error(`Could not resolve ModelID or gene column for ${geneSymbol} in the selected DepMap expression file.`);
      }
      return;
    }

    const cols = parseCsvLine(line);
    const modelId = normalizeWhitespace(cols[modelIndex] || "");
    if (!modelId || !modelIds.has(modelId)) {
      return;
    }
    if (defaultProfileOnly && defaultModelIndex >= 0) {
      const isDefault = ["1", "1.0", "true", "yes"].includes(String(cols[defaultModelIndex] || "").trim().toLowerCase());
      if (!isDefault) {
        return;
      }
    }
    const value = toFiniteNumber(cols[geneColumn.index], Number.NaN);
    if (!Number.isFinite(value)) {
      return;
    }
    valueSum += value;
    valueCount += 1;
    matchedModels.add(modelId);
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      processLine(buffer.slice(0, newlineIndex));
      buffer = buffer.slice(newlineIndex + 1);
      newlineIndex = buffer.indexOf("\n");
    }

    if (done) {
      const tail = decoder.decode();
      if (tail) buffer += tail;
      break;
    }
  }

  if (buffer.trim().length > 0) {
    processLine(buffer);
  }

  return {
    geneColumn,
    valueCount,
    valueSum,
    matchedModelCount: matchedModels.size,
  };
}

function buildDepMapExpressionSubsetMeanCacheKey({
  geneSymbol,
  subtypeCode,
  release,
  expressionDataset,
  defaultProfileOnly,
}) {
  return [
    normalizeWhitespace(geneSymbol || "").toUpperCase(),
    normalizeWhitespace(subtypeCode || ""),
    normalizeDepMapReleaseQuery(release || "") || "LATEST",
    normalizeWhitespace(expressionDataset || ""),
    defaultProfileOnly ? "default" : "all",
  ].join("::");
}

function extractDepMapEntityId(characterizationRows) {
  const rows = Array.isArray(characterizationRows) ? characterizationRows : [];
  for (const row of rows) {
    const ajaxUrl = normalizeWhitespace(row?.sublineage_plot?.ajax_url || "");
    const match = ajaxUrl.match(/[?&]entity_id=(\d+)/);
    if (match) return match[1];
  }
  return "";
}

function normalizeCellxgeneDatasetFromCollection(collection, dataset) {
  const datasetId = normalizeWhitespace(dataset?.id || dataset?.dataset_id || "");
  const collectionId = normalizeWhitespace(collection?.id || dataset?.collection_id || "");
  return {
    dataset_id: datasetId,
    collection_id: collectionId,
    title: normalizeWhitespace(dataset?.name || collection?.name || "Untitled dataset"),
    collection_name: normalizeWhitespace(collection?.name || ""),
    description: normalizeWhitespace(collection?.description || ""),
    summary_citation: normalizeWhitespace(collection?.summary_citation || ""),
    organism: Array.isArray(dataset?.organism) ? dataset.organism : [],
    disease: Array.isArray(dataset?.disease) ? dataset.disease : [],
    tissue: Array.isArray(dataset?.tissue) ? dataset.tissue : [],
    cell_type: Array.isArray(dataset?.cell_type) ? dataset.cell_type : [],
    assay: Array.isArray(dataset?.assay) ? dataset.assay : [],
    primary_cell_count: toNonNegativeInt(dataset?.primary_cell_count, 0),
    cell_count: toNonNegativeInt(dataset?.cell_count, 0),
    explorer_url: normalizeWhitespace(dataset?.dataset_deployments?.[0]?.url || ""),
  };
}

async function fetchCellxgeneDatasets() {
  const cached = getFreshCacheValue(cellxgeneDatasetCache, 60 * 60 * 1000);
  if (cached) return cached;

  const data = await fetchJsonWithRetry(`${CELLXGENE_DISCOVER_API}/collections?visibility=PUBLIC`, {
    retries: 1,
    timeoutMs: 15000,
    maxBackoffMs: 3000,
  });
  const collections = Array.isArray(data?.collections) ? data.collections : [];
  const collectionIds = dedupeArray(collections.map((row) => normalizeWhitespace(row?.id || "")).filter(Boolean));
  const allDatasets = [];

  for (let start = 0; start < collectionIds.length; start += 15) {
    const batch = collectionIds.slice(start, start + 15);
    const results = await Promise.allSettled(
      batch.map((collectionId) =>
        fetchJsonWithRetry(`${CELLXGENE_DISCOVER_API}/collections/${encodeURIComponent(collectionId)}`, {
          retries: 1,
          timeoutMs: 12000,
          maxBackoffMs: 2500,
        })
      )
    );

    for (const result of results) {
      if (result.status !== "fulfilled") continue;
      const collection = result.value;
      const datasets = Array.isArray(collection?.datasets) ? collection.datasets : [];
      for (const dataset of datasets) {
        allDatasets.push(normalizeCellxgeneDatasetFromCollection(collection, dataset));
      }
    }
  }

  const seenDatasetIds = new Set();
  const rows = [];
  for (const dataset of allDatasets) {
    const datasetId = normalizeWhitespace(dataset?.dataset_id || "");
    if (datasetId && seenDatasetIds.has(datasetId)) continue;
    if (datasetId) seenDatasetIds.add(datasetId);
    rows.push(dataset);
  }

  cellxgeneDatasetCache = storeCacheValue(cellxgeneDatasetCache, rows);
  return rows;
}

function normalizeCellxgeneOntologyOptions(items) {
  const rows = [];
  for (const item of Array.isArray(items) ? items : []) {
    if (!item || typeof item !== "object") continue;
    for (const [rawId, rawLabel] of Object.entries(item)) {
      const id = normalizeOntologyCurie(rawId);
      const label = normalizeWhitespace(rawLabel || rawId || "");
      if (!id || !label) continue;
      rows.push({ id, label });
    }
  }
  return rows;
}

function normalizeCellxgeneMatchText(value) {
  return normalizeWhitespace(value || "")
    .toLowerCase()
    .replace(/\bcells\b/g, "cell")
    .replace(/\bgenes\b/g, "gene")
    .replace(/\s+/g, " ")
    .trim();
}

function scoreCellxgeneOntologyOption(option, query) {
  if (!option || typeof option !== "object") return Number.NEGATIVE_INFINITY;
  const normalizedQuery = normalizeCellxgeneMatchText(query || "");
  if (!normalizedQuery) return Number.NEGATIVE_INFINITY;
  const queryTokens = tokenizeQuery(normalizedQuery);
  const label = normalizeCellxgeneMatchText(option?.label || "");
  const id = normalizeWhitespace(option?.id || "").toLowerCase();
  let score = 0;
  if (label === normalizedQuery) score += 1000;
  if (id === normalizedQuery) score += 950;
  if (label.replace(/[-_]/g, " ") === normalizedQuery.replace(/[-_]/g, " ")) score += 900;
  if (label.includes(normalizedQuery)) score += 200;
  if (normalizedQuery.includes(label)) score += 100;
  if (queryTokens.length > 0 && queryTokens.every((token) => label.includes(token))) {
    score += 120 + queryTokens.length;
  }
  if (label.startsWith(normalizedQuery)) score += 40;
  return score;
}

function resolveCellxgeneOntologyOption(items, query) {
  const options = normalizeCellxgeneOntologyOptions(items);
  if (!query) return null;
  const sorted = options
    .slice()
    .sort((a, b) => scoreCellxgeneOntologyOption(b, query) - scoreCellxgeneOntologyOption(a, query));
  if (sorted.length === 0) return null;
  const bestScore = scoreCellxgeneOntologyOption(sorted[0], query);
  return bestScore > 0 ? sorted[0] : null;
}

function buildCellxgeneWmgFiltersCacheKey(filterPayload) {
  try {
    return JSON.stringify(filterPayload || {});
  } catch {
    return String(filterPayload || "");
  }
}

async function fetchCellxgeneWmgPrimaryDimensions() {
  const cached = getFreshCacheValue(cellxgeneWmgPrimaryDimensionsCache, 60 * 60 * 1000);
  if (cached) return cached;
  const data = await fetchJsonWithRetry(`${CELLXGENE_WMG_API}/primary_filter_dimensions`, {
    retries: 1,
    timeoutMs: 25000,
    maxBackoffMs: 3000,
  });
  cellxgeneWmgPrimaryDimensionsCache = storeCacheValue(cellxgeneWmgPrimaryDimensionsCache, data);
  return data;
}

async function fetchCellxgeneWmgFilters(filterPayload) {
  const cacheKey = buildCellxgeneWmgFiltersCacheKey(filterPayload);
  const cached = getFreshCacheValue(cellxgeneWmgFiltersCache.get(cacheKey), 30 * 60 * 1000);
  if (cached) return cached;
  const data = await fetchJsonWithRetry(`${CELLXGENE_WMG_API}/filters`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ filter: filterPayload || {} }),
    retries: 1,
    timeoutMs: 25000,
    maxBackoffMs: 3000,
  });
  cellxgeneWmgFiltersCache.set(cacheKey, storeCacheValue(null, data));
  return data;
}

async function fetchCellxgeneDeFilters(filterPayload) {
  const cacheKey = buildCellxgeneWmgFiltersCacheKey(filterPayload);
  const cached = getFreshCacheValue(cellxgeneDeFiltersCache.get(cacheKey), 30 * 60 * 1000);
  if (cached) return cached;
  const data = await fetchJsonWithRetry(`${CELLXGENE_DE_API}/filters`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({ filter: filterPayload || {} }),
    retries: 1,
    timeoutMs: 25000,
    maxBackoffMs: 3000,
  });
  cellxgeneDeFiltersCache.set(cacheKey, storeCacheValue(null, data));
  return data;
}

async function fetchCellxgeneMarkerGenes({ cellTypeId, organismId, tissueId, nMarkers = 10, test = "ttest" }) {
  return fetchJsonWithRetry(`${CELLXGENE_WMG_API}/markers`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify({
      celltype: cellTypeId,
      organism: organismId,
      tissue: tissueId,
      n_markers: Math.max(1, Math.min(50, Math.round(nMarkers || 10))),
      test: normalizeWhitespace(test || "ttest") || "ttest",
    }),
    retries: 1,
    timeoutMs: 25000,
    maxBackoffMs: 3000,
  });
}

function buildCellxgeneGeneSymbolMap(primaryDimensions, organismId) {
  const rows = Array.isArray(primaryDimensions?.gene_terms?.[organismId]) ? primaryDimensions.gene_terms[organismId] : [];
  const out = new Map();
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    for (const [geneId, geneSymbol] of Object.entries(row)) {
      const normalizedId = normalizeWhitespace(geneId || "");
      const symbol = normalizeWhitespace(geneSymbol || geneId || "");
      if (normalizedId && symbol) {
        out.set(normalizedId, symbol);
      }
    }
  }
  return out;
}

function normalizeGdscLookupText(value) {
  return normalizeWhitespace(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

async function fetchGdscCompoundCatalog() {
  const cached = getFreshCacheValue(gdscCompoundCache, 12 * 60 * 60 * 1000);
  if (cached) return cached;

  const response = await fetchWithRetry(`${CANCERRXGENE_API}/compounds?list=all`, {
    retries: 1,
    timeoutMs: 20000,
    maxBackoffMs: 3000,
  });
  const text = await response.text();
  const rows = parseDelimitedObjects(text, "\t")
    .map((row) => {
      const synonyms = dedupeArray(
        normalizeWhitespace(row?.Synonyms || "")
          .split(",")
          .map((value) => normalizeWhitespace(value))
          .filter(Boolean)
      );
      const targets = dedupeArray(
        normalizeWhitespace(row?.Targets || "")
          .split(",")
          .map((value) => normalizeWhitespace(value))
          .filter(Boolean)
      );
      return {
        drug_id: normalizeWhitespace(row?.["Drug Id"] || ""),
        name: normalizeWhitespace(row?.Name || ""),
        synonyms,
        targets,
        target_pathway: normalizeWhitespace(row?.["Target pathway"] || ""),
        pubchem: normalizeWhitespace(row?.PubCHEM || ""),
        dataset: normalizeWhitespace(row?.Datasets || ""),
        screening_site: normalizeWhitespace(row?.["Screening site"] || ""),
        cell_line_count: toNonNegativeInt(row?.["number of cell lines"], 0),
      };
    })
    .filter((row) => row.drug_id && row.name);

  gdscCompoundCache = storeCacheValue(gdscCompoundCache, rows);
  return rows;
}

function scoreGdscCompoundMatch(row, query) {
  const normalizedQuery = normalizeGdscLookupText(query);
  if (!normalizedQuery) return 0;

  const normalizedName = normalizeGdscLookupText(row?.name || "");
  const normalizedSynonyms = (Array.isArray(row?.synonyms) ? row.synonyms : []).map(normalizeGdscLookupText);
  const queryTokens = normalizedQuery.split(/\s+/).filter(Boolean);

  if (normalizedQuery === normalizedName) return 120;
  if (normalizedSynonyms.includes(normalizedQuery)) return 115;
  if (normalizedName.startsWith(normalizedQuery)) return 100;
  if (normalizedSynonyms.some((value) => value.startsWith(normalizedQuery))) return 95;
  if (normalizedName.includes(normalizedQuery)) return 80;
  if (normalizedSynonyms.some((value) => value.includes(normalizedQuery))) return 75;
  if (queryTokens.length > 1 && queryTokens.every((token) => normalizedName.includes(token))) return 65;
  if (queryTokens.length > 1 && normalizedSynonyms.some((value) => queryTokens.every((token) => value.includes(token)))) return 60;
  return 0;
}

function resolveGdscCompoundSelection(catalogRows, rawQuery, datasetFilter = "all") {
  const rows = Array.isArray(catalogRows) ? catalogRows : [];
  const query = normalizeWhitespace(rawQuery || "");
  const requestedDataset = normalizeWhitespace(datasetFilter || "all").toUpperCase();
  const activeDataset = requestedDataset === "ALL" ? "all" : requestedDataset;

  if (!query) {
    return {
      selectedRows: [],
      matchedName: "",
      alternates: [],
      availableDatasets: [],
    };
  }

  if (/^\d+$/.test(query)) {
    const numericMatches = rows.filter((row) => row.drug_id === query);
    const filteredRows =
      activeDataset === "all" ? numericMatches : numericMatches.filter((row) => row.dataset.toUpperCase() === activeDataset);
    return {
      selectedRows: dedupeArray(filteredRows.map((row) => `${row.drug_id}::${row.dataset}`))
        .map((key) => filteredRows.find((row) => `${row.drug_id}::${row.dataset}` === key))
        .filter(Boolean),
      matchedName: filteredRows[0]?.name || numericMatches[0]?.name || query,
      alternates: [],
      availableDatasets: dedupeArray(numericMatches.map((row) => row.dataset)),
    };
  }

  const scoredRows = rows
    .map((row) => ({ row, score: scoreGdscCompoundMatch(row, query) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || b.row.cell_line_count - a.row.cell_line_count || a.row.name.localeCompare(b.row.name));

  if (scoredRows.length === 0) {
    return {
      selectedRows: [],
      matchedName: "",
      alternates: [],
      availableDatasets: [],
    };
  }

  const topName = normalizeGdscLookupText(scoredRows[0].row.name);
  const familyRows = scoredRows
    .filter((entry) => normalizeGdscLookupText(entry.row.name) === topName)
    .map((entry) => entry.row);
  const filteredRows =
    activeDataset === "all" ? familyRows : familyRows.filter((row) => row.dataset.toUpperCase() === activeDataset);
  const selectedRows = dedupeArray(filteredRows.map((row) => `${row.drug_id}::${row.dataset}`))
    .map((key) => filteredRows.find((row) => `${row.drug_id}::${row.dataset}` === key))
    .filter(Boolean);
  const alternates = dedupeArray(
    scoredRows
      .map((entry) => entry.row.name)
      .filter((name) => normalizeGdscLookupText(name) !== topName)
  ).slice(0, 5);

  return {
    selectedRows,
    matchedName: familyRows[0]?.name || "",
    alternates,
    availableDatasets: dedupeArray(familyRows.map((row) => row.dataset)),
  };
}

async function fetchGdscOverviewData(drugId, dataset) {
  const url =
    `${CANCERRXGENE_API}/compound/overview_data?id=${encodeURIComponent(drugId)}` +
    `&screening_set=${encodeURIComponent(dataset)}&tissue=`;
  const data = await fetchJsonWithRetry(url, {
    retries: 1,
    timeoutMs: 20000,
    maxBackoffMs: 3000,
    headers: { Accept: "application/json" },
  });
  return Array.isArray(data?.data) ? data.data : [];
}

function getGdscTissueLabel(record) {
  const tissueName = normalizeWhitespace(record?.tissue_name || "");
  if (tissueName && !/^no tcga classification$/i.test(tissueName)) {
    return tissueName;
  }
  const secondary = normalizeWhitespace(String(record?.gdsc_desc2 || "").replace(/_/g, " "));
  if (secondary) return secondary;
  const primary = normalizeWhitespace(String(record?.gdsc_desc1 || "").replace(/_/g, " "));
  if (primary) return primary;
  const tcga = normalizeWhitespace(record?.tcga || "");
  return tcga || "Unspecified tissue";
}

function buildGdscRecordText(record) {
  return normalizeWhitespace(
    [
      record?.cell_name,
      record?.tissue_name,
      record?.gdsc_desc1,
      record?.gdsc_desc2,
      record?.tcga,
    ].join(" ")
  ).toLowerCase();
}

function normalizePrismLookupText(value) {
  return normalizeWhitespace(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function splitPrismSynonyms(rawValue) {
  const normalized = normalizeWhitespace(rawValue || "");
  if (!normalized) return [];
  const splitter = /[;|]/;
  if (!splitter.test(normalized)) return [normalized];
  return normalized
    .split(splitter)
    .map((value) => normalizeWhitespace(value))
    .filter(Boolean);
}

async function fetchPrismCompoundCatalog() {
  const cached = getFreshCacheValue(prismCompoundCatalogCache, 12 * 60 * 60 * 1000);
  if (cached) return cached;

  const response = await fetchWithRetry(PRISM_24Q2_COMPOUND_LIST_URL, {
    retries: 1,
    timeoutMs: 45000,
    maxBackoffMs: 5000,
    headers: {
      Accept: "text/csv",
      "User-Agent": "Mozilla/5.0 research-mcp",
    },
  });
  const text = await response.text();
  const rows = parseCsvObjects(text)
    .map((row) => ({
      ids: normalizeWhitespace(row?.IDs || ""),
      name: normalizeWhitespace(row?.["Drug.Name"] || row?.drug_name || ""),
      synonyms: dedupeArray(splitPrismSynonyms(row?.Synonyms || "")),
      screen: normalizeWhitespace(row?.screen || ""),
      dose: normalizeWhitespace(row?.dose || ""),
      repurposing_target: normalizeWhitespace(row?.repurposing_target || ""),
      moa: normalizeWhitespace(row?.MOA || ""),
    }))
    .filter((row) => row.ids && row.name);

  prismCompoundCatalogCache = storeCacheValue(prismCompoundCatalogCache, rows);
  return rows;
}

function scorePrismCompoundMatch(row, query) {
  const normalizedQuery = normalizePrismLookupText(query);
  if (!normalizedQuery) return 0;

  const normalizedId = normalizeWhitespace(row?.ids || "").toUpperCase();
  const normalizedQueryUpper = normalizeWhitespace(query || "").toUpperCase();
  const normalizedName = normalizePrismLookupText(row?.name || "");
  const normalizedSynonyms = (Array.isArray(row?.synonyms) ? row.synonyms : []).map(normalizePrismLookupText);
  const queryTokens = normalizedQuery.split(/\s+/).filter(Boolean);

  if (normalizedQueryUpper && normalizedId === normalizedQueryUpper) return 160;
  if (normalizedQueryUpper && normalizedId.endsWith(normalizedQueryUpper)) return 145;
  if (normalizedQuery === normalizedName) return 130;
  if (normalizedSynonyms.includes(normalizedQuery)) return 120;
  if (normalizedName.startsWith(normalizedQuery)) return 105;
  if (normalizedSynonyms.some((value) => value.startsWith(normalizedQuery))) return 95;
  if (normalizedName.includes(normalizedQuery)) return 85;
  if (normalizedSynonyms.some((value) => value.includes(normalizedQuery))) return 80;
  if (queryTokens.length > 1 && queryTokens.every((token) => normalizedName.includes(token))) return 70;
  if (queryTokens.length > 1 && normalizedSynonyms.some((value) => queryTokens.every((token) => value.includes(token)))) return 65;
  return 0;
}

function resolvePrismCompoundSelection(catalogRows, rawQuery, screenFilter = "") {
  const rows = Array.isArray(catalogRows) ? catalogRows : [];
  const query = normalizeWhitespace(rawQuery || "");
  const requestedScreen = normalizeWhitespace(screenFilter || "").toUpperCase();

  if (!query) {
    return {
      selectedRows: [],
      matchedName: "",
      alternates: [],
      availableScreens: [],
    };
  }

  if (/^BRD[:\-]/i.test(query)) {
    const normalizedQuery = query.toUpperCase();
    const directMatches = rows.filter((row) => {
      const prismId = normalizeWhitespace(row?.ids || "").toUpperCase();
      return prismId === normalizedQuery || prismId.endsWith(normalizedQuery);
    });
    const filteredMatches =
      requestedScreen
        ? directMatches.filter((row) => row.screen.toUpperCase() === requestedScreen)
        : directMatches;
    return {
      selectedRows: filteredMatches,
      matchedName: filteredMatches[0]?.name || directMatches[0]?.name || query,
      alternates: [],
      availableScreens: dedupeArray(directMatches.map((row) => row.screen)),
    };
  }

  const scoredRows = rows
    .map((row) => ({ row, score: scorePrismCompoundMatch(row, query) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || a.row.name.localeCompare(b.row.name));

  if (scoredRows.length === 0) {
    return {
      selectedRows: [],
      matchedName: "",
      alternates: [],
      availableScreens: [],
    };
  }

  const topName = normalizePrismLookupText(scoredRows[0].row.name);
  const familyRows = scoredRows
    .filter((entry) => normalizePrismLookupText(entry.row.name) === topName)
    .map((entry) => entry.row);
  const filteredRows =
    requestedScreen
      ? familyRows.filter((row) => row.screen.toUpperCase() === requestedScreen)
      : familyRows;
  const alternates = dedupeArray(
    scoredRows
      .map((entry) => entry.row.name)
      .filter((name) => normalizePrismLookupText(name) !== topName)
  ).slice(0, 5);

  return {
    selectedRows: dedupeArray(filteredRows.map((row) => row.ids))
      .map((id) => filteredRows.find((row) => row.ids === id))
      .filter(Boolean),
    matchedName: familyRows[0]?.name || "",
    alternates,
    availableScreens: dedupeArray(familyRows.map((row) => row.screen)).filter(Boolean),
  };
}

function inferPrismTissueLabel(ccleName) {
  const normalized = normalizeWhitespace(ccleName || "");
  if (!normalized) return "Unspecified tissue";
  const idx = normalized.lastIndexOf("_");
  if (idx < 0 || idx === normalized.length - 1) return normalized;
  return normalizeWhitespace(normalized.slice(idx + 1).replace(/_/g, " ")) || "Unspecified tissue";
}

function inferPrismCellLineName(ccleName, depmapId = "") {
  const normalized = normalizeWhitespace(ccleName || "");
  if (!normalized) return normalizeWhitespace(depmapId || "Unknown cell line");
  const idx = normalized.lastIndexOf("_");
  if (idx <= 0) return normalized;
  return normalizeWhitespace(normalized.slice(0, idx).replace(/_/g, "-")) || normalized;
}

async function fetchPrismCellLineCatalog() {
  const cached = getFreshCacheValue(prismCellLineCatalogCache, 12 * 60 * 60 * 1000);
  if (cached) return cached;

  const response = await fetchWithRetry(PRISM_24Q2_CELL_LINE_METADATA_URL, {
    retries: 1,
    timeoutMs: 45000,
    maxBackoffMs: 5000,
    headers: {
      Accept: "text/csv",
      "User-Agent": "Mozilla/5.0 research-mcp",
    },
  });
  const text = await response.text();
  const seen = new Set();
  const rows = parseCsvObjects(text)
    .map((row) => {
      const depmapId = normalizeWhitespace(row?.depmap_id || "");
      const ccleName = normalizeWhitespace(row?.ccle_name || "");
      return {
        depmap_id: depmapId,
        ccle_name: ccleName,
        cell_line_name: inferPrismCellLineName(ccleName, depmapId),
        tissue_name: inferPrismTissueLabel(ccleName),
        screen: normalizeWhitespace(row?.screen || ""),
      };
    })
    .filter((row) => row.depmap_id && row.ccle_name)
    .filter((row) => {
      if (seen.has(row.depmap_id)) return false;
      seen.add(row.depmap_id);
      return true;
    });

  prismCellLineCatalogCache = storeCacheValue(prismCellLineCatalogCache, rows);
  return rows;
}

async function fetchPrismPrimaryMatrix() {
  const cached = getFreshCacheValue(prismMatrixCache, 6 * 60 * 60 * 1000);
  if (cached) return cached;

  const response = await fetchWithRetry(PRISM_24Q2_PRIMARY_MATRIX_URL, {
    retries: 1,
    timeoutMs: 120000,
    maxBackoffMs: 6000,
    headers: {
      Accept: "text/csv",
      "User-Agent": "Mozilla/5.0 research-mcp",
    },
  });
  const text = await response.text();
  const newlineIndex = text.indexOf("\n");
  const headerLine = (newlineIndex >= 0 ? text.slice(0, newlineIndex) : text).replace(/\r$/, "");
  const depmapIds = parseCsvLine(headerLine).slice(1).map((value) => normalizeWhitespace(value)).filter(Boolean);
  const payload = { text, depmapIds };
  prismMatrixCache = storeCacheValue(prismMatrixCache, payload);
  return payload;
}

function extractPrismMatrixRows(matrixPayload, targetIds) {
  const wanted = new Set(
    (Array.isArray(targetIds) ? targetIds : [])
      .map((value) => normalizeWhitespace(value))
      .filter(Boolean)
  );
  const rowMap = new Map();
  if (wanted.size === 0) return rowMap;

  const text = String(matrixPayload?.text || "");
  let cursor = 0;
  let lineNumber = 0;

  while (cursor <= text.length) {
    let nextNewline = text.indexOf("\n", cursor);
    if (nextNewline < 0) nextNewline = text.length;
    let line = text.slice(cursor, nextNewline);
    if (line.endsWith("\r")) line = line.slice(0, -1);

    if (lineNumber > 0 && line) {
      const firstComma = line.indexOf(",");
      const rowId = normalizeWhitespace(firstComma >= 0 ? line.slice(0, firstComma) : line);
      if (wanted.has(rowId)) {
        rowMap.set(
          rowId,
          parseCsvLine(line).slice(1).map((value) => toNullableNumber(value))
        );
        if (rowMap.size >= wanted.size) break;
      }
    }

    if (nextNewline >= text.length) break;
    cursor = nextNewline + 1;
    lineNumber += 1;
  }

  return rowMap;
}

function buildPrismRecordText(record) {
  return normalizeWhitespace(
    [
      record?.cell_name,
      record?.ccle_name,
      record?.tissue_name,
      record?.depmap_id,
    ].join(" ")
  ).toLowerCase();
}

function normalizePharmacoDbLookupText(value) {
  return normalizeWhitespace(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

async function fetchPharmacoDbGraphQL(query, variables = {}, timeoutMs = 45000) {
  const payload = await fetchJsonWithRetry(PHARMACODB_GRAPHQL_API, {
    retries: 1,
    timeoutMs,
    maxBackoffMs: 4000,
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json",
    },
    body: JSON.stringify({ query, variables }),
  });
  return requireGraphQLData(payload, "PharmacoDB GraphQL");
}

function scorePharmacoDbSearchHit(hit, rawQuery) {
  const normalizedQuery = normalizePharmacoDbLookupText(rawQuery);
  if (!normalizedQuery) return 0;

  const value = normalizeWhitespace(hit?.value || "");
  const uid = normalizeWhitespace(hit?.id || "");
  const type = normalizeWhitespace(hit?.type || "").toLowerCase();
  const normalizedValue = normalizePharmacoDbLookupText(value);
  const normalizedUid = uid.toUpperCase();
  const rawUpper = normalizeWhitespace(rawQuery || "").toUpperCase();
  const queryTokens = normalizedQuery.split(/\s+/).filter(Boolean);
  let score = type === "compound" ? 25 : 0;

  if (rawUpper && normalizedUid === rawUpper) score += 160;
  if (normalizedQuery === normalizedValue) score += 130;
  if (normalizedValue.startsWith(normalizedQuery)) score += 105;
  if (normalizedValue.includes(normalizedQuery)) score += 85;
  if (queryTokens.length > 1 && queryTokens.every((token) => normalizedValue.includes(token))) score += 70;
  return score;
}

async function resolvePharmacoDbCompound(rawQuery) {
  const query = normalizeWhitespace(rawQuery || "");
  if (!query) {
    return {
      compound: null,
      alternates: [],
      matchedLabel: "",
    };
  }

  const compoundQuery = `
    query($uid: String!) {
      compound(compoundUID: $uid) {
        compound {
          id
          name
          uid
          annotation {
            smiles
            chembl
            pubchem
            fda_status
          }
        }
      }
    }
  `;

  if (/^PDBC\d+$/i.test(query)) {
    const direct = await fetchPharmacoDbGraphQL(compoundQuery, { uid: query.toUpperCase() }, 25000);
    if (direct?.compound?.compound) {
      return {
        compound: direct.compound.compound,
        alternates: [],
        matchedLabel: direct.compound.compound.name || query,
      };
    }
  }

  const searchQuery = `
    query($input: String) {
      search(input: $input) {
        id
        value
        type
      }
    }
  `;
  const searchData = await fetchPharmacoDbGraphQL(searchQuery, { input: query }, 25000);
  const compoundHits = (Array.isArray(searchData?.search) ? searchData.search : [])
    .filter((hit) => normalizeWhitespace(hit?.type || "").toLowerCase() === "compound")
    .map((hit) => ({ hit, score: scorePharmacoDbSearchHit(hit, query) }))
    .filter((entry) => entry.score > 0)
    .sort((a, b) => b.score - a.score || normalizeWhitespace(a.hit?.value || "").localeCompare(normalizeWhitespace(b.hit?.value || "")));

  if (compoundHits.length === 0) {
    return {
      compound: null,
      alternates: [],
      matchedLabel: "",
    };
  }

  const best = compoundHits[0].hit;
  const detailData = await fetchPharmacoDbGraphQL(compoundQuery, { uid: normalizeWhitespace(best.id || "") }, 25000);
  return {
    compound: detailData?.compound?.compound || null,
    alternates: dedupeArray(compoundHits.slice(1).map((entry) => normalizeWhitespace(entry.hit?.value || ""))).slice(0, 5),
    matchedLabel: normalizeWhitespace(best.value || ""),
  };
}

async function fetchPharmacoDbExperiments(compoundId, tissueName = "") {
  const experimentsQuery = `
    query($compoundId: Int!, $tissueName: String) {
      experiments(compoundId: $compoundId, tissueName: $tissueName, all: true) {
        id
        dataset {
          name
        }
        tissue {
          name
        }
        cell_line {
          name
          uid
        }
        profile {
          AAC
          IC50
          EC50
          Einf
          DSS1
          DSS2
          DSS3
          HS
        }
      }
    }
  `;
  const data = await fetchPharmacoDbGraphQL(
    experimentsQuery,
    {
      compoundId: Math.trunc(Number(compoundId)),
      tissueName: normalizeWhitespace(tissueName || "") || null,
    },
    60000
  );
  return Array.isArray(data?.experiments) ? data.experiments : [];
}

function buildPharmacoDbRecordText(record) {
  return normalizeWhitespace(
    [
      record?.dataset,
      record?.tissue_name,
      record?.cell_name,
      record?.cell_uid,
    ].join(" ")
  ).toLowerCase();
}

function normalizeIntactLookupText(value) {
  return normalizeWhitespace(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function scoreIntactInteractorCandidate(candidate, rawQuery, speciesTaxId = 9606) {
  const query = normalizeWhitespace(rawQuery || "");
  const normalizedQuery = normalizeIntactLookupText(query);
  if (!normalizedQuery) return 0;

  const taxId = toNonNegativeInt(candidate?.interactorTaxId, 0);
  const name = normalizeWhitespace(candidate?.interactorName || "");
  const preferredId = normalizeWhitespace(candidate?.interactorPreferredIdentifier || "");
  const description = normalizeWhitespace(candidate?.interactorDescription || "");
  const aliases = Array.isArray(candidate?.interactorAliasNames)
    ? candidate.interactorAliasNames
    : Array.isArray(candidate?.interactorAlias)
      ? candidate.interactorAlias
      : [];
  const normalizedName = normalizeIntactLookupText(name);
  const normalizedPreferredId = normalizeIntactLookupText(preferredId);
  const normalizedAliases = aliases.map(normalizeIntactLookupText).filter(Boolean);
  const normalizedDescription = normalizeIntactLookupText(description);
  const interactionCount = toNonNegativeInt(candidate?.interactionCount, 0);

  let score = 0;
  if (taxId > 0 && speciesTaxId > 0 && taxId === speciesTaxId) score += 30;
  if (normalizeWhitespace(candidate?.interactorType || "").toLowerCase() === "protein") score += 15;
  if (normalizedQuery === normalizedPreferredId) score += 120;
  if (normalizedQuery === normalizedName) score += 110;
  if (normalizedAliases.includes(normalizedQuery)) score += 95;
  if (normalizedName.startsWith(normalizedQuery)) score += 75;
  if (normalizedPreferredId.startsWith(normalizedQuery)) score += 70;
  if (normalizedName.includes(normalizedQuery)) score += 50;
  if (normalizedAliases.some((value) => value.includes(normalizedQuery))) score += 45;
  if (normalizedDescription.includes(normalizedQuery)) score += 20;
  score += Math.min(20, Math.log10(Math.max(1, interactionCount)) * 8);
  return score;
}

async function searchIntactInteractors(query, page = 0) {
  const url = `${INTACT_API}/interactor/findInteractor/${encodeURIComponent(query)}?page=${Math.max(0, toNonNegativeInt(page, 0))}`;
  const data = await fetchJsonWithRetry(url, {
    retries: 1,
    timeoutMs: 12000,
    maxBackoffMs: 2500,
    headers: { Accept: "application/json" },
  });
  return Array.isArray(data?.content) ? data.content : [];
}

async function resolveIntactInteractor(query, speciesTaxId = 9606) {
  const candidates = await searchIntactInteractors(query, 0);
  if (candidates.length === 0) {
    return { selected: null, candidates: [] };
  }

  const ranked = candidates
    .map((candidate) => ({ candidate, score: scoreIntactInteractorCandidate(candidate, query, speciesTaxId) }))
    .sort((a, b) => b.score - a.score || toNonNegativeInt(b.candidate?.interactionCount, 0) - toNonNegativeInt(a.candidate?.interactionCount, 0));

  return {
    selected: ranked[0]?.candidate || null,
    candidates: ranked.map((entry) => entry.candidate),
  };
}

async function fetchIntactInteractionPage(query, page = 0) {
  const url = `${INTACT_API}/interaction/findInteractions/${encodeURIComponent(query)}?page=${Math.max(0, toNonNegativeInt(page, 0))}`;
  return fetchJsonWithRetry(url, {
    retries: 1,
    timeoutMs: 15000,
    maxBackoffMs: 2500,
    headers: { Accept: "application/json" },
  });
}

function extractIntactPrimaryLabel(value, fallback = "") {
  const normalized = normalizeWhitespace(value || "");
  if (!normalized) return normalizeWhitespace(fallback || "");
  const match = normalized.match(/^(.+?)\s+\([^)]+\)$/);
  return normalizeWhitespace(match ? match[1] : normalized) || normalizeWhitespace(fallback || "");
}

function resolveIntactAnchorSide(row, anchorIdentifier) {
  const anchor = normalizeWhitespace(anchorIdentifier || "").toUpperCase();
  const sideAValues = [
    normalizeWhitespace(row?.uniqueIdA || "").toUpperCase(),
    normalizeWhitespace(row?.acA || "").toUpperCase(),
    normalizeWhitespace(row?.moleculeA || "").toUpperCase(),
    extractIntactPrimaryLabel(row?.idA || "").toUpperCase(),
  ];
  if (sideAValues.includes(anchor)) return "A";
  const sideBValues = [
    normalizeWhitespace(row?.uniqueIdB || "").toUpperCase(),
    normalizeWhitespace(row?.acB || "").toUpperCase(),
    normalizeWhitespace(row?.moleculeB || "").toUpperCase(),
    extractIntactPrimaryLabel(row?.idB || "").toUpperCase(),
  ];
  if (sideBValues.includes(anchor)) return "B";
  return "A";
}

function extractIntactPartner(row, anchorIdentifier) {
  const anchorSide = resolveIntactAnchorSide(row, anchorIdentifier);
  const partnerSide = anchorSide === "A" ? "B" : "A";
  const uniqueId = normalizeWhitespace(row?.[`uniqueId${partnerSide}`] || "");
  const accession = extractIntactPrimaryLabel(row?.[`id${partnerSide}`] || "", uniqueId);
  return {
    side: partnerSide,
    accession: accession || uniqueId,
    uniqueId,
    molecule: normalizeWhitespace(row?.[`molecule${partnerSide}`] || accession || ""),
    intactName: normalizeWhitespace(row?.[`intactName${partnerSide}`] || ""),
    description: normalizeWhitespace(row?.[`description${partnerSide}`] || ""),
    species: normalizeWhitespace(row?.[`species${partnerSide}`] || ""),
    taxId: toNonNegativeInt(row?.[`taxId${partnerSide}`], 0),
    type: normalizeWhitespace(row?.[`type${partnerSide}`] || ""),
  };
}

function buildBiogridApiUrl(baseUrl, pathname, params = new URLSearchParams(), accessKey = "") {
  const searchParams = new URLSearchParams(params);
  if (accessKey) searchParams.set("accesskey", accessKey);
  const query = searchParams.toString();
  return query ? `${baseUrl}${pathname}?${query}` : `${baseUrl}${pathname}`;
}

function buildBiogridMissingKeyResponse(serviceLabel, envNames, sourceUrl, exampleQuery = "") {
  return {
    content: [{
      type: "text",
      text: renderStructuredResponse({
        summary: `${serviceLabel} service unavailable because ${envNames} is not configured in this environment.`,
        keyFields: [
          `Required environment variable(s): ${envNames}`,
          exampleQuery ? `Example once configured: ${exampleQuery}` : "",
        ].filter(Boolean),
        sources: [sourceUrl].filter(Boolean),
        limitations: [`${serviceLabel} requires a free API access key for all requests.`],
      }),
    }],
    structuredContent: {
      schema: `${serviceLabel.toLowerCase().replace(/[^a-z0-9]+/g, "_")}.status.v1`,
      result_status: "degraded",
    },
  };
}

async function fetchBiogridJson(baseUrl, pathname, params = new URLSearchParams(), options = {}) {
  const accessKey = normalizeWhitespace(options.accessKey || "");
  const serviceLabel = baseUrl === BIOGRID_ORCS_API ? "BioGRID ORCS" : "BioGRID";
  const envNames = baseUrl === BIOGRID_ORCS_API ? "BIOGRID_ORCS_ACCESS_KEY (or BIOGRID_ACCESS_KEY)" : "BIOGRID_ACCESS_KEY";
  if (!accessKey) {
    throw new Error(`${serviceLabel} service unavailable because ${envNames} is not configured in this environment.`);
  }
  const url = buildBiogridApiUrl(baseUrl, pathname, params, accessKey);
  const data = await fetchJsonWithRetry(url, {
    retries: options.retries ?? 1,
    timeoutMs: options.timeoutMs ?? 15000,
    maxBackoffMs: options.maxBackoffMs ?? 2500,
    headers: {
      Accept: "application/json",
      ...(options.headers || {}),
    },
  });
  return { url, data };
}

function biogridTaxIdToMyGeneSpecies(taxId = 9606) {
  const normalized = String(Math.max(0, toNonNegativeInt(taxId, 9606)) || 9606);
  const mapping = {
    "9606": "human",
    "10090": "mouse",
    "10116": "rat",
    "7955": "zebrafish",
    "7227": "fly",
    "6239": "worm",
    "559292": "yeast",
    "8364": "frog",
    "8355": "frog",
  };
  return mapping[normalized] || normalized;
}

async function resolveBiogridGeneSelection(query, speciesTaxId = 9606) {
  const normalizedQuery = normalizeWhitespace(query || "");
  const boundedTaxId = Math.max(0, toNonNegativeInt(speciesTaxId, 9606));
  const species = biogridTaxIdToMyGeneSpecies(boundedTaxId);
  const fields = "symbol,name,alias,entrezgene,ensembl.gene,uniprot.Swiss-Prot,uniprot.TrEMBL,taxid";
  const resolutionUrl = `${MYGENE_API}/query?${new URLSearchParams({
    q: normalizedQuery,
    species,
    size: "8",
    fields,
  }).toString()}`;

  if (!normalizedQuery) {
    return {
      query: "",
      symbol: "",
      name: "",
      aliases: [],
      entrezGene: "",
      taxid: boundedTaxId,
      resolutionUrl,
      matchedViaMyGene: false,
    };
  }

  try {
    const payload = await queryMyGene(normalizedQuery, { species, size: 8, fields });
    const hits = Array.isArray(payload?.hits) ? payload.hits : [];
    const bestHit = selectBestMyGeneHit(hits, normalizedQuery);
    const ids = normalizeMyGeneIds(bestHit || {});
    return {
      query: normalizedQuery,
      symbol: normalizeWhitespace(ids.symbol || normalizedQuery),
      name: normalizeWhitespace(ids.name || ""),
      aliases: dedupeArray(asArray(ids.aliases)).slice(0, 12),
      entrezGene: normalizeWhitespace(ids.entrezgene || ""),
      taxid: toNonNegativeInt(ids.taxid, boundedTaxId),
      resolutionUrl,
      matchedViaMyGene: Boolean(ids.symbol || ids.entrezgene),
    };
  } catch {
    return {
      query: normalizedQuery,
      symbol: normalizedQuery,
      name: "",
      aliases: [],
      entrezGene: /^\d+$/.test(normalizedQuery) ? normalizedQuery : "",
      taxid: boundedTaxId,
      resolutionUrl,
      matchedViaMyGene: false,
    };
  }
}

function normalizeBiogridLookupText(value) {
  return normalizeWhitespace(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function extractBiogridRows(data) {
  if (Array.isArray(data)) return data;
  if (!data || typeof data !== "object") return [];
  return Object.values(data).filter((row) => row && typeof row === "object");
}

function getBiogridField(row, keys) {
  for (const key of keys) {
    const normalized = normalizeWhitespace(row?.[key] || "");
    if (normalized && normalized !== "-") return normalized;
  }
  return "";
}

function extractBiogridInteractor(row, side) {
  const suffix = side === "B" ? "B" : "A";
  const symbol = getBiogridField(row, [`OFFICIAL_SYMBOL_${suffix}`, `OFFICIAL_SYMBOL_INTERACTOR_${suffix}`]);
  const systematicName = getBiogridField(row, [`SYSTEMATIC_NAME_${suffix}`, `SYSTEMATIC_NAME_INTERACTOR_${suffix}`]);
  const entrezGene = getBiogridField(row, [`ENTREZ_GENE_${suffix}`, `ENTREZ_GENE_INTERACTOR_${suffix}`]);
  const biogridId = getBiogridField(row, [`BIOGRID_ID_${suffix}`, `BIOGRID_ID_INTERACTOR_${suffix}`]);
  const taxId = toNonNegativeInt(getBiogridField(row, [`ORGANISM_${suffix}`, `ORGANISM_INTERACTOR_${suffix}`]), 0);
  const synonyms = dedupeArray(
    getBiogridField(row, [`SYNONYMS_${suffix}`, `SYNONYMS_INTERACTOR_${suffix}`])
      .split("|")
      .map((value) => normalizeWhitespace(value))
      .filter(Boolean)
  ).slice(0, 12);
  return {
    side: suffix,
    symbol: symbol || systematicName || entrezGene || biogridId || `Interactor ${suffix}`,
    systematicName,
    entrezGene,
    biogridId,
    taxId,
    synonyms,
  };
}

function scoreBiogridAnchorSide(row, side, candidateSet) {
  if (!(candidateSet instanceof Set) || candidateSet.size === 0) return 0;
  const interactor = extractBiogridInteractor(row, side);
  let score = 0;
  if (candidateSet.has(normalizeBiogridLookupText(interactor.symbol))) score += 8;
  if (candidateSet.has(normalizeBiogridLookupText(interactor.entrezGene))) score += 7;
  if (candidateSet.has(normalizeBiogridLookupText(interactor.systematicName))) score += 5;
  if (candidateSet.has(normalizeBiogridLookupText(interactor.biogridId))) score += 4;
  for (const alias of interactor.synonyms) {
    if (candidateSet.has(normalizeBiogridLookupText(alias))) score += 3;
  }
  return score;
}

function resolveBiogridAnchorSide(row, anchorCandidates = []) {
  const candidateSet = new Set(
    asArray(anchorCandidates)
      .map((value) => normalizeBiogridLookupText(value))
      .filter(Boolean)
  );
  const sideAScore = scoreBiogridAnchorSide(row, "A", candidateSet);
  const sideBScore = scoreBiogridAnchorSide(row, "B", candidateSet);
  if (sideAScore === 0 && sideBScore === 0) return "A";
  return sideAScore >= sideBScore ? "A" : "B";
}

function extractBiogridPartner(row, anchorCandidates = []) {
  const anchorSide = resolveBiogridAnchorSide(row, anchorCandidates);
  return extractBiogridInteractor(row, anchorSide === "A" ? "B" : "A");
}

function extractOrcsScoreSummaries(scoreRow, screenMetadata = {}, limit = 3) {
  const lines = [];
  for (let idx = 1; idx <= 5; idx += 1) {
    const rawValue = normalizeWhitespace(scoreRow?.[`SCORE.${idx}`] || "");
    if (!rawValue || rawValue === "-") continue;
    const scoreType = normalizeWhitespace(screenMetadata?.[`SCORE.${idx}_TYPE`] || "");
    lines.push(scoreType ? `${scoreType}: ${rawValue}` : `Score.${idx}: ${rawValue}`);
    if (lines.length >= limit) break;
  }
  return lines;
}

async function fetchBiogridOrcsScreenDetails(screenIds, accessKey) {
  const ids = dedupeArray(asArray(screenIds).map((value) => normalizeWhitespace(value)).filter(Boolean));
  const lookup = new Map();
  const sourceUrls = [];

  for (let start = 0; start < ids.length; start += 40) {
    const batch = ids.slice(start, start + 40);
    const params = new URLSearchParams({
      format: "json",
      screenID: batch.join("|"),
    });
    const payload = await fetchBiogridJson(BIOGRID_ORCS_API, "/screens/", params, {
      accessKey,
      retries: 1,
      timeoutMs: 20000,
      maxBackoffMs: 2500,
    });
    sourceUrls.push(payload.url);
    for (const row of extractBiogridRows(payload.data)) {
      const screenId = normalizeWhitespace(row?.SCREEN_ID || "");
      if (screenId) lookup.set(screenId, row);
    }
  }

  return { lookup, sourceUrls };
}

async function searchOlsTerms(query, { ontology = "", exact = true, rows = 10 } = {}) {
  const params = new URLSearchParams({
    q: query,
    exact: exact ? "true" : "false",
    rows: String(Math.max(1, Math.min(25, Math.round(rows || 10)))),
  });
  if (ontology) params.set("ontology", ontology);
  const url = `${OLS_API}/search?${params.toString()}`;
  const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 12000, maxBackoffMs: 2500 });
  return {
    url,
    docs: Array.isArray(data?.response?.docs) ? data.response.docs : [],
  };
}

function selectBestOlsDoc(docs, query) {
  const rows = Array.isArray(docs) ? docs : [];
  if (rows.length === 0) return null;
  const normalizedQuery = normalizeOntologyCurie(query).toUpperCase();
  const exactCurie = rows.find((doc) => normalizeOntologyCurie(doc?.obo_id || "").toUpperCase() === normalizedQuery);
  if (exactCurie) return exactCurie;
  const exactShortForm = rows.find((doc) => normalizeOntologyCurie(String(doc?.short_form || "").replace(/_/g, ":")).toUpperCase() === normalizedQuery);
  if (exactShortForm) return exactShortForm;
  return rows[0];
}

async function fetchOlsTermRecordByCurie(curie) {
  const searchResult = await searchOlsTerms(curie, { exact: true, rows: 10 });
  const doc = selectBestOlsDoc(searchResult.docs, curie);
  if (!doc?.ontology_name || !doc?.iri) {
    return doc ? { doc, term: null, searchUrl: searchResult.url, termUrl: "" } : null;
  }
  const encodedIri = encodeURIComponent(encodeURIComponent(doc.iri));
  const termUrl = `${OLS_API}/ontologies/${encodeURIComponent(doc.ontology_name)}/terms/${encodedIri}`;
  const term = await fetchJsonWithRetry(termUrl, { retries: 1, timeoutMs: 12000, maxBackoffMs: 2500 });
  return { doc, term, searchUrl: searchResult.url, termUrl };
}

function extractMappingsFromOlsTerm(term, allowedPrefixes = []) {
  const allowed = new Set((Array.isArray(allowedPrefixes) ? allowedPrefixes : []).map((value) => normalizeOntologyPrefix(value)).filter(Boolean));
  const xrefs = Array.isArray(term?.obo_xref) ? term.obo_xref : [];
  const seen = new Set();
  const mappings = [];
  for (const xref of xrefs) {
    const prefix = normalizeOntologyPrefix(xref?.database || "");
    const identifier = normalizeWhitespace(xref?.id || "");
    if (!prefix || !identifier) continue;
    if (allowed.size > 0 && !allowed.has(prefix)) continue;
    const curie = `${prefix}:${identifier}`;
    const dedupeKey = curie.toUpperCase();
    if (seen.has(dedupeKey)) continue;
    seen.add(dedupeKey);
    mappings.push({
      curie,
      label: "",
      targetPrefix: prefix,
      distance: 1,
      mappingOrigin: "OLS xref",
      sourceUrl: normalizeWhitespace(xref?.url || ""),
    });
  }
  return mappings;
}

async function enrichMappingsWithOlsLabels(mappings) {
  const rows = Array.isArray(mappings) ? mappings : [];
  const lookups = await Promise.allSettled(
    rows.map(async (mapping) => {
      const ontology = getOlsOntologyNameForPrefix(mapping?.targetPrefix || extractCuriePrefix(mapping?.curie || ""));
      const searchResult = await searchOlsTerms(mapping.curie, { ontology, exact: true, rows: 5 });
      const doc = selectBestOlsDoc(searchResult.docs, mapping.curie);
      return {
        curie: mapping.curie,
        label: normalizeWhitespace(doc?.label || ""),
        searchUrl: searchResult.url,
      };
    })
  );
  const extraSources = [];
  for (let i = 0; i < lookups.length; i += 1) {
    const result = lookups[i];
    if (result.status !== "fulfilled") continue;
    if (result.value.label) rows[i].label = result.value.label;
    if (result.value.searchUrl) extraSources.push(result.value.searchUrl);
  }
  return dedupeArray(extraSources);
}

async function buildOlsFallbackMappings(cleanedIds, cleanedPrefixes, boundedMappings, oxoErrorMessage = "") {
  const allowedPrefixes = new Set((Array.isArray(cleanedPrefixes) ? cleanedPrefixes : []).map((value) => normalizeOntologyPrefix(value)).filter(Boolean));
  const sources = [];
  const keyFields = [];
  const limitations = [
    oxoErrorMessage
      ? `OxO service unavailable (${compactErrorMessage(oxoErrorMessage)}); fallback uses OLS term metadata instead of OxO graph traversal.`
      : "OxO returned no mappings; fallback uses OLS term metadata instead of OxO graph traversal.",
    "OLS fallback captures direct ontology cross-references and exact-label candidates only; indirect OxO graph distances may be unavailable.",
  ];
  let recoveredTerms = 0;

  for (const inputId of cleanedIds) {
    let record = null;
    try {
      record = await fetchOlsTermRecordByCurie(inputId);
    } catch {
      record = null;
    }

    if (!record) {
      keyFields.push(`${inputId} | Unable to resolve source term in OLS.`);
      continue;
    }

    recoveredTerms += 1;
    if (record.searchUrl) sources.push(record.searchUrl);
    if (record.termUrl) sources.push(record.termUrl);

    const inputLabel = normalizeWhitespace(record?.term?.label || record?.doc?.label || "");
    const notes = [];
    const mappings = extractMappingsFromOlsTerm(record?.term, cleanedPrefixes);

    for (const prefix of allowedPrefixes) {
      if (mappings.some((mapping) => normalizeOntologyPrefix(mapping.targetPrefix) === prefix)) continue;
      const ontology = getOlsOntologyNameForPrefix(prefix);
      if (!ontology || !inputLabel) continue;

      try {
        const candidateSearch = await searchOlsTerms(inputLabel, { ontology, exact: true, rows: 5 });
        if (candidateSearch.url) sources.push(candidateSearch.url);
        const candidate = (candidateSearch.docs || []).find((doc) => {
          const curie = normalizeOntologyCurie(doc?.obo_id || "");
          const label = normalizeWhitespace(doc?.label || "").toLowerCase();
          return extractCuriePrefix(curie) === prefix && label === inputLabel.toLowerCase();
        });
        if (candidate) {
          mappings.push({
            curie: normalizeOntologyCurie(candidate.obo_id || ""),
            label: normalizeWhitespace(candidate.label || ""),
            targetPrefix: prefix,
            distance: "label-match",
            mappingOrigin: "OLS exact label candidate",
            sourceUrl: "",
          });
          continue;
        }
        const imported = (candidateSearch.docs || []).find(
          (doc) => normalizeOntologyCurie(doc?.obo_id || "").toUpperCase() === normalizeOntologyCurie(inputId).toUpperCase()
        );
        if (imported) {
          notes.push(`No distinct ${prefix} CURIE found; OLS exposes ${normalizeOntologyCurie(inputId)} as an imported term in ${prefix}.`);
        }
      } catch {
        // Ignore exact-label candidate failures in fallback mode.
      }
    }

    const dedupedMappings = [];
    const seenMappings = new Set();
    for (const mapping of mappings) {
      const curie = normalizeOntologyCurie(mapping?.curie || "");
      if (!curie) continue;
      const dedupeKey = `${curie.toUpperCase()}|${normalizeWhitespace(mapping?.mappingOrigin || "").toUpperCase()}`;
      if (seenMappings.has(dedupeKey)) continue;
      seenMappings.add(dedupeKey);
      dedupedMappings.push({ ...mapping, curie });
    }

    const trimmedMappings = dedupedMappings.slice(0, boundedMappings);
    sources.push(...trimmedMappings.map((mapping) => normalizeWhitespace(mapping?.sourceUrl || "")).filter(Boolean));

    if (trimmedMappings.length > 0) {
      const labelSources = await enrichMappingsWithOlsLabels(trimmedMappings.filter((mapping) => !mapping.label));
      sources.push(...labelSources);
    }

    const mappingText = trimmedMappings
      .map((mapping) => {
        const hopRaw = mapping?.distance;
        const hopText = Number.isFinite(Number(hopRaw))
          ? ` d=${toNonNegativeInt(hopRaw, 1)}`
          : normalizeWhitespace(hopRaw || "")
            ? ` d=${normalizeWhitespace(hopRaw)}`
            : "";
        const origin = normalizeWhitespace(mapping?.mappingOrigin || "");
        return `${mapping.curie}${mapping.label ? ` (${mapping.label})` : ""}${mapping.targetPrefix ? ` [${mapping.targetPrefix}]` : ""}${hopText}${origin ? ` via ${origin}` : ""}`;
      })
      .join("; ") || "No fallback mappings";

    const noteText = notes.length > 0 ? ` | Notes: ${notes.join(" ")}` : "";
    keyFields.push(`${normalizeOntologyCurie(inputId)}${inputLabel ? ` (${inputLabel})` : ""} | ${mappingText}${noteText}`);
  }

  return {
    recoveredTerms,
    keyFields,
    sources: dedupeArray(sources).filter(Boolean).slice(0, 12),
    limitations,
  };
}

function xmlText(node) {
  if (node === null || node === undefined) return "";
  if (typeof node === "string" || typeof node === "number" || typeof node === "boolean") {
    return normalizeWhitespace(String(node));
  }
  if (Array.isArray(node)) {
    return normalizeWhitespace(node.map((item) => xmlText(item)).filter(Boolean).join("; "));
  }
  if (typeof node === "object") {
    if (typeof node["#text"] === "string") return normalizeWhitespace(node["#text"]);
    if (typeof node.text === "string") return normalizeWhitespace(node.text);
  }
  return "";
}

function normalizeOrphanetLookupText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function extractOrphanetCode(value) {
  const raw = normalizeWhitespace(value || "");
  const curieMatch = raw.match(/^(?:orphanet|orpha)\s*[: ]\s*(\d+)$/i);
  if (curieMatch) return curieMatch[1];
  return /^\d+$/.test(raw) ? raw : "";
}

function scoreOrphanetPhenotypeFrequency(label) {
  const normalized = normalizeWhitespace(label || "").toLowerCase();
  if (!normalized) return 0;
  if (normalized.includes("obligate")) return 6;
  if (normalized.includes("very frequent")) return 5;
  if (normalized.includes("frequent")) return 4;
  if (normalized.includes("occasional")) return 3;
  if (normalized.includes("very rare")) return 2;
  if (normalized.includes("excluded")) return -1;
  return 1;
}

function scoreOrphanetDoc(doc, query) {
  const normalizedQuery = normalizeOrphanetLookupText(query);
  const exactCode = extractOrphanetCode(query);
  const curie = normalizeOntologyCurie(doc?.obo_id || doc?.short_form || "");
  const label = normalizeOrphanetLookupText(doc?.label || "");
  const synonyms = dedupeArray([
    ...asArray(doc?.exact_synonyms).map((value) => normalizeOrphanetLookupText(value)),
    ...asArray(doc?.synonym).map((value) => normalizeOrphanetLookupText(value)),
  ].filter(Boolean));
  let score = 0;
  if (curie && normalizeOntologyCurie(curie).toUpperCase() === normalizeOntologyCurie(query).toUpperCase()) score += 300;
  if (exactCode && extractOrphanetCode(curie) === exactCode) score += 240;
  if (label && label === normalizedQuery) score += 150;
  if (synonyms.includes(normalizedQuery)) score += 110;
  if (label && normalizedQuery && label.includes(normalizedQuery)) score += 45;
  if (label && normalizedQuery && normalizedQuery.includes(label)) score += 25;
  if (normalizeWhitespace(doc?.ontology_name || "").toLowerCase() === "ordo") score += 10;
  return score;
}

function scoreHpoDoc(doc, query) {
  const normalizedQuery = normalizeOrphanetLookupText(query);
  const curie = normalizeOntologyCurie(doc?.obo_id || doc?.short_form || "");
  const label = normalizeOrphanetLookupText(doc?.label || "");
  const synonyms = dedupeArray(asArray(doc?.exact_synonyms).map((value) => normalizeOrphanetLookupText(value)).filter(Boolean));
  let score = 0;
  if (curie && normalizeOntologyCurie(curie).toUpperCase() === normalizeOntologyCurie(query).toUpperCase()) score += 260;
  if (label && label === normalizedQuery) score += 150;
  if (synonyms.includes(normalizedQuery)) score += 100;
  if (label && normalizedQuery && label.includes(normalizedQuery)) score += 35;
  if (normalizeWhitespace(doc?.ontology_name || "").toLowerCase() === "hp") score += 10;
  return score;
}

function extractOrphanetExternalReferences(rawReferences) {
  return asArray(rawReferences)
    .map((reference) => {
      const source = normalizeWhitespace(reference?.Source || reference?.database || "");
      const value = normalizeWhitespace(reference?.Reference || reference?.id || "");
      if (!source || !value) return null;
      return {
        source,
        reference: value,
        relation: xmlText(reference?.DisorderMappingRelation?.Name || reference?.description || ""),
        validation: xmlText(reference?.DisorderMappingValidationStatus?.Name || ""),
        url: normalizeWhitespace(reference?.DisorderMappingICDRefUrl || reference?.url || ""),
      };
    })
    .filter(Boolean);
}

function mergeOrphanetExternalReferences(...collections) {
  const merged = [];
  const seen = new Set();
  for (const collection of collections) {
    for (const reference of asArray(collection)) {
      const source = normalizeWhitespace(reference?.source || reference?.Source || "");
      const value = normalizeWhitespace(reference?.reference || reference?.Reference || "");
      if (!source || !value) continue;
      const key = `${source.toUpperCase()}:${value.toUpperCase()}`;
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push({
        source,
        reference: value,
        relation: normalizeWhitespace(reference?.relation || reference?.description || ""),
        validation: normalizeWhitespace(reference?.validation || ""),
        url: normalizeWhitespace(reference?.url || ""),
      });
    }
  }
  return merged;
}

function extractOrphanetXrefsFromOlsTerm(term) {
  const annotations = asArray(term?.annotation?.database_cross_reference)
    .map((value) => normalizeWhitespace(value))
    .filter(Boolean)
    .map((value) => {
      const match = value.match(/^([^:]+):(.*)$/);
      if (!match) return null;
      return {
        source: normalizeOntologyPrefix(match[1]),
        reference: normalizeWhitespace(match[2]),
        relation: "",
        validation: "",
        url: "",
      };
    })
    .filter(Boolean);
  const xrefs = extractOrphanetExternalReferences(term?.obo_xref);
  return mergeOrphanetExternalReferences(annotations, xrefs);
}

function buildOrphanetSearchText(entry) {
  const parts = [
    entry?.orphaCode || "",
    entry?.curie || "",
    entry?.name || "",
    ...asArray(entry?.synonyms),
    ...asArray(entry?.xrefs).map((row) => `${row?.source || ""} ${row?.reference || ""}`.trim()),
  ];
  return normalizeOrphanetLookupText(parts.join(" "));
}

function scoreOrphanetCatalogEntry(entry, query) {
  const normalizedQuery = normalizeOrphanetLookupText(query);
  const exactCode = extractOrphanetCode(query);
  if (!normalizedQuery && !exactCode) return Number.NEGATIVE_INFINITY;
  let score = 0;
  const name = normalizeOrphanetLookupText(entry?.name || "");
  const synonyms = asArray(entry?.synonyms).map((value) => normalizeOrphanetLookupText(value)).filter(Boolean);
  if (exactCode && normalizeWhitespace(entry?.orphaCode || "") === exactCode) score += 260;
  if (name && name === normalizedQuery) score += 140;
  if (synonyms.includes(normalizedQuery)) score += 100;
  if (name && normalizedQuery && name.includes(normalizedQuery)) score += 35;
  if (entry?.searchText && normalizedQuery && entry.searchText.includes(normalizedQuery)) score += 20;
  return score;
}

function getOrphanetReferenceValue(entry, sourceName) {
  const wanted = normalizeWhitespace(sourceName || "").toUpperCase();
  const hit = asArray(entry?.xrefs).find((reference) => normalizeWhitespace(reference?.source || "").toUpperCase() === wanted);
  return normalizeWhitespace(hit?.reference || "");
}

function extractPmidsFromValidationText(value, limit = 5) {
  return dedupeArray(
    [...String(value || "").matchAll(/(\d{6,9})(?=\[PMID\])/g)].map((match) => normalizeWhitespace(match?.[1] || "")).filter(Boolean)
  ).slice(0, limit);
}

async function fetchOrphanetDisorderCatalog() {
  const cached = getFreshCacheValue(orphanetDisorderCatalogCache, 24 * 60 * 60 * 1000);
  if (cached) return cached;
  const response = await fetchWithRetry(ORPHADATA_PRODUCT1_XML, { retries: 1, timeoutMs: 45000, maxBackoffMs: 3000 });
  const xml = await response.text();
  const parsed = parseXmlDocument(xml);
  const disorders = asArray(parsed?.JDBOR?.DisorderList?.Disorder)
    .map((disorder) => {
      const orphaCode = normalizeWhitespace(disorder?.OrphaCode || "");
      const name = xmlText(disorder?.Name);
      if (!orphaCode || !name) return null;
      const entry = {
        orphaCode,
        curie: `Orphanet:${orphaCode}`,
        name,
        synonyms: asArray(disorder?.SynonymList?.Synonym).map((value) => xmlText(value)).filter(Boolean),
        expertLink: normalizeWhitespace(xmlText(disorder?.ExpertLink) || ""),
        disorderType: xmlText(disorder?.DisorderType?.Name),
        disorderGroup: xmlText(disorder?.DisorderGroup?.Name),
        xrefs: extractOrphanetExternalReferences(disorder?.ExternalReferenceList?.ExternalReference),
      };
      entry.searchText = buildOrphanetSearchText(entry);
      return entry;
    })
    .filter(Boolean);
  orphanetDisorderCatalogCache = storeCacheValue(orphanetDisorderCatalogCache, disorders);
  return disorders;
}

async function fetchOrphanetPhenotypeCatalog() {
  const cached = getFreshCacheValue(orphanetPhenotypeCatalogCache, 24 * 60 * 60 * 1000);
  if (cached) return cached;
  const response = await fetchWithRetry(ORPHADATA_PRODUCT4_XML, { retries: 1, timeoutMs: 45000, maxBackoffMs: 3000 });
  const xml = await response.text();
  const parsed = parseXmlDocument(xml);
  const byCode = {};
  for (const disorderStatus of asArray(parsed?.JDBOR?.HPODisorderSetStatusList?.HPODisorderSetStatus)) {
    const disorder = disorderStatus?.Disorder || {};
    const orphaCode = normalizeWhitespace(disorder?.OrphaCode || "");
    if (!orphaCode) continue;
    const phenotypes = asArray(disorder?.HPODisorderAssociationList?.HPODisorderAssociation)
      .map((association) => {
        const hpoId = normalizeOntologyCurie(association?.HPO?.HPOId || "");
        const label = normalizeWhitespace(association?.HPO?.HPOTerm || "");
        if (!hpoId || !label) return null;
        return {
          hpoId,
          label,
          frequency: xmlText(association?.HPOFrequency?.Name),
          diagnosticCriteria: normalizeWhitespace(association?.DiagnosticCriteria || ""),
        };
      })
      .filter(Boolean)
      .sort((a, b) => scoreOrphanetPhenotypeFrequency(b.frequency) - scoreOrphanetPhenotypeFrequency(a.frequency) || a.label.localeCompare(b.label));
    byCode[orphaCode] = phenotypes;
  }
  orphanetPhenotypeCatalogCache = storeCacheValue(orphanetPhenotypeCatalogCache, byCode);
  return byCode;
}

async function fetchOrphanetGeneCatalog() {
  const cached = getFreshCacheValue(orphanetGeneCatalogCache, 24 * 60 * 60 * 1000);
  if (cached) return cached;
  const response = await fetchWithRetry(ORPHADATA_PRODUCT6_XML, { retries: 1, timeoutMs: 45000, maxBackoffMs: 3000 });
  const xml = await response.text();
  const parsed = parseXmlDocument(xml);
  const byCode = {};
  for (const disorder of asArray(parsed?.JDBOR?.DisorderList?.Disorder)) {
    const orphaCode = normalizeWhitespace(disorder?.OrphaCode || "");
    if (!orphaCode) continue;
    const genes = asArray(disorder?.DisorderGeneAssociationList?.DisorderGeneAssociation)
      .map((association) => {
        const gene = association?.Gene || {};
        const xrefs = extractOrphanetExternalReferences(gene?.ExternalReferenceList?.ExternalReference);
        const symbol = normalizeWhitespace(gene?.Symbol || "");
        const name = xmlText(gene?.Name);
        if (!symbol && !name) return null;
        return {
          symbol: symbol || name,
          name,
          associationType: xmlText(association?.DisorderGeneAssociationType?.Name),
          status: xmlText(association?.DisorderGeneAssociationStatus?.Name),
          validation: normalizeWhitespace(association?.SourceOfValidation || ""),
          pmids: extractPmidsFromValidationText(association?.SourceOfValidation || ""),
          geneType: xmlText(gene?.GeneType?.Name),
          hgnc: getOrphanetReferenceValue({ xrefs }, "HGNC"),
          ensembl: getOrphanetReferenceValue({ xrefs }, "Ensembl"),
          omim: getOrphanetReferenceValue({ xrefs }, "OMIM"),
          swissProt: getOrphanetReferenceValue({ xrefs }, "SwissProt"),
        };
      })
      .filter(Boolean);
    byCode[orphaCode] = genes;
  }
  orphanetGeneCatalogCache = storeCacheValue(orphanetGeneCatalogCache, byCode);
  return byCode;
}

async function resolveOrphanetDisorderSelection(query) {
  const normalizedQuery = normalizeWhitespace(query || "");
  const exactCode = extractOrphanetCode(normalizedQuery);
  const catalog = await fetchOrphanetDisorderCatalog();

  let searchUrl = "";
  let record = null;
  if (exactCode) {
    record = await fetchOlsTermRecordByCurie(`Orphanet:${exactCode}`).catch(() => null);
    searchUrl = record?.searchUrl || "";
  } else if (normalizedQuery) {
    const search = await searchOlsTerms(normalizedQuery, { ontology: "ordo", exact: false, rows: 10 }).catch(() => ({ url: "", docs: [] }));
    searchUrl = search?.url || "";
    const bestDoc = [...(search?.docs || [])]
      .sort((a, b) => scoreOrphanetDoc(b, normalizedQuery) - scoreOrphanetDoc(a, normalizedQuery))[0];
    if (bestDoc?.obo_id) {
      record = await fetchOlsTermRecordByCurie(bestDoc.obo_id).catch(() => ({ doc: bestDoc, term: null, searchUrl, termUrl: "" }));
    }
  }

  const resolvedCode = extractOrphanetCode(record?.doc?.obo_id || record?.doc?.short_form || "") || exactCode;
  let selected = resolvedCode ? catalog.find((entry) => entry.orphaCode === resolvedCode) || null : null;

  if (!selected && normalizedQuery) {
    selected = [...catalog]
      .sort((a, b) => scoreOrphanetCatalogEntry(b, normalizedQuery) - scoreOrphanetCatalogEntry(a, normalizedQuery))[0] || null;
    if (selected && scoreOrphanetCatalogEntry(selected, normalizedQuery) <= 0) {
      selected = null;
    }
  }

  if (!record && selected?.curie) {
    record = await fetchOlsTermRecordByCurie(selected.curie).catch(() => null);
    searchUrl = record?.searchUrl || searchUrl;
  }

  if (selected && record?.term) {
    selected = {
      ...selected,
      definition: asArray(record.term.description).map((value) => normalizeWhitespace(value)).filter(Boolean),
      olsXrefs: extractOrphanetXrefsFromOlsTerm(record.term),
    };
  }

  return {
    selected,
    record,
    searchUrl,
  };
}

function scoreMonarchSearchLexicalMatch(item, query) {
  const normalizedQuery = normalizeOntologyCurie(query).toUpperCase();
  const queryText = normalizeOrphanetLookupText(query);
  const itemId = normalizeOntologyCurie(item?.id || "").toUpperCase();
  const name = normalizeOrphanetLookupText(item?.name || "");
  const symbol = normalizeWhitespace(item?.symbol || "").toUpperCase();
  const synonyms = [
    ...asArray(item?.synonym),
    ...asArray(item?.exact_synonym),
    ...asArray(item?.related_synonym),
  ]
    .map((value) => normalizeOrphanetLookupText(value))
    .filter(Boolean);

  let score = 0;
  if (itemId && itemId === normalizedQuery) score += 320;
  if (symbol && symbol === normalizeWhitespace(query || "").toUpperCase()) score += 150;
  if (name && name === queryText) score += 130;
  if (synonyms.includes(queryText)) score += 95;
  if (name && queryText && name.includes(queryText)) score += 35;
  if (name && queryText && queryText.includes(name)) score += 15;
  return score;
}

function scoreMonarchSearchItem(item, query, modeConfig) {
  const category = normalizeWhitespace(item?.category || "");
  let score = scoreMonarchSearchLexicalMatch(item, query);
  if (modeConfig?.inputCategories?.includes(category)) score += 120;
  return score;
}

function normalizeMonarchTaxonHint(value) {
  const raw = normalizeWhitespace(value || "");
  if (!raw) return "";
  const lower = raw.toLowerCase();
  if (lower === "9606" || lower === "ncbitaxon:9606" || lower === "homo sapiens") {
    return "NCBITaxon:9606";
  }
  return raw;
}

function getMonarchSearchItemTaxonHints(item) {
  return dedupeArray([
    normalizeMonarchTaxonHint(item?.taxon),
    normalizeMonarchTaxonHint(item?.taxon_id),
    normalizeMonarchTaxonHint(item?.in_taxon),
    normalizeMonarchTaxonHint(item?.taxon_label),
    normalizeMonarchTaxonHint(item?.in_taxon_label),
  ].filter(Boolean));
}

function isMonarchHumanGeneSearchItem(item) {
  if (normalizeWhitespace(item?.category || "") !== "biolink:Gene") return false;
  const itemId = normalizeOntologyCurie(item?.id || "").toUpperCase();
  if (/^HGNC:/.test(itemId)) return true;
  const taxonHints = getMonarchSearchItemTaxonHints(item);
  return taxonHints.includes("NCBITaxon:9606");
}

function scoreMonarchHumanGenePreference(item, modeConfig, humanOnly) {
  if (!humanOnly) return 0;
  if (!modeConfig?.inputCategories?.includes("biolink:Gene")) return 0;
  if (normalizeWhitespace(item?.category || "") !== "biolink:Gene") return 0;

  const itemId = normalizeOntologyCurie(item?.id || "").toUpperCase();
  if (isMonarchHumanGeneSearchItem(item)) return 220;
  const taxonHints = getMonarchSearchItemTaxonHints(item);
  if (/^(XENBASE|MGI|RGD|ZFIN|FB|WB|SGD):/.test(itemId)) return -160;
  if (taxonHints.some((value) => value && value !== "NCBITaxon:9606")) return -140;
  if (/^(NCBIGENE|ENSEMBL):/.test(itemId)) return 40;
  return 0;
}

async function fetchMonarchEntity(entityId) {
  const url = `${MONARCH_API}/entity/${encodeURIComponent(entityId)}`;
  const record = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 12000, maxBackoffMs: 2500 });
  return { record, url };
}

async function resolveMonarchEntity(queryOrId, associationMode, options = {}) {
  const normalizedQuery = normalizeWhitespace(queryOrId || "");
  const modeConfig = MONARCH_ASSOCIATION_MODES[associationMode];
  const humanOnly = options?.humanOnly !== false;
  if (!normalizedQuery) return { selected: null, candidates: [], searchUrl: "", entityUrl: "" };

  if (!normalizedQuery.includes(":") && modeConfig?.inputCategories?.includes("biolink:PhenotypicFeature")) {
    try {
      const hpoSearch = await searchOlsTerms(normalizedQuery, { ontology: "hp", exact: false, rows: 8 });
      const hpoDoc = [...(hpoSearch?.docs || [])]
        .sort((a, b) => scoreHpoDoc(b, normalizedQuery) - scoreHpoDoc(a, normalizedQuery))[0];
      if (hpoDoc?.obo_id) {
        const hpoEntity = await fetchMonarchEntity(normalizeOntologyCurie(hpoDoc.obo_id));
        if (modeConfig.inputCategories.includes(normalizeWhitespace(hpoEntity.record?.category || ""))) {
          return {
            selected: hpoEntity.record,
            candidates: [],
            searchUrl: hpoSearch?.url || "",
            entityUrl: hpoEntity.url,
          };
        }
      }
    } catch {
      // Fall through to direct Monarch search if HPO-assisted grounding fails.
    }
  }

  if (normalizedQuery.includes(":")) {
    try {
      const direct = await fetchMonarchEntity(normalizedQuery);
      if (!modeConfig || modeConfig.inputCategories.includes(normalizeWhitespace(direct.record?.category || ""))) {
        return { selected: direct.record, candidates: [], searchUrl: "", entityUrl: direct.url };
      }
    } catch {
      // Fall back to search-based resolution below.
    }
  }

  const searchParams = new URLSearchParams({
    q: normalizedQuery,
    limit: "8",
  });
  const searchUrl = `${MONARCH_API}/search?${searchParams.toString()}`;
  const searchData = await fetchJsonWithRetry(searchUrl, { retries: 1, timeoutMs: 12000, maxBackoffMs: 2500 });
  const items = Array.isArray(searchData?.items) ? searchData.items : [];
  let candidateItems = [...items];
  const categoryMatchedItems = candidateItems.filter((item) =>
    modeConfig?.inputCategories?.includes(normalizeWhitespace(item?.category || ""))
  );
  if (categoryMatchedItems.length > 0) {
    candidateItems = categoryMatchedItems;
  }
  if (humanOnly && modeConfig?.inputCategories?.includes("biolink:Gene")) {
    const humanGeneItems = candidateItems.filter((item) => isMonarchHumanGeneSearchItem(item));
    if (humanGeneItems.length > 0) {
      candidateItems = humanGeneItems;
    }
  }

  const ranked = [...candidateItems]
    .map((item) => ({
      item,
      lexicalScore: scoreMonarchSearchLexicalMatch(item, normalizedQuery),
      score:
        scoreMonarchSearchItem(item, normalizedQuery, modeConfig)
        + scoreMonarchHumanGenePreference(item, modeConfig, humanOnly),
    }))
    .sort((a, b) => b.score - a.score);
  const selectedSearchItem = ranked[0]?.item || null;
  const selectedScore = ranked[0]?.score || 0;
  const selectedLexicalScore = ranked[0]?.lexicalScore || 0;
  if (!selectedSearchItem?.id || selectedScore <= 0 || selectedLexicalScore <= 0) {
    return {
      selected: null,
      candidates: ranked.slice(0, 5).map(({ item }) => item),
      searchUrl,
      entityUrl: "",
    };
  }

  try {
    const entity = await fetchMonarchEntity(selectedSearchItem.id);
    return {
      selected: entity.record,
      candidates: ranked.slice(0, 5).map(({ item }) => item),
      searchUrl,
      entityUrl: entity.url,
    };
  } catch {
    return {
      selected: selectedSearchItem,
      candidates: ranked.slice(0, 5).map(({ item }) => item),
      searchUrl,
      entityUrl: "",
    };
  }
}

function getMonarchAssociationCounterpart(row, modeConfig) {
  const side = modeConfig?.counterpartSide || "object";
  return {
    id: normalizeOntologyCurie(row?.[side] || ""),
    label: normalizeWhitespace(row?.[`${side}_label`] || ""),
    category: normalizeWhitespace(row?.[`${side}_category`] || ""),
    taxon: normalizeWhitespace(row?.[`${side}_taxon`] || ""),
    taxonLabel: normalizeWhitespace(row?.[`${side}_taxon_label`] || ""),
  };
}

function keepMonarchAssociationForHuman(row, modeConfig, humanOnly) {
  if (!humanOnly) return true;
  const counterpart = getMonarchAssociationCounterpart(row, modeConfig);
  if (counterpart.category === "biolink:Gene") {
    return !counterpart.taxon || counterpart.taxon === "NCBITaxon:9606";
  }
  const contextSide = modeConfig?.contextSide || "subject";
  const contextCategory = normalizeWhitespace(row?.[`${contextSide}_category`] || "");
  const contextTaxon = normalizeWhitespace(row?.[`${contextSide}_taxon`] || "");
  if (contextCategory === "biolink:Gene") {
    return !contextTaxon || contextTaxon === "NCBITaxon:9606";
  }
  return true;
}

async function fetchMonarchAssociations(entityId, associationMode, limit) {
  const modeConfig = MONARCH_ASSOCIATION_MODES[associationMode];
  const boundedLimit = Math.max(1, Math.min(25, Math.round(limit || 10)));
  const url = `${MONARCH_API}/entity/${encodeURIComponent(entityId)}/${encodeURIComponent(modeConfig.category)}?limit=${boundedLimit}`;
  const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 });
  return {
    url,
    items: Array.isArray(data?.items) ? data.items : [],
    total: toNonNegativeInt(data?.total, 0),
  };
}

function normalizeAllianceSpeciesTaxon(value) {
  const raw = normalizeWhitespace(value || "");
  if (!raw) return "NCBITaxon:9606";
  const lower = raw.toLowerCase();
  const mapping = {
    "9606": "NCBITaxon:9606",
    human: "NCBITaxon:9606",
    "homo sapiens": "NCBITaxon:9606",
    hsa: "NCBITaxon:9606",
    "10090": "NCBITaxon:10090",
    mouse: "NCBITaxon:10090",
    "mus musculus": "NCBITaxon:10090",
    mmu: "NCBITaxon:10090",
    "10116": "NCBITaxon:10116",
    rat: "NCBITaxon:10116",
    "rattus norvegicus": "NCBITaxon:10116",
    rno: "NCBITaxon:10116",
    "7955": "NCBITaxon:7955",
    zebrafish: "NCBITaxon:7955",
    "danio rerio": "NCBITaxon:7955",
    dre: "NCBITaxon:7955",
    "7227": "NCBITaxon:7227",
    fly: "NCBITaxon:7227",
    drosophila: "NCBITaxon:7227",
    "drosophila melanogaster": "NCBITaxon:7227",
    dme: "NCBITaxon:7227",
    "6239": "NCBITaxon:6239",
    worm: "NCBITaxon:6239",
    "caenorhabditis elegans": "NCBITaxon:6239",
    cel: "NCBITaxon:6239",
    "559292": "NCBITaxon:559292",
    yeast: "NCBITaxon:559292",
    "saccharomyces cerevisiae": "NCBITaxon:559292",
    sce: "NCBITaxon:559292",
    "8364": "NCBITaxon:8364",
    frog: "NCBITaxon:8364",
    xenopus: "NCBITaxon:8364",
    "xenopus tropicalis": "NCBITaxon:8364",
    "8355": "NCBITaxon:8355",
    "xenopus laevis": "NCBITaxon:8355",
  };
  if (mapping[lower]) return mapping[lower];
  if (/^NCBITaxon:\d+$/i.test(raw)) return raw.replace(/^NCBITAXON:/i, "NCBITaxon:");
  return "NCBITaxon:9606";
}

function extractNumericTaxonId(taxonCurie) {
  const match = normalizeWhitespace(taxonCurie || "").match(/(\d+)$/);
  return match ? match[1] : "";
}

function normalizeAllianceSpeciesName(value) {
  return normalizeWhitespace(value || "").toLowerCase();
}

function allianceSpeciesMatchesTarget(speciesLabel, targetTaxon) {
  const label = normalizeAllianceSpeciesName(speciesLabel);
  const numeric = extractNumericTaxonId(targetTaxon);
  if (!label) return false;
  const aliases = {
    "NCBITaxon:9606": ["human", "homo sapiens"],
    "NCBITaxon:10090": ["mouse", "mus musculus"],
    "NCBITaxon:10116": ["rat", "rattus norvegicus"],
    "NCBITaxon:7955": ["zebrafish", "danio rerio"],
    "NCBITaxon:7227": ["fly", "drosophila melanogaster"],
    "NCBITaxon:6239": ["worm", "caenorhabditis elegans"],
    "NCBITaxon:559292": ["yeast", "saccharomyces cerevisiae"],
    "NCBITaxon:8364": ["frog", "xenopus tropicalis"],
    "NCBITaxon:8355": ["frog", "xenopus laevis"],
  };
  return (aliases[targetTaxon] || []).includes(label) || label.includes(numeric);
}

function allianceModelSpeciesRank(taxonCurie, speciesName = "") {
  if (ALLIANCE_MODEL_SPECIES_RANK[taxonCurie]) return ALLIANCE_MODEL_SPECIES_RANK[taxonCurie];
  const normalizedSpecies = normalizeAllianceSpeciesName(speciesName);
  if (normalizedSpecies.includes("mouse")) return ALLIANCE_MODEL_SPECIES_RANK["NCBITaxon:10090"];
  if (normalizedSpecies.includes("rat")) return ALLIANCE_MODEL_SPECIES_RANK["NCBITaxon:10116"];
  if (normalizedSpecies.includes("zebrafish") || normalizedSpecies.includes("danio")) return ALLIANCE_MODEL_SPECIES_RANK["NCBITaxon:7955"];
  if (normalizedSpecies.includes("drosophila") || normalizedSpecies.includes("fly")) return ALLIANCE_MODEL_SPECIES_RANK["NCBITaxon:7227"];
  if (normalizedSpecies.includes("caenorhabditis") || normalizedSpecies.includes("worm")) return ALLIANCE_MODEL_SPECIES_RANK["NCBITaxon:6239"];
  if (normalizedSpecies.includes("yeast") || normalizedSpecies.includes("saccharomyces")) return ALLIANCE_MODEL_SPECIES_RANK["NCBITaxon:559292"];
  if (normalizedSpecies.includes("xenopus") || normalizedSpecies.includes("frog")) return ALLIANCE_MODEL_SPECIES_RANK["NCBITaxon:8364"];
  return 50;
}

function buildAllianceApiUrl(pathname, params = new URLSearchParams()) {
  const query = params.toString();
  return query ? `${ALLIANCE_GENOME_API}${pathname}?${query}` : `${ALLIANCE_GENOME_API}${pathname}`;
}

async function fetchAllianceJson(pathname, params = new URLSearchParams(), options = {}) {
  const url = buildAllianceApiUrl(pathname, params);
  const data = await fetchJsonWithRetry(url, {
    retries: options.retries ?? 1,
    timeoutMs: options.timeoutMs ?? 15000,
    maxBackoffMs: options.maxBackoffMs ?? 2500,
  });
  return { url, data };
}

async function fetchAllianceGeneRecordById(geneId) {
  const normalizedGeneId = normalizeWhitespace(geneId || "");
  if (!normalizedGeneId) return null;
  const url = buildAllianceApiUrl(`/gene/${encodeURIComponent(normalizedGeneId)}`);
  try {
    const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 });
    return { url, data };
  } catch (error) {
    const message = String(error?.message || "");
    if (message.includes("Request failed (404)")) return null;
    throw error;
  }
}

function scoreAllianceGeneSearchResult(item, query, targetTaxon = "NCBITaxon:9606", preferredSymbol = "") {
  if (!item || typeof item !== "object") return Number.NEGATIVE_INFINITY;
  const normalizedQuery = normalizeWhitespace(query || "").toUpperCase();
  const preferred = normalizeWhitespace(preferredSymbol || "").toUpperCase();
  const symbol = normalizeWhitespace(item?.symbol || "").toUpperCase();
  const name = normalizeWhitespace(item?.name || "").toUpperCase();
  const id = normalizeWhitespace(item?.id || item?.primaryKey || "").toUpperCase();
  const synonyms = asArray(item?.synonyms).map((value) => normalizeWhitespace(value).toUpperCase()).filter(Boolean);
  let score = toFiniteNumber(item?.score, 0);
  if (symbol && symbol === normalizedQuery) score += 180;
  if (preferred && symbol === preferred) score += 220;
  if (id && id === normalizedQuery) score += 260;
  if (name && name === normalizedQuery) score += 90;
  if (synonyms.includes(normalizedQuery)) score += 65;
  if (synonyms.includes(preferred) && preferred) score += 80;
  if (symbol && normalizedQuery && symbol.includes(normalizedQuery)) score += 25;
  if (allianceSpeciesMatchesTarget(item?.species || "", targetTaxon)) score += 110;
  return score;
}

async function searchAllianceGenes(query, { limit = 8 } = {}) {
  const params = new URLSearchParams({
    q: query,
    category: "gene",
    limit: String(Math.max(1, Math.min(15, Math.round(limit || 8)))),
  });
  return fetchAllianceJson("/search", params, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 });
}

async function resolveAllianceGeneSelection(query, species = "human") {
  const normalizedQuery = normalizeWhitespace(query || "");
  const targetTaxon = normalizeAllianceSpeciesTaxon(species);
  if (!normalizedQuery) {
    return { selected: null, geneRecord: null, searchUrl: "", candidates: [], targetTaxon };
  }

  const directRecord = normalizedQuery.includes(":")
    ? await fetchAllianceGeneRecordById(normalizedQuery)
    : null;
  if (directRecord?.data?.id) {
    return {
      selected: {
        id: normalizeWhitespace(directRecord.data.id || normalizedQuery),
        symbol: normalizeWhitespace(directRecord.data.symbol || ""),
        species: normalizeWhitespace(directRecord.data?.species?.name || ""),
      },
      geneRecord: directRecord.data,
      searchUrl: "",
      candidates: [],
      targetTaxon,
    };
  }

  let resolvedSymbol = "";
  if (!normalizedQuery.includes(":") && targetTaxon === "NCBITaxon:9606") {
    try {
      const resolved = await resolveGeneWithMyGene(normalizedQuery, "human");
      resolvedSymbol = normalizeMyGeneIds(resolved?.bestHit || {}).symbol || "";
    } catch {
      // Continue with AGR search if MyGene resolution fails.
    }
  }

  const searchTerms = dedupeArray([resolvedSymbol, normalizedQuery]).filter(Boolean);
  const allCandidates = [];
  let searchUrl = "";
  for (const term of searchTerms) {
    const search = await searchAllianceGenes(term, { limit: 10 });
    if (!searchUrl) searchUrl = search.url;
    for (const item of asArray(search?.data?.results)) {
      if (normalizeWhitespace(item?.category || "").toLowerCase() !== "gene") continue;
      allCandidates.push(item);
    }
  }

  const dedupedCandidates = [];
  const seenIds = new Set();
  for (const candidate of allCandidates) {
    const id = normalizeWhitespace(candidate?.id || candidate?.primaryKey || "");
    if (!id) continue;
    const key = id.toUpperCase();
    if (seenIds.has(key)) continue;
    seenIds.add(key);
    dedupedCandidates.push(candidate);
  }

  const selected = dedupedCandidates
    .slice()
    .sort((a, b) =>
      scoreAllianceGeneSearchResult(b, normalizedQuery, targetTaxon, resolvedSymbol)
      - scoreAllianceGeneSearchResult(a, normalizedQuery, targetTaxon, resolvedSymbol)
    )[0] || null;

  if (!selected?.id) {
    return {
      selected: null,
      geneRecord: null,
      searchUrl,
      candidates: dedupedCandidates.slice(0, 5),
      targetTaxon,
    };
  }

  const geneRecord = await fetchAllianceGeneRecordById(selected.id);
  return {
    selected,
    geneRecord: geneRecord?.data || null,
    searchUrl,
    candidates: dedupedCandidates.slice(0, 5),
    targetTaxon,
  };
}

function extractAllianceOrthologRows(payload, targetTaxon) {
  const results = asArray(payload?.results);
  const rows = [];
  const seen = new Set();
  for (const row of results) {
    const orthology = row?.geneToGeneOrthologyGenerated || {};
    const objectGene = orthology?.objectGene || {};
    const orthologId = normalizeWhitespace(objectGene?.primaryExternalId || "");
    if (!orthologId || orthologId.toUpperCase() === normalizeWhitespace(row?.gene?.primaryExternalId || "").toUpperCase()) continue;
    const taxonCurie = normalizeWhitespace(objectGene?.taxon?.curie || "");
    const species = normalizeWhitespace(objectGene?.taxon?.name || "");
    if (taxonCurie === targetTaxon) continue;
    const key = orthologId.toUpperCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const annotationFlags = row?.geneAnnotationsMap?.[orthologId] || {};
    rows.push({
      id: orthologId,
      symbol: normalizeWhitespace(objectGene?.geneSymbol?.displayText || ""),
      species,
      taxonCurie,
      confidence: normalizeWhitespace(orthology?.confidence?.name || ""),
      bestForward: normalizeWhitespace(orthology?.isBestScore?.name || ""),
      bestReverse: normalizeWhitespace(orthology?.isBestScoreReverse?.name || ""),
      methods: dedupeArray(asArray(orthology?.predictionMethodsMatched).map((value) => normalizeWhitespace(value?.name || "")).filter(Boolean)),
      hasDiseaseAnnotations: Boolean(annotationFlags?.hasDiseaseAnnotations),
      hasExpressionAnnotations: Boolean(annotationFlags?.hasExpressionAnnotations),
    });
  }

  return rows.sort((a, b) =>
    allianceModelSpeciesRank(a.taxonCurie, a.species) - allianceModelSpeciesRank(b.taxonCurie, b.species)
    || b.methods.length - a.methods.length
    || a.symbol.localeCompare(b.symbol)
  );
}

function inferAllianceProviderSpecies(providerAbbreviation) {
  const provider = normalizeWhitespace(providerAbbreviation || "").toUpperCase();
  const mapping = {
    MGI: "Mus musculus",
    RGD: "Rattus norvegicus",
    ZFIN: "Danio rerio",
    FB: "Drosophila melanogaster",
    WB: "Caenorhabditis elegans",
    SGD: "Saccharomyces cerevisiae",
    XB: "Xenopus",
    XENBASE: "Xenopus",
  };
  return mapping[provider] || provider || "Unknown species";
}

function extractAllianceModelRows(payload) {
  const results = asArray(payload?.results);
  return results.map((row) => {
    const provider = normalizeWhitespace(row?.dataProvider || row?.model?.dataProvider?.abbreviation || "");
    const diseaseModels = asArray(row?.diseaseModels).map((entry) => normalizeWhitespace(entry?.disease?.name || entry?.diseaseModel || "")).filter(Boolean);
    return {
      modelId: normalizeWhitespace(row?.model?.primaryExternalId || ""),
      modelName: normalizeWhitespace(row?.model?.agmFullName?.formatText || row?.model?.agmFullName?.displayText || ""),
      provider,
      species: inferAllianceProviderSpecies(provider),
      diseaseModels: dedupeArray(diseaseModels).slice(0, 4),
      modifierRelationshipTypes: dedupeArray(asArray(row?.modifierRelationshipTypes).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 4),
      hasDiseaseAnnotations: Boolean(row?.hasDiseaseAnnotations),
      hasPhenotypeAnnotations: Boolean(row?.hasPhenotypeAnnotations),
    };
  }).sort((a, b) =>
    allianceModelSpeciesRank("", a.species) - allianceModelSpeciesRank("", b.species)
    || a.provider.localeCompare(b.provider)
    || a.modelId.localeCompare(b.modelId)
  );
}

function parseCsvObjectsAfterHeaderRow(text, firstHeaderCell) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
  const headerIndex = lines.findIndex((line) => {
    const cells = parseCsvLine(line).map((value) => value.trim());
    return cells[0] === firstHeaderCell;
  });
  if (headerIndex < 0) return [];
  return parseCsvObjects(lines.slice(headerIndex).join("\n"));
}

function normalizeGtopdbSpecies(value) {
  const raw = normalizeWhitespace(value || "");
  if (!raw) return "";
  const lower = raw.toLowerCase();
  if (lower === "human") return "Human";
  if (lower === "mouse") return "Mouse";
  if (lower === "rat") return "Rat";
  return raw;
}

function scoreGtopdbTarget(target, query) {
  const normalizedQuery = normalizeWhitespace(query || "").toLowerCase();
  let score = 0;
  const name = normalizeWhitespace(target?.name || "").toLowerCase();
  const abbreviation = normalizeWhitespace(target?.abbreviation || "").toLowerCase();
  if (abbreviation === normalizedQuery) score += 100;
  if (name === normalizedQuery) score += 80;
  if (abbreviation.includes(normalizedQuery) && normalizedQuery) score += 30;
  if (name.includes(normalizedQuery) && normalizedQuery) score += 15;
  return score;
}

function normalizeDailymedTitle(title) {
  return normalizeWhitespace(String(title || "").replace(/\s*\[[^\]]+\]\s*$/, ""));
}

function parsePublishedDate(value) {
  const timestamp = Date.parse(String(value || ""));
  return Number.isFinite(timestamp) ? timestamp : 0;
}

function extractDailymedSectionText(xml, displayNames, fallbackTitlePatterns = []) {
  const sections = String(xml || "").split(/<section\b[^>]*>/i).slice(1);
  for (const section of sections) {
    const hasDisplayName = displayNames.some((name) => section.includes(`displayName="${name}"`));
    const titleMatch = section.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
    const titleText = xmlToReadableText(titleMatch?.[1] || "");
    const hasTitlePattern = fallbackTitlePatterns.some((pattern) => pattern.test(titleText));
    if (!hasDisplayName && !hasTitlePattern) continue;
    const readable = xmlToReadableText(section);
    if (!readable) continue;
    return compactErrorMessage(readable, 500);
  }
  return "";
}

function extractDailymedIngredientNames(xml, limit = 6) {
  const matches = [...String(xml || "").matchAll(/<ingredientSubstance>[\s\S]*?<name>([\s\S]*?)<\/name>/gi)];
  return dedupeArray(matches.map((match) => sanitizeXmlText(match[1] || "")).filter(Boolean)).slice(0, limit);
}

function extractDailymedProductNames(xml, limit = 4) {
  const matches = [...String(xml || "").matchAll(/<manufacturedProduct>[\s\S]*?<name>([\s\S]*?)<\/name>/gi)];
  return dedupeArray(matches.map((match) => sanitizeXmlText(match[1] || "")).filter(Boolean)).slice(0, limit);
}

async function fetchClinGenGeneValidityRows() {
  const cached = getFreshCacheValue(clinGenGeneValidityCache, 12 * 60 * 60 * 1000);
  if (cached) return cached;
  const response = await fetchWithRetry(CLINGEN_GENE_VALIDITY_DOWNLOAD, { retries: 1, timeoutMs: 30000, maxBackoffMs: 2500 });
  const text = await response.text();
  const rows = parseCsvObjectsAfterHeaderRow(text, "GENE SYMBOL");
  clinGenGeneValidityCache = storeCacheValue(clinGenGeneValidityCache, rows);
  return rows;
}

async function fetchClinGenDosageRows() {
  const cached = getFreshCacheValue(clinGenDosageCache, 12 * 60 * 60 * 1000);
  if (cached) return cached;
  const response = await fetchWithRetry(CLINGEN_GENE_DOSAGE_DOWNLOAD, { retries: 1, timeoutMs: 30000, maxBackoffMs: 2500 });
  const text = await response.text();
  const rows = parseCsvObjectsAfterHeaderRow(text, "GENE SYMBOL");
  clinGenDosageCache = storeCacheValue(clinGenDosageCache, rows);
  return rows;
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
      const [job, createResponse] = await client.createQueryJob(jobOptions);
      const initialStats = createResponse?.statistics?.query || job?.metadata?.statistics?.query || {};

      if (Boolean(dryRun)) {
        const totalBytesProcessed = Number(initialStats?.totalBytesProcessed || 0);
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
      const [metadata] = await job.getMetadata();
      const stats = metadata?.statistics?.query || initialStats;
      const totalBytesProcessed = Number(stats?.totalBytesProcessed || 0);
      const totalBytesBilled = Number(stats?.totalBytesBilled || 0);
      const cacheHit = Boolean(stats?.cacheHit);
      const previewLines = (rows || [])
        .slice(0, boundedRows)
        .map((row, idx) => `${idx + 1}. ${JSON.stringify(row)}`);
      const previewTextRaw = previewLines.join("\n");
      const previewText =
        previewTextRaw.length > 7000 ? `${previewTextRaw.slice(0, 7000)}\n... (truncated)` : previewTextRaw;

      const rowCount = rows?.length || 0;
      const bqSummary = buildBigQuerySummary(rows, referencedDatasets, rowCount);

      return {
        content: [
          {
            type: "text",
            text: `${renderStructuredResponse({
              summary: bqSummary,
              keyFields: [
                `Datasets referenced: ${referencedDatasets.join(", ")}`,
                `Rows returned: ${rowCount}`,
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
      const message = normalizeWhitespace(error?.message || "unknown BigQuery error");
      const hint = buildBigQueryErrorHint(normalizedSql, message);
      return {
        content: [{
          type: "text",
          text: `Error in run_bigquery_select_query: ${message}${hint ? `\nHint: ${hint}` : ""}`,
        }],
      };
    }
  }
);

// ============================================
// TOOL 7: Open Targets association score lookup
// ============================================
server.registerTool(
  "get_open_targets_association",
  {
    description:
      "Looks up a target-disease association score from Open Targets using the official EBI parquet release archive. Prefer this over BigQuery/current Open Targets sources when the question pins a specific release such as 'September 2025' or asks for an association score for one target-disease pair.",
    inputSchema: {
      target: z
        .string()
        .describe("Gene symbol or Ensembl gene ID (for example 'HTRA1' or 'ENSG00000166033')."),
      disease: z
        .string()
        .describe("Disease, phenotype, or trait label as used by Open Targets (for example 'vital capacity')."),
      release: z
        .string()
        .optional()
        .default("latest")
        .describe("Open Targets release to query. Use YY.MM (for example '25.09') or a natural label like 'September 2025'. Use 'latest' when no release is specified."),
      maxDiseaseMatches: z
        .number()
        .int()
        .min(1)
        .max(10)
        .optional()
        .default(5)
        .describe("How many candidate disease matches to retain for debugging ambiguous disease names."),
    },
  },
  async ({ target, disease, release = "latest", maxDiseaseMatches = 5 }) => {
    const result = await runPythonJsonHelper(
      OPEN_TARGETS_RELEASE_QUERY_SCRIPT,
      {
        target,
        disease,
        release,
        max_disease_matches: maxDiseaseMatches,
      },
      { timeoutMs: 180000 }
    );

    if (!result?.ok) {
      return {
        content: [
          {
            type: "text",
            text: `Error in get_open_targets_association: ${compactErrorMessage(result?.error || "unknown error", 320)}`,
          },
        ],
      };
    }

    const releaseTag = normalizeWhitespace(result.release || release || "");
    const targetId = normalizeWhitespace(result.target_id || "");
    const targetSymbol = normalizeWhitespace(result.target_symbol || target || "");
    const targetName = normalizeWhitespace(result.target_name || "");
    const diseaseId = normalizeWhitespace(result.disease_id || "");
    const diseaseName = normalizeWhitespace(result.disease_name || disease || "");
    const scoreValue = Number(result.score);
    const evidenceCount = Number(result.evidence_count);
    const candidateDiseases = Array.isArray(result.candidate_diseases) ? result.candidate_diseases : [];

    const keyFields = [
      `Release: ${releaseTag || "latest archived release"}`,
      `Target: ${targetSymbol}${targetId ? ` (${targetId})` : ""}${targetName ? ` — ${targetName}` : ""}`,
      `Disease: ${diseaseName}${diseaseId ? ` (${diseaseId})` : ""}`,
      `Association score: ${Number.isFinite(scoreValue) ? String(scoreValue) : normalizeWhitespace(result.score || "")}`,
      `Evidence count: ${Number.isFinite(evidenceCount) ? String(evidenceCount) : normalizeWhitespace(result.evidence_count || "")}`,
    ];

    if (candidateDiseases.length > 1) {
      const preview = candidateDiseases
        .slice(0, 3)
        .map((entry) => normalizeWhitespace(entry?.disease_name || ""))
        .filter(Boolean)
        .join("; ");
      if (preview) {
        keyFields.push(`Top disease matches considered: ${preview}`);
      }
    }

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: `Open Targets association_overall_direct score for ${targetSymbol || targetId} and ${diseaseName} in release ${releaseTag || "latest archived release"} is ${Number.isFinite(scoreValue) ? String(scoreValue) : "unavailable"}.`,
            keyFields,
            sources: [
              normalizeWhitespace(result.association_source_url || ""),
              normalizeWhitespace(result.disease_resolution_source || ""),
              normalizeWhitespace(result.target_resolution_source || ""),
            ].filter(Boolean),
            limitations: [
              "This uses the official archived parquet release files rather than the current live API or BigQuery mirror.",
              "The lookup resolves one target-disease pair at a time.",
            ],
          }),
        },
      ],
    };
  }
);

// ============================================
// TOOL 8: Open Targets L2G lookup
// ============================================
server.registerTool(
  "get_open_targets_l2g",
  {
    description:
      "Looks up an Open Targets Locus-to-Gene (L2G) score for a target-disease pair using the official archived platform release files. Prefer this over BigQuery/current Open Targets sources for genetics questions about L2G, credible sets, study-locus rows, or release-specific Open Targets genetics results.",
    inputSchema: {
      target: z
        .string()
        .describe("Gene symbol or Ensembl gene ID for the candidate causal gene (for example 'SEMA7A' or 'ENSG00000138623')."),
      disease: z
        .string()
        .describe("Disease, phenotype, or trait label as used by Open Targets (for example 'caffeine metabolite measurement')."),
      variant: z
        .string()
        .optional()
        .describe("Optional variant constraint to disambiguate the study-locus. Use rsID or Open Targets variant ID such as '15_74490015_G_A'."),
      release: z
        .string()
        .optional()
        .default("latest")
        .describe("Open Targets release to query. Use YY.MM (for example '25.09') or a natural label like 'September 2025'. Uses the latest archived release when omitted unless OPEN_TARGETS_L2G_DEFAULT_RELEASE is set."),
      maxDiseaseMatches: z
        .number()
        .int()
        .min(1)
        .max(10)
        .optional()
        .default(5)
        .describe("How many candidate disease matches to retain for debugging ambiguous disease names."),
      maxStudyMatches: z
        .number()
        .int()
        .min(1)
        .max(25)
        .optional()
        .default(10)
        .describe("How many candidate Open Targets studies to retain before evaluating credible sets."),
    },
  },
  async ({ target, disease, variant, release = "latest", maxDiseaseMatches = 5, maxStudyMatches = 10 }) => {
    const result = await runPythonJsonHelper(
      OPEN_TARGETS_L2G_QUERY_SCRIPT,
      {
        target,
        disease,
        variant,
        release,
        max_disease_matches: maxDiseaseMatches,
        max_study_matches: maxStudyMatches,
      },
      { timeoutMs: 240000 }
    );

    if (!result?.ok) {
      return {
        content: [
          {
            type: "text",
            text: `Error in get_open_targets_l2g: ${compactErrorMessage(result?.error || "unknown error", 320)}`,
          },
        ],
      };
    }

    const releaseTag = normalizeWhitespace(result.release || release || "");
    const targetId = normalizeWhitespace(result.target_id || "");
    const targetSymbol = normalizeWhitespace(result.target_symbol || target || "");
    const targetName = normalizeWhitespace(result.target_name || "");
    const diseaseId = normalizeWhitespace(result.disease_id || "");
    const diseaseName = normalizeWhitespace(result.disease_name || disease || "");
    const studyId = normalizeWhitespace(result.study_id || "");
    const traitFromSource = normalizeWhitespace(result.trait_from_source || "");
    const studyLocusId = normalizeWhitespace(result.study_locus_id || "");
    const variantId = normalizeWhitespace(result.variant_id || "");
    const rsIds = Array.isArray(result.rs_ids)
      ? result.rs_ids.map((entry) => normalizeWhitespace(entry)).filter(Boolean)
      : [];
    const l2gScore = Number(result.score);
    const roundedScore = Number(result.score_rounded_3dp);
    const candidateMatches = Array.isArray(result.candidate_matches) ? result.candidate_matches : [];

    const keyFields = [
      `Release: ${releaseTag || "latest archived release"}`,
      `Target: ${targetSymbol}${targetId ? ` (${targetId})` : ""}${targetName ? ` — ${targetName}` : ""}`,
      `Disease: ${diseaseName}${diseaseId ? ` (${diseaseId})` : ""}`,
      `L2G score: ${Number.isFinite(l2gScore) ? String(l2gScore) : normalizeWhitespace(result.score || "")}`,
      `Rounded L2G score (3 d.p.): ${Number.isFinite(roundedScore) ? roundedScore.toFixed(3) : "unavailable"}`,
      `Study: ${studyId || "unavailable"}${traitFromSource ? ` — ${traitFromSource}` : ""}`,
      `Study locus: ${studyLocusId || "unavailable"}`,
      `Variant: ${variantId || "unavailable"}${rsIds.length ? ` (${rsIds.join(", ")})` : ""}`,
    ];

    if (candidateMatches.length > 1) {
      const preview = candidateMatches
        .slice(0, 3)
        .map((entry) => {
          const score = Number(entry?.score);
          const trait = normalizeWhitespace(entry?.trait_from_source || "");
          const variantPreview = normalizeWhitespace(entry?.variant_id || "");
          return `${trait || "unknown trait"}${variantPreview ? ` [${variantPreview}]` : ""}${Number.isFinite(score) ? ` = ${score}` : ""}`;
        })
        .filter(Boolean)
        .join("; ");
      if (preview) {
        keyFields.push(`Top L2G matches considered: ${preview}`);
      }
    }

    return {
      content: [
        {
          type: "text",
          text: renderStructuredResponse({
            summary: `Open Targets L2G score for ${targetSymbol || targetId} and ${diseaseName} in release ${releaseTag || "latest archived release"} is ${Number.isFinite(l2gScore) ? String(l2gScore) : "unavailable"}.`,
            keyFields,
            sources: [
              normalizeWhitespace(result.l2g_source_url || ""),
              normalizeWhitespace(result.credible_set_source_url || ""),
              normalizeWhitespace(result.study_source_url || ""),
              normalizeWhitespace(result.disease_resolution_source || ""),
              normalizeWhitespace(result.target_resolution_source || ""),
            ].filter(Boolean),
            limitations: [
              "This uses the official archived Open Targets release files rather than the current live API or BigQuery mirror.",
              "The lookup resolves one target-disease pair at a time and picks the best matching study-locus row when multiple candidates exist.",
            ],
          }),
        },
      ],
    };
  }
);

// ============================================
// TOOL 9: Benchmark dataset overview (GPQA, PubMedQA, BioASQ)
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
    const normalizedStatus = normalizeClinicalTrialStatus(status);
    const isExactNctQuery = /^NCT\d{8}$/i.test(normalizeWhitespace(query || ""));
    const minimumLimit = isExactNctQuery ? 1 : 5;
    const boundedLimit = limit ? Math.max(minimumLimit, Math.min(200, Math.round(limit))) : 50;

    try {
      const collected = await collectClinicalTrialStudies({
        query,
        normalizedStatus,
        maxStudies: boundedLimit,
        maxPagesPerVariant: Math.max(2, Math.ceil(boundedLimit / 50) + 1),
        fetchOptions: { retries: 2, timeoutMs: 15000, maxBackoffMs: 3500 },
      });
      const studies = collected.studies;
      const resultCount = Number.isFinite(collected.totalCount) ? collected.totalCount : 0;
      const hasMorePages = Boolean(collected.hasMorePages);
      studies.sort((left, right) => scoreClinicalTrialStudy(right, query) - scoreClinicalTrialStudy(left, query));
      // Do NOT set resultCount = studies.length when API did not provide totalCount —
      // that would imply "Showing 100 of 100 total" when there may be many more.
      if (studies.length === 0) {
        return {
          content: [
            {
              type: "text",
              text: `No clinical trials found for: "${query}"${normalizedStatus ? ` with status ${normalizedStatus}` : ""}`,
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

      const countLine = resultCount
        ? `Showing ${studies.length} of ${resultCount} total trials`
        : `${studies.length} trials returned (total in registry not provided by API; more may exist — increase limit to fetch more)`;
      const fallbackNote = collected.usedFallbackVariant
        ? "\nQuery normalization fallback also checked a punctuation-normalized variant of the same search to recover additional registry matches."
        : "";
      return {
        content: [
          {
            type: "text",
            text: `Clinical trials for "${query}":\n${countLine}${hasMorePages ? " (additional pages may be available)" : ""}${normalizedStatus ? `\nStatus filter: ${normalizedStatus}` : ""}${fallbackNote}\n\n${formatted}\n\nUse get_clinical_trial with the NCT ID for full details including results.`,
          },
        ],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error searching clinical trials: ${error.message}. Try again or use different search terms.` }],
      };
    }
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

function normalizePmidValue(rawPmid) {
  const value = normalizeWhitespace(rawPmid || "").replace(/^PMID:\s*/i, "");
  const match = value.match(/^(\d+)$/);
  return match?.[1] || "";
}

function normalizePmcidValue(rawPmcid) {
  const value = normalizeWhitespace(rawPmcid || "")
    .replace(/^PMCID:\s*/i, "")
    .toUpperCase()
    .replace(/\s+/g, "");
  const match = value.match(/^(PMC\d+)$/);
  return match?.[1] || "";
}

function formatAuthorList(authors, maxAuthors = 5) {
  const names = Array.isArray(authors)
    ? authors
        .map((author) => normalizeWhitespace(author?.name || author))
        .filter(Boolean)
        .slice(0, maxAuthors)
    : [];
  if (names.length === 0) return "Unknown";
  const hasMore = Array.isArray(authors) && authors.length > maxAuthors;
  return `${names.join(", ")}${hasMore ? " et al." : ""}`;
}

function parsePubmedRecordMetadata(xml) {
  const title = sanitizeXmlText(xml.match(/<ArticleTitle>([\s\S]*?)<\/ArticleTitle>/)?.[1] || "Untitled");
  const abstractParts = [...xml.matchAll(/<AbstractText[^>]*>([\s\S]*?)<\/AbstractText>/g)]
    .map((m) => sanitizeXmlText(m[1]))
    .filter(Boolean);
  const journal = sanitizeXmlText(xml.match(/<Title>([\s\S]*?)<\/Title>/)?.[1] || "");
  const year = xml.match(/<PubDate>[\s\S]*?<Year>(\d{4})<\/Year>/)?.[1] || "";
  const doi = normalizeDoiValue(xml.match(/<ArticleId IdType="doi">([\s\S]*?)<\/ArticleId>/)?.[1] || "");
  const pmcId = normalizePmcidValue(xml.match(/<ArticleId IdType="pmc">([\s\S]*?)<\/ArticleId>/)?.[1] || "");
  const pmid = normalizePmidValue(xml.match(/<PMID[^>]*>(\d+)<\/PMID>/)?.[1] || "");
  const authors = parsePubmedAuthors(xml);
  const meshTerms = [...xml.matchAll(/<DescriptorName[^>]*>([\s\S]*?)<\/DescriptorName>/g)]
    .map((m) => sanitizeXmlText(m[1]))
    .filter(Boolean)
    .slice(0, 15);
  const pubTypes = [...xml.matchAll(/<PublicationType[^>]*>([\s\S]*?)<\/PublicationType>/g)]
    .map((m) => sanitizeXmlText(m[1]))
    .filter(Boolean);
  return {
    title,
    abstractText: abstractParts.join("\n\n") || "No abstract available.",
    journal,
    year,
    doi,
    pmcId,
    pmid,
    authors,
    meshTerms,
    pubTypes,
  };
}

function parsePmcAuthors(xml) {
  const authorBlocks = [...xml.matchAll(/<contrib\b[^>]*contrib-type="author"[^>]*>([\s\S]*?)<\/contrib>/gi)];
  return authorBlocks.map((match) => {
    const block = match[1];
    const givenNames = sanitizeXmlText(block.match(/<given-names[^>]*>([\s\S]*?)<\/given-names>/i)?.[1] || "");
    const surname = sanitizeXmlText(block.match(/<surname[^>]*>([\s\S]*?)<\/surname>/i)?.[1] || "");
    const collab = sanitizeXmlText(block.match(/<collab[^>]*>([\s\S]*?)<\/collab>/i)?.[1] || "");
    const name = normalizeWhitespace(`${givenNames} ${surname}`) || collab || "Unknown";
    return { name };
  });
}

function parsePmcArticle(xml) {
  const articleTitle =
    sanitizeXmlText(xml.match(/<article-title[^>]*>([\s\S]*?)<\/article-title>/i)?.[1] || "") || "Untitled";
  const abstractXml = xml.match(/<abstract[^>]*>([\s\S]*?)<\/abstract>/i)?.[1] || "";
  const bodyXml = xml.match(/<body[^>]*>([\s\S]*?)<\/body>/i)?.[1] || "";
  const journal = sanitizeXmlText(xml.match(/<journal-title[^>]*>([\s\S]*?)<\/journal-title>/i)?.[1] || "");
  const year = xml.match(/<pub-date[\s\S]*?<year>(\d{4})<\/year>/i)?.[1] || "";
  const doi = normalizeDoiValue(
    xml.match(/<article-id[^>]*pub-id-type="doi"[^>]*>([\s\S]*?)<\/article-id>/i)?.[1] || ""
  );
  const pmid = normalizePmidValue(
    xml.match(/<article-id[^>]*pub-id-type="pmid"[^>]*>([\s\S]*?)<\/article-id>/i)?.[1] || ""
  );
  const pmcid = normalizePmcidValue(
    xml.match(/<article-id[^>]*pub-id-type="pmcid"[^>]*>([\s\S]*?)<\/article-id>/i)?.[1] ||
      xml.match(/<article-id[^>]*pub-id-type="pmc"[^>]*>([\s\S]*?)<\/article-id>/i)?.[1] ||
      ""
  );
  const authors = parsePmcAuthors(xml);
  const licenseText = xmlToReadableText(
    xml.match(/<license-p[^>]*>([\s\S]*?)<\/license-p>/i)?.[1] ||
      xml.match(/<permissions[^>]*>([\s\S]*?)<\/permissions>/i)?.[1] ||
      ""
  );
  const sectionTitles = [...bodyXml.matchAll(/<title[^>]*>([\s\S]*?)<\/title>/gi)]
    .map((match) => sanitizeXmlText(match[1]))
    .map((title) => normalizeWhitespace(title))
    .filter(Boolean);
  return {
    title: articleTitle,
    abstractText: xmlToReadableText(abstractXml) || "No abstract available.",
    fullText: xmlToReadableText(bodyXml) || "",
    journal,
    year,
    doi,
    pmid,
    pmcid,
    authors,
    licenseText,
    sectionTitles: [...new Set(sectionTitles)].slice(0, 20),
  };
}

// ============================================
// TOOL: Search PubMed
// ============================================
const PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
const PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";
const PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi";

async function resolvePmidFromDoi(rawDoi) {
  const doi = normalizeDoiValue(rawDoi);
  if (!doi) return "";
  const params = new URLSearchParams({
    db: "pubmed",
    term: `${doi}[doi]`,
    retmode: "json",
    retmax: "1",
  });
  const url = buildNcbiEutilsUrl(PUBMED_ESEARCH, params);
  const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 10000, maxBackoffMs: 2500 });
  return normalizePmidValue(data?.esearchresult?.idlist?.[0] || "");
}

async function fetchPubmedArticleXml(cleanPmid) {
  const params = new URLSearchParams({
    db: "pubmed",
    id: cleanPmid,
    retmode: "xml",
    rettype: "abstract",
  });
  const url = buildNcbiEutilsUrl(PUBMED_EFETCH, params);
  const resp = await fetchWithRetry(url, { retries: 2, timeoutMs: 12000, maxBackoffMs: 2500 });
  return resp.text();
}

async function fetchPmcFullTextXml(cleanPmcid) {
  const pmcid = normalizePmcidValue(cleanPmcid);
  if (!pmcid) {
    throw new Error("A valid PMCID is required to fetch PMC full text.");
  }
  const candidateUrls = [
    buildNcbiEutilsUrl(PUBMED_EFETCH, new URLSearchParams({
      db: "pmc",
      id: pmcid,
      retmode: "xml",
    })),
    `https://www.ebi.ac.uk/europepmc/webservices/rest/${encodeURIComponent(pmcid)}/fullTextXML`,
  ];
  let lastError = null;
  for (const url of candidateUrls) {
    try {
      const resp = await fetchWithRetry(url, { retries: 1, timeoutMs: 15000, maxBackoffMs: 3000 });
      const xml = await resp.text();
      if (xml.includes("<article") || xml.includes("<pmc-articleset")) {
        return { xml, sourceUrl: url };
      }
      lastError = new Error(`Unexpected PMC response shape from ${url}`);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError || new Error(`PMC full text unavailable for ${pmcid}`);
}

function truncateText(value, maxCharacters) {
  const text = String(value || "").trim();
  if (!text || text.length <= maxCharacters) {
    return { text, truncated: false };
  }
  return {
    text: `${text.slice(0, maxCharacters).trim()}\n\n[Full text truncated at ${maxCharacters} characters. Increase maxCharacters to retrieve more.]`,
    truncated: true,
  };
}

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

function buildPubmedRateLimitHint() {
  const hints = [
    "PubMed E-utilities is rate-limited.",
    "Retry shortly or use search_europe_pmc_literature as a fallback for retrieval.",
  ];
  if (!NCBI_API_KEY) {
    hints.push("Configuring NCBI_API_KEY will substantially reduce PubMed throttling.");
  }
  return hints.join(" ");
}

function formatPubmedToolError(error) {
  const message = normalizeWhitespace(error?.message || String(error) || "unknown PubMed error");
  if (message.includes("429") || /rate limit/i.test(message)) {
    return `${message} ${buildPubmedRateLimitHint()}`;
  }
  return message;
}

async function fetchPubmedSummaryXml(pmids) {
  const cleanPmids = dedupeArray(
    (Array.isArray(pmids) ? pmids : []).map((pmid) => normalizePmidValue(pmid)).filter(Boolean)
  );
  if (cleanPmids.length === 0) {
    throw new Error("At least one PMID is required for PubMed summary lookup.");
  }
  const params = new URLSearchParams({
    db: "pubmed",
    id: cleanPmids.join(","),
    retmode: "xml",
  });
  const summaryUrl = buildNcbiEutilsUrl(PUBMED_ESUMMARY, params);
  const response = await fetchWithRetry(summaryUrl, { retries: 2, timeoutMs: 10000, maxBackoffMs: 2500 });
  const summaryXml = await response.text();
  if (!/<DocSum>/i.test(summaryXml)) {
    const preview = compactErrorMessage(summaryXml, 220);
    throw new Error(`PubMed summary lookup returned no DocSum records. Response preview: ${preview}`);
  }
  return summaryXml;
}

function looksLikeBareGeneSymbol(query) {
  const normalizedQuery = normalizeWhitespace(query || "");
  return Boolean(normalizedQuery)
    && !/\s/.test(normalizedQuery)
    && /^[A-Za-z0-9-]+$/.test(normalizedQuery)
    && normalizedQuery.length <= 20;
}

function scoreUniProtSearchResult(entry, query) {
  const normalizedQuery = normalizeWhitespace(query || "").toUpperCase();
  if (!normalizedQuery) return 0;

  const accession = normalizeWhitespace(entry?.primaryAccession || "").toUpperCase();
  const entryId = normalizeWhitespace(entry?.uniProtkbId || "").toUpperCase();
  const geneSymbols = extractUniProtGeneSymbols(entry, 10).map((value) => normalizeWhitespace(value).toUpperCase());
  const proteinName = extractUniProtProteinName(entry).toUpperCase();

  let score = 0;
  if (accession === normalizedQuery) score += 150;
  if (entryId === normalizedQuery) score += 120;
  if (geneSymbols.includes(normalizedQuery)) score += 100;
  if (geneSymbols[0] === normalizedQuery) score += 30;
  if (proteinName === normalizedQuery) score += 40;
  if (proteinName.includes(normalizedQuery)) score += 10;
  if (normalizeWhitespace(entry?.entryType || "").toLowerCase().includes("reviewed")) score += 5;
  return score;
}

function rankUniProtResults(results, query) {
  return [...(Array.isArray(results) ? results : [])].sort((left, right) => {
    const scoreDelta = scoreUniProtSearchResult(right, query) - scoreUniProtSearchResult(left, query);
    if (scoreDelta !== 0) return scoreDelta;
    return normalizeWhitespace(left?.primaryAccession || "").localeCompare(
      normalizeWhitespace(right?.primaryAccession || "")
    );
  });
}

const CLINICAL_QUERY_STOPWORDS = new Set([
  "and",
  "the",
  "with",
  "for",
  "study",
  "trial",
  "trials",
  "disease",
  "syndrome",
  "disorder",
  "phase",
  "biomarker",
  "biomarkers",
  "treatment",
]);

function tokenizeClinicalQuery(value) {
  return dedupeArray(
    normalizeWhitespace(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .split(/\s+/)
      .map((token) => token.trim())
      .filter((token) => token.length >= 3 && !CLINICAL_QUERY_STOPWORDS.has(token))
  );
}

function scoreClinicalTrialStudy(study, query) {
  const tokens = tokenizeClinicalQuery(query);
  if (tokens.length === 0) return 0;

  const protocol = study?.protocolSection || {};
  const title = normalizeWhitespace(protocol?.identificationModule?.briefTitle || "").toLowerCase();
  const conditions = normalizeWhitespace((protocol?.conditionsModule?.conditions || []).join(" ")).toLowerCase();
  const interventions = normalizeWhitespace(
    (protocol?.armsInterventionsModule?.interventions || []).map((item) => item?.name || "").join(" ")
  ).toLowerCase();
  const sponsor = normalizeWhitespace(protocol?.sponsorCollaboratorsModule?.leadSponsor?.name || "").toLowerCase();

  let score = 0;
  for (const token of tokens) {
    if (title.includes(token)) score += 4;
    if (conditions.includes(token)) score += 5;
    if (interventions.includes(token)) score += 7;
    if (sponsor.includes(token)) score += 1;
  }
  return score;
}

function buildClinicalTrialQueryVariants(query) {
  const normalized = normalizeWhitespace(query || "");
  if (!normalized) return [];
  const variants = [];
  const seen = new Set();
  const pushVariant = (value) => {
    const candidate = normalizeWhitespace(value || "");
    if (!candidate) return;
    const key = candidate.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    variants.push(candidate);
  };
  pushVariant(normalized);
  pushVariant(
    normalized
      .replace(/\b([A-Za-z]+)'s\b/g, "$1")
      .replace(/'/g, "")
  );
  return variants;
}

function getClinicalTrialStudyKey(study) {
  const protocol = study?.protocolSection || {};
  const identification = protocol?.identificationModule || {};
  const nctId = normalizeWhitespace(identification?.nctId || "").toUpperCase();
  if (nctId) return nctId;
  const orgId = normalizeWhitespace(identification?.orgStudyIdInfo?.id || "").toUpperCase();
  if (orgId) return `ORG:${orgId}`;
  const title = normalizeWhitespace(identification?.briefTitle || "").toUpperCase();
  return title || JSON.stringify(study || {});
}

async function collectClinicalTrialStudies({
  query,
  normalizedStatus = "",
  maxStudies = 50,
  maxPagesPerVariant = 3,
  fetchOptions = { retries: 2, timeoutMs: 15000, maxBackoffMs: 3500 },
}) {
  const boundedStudies = Math.max(1, Math.min(200, Math.round(Number(maxStudies) || 50)));
  const boundedPagesPerVariant = Math.max(1, Math.min(8, Math.round(Number(maxPagesPerVariant) || 3)));
  const queryVariants = buildClinicalTrialQueryVariants(query);
  const studies = [];
  const sources = [];
  const seenStudyKeys = new Set();
  let totalCount = null;
  let hasMorePages = false;
  let usedFallbackVariant = false;
  const usedQueryVariants = [];

  for (let variantIndex = 0; variantIndex < queryVariants.length && studies.length < boundedStudies; variantIndex++) {
    const queryVariant = queryVariants[variantIndex];
    usedQueryVariants.push(queryVariant);
    let nextPageToken = "";

    for (let page = 0; page < boundedPagesPerVariant && studies.length < boundedStudies; page++) {
      const pageSize = Math.min(50, boundedStudies - studies.length);
      if (pageSize <= 0) break;

      const params = new URLSearchParams({
        "query.term": queryVariant,
        pageSize: String(pageSize),
        format: "json",
      });
      if (normalizedStatus) {
        params.append("filter.overallStatus", normalizedStatus);
      }
      if (nextPageToken) {
        params.append("pageToken", nextPageToken);
      }

      const url = `${CLINICAL_TRIALS_API}/studies?${params.toString()}`;
      sources.push(url);
      const data = await fetchJsonWithRetry(url, fetchOptions);
      const pageStudies = Array.isArray(data?.studies) ? data.studies : [];

      if (Number.isFinite(data?.totalCount) && usedQueryVariants.length === 1) {
        totalCount = data.totalCount;
      }

      let uniqueAddedThisPage = 0;
      for (const study of pageStudies) {
        const studyKey = getClinicalTrialStudyKey(study);
        if (seenStudyKeys.has(studyKey)) continue;
        seenStudyKeys.add(studyKey);
        studies.push(study);
        uniqueAddedThisPage += 1;
        if (studies.length >= boundedStudies) break;
      }

      const returnedNextPageToken = data?.nextPageToken || "";
      if (returnedNextPageToken) {
        hasMorePages = true;
      }
      if (variantIndex > 0 && uniqueAddedThisPage > 0) {
        usedFallbackVariant = true;
      }

      const shouldContinue =
        Boolean(returnedNextPageToken)
        && pageStudies.length > 0
        && studies.length < boundedStudies
        && page < boundedPagesPerVariant - 1;
      if (!shouldContinue && returnedNextPageToken) {
        hasMorePages = true;
      }
      nextPageToken = shouldContinue ? returnedNextPageToken : "";
      if (!shouldContinue) {
        break;
      }
    }
  }

  if (usedQueryVariants.length > 1) {
    totalCount = null;
  }

  return {
    studies,
    sources,
    totalCount,
    hasMorePages,
    usedFallbackVariant,
    usedQueryVariants,
  };
}

function normalizeGeoIdentifier(value) {
  return normalizeWhitespace(value || "").toUpperCase().replace(/\s+/g, "");
}

function buildGeoRecordUrl(accession) {
  const cleanAccession = normalizeGeoIdentifier(accession);
  if (!cleanAccession) return "https://www.ncbi.nlm.nih.gov/geo/";
  return `https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=${encodeURIComponent(cleanAccession)}`;
}

function buildGeoSeriesFtpRoot(accession) {
  const cleanAccession = normalizeGeoIdentifier(accession);
  if (!/^GSE\d+$/.test(cleanAccession)) return "";
  const digits = cleanAccession.replace(/^GSE/i, "");
  const prefixDigits = digits.length > 3 ? digits.slice(0, -3) : digits;
  return `https://ftp.ncbi.nlm.nih.gov/geo/series/GSE${prefixDigits}nnn/${cleanAccession}`;
}

function buildGeoSupplementaryArchiveUrl(accession) {
  const root = buildGeoSeriesFtpRoot(accession);
  const cleanAccession = normalizeGeoIdentifier(accession);
  if (!root || !cleanAccession) return "";
  return `${root}/suppl/${cleanAccession}_RAW.tar`;
}

function buildGeoSupplementaryFileListUrl(accession) {
  const root = buildGeoSeriesFtpRoot(accession);
  return root ? `${root}/suppl/filelist.txt` : "";
}

function parseGeoFileList(text) {
  return String(text || "")
    .split(/\r?\n/)
    .slice(1)
    .map((line) => line.split("\t"))
    .filter((parts) => parts.length >= 5)
    .map((parts) => ({
      archive_or_file: normalizeWhitespace(parts[0]),
      name: normalizeWhitespace(parts[1]),
      size: normalizeWhitespace(parts[3]),
      type: normalizeWhitespace(parts[4]),
    }))
    .filter((row) => row.name);
}

function buildTarEntryIndex(buffer) {
  const entries = new Map();
  if (!Buffer.isBuffer(buffer)) return entries;
  let offset = 0;
  while (offset + 512 <= buffer.length) {
    const header = buffer.subarray(offset, offset + 512);
    if (header.every((byte) => byte === 0)) {
      break;
    }
    const name = header.subarray(0, 100).toString("utf8").replace(/\0.*$/, "").trim();
    const sizeOctal = header.subarray(124, 136).toString("utf8").replace(/\0.*$/, "").trim() || "0";
    const size = Number.parseInt(sizeOctal, 8) || 0;
    const contentStart = offset + 512;
    if (name) {
      entries.set(name, {
        name,
        size,
        contentStart,
        contentEnd: contentStart + size,
      });
    }
    offset = contentStart + Math.ceil(size / 512) * 512;
  }
  return entries;
}

async function fetchGeoSupplementaryArchive(accession) {
  const cleanAccession = normalizeGeoIdentifier(accession);
  const cacheKey = cleanAccession;
  const cached = getFreshCacheValue(geoSupplementaryArchiveCache.get(cacheKey), 6 * 60 * 60 * 1000);
  if (cached) {
    return cached;
  }

  const archiveUrl = buildGeoSupplementaryArchiveUrl(cleanAccession);
  if (!archiveUrl) {
    throw new Error(`Unsupported GEO series accession: ${accession}`);
  }
  const buffer = await fetchBufferWithRetry(archiveUrl, { retries: 1, timeoutMs: 120000, maxBackoffMs: 4000 });
  const payload = {
    archiveUrl,
    buffer,
    entryIndex: buildTarEntryIndex(buffer),
  };
  geoSupplementaryArchiveCache.set(cacheKey, storeCacheValue(null, payload));
  return payload;
}

function extractGeoTarEntryBuffer(archivePayload, memberName) {
  const entryIndex = archivePayload?.entryIndex;
  const buffer = archivePayload?.buffer;
  if (!(entryIndex instanceof Map) || !Buffer.isBuffer(buffer)) {
    return null;
  }
  const entry = entryIndex.get(memberName);
  if (!entry) return null;
  return buffer.subarray(entry.contentStart, entry.contentEnd);
}

function parseGeoSampleQuickMetadata(text) {
  const lines = String(text || "").split(/\r?\n/);
  const characteristics = {};
  let organism = "";
  let sourceName = "";
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    if (line.startsWith("!Sample_organism_ch1 = ")) {
      organism = normalizeWhitespace(line.split("=", 2)[1] || "");
      continue;
    }
    if (line.startsWith("!Sample_source_name_ch1 = ")) {
      sourceName = normalizeWhitespace(line.split("=", 2)[1] || "");
      continue;
    }
    if (!line.startsWith("!Sample_characteristics_ch1 = ")) {
      continue;
    }
    const value = normalizeWhitespace(line.split("=", 2)[1] || "");
    if (!value) continue;
    const separatorIndex = value.indexOf(":");
    if (separatorIndex === -1) {
      characteristics[value.toLowerCase()] = value;
      continue;
    }
    const key = normalizeWhitespace(value.slice(0, separatorIndex)).toLowerCase();
    const fieldValue = normalizeWhitespace(value.slice(separatorIndex + 1));
    if (key) {
      characteristics[key] = fieldValue;
    }
  }
  return {
    organism,
    source_name: sourceName,
    characteristics,
  };
}

async function fetchGeoSampleQuickMetadata(accession) {
  const cleanAccession = normalizeGeoIdentifier(accession);
  const cached = getFreshCacheValue(geoSampleQuickMetadataCache.get(cleanAccession), 6 * 60 * 60 * 1000);
  if (cached) {
    return cached;
  }
  const url = `${buildGeoRecordUrl(cleanAccession)}&targ=self&form=text&view=quick`;
  const response = await fetchWithRetry(url, { retries: 1, timeoutMs: 20000, maxBackoffMs: 2500 });
  const text = await response.text();
  const payload = parseGeoSampleQuickMetadata(text);
  geoSampleQuickMetadataCache.set(cleanAccession, storeCacheValue(null, payload));
  return payload;
}

function computeGeoClusterProportionsFromCountsCsv(csvText, clusterColumn = "assigned_cluster") {
  const lines = String(csvText || "")
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0);
  if (lines.length < 2) {
    throw new Error("The GEO supplementary CSV did not contain any data rows.");
  }
  const header = parseDelimitedLine(lines[0], ",");
  const clusterIndex = header.findIndex((column) => normalizeWhitespace(column).toLowerCase() === clusterColumn.toLowerCase());
  if (clusterIndex === -1) {
    throw new Error(`Column ${clusterColumn} was not found in the GEO supplementary CSV header.`);
  }

  const counts = new Map();
  let totalRows = 0;
  for (let idx = 1; idx < lines.length; idx += 1) {
    const values = parseDelimitedLine(lines[idx], ",");
    const cluster = normalizeWhitespace(values[clusterIndex] || "");
    if (!cluster) continue;
    totalRows += 1;
    counts.set(cluster, (counts.get(cluster) || 0) + 1);
  }
  if (totalRows === 0) {
    throw new Error("No labeled cells were found in the GEO supplementary CSV.");
  }

  const proportions = Array.from(counts.entries())
    .map(([cellType, count]) => ({
      cell_type: cellType,
      count,
      proportion: count / totalRows,
      proportion_rounded: Number((count / totalRows).toFixed(2)),
    }))
    .sort((left, right) => right.count - left.count);

  return {
    total_rows: totalRows,
    cluster_column: clusterColumn,
    counts: proportions,
  };
}

function buildGeoDocsumMap(summaryResult) {
  const uids = Array.isArray(summaryResult?.uids) ? summaryResult.uids : [];
  const docs = [];
  for (const uid of uids) {
    const doc = summaryResult?.[uid];
    if (doc && typeof doc === "object") {
      docs.push(doc);
    }
  }
  return docs;
}

async function fetchGeoDocSumsByIds(ids) {
  const cleanIds = [...new Set((Array.isArray(ids) ? ids : []).map((id) => normalizeWhitespace(id)).filter(Boolean))];
  if (cleanIds.length === 0) return [];
  const params = new URLSearchParams({
    db: "gds",
    id: cleanIds.join(","),
    retmode: "json",
  });
  const url = buildNcbiEutilsUrl(PUBMED_ESUMMARY, params);
  const data = await fetchJsonWithRetry(url, { retries: 2, timeoutMs: 10000, maxBackoffMs: 2500 });
  return buildGeoDocsumMap(data?.result);
}

async function resolveGeoDocSum(identifier) {
  const normalized = normalizeGeoIdentifier(identifier);
  if (!normalized) return null;

  if (/^\d+$/.test(normalized)) {
    const docs = await fetchGeoDocSumsByIds([normalized]);
    return docs[0] || null;
  }

  const params = new URLSearchParams({
    db: "gds",
    term: `${normalized}[ACCN]`,
    retmax: "20",
    retmode: "json",
  });
  const url = buildNcbiEutilsUrl(PUBMED_ESEARCH, params);
  const searchData = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 10000, maxBackoffMs: 2500 });
  const idList = searchData?.esearchresult?.idlist ?? [];
  if (!Array.isArray(idList) || idList.length === 0) {
    return null;
  }

  const docs = await fetchGeoDocSumsByIds(idList);
  const exact = docs.find((doc) => normalizeGeoIdentifier(doc?.accession) === normalized);
  if (exact) return exact;

  const numericSuffix = normalized.match(/^(GSE|GSM|GPL|GDS)(\d+)$/)?.[2] || "";
  if (numericSuffix) {
    const prefix = normalized.slice(0, 3);
    const byFamily = docs.find((doc) => {
      const docPrefix = normalizeGeoIdentifier(doc?.entrytype || "").slice(0, 3);
      const accessionSuffix = normalizeGeoIdentifier(doc?.accession).replace(/^[A-Z]+/, "");
      return accessionSuffix === numericSuffix && docPrefix === prefix;
    });
    if (byFamily) return byFamily;
  }

  return docs[0] || null;
}

function summarizeGeoDocsum(doc) {
  const accession = normalizeGeoIdentifier(doc?.accession || "");
  const title = normalizeWhitespace(doc?.title || "Untitled GEO record");
  const entryType = normalizeWhitespace(doc?.entrytype || "GEO");
  const datasetType = normalizeWhitespace(doc?.gdstype || "");
  const taxon = normalizeWhitespace(doc?.taxon || "");
  const nSamples = Number(doc?.n_samples || 0);
  const pdat = normalizeWhitespace(doc?.pdat || "");
  const pubmedIds = Array.isArray(doc?.pubmedids) ? doc.pubmedids.filter(Boolean).slice(0, 3) : [];
  const summary = normalizeWhitespace(doc?.summary || "");
  const summaryPreview = summary ? (summary.length > 220 ? `${summary.slice(0, 217)}...` : summary) : "No summary provided.";
  const parts = [
    accession || `UID:${normalizeWhitespace(doc?.uid || "") || "unknown"}`,
    `${entryType}${datasetType ? ` | ${datasetType}` : ""}`,
    taxon || "organism unknown",
    nSamples > 0 ? `${nSamples} sample(s)` : "sample count unavailable",
    pdat ? `updated ${pdat}` : "",
    title,
  ].filter(Boolean);
  if (pubmedIds.length > 0) {
    parts.push(`PMID:${pubmedIds.join(", ")}`);
  }
  parts.push(summaryPreview);
  return parts;
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

      const searchUrl = buildNcbiEutilsUrl(PUBMED_ESEARCH, params);
      const searchData = await fetchJsonWithRetry(searchUrl, { retries: 2, timeoutMs: 10000 });
      const idList = searchData?.esearchresult?.idlist ?? [];
      const totalCount = parseInt(searchData?.esearchresult?.count || "0", 10);

      if (idList.length === 0) {
        return {
          content: [{ type: "text", text: `No PubMed results found for: "${query}"${minDate ? ` (from ${minDate})` : ""}${maxDate ? ` (to ${maxDate})` : ""}. Try broader terms or remove date filters.` }],
        };
      }

      const summaryXml = await fetchPubmedSummaryXml(idList);
      const articles = parsePubmedArticleSummary(summaryXml);
      if (articles.length === 0 && totalCount > 0) {
        throw new Error(`PubMed summary parse returned zero article records despite ${totalCount} hit(s).`);
      }

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
          text: `PubMed search for "${query}":\nShowing ${articles.length} of ${totalCount} results${minDate ? ` | From: ${minDate}` : ""}${maxDate ? ` | To: ${maxDate}` : ""}\n\n${formatted}\n\nUse get_pubmed_abstract with a PMID for the abstract, or get_paper_fulltext for PMC full text when available.`,
        }],
      };
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: `Error searching PubMed: ${formatPubmedToolError(error)} Try different search terms.`,
        }],
      };
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
      const cleanPmid = normalizePmidValue(pmid);
      const xml = await fetchPubmedArticleXml(cleanPmid);

      if (!xml || xml.includes("<ERROR>") || !xml.includes("<PubmedArticle>")) {
        return { content: [{ type: "text", text: `PubMed article not found for PMID: ${cleanPmid}` }] };
      }

      const meta = parsePubmedRecordMetadata(xml);
      const authorStr = formatAuthorList(meta.authors);

      let text = `Title: ${meta.title}\nPMID: ${cleanPmid}`;
      if (meta.doi) text += ` | DOI: ${meta.doi}`;
      if (meta.pmcId) text += ` | ${meta.pmcId}`;
      text += `\nAuthors: ${authorStr}\nJournal: ${meta.journal} (${meta.year})`;
      if (meta.pubTypes.length) text += `\nType: ${meta.pubTypes.join(", ")}`;
      text += `\n\nAbstract:\n${meta.abstractText}`;
      if (meta.meshTerms.length) text += `\n\nMeSH Terms: ${meta.meshTerms.join("; ")}`;
      if (meta.pmcId) text += `\n\nPMC full text may be available via get_paper_fulltext.`;
      text += `\n\nLink: https://pubmed.ncbi.nlm.nih.gov/${cleanPmid}/`;

      return { content: [{ type: "text", text }] };
    } catch (error) {
      return { content: [{ type: "text", text: `Error fetching PubMed abstract: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Get full paper text via PMC
// ============================================
server.registerTool(
  "get_paper_fulltext",
  {
    description:
      "Fetches full paper text when an open-access PubMed Central copy exists. Accepts PMID, PMCID, or DOI, resolves to PMC when possible, and falls back to abstract-level metadata when no PMC full text is available.",
    inputSchema: {
      pmid: z.string().optional().describe("PubMed ID (e.g., '12345678' or 'PMID:12345678')"),
      pmcid: z.string().optional().describe("PubMed Central ID (e.g., 'PMC1234567' or 'PMCID:PMC1234567')"),
      doi: z.string().optional().describe("DOI for lookup when PMID/PMCID is unknown"),
      maxCharacters: z.number().optional().describe("Maximum number of full-text characters to return (default 30000, max 100000)."),
    },
  },
  async ({ pmid, pmcid, doi, maxCharacters }) => {
    const boundedMax = Math.max(2000, Math.min(100000, Math.round(maxCharacters || 30000)));
    const cleanPmid = normalizePmidValue(pmid);
    const cleanPmcid = normalizePmcidValue(pmcid);
    const cleanDoi = normalizeDoiValue(doi);
    if (!cleanPmid && !cleanPmcid && !cleanDoi) {
      return {
        content: [{
          type: "text",
          text: "Provide at least one identifier: pmid, pmcid, or doi.",
        }],
      };
    }
    try {
      let resolvedPmid = cleanPmid;
      let resolvedPmcid = cleanPmcid;
      let resolvedDoi = cleanDoi;
      let pubmedMeta = null;

      if (!resolvedPmid && resolvedDoi) {
        resolvedPmid = await resolvePmidFromDoi(resolvedDoi);
      }
      if (resolvedPmid) {
        const pubmedXml = await fetchPubmedArticleXml(resolvedPmid);
        if (pubmedXml && !pubmedXml.includes("<ERROR>") && pubmedXml.includes("<PubmedArticle>")) {
          pubmedMeta = parsePubmedRecordMetadata(pubmedXml);
          resolvedPmid = resolvedPmid || pubmedMeta.pmid;
          resolvedPmcid = resolvedPmcid || pubmedMeta.pmcId;
          resolvedDoi = resolvedDoi || pubmedMeta.doi;
        }
      }

      if (!resolvedPmcid) {
        let text = "No PMC full text is available for the supplied identifier.";
        if (resolvedPmid) text += `\nPMID: ${resolvedPmid}`;
        if (resolvedDoi) text += `${resolvedPmid ? " |" : "\n"} DOI: ${resolvedDoi}`;
        text += "\nThis usually means PubMed has metadata/abstract only, but no linked open-access PMC copy.";
        if (pubmedMeta) {
          text += `\n\nTitle: ${pubmedMeta.title}`;
          text += `\nAuthors: ${formatAuthorList(pubmedMeta.authors)}`;
          text += `\nJournal: ${pubmedMeta.journal} (${pubmedMeta.year})`;
          text += `\n\nAbstract:\n${pubmedMeta.abstractText}`;
        }
        return {
          content: [{ type: "text", text }],
          structuredContent: {
            schema: "get_paper_fulltext.v1",
            result_status: "not_found_or_empty",
            full_text_available: false,
            pmid: resolvedPmid || null,
            pmcid: null,
            doi: resolvedDoi || null,
            max_characters: boundedMax,
          },
        };
      }

      const { xml: pmcXml, sourceUrl } = await fetchPmcFullTextXml(resolvedPmcid);
      const article = parsePmcArticle(pmcXml);
      const resolvedTitle = article.title || pubmedMeta?.title || "Untitled";
      const resolvedJournal = article.journal || pubmedMeta?.journal || "";
      const resolvedYear = article.year || pubmedMeta?.year || "";
      const resolvedAuthors = article.authors.length > 0 ? article.authors : pubmedMeta?.authors || [];
      const resolvedAbstract = article.abstractText || pubmedMeta?.abstractText || "No abstract available.";
      resolvedPmid = resolvedPmid || article.pmid || pubmedMeta?.pmid || "";
      resolvedDoi = resolvedDoi || article.doi || pubmedMeta?.doi || "";
      const { text: fullText, truncated } = truncateText(article.fullText, boundedMax);

      let text = `Title: ${resolvedTitle}\nPMCID: ${resolvedPmcid}`;
      if (resolvedPmid) text += ` | PMID: ${resolvedPmid}`;
      if (resolvedDoi) text += ` | DOI: ${resolvedDoi}`;
      text += `\nAuthors: ${formatAuthorList(resolvedAuthors, 10)}`;
      text += `\nJournal: ${resolvedJournal}${resolvedYear ? ` (${resolvedYear})` : ""}`;
      if (article.licenseText) text += `\nLicense: ${article.licenseText}`;
      if (article.sectionTitles.length) text += `\nSections: ${article.sectionTitles.slice(0, 12).join("; ")}`;
      text += `\n\nAbstract:\n${resolvedAbstract}`;
      text += `\n\nFull Text:\n${fullText || "No PMC body text was extracted."}`;
      text += `\n\nLinks:\n- PMC: https://pmc.ncbi.nlm.nih.gov/articles/${resolvedPmcid}/`;
      if (resolvedPmid) text += `\n- PubMed: https://pubmed.ncbi.nlm.nih.gov/${resolvedPmid}/`;
      if (resolvedDoi) text += `\n- DOI: https://doi.org/${resolvedDoi}`;

      return {
        content: [{ type: "text", text }],
        structuredContent: {
          schema: "get_paper_fulltext.v1",
          result_status: "ok",
          full_text_available: true,
          truncated,
          max_characters: boundedMax,
          title: resolvedTitle,
          pmid: resolvedPmid || null,
          pmcid: resolvedPmcid,
          doi: resolvedDoi || null,
          journal: resolvedJournal || null,
          publication_year: resolvedYear || null,
          source_url: sourceUrl,
          section_titles: article.sectionTitles,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{
          type: "text",
          text: `Error fetching paper full text: ${detail}`,
        }],
        structuredContent: {
          schema: "get_paper_fulltext.v1",
          result_status: "error",
          full_text_available: false,
          pmid: cleanPmid || null,
          pmcid: cleanPmcid || null,
          doi: cleanDoi || null,
          max_characters: boundedMax,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: IEDB epitope / assay evidence search
// ============================================
server.registerTool(
  "search_iedb_epitope_evidence",
  {
    description:
      "Searches IEDB for epitope, T-cell, and MHC ligand evidence. " +
      "Best used with an exact or partial peptide sequence, and can also resolve an antigen/protein name " +
      "to a UniProt accession before querying IEDB. Returns matching sequences, alleles, assay evidence, and PMIDs.",
    inputSchema: {
      peptide: z.string().optional().describe("Exact or partial peptide sequence, e.g. 'SIINFEKL' or 'KLVVVGAGGVGKSAL'. Strongest input when known."),
      antigen: z.string().optional().describe("Antigen or protein text, e.g. 'KRAS', 'KRAS G12D', or 'spike glycoprotein'. Resolved to a UniProt accession when possible."),
      allele: z.string().optional().describe("Exact HLA / MHC allele name, e.g. 'HLA-A*11:01'."),
      disease: z.string().optional().describe("Optional disease context filter applied after retrieval, e.g. 'glioma' or 'melanoma'."),
      hostOrganism: z.string().optional().describe("Optional host organism filter applied after retrieval, e.g. 'human' or 'mouse'."),
      assay: z.enum(["any", "epitope", "tcell", "mhc"]).optional().default("any")
        .describe("Which IEDB endpoint family to query. 'any' searches epitope summaries plus T-cell and MHC assay evidence."),
      positiveOnly: z.boolean().optional().default(false)
        .describe("If true, keep only positive assay/evidence rows when the endpoint exposes qualitative positivity."),
      maxResults: z.number().optional().default(10)
        .describe("Maximum number of normalized evidence records to return (default 10, max 20)."),
    },
  },
  async ({
    peptide,
    antigen,
    allele,
    disease,
    hostOrganism,
    assay = "any",
    positiveOnly = false,
    maxResults = 10,
  }) => {
    const normalizedPeptide = normalizePeptideSequence(peptide || "");
    const normalizedAntigen = normalizeWhitespace(antigen || "");
    const normalizedAllele = normalizeWhitespace(allele || "");
    const normalizedDisease = normalizeWhitespace(disease || "");
    const normalizedHost = normalizeWhitespace(hostOrganism || "");
    const boundedMaxResults = Math.max(1, Math.min(20, Math.round(Number(maxResults) || 10)));

    if (!normalizedPeptide && !normalizedAntigen && !normalizedAllele) {
      return {
        content: [{
          type: "text",
          text: "Provide at least one of: peptide, antigen, or allele.",
        }],
        structuredContent: {
          schema: "search_iedb_epitope_evidence.v1",
          result_status: "error",
          error: "Missing required search input: peptide, antigen, or allele.",
        },
      };
    }

    let antigenTarget = null;
    if (normalizedAntigen) {
      antigenTarget = await resolveIedbAntigenTarget(normalizedAntigen);
      if (!antigenTarget && !normalizedPeptide && !normalizedAllele) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Could not resolve antigen "${normalizedAntigen}" to a UniProt accession for IEDB querying.`,
              keyFields: [
                `Antigen query: ${normalizedAntigen}`,
                "IEDB antigen searches work best with an exact peptide sequence or a resolvable antigen/protein accession.",
              ],
              sources: [
                "https://help.iedb.org/hc/en-us/articles/4402872882189-Immune-Epitope-Database-Query-API-IQ-API",
              ],
              limitations: [
                "Mutation names alone often do not map cleanly to IEDB records. Supply a peptide sequence when possible.",
              ],
            }),
          }],
          structuredContent: {
            schema: "search_iedb_epitope_evidence.v1",
            result_status: "not_found_or_empty",
            peptide: normalizedPeptide || null,
            antigen: normalizedAntigen || null,
            allele: normalizedAllele || null,
            disease: normalizedDisease || null,
            host_organism: normalizedHost || null,
            assay,
            positive_only: Boolean(positiveOnly),
            antigen_resolution: null,
            records: [],
          },
        };
      }
    }

    const endpoints = assay === "any" ? ["epitope", "tcell", "mhc"] : [assay];
    const perEndpointLimit = assay === "any"
      ? Math.max(8, Math.min(40, boundedMaxResults * 3))
      : Math.max(8, Math.min(50, boundedMaxResults * 4));

    const settled = await Promise.allSettled(
      endpoints.map((endpoint) =>
        fetchIedbEndpoint(endpoint, {
          peptide: normalizedPeptide,
          antigenTarget,
          allele: normalizedAllele,
          positiveOnly: Boolean(positiveOnly),
          limit: perEndpointLimit,
        })
      )
    );

    const endpointCounts = {};
    const sourceUrls = [];
    const rawRecords = [];
    const errors = [];

    settled.forEach((result, idx) => {
      const endpoint = endpoints[idx];
      if (result.status === "fulfilled") {
        const rows = Array.isArray(result.value?.rows) ? result.value.rows : [];
        endpointCounts[endpoint] = {
          retrieved: rows.length,
          total_count: result.value?.totalCount ?? null,
        };
        if (result.value?.url) sourceUrls.push(result.value.url);
        for (const row of rows) {
          rawRecords.push(normalizeIedbRecord(endpoint, row));
        }
      } else {
        endpointCounts[endpoint] = {
          retrieved: 0,
          total_count: null,
          error: compactErrorMessage(result.reason?.message || String(result.reason), 180),
        };
        errors.push(`${endpoint}: ${endpointCounts[endpoint].error}`);
      }
    });

    const filteredRecords = rawRecords.filter((record) => {
      if (normalizedDisease && !matchesIedbText(record.disease_names, normalizedDisease)) return false;
      if (normalizedHost && !matchesIedbText(record.host_organism_names, normalizedHost)) return false;
      if (normalizedAntigen && !antigenTarget && !matchesIedbText(record.antigen_names, normalizedAntigen)) return false;
      return true;
    });

    const scoredRecords = filteredRecords
      .map((record) => ({
        ...record,
        score: scoreIedbRecord(record, {
          peptide: normalizedPeptide,
          allele: normalizedAllele,
          antigenQuery: antigenTarget?.query || normalizedAntigen,
          disease: normalizedDisease,
          hostOrganism: normalizedHost,
        }),
      }))
      .sort((left, right) => {
        const scoreDelta = (right.score || 0) - (left.score || 0);
        if (scoreDelta !== 0) return scoreDelta;
        return normalizeWhitespace(left.sequence || "").localeCompare(normalizeWhitespace(right.sequence || ""));
      })
      .slice(0, boundedMaxResults);

    const uniquePmids = dedupeArray(scoredRecords.flatMap((record) => record.pubmed_ids)).slice(0, 20);
    const uniqueAlleles = dedupeArray(scoredRecords.flatMap((record) => record.allele_names)).slice(0, 12);
    const uniqueSequences = dedupeArray(scoredRecords.map((record) => record.sequence).filter(Boolean)).slice(0, 12);

    if (antigenTarget?.source_url) sourceUrls.push(antigenTarget.source_url);
    const dedupedSources = dedupeArray(sourceUrls);
    const keyFields = [];
    if (normalizedPeptide) keyFields.push(`Peptide filter: ${normalizedPeptide}`);
    if (normalizedAntigen) {
      keyFields.push(`Antigen query: ${normalizedAntigen}`);
      if (antigenTarget?.iri) {
        const geneText = antigenTarget.gene_symbols?.length ? ` | genes: ${antigenTarget.gene_symbols.join(", ")}` : "";
        keyFields.push(`Resolved antigen: ${antigenTarget.label} (${antigenTarget.iri})${geneText}`);
      }
    }
    if (normalizedAllele) keyFields.push(`Allele filter: ${normalizedAllele}`);
    if (normalizedDisease) keyFields.push(`Disease filter: ${normalizedDisease}`);
    if (normalizedHost) keyFields.push(`Host organism filter: ${normalizedHost}`);
    keyFields.push(`Assay mode: ${assay}`);
    keyFields.push(`Positive-only: ${positiveOnly ? "yes" : "no"}`);
    keyFields.push(`Endpoint hits: ${endpoints.map((endpoint) => {
      const meta = endpointCounts[endpoint] || {};
      return `${endpoint} ${meta.retrieved || 0}${meta.total_count != null ? ` of ${meta.total_count}` : ""}`;
    }).join("; ")}`);
    if (uniqueSequences.length > 0) keyFields.push(`Unique sequences shown: ${uniqueSequences.join(", ")}`);
    if (uniqueAlleles.length > 0) keyFields.push(`Unique alleles shown: ${uniqueAlleles.join(", ")}`);
    if (uniquePmids.length > 0) keyFields.push(`Representative PMIDs: ${uniquePmids.join(", ")}`);

    if (scoredRecords.length > 0) {
      keyFields.push("\nTop evidence:");
      keyFields.push(...scoredRecords.map((record, idx) => formatIedbRecordLine(record, idx)));
    }

    const mutationOnlyWarning = normalizedAntigen && looksLikeMutationToken(normalizedAntigen) && !normalizedPeptide;
    const limitations = [
      "IEDB retrieval is strongest when you provide an exact peptide sequence and, when known, an exact HLA/MHC allele.",
      "Positive and negative assay rows can coexist for the same peptide and allele across different studies.",
      "Antigen text is resolved to the top UniProt hit when possible; broad names can still pull mixed evidence.",
    ];
    if (mutationOnlyWarning) {
      limitations.push("A mutation-like antigen query was simplified for antigen resolution. Supply the exact mutant peptide sequence for mutation-specific retrieval.");
    }
    if (errors.length > 0) {
      limitations.push(`Some IEDB endpoints failed: ${errors.join(" | ")}`);
    }

    const resultStatus = scoredRecords.length > 0
      ? (errors.length > 0 ? "degraded" : "ok")
      : (errors.length > 0 ? "error" : "not_found_or_empty");

    const summary = scoredRecords.length > 0
      ? `IEDB returned ${scoredRecords.length} normalized evidence record(s) across ${endpoints.length} endpoint${endpoints.length === 1 ? "" : "s"}${uniquePmids.length > 0 ? ` with PMID support from ${uniquePmids.length} article(s)` : ""}.`
      : `No IEDB evidence records matched the supplied filters${errors.length > 0 ? `; endpoint errors were observed` : ""}.`;

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary,
          keyFields,
          sources: dedupedSources.length > 0
            ? dedupedSources
            : ["https://help.iedb.org/hc/en-us/articles/4402872882189-Immune-Epitope-Database-Query-API-IQ-API"],
          limitations,
        }),
      }],
      structuredContent: {
        schema: "search_iedb_epitope_evidence.v1",
        result_status: resultStatus,
        peptide: normalizedPeptide || null,
        antigen: normalizedAntigen || null,
        allele: normalizedAllele || null,
        disease: normalizedDisease || null,
        host_organism: normalizedHost || null,
        assay,
        positive_only: Boolean(positiveOnly),
        antigen_resolution: antigenTarget
          ? {
            query_used: antigenTarget.query,
            accession: antigenTarget.accession,
            iri: antigenTarget.iri,
            label: antigenTarget.label,
            gene_symbols: antigenTarget.gene_symbols || [],
          }
          : null,
        endpoint_counts: endpointCounts,
        records: scoredRecords.map((record) => ({
          endpoint: record.endpoint,
          record_id: record.record_id || null,
          structure_id: record.structure_id || null,
          sequence: record.sequence || null,
          antigen_names: record.antigen_names,
          antigen_iris: record.antigen_iris,
          allele_names: record.allele_names,
          qualitative_measures: record.qualitative_measures,
          quantitative_measure: record.quantitative_measure,
          pubmed_ids: record.pubmed_ids,
          reference_titles: record.reference_titles.slice(0, 3),
          disease_names: record.disease_names,
          host_organism_names: record.host_organism_names,
          assay_names: record.assay_names.slice(0, 4),
          score: record.score,
        })),
        source_urls: dedupedSources,
        errors,
      },
    };
  }
);

// ============================================
// RefSeq helpers (NCBI Entrez nuccore / protein)
// ============================================
const REFSEQ_PROTEIN_ACCESSION_PREFIX = /^(NP_|XP_|YP_|AP_|WP_)/i;

function inferRefseqEntrezDbFromIdentifier(rawIdentifier) {
  const token = String(rawIdentifier || "").trim().split(/\s+/)[0] || "";
  if (REFSEQ_PROTEIN_ACCESSION_PREFIX.test(token)) {
    return "protein";
  }
  return "nuccore";
}

function buildRefseqRecordUrl(db, accessionVersion, caption) {
  const acc = String(accessionVersion || caption || "").trim();
  if (!acc) {
    return "https://www.ncbi.nlm.nih.gov/refseq/";
  }
  if (db === "protein") {
    return `https://www.ncbi.nlm.nih.gov/protein/${encodeURIComponent(acc)}`;
  }
  return `https://www.ncbi.nlm.nih.gov/nuccore/${encodeURIComponent(acc)}`;
}

function parseRefseqEsummaryDocs(payload) {
  const result = payload?.result;
  const uids = result?.uids;
  if (!Array.isArray(uids) || uids.length === 0) {
    return [];
  }
  return uids
    .map((uid) => result?.[uid])
    .filter((doc) => doc && typeof doc === "object");
}

function summarizeRefseqDocSum(doc, db) {
  const accVer = normalizeWhitespace(doc?.accessionversion || doc?.caption || "") || "";
  const title = normalizeWhitespace(doc?.title || "") || "";
  const organism = normalizeWhitespace(doc?.organism || "") || "";
  const slen = Number(doc?.slen || 0) || 0;
  const molecule = normalizeWhitespace(doc?.biomol || doc?.moltype || "") || "";
  const location = normalizeWhitespace(doc?.subname || "") || "";
  const uid = normalizeWhitespace(doc?.uid || "") || "";
  const caption = normalizeWhitespace(doc?.caption || "") || null;
  const capOrVer = accVer || caption || "";
  return {
    uid,
    accessionversion: accVer || null,
    caption,
    title,
    organism,
    length: slen,
    molecule,
    location: location || null,
    sourcedb: normalizeWhitespace(doc?.sourcedb || "") || null,
    record_url: buildRefseqRecordUrl(db, capOrVer, caption),
  };
}

async function refseqEsearch(db, term, retmax) {
  const params = new URLSearchParams({
    db,
    term,
    retmax: String(retmax),
    retmode: "json",
  });
  const url = buildNcbiEutilsUrl(PUBMED_ESEARCH, params);
  return fetchJsonWithRetry(url, { retries: 2, timeoutMs: 10000, maxBackoffMs: 2500 });
}

async function refseqEsummary(db, idList) {
  const cleanIds = idList.map((id) => String(id).trim()).filter(Boolean);
  if (cleanIds.length === 0) {
    return { result: { uids: [] } };
  }
  const params = new URLSearchParams({
    db,
    id: cleanIds.join(","),
    retmode: "json",
  });
  const url = buildNcbiEutilsUrl(PUBMED_ESUMMARY, params);
  return fetchJsonWithRetry(url, { retries: 2, timeoutMs: 12000, maxBackoffMs: 2500 });
}

async function resolveRefseqUidsForLookup(db, identifier, retmax) {
  const raw = String(identifier || "").trim();
  if (/^\d+$/.test(raw)) {
    const summaryPayload = await refseqEsummary(db, [raw]);
    const docs = parseRefseqEsummaryDocs(summaryPayload);
    if (docs.length > 0) {
      return { uids: [raw], totalCount: 1 };
    }
  }
  const data = await refseqEsearch(db, raw, retmax);
  const uids = Array.isArray(data?.esearchresult?.idlist) ? data.esearchresult.idlist : [];
  const totalCount = Number(data?.esearchresult?.count || 0);
  return { uids, totalCount };
}

async function refseqEfetchXml(db, id) {
  const params = new URLSearchParams({
    db,
    id: String(id || "").trim(),
    rettype: db === "protein" ? "gp" : "gbc",
    retmode: "xml",
  });
  const url = buildNcbiEutilsUrl(PUBMED_EFETCH, params);
  const response = await fetchWithRetry(url, { retries: 2, timeoutMs: 20000, maxBackoffMs: 2500 });
  return response.text();
}

const REFSEQ_FEATURE_QUALIFIER_SKIP = new Set(["translation"]);
const REFSEQ_FEATURE_SUMMARY_QUALIFIER_PRIORITY = ["gene", "product", "protein_id", "peptide", "note", "inference"];

function summarizeRefseqFeatureAnnotation(feature) {
  if (!feature || typeof feature !== "object") {
    return null;
  }
  const key = normalizeWhitespace(feature?.INSDFeature_key || "");
  const location = normalizeWhitespace(feature?.INSDFeature_location || "");
  if (!key || !location) {
    return null;
  }

  const intervals = asArray(feature?.INSDFeature_intervals?.INSDInterval)
    .map((interval) => ({
      from: normalizeWhitespace(interval?.INSDInterval_from || "") || null,
      to: normalizeWhitespace(interval?.INSDInterval_to || "") || null,
      accession: normalizeWhitespace(interval?.INSDInterval_accession || "") || null,
    }))
    .filter((interval) => interval.from || interval.to || interval.accession);

  const qualifiers = {};
  for (const qualifier of asArray(feature?.INSDFeature_quals?.INSDQualifier)) {
    const name = normalizeWhitespace(qualifier?.INSDQualifier_name || "").toLowerCase();
    const value = normalizeWhitespace(qualifier?.INSDQualifier_value || "");
    if (!name || !value || REFSEQ_FEATURE_QUALIFIER_SKIP.has(name)) {
      continue;
    }
    if (!qualifiers[name]) {
      qualifiers[name] = [];
    }
    if (qualifiers[name].length < 4) {
      qualifiers[name].push(compactErrorMessage(value, 180));
    }
  }

  const summaryBits = [`${key} ${location}`];
  for (const qualifierName of REFSEQ_FEATURE_SUMMARY_QUALIFIER_PRIORITY) {
    const firstValue = qualifiers[qualifierName]?.[0];
    if (firstValue) {
      summaryBits.push(`${qualifierName}: ${firstValue}`);
    }
  }

  return {
    key,
    location,
    intervals,
    qualifiers,
    summary_line: summaryBits.join(" | "),
  };
}

function parseRefseqFeatureAnnotations(xmlText, { maxFeatures = 20, highlightKeys = [] } = {}) {
  const parsed = parseXmlDocument(xmlText);
  const record = Array.isArray(parsed?.INSDSet?.INSDSeq) ? parsed.INSDSet.INSDSeq[0] : parsed?.INSDSet?.INSDSeq;
  const rawFeatures = asArray(record?.["INSDSeq_feature-table"]?.INSDFeature);
  const features = rawFeatures
    .map((feature) => summarizeRefseqFeatureAnnotation(feature))
    .filter(Boolean);

  const normalizedHighlightKeys = new Set(
    asArray(highlightKeys).map((value) => normalizeWhitespace(value || "").toLowerCase()).filter(Boolean)
  );
  const highlightedFeatures = normalizedHighlightKeys.size > 0
    ? features.filter((feature) => normalizedHighlightKeys.has(feature.key.toLowerCase()))
    : features.slice(0, maxFeatures);

  const featureLookup = {};
  for (const feature of features) {
    if (!featureLookup[feature.key]) {
      featureLookup[feature.key] = [];
    }
    featureLookup[feature.key].push(feature.location);
  }

  return {
    feature_count: features.length,
    features: features.slice(0, maxFeatures),
    highlighted_features: highlightedFeatures.slice(0, maxFeatures),
    feature_lookup: featureLookup,
  };
}

// ============================================
// TOOL: Search GEO datasets
// ============================================
server.registerTool(
  "search_geo_datasets",
  {
    description:
      "Searches NCBI GEO (Gene Expression Omnibus) for functional genomics records, including series (GSE), samples (GSM), platforms (GPL), and curated datasets (GDS). " +
      "Useful for finding transcriptomics, microarray, and sequencing studies by disease, gene, perturbation, tissue, or accession.",
    inputSchema: {
      query: z.string().describe("GEO search query, e.g. 'Parkinson disease substantia nigra RNA-seq', 'TP53 knockdown', or an accession like 'GSE124700'."),
      maxResults: z.number().optional().describe("Maximum results to return (default 10, max 50)."),
      entryType: z.enum(["all", "GSE", "GDS", "GSM", "GPL"]).optional().describe("Optional GEO record family filter. Default 'all'."),
    },
  },
  async ({ query, maxResults, entryType }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a GEO search query or accession." }] };
    }

    try {
      const limit = Math.min(Math.max(1, Math.round(maxResults || 10)), 50);
      const family = normalizeGeoIdentifier(entryType || "all");
      const term = family && family !== "ALL"
        ? `${normalizedQuery} AND ${family}[ETYP]`
        : normalizedQuery;
      const searchParams = new URLSearchParams({
        db: "gds",
        term,
        retmax: String(limit),
        retmode: "json",
      });
      const searchUrl = buildNcbiEutilsUrl(PUBMED_ESEARCH, searchParams);
      const searchData = await fetchJsonWithRetry(searchUrl, { retries: 2, timeoutMs: 10000, maxBackoffMs: 2500 });
      const idList = searchData?.esearchresult?.idlist ?? [];
      const totalCount = Number(searchData?.esearchresult?.count || 0);

      if (!Array.isArray(idList) || idList.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No GEO records found for "${normalizedQuery}".`,
              keyFields: [
                `Query: ${normalizedQuery}`,
                `Entry type filter: ${family && family !== "ALL" ? family : "all"}`,
              ],
              sources: [
                "https://www.ncbi.nlm.nih.gov/geo/",
                `https://www.ncbi.nlm.nih.gov/gds/?term=${encodeURIComponent(normalizedQuery)}`,
              ],
              limitations: [
                "GEO metadata search can be noisy. Try adding tissue, organism, assay type, or accession terms like GSE12345.",
              ],
            }),
          }],
          structuredContent: {
            schema: "search_geo_datasets.v1",
            result_status: "not_found_or_empty",
            query: normalizedQuery,
            entry_type: family && family !== "ALL" ? family : null,
            total_count: 0,
            records: [],
          },
        };
      }

      const docs = await fetchGeoDocSumsByIds(idList);
      const lines = docs.map((doc, idx) => {
        const parts = summarizeGeoDocsum(doc);
        return `${String(idx + 1).padStart(3)}. ${parts.join(" | ")}`;
      });
      const records = docs.map((doc) => ({
        uid: normalizeWhitespace(doc?.uid || "") || null,
        accession: normalizeGeoIdentifier(doc?.accession || "") || null,
        title: normalizeWhitespace(doc?.title || "") || null,
        entry_type: normalizeWhitespace(doc?.entrytype || "") || null,
        dataset_type: normalizeWhitespace(doc?.gdstype || "") || null,
        organism: normalizeWhitespace(doc?.taxon || "") || null,
        n_samples: Number(doc?.n_samples || 0) || 0,
        pubmed_ids: Array.isArray(doc?.pubmedids) ? doc.pubmedids.filter(Boolean) : [],
        record_url: buildGeoRecordUrl(doc?.accession),
      }));

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `GEO search: ${totalCount} record(s) for "${normalizedQuery}". Showing top ${docs.length}.`,
            keyFields: [
              `Query: ${normalizedQuery}`,
              `Entry type filter: ${family && family !== "ALL" ? family : "all"}`,
              `Total matches: ${totalCount}`,
              "\nResults:",
              ...lines,
            ],
            sources: [
              "https://www.ncbi.nlm.nih.gov/geo/",
              `https://www.ncbi.nlm.nih.gov/gds/?term=${encodeURIComponent(normalizedQuery)}`,
            ],
            limitations: [
              "GEO mixes series, samples, platforms, and curated datasets in one search index.",
              "Use get_geo_dataset with a returned accession to inspect full metadata for a specific record.",
            ],
          }),
        }],
        structuredContent: {
          schema: "search_geo_datasets.v1",
          result_status: "ok",
          query: normalizedQuery,
          entry_type: family && family !== "ALL" ? family : null,
          total_count: totalCount,
          records,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error searching GEO: ${detail}` }],
        structuredContent: {
          schema: "search_geo_datasets.v1",
          result_status: "error",
          query: normalizedQuery,
          entry_type: normalizeGeoIdentifier(entryType || "all") || null,
          total_count: 0,
          records: [],
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: Get GEO dataset details
// ============================================
server.registerTool(
  "get_geo_dataset",
  {
    description:
      "Fetches detailed metadata for a GEO accession or GEO UID, including title, organism, study type, sample count, platform/series links, PubMed IDs, and summary text. " +
      "Accepts accessions such as GSE124700, GDS507, GPL570, or GSM3543597.",
    inputSchema: {
      identifier: z.string().describe("GEO accession or numeric GEO UID, e.g. 'GSE124700', 'GDS507', 'GPL570', 'GSM3543597', or '200124700'."),
    },
  },
  async ({ identifier }) => {
    const normalizedIdentifier = normalizeGeoIdentifier(identifier);
    if (!normalizedIdentifier) {
      return { content: [{ type: "text", text: "Provide a GEO accession or GEO UID." }] };
    }

    try {
      const doc = await resolveGeoDocSum(normalizedIdentifier);
      if (!doc) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No GEO record found for "${normalizedIdentifier}".`,
              keyFields: [`Identifier: ${normalizedIdentifier}`],
              sources: [
                "https://www.ncbi.nlm.nih.gov/geo/",
                `https://www.ncbi.nlm.nih.gov/gds/?term=${encodeURIComponent(normalizedIdentifier)}`,
              ],
              limitations: [
                "Check whether the accession is valid and note that GEO contains multiple record families (GSE, GSM, GPL, GDS).",
              ],
            }),
          }],
          structuredContent: {
            schema: "get_geo_dataset.v1",
            result_status: "not_found_or_empty",
            identifier: normalizedIdentifier,
          },
        };
      }

      const accession = normalizeGeoIdentifier(doc?.accession || "");
      const keyFields = [
        `Identifier: ${normalizedIdentifier}`,
        `Accession: ${accession || "n/a"}`,
        `UID: ${normalizeWhitespace(doc?.uid || "") || "n/a"}`,
        `Title: ${normalizeWhitespace(doc?.title || "Untitled GEO record")}`,
        `Entry type: ${normalizeWhitespace(doc?.entrytype || "unknown")}`,
      ];
      if (doc?.gdstype) keyFields.push(`Dataset type: ${normalizeWhitespace(doc.gdstype)}`);
      if (doc?.taxon) keyFields.push(`Organism: ${normalizeWhitespace(doc.taxon)}`);
      if (doc?.n_samples !== undefined) keyFields.push(`Samples: ${Number(doc.n_samples || 0)}`);
      if (doc?.gpl) keyFields.push(`Platform: GPL${String(doc.gpl).replace(/^GPL/i, "")}`);
      if (doc?.gse) keyFields.push(`Series: GSE${String(doc.gse).replace(/^GSE/i, "")}`);
      if (doc?.pdat) keyFields.push(`Published/updated: ${normalizeWhitespace(doc.pdat)}`);
      if (doc?.suppfile) keyFields.push(`Supplementary files: ${normalizeWhitespace(doc.suppfile)}`);
      if (doc?.geo2r) keyFields.push(`GEO2R available: ${String(doc.geo2r).toLowerCase() === "yes" ? "yes" : doc.geo2r}`);
      if (doc?.bioproject) keyFields.push(`BioProject: ${normalizeWhitespace(doc.bioproject)}`);

      const samples = Array.isArray(doc?.samples) ? doc.samples.slice(0, 8) : [];
      if (samples.length > 0) {
        keyFields.push("\nSample examples:");
        for (const sample of samples) {
          const sampleAccession = normalizeGeoIdentifier(sample?.accession || "");
          const sampleTitle = normalizeWhitespace(sample?.title || "");
          keyFields.push(`${sampleAccession || "sample"}${sampleTitle ? `: ${sampleTitle}` : ""}`);
        }
      }

      const summary = normalizeWhitespace(doc?.summary || "");
      if (summary) {
        keyFields.push(`\nSummary text:\n${summary.length > 1200 ? `${summary.slice(0, 1197)}...` : summary}`);
      }

      const pubmedIds = Array.isArray(doc?.pubmedids) ? doc.pubmedids.filter(Boolean).slice(0, 8) : [];
      if (pubmedIds.length > 0) {
        keyFields.push(`PubMed IDs: ${pubmedIds.join(", ")}`);
      }

      const sources = [buildGeoRecordUrl(accession || normalizedIdentifier)];
      if (doc?.ftplink) sources.push(normalizeWhitespace(doc.ftplink));
      for (const pmid of pubmedIds.slice(0, 3)) {
        sources.push(`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(pmid)}/`);
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `GEO metadata retrieved for ${accession || normalizedIdentifier}.`,
            keyFields,
            sources,
            limitations: [
              "This tool returns GEO metadata and sample previews, not full expression matrices or raw sequencing files.",
              "Study design, normalization details, and downloadable assets should be verified on the GEO record page and linked supplementary files.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_geo_dataset.v1",
          result_status: "ok",
          identifier: normalizedIdentifier,
          uid: normalizeWhitespace(doc?.uid || "") || null,
          accession: accession || null,
          title: normalizeWhitespace(doc?.title || "") || null,
          entry_type: normalizeWhitespace(doc?.entrytype || "") || null,
          dataset_type: normalizeWhitespace(doc?.gdstype || "") || null,
          organism: normalizeWhitespace(doc?.taxon || "") || null,
          n_samples: Number(doc?.n_samples || 0) || 0,
          pubmed_ids: pubmedIds,
          ftp_link: normalizeWhitespace(doc?.ftplink || "") || null,
          bioproject: normalizeWhitespace(doc?.bioproject || "") || null,
          geo2r_available: String(doc?.geo2r || "").toLowerCase() === "yes",
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error fetching GEO record: ${detail}` }],
        structuredContent: {
          schema: "get_geo_dataset.v1",
          result_status: "error",
          identifier: normalizedIdentifier,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: Compute GEO cell-type proportions from supplementary count matrices
// ============================================
server.registerTool(
  "get_geo_cell_type_proportions",
  {
    description:
      "Compute cell-type proportions from GEO supplementary count matrices when the raw per-cell CSV includes a cell-type/cluster label column " +
      "(for example GSE84133 `assigned_cluster`). Can filter donors by sample metadata such as disease status before aggregating counts.",
    inputSchema: {
      accession: z.string().describe("GEO series accession (GSE...), e.g. 'GSE84133'."),
      cellTypes: z.array(z.string()).optional().describe("Optional list of cell types/clusters to report explicitly, e.g. ['beta', 'ductal']."),
      sampleAccession: z.string().optional().describe("Optional specific GEO sample accession (GSM...) to use instead of filtering all samples in the series."),
      organism: z.string().optional().describe("Optional organism filter for sample metadata, default 'Homo sapiens'."),
      donorDiseaseField: z.string().optional().describe("Sample metadata field used to identify a donor subset, default 'type 2 diabetes mellitus'."),
      donorDiseaseValue: z.string().optional().describe("Required value for donorDiseaseField, default 'Yes'."),
      clusterColumn: z.string().optional().describe("CSV column containing cell-type labels, default 'assigned_cluster'."),
    },
  },
  async ({ accession, cellTypes, sampleAccession, organism, donorDiseaseField, donorDiseaseValue, clusterColumn }) => {
    const cleanAccession = normalizeGeoIdentifier(accession);
    if (!/^GSE\d+$/.test(cleanAccession)) {
      return { content: [{ type: "text", text: "Provide a GEO series accession such as GSE84133." }] };
    }

    const targetCellTypes = Array.isArray(cellTypes)
      ? cellTypes.map((value) => normalizeWhitespace(value)).filter(Boolean)
      : [];
    const cleanSampleAccession = normalizeGeoIdentifier(sampleAccession);
    const organismFilter = normalizeWhitespace(organism || "Homo sapiens");
    const diseaseField = normalizeWhitespace(donorDiseaseField || "type 2 diabetes mellitus").toLowerCase();
    const diseaseValue = normalizeWhitespace(donorDiseaseValue || "Yes").toLowerCase();
    const clusterField = normalizeWhitespace(clusterColumn || "assigned_cluster") || "assigned_cluster";

    try {
      const fileListUrl = buildGeoSupplementaryFileListUrl(cleanAccession);
      if (!fileListUrl) {
        throw new Error(`Could not derive a GEO supplementary-file URL for ${cleanAccession}.`);
      }
      const fileListResponse = await fetchWithRetry(fileListUrl, { retries: 1, timeoutMs: 20000, maxBackoffMs: 2500 });
      const fileListRows = parseGeoFileList(await fileListResponse.text());
      const csvMembers = fileListRows
        .map((row) => row.name)
        .filter((name) => /^GSM\d+_.+\.csv\.gz$/i.test(name));

      if (csvMembers.length === 0) {
        throw new Error(`No supplementary per-cell CSV files were found for ${cleanAccession}.`);
      }

      const matchingMembers = [];
      for (const memberName of csvMembers) {
        const gsmMatch = memberName.match(/^(GSM\d+)_/i);
        const gsmAccession = normalizeGeoIdentifier(gsmMatch?.[1] || "");
        if (!gsmAccession) continue;
        if (cleanSampleAccession && gsmAccession !== cleanSampleAccession) {
          continue;
        }
        const sampleMetadata = await fetchGeoSampleQuickMetadata(gsmAccession);
        const sampleOrganism = normalizeWhitespace(sampleMetadata?.organism || "");
        if (organismFilter && sampleOrganism && sampleOrganism.toLowerCase() !== organismFilter.toLowerCase()) {
          continue;
        }
        if (!cleanSampleAccession) {
          const characteristicValue = normalizeWhitespace(sampleMetadata?.characteristics?.[diseaseField] || "").toLowerCase();
          if (!characteristicValue || characteristicValue !== diseaseValue) {
            continue;
          }
        }
        matchingMembers.push({
          gsm_accession: gsmAccession,
          member_name: memberName,
          metadata: sampleMetadata,
        });
      }

      if (matchingMembers.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No supplementary GEO samples matched the requested donor filter in ${cleanAccession}.`,
              keyFields: [
                `Series: ${cleanAccession}`,
                `Requested sample: ${cleanSampleAccession || "none"}`,
                `Organism filter: ${organismFilter || "none"}`,
                `Donor filter: ${diseaseField || "none"} = ${diseaseValue || "none"}`,
              ],
              sources: [buildGeoRecordUrl(cleanAccession), fileListUrl],
              limitations: [
                "This tool only works when GEO supplementary files include a per-cell label column such as `assigned_cluster`.",
                "Sample-level donor filtering relies on GEO sample metadata fields and may not cover every study design.",
              ],
            }),
          }],
          structuredContent: {
            schema: "get_geo_cell_type_proportions.v1",
            result_status: "not_found_or_empty",
            accession: cleanAccession,
            sample_accessions: [],
            cell_type_proportions: [],
          },
        };
      }

      const archivePayload = await fetchGeoSupplementaryArchive(cleanAccession);
      const aggregatedCounts = new Map();
      let totalRows = 0;
      const sampleSummaries = [];

      for (const member of matchingMembers) {
        const gzBuffer = extractGeoTarEntryBuffer(archivePayload, member.member_name);
        if (!gzBuffer) {
          throw new Error(`Supplementary member ${member.member_name} was not found inside ${cleanAccession}_RAW.tar.`);
        }
        const csvText = gunzipSync(gzBuffer).toString("utf8");
        const sampleResult = computeGeoClusterProportionsFromCountsCsv(csvText, clusterField);
        totalRows += sampleResult.total_rows;
        for (const row of sampleResult.counts) {
          aggregatedCounts.set(row.cell_type, (aggregatedCounts.get(row.cell_type) || 0) + row.count);
        }
        sampleSummaries.push({
          gsm_accession: member.gsm_accession,
          member_name: member.member_name,
          total_rows: sampleResult.total_rows,
          source_name: normalizeWhitespace(member.metadata?.source_name || "") || null,
          donor_characteristics: member.metadata?.characteristics || {},
        });
      }

      const allProportions = Array.from(aggregatedCounts.entries())
        .map(([cellType, count]) => ({
          cell_type: cellType,
          count,
          proportion: count / totalRows,
          proportion_rounded: Number((count / totalRows).toFixed(2)),
        }))
        .sort((left, right) => right.count - left.count);

      const filteredProportions = targetCellTypes.length > 0
        ? allProportions.filter((row) => targetCellTypes.some((cellType) => cellType.toLowerCase() === row.cell_type.toLowerCase()))
        : allProportions;

      const keyFields = [
        `Series: ${cleanAccession}`,
        `Matched samples: ${matchingMembers.map((member) => member.gsm_accession).join(", ")}`,
        `Organism filter: ${organismFilter}`,
        cleanSampleAccession
          ? `Specific sample: ${cleanSampleAccession}`
          : `Donor filter: ${diseaseField} = ${diseaseValue}`,
        `Cluster column: ${clusterField}`,
        `Total labeled cells: ${totalRows}`,
        "\nComputed proportions:",
        ...filteredProportions.map((row) => `${row.cell_type}: ${row.proportion.toFixed(4)} (${row.count}/${totalRows})`),
      ];

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Computed GEO cell-type proportions from ${cleanAccession} supplementary count matrices.`,
            keyFields,
            sources: [
              buildGeoRecordUrl(cleanAccession),
              fileListUrl,
              archivePayload.archiveUrl,
              ...matchingMembers.slice(0, 4).map((member) => buildGeoRecordUrl(member.gsm_accession)),
            ],
            limitations: [
              "This tool depends on a per-cell label column already being present in the supplementary CSV.",
              "Donor subset filtering is based on GEO sample metadata rather than re-inferred from the expression matrix.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_geo_cell_type_proportions.v1",
          result_status: "ok",
          accession: cleanAccession,
          organism: organismFilter || null,
          sample_accessions: matchingMembers.map((member) => member.gsm_accession),
          donor_filter: cleanSampleAccession ? null : { field: diseaseField, value: diseaseValue },
          cluster_column: clusterField,
          total_cells: totalRows,
          cell_type_proportions: filteredProportions,
          all_cell_type_proportions: allProportions,
          sample_summaries: sampleSummaries,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error computing GEO cell-type proportions: ${detail}` }],
        structuredContent: {
          schema: "get_geo_cell_type_proportions.v1",
          result_status: "error",
          accession: cleanAccession,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: Search RefSeq (NCBI Entrez)
// ============================================
server.registerTool(
  "search_refseq_sequences",
  {
    description:
      "Searches NCBI RefSeq nucleotide records (nuccore; NM/NR/NC/NG/XM/XR accessions) or RefSeq protein records (NP/XP/WP accessions) using Entrez. " +
      "Use for locating curated RefSeq accessions by gene symbol, keyword, or accession stem; follow up with get_refseq_record for a specific hit.",
    inputSchema: {
      query: z.string().describe("Entrez query, e.g. 'TP53[Gene Name]', 'NM_000546', 'brca1 AND human', or 'BRCA1[Gene Name]'."),
      moleculeType: z.enum(["nucleotide", "protein"]).optional().describe("Search nuccore (transcripts/genomic RefSeq) or protein. Default 'nucleotide'."),
      organism: z.string().optional().describe("Optional organism filter, e.g. 'Homo sapiens' or 'Mus musculus' (adds Organism field to the query)."),
      refseqOnly: z.boolean().optional().describe("When true (default), restricts to refseq[filter]. Set false to search the full nucleotide/protein index."),
      maxResults: z.number().optional().describe("Maximum records to return (default 15, max 50)."),
    },
  },
  async ({ query, moleculeType, organism, refseqOnly, maxResults }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a RefSeq search query." }] };
    }

    const db = (moleculeType || "nucleotide") === "protein" ? "protein" : "nuccore";
    const useRefseqOnly = refseqOnly !== false;
    const limit = Math.min(Math.max(1, Math.round(maxResults || 15)), 50);
    let term = `(${normalizedQuery})`;
    if (useRefseqOnly) {
      term += " AND refseq[filter]";
    }
    const org = normalizeWhitespace(organism || "");
    if (org) {
      term += ` AND "${org}"[Organism]`;
    }

    try {
      const searchData = await refseqEsearch(db, term, limit);
      const idList = Array.isArray(searchData?.esearchresult?.idlist) ? searchData.esearchresult.idlist : [];
      const totalCount = Number(searchData?.esearchresult?.count || 0);

      if (idList.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No RefSeq (${db}) matches for this query.`,
              keyFields: [
                `Query: ${normalizedQuery}`,
                `Entrez db: ${db}`,
                `RefSeq filter: ${useRefseqOnly ? "on" : "off"}`,
                org ? `Organism: ${org}` : null,
              ].filter(Boolean),
              sources: [
                "https://www.ncbi.nlm.nih.gov/refseq/",
                `https://www.ncbi.nlm.nih.gov/${db === "protein" ? "protein" : "nuccore"}/?term=${encodeURIComponent(term)}`,
              ],
              limitations: [
                "Try a simpler gene symbol query with organism filter, or search the opposite moleculeType (nuccore vs protein).",
              ],
            }),
          }],
          structuredContent: {
            schema: "search_refseq_sequences.v1",
            result_status: "not_found_or_empty",
            entrez_db: db,
            query: normalizedQuery,
            refseq_filter: useRefseqOnly,
            organism: org || null,
            total_count: 0,
            records: [],
          },
        };
      }

      const summaryPayload = await refseqEsummary(db, idList);
      const docs = parseRefseqEsummaryDocs(summaryPayload);
      const records = docs.map((doc) => summarizeRefseqDocSum(doc, db));
      const lines = records.map((rec, idx) => {
        const acc = rec.accessionversion || rec.caption || rec.uid;
        const bit = [
          `${String(idx + 1).padStart(3)}. ${acc}`,
          rec.title && rec.title.length > 140 ? `${rec.title.slice(0, 137)}...` : rec.title,
          rec.organism ? `[${rec.organism}]` : "",
          rec.length ? `len=${rec.length}` : "",
        ].filter(Boolean);
        return bit.join(" | ");
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `RefSeq search (${db}): ${totalCount} hit(s). Showing ${records.length}.`,
            keyFields: [
              `Query: ${normalizedQuery}`,
              `Entrez db: ${db}`,
              `RefSeq filter: ${useRefseqOnly ? "on" : "off"}`,
              org ? `Organism: ${org}` : null,
              "\nResults:",
              ...lines,
            ].filter(Boolean),
            sources: [
              "https://www.ncbi.nlm.nih.gov/refseq/",
              `https://www.ncbi.nlm.nih.gov/${db === "protein" ? "protein" : "nuccore"}/?term=${encodeURIComponent(term)}`,
            ],
            limitations: [
              "Summaries are Entrez esummary fields only—no raw FASTA or full GenBank flatfile here. Use get_refseq_record on an accession or UID for consistent metadata.",
            ],
          }),
        }],
        structuredContent: {
          schema: "search_refseq_sequences.v1",
          result_status: "ok",
          entrez_db: db,
          query: normalizedQuery,
          refseq_filter: useRefseqOnly,
          organism: org || null,
          total_count: totalCount,
          records,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error searching RefSeq: ${detail}` }],
        structuredContent: {
          schema: "search_refseq_sequences.v1",
          result_status: "error",
          entrez_db: db,
          query: normalizedQuery,
          refseq_filter: useRefseqOnly,
          organism: org || null,
          total_count: 0,
          records: [],
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: Get RefSeq record metadata
// ============================================
server.registerTool(
  "get_refseq_record",
  {
    description:
      "Fetches detailed RefSeq metadata from NCBI Entrez (nuccore or protein) for a RefSeq accession (e.g. NM_000546.6, NP_000537.3) or numeric GI/UID. " +
      "Returns accession version, title, organism, sequence length, molecule type, map location when present, a stable NCBI link, and GenBank feature annotations when available.",
    inputSchema: {
      identifier: z.string().describe("RefSeq accession with or without version, Entrez UID, or accession text accepted by Entrez, e.g. 'NM_000546', '1808862652'."),
      moleculeType: z.enum(["auto", "nucleotide", "protein"]).optional().describe("Force Entrez db: nuccore vs protein. Default 'auto' uses NP/XP/WP/... → protein, otherwise nuccore."),
    },
  },
  async ({ identifier, moleculeType }) => {
    const rawId = normalizeWhitespace(identifier || "");
    if (!rawId) {
      return { content: [{ type: "text", text: "Provide a RefSeq accession or Entrez UID." }] };
    }

    const mode = moleculeType || "auto";
    const db = mode === "protein"
      ? "protein"
      : mode === "nucleotide"
        ? "nuccore"
        : inferRefseqEntrezDbFromIdentifier(rawId);

    try {
      const { uids, totalCount } = await resolveRefseqUidsForLookup(db, rawId, 1);
      if (!uids.length) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No RefSeq (${db}) record found for "${rawId}".`,
              keyFields: [`Identifier: ${rawId}`, `Entrez db: ${db}`, totalCount > 1 ? `Entrez reports ${totalCount} match(es); refine the accession.` : null].filter(Boolean),
              sources: [
                "https://www.ncbi.nlm.nih.gov/refseq/",
                `https://www.ncbi.nlm.nih.gov/${db === "protein" ? "protein" : "nuccore"}/?term=${encodeURIComponent(rawId)}`,
              ],
              limitations: [
                "If an accession moved, try without the version number or use search_refseq_sequences to list current RefSeq accessions.",
              ],
            }),
          }],
          structuredContent: {
            schema: "get_refseq_record.v1",
            result_status: "not_found_or_empty",
            entrez_db: db,
            identifier: rawId,
          },
        };
      }

      const summaryPayload = await refseqEsummary(db, uids);
      const docs = parseRefseqEsummaryDocs(summaryPayload);
      if (!docs.length) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `RefSeq esummary returned no document for "${rawId}".`,
              keyFields: [`Identifier: ${rawId}`, `Entrez db: ${db}`],
              sources: [`https://www.ncbi.nlm.nih.gov/${db === "protein" ? "protein" : "nuccore"}/${encodeURIComponent(uids[0])}`],
              limitations: ["Try moleculeType 'nucleotide' vs 'protein' if the accession class was mis-inferred."],
            }),
          }],
          structuredContent: {
            schema: "get_refseq_record.v1",
            result_status: "not_found_or_empty",
            entrez_db: db,
            identifier: rawId,
          },
        };
      }

      const doc = docs[0];
      const rec = summarizeRefseqDocSum(doc, db);
      let featureAnnotations = null;
      let featureAnnotationError = "";
      if (db === "nuccore") {
        try {
          const xml = await refseqEfetchXml(db, rec.accessionversion || rec.uid || rawId);
          featureAnnotations = parseRefseqFeatureAnnotations(xml, {
            maxFeatures: 20,
            highlightKeys: ["CDS", "sig_peptide", "mat_peptide"],
          });
        } catch (error) {
          featureAnnotationError = compactErrorMessage(error?.message || "feature annotations unavailable", 180);
        }
      }
      const keyFields = [
        `Identifier: ${rawId}`,
        `Entrez db: ${db}`,
        `UID: ${rec.uid}`,
        `Accession: ${rec.accessionversion || rec.caption || "n/a"}`,
        `Title: ${rec.title || "n/a"}`,
      ];
      if (rec.organism) keyFields.push(`Organism: ${rec.organism}`);
      if (rec.length) keyFields.push(`${db === "protein" ? "Length (aa)" : "Length (bases)"}: ${rec.length}`);
      if (rec.molecule) keyFields.push(`Molecule: ${rec.molecule}`);
      if (rec.location) keyFields.push(`Map / context: ${rec.location}`);
      if (rec.sourcedb) keyFields.push(`Source: ${rec.sourcedb}`);
      if (featureAnnotations?.highlighted_features?.length) {
        keyFields.push("\nFeature annotations:");
        for (const feature of featureAnnotations.highlighted_features.slice(0, 8)) {
          keyFields.push(feature.summary_line);
        }
      } else if (featureAnnotationError) {
        keyFields.push(`Feature annotations unavailable: ${featureAnnotationError}`);
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `RefSeq metadata for ${rec.accessionversion || rec.caption || rec.uid}.`,
            keyFields,
            sources: [rec.record_url],
            limitations: [
              "This tool does not return raw sequence text; open the NCBI link or use efetch for FASTA/GenBank when needed.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_refseq_record.v1",
          result_status: "ok",
          entrez_db: db,
          identifier: rawId,
          feature_count: featureAnnotations?.feature_count || 0,
          highlighted_features: featureAnnotations?.highlighted_features || [],
          features: featureAnnotations?.features || [],
          feature_lookup: featureAnnotations?.feature_lookup || {},
          feature_annotation_error: featureAnnotationError || null,
          ...rec,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error fetching RefSeq record: ${detail}` }],
        structuredContent: {
          schema: "get_refseq_record.v1",
          result_status: "error",
          entrez_db: db,
          identifier: rawId,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// UCSC Genome Browser REST API (api.genome.ucsc.edu)
// ============================================
const UCSC_API_BASE = "https://api.genome.ucsc.edu";

function safeDecodeURIComponent(value) {
  const raw = String(value || "");
  try {
    return decodeURIComponent(raw.replace(/\+/g, " "));
  } catch {
    return raw;
  }
}

function buildUcscBrowserUrl(genome, position) {
  const db = String(genome || "").trim();
  const pos = String(position || "").trim();
  if (!db || !pos) {
    return "https://genome.ucsc.edu/";
  }
  return `https://genome.ucsc.edu/cgi-bin/hgTracks?db=${encodeURIComponent(db)}&position=${encodeURIComponent(pos)}`;
}

function buildUcscApiSearchUrl(search, genome, categories) {
  const params = new URLSearchParams({
    search: String(search || "").trim(),
    genome: String(genome || "").trim(),
  });
  const cat = String(categories || "").trim();
  if (cat) {
    params.set("categories", cat);
  }
  return `${UCSC_API_BASE}/search?${params.toString()}`;
}

function buildUcscGetDataSequenceParams({ genome, chrom, start, end, revComp, hubUrl }) {
  const parts = [];
  const hub = String(hubUrl || "").trim();
  if (hub) {
    parts.push(`hubUrl=${encodeURIComponent(hub)}`);
  }
  parts.push(`genome=${encodeURIComponent(String(genome || "").trim())}`);
  parts.push(`chrom=${encodeURIComponent(String(chrom || "").trim())}`);
  parts.push(`start=${Math.round(Number(start))}`);
  parts.push(`end=${Math.round(Number(end))}`);
  if (revComp) {
    parts.push("revComp=1");
  }
  return parts.join(";");
}

function buildUcscGetDataTrackParams({ genome, track, chrom, start, end, maxItemsOutput, hubUrl }) {
  const parts = [];
  const hub = String(hubUrl || "").trim();
  if (hub) {
    parts.push(`hubUrl=${encodeURIComponent(hub)}`);
  }
  parts.push(`genome=${encodeURIComponent(String(genome || "").trim())}`);
  parts.push(`track=${encodeURIComponent(String(track || "").trim())}`);
  const chr = String(chrom || "").trim();
  if (chr) {
    parts.push(`chrom=${encodeURIComponent(chr)}`);
  }
  if (start !== undefined && end !== undefined && String(chrom || "").trim()) {
    parts.push(`start=${Math.round(Number(start))}`);
    parts.push(`end=${Math.round(Number(end))}`);
  }
  const cap = Math.max(1, Math.min(500, Math.round(maxItemsOutput || 25)));
  parts.push(`maxItemsOutput=${cap}`);
  return parts.join(";");
}

function flattenUcscSearchMatches(payload, maxTotal) {
  const limit = Math.max(1, Math.min(100, Math.round(maxTotal || 40)));
  const sections = Array.isArray(payload?.positionMatches) ? payload.positionMatches : [];
  const out = [];
  for (const section of sections) {
    const trackName = normalizeWhitespace(section?.trackName || section?.name || "") || "";
    const trackDesc = normalizeWhitespace(section?.description || "") || null;
    const matches = Array.isArray(section?.matches) ? section.matches : [];
    for (const m of matches) {
      if (out.length >= limit) {
        return out;
      }
      const position = normalizeWhitespace(m?.position || "") || null;
      out.push({
        track: trackName,
        track_description: trackDesc,
        position,
        name: normalizeWhitespace(m?.posName || "") || null,
        match_id: normalizeWhitespace(safeDecodeURIComponent(m?.hgFindMatches || "")) || null,
        description: normalizeWhitespace(m?.description || "") || null,
        canonical: Boolean(m?.canonical),
        browser_url: position ? buildUcscBrowserUrl(payload?.genome, position) : buildUcscBrowserUrl(payload?.genome, ""),
      });
    }
  }
  return out;
}

function extractUcscTrackRows(payload, trackName) {
  if (!payload || typeof payload !== "object") {
    return [];
  }
  const t = String(trackName || "").trim();
  if (t && Array.isArray(payload[t])) {
    return payload[t];
  }
  const skip = new Set([
    "downloadTime",
    "downloadTimeStamp",
    "genome",
    "dataTime",
    "dataTimeStamp",
    "trackType",
    "track",
    "chrom",
    "chromSize",
    "bigDataUrl",
    "start",
    "end",
  ]);
  for (const [key, val] of Object.entries(payload)) {
    if (skip.has(key)) {
      continue;
    }
    if (Array.isArray(val)) {
      return val;
    }
  }
  return [];
}

function summarizeUcscTrackRow(row) {
  if (!row || typeof row !== "object") {
    return null;
  }
  const gene = normalizeWhitespace(row.geneName || row.name2 || "") || null;
  const tx = normalizeWhitespace(row.name || "") || null;
  const start = Number.isFinite(Number(row.chromStart)) ? Number(row.chromStart) : null;
  const end = Number.isFinite(Number(row.chromEnd)) ? Number(row.chromEnd) : null;
  const chrom = normalizeWhitespace(row.chrom || "") || null;
  const strand = normalizeWhitespace(row.strand || "") || null;
  const interval = chrom && start !== null && end !== null ? `${chrom}:${start}-${end}` : null;
  const bits = [
    gene || tx || "record",
    interval,
    strand ? `(${strand})` : null,
    tx && gene ? `tx ${tx}` : null,
  ].filter(Boolean);
  return {
    summary: bits.join(" "),
    gene,
    transcript: tx,
    chrom,
    chromStart: start,
    chromEnd: end,
    strand,
  };
}

function summarizeDnaComposition(dna) {
  const text = typeof dna === "string" ? dna.toUpperCase() : "";
  let aCount = 0;
  let cCount = 0;
  let gCount = 0;
  let tCount = 0;
  let nonAcgtCount = 0;

  for (const base of text) {
    if (base === "A") aCount += 1;
    else if (base === "C") cCount += 1;
    else if (base === "G") gCount += 1;
    else if (base === "T") tCount += 1;
    else nonAcgtCount += 1;
  }

  const canonicalBaseCount = aCount + cCount + gCount + tCount;
  const gcCount = cCount + gCount;
  const atCount = aCount + tCount;
  const gcFraction = canonicalBaseCount > 0 ? gcCount / canonicalBaseCount : null;
  const atFraction = canonicalBaseCount > 0 ? atCount / canonicalBaseCount : null;

  return {
    aCount,
    cCount,
    gCount,
    tCount,
    gcCount,
    atCount,
    canonicalBaseCount,
    nonAcgtCount,
    gcFraction,
    gcPercent: gcFraction === null ? null : gcFraction * 100,
    atFraction,
    atPercent: atFraction === null ? null : atFraction * 100,
  };
}

// ============================================
// TOOL: UCSC genome search
// ============================================
server.registerTool(
  "search_ucsc_genome",
  {
    description:
      "Searches within a UCSC Genome Browser assembly ( positions, genes, tracks metadata ) via the public REST /search API. " +
      "Use to resolve gene symbols (e.g. BRCA1) or features to chromosomal coordinates and to obtain direct Genome Browser links. Default assembly is hg38.",
    inputSchema: {
      search: z.string().describe("Search string, e.g. gene symbol 'BRCA1', 'HOXA1', region text, or track keyword."),
      genome: z.string().optional().describe("UCSC assembly name, e.g. hg38, hg19, mm39, mm10 (default hg38)."),
      categories: z.enum(["helpDocs", "publicHubs", "trackDb"]).optional().describe(
        "Restrict search scope: help documentation, public hubs, or trackDb settings. Omit for the default feature/position search.",
      ),
      maxMatches: z.number().optional().describe("Maximum flattened hits to return (default 40, max 100)."),
    },
  },
  async ({ search, genome, categories, maxMatches }) => {
    const q = normalizeWhitespace(search || "");
    if (!q) {
      return { content: [{ type: "text", text: "Provide a UCSC search string." }] };
    }
    const asm = normalizeWhitespace(genome || "") || "hg38";
    const cap = Math.max(1, Math.min(100, Math.round(maxMatches || 40)));

    try {
      const url = buildUcscApiSearchUrl(q, asm, categories || "");
      const payload = await fetchJsonWithRetry(url, { retries: 2, timeoutMs: 20000, maxBackoffMs: 4000 });
      const matches = categories
        ? []
        : flattenUcscSearchMatches({ ...payload, genome: asm }, cap);

      if (!categories && matches.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No UCSC position matches for "${q}" on ${asm}.`,
              keyFields: [`Search: ${q}`, `Genome: ${asm}`],
              sources: [url, buildUcscBrowserUrl(asm, q)],
              limitations: [
                "Try hg19 if the assembly is wrong, or a more specific symbol. For raw sequence use get_ucsc_genomic_sequence with coordinates from a hit.",
              ],
            }),
          }],
          structuredContent: {
            schema: "search_ucsc_genome.v1",
            result_status: "not_found_or_empty",
            genome: asm,
            search: q,
            categories: categories || null,
            matches: [],
          },
        };
      }

      if (categories) {
        const textJson = JSON.stringify(payload, null, 2);
        const clipped = textJson.length > 12000 ? `${textJson.slice(0, 11997)}...` : textJson;
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `UCSC search (${asm}, categories=${categories}) for "${q}".`,
              keyFields: ["Raw JSON (trimmed if long):", clipped],
              sources: [url, "https://api.genome.ucsc.edu/goldenPath/help/api.html"],
              limitations: ["Category searches return API-specific JSON shapes; inspect fields for hub or help hits."],
            }),
          }],
          structuredContent: {
            schema: "search_ucsc_genome.v1",
            result_status: "ok",
            genome: asm,
            search: q,
            categories,
            raw: payload,
          },
        };
      }

      const lines = matches.map((m, i) => {
        const head = `${String(i + 1).padStart(3)}. [${m.track}] ${m.name || "—"} @ ${m.position || "—"}`;
        const tail = m.description && m.description.length > 0
          ? (m.description.length > 160 ? `${m.description.slice(0, 157)}...` : m.description)
          : "";
        return tail ? `${head}\n    ${tail}` : head;
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `UCSC search on ${asm}: ${matches.length} hit(s) for "${q}".`,
            keyFields: [
              `Search: ${q}`,
              `Genome: ${asm}`,
              "\nHits:",
              ...lines,
            ],
            sources: [
              url,
              ...matches.map((m) => m.browser_url).filter(Boolean).slice(0, 5),
            ],
            limitations: [
              "Hits can include secondary matches (e.g. similarly named genes). Prefer canonical rows and verify on the browser.",
              "Respect UCSC API usage: avoid rapid-fire requests; prefer narrow intervals for track/sequence follow-ups.",
            ],
          }),
        }],
        structuredContent: {
          schema: "search_ucsc_genome.v1",
          result_status: "ok",
          genome: asm,
          search: q,
          categories: null,
          matches,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error querying UCSC search API: ${detail}` }],
        structuredContent: {
          schema: "search_ucsc_genome.v1",
          result_status: "error",
          genome: asm,
          search: q,
          categories: categories || null,
          matches: [],
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: UCSC genomic sequence
// ============================================
server.registerTool(
  "get_ucsc_genomic_sequence",
  {
    description:
      "Returns genomic DNA via UCSC REST getData/sequence for a UCSC assembly, chromosome, and half-open interval (start 0-based, end 1-based per UCSC API). " +
      "Use after search_ucsc_genome for coordinates. Enforces a maximum span to avoid huge downloads.",
    inputSchema: {
      genome: z.string().describe("UCSC database, e.g. hg38, hg19, mm39."),
      chrom: z.string().describe("Chromosome, e.g. chr17, chrM."),
      start: z.number().describe("Start position (0-based, inclusive)."),
      end: z.number().describe("End position (1-based per UCSC; exclusive in zero-based terms)."),
      revComp: z.boolean().optional().describe("If true, return reverse complement."),
      maxBases: z.number().optional().describe("Maximum interval width allowed (default 250000, hard cap 1000000)."),
      hubUrl: z.string().optional().describe("Optional track/assembly hub URL when sequence is defined on a hub."),
    },
  },
  async ({ genome, chrom, start, end, revComp, maxBases, hubUrl }) => {
    const asm = normalizeWhitespace(genome || "");
    const chr = normalizeWhitespace(chrom || "");
    const s = Math.round(Number(start));
    const e = Math.round(Number(end));
    if (!asm || !chr || !Number.isFinite(s) || !Number.isFinite(e)) {
      return { content: [{ type: "text", text: "Provide genome, chrom, start, and end." }] };
    }
    if (e <= s) {
      return { content: [{ type: "text", text: "UCSC sequence requires end > start." }] };
    }
    const widthLimit = Math.max(1000, Math.min(1_000_000, Math.round(maxBases || 250_000)));
    const width = e - s;
    if (width > widthLimit) {
      return {
        content: [{
          type: "text",
          text: `Interval width ${width} bp exceeds maxBases=${widthLimit}. Narrow the range or raise maxBases up to 1000000.`,
        }],
      };
    }

    try {
      const query = buildUcscGetDataSequenceParams({
        genome: asm,
        chrom: chr,
        start: s,
        end: e,
        revComp: Boolean(revComp),
        hubUrl,
      });
      const url = `${UCSC_API_BASE}/getData/sequence?${query}`;
      const payload = await fetchJsonWithRetry(url, { retries: 2, timeoutMs: Math.min(120_000, 15_000 + width / 50), maxBackoffMs: 4000 });
      const dnaRaw = payload?.dna;
      const dna = typeof dnaRaw === "string" ? dnaRaw : "";
      if (!dna) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: "UCSC returned no sequence for this request.",
              keyFields: [`URL: ${url}`],
              sources: [url, "https://api.genome.ucsc.edu/goldenPath/help/api.html"],
              limitations: ["Check assembly, chromosome name (e.g. chr1 vs 1), and coordinates."],
            }),
          }],
          structuredContent: {
            schema: "get_ucsc_genomic_sequence.v1",
            result_status: "not_found_or_empty",
            genome: asm,
            chrom: chr,
            start: s,
            end: e,
          },
        };
      }

      const pos = `${chr}:${s}-${e}`;
      const previewCap = 400;
      const preview = dna.length > previewCap ? `${dna.slice(0, previewCap)}...` : dna;
      const composition = summarizeDnaComposition(dna);
      const structSeqMax = Math.max(2048, Math.min(50_000, widthLimit));
      const structSeq = dna.length <= structSeqMax ? dna : `${dna.slice(0, structSeqMax)}...`;

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `UCSC sequence ${asm} ${pos} (${dna.length} bp).`,
            keyFields: [
              `Genome: ${asm}`,
              `Interval: ${pos}`,
              `Length: ${dna.length}`,
              `Canonical bases counted: ${composition.canonicalBaseCount}`,
              `GC fraction: ${composition.gcFraction === null ? "n/a" : composition.gcFraction.toFixed(4)}`,
              `GC percent: ${composition.gcPercent === null ? "n/a" : `${composition.gcPercent.toFixed(2)}%`}`,
              composition.nonAcgtCount > 0 ? `Non-ACGT bases: ${composition.nonAcgtCount}` : null,
              revComp ? "strand: reverse complement" : null,
              "\nSequence preview:",
              preview,
            ].filter(Boolean),
            sources: [url, buildUcscBrowserUrl(asm, pos)],
            limitations: [
              "Coordinates follow the UCSC REST API convention; confirm numbering when comparing to other tools.",
              "Large intervals can be slow; keep windows as small as practical.",
              dna.length > structSeqMax
                ? `Structured JSON sequence is truncated after ${structSeqMax} characters; narrow the interval if you need the entire sequence returned in one payload.`
                : "Structured JSON includes the full sequence for this interval.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_ucsc_genomic_sequence.v1",
          result_status: "ok",
          genome: asm,
          chrom: chr,
          start: s,
          end: e,
          rev_comp: Boolean(revComp),
          length: dna.length,
          canonical_base_count: composition.canonicalBaseCount,
          gc_fraction: composition.gcFraction,
          gc_percent: composition.gcPercent,
          at_fraction: composition.atFraction,
          at_percent: composition.atPercent,
          gc_count: composition.gcCount,
          at_count: composition.atCount,
          a_count: composition.aCount,
          c_count: composition.cCount,
          g_count: composition.gCount,
          t_count: composition.tCount,
          non_acgt_count: composition.nonAcgtCount,
          sequence: structSeq,
          sequence_truncated: dna.length > structSeqMax,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error fetching UCSC sequence: ${detail}` }],
        structuredContent: {
          schema: "get_ucsc_genomic_sequence.v1",
          result_status: "error",
          genome: asm,
          chrom: chr,
          start: s,
          end: e,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: UCSC track data (interval)
// ============================================
server.registerTool(
  "get_ucsc_track_data",
  {
    description:
      "Fetches rows from a UCSC Genome Browser track via REST getData/track (genePred, bed, bigBed-supported tracks, etc.). " +
      "Always specify genome and track; strongly prefer chrom with start/end to limit output size. Default item cap is 25 (max 500).",
    inputSchema: {
      genome: z.string().describe("UCSC assembly, e.g. hg38."),
      track: z.string().describe("Track name, e.g. knownGene, ncbiRefSeq, refGene (assembly-dependent)."),
      chrom: z.string().optional().describe("Chromosome, e.g. chr17—recommended whenever possible."),
      start: z.number().optional().describe("Start (0-based) when chrom is set; must be used with end."),
      end: z.number().optional().describe("End (UCSC API end coordinate) when chrom is set; must be used with start."),
      maxItems: z.number().optional().describe("maxItemsOutput (default 25, max 500)."),
      hubUrl: z.string().optional().describe("Hub URL when querying hub tracks."),
    },
  },
  async ({ genome, track, chrom, start, end, maxItems, hubUrl }) => {
    const asm = normalizeWhitespace(genome || "");
    const tr = normalizeWhitespace(track || "");
    if (!asm || !tr) {
      return { content: [{ type: "text", text: "Provide genome and track name." }] };
    }
    const chr = normalizeWhitespace(chrom || "");
    const hasInterval = chr && start !== undefined && end !== undefined;
    if (chr && !hasInterval) {
      return {
        content: [{
          type: "text",
          text: "When chrom is provided, include both start and end to bound the query and avoid huge responses.",
        }],
      };
    }
    if (hasInterval) {
      const s = Math.round(Number(start));
      const e = Math.round(Number(end));
      if (!Number.isFinite(s) || !Number.isFinite(e) || e <= s) {
        return { content: [{ type: "text", text: "Provide valid start and end with end > start." }] };
      }
    }

    try {
      const query = buildUcscGetDataTrackParams({
        genome: asm,
        track: tr,
        chrom: chr || undefined,
        start: hasInterval ? Math.round(Number(start)) : undefined,
        end: hasInterval ? Math.round(Number(end)) : undefined,
        maxItemsOutput: maxItems,
        hubUrl,
      });
      const url = `${UCSC_API_BASE}/getData/track?${query}`;
      const payload = await fetchJsonWithRetry(url, { retries: 2, timeoutMs: 25000, maxBackoffMs: 4000 });
      const rows = extractUcscTrackRows(payload, tr);
      const cap = Math.max(1, Math.min(500, Math.round(maxItems || 25)));
      const sliced = rows.slice(0, cap);
      const summaries = sliced.map(summarizeUcscTrackRow).filter(Boolean);
      const lines = summaries.map((row, i) => `${String(i + 1).padStart(3)}. ${row.summary}`);

      if (sliced.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No rows returned for track "${tr}" on ${asm} (with current filters).`,
              keyFields: [`URL: ${url}`],
              sources: [url, buildUcscBrowserUrl(asm, chr || "")],
              limitations: [
                "Track names vary by assembly; use search_ucsc_genome or the UCSC track list if unsure.",
                "Some tracks are restricted or empty on certain chromosomes.",
              ],
            }),
          }],
          structuredContent: {
            schema: "get_ucsc_track_data.v1",
            result_status: "not_found_or_empty",
            genome: asm,
            track: tr,
            chrom: chr || null,
            items: [],
          },
        };
      }

      const posLabel = hasInterval ? `${chr}:${Math.round(Number(start))}-${Math.round(Number(end))}` : (chr || "all");

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `UCSC track "${tr}" on ${asm} (${posLabel}): ${sliced.length} row(s).`,
            keyFields: [
              `Genome: ${asm}`,
              `Track: ${tr}`,
              `Region: ${posLabel}`,
              "\nRows:",
              ...lines,
            ],
            sources: [
              url,
              buildUcscBrowserUrl(asm, hasInterval ? posLabel : ""),
            ],
            limitations: [
              "Rows are capped; widen maxItems (up to 500) or narrow coordinates if needed.",
              "Binary or summary tracks may return fewer interpretable fields than gene tracks.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_ucsc_track_data.v1",
          result_status: "ok",
          genome: asm,
          track: tr,
          chrom: chr || null,
          start: hasInterval ? Math.round(Number(start)) : null,
          end: hasInterval ? Math.round(Number(end)) : null,
          items_returned: sliced.length,
          items: sliced,
          summaries,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error fetching UCSC track data: ${detail}` }],
        structuredContent: {
          schema: "get_ucsc_track_data.v1",
          result_status: "error",
          genome: asm,
          track: tr,
          chrom: chr || null,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// Ensembl canonical transcript / TSS lookup
// ============================================
function normalizeEnsemblSpecies(species) {
  const raw = normalizeWhitespace(species || "").toLowerCase();
  if (!raw || raw === "human" || raw === "homo sapiens" || raw === "homo_sapiens") {
    return "homo_sapiens";
  }
  if (raw === "mouse" || raw === "mus musculus" || raw === "mus_musculus") {
    return "mus_musculus";
  }
  return raw.replace(/\s+/g, "_");
}

function normalizeEnsemblSequenceSpecies(species) {
  const normalized = normalizeEnsemblSpecies(species);
  if (normalized === "homo_sapiens") return "human";
  if (normalized === "mus_musculus") return "mouse";
  return normalized;
}

function stripEnsemblVersion(identifier) {
  return String(identifier || "").trim().split(".")[0] || "";
}

function getEnsemblRestBaseForAssembly(assembly) {
  const normalized = normalizeWhitespace(assembly || "").toUpperCase();
  return normalized === "GRCH37" ? "https://grch37.rest.ensembl.org" : ENSEMBL_REST_API;
}

function buildEnsemblLookupUrl({ identifier, species, assembly }) {
  const base = getEnsemblRestBaseForAssembly(assembly);
  const raw = normalizeWhitespace(identifier || "");
  const encoded = encodeURIComponent(raw);
  if (/^ENSG/i.test(raw)) {
    return `${base}/lookup/id/${encoded}?expand=1;content-type=application/json`;
  }
  const normalizedSpecies = normalizeEnsemblSpecies(species);
  return `${base}/lookup/symbol/${encodeURIComponent(normalizedSpecies)}/${encoded}?expand=1;content-type=application/json`;
}

function buildEnsemblRegionSequenceUrl({ species, assembly, seqRegionName, start, end, strand }) {
  const base = getEnsemblRestBaseForAssembly(assembly);
  const sequenceSpecies = normalizeEnsemblSequenceSpecies(species);
  const region = `${seqRegionName}:${Math.round(Number(start))}..${Math.round(Number(end))}:${Math.round(Number(strand))}`;
  return `${base}/sequence/region/${encodeURIComponent(sequenceSpecies)}/${encodeURIComponent(region)}?content-type=text/plain`;
}

function resolveCanonicalTranscript(geneRecord) {
  const transcripts = Array.isArray(geneRecord?.Transcript) ? geneRecord.Transcript : [];
  const canonicalStableId = stripEnsemblVersion(geneRecord?.canonical_transcript || "");
  if (canonicalStableId) {
    const direct = transcripts.find((tx) => stripEnsemblVersion(tx?.id || "") === canonicalStableId);
    if (direct) {
      return direct;
    }
  }
  return transcripts.find((tx) => Number(tx?.is_canonical || 0) === 1) || transcripts[0] || null;
}

server.registerTool(
  "get_ensembl_canonical_transcript",
  {
    description:
      "Resolve a gene symbol or Ensembl gene ID to Ensembl's canonical transcript, transcript bounds, strand, and TSS. " +
      "Optionally return a TSS-centered genomic sequence window on GRCh38/GRCh37, which is useful for promoter and upstream/downstream sequence questions.",
    inputSchema: {
      identifier: z.string().describe("Gene symbol or Ensembl gene ID, e.g. 'HTRA1' or 'ENSG00000166033'."),
      species: z.string().optional().describe("Ensembl species name, e.g. 'human', 'homo_sapiens', or 'mouse'. Default human."),
      assembly: z.enum(["GRCh38", "GRCh37"]).optional().describe("Human Ensembl assembly. Default GRCh38."),
      includeSequenceWindow: z.boolean().optional().describe("If true, fetch a genomic sequence window around the canonical TSS."),
      upstreamBp: z.number().optional().describe("Bases upstream of the canonical TSS to include when fetching sequence (default 0)."),
      downstreamBp: z.number().optional().describe("Bases downstream of the canonical TSS to include when fetching sequence (default 0)."),
      sequenceOrientation: z.enum(["reference", "transcript"]).optional().describe(
        "Sequence orientation when a window is requested: reference strand (default) or transcript strand.",
      ),
    },
  },
  async ({ identifier, species, assembly, includeSequenceWindow, upstreamBp, downstreamBp, sequenceOrientation }) => {
    const rawIdentifier = normalizeWhitespace(identifier || "");
    if (!rawIdentifier) {
      return { content: [{ type: "text", text: "Provide a gene symbol or Ensembl gene ID." }] };
    }

    const normalizedSpecies = normalizeEnsemblSpecies(species);
    const normalizedAssembly = normalizeWhitespace(assembly || "") || "GRCh38";
    const upstream = Math.max(0, Math.round(Number(upstreamBp ?? 0)));
    const downstream = Math.max(0, Math.round(Number(downstreamBp ?? 0)));
    const windowRequested = Boolean(includeSequenceWindow) || upstream > 0 || downstream > 0;
    const requestedWindowWidth = upstream + downstream + 1;
    if (windowRequested && requestedWindowWidth > 50_000) {
      return {
        content: [{
          type: "text",
          text: `Requested TSS window (${requestedWindowWidth} bp) exceeds the 50000 bp limit. Narrow upstreamBp/downstreamBp.`,
        }],
      };
    }

    const lookupUrl = buildEnsemblLookupUrl({
      identifier: rawIdentifier,
      species: normalizedSpecies,
      assembly: normalizedAssembly,
    });

    try {
      const geneRecord = await fetchJsonWithRetry(lookupUrl, {
        retries: 2,
        timeoutMs: 20000,
        maxBackoffMs: 3000,
      });
      const canonicalTranscript = resolveCanonicalTranscript(geneRecord);
      if (!canonicalTranscript) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No transcript models returned by Ensembl for "${rawIdentifier}".`,
              keyFields: [
                `Identifier: ${rawIdentifier}`,
                `Species: ${normalizedSpecies}`,
                `Assembly: ${normalizedAssembly}`,
              ],
              sources: [lookupUrl],
              limitations: [
                "Try a canonical gene symbol or a stable Ensembl gene ID if the identifier was an alias.",
              ],
            }),
          }],
          structuredContent: {
            schema: "get_ensembl_canonical_transcript.v1",
            result_status: "not_found_or_empty",
            identifier: rawIdentifier,
            species: normalizedSpecies,
            assembly: normalizedAssembly,
          },
        };
      }

      const transcriptStart = Number(canonicalTranscript?.start);
      const transcriptEnd = Number(canonicalTranscript?.end);
      const transcriptStrand = Number(canonicalTranscript?.strand);
      const seqRegionName = normalizeWhitespace(canonicalTranscript?.seq_region_name || geneRecord?.seq_region_name || "");
      if (!Number.isFinite(transcriptStart) || !Number.isFinite(transcriptEnd) || !Number.isFinite(transcriptStrand) || !seqRegionName) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Ensembl returned an incomplete canonical transcript record for "${rawIdentifier}".`,
              keyFields: [
                `Identifier: ${rawIdentifier}`,
                `Species: ${normalizedSpecies}`,
                `Assembly: ${normalizedAssembly}`,
              ],
              sources: [lookupUrl],
              limitations: [
                "Transcript coordinate fields were missing, so TSS and sequence context could not be derived.",
              ],
            }),
          }],
          structuredContent: {
            schema: "get_ensembl_canonical_transcript.v1",
            result_status: "degraded",
            identifier: rawIdentifier,
            species: normalizedSpecies,
            assembly: normalizedAssembly,
          },
        };
      }

      const tss1Based = transcriptStrand >= 0 ? transcriptStart : transcriptEnd;
      const windowStart1Based = windowRequested
        ? Math.max(1, transcriptStrand >= 0 ? tss1Based - upstream : tss1Based - downstream)
        : null;
      const windowEnd1Based = windowRequested
        ? transcriptStrand >= 0 ? tss1Based + downstream : tss1Based + upstream
        : null;
      const sequenceStrand = windowRequested && sequenceOrientation === "transcript" ? transcriptStrand : 1;

      let sequence = null;
      let sequenceUrl = null;
      let sequenceError = "";
      if (windowRequested && windowStart1Based !== null && windowEnd1Based !== null) {
        sequenceUrl = buildEnsemblRegionSequenceUrl({
          species: normalizedSpecies,
          assembly: normalizedAssembly,
          seqRegionName,
          start: windowStart1Based,
          end: windowEnd1Based,
          strand: sequenceStrand,
        });
        try {
          const response = await fetchWithRetry(sequenceUrl, {
            retries: 2,
            timeoutMs: 20000,
            maxBackoffMs: 3000,
          });
          sequence = String(await response.text()).replace(/\s+/g, "").toUpperCase() || null;
        } catch (error) {
          sequenceError = compactErrorMessage(error?.message || "sequence unavailable", 180);
        }
      }

      const canonicalTranscriptId = normalizeWhitespace(canonicalTranscript?.id || "") || null;
      const canonicalTranscriptVersion = normalizeWhitespace(geneRecord?.canonical_transcript || "") || canonicalTranscriptId;
      const geneId = normalizeWhitespace(geneRecord?.id || "") || null;
      const geneSymbol = normalizeWhitespace(geneRecord?.display_name || rawIdentifier) || rawIdentifier;
      const transcriptDisplayName = normalizeWhitespace(canonicalTranscript?.display_name || "") || null;
      const transcriptBiotype = normalizeWhitespace(canonicalTranscript?.biotype || "") || null;
      const strandSymbol = transcriptStrand >= 0 ? "+" : "-";
      const transcriptInterval = `${seqRegionName}:${transcriptStart}-${transcriptEnd}`;
      const tssInterval = `${seqRegionName}:${tss1Based}`;
      const windowInterval = windowRequested && windowStart1Based !== null && windowEnd1Based !== null
        ? `${seqRegionName}:${windowStart1Based}-${windowEnd1Based}`
        : null;
      const resultStatus = windowRequested && sequenceError ? "degraded" : "ok";

      const keyFields = [
        `Identifier: ${rawIdentifier}`,
        `Species: ${normalizedSpecies}`,
        `Assembly: ${normalizedAssembly}`,
        `Gene: ${geneSymbol}${geneId ? ` (${geneId})` : ""}`,
        `Canonical transcript: ${canonicalTranscriptVersion}${transcriptDisplayName ? ` (${transcriptDisplayName})` : ""}`,
        transcriptBiotype ? `Biotype: ${transcriptBiotype}` : null,
        `Transcript interval: ${transcriptInterval} (${strandSymbol})`,
        `Canonical TSS (1-based): ${tssInterval}`,
      ].filter(Boolean);

      if (windowInterval) {
        keyFields.push(`Requested TSS window: ${windowInterval} (${windowEnd1Based - windowStart1Based + 1} bp, ${sequenceStrand === 1 ? "reference" : "transcript"} orientation)`);
        if (sequence) {
          keyFields.push("\nSequence:");
          keyFields.push(sequence);
        } else if (sequenceError) {
          keyFields.push(`Sequence unavailable: ${sequenceError}`);
        }
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Ensembl canonical transcript for ${geneSymbol}: ${canonicalTranscriptVersion || "n/a"} with TSS at ${tssInterval}.`,
            keyFields,
            sources: [lookupUrl, sequenceUrl].filter(Boolean),
            limitations: [
              "Canonical transcript selection follows Ensembl's current canonical transcript assignment for the requested assembly.",
              windowRequested && sequenceStrand !== 1
                ? "Sequence was returned in transcript orientation because sequenceOrientation='transcript' was requested."
                : "Sequence, when requested, is returned in reference orientation unless sequenceOrientation='transcript' is set.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_ensembl_canonical_transcript.v1",
          result_status: resultStatus,
          identifier: rawIdentifier,
          species: normalizedSpecies,
          assembly: normalizedAssembly,
          gene_id: geneId,
          gene_symbol: geneSymbol,
          canonical_transcript_id: canonicalTranscriptId,
          canonical_transcript: canonicalTranscriptVersion,
          canonical_transcript_display_name: transcriptDisplayName,
          canonical_transcript_biotype: transcriptBiotype,
          seq_region_name: seqRegionName,
          strand: transcriptStrand,
          transcript_start_1based: transcriptStart,
          transcript_end_1based: transcriptEnd,
          tss_1based: tss1Based,
          transcript_interval: transcriptInterval,
          tss_interval: tssInterval,
          upstream_bp: windowRequested ? upstream : null,
          downstream_bp: windowRequested ? downstream : null,
          sequence_orientation: windowRequested ? (sequenceStrand === 1 ? "reference" : "transcript") : null,
          window_start_1based: windowStart1Based,
          window_end_1based: windowEnd1Based,
          window_interval: windowInterval,
          sequence,
          sequence_error: sequenceError || null,
          lookup_url: lookupUrl,
          sequence_url: sequenceUrl,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error fetching Ensembl canonical transcript: ${detail}` }],
        structuredContent: {
          schema: "get_ensembl_canonical_transcript.v1",
          result_status: "error",
          identifier: rawIdentifier,
          species: normalizedSpecies,
          assembly: normalizedAssembly,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// ENCODE Portal REST API (www.encodeproject.org)
// ============================================
const ENCODE_PORTAL_BASE = "https://www.encodeproject.org";

const ENCODE_JSON_HEADERS = { Accept: "application/json", "User-Agent": "research-mcp/encode" };

function describeEncodePortalRef(ref) {
  if (ref === null || ref === undefined) {
    return null;
  }
  if (typeof ref === "object") {
    return (
      normalizeWhitespace(ref.label || ref.name || ref.term_name || ref.accession || "") || null
    );
  }
  const s = String(ref).trim();
  if (!s) {
    return null;
  }
  if (s.includes("/")) {
    const tail = s.split("/").filter(Boolean).pop() || "";
    return tail.replace(/\+/g, " ") || null;
  }
  return normalizeWhitespace(s) || null;
}

function buildEncodePortalItemUrl(item) {
  const href = item?.["@id"];
  if (typeof href === "string" && href.startsWith("/")) {
    return `${ENCODE_PORTAL_BASE}${href}`;
  }
  const acc = normalizeWhitespace(item?.accession || "");
  if (acc) {
    return `${ENCODE_PORTAL_BASE}/${encodeURIComponent(acc)}/`;
  }
  return ENCODE_PORTAL_BASE;
}

function summarizeEncodeItem(item) {
  if (!item || typeof item !== "object") {
    return null;
  }
  const types = Array.isArray(item["@type"]) ? item["@type"] : [];
  const primaryType = types.find((t) => t !== "Item") || types[0] || "";
  const accession = normalizeWhitespace(item.accession || "") || null;
  const assayTitle = normalizeWhitespace(item.assay_title || item.assay_term_name || "") || null;
  const targetLabel = describeEncodePortalRef(item.target);
  const biosampleSummary = normalizeWhitespace(item.biosample_summary || "") || null;
  const biosampleOntology = describeEncodePortalRef(item.biosample_ontology);
  let description = normalizeWhitespace(item.description || item.summary || "") || null;
  if (description && description.length > 220) {
    description = `${description.slice(0, 217)}...`;
  }
  const status = normalizeWhitespace(item.status || "") || null;
  const outputType = normalizeWhitespace(item.output_type || "") || null;
  const fileFormat = normalizeWhitespace(item.file_format || "") || null;
  const assayTerm = normalizeWhitespace(item.assay_term_name || "") || null;
  const portalUrl = buildEncodePortalItemUrl(item);
  const line = [
    accession || "—",
    primaryType,
    assayTitle || assayTerm,
    targetLabel ? `target ${targetLabel}` : null,
    biosampleOntology || biosampleSummary,
    fileFormat || outputType,
    status,
  ]
    .filter(Boolean)
    .join(" | ");
  return {
    accession,
    object_type: primaryType || null,
    assay_title: assayTitle || assayTerm,
    target_label: targetLabel,
    biosample_summary: biosampleSummary,
    biosample_ontology: biosampleOntology,
    description,
    status,
    output_type: outputType,
    file_format: fileFormat,
    portal_url: portalUrl,
    summary_line: line,
  };
}

function readEncodeApiError(payload) {
  if (!payload || typeof payload !== "object") {
    return "";
  }
  if (Number(payload._http_status) === 404 && Array.isArray(payload["@graph"])) {
    return "";
  }
  if (payload.status === "error" && payload.description) {
    return normalizeWhitespace(String(payload.description)) || "ENCODE API error";
  }
  if (Number(payload.code) === 404) {
    return normalizeWhitespace(String(payload.description || "Not found")) || "Not found";
  }
  return "";
}

function buildEncodeSearchUrl({
  objectType,
  searchTerm,
  organism,
  assayTitle,
  status,
  limit,
  frame,
}) {
  const params = new URLSearchParams();
  params.set("type", normalizeWhitespace(objectType || "") || "Experiment");
  const term = normalizeWhitespace(searchTerm || "");
  if (term) {
    params.set("searchTerm", term);
  }
  const org = normalizeWhitespace(organism || "");
  if (org) {
    params.set("organism.scientific_name", org);
  }
  const assay = normalizeWhitespace(assayTitle || "");
  if (assay) {
    params.set("assay_title", assay);
  }
  const stat = normalizeWhitespace(status || "");
  if (stat && stat.toLowerCase() !== "any") {
    params.set("status", stat);
  }
  const lim = Math.max(1, Math.min(50, Math.round(limit || 15)));
  params.set("limit", String(lim));
  params.set("frame", normalizeWhitespace(frame || "") || "object");
  return `${ENCODE_PORTAL_BASE}/search/?${params.toString()}`;
}

function buildEncodeRecordFetchUrl(accessionOrPath, frame) {
  let path = String(accessionOrPath || "").trim();
  if (!path) {
    return "";
  }
  if (!path.startsWith("/")) {
    path = `/${path.replace(/^\/+/, "")}`;
  }
  if (!path.endsWith("/")) {
    path = `${path}/`;
  }
  const params = new URLSearchParams();
  const fr = normalizeWhitespace(frame || "") || "object";
  if (fr) {
    params.set("frame", fr);
  }
  return `${ENCODE_PORTAL_BASE}${path}?${params.toString()}`;
}

async function fetchEncodeJsonWithRetry(url, options = {}) {
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
      const rawText = await response.text().catch(() => "");
      let parsed = null;
      if (rawText) {
        try {
          parsed = JSON.parse(rawText);
        } catch {
          parsed = null;
        }
      }

      if (response.ok) {
        return parsed;
      }

      // ENCODE search often returns a 404 HTTP status for "no hits" while still providing a structured JSON body.
      if (response.status === 404 && parsed && typeof parsed === "object") {
        parsed._http_status = response.status;
        return parsed;
      }

      const retryable = response.status === 429 || response.status >= 500;
      lastError = new Error(
        `Request failed (${response.status}): ${url}${
          rawText ? ` | ${rawText.slice(0, 220).replace(/\s+/g, " ").trim()}` : ""
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

function shouldSearchEncodeFunctionalCharacterization({ objectType, searchTerm, assayTitle }) {
  if (normalizeWhitespace(objectType || "") !== "Experiment") {
    return false;
  }
  const haystack = `${normalizeWhitespace(searchTerm || "")} ${normalizeWhitespace(assayTitle || "")}`.toLowerCase();
  return /\bmpra\b|\blentimpra\b|\bstarr\b|massively parallel reporter|reporter assay|crispr screen|functional characterization/.test(haystack);
}

async function runEncodeSearchQuery({ objectType, searchTerm, organism, assayTitle, status, maxResults, frame }) {
  const url = buildEncodeSearchUrl({
    objectType,
    searchTerm,
    organism,
    assayTitle,
    status,
    limit: maxResults,
    frame,
  });
  const payload = await fetchEncodeJsonWithRetry(url, {
    retries: 2,
    timeoutMs: 25000,
    maxBackoffMs: 5000,
    headers: ENCODE_JSON_HEADERS,
  });
  const errText = readEncodeApiError(payload);
  const graph = Array.isArray(payload?.["@graph"]) ? payload["@graph"] : [];
  const total = Number(payload?.total) || graph.length;
  return { objectType, url, payload, errText, graph, total };
}

// ============================================
// TOOL: Search ENCODE metadata
// ============================================
server.registerTool(
  "search_encode_metadata",
  {
    description:
      "Searches the ENCODE Portal metadata index (experiments, files, biosamples, and other object types) via the public REST API. " +
      "Use for ChIP-seq, DNase-seq, RNA-seq, MPRA / functional-characterization assays, assay targets, biosamples, and file accessions. Defaults to released human/mouse experiments when filters are added. Respects ENCODE rate guidance (~10 GET/s).",
    inputSchema: {
      objectType: z.enum([
        "Experiment",
        "FunctionalCharacterizationExperiment",
        "File",
        "Biosample",
        "Dataset",
        "Reference",
        "AntibodyLot",
        "Library",
        "GeneticModification",
        "Treatment",
      ]).optional().describe("ENCODE @type filter for /search (default Experiment)."),
      searchTerm: z.string().optional().describe("Free-text search term (e.g. 'H3K4me3', 'DNase', 'ENCFF123ABC')."),
      organism: z.string().optional().describe("Optional scientific name filter, e.g. 'Homo sapiens' or 'Mus musculus'."),
      assayTitle: z.string().optional().describe("Optional assay_title facet, e.g. 'ChIP-seq', 'DNase-seq'."),
      status: z.string().optional().describe("Metadata status filter (default 'released'). Pass 'any' to omit."),
      maxResults: z.number().optional().describe("Max records (default 15, max 50)."),
      frame: z.enum(["object", "embedded"]).optional().describe("Search result frame; default 'object' (smaller JSON)."),
    },
  },
  async ({ objectType, searchTerm, organism, assayTitle, status, maxResults, frame }) => {
    const requestedTypeFilter = objectType || "Experiment";
    const stat = status !== undefined && status !== null && String(status).trim() !== ""
      ? String(status).trim()
      : "released";

    const focusedTerm = normalizeWhitespace(searchTerm || "");
    const focusedOrg = normalizeWhitespace(organism || "");
    const focusedAssay = normalizeWhitespace(assayTitle || "");
    if (!focusedTerm && !focusedOrg && !focusedAssay) {
      return {
        content: [{
          type: "text",
          text: "Provide at least one of searchTerm, organism, or assayTitle so ENCODE search stays specific (required to avoid scanning the entire ENCODE catalog).",
        }],
      };
    }

    try {
      const candidateObjectTypes = dedupeArray([
        requestedTypeFilter,
        shouldSearchEncodeFunctionalCharacterization({
          objectType: requestedTypeFilter,
          searchTerm,
          assayTitle,
        })
          ? "FunctionalCharacterizationExperiment"
          : null,
      ]);

      const attemptedSearches = [];
      let selectedSearch = null;
      let firstNotFoundSearch = null;
      let firstErrorSearch = null;

      for (const candidateType of candidateObjectTypes) {
        const searchResult = await runEncodeSearchQuery({
          objectType: candidateType,
          searchTerm,
          organism,
          assayTitle,
          status: stat,
          maxResults,
          frame,
        });
        attemptedSearches.push({
          object_type: candidateType,
          search_url: searchResult.url,
          total: searchResult.total,
          records_returned: searchResult.graph.length,
          error: searchResult.errText || null,
        });
        if (searchResult.errText) {
          firstErrorSearch ||= searchResult;
          continue;
        }
        if (searchResult.graph.length === 0) {
          firstNotFoundSearch ||= searchResult;
          continue;
        }
        selectedSearch = searchResult;
        break;
      }

      if (!selectedSearch && firstNotFoundSearch) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No ENCODE ${requestedTypeFilter} results for this query.`,
              keyFields: [
                `Requested search type: ${requestedTypeFilter}`,
                `Resolved search type: ${firstNotFoundSearch.objectType}`,
                `Search URL: ${firstNotFoundSearch.url}`,
                `Total reported: ${firstNotFoundSearch.total}`,
              ],
              sources: [firstNotFoundSearch.url, "https://www.encodeproject.org/help/rest-api/"],
              limitations: [
                "Broaden searchTerm, clear organism/assay filters, or set status to 'any' if you expect unreleased or non-human records.",
              ],
            }),
          }],
          structuredContent: {
            schema: "search_encode_metadata.v1",
            result_status: "not_found_or_empty",
            requested_object_type: requestedTypeFilter,
            object_type: firstNotFoundSearch.objectType,
            total: firstNotFoundSearch.total,
            records: [],
            search_url: firstNotFoundSearch.url,
            attempted_searches: attemptedSearches,
          },
        };
      }

      if (!selectedSearch) {
        const detail = firstErrorSearch?.errText || "unknown ENCODE search error";
        return {
          content: [{ type: "text", text: `Error searching ENCODE: ${detail}` }],
          structuredContent: {
            schema: "search_encode_metadata.v1",
            result_status: "error",
            requested_object_type: requestedTypeFilter,
            object_type: firstErrorSearch?.objectType || requestedTypeFilter,
            error: detail,
            attempted_searches: attemptedSearches,
          },
        };
      }

      const { objectType: resolvedTypeFilter, url, graph, total } = selectedSearch;

      const records = graph
        .map((row) => summarizeEncodeItem(row))
        .filter(Boolean);
      const lines = records.map((rec, idx) => `${String(idx + 1).padStart(3)}. ${rec.summary_line}`);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `ENCODE search: ${total} total match(es); showing ${records.length} ${resolvedTypeFilter} record(s).`,
            keyFields: [
              `Requested search type: ${requestedTypeFilter}`,
              resolvedTypeFilter !== requestedTypeFilter ? `Resolved search type: ${resolvedTypeFilter}` : null,
              "Results:",
              ...lines,
            ].filter(Boolean),
            sources: [
              url,
              ...records.map((r) => r.portal_url).filter(Boolean).slice(0, 6),
            ],
            limitations: [
              "Programmatic limit is 10 GET requests/sec per ENCODE policy.",
              "Use get_encode_record with an accession or @id path for full metadata and file links.",
            ],
          }),
        }],
        structuredContent: {
          schema: "search_encode_metadata.v1",
          result_status: "ok",
          requested_object_type: requestedTypeFilter,
          object_type: resolvedTypeFilter,
          total,
          records_returned: records.length,
          records,
          search_url: url,
          attempted_searches: attemptedSearches,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error searching ENCODE: ${detail}` }],
        structuredContent: {
          schema: "search_encode_metadata.v1",
          result_status: "error",
          object_type: requestedTypeFilter,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: Get ENCODE record
// ============================================
server.registerTool(
  "get_encode_record",
  {
    description:
      "Retrieves one ENCODE metadata object by accession (ENCSR..., ENCFF..., ENCBS..., ...) or API path (e.g. /experiments/ENCSR123/) using GET + JSON. " +
      "Use after search_encode_metadata or when the accession is known. Default frame is 'object'; use 'embedded' for expanded nested links (larger payload).",
    inputSchema: {
      accessionOrPath: z.string().describe("ENCODE accession, or path starting with /experiments/, /files/, /biosamples/, etc."),
      frame: z.enum(["object", "embedded"]).optional().describe("Record frame (default 'object')."),
    },
  },
  async ({ accessionOrPath, frame }) => {
    const raw = String(accessionOrPath || "").trim();
    if (!raw) {
      return { content: [{ type: "text", text: "Provide an ENCODE accession or /type/accession/ path." }] };
    }

    const frameMode = frame || "object";
    try {
      const url = buildEncodeRecordFetchUrl(raw, frameMode);
      if (!url) {
        return { content: [{ type: "text", text: "Invalid ENCODE accession or path." }] };
      }
      const fetched = await fetchEncodeJsonWithRetry(url, {
        retries: 2,
        timeoutMs: frameMode === "embedded" ? 45000 : 20000,
        maxBackoffMs: 5000,
        headers: ENCODE_JSON_HEADERS,
      });
      const errText = readEncodeApiError(fetched);
      if (errText) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `ENCODE record not retrieved: ${errText}`,
              keyFields: [`URL: ${url}`],
              sources: [url, `${ENCODE_PORTAL_BASE}/help/rest-api/`],
              limitations: ["Verify accession spelling; use search_encode_metadata to discover identifiers."],
            }),
          }],
          structuredContent: {
            schema: "get_encode_record.v1",
            result_status: "not_found_or_empty",
            accession_or_path: raw,
            error: errText,
          },
        };
      }

      let item = fetched;
      if (Array.isArray(fetched?.["@graph"]) && fetched["@graph"].length === 1) {
        item = fetched["@graph"][0];
      }

      const summary = summarizeEncodeItem(item);
      const files = Array.isArray(item.files) ? item.files : [];
      const filePreview = files.slice(0, 8).map((f) => {
        if (typeof f === "string") {
          return f;
        }
        return normalizeWhitespace(f?.accession || f?.["@id"] || "") || "";
      }).filter(Boolean);

      const keyFields = [
        summary?.accession ? `Accession: ${summary.accession}` : null,
        summary?.object_type ? `Type: ${summary.object_type}` : null,
        summary?.assay_title ? `Assay: ${summary.assay_title}` : null,
        summary?.target_label ? `Target: ${summary.target_label}` : null,
        summary?.biosample_summary ? `Biosample: ${summary.biosample_summary}` : null,
        summary?.status ? `Status: ${summary.status}` : null,
        files.length ? `Linked files: ${files.length}${filePreview.length ? ` (e.g. ${filePreview.join(", ")})` : ""}` : null,
        summary?.description ? `Description: ${summary.description}` : null,
      ].filter(Boolean);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: summary?.summary_line
              ? `ENCODE ${summary.summary_line}`
              : "ENCODE record retrieved.",
            keyFields: keyFields.length > 0 ? keyFields : [JSON.stringify(item, null, 2).slice(0, 1500)],
            sources: [summary?.portal_url || url, url],
            limitations: [
              "Large file lists and raw metadata may be truncated in chat; open the portal link for full detail and downloads.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_encode_record.v1",
          result_status: "ok",
          accession_or_path: raw,
          frame: frameMode,
          summary,
          file_count: files.length,
          file_sample: filePreview,
          record: frameMode === "object" ? item : null,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error fetching ENCODE record: ${detail}` }],
        structuredContent: {
          schema: "get_encode_record.v1",
          result_status: "error",
          accession_or_path: raw,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// Zenodo REST API (zenodo.org/api)
// ============================================
const ZENODO_API_BASE = "https://zenodo.org/api";

function zenodoRequestHeaders() {
  const headers = {
    Accept: "application/json",
    "User-Agent": "research-mcp/zenodo",
  };
  const token = String(process.env.ZENODO_ACCESS_TOKEN || "").trim();
  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }
  return headers;
}

function zenodoAnonymousMaxPageSize() {
  return String(process.env.ZENODO_ACCESS_TOKEN || "").trim() ? 100 : 25;
}

function stripHtmlSnippet(value, maxLen) {
  const raw = String(value || "").replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  if (!maxLen || raw.length <= maxLen) {
    return raw;
  }
  return `${raw.slice(0, Math.max(0, maxLen - 3)).trim()}...`;
}

function parseZenodoRecordId(raw) {
  const s = String(raw || "").trim();
  if (!s) {
    return null;
  }
  const lower = s.toLowerCase();
  const recordPath = lower.match(/zenodo\.org\/(?:record|records)\/(\d+)/);
  if (recordPath) {
    return recordPath[1];
  }
  const doiMatch = s.match(/10\.5281\/zenodo\.(\d+)/i);
  if (doiMatch) {
    return doiMatch[1];
  }
  if (/^\d+$/.test(s)) {
    return s;
  }
  return null;
}

function summarizeZenodoHit(hit) {
  if (!hit || typeof hit !== "object") {
    return null;
  }
  const md = hit.metadata && typeof hit.metadata === "object" ? hit.metadata : {};
  const title = normalizeWhitespace(md.title || "") || null;
  const pubDate = normalizeWhitespace(md.publication_date || "") || null;
  const access = normalizeWhitespace(md.access_right || "") || null;
  const rtype = md.resource_type && typeof md.resource_type === "object"
    ? normalizeWhitespace(md.resource_type.title || md.resource_type.type || "")
    : null;
  const creators = Array.isArray(md.creators) ? md.creators : [];
  const creatorNames = creators
    .map((c) => normalizeWhitespace(c?.name || ""))
    .filter(Boolean)
    .slice(0, 4);
  const kw = Array.isArray(md.keywords) ? md.keywords.filter(Boolean).slice(0, 6) : [];
  const concept = hit.conceptrecid !== undefined && hit.conceptrecid !== null
    ? String(hit.conceptrecid)
    : null;
  const id = hit.id !== undefined && hit.id !== null ? String(hit.id) : null;
  const doi = normalizeWhitespace(hit.doi || md.doi || "") || null;
  const doiUrl = normalizeWhitespace(hit.doi_url || "") || (doi ? `https://doi.org/${encodeURIComponent(doi)}` : null);
  const htmlLink = typeof hit.links?.html === "string" ? hit.links.html : null;
  const apiSelf = typeof hit.links?.self === "string" ? hit.links.self : null;
  const portal = htmlLink || (id ? `https://zenodo.org/records/${encodeURIComponent(id)}` : "https://zenodo.org/");
  const desc = stripHtmlSnippet(md.description || "", 220);
  const line = [
    id ? `id ${id}` : null,
    concept ? `concept ${concept}` : null,
    title,
    rtype ? `[${rtype}]` : null,
    pubDate,
    access,
    creatorNames.length ? creatorNames.join("; ") : null,
  ]
    .filter(Boolean)
    .join(" | ");
  return {
    id,
    conceptrecid: concept,
    title,
    publication_date: pubDate,
    resource_type: rtype,
    access_right: access,
    creators: creatorNames,
    keywords: kw,
    description: desc || null,
    doi,
    doi_url: doiUrl,
    portal_url: portal,
    api_url: apiSelf,
    summary_line: line,
  };
}

function buildZenodoSearchUrl({
  searchQuery,
  sort,
  size,
  recordType,
  community,
  status,
  allVersions,
  page,
}) {
  const params = new URLSearchParams();
  const q = normalizeWhitespace(searchQuery || "");
  if (q) {
    params.set("q", q);
  }
  const s = normalizeWhitespace(sort || "") || "bestmatch";
  params.set("sort", s);
  const lim = Math.max(1, Math.min(zenodoAnonymousMaxPageSize(), Math.round(size || 10)));
  params.set("size", String(lim));
  const p = Math.max(1, Math.round(page || 1));
  if (p > 1) {
    params.set("page", String(p));
  }
  const rt = normalizeWhitespace(recordType || "");
  if (rt) {
    params.set("type", rt.toLowerCase());
  }
  const comm = normalizeWhitespace(community || "");
  if (comm) {
    params.set("communities", comm);
  }
  const st = normalizeWhitespace(status || "");
  if (st) {
    params.set("status", st);
  }
  if (allVersions === true || allVersions === 1 || String(allVersions).toLowerCase() === "true") {
    params.set("all_versions", "1");
  }
  return `${ZENODO_API_BASE}/records?${params.toString()}`;
}

// ============================================
// TOOL: Search Zenodo records
// ============================================
server.registerTool(
  "search_zenodo_records",
  {
    description:
      "Searches published Zenodo records (datasets, software, publications, and other uploads) via GET /api/records. " +
      "Supports Elasticsearch-style query strings in searchQuery. Anonymous requests are capped at 25 hits per page (~30 requests/min per Zenodo policy); set ZENODO_ACCESS_TOKEN for higher page sizes.",
    inputSchema: {
      searchQuery: z.string().describe("Search string (e.g. 'single cell RNA', 'fMRI', 'resource_type.type:dataset')."),
      maxResults: z.number().optional().describe("Page size (default 10; max 25 without token, 100 with ZENODO_ACCESS_TOKEN)."),
      sort: z.enum(["bestmatch", "mostrecent", "newest", "-mostrecent"]).optional().describe("Sort order (default bestmatch). Use mostrecent or -mostrecent per Zenodo API."),
      recordType: z.string().optional().describe("Zenodo type filter, e.g. 'dataset', 'software', 'publication' (passed as the API type= parameter)."),
      community: z.string().optional().describe("Zenodo community identifier to restrict results."),
      status: z.string().optional().describe("Deposit status filter, typically 'published'."),
      allVersions: z.boolean().optional().describe("If true, include all versions of a record."),
      page: z.number().optional().describe("Page number (default 1)."),
    },
  },
  async ({ searchQuery, maxResults, sort, recordType, community, status, allVersions, page }) => {
    const q = normalizeWhitespace(searchQuery || "");
    if (!q) {
      return { content: [{ type: "text", text: "Provide a Zenodo searchQuery." }] };
    }

    const sortValue = (() => {
      const s = String(sort || "bestmatch").trim();
      if (s === "newest") {
        return "mostrecent";
      }
      return s;
    })();

    try {
      const url = buildZenodoSearchUrl({
        searchQuery: q,
        sort: sortValue,
        size: maxResults,
        recordType,
        community,
        status,
        allVersions,
        page,
      });
      const payload = await fetchJsonWithRetry(url, {
        retries: 2,
        timeoutMs: 25000,
        maxBackoffMs: 5000,
        headers: zenodoRequestHeaders(),
      });
      const total = Number(payload?.hits?.total ?? 0);
      const hits = Array.isArray(payload?.hits?.hits) ? payload.hits.hits : [];
      if (hits.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No Zenodo records matched "${q}".`,
              keyFields: [`Request: ${url}`, `Reported total (index): ${total}`],
              sources: [url, "https://developers.zenodo.org/", "https://help.zenodo.org/guides/search/"],
              limitations: ["Try broader terms, a different type= filter, or fielded search syntax from the Zenodo search guide."],
            }),
          }],
          structuredContent: {
            schema: "search_zenodo_records.v1",
            result_status: "not_found_or_empty",
            search_query: q,
            total,
            records: [],
            request_url: url,
          },
        };
      }

      const records = hits.map((h) => summarizeZenodoHit(h)).filter(Boolean);
      const lines = records.map((rec, idx) => `${String(idx + 1).padStart(3)}. ${rec.summary_line}`);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Zenodo: ${total} total hit(s); showing ${records.length} on this page for "${q}".`,
            keyFields: ["Results:", ...lines],
            sources: [
              url,
              ...records.map((r) => r.portal_url).filter(Boolean).slice(0, 6),
            ],
            limitations: [
              "Zenodo limits anonymous API page size to 25 and throttles ~30 GETs/min; configure ZENODO_ACCESS_TOKEN if you need larger pages.",
              "Use get_zenodo_record with a numeric id, DOI (10.5281/zenodo.N), or Zenodo record URL for full metadata and file links.",
            ],
          }),
        }],
        structuredContent: {
          schema: "search_zenodo_records.v1",
          result_status: "ok",
          search_query: q,
          total,
          records_returned: records.length,
          records,
          request_url: url,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error searching Zenodo: ${detail}` }],
        structuredContent: {
          schema: "search_zenodo_records.v1",
          result_status: "error",
          search_query: q,
          error: detail,
        },
      };
    }
  }
);

// ============================================
// TOOL: Get Zenodo record
// ============================================
server.registerTool(
  "get_zenodo_record",
  {
    description:
      "Retrieves one published Zenodo record via GET /api/records/:id. Accepts numeric record id, DOI (10.5281/zenodo.N), or a zenodo.org record URL. " +
      "Returns title, type, dates, access rights, creators, DOI, and a compact file list with download links when present.",
    inputSchema: {
      recordId: z.string().describe("Numeric Zenodo id, DOI 10.5281/zenodo.N, or URL containing /records/N."),
    },
  },
  async ({ recordId }) => {
    const resolved = parseZenodoRecordId(recordId);
    if (!resolved) {
      return {
        content: [{
          type: "text",
          text: "Provide a Zenodo numeric id, a 10.5281/zenodo.N DOI, or a https://zenodo.org/records/N URL.",
        }],
      };
    }

    try {
      const url = `${ZENODO_API_BASE}/records/${encodeURIComponent(resolved)}`;
      const record = await fetchJsonWithRetry(url, {
        retries: 2,
        timeoutMs: 25000,
        maxBackoffMs: 5000,
        headers: zenodoRequestHeaders(),
      });
      if (record?.status === "error" || record?.message) {
        const msg = normalizeWhitespace(record?.message || record?.description || "Zenodo error") || "Zenodo error";
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Zenodo record error: ${msg}`,
              keyFields: [`URL: ${url}`],
              sources: [url, "https://developers.zenodo.org/"],
              limitations: ["Verify the id or DOI refers to a published record."],
            }),
          }],
          structuredContent: {
            schema: "get_zenodo_record.v1",
            result_status: "error",
            record_id: resolved,
            error: msg,
          },
        };
      }

      const summary = summarizeZenodoHit(record);
      const files = Array.isArray(record.files) ? record.files : [];
      const fileRows = files.slice(0, 12).map((f) => ({
        key: normalizeWhitespace(f?.key || "") || null,
        size: f?.size !== undefined ? Number(f.size) : null,
        type: normalizeWhitespace(f?.type || "") || null,
        link: typeof f?.links?.self === "string" ? f.links.self : null,
      }));

      const keyFields = [
        summary?.title ? `Title: ${summary.title}` : null,
        summary?.id ? `Id: ${summary.id} (concept ${summary.conceptrecid || "—"})` : null,
        summary?.resource_type ? `Type: ${summary.resource_type}` : null,
        summary?.publication_date ? `Date: ${summary.publication_date}` : null,
        summary?.access_right ? `Access: ${summary.access_right}` : null,
        summary?.doi ? `DOI: ${summary.doi}` : null,
        summary?.creators?.length ? `Creators: ${summary.creators.join("; ")}` : null,
        summary?.description ? `About: ${summary.description}` : null,
        files.length ? `Files: ${files.length}` : null,
        ...fileRows
          .filter((r) => r.key)
          .map((r) => `  - ${r.key}${r.size ? ` (${r.size} B)` : ""}`),
      ].filter(Boolean);

      let recordForStruct = record;
      try {
        if (JSON.stringify(recordForStruct).length > 100000) {
          recordForStruct = {
            id: record.id,
            conceptrecid: record.conceptrecid,
            doi: record.doi,
            links: record.links,
            metadata: record.metadata,
            files: fileRows,
            _note: "Full record JSON exceeded size cap in structured output.",
          };
        }
      } catch {
        recordForStruct = { id: record.id, conceptrecid: record.conceptrecid };
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: summary?.summary_line ? `Zenodo record: ${summary.summary_line}` : `Zenodo record ${resolved}`,
            keyFields: keyFields.length > 0 ? keyFields : [JSON.stringify(record, null, 2).slice(0, 1200)],
            sources: [summary?.portal_url || url, url],
            limitations: [
              "Binary attachments and very large metadata should be downloaded from Zenodo directly, not via chat transcripts.",
            ],
          }),
        }],
        structuredContent: {
          schema: "get_zenodo_record.v1",
          result_status: "ok",
          record_id: resolved,
          summary,
          file_count: files.length,
          files: fileRows,
          record: recordForStruct,
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error fetching Zenodo record: ${detail}` }],
        structuredContent: {
          schema: "get_zenodo_record.v1",
          result_status: "error",
          record_id: resolved,
          error: detail,
        },
      };
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

      const searchUrl = buildNcbiEutilsUrl(PUBMED_ESEARCH, params);
      const searchData = await fetchJsonWithRetry(searchUrl, { retries: 2, timeoutMs: 10000 });
      const idList = searchData?.esearchresult?.idlist ?? [];
      const totalCount = parseInt(searchData?.esearchresult?.count || "0", 10);

      if (idList.length === 0) {
        return {
          content: [{ type: "text", text: `No PubMed results for advanced query: ${query}. Check field tags and try broader terms.` }],
        };
      }

      const summaryXml = await fetchPubmedSummaryXml(idList);
      const articles = parsePubmedArticleSummary(summaryXml);
      if (articles.length === 0 && totalCount > 0) {
        throw new Error(`PubMed summary parse returned zero article records despite ${totalCount} hit(s).`);
      }

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
          text: `PubMed advanced search:\nShowing ${articles.length} of ${totalCount} results\nQuery: ${query}\n\n${formatted}\n\nUse get_pubmed_abstract with a PMID for the abstract, or get_paper_fulltext for PMC full text when available.`,
        }],
      };
    } catch (error) {
      return {
        content: [{ type: "text", text: `Error in advanced PubMed search: ${formatPubmedToolError(error)}` }],
      };
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
// TOOL: MyGene.info gene identifier resolution
// ============================================
server.registerTool(
  "resolve_gene_identifiers",
  {
    description:
      "Resolves a gene symbol, alias, or identifier to canonical gene IDs and aliases using MyGene.info. " +
      "Use for identifier normalization before querying other sources.",
    inputSchema: {
      query: z.string().describe("Gene symbol, alias, Entrez ID, or Ensembl gene ID (e.g. 'TP53', 'P53', 'ENSG00000141510')."),
      species: z.string().optional().default("human").describe("Species name or taxon restriction (default: human)."),
      limit: z.number().optional().default(5).describe("Maximum hits to return (1-10)."),
    },
  },
  async ({ query, species = "human", limit = 5 }) => {
    const boundedLimit = Math.max(1, Math.min(10, Math.round(limit || 5)));
    const params = new URLSearchParams({
      q: query,
      species,
      size: String(boundedLimit),
      fields: "symbol,name,alias,entrezgene,ensembl.gene,uniprot.Swiss-Prot,uniprot.TrEMBL,taxid",
    });
    const url = `${MYGENE_API}/query?${params.toString()}`;

    try {
      const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 12000, maxBackoffMs: 2500 });
      const hits = Array.isArray(data?.hits) ? data.hits.slice(0, boundedLimit) : [];
      if (hits.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No MyGene.info hits found for "${query}".`,
              keyFields: [`Species: ${species}`],
              sources: [url],
              limitations: ["Try a canonical HGNC gene symbol, Entrez Gene ID, or Ensembl gene ID."],
            }),
          }],
        };
      }

      const keyFields = hits.map((hit, idx) => {
        const ids = normalizeMyGeneIds(hit);
        const aliasText = ids.aliases.length > 0 ? ids.aliases.slice(0, 6).join(", ") : "none";
        const ensemblText = ids.ensemblGenes.length > 0 ? ids.ensemblGenes.join(", ") : "none";
        const swissProtText = ids.swissProtIds.length > 0 ? ids.swissProtIds.join(", ") : "none";
        return (
          `${idx + 1}. ${ids.symbol || "Unknown"} — ${ids.name || "No name"} ` +
          `| Taxon: ${ids.taxid || "unknown"} ` +
          `| Entrez: ${ids.entrezgene || "none"} ` +
          `| Ensembl: ${ensemblText} ` +
          `| UniProt: ${swissProtText} ` +
          `| Aliases: ${aliasText}`
        );
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `MyGene.info returned ${hits.length} candidate gene hit(s) for "${query}".`,
            keyFields,
            sources: [url],
            limitations: ["Alias-heavy queries can return multiple plausible genes; verify the top hit before using downstream tools."],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in resolve_gene_identifiers: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: OxO ontology cross-reference mapping
// ============================================
server.registerTool(
  "map_ontology_terms_oxo",
  {
    description:
      "Maps ontology terms across identifier systems using EBI OxO. " +
      "Use for MONDO/EFO/DOID/MeSH/OMIM/UMLS crosswalks before joining evidence across databases.",
    inputSchema: {
      ids: z.array(z.string()).describe("One or more CURIEs (e.g. ['MONDO:0005148', 'EFO:0003767'])."),
      targetPrefixes: z.array(z.string()).optional().describe("Optional target prefixes to constrain mappings (e.g. ['EFO', 'MeSH', 'DOID'])."),
      distance: z.number().optional().default(1).describe("Maximum graph distance in OxO (1-3 recommended)."),
      mappingsPerId: z.number().optional().default(10).describe("Maximum mappings to return per input ID (1-25)."),
    },
  },
  async ({ ids, targetPrefixes = [], distance = 1, mappingsPerId = 10 }) => {
    const cleanedIds = dedupeArray((Array.isArray(ids) ? ids : []).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 10);
    const cleanedPrefixes = dedupeArray((Array.isArray(targetPrefixes) ? targetPrefixes : []).map((value) => normalizeOntologyPrefix(value)).filter(Boolean)).slice(0, 8);
    const boundedDistance = Math.max(1, Math.min(3, Math.round(distance || 1)));
    const boundedMappings = Math.max(1, Math.min(25, Math.round(mappingsPerId || 10)));
    if (cleanedIds.length === 0) {
      return { content: [{ type: "text", text: "Provide at least one ontology CURIE in `ids`." }] };
    }

    const params = new URLSearchParams({
      distance: String(boundedDistance),
      size: String(cleanedIds.length),
    });
    for (const id of cleanedIds) params.append("ids", id);
    for (const prefix of cleanedPrefixes) params.append("targetPrefix", prefix);
    const url = `${OXO_API}/search?${params.toString()}`;

    try {
      const data = await fetchJsonWithRetry(url, { retries: 0, timeoutMs: 9000, maxBackoffMs: 1500 });
      const results = Array.isArray(data?._embedded?.searchResults) ? data._embedded.searchResults : [];
      if (results.length === 0) {
        const fallback = await buildOlsFallbackMappings(cleanedIds, cleanedPrefixes, boundedMappings);
        if (fallback.recoveredTerms > 0) {
          return {
            content: [{
              type: "text",
              text: renderStructuredResponse({
                summary: `OxO returned no mappings; OLS fallback recovered results for ${fallback.recoveredTerms} ontology term(s).`,
                keyFields: fallback.keyFields,
                sources: [url, ...fallback.sources],
                limitations: fallback.limitations,
              }),
            }],
          };
        }
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `OxO returned no ontology mappings for ${cleanedIds.join(", ")}.`,
              keyFields: [
                `Distance: ${boundedDistance}`,
                cleanedPrefixes.length > 0 ? `Target prefixes: ${cleanedPrefixes.join(", ")}` : "Target prefixes: all",
              ],
              sources: [url],
              limitations: ["Some ontology pairs are not directly cross-referenced in OxO; try a larger distance or a different source ontology."],
            }),
          }],
        };
      }

      const keyFields = results.map((result, idx) => {
        const label = normalizeWhitespace(result?.label || "");
        const mappings = Array.isArray(result?.mappingResponseList) ? result.mappingResponseList : [];
        const mappingText = mappings
          .slice(0, boundedMappings)
          .map((row) => {
            const curie = normalizeOntologyCurie(row?.curie || "");
            const mappedLabel = normalizeWhitespace(row?.label || "");
            const prefix = normalizeOntologyPrefix(row?.targetPrefix || extractCuriePrefix(curie));
            const hopRaw = row?.distance;
            const hopText = Number.isFinite(Number(hopRaw))
              ? ` d=${toNonNegativeInt(hopRaw, boundedDistance)}`
              : normalizeWhitespace(hopRaw || "")
                ? ` d=${normalizeWhitespace(hopRaw)}`
                : "";
            return `${curie}${mappedLabel ? ` (${mappedLabel})` : ""}${prefix ? ` [${prefix}]` : ""}${hopText}`;
          })
          .join("; ") || "No mappings";
        return `${idx + 1}. ${normalizeOntologyCurie(result?.curie || "Unknown")}${label ? ` (${label})` : ""} | ${mappingText}`;
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OxO returned mappings for ${results.length} ontology term(s).`,
            keyFields,
            sources: [url],
            limitations: ["OxO provides cross-references, not semantic equivalence guarantees; review mappings before using them in automated joins."],
          }),
        }],
      };
    } catch (error) {
      const fallback = await buildOlsFallbackMappings(cleanedIds, cleanedPrefixes, boundedMappings, error.message);
      if (fallback.recoveredTerms > 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `OxO service unavailable; OLS fallback recovered mappings for ${fallback.recoveredTerms} ontology term(s).`,
              keyFields: fallback.keyFields,
              sources: [url, ...fallback.sources],
              limitations: fallback.limitations,
            }),
          }],
        };
      }
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OxO service unavailable and no OLS fallback mappings were recovered for ${cleanedIds.join(", ")}.`,
            keyFields: [
              `Distance: ${boundedDistance}`,
              cleanedPrefixes.length > 0 ? `Target prefixes: ${cleanedPrefixes.join(", ")}` : "Target prefixes: all",
            ],
            sources: [url, ...fallback.sources],
            limitations: fallback.limitations,
          }),
        }],
      };
    }
  }
);

// ============================================
// TOOL: HPO term search
// ============================================
server.registerTool(
  "search_hpo_terms",
  {
    description:
      "Search Human Phenotype Ontology (HPO) terms via EBI OLS. " +
      "Use for phenotype-term normalization before rare-disease or phenotype-driven association queries.",
    inputSchema: {
      query: z.string().describe("HPO term text or CURIE (for example 'ataxia', 'microcephaly', or 'HP:0001250')."),
      exact: z.boolean().optional().default(false).describe("If true, require exact matching in OLS."),
      limit: z.number().optional().default(8).describe("Maximum HPO terms to return (1-15)."),
    },
  },
  async ({ query, exact = false, limit = 8 }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    const boundedLimit = Math.max(1, Math.min(15, Math.round(limit || 8)));
    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide an HPO term query or CURIE (for example ataxia or HP:0001250)." }] };
    }

    try {
      const search = await searchOlsTerms(normalizedQuery, { ontology: "hp", exact, rows: boundedLimit });
      const docs = (search?.docs || []).slice(0, boundedLimit);
      if (docs.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No HPO terms matched "${normalizedQuery}".`,
              keyFields: [`Exact match: ${exact ? "yes" : "no"}`],
              sources: [search?.url || `${OLS_API}/search?ontology=hp&q=${encodeURIComponent(normalizedQuery)}`],
              limitations: ["Try a broader phenotype phrase or search without exact matching."],
            }),
          }],
        };
      }

      const keyFields = docs.map((doc, idx) => {
        const curie = normalizeOntologyCurie(doc?.obo_id || "");
        const label = normalizeWhitespace(doc?.label || curie || "Unnamed term");
        const synonyms = dedupeArray(asArray(doc?.exact_synonyms).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 3);
        const definition = asArray(doc?.description).map((value) => normalizeWhitespace(value)).filter(Boolean)[0] || "";
        const parts = [`${idx + 1}. ${label} | ${curie || "no CURIE"}`];
        if (synonyms.length > 0) parts.push(`Synonyms: ${synonyms.join(", ")}`);
        if (definition) parts.push(`Definition: ${compactErrorMessage(definition, 220)}`);
        return parts.join(" | ");
      });

      const sources = [
        search.url,
        ...docs
          .map((doc) => normalizeOntologyCurie(doc?.obo_id || ""))
          .filter(Boolean)
          .map((curie) => `https://hpo.jax.org/browse/term/${encodeURIComponent(curie)}`),
      ];

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `HPO search returned ${docs.length} term(s) for "${normalizedQuery}".`,
            keyFields,
            sources: dedupeArray(sources).slice(0, 8),
            limitations: [
              "HPO term search normalizes phenotype concepts; disease interpretation still needs a disease or gene evidence source.",
              "OLS search ranking may include parent or closely related phenotype terms for broad queries.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_hpo_terms: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Orphanet rare-disease profile
// ============================================
server.registerTool(
  "get_orphanet_disease_profile",
  {
    description:
      "Retrieve an Orphanet / ORDO rare-disease profile by name or OrphaCode, " +
      "including cross-references, onset/inheritance metadata, phenotype associations, and curated disease-gene links.",
    inputSchema: {
      query: z.string().describe("Rare-disease name, OrphaCode, or CURIE (for example 'Rett syndrome', '778', or 'Orphanet:778')."),
      includePhenotypes: z.boolean().optional().default(true).describe("If true, include top HPO phenotype associations from Orphadata."),
      includeGeneAssociations: z.boolean().optional().default(true).describe("If true, include curated disease-gene associations from Orphadata."),
      phenotypeLimit: z.number().optional().default(10).describe("Maximum phenotype associations to summarize (1-15)."),
      geneLimit: z.number().optional().default(8).describe("Maximum disease-gene associations to summarize (1-12)."),
    },
  },
  async ({ query, includePhenotypes = true, includeGeneAssociations = true, phenotypeLimit = 10, geneLimit = 8 }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    const boundedPhenotypeLimit = Math.max(1, Math.min(15, Math.round(phenotypeLimit || 10)));
    const boundedGeneLimit = Math.max(1, Math.min(12, Math.round(geneLimit || 8)));
    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a rare-disease name or Orphanet identifier (for example Rett syndrome or Orphanet:778)." }] };
    }

    try {
      const resolution = await resolveOrphanetDisorderSelection(normalizedQuery);
      const selected = resolution?.selected || null;
      const record = resolution?.record || null;
      if (!selected) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `No Orphanet / ORDO disease profile matched "${normalizedQuery}".`,
              keyFields: ["Try a more specific rare-disease name or an explicit Orphanet CURIE such as Orphanet:778."],
              sources: [resolution?.searchUrl || `${OLS_API}/search?ontology=ordo&q=${encodeURIComponent(normalizedQuery)}`],
              limitations: ["Name-based rare-disease lookup can be sensitive to synonyms and historical labels."],
            }),
          }],
        };
      }

      const [phenotypeCatalog, geneCatalog] = await Promise.all([
        includePhenotypes ? fetchOrphanetPhenotypeCatalog() : Promise.resolve({}),
        includeGeneAssociations ? fetchOrphanetGeneCatalog() : Promise.resolve({}),
      ]);

      const phenotypes = includePhenotypes
        ? asArray(phenotypeCatalog?.[selected.orphaCode]).slice(0, boundedPhenotypeLimit)
        : [];
      const genes = includeGeneAssociations
        ? asArray(geneCatalog?.[selected.orphaCode]).slice(0, boundedGeneLimit)
        : [];

      const term = record?.term || null;
      const definition = asArray(term?.description || selected?.definition).map((value) => normalizeWhitespace(value)).filter(Boolean)[0] || "";
      const inheritance = dedupeArray(asArray(term?.annotation?.has_inheritance).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 4);
      const onset = dedupeArray(asArray(term?.annotation?.has_age_of_onset).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 4);
      const prevalence = dedupeArray(asArray(term?.annotation?.present_in).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 4);
      const xrefs = mergeOrphanetExternalReferences(selected?.xrefs, selected?.olsXrefs);
      const keyFields = [
        `Disease: ${selected.name} | Orphanet: ${selected.orphaCode}`,
        definition ? `Definition: ${compactErrorMessage(definition, 320)}` : "",
        selected.disorderType ? `Type: ${selected.disorderType}` : "",
        selected.disorderGroup ? `Group: ${selected.disorderGroup}` : "",
        inheritance.length > 0 ? `Inheritance: ${inheritance.join(", ")}` : "",
        onset.length > 0 ? `Typical onset: ${onset.join(", ")}` : "",
        xrefs.length > 0
          ? `Cross-references: ${xrefs.slice(0, 8).map((row) => `${row.source}:${row.reference}`).join(", ")}`
          : "",
        prevalence.length > 0 ? `Selected prevalence / geography notes: ${prevalence.join(" | ")}` : "",
      ].filter(Boolean);

      if (selected.synonyms?.length > 0) {
        keyFields.push(`Synonyms: ${selected.synonyms.slice(0, 6).join(", ")}`);
      }

      if (phenotypes.length > 0) {
        keyFields.push("Top phenotype associations:");
        keyFields.push(
          ...phenotypes.map((row, idx) =>
            `${idx + 1}. ${row.label} (${row.hpoId})${row.frequency ? ` | Frequency: ${row.frequency}` : ""}${
              row.diagnosticCriteria ? ` | Diagnostic criteria: ${row.diagnosticCriteria}` : ""
            }`
          )
        );
      }

      if (genes.length > 0) {
        keyFields.push("Curated disease-gene associations:");
        keyFields.push(
          ...genes.map((row, idx) => {
            const refBits = [
              row.hgnc ? `HGNC:${row.hgnc}` : "",
              row.ensembl ? `Ensembl:${row.ensembl}` : "",
              row.omim ? `OMIM:${row.omim}` : "",
            ].filter(Boolean);
            return `${idx + 1}. ${row.symbol}${row.name ? ` — ${row.name}` : ""}${
              row.associationType ? ` | ${row.associationType}` : ""
            }${row.status ? ` | Status: ${row.status}` : ""}${row.geneType ? ` | Type: ${row.geneType}` : ""}${
              refBits.length > 0 ? ` | ${refBits.join(", ")}` : ""
            }${row.pmids.length > 0 ? ` | PMIDs: ${row.pmids.join(", ")}` : ""}`;
          })
        );
      }

      const orphaPage = `https://www.orpha.net/en/disease/detail/${encodeURIComponent(selected.orphaCode)}`;
      const sources = [
        resolution?.searchUrl || "",
        record?.termUrl || "",
        orphaPage,
        ORPHADATA_PRODUCT1_XML,
        includePhenotypes ? ORPHADATA_PRODUCT4_XML : "",
        includeGeneAssociations ? ORPHADATA_PRODUCT6_XML : "",
        selected.expertLink || "",
      ].filter(Boolean);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary:
              `Orphanet / ORDO profile for ${selected.name} (Orphanet:${selected.orphaCode})` +
              `${includePhenotypes ? ` with ${phenotypes.length} phenotype association(s)` : ""}` +
              `${includeGeneAssociations ? `${includePhenotypes ? " and" : " with"} ${genes.length} disease-gene association(s)` : ""}.`,
            keyFields,
            sources: dedupeArray(sources).slice(0, 10),
            limitations: [
              "Phenotype and disease-gene sections summarize curated Orphadata snapshots rather than the full narrative pages.",
              "Rare-disease cross-references and epidemiology notes depend on Orphanet / ORDO release cadence and may lag recent literature.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_orphanet_disease_profile: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Monarch phenotype/disease associations
// ============================================
server.registerTool(
  "query_monarch_associations",
  {
    description:
      "Query Monarch phenotype- and rare-disease-centric associations. " +
      "Supports disease->phenotype, phenotype->gene, disease->gene, and gene->phenotype modes.",
    inputSchema: {
      query: z.string().optional().describe("Free-text entity query or CURIE (for example 'ataxia', 'Rett syndrome', 'TP53', 'HP:0001251')."),
      entityId: z.string().optional().describe("Optional explicit Monarch entity ID/CURIE to skip search resolution."),
      associationMode: z.enum([
        "disease_to_phenotype",
        "phenotype_to_gene",
        "disease_to_gene_causal",
        "disease_to_gene_correlated",
        "gene_to_phenotype",
      ]).describe("Association mode to query in Monarch."),
      limit: z.number().optional().default(10).describe("Maximum associations to return (1-20)."),
      humanOnly: z.boolean().optional().default(true).describe("If true, keep only human gene associations when a gene node is involved."),
    },
  },
  async ({ query = "", entityId = "", associationMode, limit = 10, humanOnly = true }) => {
    const normalizedTarget = normalizeWhitespace(entityId || query || "");
    const modeConfig = MONARCH_ASSOCIATION_MODES[associationMode];
    const boundedLimit = Math.max(1, Math.min(20, Math.round(limit || 10)));
    if (!normalizedTarget) {
      return { content: [{ type: "text", text: "Provide a query or entityId for Monarch (for example ataxia, Rett syndrome, or HP:0001251)." }] };
    }

    try {
      const resolution = await resolveMonarchEntity(normalizedTarget, associationMode, { humanOnly });
      const selected = resolution?.selected || null;
      if (!selected?.id) {
        const candidates = asArray(resolution?.candidates)
          .slice(0, 5)
          .map((item) => `${normalizeWhitespace(item?.name || item?.label || "Unknown")} (${normalizeOntologyCurie(item?.id || "") || "no id"}) | ${normalizeWhitespace(item?.category || "unknown category")}`);
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Monarch could not resolve "${normalizedTarget}" for association mode ${associationMode}.`,
              keyFields: candidates.length > 0 ? candidates : ["No candidate Monarch entities were returned."],
              sources: [resolution?.searchUrl || `${MONARCH_API}/search?q=${encodeURIComponent(normalizedTarget)}`],
              limitations: ["Try an explicit CURIE such as HP:0001251, MONDO:0010726, or HGNC:11998 if free-text resolution is ambiguous."],
            }),
          }],
        };
      }

      const associationPayload = await fetchMonarchAssociations(selected.id, associationMode, boundedLimit);
      const rows = asArray(associationPayload?.items)
        .filter((row) => keepMonarchAssociationForHuman(row, modeConfig, humanOnly))
        .slice(0, boundedLimit);

      const resolvedLabel = normalizeWhitespace(selected?.name || selected?.label || selected?.id || normalizedTarget);
      const resolvedCategory = normalizeWhitespace(selected?.category || "unknown category");
      if (rows.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Monarch resolved ${resolvedLabel} (${normalizeOntologyCurie(selected.id)}) but returned no associations for mode ${associationMode}.`,
              keyFields: [
                `Resolved entity: ${resolvedLabel}`,
                `Category: ${resolvedCategory}`,
                `Mode: ${associationMode} — ${modeConfig?.description || ""}`.trim(),
              ],
              sources: [resolution?.searchUrl || "", resolution?.entityUrl || "", associationPayload?.url || ""].filter(Boolean),
              limitations: [
                "Association coverage varies by upstream knowledge source and mode.",
                humanOnly ? "Human-only filtering can remove model-organism associations from the result set." : "No human-only filtering applied.",
              ],
            }),
          }],
        };
      }

      const keyFields = [
        `Resolved entity: ${resolvedLabel} | ${normalizeOntologyCurie(selected.id)} | ${resolvedCategory}`,
        `Mode: ${associationMode} — ${modeConfig?.description || ""}`.trim(),
        `Associations shown: ${rows.length}${associationPayload?.total ? ` of ${associationPayload.total} reported` : ""}`,
      ];

      const candidateNotes = asArray(resolution?.candidates)
        .slice(0, 3)
        .map((item) => `${normalizeWhitespace(item?.name || item?.label || "Unknown")} (${normalizeOntologyCurie(item?.id || "")})`);
      if (candidateNotes.length > 1) {
        keyFields.push(`Closest Monarch matches considered: ${candidateNotes.join(" | ")}`);
      }

      keyFields.push(
        ...rows.map((row, idx) => {
          const counterpart = getMonarchAssociationCounterpart(row, modeConfig);
          const predicate = normalizeWhitespace(row?.predicate || "association");
          const sourceBits = dedupeArray([
            normalizeWhitespace(row?.primary_knowledge_source || ""),
            ...asArray(row?.aggregator_knowledge_source).map((value) => normalizeWhitespace(value)).filter(Boolean),
            normalizeWhitespace(row?.provided_by || ""),
          ]).slice(0, 3);
          const evidenceCount = toNonNegativeInt(row?.evidence_count, 0);
          const publicationCount = Math.max(asArray(row?.publications).length, asArray(row?.publications_links).length);
          const qualifierBits = [
            normalizeWhitespace(row?.frequency_qualifier_label || ""),
            normalizeWhitespace(row?.onset_qualifier_label || ""),
            normalizeWhitespace(row?.sex_qualifier_label || ""),
            normalizeWhitespace(row?.stage_qualifier_label || ""),
          ].filter(Boolean);
          return `${idx + 1}. ${counterpart.label || counterpart.id || "Unnamed node"}${counterpart.id ? ` (${counterpart.id})` : ""}${
            counterpart.taxonLabel ? ` | Taxon: ${counterpart.taxonLabel}` : ""
          } | Predicate: ${predicate}${sourceBits.length > 0 ? ` | Source: ${sourceBits.join(", ")}` : ""}${
            evidenceCount > 0 ? ` | Evidence count: ${evidenceCount}` : ""
          }${publicationCount > 0 ? ` | Publications: ${publicationCount}` : ""}${
            qualifierBits.length > 0 ? ` | Qualifiers: ${qualifierBits.join("; ")}` : ""
          }`;
        })
      );

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Monarch returned ${rows.length} ${associationMode.replaceAll("_", " ")} association(s) for ${resolvedLabel}.`,
            keyFields,
            sources: [resolution?.searchUrl || "", resolution?.entityUrl || "", associationPayload?.url || ""].filter(Boolean),
            limitations: [
              "Monarch aggregates heterogeneous upstream sources; evidence counts and publication fields vary by provider.",
              humanOnly ? "Human-only filtering was applied when gene nodes were present." : "Model-organism and cross-species associations were retained.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in query_monarch_associations: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: QuickGO term search
// ============================================
server.registerTool(
  "search_quickgo_terms",
  {
    description:
      "Searches Gene Ontology (GO) terms via QuickGO. " +
      "Use for ontology lookup before running GO annotation queries or when grounding a process/function/component concept.",
    inputSchema: {
      query: z.string().describe("GO term text query (e.g. 'apoptosis', 'mitochondrial membrane')."),
      aspect: z.string().optional().describe("Optional aspect: biological_process, molecular_function, or cellular_component."),
      limit: z.number().optional().default(10).describe("Maximum terms to return (1-25)."),
    },
  },
  async ({ query, aspect, limit = 10 }) => {
    const boundedLimit = Math.max(1, Math.min(25, Math.round(limit || 10)));
    const normalizedAspect = normalizeQuickGoAspect(aspect);
    const params = new URLSearchParams({
      query,
      limit: String(boundedLimit),
    });
    if (normalizedAspect) params.set("aspect", normalizedAspect);
    const url = `${QUICKGO_API}/ontology/go/search?${params.toString()}`;

    try {
      const data = await fetchJsonWithRetry(url, {
        headers: { Accept: "application/json" },
        retries: 1,
        timeoutMs: 15000,
        maxBackoffMs: 2500,
      });
      const results = Array.isArray(data?.results) ? data.results : [];
      if (results.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `QuickGO found no GO terms for "${query}".`,
              keyFields: [normalizedAspect ? `Aspect: ${normalizedAspect}` : "Aspect: all"],
              sources: [url],
              limitations: ["Try broader wording or remove the aspect filter."],
            }),
          }],
        };
      }

      const keyFields = results.map((row, idx) => {
        const definition = compactErrorMessage(row?.definition?.text || "", 150);
        return `${idx + 1}. ${row?.id || "GO:?"} — ${row?.name || "Unnamed"} | Aspect: ${row?.aspect || "unknown"}${definition ? ` | ${definition}` : ""}`;
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `QuickGO returned ${results.length} GO term(s) for "${query}".`,
            keyFields,
            sources: [url],
            limitations: ["Ontology search ranks lexical matches; nearby child/parent terms may still be more appropriate than the top result."],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_quickgo_terms: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: QuickGO annotations by gene product
// ============================================
server.registerTool(
  "get_quickgo_annotations",
  {
    description:
      "Fetches GO annotations for a gene product via QuickGO. " +
      "Accepts a QuickGO geneProductId (e.g. UniProtKB:P04637) or a gene symbol resolved via MyGene.info.",
    inputSchema: {
      geneProductId: z.string().optional().describe("QuickGO gene product identifier, usually UniProtKB:ACCESSION."),
      gene: z.string().optional().describe("Gene symbol or alias to resolve via MyGene.info (e.g. 'TP53')."),
      species: z.string().optional().default("human").describe("Species restriction used when resolving `gene`."),
      aspect: z.string().optional().describe("Optional aspect: biological_process, molecular_function, or cellular_component."),
      limit: z.number().optional().default(20).describe("Maximum annotations to return (1-50)."),
    },
  },
  async ({ geneProductId = "", gene = "", species = "human", aspect, limit = 20 }) => {
    const normalizedAspect = normalizeQuickGoAspect(aspect);
    const boundedLimit = Math.max(1, Math.min(50, Math.round(limit || 20)));
    let geneProductIds = [];
    let resolutionNote = "";

    try {
      if (geneProductId) {
        geneProductIds = [normalizeQuickGoGeneProductId(geneProductId)].filter(Boolean);
      } else if (gene) {
        const resolved = await resolveGeneWithMyGene(gene, species);
        const ids = normalizeMyGeneIds(resolved.bestHit || {});
        geneProductIds = ids.swissProtIds.slice(0, 3).map((value) => `UniProtKB:${value}`);
        if (!resolved.bestHit || geneProductIds.length === 0) {
          return {
            content: [{
              type: "text",
              text: renderStructuredResponse({
                summary: `Could not resolve a QuickGO-compatible UniProt identifier for "${gene}".`,
                keyFields: [`Species: ${species}`],
                sources: [`${MYGENE_API}/query?q=${encodeURIComponent(gene)}&species=${encodeURIComponent(species)}`],
                limitations: ["QuickGO gene-product annotations are easiest to access via UniProtKB accessions. Try passing `geneProductId` directly."],
              }),
            }],
          };
        }
        resolutionNote = `Resolved ${gene} to ${geneProductIds.join(", ")} via MyGene.info.`;
      } else {
        return { content: [{ type: "text", text: "Provide either `geneProductId` or `gene`." }] };
      }

      const params = new URLSearchParams({ limit: String(boundedLimit) });
      if (normalizedAspect) params.set("aspect", normalizedAspect);
      for (const id of geneProductIds) params.append("geneProductId", id);
      const url = `${QUICKGO_API}/annotation/search?${params.toString()}`;
      const data = await fetchJsonWithRetry(url, {
        headers: { Accept: "application/json" },
        retries: 1,
        timeoutMs: 18000,
        maxBackoffMs: 2500,
      });
      const results = Array.isArray(data?.results) ? data.results : [];
      if (results.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `QuickGO found no annotations for ${geneProductIds.join(", ")}.`,
              keyFields: [
                resolutionNote || `Gene product ID: ${geneProductIds.join(", ")}`,
                normalizedAspect ? `Aspect: ${normalizedAspect}` : "Aspect: all",
              ],
              sources: [url],
              limitations: ["The gene product may lack reviewed GO annotations in QuickGO for the selected aspect."],
            }),
          }],
        };
      }

      const termMap = await fetchQuickGoTerms(dedupeArray(results.map((row) => normalizeWhitespace(row?.goId || "")).filter(Boolean)));
      const keyFields = results.slice(0, boundedLimit).map((row, idx) => {
        const goId = normalizeWhitespace(row?.goId || "");
        const term = termMap.get(goId) || {};
        const name = normalizeWhitespace(row?.goName || term?.name || "");
        const aspectText = normalizeWhitespace(row?.goAspect || term?.aspect || "");
        const evidence = normalizeWhitespace(row?.evidenceCode || row?.goEvidence || "");
        const reference = normalizeWhitespace(row?.reference || "");
        const qualifier = normalizeWhitespace(row?.qualifier || "");
        return (
          `${idx + 1}. ${goId}${name ? ` — ${name}` : ""}` +
          `${aspectText ? ` | Aspect: ${aspectText}` : ""}` +
          `${evidence ? ` | Evidence: ${evidence}` : ""}` +
          `${qualifier ? ` | Qualifier: ${qualifier}` : ""}` +
          `${reference ? ` | Ref: ${reference}` : ""}`
        );
      });

      const limitations = [];
      if (resolutionNote) limitations.push(resolutionNote);
      limitations.push("GO annotations mix evidence types and curator sources; review evidence codes before making mechanistic claims.");

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `QuickGO returned ${results.length} annotation(s) for ${geneProductIds.join(", ")}.`,
            keyFields,
            sources: [url, ...geneProductIds.slice(0, 3).map((id) => `https://www.ebi.ac.uk/QuickGO/annotations?geneProductId=${encodeURIComponent(id)}`)],
            limitations,
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_quickgo_annotations: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: CELLxGENE Discover / Census metadata search
// ============================================
server.registerTool(
  "search_cellxgene_datasets",
  {
    description:
      "Searches public single-cell dataset metadata from CELLxGENE Discover, which underlies CELLxGENE Census releases. " +
      "Use for cell-type, tissue, disease, assay, and organism-level dataset discovery.",
    inputSchema: {
      query: z.string().optional().describe("Free-text query over title, collection, disease, tissue, cell type, assay, and organism."),
      organism: z.string().optional().describe("Optional organism label filter (e.g. 'Homo sapiens')."),
      disease: z.string().optional().describe("Optional disease label filter."),
      tissue: z.string().optional().describe("Optional tissue label filter."),
      cellType: z.string().optional().describe("Optional cell-type label filter."),
      assay: z.string().optional().describe("Optional assay label filter."),
      limit: z.number().optional().default(10).describe("Maximum datasets to return (1-25)."),
    },
  },
  async ({ query = "", organism = "", disease = "", tissue = "", cellType = "", assay = "", limit = 10 }) => {
    const boundedLimit = Math.max(1, Math.min(25, Math.round(limit || 10)));
    const queryTokens = tokenizeQuery(query);
    const organismToken = normalizeWhitespace(organism || "").toLowerCase();
    const diseaseToken = normalizeWhitespace(disease || "").toLowerCase();
    const tissueToken = normalizeWhitespace(tissue || "").toLowerCase();
    const cellTypeToken = normalizeWhitespace(cellType || "").toLowerCase();
    const assayToken = normalizeWhitespace(assay || "").toLowerCase();

    try {
      const datasets = await fetchCellxgeneDatasets();
      const filtered = datasets.filter((dataset) => {
        const text = buildCellxgeneDatasetText(dataset);
        if (queryTokens.length > 0 && !matchesAllTokens(text, queryTokens)) return false;
        if (organismToken && !flattenOntologyLabels(dataset?.organism, 12).some((value) => value.toLowerCase().includes(organismToken))) return false;
        if (diseaseToken && !flattenOntologyLabels(dataset?.disease, 20).some((value) => value.toLowerCase().includes(diseaseToken))) return false;
        if (tissueToken && !flattenOntologyLabels(dataset?.tissue, 20).some((value) => value.toLowerCase().includes(tissueToken))) return false;
        if (cellTypeToken && !flattenOntologyLabels(dataset?.cell_type, 60).some((value) => value.toLowerCase().includes(cellTypeToken))) return false;
        if (assayToken && !flattenOntologyLabels(dataset?.assay, 12).some((value) => value.toLowerCase().includes(assayToken))) return false;
        return true;
      });

      const ranked = filtered
        .slice()
        .sort((a, b) => {
          const aCells = Math.max(toNonNegativeInt(a?.primary_cell_count, 0), toNonNegativeInt(a?.cell_count, 0));
          const bCells = Math.max(toNonNegativeInt(b?.primary_cell_count, 0), toNonNegativeInt(b?.cell_count, 0));
          return bCells - aCells;
        })
        .slice(0, boundedLimit);

      if (ranked.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: "No CELLxGENE datasets matched the requested filters.",
              keyFields: [
                query ? `Query: ${query}` : "Query: none",
                organism ? `Organism: ${organism}` : "Organism: any",
                disease ? `Disease: ${disease}` : "Disease: any",
                tissue ? `Tissue: ${tissue}` : "Tissue: any",
                cellType ? `Cell type: ${cellType}` : "Cell type: any",
                assay ? `Assay: ${assay}` : "Assay: any",
              ],
              sources: [`${CELLXGENE_DISCOVER_API}/collections?visibility=PUBLIC`],
              limitations: ["This integration searches public Discover metadata; it does not yet query the full Census SOMA object directly."],
            }),
          }],
        };
      }

      const keyFields = ranked.map((dataset, idx) => {
        const title = normalizeWhitespace(dataset?.title || "Untitled dataset");
        const collectionName = normalizeWhitespace(dataset?.collection_name || "Unknown collection");
        const organismText = flattenOntologyLabels(dataset?.organism, 3).join(", ") || "unknown organism";
        const diseaseText = flattenOntologyLabels(dataset?.disease, 3).join(", ") || "unspecified";
        const tissueText = flattenOntologyLabels(dataset?.tissue, 3).join(", ") || "unspecified";
        const cellTypeText = flattenOntologyLabels(dataset?.cell_type, 4).join(", ") || "unspecified";
        const assayText = flattenOntologyLabels(dataset?.assay, 3).join(", ") || "unspecified";
        const primaryCellCount = toNonNegativeInt(dataset?.primary_cell_count, 0);
        const totalCellCount = toNonNegativeInt(dataset?.cell_count, 0);
        const cellsLabel = primaryCellCount > 0 ? `${primaryCellCount.toLocaleString()} primary cells` : `${totalCellCount.toLocaleString()} cells`;
        return (
          `${idx + 1}. ${title} | Collection: ${collectionName} | Organism: ${organismText} ` +
          `| Disease: ${diseaseText} | Tissue: ${tissueText} | Cell types: ${cellTypeText} ` +
          `| Assay: ${assayText} | ${cellsLabel} | Dataset ID: ${dataset?.dataset_id || "unknown"}`
        );
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `CELLxGENE returned ${ranked.length} public dataset(s) matching the requested filters.`,
            keyFields,
            sources: [
              `${CELLXGENE_DISCOVER_API}/collections?visibility=PUBLIC`,
              ...ranked.slice(0, 5).map((dataset) => normalizeWhitespace(dataset?.explorer_url || "")).filter(Boolean),
            ],
            limitations: [
              "This tool currently searches the public Discover metadata API rather than opening the full Census SOMA object directly.",
              "Dataset-level metadata can show many cell types; follow up in the Explorer or Census APIs for per-cell analyses.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_cellxgene_datasets: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: CELLxGENE marker genes
// ============================================
server.registerTool(
  "get_cellxgene_marker_genes",
  {
    description:
      "Get top marker genes for a CELLxGENE / Census cell type within an organism+tissue context using the public WMG marker-gene API. " +
      "Prefer this for Cell X Gene questions about highest marker genes or marker-effect rankings, not dataset discovery.",
    inputSchema: {
      cellType: z.string().describe("Cell type label or ontology ID (for example 'mononuclear cell' or 'CL:0000842')."),
      tissue: z.string().describe("Tissue label or ontology ID (for example 'eye' or 'UBERON:0000970')."),
      organism: z.string().optional().default("Homo sapiens").describe("Organism label or ontology ID (default 'Homo sapiens')."),
      disease: z.string().optional().describe("Optional disease label used to refine CELLxGENE ontology resolution before the marker-gene lookup."),
      test: z.enum(["ttest", "binomtest"]).optional().default("ttest").describe("Marker-gene scoring method supported by the public WMG API."),
      nMarkers: z.number().optional().default(10).describe("Maximum marker genes to return (1-50)."),
    },
  },
  async ({ cellType, tissue, organism = "Homo sapiens", disease = "", test = "ttest", nMarkers = 10 }) => {
    const requestedCellType = normalizeWhitespace(cellType || "");
    const requestedTissue = normalizeWhitespace(tissue || "");
    const requestedOrganism = normalizeWhitespace(organism || "Homo sapiens") || "Homo sapiens";
    const requestedDisease = normalizeWhitespace(disease || "");
    const boundedMarkers = Math.max(1, Math.min(50, Math.round(nMarkers || 10)));

    if (!requestedCellType || !requestedTissue) {
      return { content: [{ type: "text", text: "Provide both a CELLxGENE cell type and tissue." }] };
    }

    try {
      const primaryDimensions = await fetchCellxgeneWmgPrimaryDimensions();
      const organismOption = resolveCellxgeneOntologyOption(primaryDimensions?.organism_terms, requestedOrganism);
      if (!organismOption) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `CELLxGENE could not resolve organism "${requestedOrganism}".`,
              keyFields: [`Requested organism: ${requestedOrganism}`],
              sources: [`${CELLXGENE_WMG_API}/primary_filter_dimensions`],
              limitations: ["Use a CELLxGENE-supported organism label such as 'Homo sapiens' or an NCBITaxon CURIE."],
            }),
          }],
        };
      }

      const tissueOptions = primaryDimensions?.tissue_terms?.[organismOption.id];
      const tissueOption = resolveCellxgeneOntologyOption(tissueOptions, requestedTissue);
      if (!tissueOption) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `CELLxGENE could not resolve tissue "${requestedTissue}" for ${organismOption.label}.`,
              keyFields: [
                `Requested tissue: ${requestedTissue}`,
                `Resolved organism: ${organismOption.label} (${organismOption.id})`,
              ],
              sources: [`${CELLXGENE_WMG_API}/primary_filter_dimensions`],
              limitations: ["The tissue may not be represented in the public WMG release for the requested organism."],
            }),
          }],
        };
      }

      const baseFilter = {
        organism_ontology_term_id: organismOption.id,
        tissue_ontology_term_ids: [tissueOption.id],
      };
      const deFilterResponse = await fetchCellxgeneDeFilters(baseFilter);
      let diseaseScopedDeFilterResponse = null;
      let diseaseOption = null;
      if (requestedDisease) {
        diseaseOption = resolveCellxgeneOntologyOption(deFilterResponse?.filter_dims?.disease_terms, requestedDisease);
        if (diseaseOption) {
          diseaseScopedDeFilterResponse = await fetchCellxgeneDeFilters({
            ...baseFilter,
            disease_ontology_term_ids: [diseaseOption.id],
          });
        }
      }

      const cellTypeResolutionSource = diseaseScopedDeFilterResponse || deFilterResponse;
      const cellTypeOption = resolveCellxgeneOntologyOption(
        cellTypeResolutionSource?.filter_dims?.cell_type_terms,
        requestedCellType,
      );
      if (!cellTypeOption) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `CELLxGENE could not resolve cell type "${requestedCellType}" in ${tissueOption.label}.`,
              keyFields: [
                `Requested cell type: ${requestedCellType}`,
                `Resolved organism: ${organismOption.label} (${organismOption.id})`,
                `Resolved tissue: ${tissueOption.label} (${tissueOption.id})`,
                ...(diseaseOption ? [`Disease context: ${diseaseOption.label} (${diseaseOption.id})`] : []),
              ],
              sources: [`${CELLXGENE_DE_API}/filters`],
              limitations: ["The requested cell type may not be represented in the public CELLxGENE WMG release for that context."],
            }),
          }],
        };
      }

      const markerResponse = await fetchCellxgeneMarkerGenes({
        cellTypeId: cellTypeOption.id,
        organismId: organismOption.id,
        tissueId: tissueOption.id,
        nMarkers: boundedMarkers,
        test,
      });
      const markerRows = Array.isArray(markerResponse?.marker_genes) ? markerResponse.marker_genes : [];
      const geneSymbols = buildCellxgeneGeneSymbolMap(primaryDimensions, organismOption.id);
      const rankedMarkers = markerRows.map((row) => {
        const geneId = normalizeWhitespace(row?.gene_ontology_term_id || "");
        const geneSymbol = geneSymbols.get(geneId) || geneId || "unknown gene";
        const markerScore = toNullableNumber(row?.marker_score);
        const specificity = toNullableNumber(row?.specificity);
        return {
          geneId,
          geneSymbol,
          markerScore,
          specificity,
        };
      });

      if (rankedMarkers.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `CELLxGENE returned no marker genes for ${cellTypeOption.label} in ${tissueOption.label}.`,
              keyFields: [
                `Resolved organism: ${organismOption.label} (${organismOption.id})`,
                `Resolved tissue: ${tissueOption.label} (${tissueOption.id})`,
                `Resolved cell type: ${cellTypeOption.label} (${cellTypeOption.id})`,
                ...(diseaseOption ? [`Disease context: ${diseaseOption.label} (${diseaseOption.id})`] : []),
                `Marker test: ${test}`,
              ],
              sources: [`${CELLXGENE_WMG_API}/markers`, `${CELLXGENE_WMG_API}/filters`],
              limitations: ["The public WMG marker endpoint can return empty marker lists for some cell-type/tissue contexts."],
            }),
          }],
        };
      }

      const topMarker = rankedMarkers[0];
      const keyFields = [
        `Top marker gene: ${topMarker.geneSymbol}${Number.isFinite(topMarker.markerScore) ? ` (marker score ${topMarker.markerScore.toFixed(4)})` : ""}`,
        `Resolved organism: ${organismOption.label} (${organismOption.id})`,
        `Resolved tissue: ${tissueOption.label} (${tissueOption.id})`,
        `Resolved cell type: ${cellTypeOption.label} (${cellTypeOption.id})`,
        ...(diseaseOption ? [`Disease context used for ontology resolution: ${diseaseOption.label} (${diseaseOption.id})`] : []),
        `Marker test: ${test}`,
        ...rankedMarkers.slice(0, Math.min(10, rankedMarkers.length)).map((row, idx) => (
          `${idx + 1}. ${row.geneSymbol}` +
          `${Number.isFinite(row.markerScore) ? ` | marker score ${row.markerScore.toFixed(4)}` : ""}` +
          `${Number.isFinite(row.specificity) ? ` | specificity ${row.specificity.toFixed(4)}` : ""}` +
          `${row.geneId ? ` | ${row.geneId}` : ""}`
        )),
      ];

      const sources = [
        `${CELLXGENE_WMG_API}/primary_filter_dimensions`,
        `${CELLXGENE_WMG_API}/filters`,
        `${CELLXGENE_WMG_API}/markers`,
      ];
      const contextDatasetsSource = diseaseScopedDeFilterResponse || deFilterResponse;
      const datasets = Array.isArray(contextDatasetsSource?.filter_dims?.datasets) ? contextDatasetsSource.filter_dims.datasets : [];
      for (const dataset of datasets.slice(0, 3)) {
        const collectionId = normalizeWhitespace(dataset?.collection_id || "");
        if (collectionId) {
          sources.push(`${CELLXGENE_DISCOVER_API}/collections/${encodeURIComponent(collectionId)}`);
        }
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `CELLxGENE marker-gene lookup returned ${rankedMarkers.length} marker gene(s) for ${cellTypeOption.label} in ${tissueOption.label}.`,
            keyFields,
            sources: dedupeArray(sources),
            limitations: [
              "The public WMG marker endpoint ranks genes by a marker score, not by a raw differential-expression coefficient.",
              "Optional disease context is used here to refine ontology resolution; the marker API itself is organism+tissue+cell-type based.",
            ],
          }),
        }],
        structuredContent: {
          schema: "cellxgene_marker_genes.v1",
          result_status: "ok",
          organism: organismOption,
          tissue: tissueOption,
          cell_type: cellTypeOption,
          disease: diseaseOption,
          marker_test: test,
          top_marker_gene: topMarker.geneSymbol,
          marker_genes: rankedMarkers,
        },
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_cellxgene_marker_genes: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Europe PMC literature search
// ============================================
server.registerTool(
  "search_europe_pmc_literature",
  {
    description:
      "Search Europe PMC for literature, preprints, open-access records, citation counts, and linked metadata. " +
      "Use when PubMed is too narrow or you specifically want preprints and Europe PMC citation metadata.",
    inputSchema: {
      query: z.string().describe("Search query for Europe PMC (e.g. 'TP53 cancer', 'LRRK2 Parkinson')."),
      source: z.string().optional().describe("Optional source filter such as MED, PMC, or PPR (preprints)."),
      openAccessOnly: z.boolean().optional().default(false).describe("If true, restrict to open-access records."),
      limit: z.number().optional().default(5).describe("Maximum records to return (1-10)."),
    },
  },
  async ({ query, source = "", openAccessOnly = false, limit = 5 }) => {
    const boundedLimit = Math.max(1, Math.min(10, Math.round(limit || 5)));
    const sourceFilter = normalizeWhitespace(source || "").toUpperCase();
    let europePmcQuery = query;
    if (sourceFilter) europePmcQuery = `SRC:${sourceFilter} AND (${europePmcQuery})`;
    if (openAccessOnly) europePmcQuery = `(${europePmcQuery}) AND OPEN_ACCESS:y`;

    const params = new URLSearchParams({
      query: europePmcQuery,
      format: "json",
      pageSize: String(boundedLimit),
      resultType: "core",
    });
    const url = `${EUROPE_PMC_API}/search?${params.toString()}`;

    try {
      const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 });
      const results = Array.isArray(data?.resultList?.result) ? data.resultList.result.slice(0, boundedLimit) : [];
      if (results.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Europe PMC returned no records for "${query}".`,
              keyFields: [
                sourceFilter ? `Source filter: ${sourceFilter}` : "Source filter: any",
                openAccessOnly ? "Open access only: yes" : "Open access only: no",
              ],
              sources: [url],
              limitations: ["Try broader keywords, omit the source filter, or remove the open-access restriction."],
            }),
          }],
        };
      }

      const keyFields = results.map((row, idx) => {
        const articleUrl = normalizeWhitespace(row?.source && row?.id ? `https://europepmc.org/article/${row.source}/${row.id}` : "");
        const pmid = normalizeWhitespace(row?.pmid || "");
        const doi = normalizeWhitespace(row?.doi || "");
        const citationCount = toNonNegativeInt(row?.citedByCount, 0);
        return (
          `${idx + 1}. ${normalizeWhitespace(row?.title || "Untitled")} ` +
          `| Source: ${normalizeWhitespace(row?.source || "unknown")} ` +
          `| Year: ${normalizeWhitespace(row?.pubYear || "unknown")} ` +
          `| PMID: ${pmid || "n/a"} | DOI: ${doi || "n/a"} ` +
          `| Citations: ${citationCount} | Open access: ${normalizeWhitespace(row?.isOpenAccess || "unknown")}` +
          `${articleUrl ? ` | ${articleUrl}` : ""}`
        );
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Europe PMC returned ${results.length} record(s) for "${query}".`,
            keyFields,
            sources: [url, ...results.map((row) => normalizeWhitespace(row?.source && row?.id ? `https://europepmc.org/article/${row.source}/${row.id}` : "")).filter(Boolean).slice(0, 5)],
            limitations: ["Europe PMC includes preprints and multiple record sources; review publication status before treating findings as peer reviewed."],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_europe_pmc_literature: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Pathway Commons top pathways
// ============================================
server.registerTool(
  "search_pathway_commons_top_pathways",
  {
    description:
      "Search top pathways in Pathway Commons across integrated pathway sources. " +
      "Use to broaden pathway/context retrieval beyond Reactome alone.",
    inputSchema: {
      query: z.string().describe("Gene, disease, or pathway keyword query (e.g. 'EGFR', 'apoptosis', 'Parkinson disease')."),
      organism: z.string().optional().default("9606").describe("NCBI taxonomy ID or common species label (default: 9606 = human)."),
      limit: z.number().optional().default(5).describe("Maximum pathways to return (1-10)."),
    },
  },
  async ({ query, organism = "9606", limit = 5 }) => {
    const boundedLimit = Math.max(1, Math.min(10, Math.round(limit || 5)));
    const normalizedOrganism = (() => {
      const raw = normalizeWhitespace(organism || "");
      const lower = raw.toLowerCase();
      if (!raw || lower === "human" || lower === "homo sapiens") return "9606";
      if (lower === "mouse" || lower === "mus musculus") return "10090";
      if (lower === "rat" || lower === "rattus norvegicus") return "10116";
      return raw;
    })();

    const params = new URLSearchParams({
      q: query,
      organism: normalizedOrganism,
    });
    const url = `${PATHWAY_COMMONS_API}/top_pathways?${params.toString()}`;

    try {
      const data = await fetchJsonWithRetry(url, {
        headers: { Accept: "application/json" },
        retries: 1,
        timeoutMs: 15000,
        maxBackoffMs: 2500,
      });
      const hits = Array.isArray(data?.searchHit) ? data.searchHit.slice(0, boundedLimit) : [];
      if (hits.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Pathway Commons returned no pathways for "${query}".`,
              keyFields: [`Organism: ${normalizedOrganism}`],
              sources: [url],
              limitations: ["Try a gene symbol, a shorter phrase, or a broader disease/pathway keyword."],
            }),
          }],
        };
      }

      const keyFields = hits.map((hit, idx) => {
        const dataSources = Array.isArray(hit?.dataSource) ? hit.dataSource.join(", ") : "unknown";
        return (
          `${idx + 1}. ${normalizeWhitespace(hit?.name || "Unnamed pathway")} ` +
          `| Source DBs: ${dataSources} ` +
          `| Participants: ${toNonNegativeInt(hit?.numParticipants, 0)} ` +
          `| Processes: ${toNonNegativeInt(hit?.numProcesses, 0)} ` +
          `| URI: ${normalizeWhitespace(hit?.uri || "n/a")}`
        );
      });

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Pathway Commons returned ${hits.length} top pathway result(s) for "${query}".`,
            keyFields,
            sources: [url, ...hits.map((hit) => normalizeWhitespace(hit?.uri || "")).filter(Boolean).slice(0, 5)],
            limitations: ["Pathway Commons aggregates many providers; pathway names and granularity can vary across source databases."],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in search_pathway_commons_top_pathways: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Guide to Pharmacology curated target profile
// ============================================
server.registerTool(
  "get_guidetopharmacology_target",
  {
    description:
      "Retrieve a curated Guide to Pharmacology target profile and representative ligand interactions. " +
      "Use for curated target-ligand relationships, action types, and affinity evidence.",
    inputSchema: {
      query: z.string().describe("Gene symbol or target name (e.g. 'EGFR', 'HER3', 'PPARG')."),
      species: z.string().optional().default("Human").describe("Species filter for interactions (default: Human)."),
      approvedOnly: z.boolean().optional().default(false).describe("If true, return approved-drug interactions only."),
      primaryTargetOnly: z.boolean().optional().default(false).describe("If true, restrict interactions to primary targets."),
      interactionLimit: z.number().optional().default(5).describe("Maximum ligand interactions to return (1-10)."),
    },
  },
  async ({ query, species = "Human", approvedOnly = false, primaryTargetOnly = false, interactionLimit = 5 }) => {
    const normalizedSpecies = normalizeGtopdbSpecies(species || "Human") || "Human";
    const boundedLimit = Math.max(1, Math.min(10, Math.round(interactionLimit || 5)));

    try {
      const geneSymbolUrl = `${GTOPDB_API}/targets?${new URLSearchParams({ geneSymbol: query }).toString()}`;
      const nameUrl = `${GTOPDB_API}/targets?${new URLSearchParams({ name: query }).toString()}`;
      const [geneSymbolHits, nameHits] = await Promise.all([
        fetchJsonWithRetry(geneSymbolUrl, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 }).catch(() => []),
        fetchJsonWithRetry(nameUrl, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 }).catch(() => []),
      ]);
      const mergedTargets = dedupeArray(
        [...(Array.isArray(geneSymbolHits) ? geneSymbolHits : []), ...(Array.isArray(nameHits) ? nameHits : [])]
          .map((row) => JSON.stringify(row))
      ).map((row) => JSON.parse(row));
      const targets = mergedTargets.sort((a, b) => scoreGtopdbTarget(b, query) - scoreGtopdbTarget(a, query));

      if (targets.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Guide to Pharmacology returned no targets for "${query}".`,
              keyFields: [`Species for interactions: ${normalizedSpecies}`],
              sources: [geneSymbolUrl, nameUrl],
              limitations: ["Try a canonical HGNC symbol or a full target name."],
            }),
          }],
        };
      }

      const bestTarget = targets[0];
      const interactionParams = new URLSearchParams({ species: normalizedSpecies });
      if (approvedOnly) interactionParams.set("approved", "true");
      if (primaryTargetOnly) interactionParams.set("primaryTarget", "true");
      const interactionsUrl = `${GTOPDB_API}/targets/${encodeURIComponent(bestTarget.targetId)}/interactions?${interactionParams.toString()}`;
      const interactions = await fetchJsonWithRetry(interactionsUrl, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 }).catch(() => []);
      const topInteractions = Array.isArray(interactions) ? interactions.slice(0, boundedLimit) : [];

      const candidateText = targets
        .slice(0, 3)
        .map((target) => `${normalizeWhitespace(target?.abbreviation || target?.name || "unknown")} (targetId=${target?.targetId || "n/a"})`)
        .join(", ");

      const keyFields = [
        `Target: ${normalizeWhitespace(bestTarget?.abbreviation || "") || query} — ${normalizeWhitespace(bestTarget?.name || "Unknown target")} | Target ID: ${bestTarget?.targetId || "n/a"} | Type: ${normalizeWhitespace(bestTarget?.type || "unknown")}`,
        candidateText ? `Candidate matches: ${candidateText}` : "",
        ...topInteractions.map((row, idx) => (
          `${idx + 1}. ${normalizeWhitespace(row?.ligandName || "Unknown ligand")} ` +
          `| Action type: ${normalizeWhitespace(row?.type || "n/a")} ` +
          `| Action: ${normalizeWhitespace(row?.action || "n/a")} ` +
          `| Affinity: ${normalizeWhitespace(row?.affinity || "n/a")} ${normalizeWhitespace(row?.affinityParameter || "")}` +
          ` | Primary target: ${row?.primaryTarget === true ? "yes" : "no"}`
        )),
      ].filter(Boolean);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Guide to Pharmacology found ${targets.length} target candidate(s) and ${topInteractions.length} representative interaction(s) for "${query}".`,
            keyFields,
            sources: [geneSymbolUrl, nameUrl, interactionsUrl],
            limitations: [
              "Guide to Pharmacology is curated and selective rather than exhaustive; absence of an interaction is not evidence of no activity.",
              approvedOnly ? "Approved-only filtering can exclude investigational ligands with strong affinity data." : "Interaction summaries can mix approved and investigational ligands unless you enable approvedOnly.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_guidetopharmacology_target: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: DailyMed drug label summary
// ============================================
server.registerTool(
  "get_dailymed_drug_label",
  {
    description:
      "Search DailyMed labels and summarize key label sections such as boxed warnings, indications, contraindications, and warnings/precautions. " +
      "Use for current US label context, not post-marketing signal counts.",
    inputSchema: {
      drugName: z.string().optional().describe("Drug or ingredient name to search in DailyMed (e.g. 'metformin', 'osimertinib')."),
      setId: z.string().optional().describe("Optional DailyMed SETID to fetch directly."),
      resultLimit: z.number().optional().default(3).describe("Maximum matching labels to consider from search (1-5)."),
    },
  },
  async ({ drugName = "", setId = "", resultLimit = 3 }) => {
    const boundedLimit = Math.max(1, Math.min(5, Math.round(resultLimit || 3)));
    const normalizedSetId = normalizeWhitespace(setId || "");
    const normalizedDrugName = normalizeWhitespace(drugName || "");
    if (!normalizedSetId && !normalizedDrugName) {
      return { content: [{ type: "text", text: "Provide either `drugName` or `setId` for DailyMed lookup." }] };
    }

    try {
      let chosenLabel = null;
      let searchUrl = "";
      let searchResults = [];
      if (!normalizedSetId) {
        searchUrl = `${DAILYMED_API}/spls.json?${new URLSearchParams({
          drug_name: normalizedDrugName,
          pagesize: String(boundedLimit),
        }).toString()}`;
        const searchData = await fetchJsonWithRetry(searchUrl, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 });
        searchResults = Array.isArray(searchData?.data) ? searchData.data.slice(0, boundedLimit) : [];
        searchResults.sort((a, b) => parsePublishedDate(b?.published_date) - parsePublishedDate(a?.published_date));
        chosenLabel = searchResults[0] || null;
      } else {
        chosenLabel = { setid: normalizedSetId, title: "", published_date: "" };
      }

      if (!chosenLabel?.setid) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `DailyMed returned no labels for "${normalizedDrugName}".`,
              keyFields: [],
              sources: searchUrl ? [searchUrl] : [],
              limitations: ["Try a generic ingredient name, a more specific product name, or fetch by SETID."],
            }),
          }],
        };
      }

      const xmlUrl = `${DAILYMED_API}/spls/${encodeURIComponent(chosenLabel.setid)}.xml`;
      const response = await fetchWithRetry(xmlUrl, { retries: 1, timeoutMs: 20000, maxBackoffMs: 2500 });
      const xml = await response.text();
      const productNames = extractDailymedProductNames(xml, 4);
      const ingredientNames = extractDailymedIngredientNames(xml, 6);
      const boxedWarning = extractDailymedSectionText(xml, ["BOXED WARNING SECTION"], [/^warning:/i]);
      const indications = extractDailymedSectionText(xml, ["INDICATIONS &amp; USAGE SECTION", "INDICATIONS & USAGE SECTION"], [/indications and usage/i]);
      const contraindications = extractDailymedSectionText(xml, ["CONTRAINDICATIONS SECTION"], [/contraindications/i]);
      const warnings = extractDailymedSectionText(xml, ["WARNINGS AND PRECAUTIONS SECTION", "WARNINGS SECTION"], [/warnings and precautions/i, /^warnings$/i]);
      const labelTitle = normalizeWhitespace(chosenLabel?.title || "") || productNames[0] || normalizedDrugName || normalizedSetId;
      const labelCandidates = searchResults
        .slice(0, boundedLimit)
        .map((row) => `${normalizeDailymedTitle(row?.title || "")} (${normalizeWhitespace(row?.published_date || "unknown date")})`)
        .join(" | ");

      const keyFields = [
        `Label: ${labelTitle} | SETID: ${chosenLabel.setid} | Published: ${normalizeWhitespace(chosenLabel?.published_date || "unknown")}`,
        productNames.length > 0 ? `Product names: ${productNames.join(", ")}` : "",
        ingredientNames.length > 0 ? `Active / listed ingredients: ${ingredientNames.join(", ")}` : "",
        labelCandidates ? `Recent matching labels: ${labelCandidates}` : "",
        boxedWarning ? `Boxed warning: ${boxedWarning}` : "Boxed warning: none found in retrieved SPL",
        indications ? `Indications and usage: ${indications}` : "",
        contraindications ? `Contraindications: ${contraindications}` : "",
        warnings ? `Warnings / precautions: ${warnings}` : "",
      ].filter(Boolean);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `DailyMed returned a label summary for ${labelTitle}.`,
            keyFields,
            sources: [searchUrl, xmlUrl, `https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=${encodeURIComponent(chosenLabel.setid)}`].filter(Boolean),
            limitations: [
              "This tool summarizes the latest retrieved SPL sections; review the full label for complete prescribing context.",
              "DailyMed label searches can return many manufacturer-specific variants for the same ingredient.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_dailymed_drug_label: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: ClinGen gene curation summary
// ============================================
server.registerTool(
  "get_clingen_gene_curation",
  {
    description:
      "Summarize ClinGen gene-disease validity and dosage sensitivity curations for a gene. " +
      "Use for curated evidence strength, dosage sensitivity, and expert-panel attributions.",
    inputSchema: {
      gene: z.string().describe("Gene symbol or HGNC identifier (e.g. 'TP53', 'HGNC:11998')."),
      disease: z.string().optional().describe("Optional disease name substring filter."),
      includeDosage: z.boolean().optional().default(true).describe("If true, include ClinGen dosage sensitivity curation when available."),
      limit: z.number().optional().default(5).describe("Maximum gene-disease validity curations to return (1-10)."),
    },
  },
  async ({ gene, disease = "", includeDosage = true, limit = 5 }) => {
    const boundedLimit = Math.max(1, Math.min(10, Math.round(limit || 5)));
    const normalizedGene = normalizeWhitespace(gene || "");
    const diseaseToken = normalizeWhitespace(disease || "").toLowerCase();

    try {
      const validityRows = await fetchClinGenGeneValidityRows();
      const dosageRows = includeDosage ? await fetchClinGenDosageRows() : [];

      const geneQueries = [normalizedGene.toUpperCase()];
      let resolvedSymbol = normalizedGene;
      let resolvedHgnc = "";
      if (!normalizedGene.toUpperCase().startsWith("HGNC:")) {
        try {
          const resolved = await resolveGeneWithMyGene(normalizedGene, "human");
          const ids = normalizeMyGeneIds(resolved?.bestHit || {});
          if (ids.symbol) {
            resolvedSymbol = ids.symbol;
            geneQueries.push(ids.symbol.toUpperCase());
          }
          if (normalizeWhitespace(resolved?.bestHit?.HGNC || "")) {
            resolvedHgnc = normalizeWhitespace(resolved.bestHit.HGNC);
            geneQueries.push(resolvedHgnc.toUpperCase());
          } else if (ids.symbol) {
            const geneLookupUrl = `https://search.clinicalgenome.org/api/genes/look/${encodeURIComponent(ids.symbol)}`;
            const geneLookupRows = await fetchJsonWithRetry(geneLookupUrl, { retries: 1, timeoutMs: 10000, maxBackoffMs: 2500 }).catch(() => []);
            const matchedGene = Array.isArray(geneLookupRows)
              ? geneLookupRows.find((row) => normalizeWhitespace(row?.label || "").toUpperCase() === ids.symbol.toUpperCase())
              : null;
            resolvedHgnc = normalizeWhitespace(matchedGene?.hgnc || "");
            if (resolvedHgnc) geneQueries.push(resolvedHgnc.toUpperCase());
          }
        } catch {
          // Ignore MyGene/ClinGen lookup fallback failures.
        }
      } else {
        resolvedHgnc = normalizedGene;
      }

      const filteredValidity = validityRows.filter((row) => {
        const rowSymbol = normalizeWhitespace(row?.["GENE SYMBOL"] || "").toUpperCase();
        const rowHgnc = normalizeWhitespace(row?.["GENE ID (HGNC)"] || "").toUpperCase();
        if (!geneQueries.includes(rowSymbol) && !geneQueries.includes(rowHgnc)) return false;
        if (diseaseToken && !normalizeWhitespace(row?.["DISEASE LABEL"] || "").toLowerCase().includes(diseaseToken)) return false;
        return true;
      }).slice(0, boundedLimit);

      const dosageMatch = Array.isArray(dosageRows)
        ? dosageRows.find((row) => {
            const rowSymbol = normalizeWhitespace(row?.["GENE SYMBOL"] || "").toUpperCase();
            const rowHgnc = normalizeWhitespace(row?.["HGNC ID"] || "").toUpperCase();
            return geneQueries.includes(rowSymbol) || geneQueries.includes(rowHgnc);
          })
        : null;

      if (filteredValidity.length === 0 && !dosageMatch) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `ClinGen returned no curations for "${gene}".`,
              keyFields: [disease ? `Disease filter: ${disease}` : "Disease filter: none"],
              sources: [CLINGEN_GENE_VALIDITY_DOWNLOAD, includeDosage ? CLINGEN_GENE_DOSAGE_DOWNLOAD : ""].filter(Boolean),
              limitations: ["ClinGen curation coverage is selective and gene aliases may require prior normalization."],
            }),
          }],
        };
      }

      const keyFields = [
        ...filteredValidity.map((row, idx) => (
          `${idx + 1}. ${normalizeWhitespace(row?.["GENE SYMBOL"] || resolvedSymbol || gene)} ` +
          `| Disease: ${normalizeWhitespace(row?.["DISEASE LABEL"] || "n/a")} ` +
          `| MONDO: ${normalizeWhitespace(row?.["DISEASE ID (MONDO)"] || "n/a")} ` +
          `| MOI: ${normalizeWhitespace(row?.MOI || "n/a")} ` +
          `| Classification: ${normalizeWhitespace(row?.CLASSIFICATION || "n/a")} ` +
          `| GCEP: ${normalizeWhitespace(row?.GCEP || "n/a")} ` +
          `| Date: ${normalizeWhitespace(row?.["CLASSIFICATION DATE"] || "n/a")}`
        )),
      ];

      if (dosageMatch) {
        keyFields.push(
          `Dosage sensitivity | Haploinsufficiency: ${normalizeWhitespace(dosageMatch?.HAPLOINSUFFICIENCY || "n/a")} ` +
          `| Triplosensitivity: ${normalizeWhitespace(dosageMatch?.TRIPLOSENSITIVITY || "n/a")} ` +
          `| Date: ${normalizeWhitespace(dosageMatch?.DATE || "n/a")}`
        );
      }

      const reportUrls = [
        ...filteredValidity.map((row) => normalizeWhitespace(row?.["ONLINE REPORT"] || "")),
        normalizeWhitespace(dosageMatch?.["ONLINE REPORT"] || ""),
      ].filter(Boolean).slice(0, 6);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `ClinGen returned ${filteredValidity.length} gene-disease validity curation(s)${dosageMatch ? " and dosage sensitivity data" : ""} for "${gene}".`,
            keyFields,
            sources: [CLINGEN_GENE_VALIDITY_DOWNLOAD, includeDosage ? CLINGEN_GENE_DOSAGE_DOWNLOAD : "", ...reportUrls].filter(Boolean),
            limitations: [
              "This tool summarizes released ClinGen curations from download files rather than the full evidence narrative.",
              "Use resolve_gene_identifiers first if you are querying historical aliases or non-HGNC identifiers.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_clingen_gene_curation: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: Alliance Genome Resources gene profile
// ============================================
server.registerTool(
  "get_alliance_genome_gene_profile",
  {
    description:
      "Summarize Alliance Genome Resources gene evidence for translational and model-organism research. " +
      "Returns resolved gene identity, model-species orthologs, disease/phenotype evidence counts, and representative disease models.",
    inputSchema: {
      query: z.string().describe("Gene symbol, alias, or gene identifier (for example 'TP53', 'LRRK2', or 'HGNC:11998')."),
      species: z.string().optional().default("human").describe("Preferred source species for resolving the query gene (default: human)."),
      orthologLimit: z.number().optional().default(8).describe("Maximum ortholog rows to summarize (1-12)."),
      modelLimit: z.number().optional().default(5).describe("Maximum disease-model rows to summarize (1-10)."),
    },
  },
  async ({ query, species = "human", orthologLimit = 8, modelLimit = 5 }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    const boundedOrthologLimit = Math.max(1, Math.min(12, Math.round(orthologLimit || 8)));
    const boundedModelLimit = Math.max(1, Math.min(10, Math.round(modelLimit || 5)));
    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a gene symbol, alias, or gene identifier for Alliance Genome Resources (for example TP53)." }] };
    }

    try {
      const resolution = await resolveAllianceGeneSelection(normalizedQuery, species);
      const gene = resolution?.geneRecord || null;
      const selected = resolution?.selected || null;
      if (!gene?.id && !selected?.id) {
        const candidateText = asArray(resolution?.candidates)
          .slice(0, 5)
          .map((item, idx) => `${idx + 1}. ${normalizeWhitespace(item?.symbol || "Unknown")} | ${normalizeWhitespace(item?.species || "unknown species")} | ${normalizeWhitespace(item?.id || item?.primaryKey || "")}`);
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Alliance Genome Resources found no gene profile for "${normalizedQuery}".`,
              keyFields: candidateText.length > 0 ? candidateText : [`Preferred species bias: ${species || "human"}`],
              sources: [resolution?.searchUrl || buildAllianceApiUrl("/search", new URLSearchParams({ q: normalizedQuery, category: "gene", limit: "8" }))],
              limitations: ["Try a canonical gene symbol or an explicit identifier such as HGNC:11998 or MGI:98834."],
            }),
          }],
        };
      }

      const resolvedId = normalizeWhitespace(gene?.id || selected?.id || "");
      const [experimentDiseaseSummary, orthologyDiseaseSummary, phenotypeSummary, orthologPayload, modelPayload] = await Promise.all([
        fetchAllianceJson(`/gene/${encodeURIComponent(resolvedId)}/disease-summary`, new URLSearchParams({ type: "experiment" }), { retries: 1, timeoutMs: 15000 }).catch(() => ({ url: "", data: {} })),
        fetchAllianceJson(`/gene/${encodeURIComponent(resolvedId)}/disease-summary`, new URLSearchParams({ type: "orthology" }), { retries: 1, timeoutMs: 15000 }).catch(() => ({ url: "", data: {} })),
        fetchAllianceJson(`/gene/${encodeURIComponent(resolvedId)}/phenotype-summary`, new URLSearchParams(), { retries: 1, timeoutMs: 15000 }).catch(() => ({ url: "", data: {} })),
        fetchAllianceJson(`/gene/${encodeURIComponent(resolvedId)}/orthologs`, new URLSearchParams({ limit: "50" }), { retries: 1, timeoutMs: 20000 }).catch(() => ({ url: "", data: {} })),
        fetchAllianceJson(`/gene/${encodeURIComponent(resolvedId)}/models`, new URLSearchParams({ limit: String(Math.max(20, boundedModelLimit * 4)) }), { retries: 1, timeoutMs: 20000 }).catch(() => ({ url: "", data: {} })),
      ]);

      const orthologRows = extractAllianceOrthologRows(orthologPayload?.data, resolution?.targetTaxon || normalizeAllianceSpeciesTaxon(species)).slice(0, boundedOrthologLimit);
      const modelRows = extractAllianceModelRows(modelPayload?.data).slice(0, boundedModelLimit);
      const speciesName = normalizeWhitespace(gene?.species?.name || selected?.species || "");
      const symbol = normalizeWhitespace(gene?.symbol || selected?.symbol || normalizedQuery);
      const geneName = normalizeWhitespace(gene?.name || "");
      const synopsis = normalizeWhitespace(gene?.geneSynopsis || gene?.automatedGeneSynopsis || "");
      const synonyms = dedupeArray(asArray(gene?.synonyms).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 8);
      const secondaryIds = dedupeArray(asArray(gene?.secondaryIds).map((value) => normalizeWhitespace(value)).filter(Boolean)).slice(0, 6);
      const provider = normalizeWhitespace(gene?.dataProvider || gene?.species?.dataProviderShortName || "");

      const keyFields = [
        `Gene: ${symbol}${geneName ? ` — ${geneName}` : ""} | AGR ID: ${resolvedId}${speciesName ? ` | Species: ${speciesName}` : ""}${provider ? ` | Provider: ${provider}` : ""}`,
        synopsis ? `Synopsis: ${compactErrorMessage(synopsis, 360)}` : "",
        secondaryIds.length > 0 ? `Secondary IDs: ${secondaryIds.join(", ")}` : "",
        synonyms.length > 0 ? `Synonyms: ${synonyms.join(", ")}` : "",
        Number.isFinite(Number(experimentDiseaseSummary?.data?.numberOfEntities))
          ? `Direct disease evidence: ${toNonNegativeInt(experimentDiseaseSummary.data.numberOfAnnotations)} annotation(s) across ${toNonNegativeInt(experimentDiseaseSummary.data.numberOfEntities)} disease entity(ies)`
          : "",
        Number.isFinite(Number(orthologyDiseaseSummary?.data?.numberOfEntities))
          ? `Orthology-transferred disease evidence: ${toNonNegativeInt(orthologyDiseaseSummary.data.numberOfAnnotations)} annotation(s) across ${toNonNegativeInt(orthologyDiseaseSummary.data.numberOfEntities)} disease entity(ies)`
          : "",
        Number.isFinite(Number(phenotypeSummary?.data?.numberOfEntities))
          ? `Phenotype evidence: ${toNonNegativeInt(phenotypeSummary.data.numberOfAnnotations)} annotation(s) across ${toNonNegativeInt(phenotypeSummary.data.numberOfEntities)} phenotype entity(ies)`
          : "",
      ].filter(Boolean);

      if (orthologRows.length > 0) {
        keyFields.push("Representative model-species orthologs:");
        keyFields.push(
          ...orthologRows.map((row, idx) =>
            `${idx + 1}. ${row.species} — ${row.symbol || "unknown"} (${row.id})` +
            `${row.confidence ? ` | Confidence: ${row.confidence}` : ""}` +
            `${row.methods.length > 0 ? ` | Methods: ${row.methods.join(", ")}` : ""}` +
            `${row.hasDiseaseAnnotations ? " | Has disease annotations" : ""}` +
            `${row.hasExpressionAnnotations ? " | Has expression annotations" : ""}`
          )
        );
      }

      if (modelRows.length > 0) {
        keyFields.push("Representative disease models:");
        keyFields.push(
          ...modelRows.map((row, idx) =>
            `${idx + 1}. ${row.species} (${row.provider}) — ${row.modelName || row.modelId || "unnamed model"}` +
            `${row.modelId ? ` | ${row.modelId}` : ""}` +
            `${row.diseaseModels.length > 0 ? ` | Diseases: ${row.diseaseModels.join(", ")}` : ""}` +
            `${row.modifierRelationshipTypes.length > 0 ? ` | Modifiers: ${row.modifierRelationshipTypes.join(", ")}` : ""}` +
            `${row.hasPhenotypeAnnotations ? " | Has phenotype annotations" : ""}`
          )
        );
      }

      const sources = [
        resolution?.searchUrl || "",
        buildAllianceApiUrl(`/gene/${encodeURIComponent(resolvedId)}`),
        experimentDiseaseSummary?.url || "",
        orthologyDiseaseSummary?.url || "",
        phenotypeSummary?.url || "",
        orthologPayload?.url || "",
        modelPayload?.url || "",
        `https://www.alliancegenome.org/gene/${encodeURIComponent(resolvedId)}`,
        normalizeWhitespace(gene?.modCrossRefCompleteUrl || ""),
      ].filter(Boolean);

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary:
              `Alliance Genome Resources profile for ${symbol} (${resolvedId}) with ${orthologRows.length} ortholog summary row(s)` +
              `${modelRows.length > 0 ? ` and ${modelRows.length} disease-model row(s)` : ""}.`,
            keyFields,
            sources: dedupeArray(sources).slice(0, 10),
            limitations: [
              "AGR is strongest for cross-species and model-organism context; it complements rather than replaces human-only clinical curation sources.",
              "Search resolution can be ambiguous for short aliases, and some orthology/model endpoints return broad result sets that are summarized here.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_alliance_genome_gene_profile: ${error.message}` }] };
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
// TOOL 26: IntAct experimental interactions
// ============================================
server.registerTool(
  "get_intact_interactions",
  {
    description:
      "Retrieve curated experimental molecular interactions from IntAct for a gene/protein identifier. " +
      "Returns exact interactor resolution, top partners, interaction types, detection methods, and publication support.",
    inputSchema: {
      query: z.string().describe("Gene symbol, protein name, UniProt accession, or IntAct interactor identifier (e.g. 'TP53', 'P04637')."),
      speciesTaxId: z.number().optional().default(9606).describe("Interactor species taxon filter (default: 9606 for human)."),
      intraSpeciesOnly: z.boolean().optional().default(true).describe("If true, keep only interactions where both partners match speciesTaxId."),
      proteinOnly: z.boolean().optional().default(true).describe("If true, keep only protein-protein interactions."),
      includeSelfInteractions: z.boolean().optional().default(false).describe("If false, drop self/isoform-self interactions so distinct partners are easier to inspect."),
      maxPages: z.number().optional().default(2).describe("Maximum IntAct result pages to scan (1-5, ~20 interactions per page)."),
      topPartnerLimit: z.number().optional().default(8).describe("Maximum partner proteins/entities to summarize (1-15)."),
    },
  },
  async ({
    query,
    speciesTaxId = 9606,
    intraSpeciesOnly = true,
    proteinOnly = true,
    includeSelfInteractions = false,
    maxPages = 2,
    topPartnerLimit = 8,
  }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    const boundedSpeciesTaxId = Math.max(0, toNonNegativeInt(speciesTaxId, 9606));
    const boundedPages = Math.max(1, Math.min(5, Math.round(maxPages || 2)));
    const boundedPartnerLimit = Math.max(1, Math.min(15, Math.round(topPartnerLimit || 8)));

    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a gene symbol, UniProt accession, or IntAct interactor query (for example TP53)." }] };
    }

    try {
      const { selected, candidates } = await resolveIntactInteractor(normalizedQuery, boundedSpeciesTaxId);
      if (!selected) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `IntAct returned no interactors for "${normalizedQuery}".`,
              keyFields: [`Query: ${normalizedQuery}`],
              sources: [`${INTACT_API}/interactor/findInteractor/${encodeURIComponent(normalizedQuery)}`],
              limitations: ["Try a UniProt accession or a canonical gene symbol if free-text search is ambiguous."],
            }),
          }],
        };
      }

      const anchorIdentifier =
        normalizeWhitespace(selected?.interactorPreferredIdentifier || "")
        || normalizeWhitespace(selected?.interactorName || "")
        || normalizedQuery;

      const pagePayloads = [];
      for (let page = 0; page < boundedPages; page += 1) {
        const payload = await fetchIntactInteractionPage(anchorIdentifier, page);
        pagePayloads.push(payload);
        if (payload?.last === true) break;
      }

      const scannedRows = pagePayloads.flatMap((payload) => Array.isArray(payload?.content) ? payload.content : []);
      const totalReported = toNonNegativeInt(pagePayloads[0]?.totalElements, scannedRows.length);
      const totalPages = Math.max(1, toNonNegativeInt(pagePayloads[0]?.totalPages, pagePayloads.length));

      const filteredRows = scannedRows.filter((row) => {
        if (Boolean(row?.negative)) return false;
        if (proteinOnly) {
          const typeA = normalizeWhitespace(row?.typeA || "").toLowerCase();
          const typeB = normalizeWhitespace(row?.typeB || "").toLowerCase();
          if (typeA !== "protein" || typeB !== "protein") return false;
        }
        if (boundedSpeciesTaxId > 0) {
          const taxIdA = toNonNegativeInt(row?.taxIdA, 0);
          const taxIdB = toNonNegativeInt(row?.taxIdB, 0);
          if (intraSpeciesOnly) {
            if (taxIdA !== boundedSpeciesTaxId || taxIdB !== boundedSpeciesTaxId) return false;
          } else if (taxIdA !== boundedSpeciesTaxId && taxIdB !== boundedSpeciesTaxId) {
            return false;
          }
        }
        return true;
      }).filter((row) => {
        if (includeSelfInteractions) return true;
        const partner = extractIntactPartner(row, anchorIdentifier);
        const anchorNormalized = normalizeIntactLookupText(anchorIdentifier);
        const anchorNameNormalized = normalizeIntactLookupText(selected?.interactorName || "");
        const partnerIdNormalized = normalizeIntactLookupText(partner.accession || partner.uniqueId || "");
        const partnerLabelNormalized = normalizeIntactLookupText(partner.molecule || partner.description || "");
        if (!partnerIdNormalized && !partnerLabelNormalized) return true;
        return (
          partnerIdNormalized !== anchorNormalized
          && partnerLabelNormalized !== anchorNameNormalized
        );
      });

      if (filteredRows.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `IntAct resolved "${normalizedQuery}" to ${anchorIdentifier}, but no interactions remained after the requested filters.`,
              keyFields: [
                `Resolved interactor: ${normalizeWhitespace(selected?.interactorName || anchorIdentifier)} | Preferred ID: ${anchorIdentifier}`,
                `Species filter: ${boundedSpeciesTaxId > 0 ? boundedSpeciesTaxId : "none"} | Intra-species only: ${intraSpeciesOnly ? "yes" : "no"} | Protein-only: ${proteinOnly ? "yes" : "no"}`,
                `Pages scanned: ${pagePayloads.length}/${totalPages}`,
              ],
              sources: [
                `${INTACT_API}/interactor/findInteractor/${encodeURIComponent(normalizedQuery)}`,
                `${INTACT_API}/interaction/findInteractions/${encodeURIComponent(anchorIdentifier)}?page=0`,
              ],
              limitations: [
                "Filtered IntAct interactions can be sparse for less-studied proteins or non-protein interactors.",
                "Relax protein-only or intra-species constraints if you need broader experimental context.",
              ],
            }),
          }],
        };
      }

      const partnerMap = new Map();
      const detectionMethodCounts = new Map();
      const interactionTypeCounts = new Map();
      const hostOrganismCounts = new Map();
      const publicationIds = new Set();

      for (const row of filteredRows) {
        const partner = extractIntactPartner(row, anchorIdentifier);
        const partnerKey = `${partner.uniqueId || partner.accession || partner.molecule}::${partner.taxId || 0}`;
        const score = toFiniteNumber(row?.intactMiscore, Number.NaN);
        const pubmedId = normalizeWhitespace(row?.publicationPubmedIdentifier || "");
        const detectionMethod = normalizeWhitespace(row?.detectionMethod || "");
        const interactionType = normalizeWhitespace(row?.type || "");
        const hostOrganism = normalizeWhitespace(row?.hostOrganism || "");

        if (detectionMethod) incrementCount(detectionMethodCounts, detectionMethod);
        if (interactionType) incrementCount(interactionTypeCounts, interactionType);
        if (hostOrganism) incrementCount(hostOrganismCounts, hostOrganism);
        if (pubmedId) publicationIds.add(pubmedId);

        const existing = partnerMap.get(partnerKey) || {
          label: partner.molecule || partner.accession || "Unknown partner",
          accession: partner.accession || partner.uniqueId || "",
          species: partner.species || "",
          taxId: partner.taxId || 0,
          type: partner.type || "",
          description: partner.description || "",
          interactionCount: 0,
          maxMiscore: Number.NaN,
          detectionMethods: new Set(),
          interactionTypes: new Set(),
          publications: new Set(),
          sourceDatabases: new Set(),
        };

        existing.interactionCount += 1;
        if (Number.isFinite(score)) {
          existing.maxMiscore = Number.isFinite(existing.maxMiscore) ? Math.max(existing.maxMiscore, score) : score;
        }
        if (detectionMethod) existing.detectionMethods.add(detectionMethod);
        if (interactionType) existing.interactionTypes.add(interactionType);
        if (pubmedId) existing.publications.add(pubmedId);
        if (normalizeWhitespace(row?.sourceDatabase || "")) existing.sourceDatabases.add(normalizeWhitespace(row.sourceDatabase));
        partnerMap.set(partnerKey, existing);
      }

      const topPartners = [...partnerMap.values()]
        .sort((a, b) => {
          if (b.interactionCount !== a.interactionCount) return b.interactionCount - a.interactionCount;
          const scoreA = Number.isFinite(a.maxMiscore) ? a.maxMiscore : -1;
          const scoreB = Number.isFinite(b.maxMiscore) ? b.maxMiscore : -1;
          return scoreB - scoreA;
        })
        .slice(0, boundedPartnerLimit);

      const topInteractionRows = filteredRows
        .slice()
        .sort((a, b) => toFiniteNumber(b?.intactMiscore, -1) - toFiniteNumber(a?.intactMiscore, -1))
        .slice(0, Math.min(5, filteredRows.length));

      const keyFields = [
        `Resolved interactor: ${normalizeWhitespace(selected?.interactorName || anchorIdentifier)} | Preferred ID: ${anchorIdentifier}`,
        `Interactor species: ${normalizeWhitespace(selected?.interactorSpecies || "n/a")}${selected?.interactorTaxId ? ` (taxon ${selected.interactorTaxId})` : ""}`,
        `Reported interaction count on interactor card: ${toNonNegativeInt(selected?.interactionCount, filteredRows.length).toLocaleString()}`,
        `Interactions analyzed after filters: ${filteredRows.length.toLocaleString()} of ${scannedRows.length.toLocaleString()} scanned (${totalReported.toLocaleString()} total reported by IntAct)`,
        `Pages scanned: ${pagePayloads.length}/${totalPages}`,
        `Unique partner entities: ${partnerMap.size.toLocaleString()}`,
        `Unique PubMed IDs: ${publicationIds.size.toLocaleString()}`,
        `Top interaction types: ${summarizeTopCounts(interactionTypeCounts, 5).join("; ") || "n/a"}`,
        `Top detection methods: ${summarizeTopCounts(detectionMethodCounts, 5).join("; ") || "n/a"}`,
        `Self-interactions included: ${includeSelfInteractions ? "yes" : "no"}`,
      ];
      const hostSummary = summarizeTopCounts(hostOrganismCounts, 4);
      if (hostSummary.length > 0) {
        keyFields.push(`Top host organisms: ${hostSummary.join("; ")}`);
      }
      const alternateMatches = dedupeArray(candidates.slice(1, 8).map((candidate) =>
        `${normalizeWhitespace(candidate?.interactorName || candidate?.interactorPreferredIdentifier || "unknown")} (${normalizeWhitespace(candidate?.interactorSpecies || "species n/a")})`
      )).slice(0, 4);
      if (alternateMatches.length > 0) {
        keyFields.push(`Other interactor matches: ${alternateMatches.join("; ")}`);
      }
      if (topPartners.length > 0) {
        keyFields.push("Top partners:");
        keyFields.push(
          ...topPartners.map((partner, idx) =>
            `${idx + 1}. ${partner.label}${partner.accession ? ` [${partner.accession}]` : ""}` +
            `${partner.species ? ` | ${partner.species}` : ""}` +
            ` | evidence rows ${partner.interactionCount}` +
            `${Number.isFinite(partner.maxMiscore) ? ` | max miscore ${partner.maxMiscore.toFixed(2)}` : ""}` +
            `${partner.interactionTypes.size > 0 ? ` | types: ${[...partner.interactionTypes].slice(0, 2).join(", ")}` : ""}` +
            `${partner.publications.size > 0 ? ` | PMIDs: ${[...partner.publications].slice(0, 3).join(", ")}` : ""}`
          )
        );
      }
      if (topInteractionRows.length > 0) {
        keyFields.push("Top individual interaction records:");
        keyFields.push(
          ...topInteractionRows.map((row, idx) => {
            const partner = extractIntactPartner(row, anchorIdentifier);
            return (
              `${idx + 1}. ${normalizeWhitespace(selected?.interactorName || anchorIdentifier)} -> ${partner.molecule || partner.accession || "partner"}` +
              `${normalizeWhitespace(row?.type || "") ? ` | ${normalizeWhitespace(row.type)}` : ""}` +
              `${normalizeWhitespace(row?.detectionMethod || "") ? ` | ${normalizeWhitespace(row.detectionMethod)}` : ""}` +
              `${Number.isFinite(toFiniteNumber(row?.intactMiscore, Number.NaN)) ? ` | miscore ${toFiniteNumber(row.intactMiscore, Number.NaN).toFixed(2)}` : ""}` +
              `${normalizeWhitespace(row?.publicationPubmedIdentifier || "") ? ` | PMID ${normalizeWhitespace(row.publicationPubmedIdentifier)}` : ""}`
            );
          })
        );
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary:
              `IntAct resolved "${normalizedQuery}" to ${anchorIdentifier} and found ${filteredRows.length.toLocaleString()} filtered experimental interaction record` +
              `${filteredRows.length === 1 ? "" : "s"} across ${partnerMap.size.toLocaleString()} partner` +
              `${partnerMap.size === 1 ? "" : "s"}.`,
            keyFields,
            sources: dedupeArray([
              `${INTACT_API}/interactor/findInteractor/${encodeURIComponent(normalizedQuery)}`,
              ...pagePayloads.slice(0, 2).map((_, idx) => `${INTACT_API}/interaction/findInteractions/${encodeURIComponent(anchorIdentifier)}?page=${idx}`),
            ]),
            limitations: [
              `Only the first ${pagePayloads.length} IntAct result page${pagePayloads.length === 1 ? "" : "s"} were scanned; lower-ranked interactions may be omitted.`,
              "IntAct captures curated experimental evidence and may underrepresent true biology for less-studied proteins.",
              "This complements STRING: IntAct emphasizes experimentally curated records rather than integrated predictive/network evidence.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_intact_interactions: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: BioGRID experimental interactions
// ============================================
server.registerTool(
  "get_biogrid_interactions",
  {
    description:
      "Retrieve experimental interaction evidence from BioGRID for a gene. " +
      "Returns broader physical/genetic interaction coverage, experimental systems, throughput tags, partner summaries, and supporting PMIDs.",
    inputSchema: {
      query: z.string().describe("Gene symbol, alias, or Entrez gene ID (for example 'TP53', 'EGFR', or '7157')."),
      speciesTaxId: z.number().optional().default(9606).describe("NCBI taxonomy ID filter for interactors (default: 9606 for human)."),
      interactionType: z.enum(["any", "physical", "genetic"]).optional().default("any")
        .describe("Restrict to physical or genetic interaction evidence, or keep both."),
      throughput: z.enum(["any", "low", "high"]).optional().default("any")
        .describe("Optional BioGRID throughput filter."),
      interSpeciesExcluded: z.boolean().optional().default(true)
        .describe("If true, keep only intra-species interactions matching speciesTaxId."),
      includeSelfInteractions: z.boolean().optional().default(false)
        .describe("If false, drop self-interactions so distinct partners are easier to inspect."),
      maxRecords: z.number().optional().default(250).describe("Maximum BioGRID interaction rows to request (25-1000)."),
      topPartnerLimit: z.number().optional().default(8).describe("Maximum partner entities to summarize (1-15)."),
    },
  },
  async ({
    query,
    speciesTaxId = 9606,
    interactionType = "any",
    throughput = "any",
    interSpeciesExcluded = true,
    includeSelfInteractions = false,
    maxRecords = 250,
    topPartnerLimit = 8,
  }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    const boundedSpeciesTaxId = Math.max(0, toNonNegativeInt(speciesTaxId, 9606));
    const boundedMaxRecords = Math.max(25, Math.min(1000, Math.round(maxRecords || 250)));
    const boundedPartnerLimit = Math.max(1, Math.min(15, Math.round(topPartnerLimit || 8)));

    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a gene symbol, alias, or Entrez gene ID for BioGRID (for example TP53)." }] };
    }
    if (!BIOGRID_ACCESS_KEY) {
      return buildBiogridMissingKeyResponse(
        "BioGRID",
        "BIOGRID_ACCESS_KEY",
        BIOGRID_API,
        `get_biogrid_interactions(query="${normalizedQuery}")`
      );
    }

    try {
      const resolved = await resolveBiogridGeneSelection(normalizedQuery, boundedSpeciesTaxId);
      const geneListValue = resolved.entrezGene || resolved.symbol || normalizedQuery;
      const params = new URLSearchParams({
        format: "json",
        geneList: geneListValue,
        searchNames: "true",
        searchIds: "true",
        searchSynonyms: "true",
        includeInteractors: "true",
        includeInteractorInteractions: "false",
        selfInteractionsExcluded: includeSelfInteractions ? "false" : "true",
        interSpeciesExcluded: interSpeciesExcluded ? "true" : "false",
        max: String(boundedMaxRecords),
      });
      if (boundedSpeciesTaxId > 0) params.set("taxId", String(boundedSpeciesTaxId));
      if (throughput !== "any") params.set("throughputTag", throughput);

      const payload = await fetchBiogridJson(BIOGRID_API, "/interactions", params, {
        accessKey: BIOGRID_ACCESS_KEY,
        retries: 1,
        timeoutMs: 20000,
        maxBackoffMs: 3000,
      });

      const rows = extractBiogridRows(payload.data);
      const anchorCandidates = dedupeArray([
        normalizedQuery,
        resolved.symbol,
        resolved.entrezGene,
        ...resolved.aliases,
      ]).filter(Boolean);

      const filteredRows = rows.filter((row) => {
        const evidenceType = normalizeWhitespace(getBiogridField(row, ["EXPERIMENTAL_SYSTEM_TYPE"])).toLowerCase();
        if (interactionType !== "any" && evidenceType !== interactionType) return false;

        const interactorA = extractBiogridInteractor(row, "A");
        const interactorB = extractBiogridInteractor(row, "B");
        if (boundedSpeciesTaxId > 0) {
          const isARequested = interactorA.taxId === boundedSpeciesTaxId;
          const isBRequested = interactorB.taxId === boundedSpeciesTaxId;
          if (interSpeciesExcluded) {
            if (!isARequested || !isBRequested) return false;
          } else if (!isARequested && !isBRequested) {
            return false;
          }
        }

        if (!includeSelfInteractions) {
          const sameEntrez = interactorA.entrezGene && interactorA.entrezGene === interactorB.entrezGene;
          const sameSymbol =
            normalizeBiogridLookupText(interactorA.symbol)
            && normalizeBiogridLookupText(interactorA.symbol) === normalizeBiogridLookupText(interactorB.symbol);
          if (sameEntrez || sameSymbol) return false;
        }
        return true;
      });

      if (filteredRows.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `BioGRID returned no interaction rows for "${normalizedQuery}" after applying the requested filters.`,
              keyFields: [
                `Resolved query: ${resolved.symbol || normalizedQuery}${resolved.entrezGene ? ` | Entrez: ${resolved.entrezGene}` : ""}`,
                `Rows returned before filtering: ${rows.length.toLocaleString()}`,
                `Requested interaction type: ${interactionType}`,
                `Requested throughput: ${throughput}`,
              ],
              sources: [resolved.resolutionUrl, payload.url].filter(Boolean),
              limitations: [
                "BioGRID coverage varies by gene and assay type.",
                "Relax the interaction type, throughput, or species filters if you want broader experimental coverage.",
              ],
            }),
          }],
        };
      }

      const interactionTypeCounts = new Map();
      const systemCounts = new Map();
      const throughputCounts = new Map();
      const sourceDbCounts = new Map();
      const pmids = new Set();
      const partnerMap = new Map();

      for (const row of filteredRows) {
        const partner = extractBiogridPartner(row, anchorCandidates);
        const partnerKey = `${partner.entrezGene || partner.symbol || partner.biogridId}::${partner.taxId || 0}`;
        const evidenceType = normalizeWhitespace(getBiogridField(row, ["EXPERIMENTAL_SYSTEM_TYPE"])) || "unknown";
        const system = normalizeWhitespace(getBiogridField(row, ["EXPERIMENTAL_SYSTEM"])) || "unknown";
        const throughputLabel = normalizeWhitespace(getBiogridField(row, ["THROUGHPUT"])) || "unknown";
        const sourceDb = normalizeWhitespace(getBiogridField(row, ["SOURCEDB", "SOURCE_DATABASE", "SOURCE"])) || "BioGRID";
        const pubmedId = normalizeWhitespace(getBiogridField(row, ["PUBMED_ID"]));
        const qualification = normalizeWhitespace(getBiogridField(row, ["QUALIFICATIONS"]));

        incrementCount(interactionTypeCounts, evidenceType);
        incrementCount(systemCounts, system);
        incrementCount(throughputCounts, throughputLabel);
        incrementCount(sourceDbCounts, sourceDb);
        if (pubmedId) pmids.add(pubmedId);

        const existing = partnerMap.get(partnerKey) || {
          label: partner.symbol || partner.systematicName || partner.entrezGene || "Unknown partner",
          entrezGene: partner.entrezGene,
          biogridId: partner.biogridId,
          taxId: partner.taxId,
          interactionCount: 0,
          evidenceTypes: new Set(),
          systems: new Set(),
          pmids: new Set(),
          qualifications: new Set(),
        };
        existing.interactionCount += 1;
        existing.evidenceTypes.add(evidenceType);
        existing.systems.add(system);
        if (pubmedId) existing.pmids.add(pubmedId);
        if (qualification) existing.qualifications.add(qualification);
        partnerMap.set(partnerKey, existing);
      }

      const topPartners = [...partnerMap.values()]
        .sort((a, b) => b.interactionCount - a.interactionCount || b.pmids.size - a.pmids.size)
        .slice(0, boundedPartnerLimit);
      const topRows = filteredRows.slice(0, Math.min(5, filteredRows.length));

      const keyFields = [
        `Resolved query: ${resolved.symbol || normalizedQuery}${resolved.name ? ` — ${resolved.name}` : ""}${resolved.entrezGene ? ` | Entrez: ${resolved.entrezGene}` : ""}`,
        `Species filter: ${boundedSpeciesTaxId || "none"} | Intra-species only: ${interSpeciesExcluded ? "yes" : "no"} | Self-interactions included: ${includeSelfInteractions ? "yes" : "no"}`,
        resolved.matchedViaMyGene ? "Identifier grounding: MyGene.info resolved the query before BioGRID lookup." : "Identifier grounding: BioGRID queried the original user string directly.",
        `BioGRID rows analyzed after filters: ${filteredRows.length.toLocaleString()} of ${rows.length.toLocaleString()} returned`,
        `Unique partner entities: ${partnerMap.size.toLocaleString()}`,
        `Unique PubMed IDs: ${pmids.size.toLocaleString()}`,
        `Interaction evidence classes: ${summarizeTopCounts(interactionTypeCounts, 5).join("; ") || "n/a"}`,
        `Experimental systems: ${summarizeTopCounts(systemCounts, 5).join("; ") || "n/a"}`,
        `Throughput mix: ${summarizeTopCounts(throughputCounts, 4).join("; ") || "n/a"}`,
      ];
      const sourceDbSummary = summarizeTopCounts(sourceDbCounts, 4);
      if (sourceDbSummary.length > 0) {
        keyFields.push(`Source databases: ${sourceDbSummary.join("; ")}`);
      }
      if (topPartners.length > 0) {
        keyFields.push("Top BioGRID partners:");
        keyFields.push(
          ...topPartners.map((partner, idx) =>
            `${idx + 1}. ${partner.label}` +
            `${partner.entrezGene ? ` | Entrez ${partner.entrezGene}` : ""}` +
            ` | rows ${partner.interactionCount}` +
            `${partner.evidenceTypes.size > 0 ? ` | types: ${[...partner.evidenceTypes].slice(0, 2).join(", ")}` : ""}` +
            `${partner.systems.size > 0 ? ` | systems: ${[...partner.systems].slice(0, 2).join(", ")}` : ""}` +
            `${partner.pmids.size > 0 ? ` | PMIDs: ${[...partner.pmids].slice(0, 3).join(", ")}` : ""}`
          )
        );
      }
      if (topRows.length > 0) {
        keyFields.push("Representative interaction records:");
        keyFields.push(
          ...topRows.map((row, idx) => {
            const partner = extractBiogridPartner(row, anchorCandidates);
            const system = normalizeWhitespace(getBiogridField(row, ["EXPERIMENTAL_SYSTEM"])) || "unknown system";
            const evidenceType = normalizeWhitespace(getBiogridField(row, ["EXPERIMENTAL_SYSTEM_TYPE"])) || "unknown type";
            const author = normalizeWhitespace(getBiogridField(row, ["PUBMED_AUTHOR", "AUTHOR"]));
            const pubmedId = normalizeWhitespace(getBiogridField(row, ["PUBMED_ID"]));
            const qualification = normalizeWhitespace(getBiogridField(row, ["QUALIFICATIONS"]));
            return (
              `${idx + 1}. ${resolved.symbol || normalizedQuery} -> ${partner.symbol || partner.systematicName || partner.entrezGene || "partner"}` +
              ` | ${system}` +
              ` | ${evidenceType}` +
              `${pubmedId ? ` | PMID ${pubmedId}` : ""}` +
              `${author ? ` | ${author}` : ""}` +
              `${qualification ? ` | qualification: ${qualification}` : ""}`
            );
          })
        );
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary:
              `BioGRID found ${filteredRows.length.toLocaleString()} filtered interaction record` +
              `${filteredRows.length === 1 ? "" : "s"} for ${resolved.symbol || normalizedQuery} across ${partnerMap.size.toLocaleString()} partner` +
              `${partnerMap.size === 1 ? "" : "s"}.`,
            keyFields,
            sources: [resolved.resolutionUrl, payload.url].filter(Boolean),
            limitations: [
              `Only up to ${boundedMaxRecords.toLocaleString()} BioGRID rows were requested for this summary.`,
              "BioGRID interaction evidence mixes assay modalities and throughput regimes; physical vs genetic labels are broad evidence classes rather than direct mechanism claims.",
              "This complements IntAct: BioGRID is useful for wider experimental coverage, while IntAct emphasizes deeply curated molecular interaction records and detection methods.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_biogrid_interactions: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL: BioGRID ORCS CRISPR screen summaries
// ============================================
server.registerTool(
  "get_biogrid_orcs_gene_summary",
  {
    description:
      "Summarize BioGRID ORCS CRISPR-screen evidence for a gene across published screens. " +
      "Returns screen counts, hit status, top phenotypes/cell lines, methodologies, and representative screen summaries.",
    inputSchema: {
      query: z.string().describe("Gene symbol, alias, or Entrez gene ID (for example 'TP53', 'EGFR', or '7157')."),
      organismTaxId: z.number().optional().default(9606).describe("NCBI taxonomy ID filter for ORCS screens (default: 9606 for human)."),
      hitFilter: z.enum(["all", "yes", "no"]).optional().default("yes")
        .describe("Whether to keep only significant hits, only non-hits, or all returned score rows."),
      topScreenLimit: z.number().optional().default(6).describe("Maximum representative screens to summarize (1-12)."),
      topPhenotypeLimit: z.number().optional().default(5).describe("Maximum phenotype labels to summarize (1-10)."),
      topCellLineLimit: z.number().optional().default(5).describe("Maximum cell lines to summarize (1-10)."),
    },
  },
  async ({ query, organismTaxId = 9606, hitFilter = "yes", topScreenLimit = 6, topPhenotypeLimit = 5, topCellLineLimit = 5 }) => {
    const normalizedQuery = normalizeWhitespace(query || "");
    const boundedOrganismTaxId = Math.max(0, toNonNegativeInt(organismTaxId, 9606));
    const boundedScreenLimit = Math.max(1, Math.min(12, Math.round(topScreenLimit || 6)));
    const boundedPhenotypeLimit = Math.max(1, Math.min(10, Math.round(topPhenotypeLimit || 5)));
    const boundedCellLineLimit = Math.max(1, Math.min(10, Math.round(topCellLineLimit || 5)));

    if (!normalizedQuery) {
      return { content: [{ type: "text", text: "Provide a gene symbol, alias, or Entrez gene ID for BioGRID ORCS (for example TP53)." }] };
    }
    if (!BIOGRID_ORCS_ACCESS_KEY) {
      return buildBiogridMissingKeyResponse(
        "BioGRID ORCS",
        "BIOGRID_ORCS_ACCESS_KEY (or BIOGRID_ACCESS_KEY)",
        BIOGRID_ORCS_API,
        `get_biogrid_orcs_gene_summary(query="${normalizedQuery}")`
      );
    }

    try {
      const resolved = await resolveBiogridGeneSelection(normalizedQuery, boundedOrganismTaxId);
      const params = new URLSearchParams({
        format: "json",
        organismID: String(boundedOrganismTaxId),
        hit: hitFilter,
        max: "10000",
      });
      if (resolved.entrezGene) params.set("geneID", resolved.entrezGene);
      params.set("name", resolved.symbol || normalizedQuery);

      const payload = await fetchBiogridJson(BIOGRID_ORCS_API, "/genes/", params, {
        accessKey: BIOGRID_ORCS_ACCESS_KEY,
        retries: 1,
        timeoutMs: 20000,
        maxBackoffMs: 3000,
      });

      const rawRows = extractBiogridRows(payload.data);
      const scoreRows = rawRows.filter((row) => {
        const rowEntrez = normalizeWhitespace(row?.IDENTIFIER_ID || "");
        const rowSymbol = normalizeWhitespace(row?.OFFICIAL_SYMBOL || "").toUpperCase();
        if (resolved.entrezGene && rowEntrez === resolved.entrezGene) return true;
        if (resolved.symbol && rowSymbol === resolved.symbol.toUpperCase()) return true;
        return !resolved.entrezGene && !resolved.symbol;
      });

      if (scoreRows.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `BioGRID ORCS returned no CRISPR screen rows for "${normalizedQuery}".`,
              keyFields: [
                `Resolved query: ${resolved.symbol || normalizedQuery}${resolved.entrezGene ? ` | Entrez: ${resolved.entrezGene}` : ""}`,
                `Organism filter: ${boundedOrganismTaxId}`,
                `Hit filter: ${hitFilter}`,
              ],
              sources: [resolved.resolutionUrl, payload.url].filter(Boolean),
              limitations: [
                "ORCS search by official symbol is stricter than free-text literature search and may miss unresolved aliases.",
                "Try a canonical HGNC/MGI symbol or switch hitFilter to `all` if you expected sparse screen coverage.",
              ],
            }),
          }],
        };
      }

      const screenIds = dedupeArray(scoreRows.map((row) => normalizeWhitespace(row?.SCREEN_ID || "")).filter(Boolean));
      const { lookup: screenLookup, sourceUrls: screenSourceUrls } = await fetchBiogridOrcsScreenDetails(screenIds, BIOGRID_ORCS_ACCESS_KEY);

      const phenotypeCounts = new Map();
      const cellLineCounts = new Map();
      const methodologyCounts = new Map();
      const analysisCounts = new Map();
      const screenTypeCounts = new Map();
      const hitCounts = new Map();
      const representativeScreens = [];

      for (const row of scoreRows) {
        const screenId = normalizeWhitespace(row?.SCREEN_ID || "");
        const metadata = screenLookup.get(screenId) || {};
        const hitValue = normalizeWhitespace(row?.HIT || "UNKNOWN").toUpperCase() || "UNKNOWN";
        const phenotype = normalizeWhitespace(metadata?.PHENOTYPE || "");
        const cellLine = normalizeWhitespace(metadata?.CELL_LINE || "");
        const methodology = normalizeWhitespace(metadata?.METHODOLOGY || metadata?.LIBRARY_METHODOLOGY || "");
        const analysis = normalizeWhitespace(metadata?.ANALYSIS || "");
        const screenType = normalizeWhitespace(metadata?.SCREEN_TYPE || "");

        incrementCount(hitCounts, hitValue);
        if (phenotype) incrementCount(phenotypeCounts, phenotype);
        if (cellLine) incrementCount(cellLineCounts, cellLine);
        if (methodology) incrementCount(methodologyCounts, methodology);
        if (analysis) incrementCount(analysisCounts, analysis);
        if (screenType) incrementCount(screenTypeCounts, screenType);

        representativeScreens.push({
          screenId,
          hitValue,
          scoreRow: row,
          metadata,
          scoreSummaries: extractOrcsScoreSummaries(row, metadata, 3),
        });
      }

      const uniqueRepresentativeScreens = [];
      const seenScreens = new Set();
      for (const row of representativeScreens) {
        if (!row.screenId || seenScreens.has(row.screenId)) continue;
        seenScreens.add(row.screenId);
        uniqueRepresentativeScreens.push(row);
      }
      uniqueRepresentativeScreens.sort((a, b) => {
        const hitScoreA = a.hitValue === "YES" ? 2 : a.hitValue === "NO" ? 1 : 0;
        const hitScoreB = b.hitValue === "YES" ? 2 : b.hitValue === "NO" ? 1 : 0;
        if (hitScoreB !== hitScoreA) return hitScoreB - hitScoreA;
        return toNonNegativeInt(b.screenId, 0) - toNonNegativeInt(a.screenId, 0);
      });

      const keyFields = [
        `Resolved query: ${resolved.symbol || normalizedQuery}${resolved.name ? ` — ${resolved.name}` : ""}${resolved.entrezGene ? ` | Entrez: ${resolved.entrezGene}` : ""}`,
        `Organism filter: ${boundedOrganismTaxId} | Hit filter: ${hitFilter}`,
        resolved.matchedViaMyGene ? "Identifier grounding: MyGene.info resolved the query before BioGRID ORCS lookup." : "Identifier grounding: ORCS queried the original user string directly.",
        `ORCS score rows analyzed: ${scoreRows.length.toLocaleString()}`,
        `Unique screens: ${screenIds.length.toLocaleString()}`,
        `Hit breakdown: ${summarizeTopCounts(hitCounts, 3).join("; ") || "n/a"}`,
        `Top phenotypes: ${summarizeTopCounts(phenotypeCounts, boundedPhenotypeLimit).join("; ") || "n/a"}`,
        `Top cell lines: ${summarizeTopCounts(cellLineCounts, boundedCellLineLimit).join("; ") || "n/a"}`,
        `Top methodologies: ${summarizeTopCounts(methodologyCounts, 5).join("; ") || "n/a"}`,
      ];
      const analysisSummary = summarizeTopCounts(analysisCounts, 5);
      if (analysisSummary.length > 0) {
        keyFields.push(`Top analyses: ${analysisSummary.join("; ")}`);
      }
      const screenTypeSummary = summarizeTopCounts(screenTypeCounts, 5);
      if (screenTypeSummary.length > 0) {
        keyFields.push(`Top screen types: ${screenTypeSummary.join("; ")}`);
      }
      if (uniqueRepresentativeScreens.length > 0) {
        keyFields.push("Representative ORCS screens:");
        keyFields.push(
          ...uniqueRepresentativeScreens.slice(0, boundedScreenLimit).map((entry, idx) => {
            const metadata = entry.metadata || {};
            const scoreText = entry.scoreSummaries.length > 0 ? ` | ${entry.scoreSummaries.join(" | ")}` : "";
            return (
              `${idx + 1}. ${normalizeWhitespace(metadata?.SCREEN_NAME || `Screen ${entry.screenId}`)}` +
              ` | phenotype: ${normalizeWhitespace(metadata?.PHENOTYPE || "n/a")}` +
              ` | cell line: ${normalizeWhitespace(metadata?.CELL_LINE || "n/a")}` +
              ` | methodology: ${normalizeWhitespace(metadata?.METHODOLOGY || metadata?.LIBRARY_METHODOLOGY || "n/a")}` +
              ` | analysis: ${normalizeWhitespace(metadata?.ANALYSIS || "n/a")}` +
              ` | hit: ${entry.hitValue}` +
              `${normalizeWhitespace(metadata?.SOURCE_ID || "") ? ` | PMID ${normalizeWhitespace(metadata.SOURCE_ID)}` : ""}` +
              `${scoreText}`
            );
          })
        );
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary:
              `BioGRID ORCS found ${scoreRows.length.toLocaleString()} score row` +
              `${scoreRows.length === 1 ? "" : "s"} for ${resolved.symbol || normalizedQuery} across ${screenIds.length.toLocaleString()} screen` +
              `${screenIds.length === 1 ? "" : "s"}.`,
            keyFields,
            sources: [resolved.resolutionUrl, payload.url, ...screenSourceUrls.slice(0, 3)].filter(Boolean),
            limitations: [
              "ORCS score columns are screen-specific; CERES, Bayes factors, FDRs, and other metrics are not directly comparable across all screens.",
              "Hit labels and significance criteria are defined per screen and should be interpreted alongside the screen metadata.",
              "This complements DepMap: ORCS emphasizes published screen-level evidence with phenotype and cell-line context, whereas DepMap focuses on release-level dependency summaries.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_biogrid_orcs_gene_summary: ${error.message}` }] };
    }
  }
);

// ============================================
// TOOL 27: Summarize clinical-trial landscape
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
      const normalizedStatus = normalizeClinicalTrialStatus(status);
      const boundedStudies = Math.max(10, Math.min(200, Math.round(maxStudies)));
      const boundedPages = Math.max(1, Math.min(8, Math.round(maxPages)));
      const collected = await collectClinicalTrialStudies({
        query,
        normalizedStatus,
        maxStudies: boundedStudies,
        maxPagesPerVariant: boundedPages,
        fetchOptions: { retries: 2, timeoutMs: 15000, maxBackoffMs: 3500 },
      });
      const studies = collected.studies;
      const sources = collected.sources;
      const totalCount = collected.totalCount;

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
      const hasMore = Boolean(collected.hasMorePages);

      const countQualifier = Number.isFinite(totalCount)
        ? ` (reported total in registry: ${totalCount})`
        : " (registry total not provided; count reflects only studies fetched within pagination limits — more may exist)";
      const keyFields = [
        `Query: ${query}`,
        `Studies analyzed: ${analyzed}${countQualifier}`,
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
      if (collected.usedFallbackVariant) {
        keyFields.push("Query normalization fallback recovered additional matches from a punctuation-normalized variant of the same search.");
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
          ...(collected.usedFallbackVariant
            ? ["A punctuation-normalized query variant was also scanned to recover additional semantically equivalent matches."]
            : []),
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
                ...(!Number.isFinite(totalCount) && analyzed > 0
                  ? ["Total matching studies in the registry was not provided by the API; the count reflects only fetched studies — more may exist."]
                  : []),
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
      const rawResults = Array.isArray(data?.results) ? data.results : [];
      const rankedResults = rankUniProtResults(rawResults, normalizedQuery);
      let results = rankedResults;
      if (looksLikeBareGeneSymbol(normalizedQuery)) {
        const exactishMatches = rankedResults.filter((entry) => scoreUniProtSearchResult(entry, normalizedQuery) >= 100);
        if (exactishMatches.length > 0) {
          results = exactishMatches;
        }
      }
      results = results.slice(0, boundedLimit);

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
      "Get a concise UniProt protein profile by accession, including function, localization, sequence facts, disease-linked natural variants, and major cross-references.",
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
      const diseaseVariantAnnotations = extractUniProtDiseaseVariantAnnotations(record, 6, 8);
      const ensemblGeneIds = extractUniProtEnsemblGeneIds(record, 8);
      const pdbIds = extractUniProtCrossRefs(record, "PDB", 8);
      const reactomeIds = extractUniProtCrossRefs(record, "Reactome", 8);
      const featureSummary = summarizeUniProtFeatureTypes(record, 10);
      const lastUpdate = normalizeWhitespace(record?.entryAudit?.lastAnnotationUpdateDate || "");
      const diseaseVariantPositionSummary = diseaseVariantAnnotations
        .map((item) => `${item.label}: ${item.positions.join(", ")}`)
        .filter(Boolean);
      const diseaseVariantNotationSummary = diseaseVariantAnnotations
        .map((item) => {
          const variantBits = item.variants
            .map((variant) => {
              const notation = normalizeWhitespace(variant?.notation || "");
              const dbsnpId = normalizeWhitespace(variant?.dbsnpId || "");
              if (notation && dbsnpId) return `${notation} (${dbsnpId})`;
              return notation || dbsnpId;
            })
            .filter(Boolean)
            .slice(0, 4);
          return variantBits.length > 0 ? `${item.label}: ${variantBits.join(", ")}` : "";
        })
        .filter(Boolean);

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
        ...(diseaseVariantPositionSummary.length > 0
          ? [`Disease-associated natural variant positions: ${diseaseVariantPositionSummary.join(" | ")}`]
          : []),
        ...(diseaseVariantNotationSummary.length > 0
          ? [`Disease-associated variant examples: ${diseaseVariantNotationSummary.join(" | ")}`]
          : []),
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
        structuredContent: {
          accession: primaryAccession,
          uniprotId: entryId || null,
          diseaseVariantAnnotations,
        },
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
      "and AlphaMissense predictions. Use for assessing pathogenicity of specific mutations. " +
      "REQUIRES variant-level input: HGVS notation (e.g. BRAF:p.Val600Glu, chr2:g.166210844T>C). Does NOT accept gene symbols. " +
      "Accepts both hg19 (from MyVariant) and hg38 coordinates — auto-detects assembly. " +
      "If only a gene is known, first use search_variants_by_gene or search_civic_variants to discover variants.",
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
    const normalizeHgvs = (v) => {
      const s = normalizeWhitespace(v);
      const ncMatch = s.match(/^NC_0*(\d{2})\.\d+:(.+)$/);
      if (ncMatch) {
        const num = parseInt(ncMatch[1], 10);
        const rest = ncMatch[2];
        const chr = num <= 22 ? `chr${num}` : num === 23 ? "chrX" : num === 24 ? "chrY" : null;
        if (chr) return `${chr}:${rest}`;
      }
      return s;
    };
    const cleaned = (variants || []).map(normalizeHgvs).filter(Boolean).slice(0, 10);
    if (cleaned.length === 0) {
      return { content: [{ type: "text", text: "Provide at least one variant in HGVS notation." }] };
    }

    const looksLikeGeneSymbol = (s) => {
      const token = s.split(/\s+/)[0] || s;
      return (
        token.length >= 2 &&
        token.length <= 20 &&
        /^[A-Z0-9][A-Za-z0-9_-]*$/i.test(token) &&
        !/^rs\d+$/i.test(token) &&
        !/[>:]|\.g\.|\.p\.|\.c\./.test(token)
      );
    };
    const getGeneSymbol = (s) => (s.split(/\s+/)[0] || s).toUpperCase();

    const geneLike = cleaned.filter(looksLikeGeneSymbol);
    if (geneLike.length > 0) {
      const gene = getGeneSymbol(geneLike[0]);
      const geneQuery = `dbnsfp.genename:${gene}`;
      const geneParams = new URLSearchParams({
        q: geneQuery,
        fields: "_id,dbsnp.rsid,cadd.consequence,gnomad_exome.af",
        size: "15",
      });
      const geneUrl = `${MYVARIANT_API}/query?${geneParams}`;
      const geneData = await fetchJsonWithRetry(geneUrl, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });
      const geneHits = Array.isArray(geneData?.hits) ? geneData.hits : [];
      if (geneHits.length > 0) {
        const keyFields = geneHits.map((h) => {
          const hgvs = normalizeWhitespace(h._id || "");
          const rsid = normalizeWhitespace(h.dbsnp?.rsid || "");
          const cons = normalizeWhitespace(h.cadd?.consequence || "");
          return [hgvs, rsid ? `rsID: ${rsid}` : "", cons].filter(Boolean).join(" | ");
        });
        return {
          content: [
            {
              type: "text",
              text: renderStructuredResponse({
                summary: `Input "${gene}" appears to be a gene symbol. VEP requires HGVS notation. Found ${geneHits.length} variant(s) in that gene—call annotate_variants_vep(variants=["chrX:g...."]) with one of the HGVS _id below.`,
                keyFields,
                sources: [geneUrl],
                limitations: [
                  "Use the HGVS _id (e.g. chr2:g.165310468G>A) from above with annotate_variants_vep for SIFT/PolyPhen/AlphaMissense.",
                ],
              }),
            },
          ],
        };
      }
    }

    // MyVariant.info returns hg19 coordinates by default (e.g. chr2:g.166210844T>C).
    // Ensembl VEP REST API defaults to GRCh38 and will reject hg19 coords with ref-allele mismatches.
    // Detect hg19-style chr coords and use the GRCh37 endpoint; try GRCh38 first for NC_ / protein HGVS.
    const hasGenomicChrCoord = cleaned.some((v) => /^chr\d+:g\.\d+/i.test(v));
    const hasProteinOrGeneHgvs = cleaned.some((v) => /^[A-Z][A-Za-z0-9]+:p\./i.test(v) || /^[A-Z][A-Za-z0-9]+:c\./i.test(v));

    const body = { hgvs_notations: cleaned };
    if (includeAlphaMissense) body.AlphaMissense = 1;

    // Use GRCh37 endpoint for hg19 genomic coords (from MyVariant), GRCh38 for protein/coding HGVS
    const useGrch37 = hasGenomicChrCoord && !hasProteinOrGeneHgvs;
    const vepBase = useGrch37 ? "https://grch37.rest.ensembl.org" : ENSEMBL_REST_API;
    const url = `${vepBase}/vep/human/hgvs`;
    let data;
    try {
      data = await fetchJsonWithRetry(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify(body),
        retries: 1,
        timeoutMs: 20000,
        maxBackoffMs: 3000,
      });
    } catch (err) {
      // If ref-allele mismatch on GRCh38, retry with GRCh37 (likely hg19 coords)
      if (!useGrch37 && /reference allele/i.test(String(err?.message || ""))) {
        const fallbackUrl = `https://grch37.rest.ensembl.org/vep/human/hgvs`;
        data = await fetchJsonWithRetry(fallbackUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(body),
          retries: 1,
          timeoutMs: 20000,
          maxBackoffMs: 3000,
        });
      } else {
        throw err;
      }
    }

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
      limit: z.coerce.number().optional().default(10).describe("Max variants to return (default 10, max 25)"),
    },
  },
  async ({ gene, variantName, limit = 10 }) => {
    const normalizedGene = normalizeWhitespace(gene || "").toUpperCase();
    if (!normalizedGene) {
      return { content: [{ type: "text", text: "Provide a gene symbol (e.g. BRAF)." }] };
    }
    const baseLimit = Math.max(1, Math.min(25, Math.round(Number(limit) || 10)));

    let variants = [];
    let totalCount = 0;
    let filtered = [];

    if (variantName) {
      const normalizedVariant = normalizeWhitespace(variantName);
      const browseQuery = `
        query BrowseCivicByVariant($featureName: String!, $variantName: String!) {
          browseMolecularProfiles(
            featureName: $featureName
            variantName: $variantName
            first: 15
          ) {
            nodes {
              id
              name
              evidenceItemCount
            }
          }
        }
      `;
      const browseData = await fetchCivicGraphQL(browseQuery, {
        featureName: normalizedGene,
        variantName: normalizedVariant,
      });
      const profiles = browseData?.browseMolecularProfiles?.nodes || [];
      if (profiles.length === 0) {
        const genesQuery = `
          query GenesFallback($entrezSymbols: [String!]!, $first: Int) {
            genes(entrezSymbols: $entrezSymbols, first: 1) {
              nodes {
                variants(first: $first) { totalCount nodes { id name molecularProfiles(first: 5) {
                  nodes { evidenceItems(first: 5) { nodes { id status evidenceLevel evidenceDirection significance therapies { name } disease { name } source { citation } } }
                } } }
              }
            }
          }
        }
        `;
        const fallbackData = await fetchCivicGraphQL(genesQuery, {
          entrezSymbols: [normalizedGene],
          first: 100,
        });
        const geneNode = fallbackData?.genes?.nodes?.[0];
        variants = geneNode?.variants?.nodes || [];
        totalCount = geneNode?.variants?.totalCount || 0;
        filtered = variants.filter((v) =>
          normalizeWhitespace(v.name || "").toLowerCase().includes(normalizedVariant.toLowerCase())
        );
      } else {
        const topIds = profiles
          .sort((a, b) => (b.evidenceItemCount || 0) - (a.evidenceItemCount || 0))
          .slice(0, 5)
          .map((p) => p.id);
        const frag = `
          id name evidenceItems(first: 8) {
            nodes { id status evidenceLevel evidenceDirection significance therapies { name } disease { name } source { citation } }
          }
        `;
        const aliasQueries = topIds.map((id, i) => `p${i}: molecularProfile(id: ${id}) { ${frag} }`).join("\n");
        const detailQuery = `query { ${aliasQueries} }`;
        const detailData = await fetchCivicGraphQL(detailQuery);
        const profileNodes = topIds.map((_, i) => detailData?.[`p${i}`]).filter(Boolean);
        variants = profileNodes.map((p) => ({
          id: p.id,
          name: p.name,
          molecularProfiles: { nodes: [{ evidenceItems: p.evidenceItems }] },
        }));
        totalCount = profiles.length;
        filtered = variants;
      }
    } else {
      const boundedLimit = baseLimit;
      const query = `
        query SearchCivicVariants($entrezSymbols: [String!]!, $first: Int) {
          genes(entrezSymbols: $entrezSymbols, first: 1) {
            nodes {
              id
              name
              variants(first: $first) {
                totalCount
                nodes {
                  id
                  name
                  molecularProfiles(first: 5) {
                    nodes {
                      id
                      name
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
          }
        }
      `;
      const data = await fetchCivicGraphQL(query, {
        entrezSymbols: [normalizedGene],
        first: boundedLimit,
      });
      const geneNode = data?.genes?.nodes?.[0];
      variants = geneNode?.variants?.nodes || [];
      totalCount = geneNode?.variants?.totalCount || 0;
      filtered = variants;
    }

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
      query GeneDetail($entrezSymbols: [String!]!) {
        genes(entrezSymbols: $entrezSymbols, first: 1) {
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

    const data = await fetchCivicGraphQL(query, { entrezSymbols: [normalizedName] });
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
// MyVariant.info — gene-level variant discovery (bridge to variant-level tools)
// ---------------------------------------------------------------------------

server.registerTool(
  "search_variants_by_gene",
  {
    description:
      "Search MyVariant.info for variants in a gene by symbol. Returns hg19 HGVS IDs and basic annotations (ClinVar, gnomAD AF) " +
      "for variants that can then be passed to get_variant_annotations or annotate_variants_vep (both accept hg19 coords). " +
      "Use when only a gene symbol is known (e.g. SCN2A) and variant coordinates are needed for downstream tools. " +
      "Supports optional consequence filter (e.g. missense) to narrow results.",
    inputSchema: {
      gene: z.string().describe("Gene symbol (e.g. SCN2A, BRAF, TP53)"),
      consequenceFilter: z
        .string()
        .optional()
        .describe("Optional: filter by consequence (e.g. missense, synonymous, nonsense, frameshift, splice). 'missense' maps to CADD NON_SYNONYMOUS."),
      limit: z.coerce.number().optional().default(25).describe("Max variants to return (default 25, max 100)"),
    },
  },
  async ({ gene, consequenceFilter, limit = 25 }) => {
    const normalizedGene = normalizeWhitespace(gene || "").toUpperCase();
    if (!normalizedGene) {
      return { content: [{ type: "text", text: "Provide a gene symbol (e.g. SCN2A)." }] };
    }
    const boundedLimit = Math.max(1, Math.min(100, Math.round(Number(limit) || 25)));

    const consequenceAliases = {
      missense: "NON_SYNONYMOUS",
      nonsynonymous: "NON_SYNONYMOUS",
      non_synonymous: "NON_SYNONYMOUS",
      synonymous: "SYNONYMOUS",
      stop_gained: "STOP_GAINED",
      nonsense: "STOP_GAINED",
      frameshift: "FRAME_SHIFT",
      splice: "SPLICE_SITE",
      splice_site: "SPLICE_SITE",
    };

    let query = `dbnsfp.genename:${normalizedGene}`;
    if (consequenceFilter) {
      const cf = normalizeWhitespace(consequenceFilter).toLowerCase().replace(/["\\]/g, "").replace(/\s+/g, "_");
      const mapped = consequenceAliases[cf] || cf.toUpperCase();
      if (mapped) query += ` AND cadd.consequence:${mapped}`;
    }

    const fields = "_id,dbsnp.rsid,dbsnp.gene,cadd.consequence,clinvar.rcv,gnomad_exome.af,gnomad_genome.af,dbnsfp.sift,dbnsfp.polyphen2";
    const params = new URLSearchParams({
      q: query,
      fields,
      size: String(boundedLimit),
    });
    const url = `${MYVARIANT_API}/query?${params}`;
    const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 15000, maxBackoffMs: 3000 });

    const hits = Array.isArray(data?.hits) ? data.hits : [];
    const total = Number(data?.total) ?? 0;

    const keyFields = hits.map((h) => {
      const hgvs = normalizeWhitespace(h._id || "");
      const rsid = normalizeWhitespace(h.dbsnp?.rsid || "");
      const consequence = normalizeWhitespace(h.cadd?.consequence || "");
      let af = "";
      if (h.gnomad_exome?.af != null) af = `gnomAD exome AF: ${Number(h.gnomad_exome.af).toExponential(2)}`;
      else if (h.gnomad_genome?.af != null) af = `gnomAD genome AF: ${Number(h.gnomad_genome.af).toExponential(2)}`;
      const parts = [hgvs];
      if (rsid) parts.push(`rsID: ${rsid}`);
      if (consequence) parts.push(consequence);
      if (af) parts.push(af);
      return parts.join(" | ");
    });

    if (hits.length === 0) {
      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `No variants found for gene ${normalizedGene} in MyVariant.info.`,
              keyFields: [`Gene: ${normalizedGene}`, `Total matching: ${total}`],
              sources: [url],
              limitations: [
                "Check gene symbol spelling. Use resolve_gene_identifiers first if uncertain.",
                "Some genes may have limited variant coverage in dbNSFP.",
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
            summary: `Found ${hits.length} variant${hits.length === 1 ? "" : "s"} for ${normalizedGene}${total > boundedLimit ? ` (${total} total, showing ${boundedLimit})` : ""}. Use get_variant_annotations with the HGVS _id or rsID for full annotations.`,
            keyFields,
            sources: [url, "https://myvariant.info/"],
            limitations: [
              "Results include HGVS IDs (_id) that can be passed to get_variant_annotations or annotate_variants_vep.",
              "gnomAD AF is hg19. For rare-disease genes, many variants may have low/absent population frequency.",
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
      "dbSNP, CADD, gnomAD, COSMIC, and more. " +
      "Accepts rsID (e.g. rs113488022), genomic HGVS (e.g. chr2:g.165310413C>T or NC_000002.12:g.165310413C>T), " +
      "gene+protein HGVS such as KRAS:p.Gly12Cys, or shorthand like KRAS G12C. " +
      "Does NOT accept gene symbols. If only a gene is known, use search_variants_by_gene first. " +
      "gnomAD fields: gnomad_exome, gnomad_genome (not gnomad_genomes).",
    inputSchema: {
      variantId: z
        .string()
        .describe('Variant identifier: rsID, genomic HGVS, gene+protein HGVS, or shorthand like "KRAS G12C" (e.g. "rs113488022", "chr7:g.140753336A>T", "KRAS:p.Gly12Cys")'),
      fields: z
        .string()
        .optional()
        .describe("Comma-separated fields to return (e.g. clinvar,cadd,dbsnp,gnomad_exome). Default returns key fields."),
    },
  },
  async ({ variantId, fields }) => {
    const rawId = variantId ?? "";
    const normalizedId = normalizeWhitespace(String(rawId));
    if (!normalizedId) {
      return { content: [{ type: "text", text: 'Provide variantId (rsID or HGVS notation, e.g. "rs113488022" or "chr2:g.165310413C>T").' }] };
    }

    const defaultFields = "clinvar,cadd,dbsnp,gnomad_exome,gnomad_genome,cosmic,dbnsfp,snpeff";
    let requestFields = normalizeWhitespace(fields || "") || defaultFields;
    if (/gnomad_genomes/i.test(requestFields)) {
      requestFields = requestFields.replace(/gnomad_genomes/g, "gnomad_genome");
    }

    const proteinHgvsMatch = normalizedId.match(/^([A-Za-z0-9_-]+)[:\s]+(p\.[A-Za-z][A-Za-z0-9*]+)$/i);
    const proteinShorthandMatch = normalizedId.match(/^([A-Za-z0-9_-]+)[:\s]+([A-Za-z]\d+[A-Za-z*])$/i);
    const codingHgvsMatch = normalizedId.match(/^([A-Za-z0-9_-]+)[:\s]+(c\.\d+[ACGT]>[ACGT])$/i);
    const codingShorthandMatch = normalizedId.match(/^([A-Za-z0-9_-]+)[:\s]+(\d+[ACGT]>[ACGT])$/i);
    const proteinQueryMeta = proteinHgvsMatch
      ? { gene: proteinHgvsMatch[1].toUpperCase(), protein: proteinHgvsMatch[2] }
      : proteinShorthandMatch
        ? { gene: proteinShorthandMatch[1].toUpperCase(), protein: `p.${proteinShorthandMatch[2]}` }
        : null;
    const codingQueryMeta = codingHgvsMatch
      ? {
          gene: codingHgvsMatch[1].toUpperCase(),
          coding: `c.${codingHgvsMatch[2].replace(/^c\./i, "").toUpperCase()}`,
          shorthand: codingHgvsMatch[2].replace(/^c\./i, "").toUpperCase(),
        }
      : codingShorthandMatch
        ? {
            gene: codingShorthandMatch[1].toUpperCase(),
            coding: `c.${codingShorthandMatch[2].toUpperCase()}`,
            shorthand: codingShorthandMatch[2].toUpperCase(),
          }
        : null;

    const extractHitGeneValues = (hit) =>
      dedupeArray(
        [
          ...asArray(hit?.dbnsfp?.genename),
          normalizeWhitespace(hit?.clinvar?.gene?.symbol || ""),
          normalizeWhitespace(hit?.dbsnp?.gene?.symbol || ""),
        ]
          .map((value) => normalizeWhitespace(value).toUpperCase())
          .filter(Boolean)
      );

    const extractHitCodingValues = (hit) =>
      dedupeArray(
        [
          ...asArray(hit?.dbnsfp?.hgvsc),
          ...asArray(hit?.clinvar?.hgvs?.coding),
          ...asArray(hit?.dbsnp?.gene?.rnas).map((entry) => normalizeWhitespace(entry?.hgvs || "")),
        ]
          .map((value) => normalizeWhitespace(value))
          .filter(Boolean)
      );

    const extractHitProteinValues = (hit) =>
      dedupeArray(
        [
          ...asArray(hit?.dbnsfp?.hgvsp),
          ...asArray(hit?.clinvar?.hgvs?.protein),
        ]
          .map((value) => normalizeWhitespace(value))
          .filter(Boolean)
      );

    const scoreProteinQueryHit = (hit, { gene, protein }) => {
      const hitGeneValues = extractHitGeneValues(hit);
      const hitProteinValues = extractHitProteinValues(hit);
      let score = 0;
      if (hitGeneValues.includes(gene)) score += 6;
      if (hitProteinValues.includes(protein)) score += 8;
      if (/^chr[\w]+:g\.\d+[ACGT]>[ACGT]$/i.test(normalizeWhitespace(hit?._id || ""))) score += 2;
      return score;
    };

    const scoreCodingQueryHit = (hit, { gene, coding, shorthand }) => {
      const hitGeneValues = extractHitGeneValues(hit);
      const hitCodingValues = extractHitCodingValues(hit);
      let score = 0;
      if (hitGeneValues.includes(gene)) score += 6;
      if (hitCodingValues.includes(coding)) score += 10;
      if (shorthand && hitCodingValues.some((value) => value.endsWith(`:${coding}`) || value.endsWith(shorthand))) score += 4;
      if (/^chr[\w]+:g\.\d+[ACGT]>[ACGT]$/i.test(normalizeWhitespace(hit?._id || ""))) score += 2;
      return score;
    };

    const fetchProteinHgvsHit = async ({ gene, protein }) => {
      const searchQueries = [
        `dbnsfp.genename:${gene} AND dbnsfp.hgvsp:${protein}`,
        `${gene} ${protein}`,
      ];
      for (const queryText of searchQueries) {
        const params = new URLSearchParams({
          q: queryText,
          fields: `_id,${requestFields}`,
          size: "10",
        });
        const queryUrl = `${MYVARIANT_API}/query?${params}`;
        const queryData = await fetchJsonWithRetry(queryUrl, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });
        const queryHits = Array.isArray(queryData?.hits) ? queryData.hits : [];
        if (queryHits.length === 0) continue;
        const ranked = [...queryHits].sort(
          (a, b) => scoreProteinQueryHit(b, { gene, protein }) - scoreProteinQueryHit(a, { gene, protein })
        );
        return { hit: ranked[0], url: queryUrl };
      }
      return { hit: null, url: `${MYVARIANT_API}/query?q=${encodeURIComponent(`${gene} ${protein}`)}` };
    };

    const fetchCodingHgvsHit = async ({ gene, coding, shorthand }) => {
      const searchQueries = dedupeArray([
        `dbnsfp.genename:${gene} AND ${shorthand}`,
        `dbnsfp.genename:${gene} AND ${coding}`,
        `${gene} ${coding}`,
        `${gene} ${shorthand}`,
      ]);
      for (const queryText of searchQueries) {
        const params = new URLSearchParams({
          q: queryText,
          fields: `_id,${requestFields}`,
          size: "10",
        });
        const queryUrl = `${MYVARIANT_API}/query?${params}`;
        const queryData = await fetchJsonWithRetry(queryUrl, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });
        const queryHits = Array.isArray(queryData?.hits) ? queryData.hits : [];
        if (queryHits.length === 0) continue;
        const ranked = [...queryHits].sort(
          (a, b) => scoreCodingQueryHit(b, { gene, coding, shorthand }) - scoreCodingQueryHit(a, { gene, coding, shorthand })
        );
        return { hit: ranked[0], url: queryUrl };
      }
      return { hit: null, url: `${MYVARIANT_API}/query?q=${encodeURIComponent(`${gene} ${coding}`)}` };
    };

    let resolvedId = normalizedId;
    let useHg38 = false;
    const ncMatch = normalizedId.match(/^NC_0*(\d{2})\.\d+:(.+)$/);
    if (ncMatch) {
      const num = parseInt(ncMatch[1], 10);
      const rest = ncMatch[2];
      const chr = num <= 22 ? `chr${num}` : num === 23 ? "chrX" : num === 24 ? "chrY" : null;
      if (chr) {
        resolvedId = `${chr}:${rest}`;
        useHg38 = true;
      }
    }

    const isRsId = /^rs\d+$/i.test(resolvedId);
    let url = "";
    let hit = null;
    if (proteinQueryMeta) {
      const proteinResolution = await fetchProteinHgvsHit(proteinQueryMeta);
      hit = proteinResolution.hit;
      url = proteinResolution.url;
    } else if (codingQueryMeta) {
      const codingResolution = await fetchCodingHgvsHit(codingQueryMeta);
      hit = codingResolution.hit;
      url = codingResolution.url;
    } else if (isRsId) {
      const params = new URLSearchParams({ q: `dbsnp.rsid:${resolvedId}`, fields: requestFields, size: "1" });
      url = `${MYVARIANT_API}/query?${params}`;
      const data = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });
      hit = Array.isArray(data?.hits) ? data.hits[0] : null;
    } else {
      const params = new URLSearchParams({ fields: requestFields });
      if (useHg38) params.set("assembly", "hg38");
      url = `${MYVARIANT_API}/variant/${encodeURIComponent(resolvedId)}?${params}`;

      try {
        hit = await fetchJsonWithRetry(url, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });
      } catch (err) {
        const errorMessage = String(err?.message || "");
        if (useHg38 && /404/.test(errorMessage)) {
          const paramsHg19 = new URLSearchParams({ fields: requestFields });
          const urlHg19 = `${MYVARIANT_API}/variant/${encodeURIComponent(resolvedId)}?${paramsHg19}`;
          try {
            hit = await fetchJsonWithRetry(urlHg19, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });
            url = urlHg19;
            useHg38 = false;
          } catch (fallbackErr) {
            if (/404/.test(String(fallbackErr?.message || ""))) {
              hit = null;
            } else {
              throw fallbackErr;
            }
          }
        } else if (/404/.test(errorMessage)) {
          hit = null;
        } else {
          throw err;
        }
      }
    }

    if (!hit || hit.notfound) {
      // If input looks like a gene symbol, try search_variants_by_gene logic to return useful results
      const firstToken = normalizedId.split(/\s+/)[0] || normalizedId;
      const looksLikeGeneSymbol =
        firstToken.length >= 2 &&
        firstToken.length <= 20 &&
        /^[A-Z0-9][A-Za-z0-9_-]*$/i.test(firstToken) &&
        !/^rs\d+$/i.test(firstToken) &&
        !/[>:]|\.g\.|\.p\.|\.c\./.test(firstToken);

      if (looksLikeGeneSymbol) {
        const geneSymbol = firstToken.toUpperCase();
        const geneQuery = `dbnsfp.genename:${geneSymbol}`;
        const geneParams = new URLSearchParams({
          q: geneQuery,
          fields: "_id,dbsnp.rsid,dbsnp.gene,cadd.consequence,gnomad_exome.af,gnomad_genome.af",
          size: "15",
        });
        const geneUrl = `${MYVARIANT_API}/query?${geneParams}`;
        const geneData = await fetchJsonWithRetry(geneUrl, { retries: 1, timeoutMs: 12000, maxBackoffMs: 3000 });
        const geneHits = Array.isArray(geneData?.hits) ? geneData.hits : [];
        const total = Number(geneData?.total) ?? 0;

        if (geneHits.length > 0) {
          const keyFields = geneHits.map((h) => {
            const hgvs = normalizeWhitespace(h._id || "");
            const rsid = normalizeWhitespace(h.dbsnp?.rsid || "");
            const cons = normalizeWhitespace(h.cadd?.consequence || "");
            let af = "";
            if (h.gnomad_exome?.af != null) af = `gnomAD AF: ${Number(h.gnomad_exome.af).toExponential(2)}`;
            else if (h.gnomad_genome?.af != null) af = `gnomAD AF: ${Number(h.gnomad_genome.af).toExponential(2)}`;
            const parts = [hgvs];
            if (rsid) parts.push(`rsID: ${rsid}`);
            if (cons) parts.push(cons);
            if (af) parts.push(af);
            return parts.join(" | ");
          });
          return {
            content: [
              {
                type: "text",
                text: renderStructuredResponse({
                  summary: `Input "${normalizedId}" appears to be a gene symbol. Found ${geneHits.length} variant(s) in gene ${geneSymbol}. Call get_variant_annotations(variantId="...") with one of the HGVS _id or rsID below for full annotations.`,
                  keyFields,
                  sources: [geneUrl],
                  limitations: [
                    "Pass the HGVS _id (e.g. chr2:g.165310468G>A) or rsID as variantId for full ClinVar/gnomAD/CADD annotations.",
                    "For VEP predictions, use annotate_variants_vep with the same HGVS ID.",
                  ],
                }),
              },
            ],
          };
        }
      }

      return {
        content: [
          {
            type: "text",
            text: renderStructuredResponse({
              summary: `Variant ${normalizedId} not found in MyVariant.info.`,
              keyFields: [`Queried: ${normalizedId}`],
              sources: [url],
              limitations: [
                'Check variant format. Examples: "rs113488022", "chr7:g.140753336A>T", "KRAS:p.Gly12Cys", "KRAS G12C".',
                "If this is a gene symbol, use search_variants_by_gene(gene='...') first to discover variants.",
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
      const clinvarEntries = asArray(cv.rcv);
      const sig = dedupeArray(
        clinvarEntries.map((entry) => normalizeWhitespace(entry?.clinical_significance || "")).filter(Boolean)
      ).join("; ");
      if (sig) keyFields.push(`ClinVar significance: ${sig}`);
      const conditions = dedupeArray(
        clinvarEntries
          .map((entry) =>
            normalizeWhitespace(
              entry?.conditions?.name
              || entry?.conditions?.identifiers?.medgen
              || ""
            )
          )
          .filter(Boolean)
      ).slice(0, 3);
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

server.registerTool(
  "get_alphafold_domain_plddt",
  {
    description:
      "Compute domain-level mean pLDDT values from AlphaFold models for a UniProt accession. Supports the latest AlphaFold model via API and historical archive-backed versions such as v4, then combines the model with UniProt feature annotations (topological domains, transmembrane regions, signal peptides, and a few clearly-labeled derived regions).",
    inputSchema: {
      uniprotId: z.string().describe("UniProt accession (for example P02786 for TFRC)."),
      version: z.string().optional().describe("AlphaFold model version such as 'latest', 'v4', or '6'. Default 'latest'."),
      domains: z.array(z.string()).optional().describe("Optional domain labels to prioritize in the response, for example ['signal peptide', 'extracellular', 'transmembrane', 'cytoplasmic']."),
    },
  },
  async ({ uniprotId, version, domains }) => {
    const normalizedId = normalizeWhitespace(uniprotId || "").toUpperCase();
    if (!normalizedId) {
      return { content: [{ type: "text", text: "Provide a UniProt accession (for example P02786)." }] };
    }

    try {
      const helperResult = await runPythonJsonHelper(
        ALPHAFOLD_DOMAIN_QUERY_SCRIPT,
        {
          uniprotId: normalizedId,
          version: normalizeWhitespace(version || "latest") || "latest",
          domains: Array.isArray(domains) ? domains : [],
        },
        { timeoutMs: 240000 }
      );

      if (normalizeWhitespace(helperResult?.status || "") !== "ok") {
        throw new Error(normalizeWhitespace(helperResult?.error || "AlphaFold domain pLDDT helper failed."));
      }

      const requestedRows = Array.isArray(helperResult?.requested_domain_means) ? helperResult.requested_domain_means : [];
      const domainRows = Array.isArray(helperResult?.domain_means) ? helperResult.domain_means : [];
      const rowsToReport = requestedRows.length > 0 ? requestedRows : domainRows;
      const versionLabel = helperResult?.version != null ? `v${helperResult.version}` : String(version || "latest");
      const keyFields = [
        `Entry: ${normalizeWhitespace(helperResult?.entry_id || `AF-${normalizedId}-F1`)}`,
        `UniProt: [${normalizedId}](https://www.uniprot.org/uniprotkb/${normalizedId})`,
        `Model version: ${versionLabel}`,
      ];
      if (helperResult?.global_plddt != null) {
        keyFields.push(`Global pLDDT (latest API summary): ${Number(helperResult.global_plddt).toFixed(1)}`);
      }
      keyFields.push(
        "\nDomain pLDDT means:",
        ...rowsToReport.map((row) => {
          const label = normalizeWhitespace(row?.label || "domain");
          const description = normalizeWhitespace(row?.description || "");
          const start = Number(row?.start || 0);
          const end = Number(row?.end || 0);
          const meanPlddt = row?.mean_plddt != null ? Number(row.mean_plddt).toFixed(1) : "n/a";
          const atomMean = row?.all_atom_mean_plddt != null ? Number(row.all_atom_mean_plddt).toFixed(1) : "n/a";
          const derivedTag = row?.derived ? "derived" : "annotated";
          return `${label}: ${meanPlddt} pLDDT (all-atom mean ${atomMean}; residues ${start}-${end}; ${derivedTag}${description ? `; ${description}` : ""})`;
        })
      );

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Computed AlphaFold domain-level pLDDT values for ${normalizedId} using ${versionLabel}.`,
            keyFields,
            sources: [
              normalizeWhitespace(helperResult?.model_source || ""),
              `https://alphafold.ebi.ac.uk/entry/${normalizeWhitespace(helperResult?.entry_id || `AF-${normalizedId}-F1`)}`,
              `https://www.uniprot.org/uniprotkb/${normalizedId}`,
            ].filter(Boolean),
            limitations: Array.isArray(helperResult?.notes) && helperResult.notes.length > 0
              ? helperResult.notes
              : [
                  "Domain boundaries come from UniProt feature annotations plus a small number of explicitly labeled derived regions when UniProt indicates a signal-anchor or cleaved alternate chain.",
                ],
          }),
        }],
        structuredContent: {
          schema: "get_alphafold_domain_plddt.v1",
          result_status: "ok",
          uniprot_id: normalizedId,
          entry_id: normalizeWhitespace(helperResult?.entry_id || `AF-${normalizedId}-F1`) || null,
          version: helperResult?.version ?? null,
          latest_version: helperResult?.latest_version ?? null,
          global_plddt: toNullableNumber(helperResult?.global_plddt),
          model_source: normalizeWhitespace(helperResult?.model_source || "") || null,
          requested_domain_means: requestedRows,
          domain_means: domainRows,
          notes: Array.isArray(helperResult?.notes) ? helperResult.notes : [],
        },
      };
    } catch (error) {
      const detail = compactErrorMessage(error?.message || "unknown error", 220);
      return {
        content: [{ type: "text", text: `Error computing AlphaFold domain pLDDT values: ${detail}` }],
        structuredContent: {
          schema: "get_alphafold_domain_plddt.v1",
          result_status: "error",
          uniprot_id: normalizedId,
          error: detail,
        },
      };
    }
  }
);

// ---------------------------------------------------------------------------
// GWAS Catalog — trait-variant associations from genome-wide association studies
// ---------------------------------------------------------------------------

function normalizeGwasRiskAlleleName(value) {
  const normalized = normalizeWhitespace(value || "");
  if (!normalized) return { variantId: "", allele: "", combined: "" };
  const match = normalized.match(/^(rs\d+)-([A-Za-z?]+)$/i);
  if (!match) {
    return {
      variantId: normalized,
      allele: "",
      combined: normalized.toUpperCase(),
    };
  }
  return {
    variantId: normalizeWhitespace(match[1]),
    allele: normalizeWhitespace(match[2]).toUpperCase(),
    combined: `${normalizeWhitespace(match[1]).toLowerCase()}-${normalizeWhitespace(match[2]).toUpperCase()}`,
  };
}

function extractGwasAssociationRiskAlleles(association) {
  const rows = [];
  const loci = Array.isArray(association?.loci) ? association.loci : [];
  for (const locus of loci) {
    const strongest = Array.isArray(locus?.strongestRiskAlleles) ? locus.strongestRiskAlleles : [];
    for (const row of strongest) {
      const parsed = normalizeGwasRiskAlleleName(row?.riskAlleleName || "");
      if (!parsed.variantId) continue;
      rows.push({
        risk_allele_name: normalizeWhitespace(row?.riskAlleleName || ""),
        variant_id: parsed.variantId,
        allele: parsed.allele,
        risk_frequency: normalizeWhitespace(row?.riskFrequency || ""),
      });
    }
  }
  return rows;
}

function pickBestGwasStudyVariantAssociation(associations, variantId, riskAllele) {
  const wantedVariant = normalizeWhitespace(variantId || "").toLowerCase();
  const wantedAllele = normalizeWhitespace(riskAllele || "").toUpperCase();
  let best = null;
  for (const association of Array.isArray(associations) ? associations : []) {
    const riskAlleles = extractGwasAssociationRiskAlleles(association);
    for (const risk of riskAlleles) {
      if (wantedVariant && risk.variant_id.toLowerCase() !== wantedVariant) continue;
      const hasExactAllele = Boolean(wantedAllele) && risk.allele === wantedAllele;
      const associationRiskFrequency = normalizeWhitespace(association?.riskFrequency || "");
      const locusRiskFrequency = normalizeWhitespace(risk.risk_frequency || "");
      const resolvedRiskFrequency = locusRiskFrequency || associationRiskFrequency;
      const numericRiskFrequency = toFiniteNumber(resolvedRiskFrequency, Number.NaN);
      const score =
        (hasExactAllele ? 1000 : 0) +
        (Number.isFinite(numericRiskFrequency) ? 100 : 0) +
        (normalizeWhitespace(association?.pvalueDescription || "") ? 0 : 1);
      if (!best || score > best.score) {
        best = {
          score,
          association,
          risk,
          resolvedRiskFrequency,
        };
      }
    }
  }
  return best;
}

function rankJasparMatrixCandidate(candidate, tfName, speciesTaxId) {
  const name = normalizeWhitespace(candidate?.name || "");
  const species = Array.isArray(candidate?.species) ? candidate.species : [];
  const version = toNonNegativeInt(candidate?.version, 0);
  const wantedName = normalizeWhitespace(tfName || "");
  const wantedSpecies = toNonNegativeInt(speciesTaxId, 0);
  let score = version;
  if (wantedName && name.localeCompare(wantedName, undefined, { sensitivity: "accent" }) === 0) {
    score += 1000;
  } else if (wantedName && name.toLowerCase() === wantedName.toLowerCase()) {
    score += 900;
  }
  if (wantedSpecies > 0 && species.some((entry) => toNonNegativeInt(entry?.tax_id, 0) === wantedSpecies)) {
    score += 500;
  }
  if (normalizeWhitespace(candidate?.collection || "").toUpperCase() === "CORE") {
    score += 50;
  }
  return score;
}

function computeJasparConsensusAndInformationContent(pfm) {
  const bases = ["A", "C", "G", "T"];
  const columns = Math.max(
    ...bases.map((base) => (Array.isArray(pfm?.[base]) ? pfm[base].length : 0))
  );
  if (!Number.isFinite(columns) || columns <= 0) {
    return {
      consensus: "",
      totalInformationContent: Number.NaN,
      perPositionBits: [],
    };
  }

  const consensus = [];
  const perPositionBits = [];
  let totalInformationContent = 0;

  for (let idx = 0; idx < columns; idx += 1) {
    const counts = bases.map((base) => toFiniteNumber(pfm?.[base]?.[idx], 0));
    const total = counts.reduce((sum, value) => sum + value, 0);
    if (!(total > 0)) {
      consensus.push("N");
      perPositionBits.push(0);
      continue;
    }
    const probs = counts.map((value) => value / total);
    const bestBaseIndex = probs.reduce((bestIdx, value, currentIdx, arr) => (
      value > arr[bestIdx] ? currentIdx : bestIdx
    ), 0);
    consensus.push(bases[bestBaseIndex]);
    const entropy = probs.reduce((sum, value) => {
      if (!(value > 0)) return sum;
      return sum - (value * Math.log2(value));
    }, 0);
    const infoBits = 2 - entropy;
    perPositionBits.push(infoBits);
    totalInformationContent += infoBits;
  }

  return {
    consensus: consensus.join(""),
    totalInformationContent,
    perPositionBits,
  };
}

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

server.registerTool(
  "get_gwas_study_variant_association",
  {
    description:
      "Retrieve a specific GWAS Catalog association for one study accession and one variant, including RAF / risk frequency, p-value, effect size, and risk allele details. Prefer this over broad GWAS search when the question names a GCST study and rsID.",
    inputSchema: {
      studyAccession: z.string().describe("GWAS Catalog study accession (for example 'GCST011946')."),
      variantId: z.string().describe("Variant rsID (for example 'rs79043147')."),
      riskAllele: z.string().optional().describe("Optional effect/risk allele to require (for example 'T')."),
      trait: z.string().optional().describe("Optional trait label for user-facing context."),
      pageSize: z.number().int().min(25).max(2000).optional().default(500).describe("How many study associations to scan per page."),
      maxPages: z.number().int().min(1).max(20).optional().default(8).describe("Maximum pages to scan before giving up."),
    },
  },
  async ({ studyAccession, variantId, riskAllele = "", trait = "", pageSize = 500, maxPages = 8 }) => {
    const cleanStudy = normalizeWhitespace(studyAccession || "").toUpperCase();
    const cleanVariant = normalizeWhitespace(variantId || "");
    const cleanRiskAllele = normalizeWhitespace(riskAllele || "").toUpperCase();
    const cleanTrait = normalizeWhitespace(trait || "");

    if (!cleanStudy || !cleanVariant) {
      return {
        content: [{ type: "text", text: "Provide both `studyAccession` (GCST...) and `variantId` (rs...)." }],
      };
    }

    const boundedPageSize = Math.max(25, Math.min(2000, Math.round(pageSize || 500)));
    const boundedMaxPages = Math.max(1, Math.min(20, Math.round(maxPages || 8)));
    const scannedUrls = [];
    const candidateMatches = [];
    let bestMatch = null;

    for (let pageIndex = 0; pageIndex < boundedMaxPages; pageIndex += 1) {
      const params = new URLSearchParams({
        projection: "associationByStudy",
        size: String(boundedPageSize),
        page: String(pageIndex),
      });
      const url = `${GWAS_CATALOG_LEGACY_API}/studies/${encodeURIComponent(cleanStudy)}/associations?${params}`;
      scannedUrls.push(url);
      let data;
      try {
        data = await fetchJsonWithRetry(url, {
          headers: { Accept: "application/json" },
          retries: 1,
          timeoutMs: 20000,
          maxBackoffMs: 2500,
        });
      } catch (err) {
        return {
          content: [{ type: "text", text: renderStructuredResponse({
            summary: `GWAS Catalog study-specific lookup failed: ${err.message}`,
            keyFields: [
              `Study: ${cleanStudy}`,
              `Variant: ${cleanVariant}${cleanRiskAllele ? `-${cleanRiskAllele}` : ""}`,
            ],
            sources: scannedUrls,
            limitations: ["The legacy GWAS Catalog study-association endpoint may be temporarily unavailable."],
          }) }],
        };
      }

      const associations = Array.isArray(data?._embedded?.associations) ? data._embedded.associations : [];
      const match = pickBestGwasStudyVariantAssociation(associations, cleanVariant, cleanRiskAllele);
      if (match) {
        bestMatch = match;
        break;
      }

      for (const association of associations) {
        const riskAlleles = extractGwasAssociationRiskAlleles(association)
          .filter((row) => row.variant_id.toLowerCase() === cleanVariant.toLowerCase())
          .map((row) => row.risk_allele_name)
          .filter(Boolean);
        if (riskAlleles.length > 0) {
          candidateMatches.push(...riskAlleles);
        }
      }

      const totalPages = toNonNegativeInt(data?.page?.totalPages, 0);
      if (totalPages > 0 && pageIndex + 1 >= totalPages) {
        break;
      }
      if (associations.length < boundedPageSize) {
        break;
      }
    }

    if (!bestMatch) {
      const variantSummary = candidateMatches.length > 0
        ? `Risk-allele names seen for ${cleanVariant}: ${dedupeArray(candidateMatches).slice(0, 6).join(", ")}`
        : `No ${cleanVariant} association rows were recovered from study ${cleanStudy}.`;
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `No GWAS Catalog association match was found for ${cleanVariant}${cleanRiskAllele ? `-${cleanRiskAllele}` : ""} in study ${cleanStudy}.`,
          keyFields: [
            `Study: ${cleanStudy}`,
            `Variant: ${cleanVariant}`,
            cleanRiskAllele ? `Requested risk allele: ${cleanRiskAllele}` : "",
            variantSummary,
          ].filter(Boolean),
          sources: scannedUrls,
          limitations: [
            "Study-level association pages were scanned from the legacy GWAS Catalog API because the v2 API does not expose equivalent study+variant association endpoints.",
          ],
        }) }],
      };
    }

    const association = bestMatch.association || {};
    const risk = bestMatch.risk || {};
    const study = association?.study || {};
    const resolvedTrait = normalizeWhitespace(
      study?.diseaseTrait?.trait
      || (Array.isArray(association?.efoTraits) ? association.efoTraits[0]?.trait : "")
      || cleanTrait
    );
    const riskFrequencyRaw = normalizeWhitespace(bestMatch.resolvedRiskFrequency || "");
    const riskFrequencyNum = toFiniteNumber(riskFrequencyRaw, Number.NaN);
    const beta = toFiniteNumber(association?.betaNum, Number.NaN);
    const standardError = toFiniteNumber(association?.standardError, Number.NaN);
    const oddsRatio = toFiniteNumber(association?.orPerCopyNum, Number.NaN);
    const pValue = toFiniteNumber(association?.pvalue, Number.NaN);
    const pubmedId = normalizeWhitespace(study?.publicationInfo?.pubmedId || "");
    const studyUrl = `${GWAS_CATALOG_LEGACY_API}/studies/${encodeURIComponent(cleanStudy)}`;
    const associationUrl = normalizeWhitespace(association?._links?.self?.href || "");

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary:
            `GWAS Catalog study ${cleanStudy} reports ${risk.risk_allele_name || `${cleanVariant}${cleanRiskAllele ? `-${cleanRiskAllele}` : ""}`} ` +
            `for ${resolvedTrait || "the requested trait"} with RAF ${Number.isFinite(riskFrequencyNum) ? riskFrequencyRaw : "unavailable"}.`,
          keyFields: [
            `Study: ${cleanStudy}`,
            resolvedTrait ? `Trait: ${resolvedTrait}` : "",
            `Variant: ${cleanVariant}`,
            `Risk allele: ${risk.allele || cleanRiskAllele || "unavailable"}`,
            `Risk allele label: ${risk.risk_allele_name || "unavailable"}`,
            `RAF / risk frequency: ${Number.isFinite(riskFrequencyNum) ? riskFrequencyRaw : "unavailable"}`,
            Number.isFinite(beta) ? `Beta: ${beta}` : "",
            Number.isFinite(standardError) ? `Standard error: ${standardError}` : "",
            Number.isFinite(oddsRatio) ? `Odds ratio: ${oddsRatio}` : "",
            Number.isFinite(pValue) ? `P-value: ${pValue}` : "",
            pubmedId ? `PMID: ${pubmedId}` : "",
          ].filter(Boolean),
          sources: [associationUrl, studyUrl, ...scannedUrls].filter(Boolean),
          limitations: [
            "This uses the study-specific association pages from the legacy GWAS Catalog REST API because those details are not exposed on the v2 search endpoints.",
          ],
        }),
      }],
    };
  }
);

server.registerTool(
  "get_jaspar_motif_profile",
  {
    description:
      "Retrieve one JASPAR transcription-factor motif profile and compute its consensus sequence plus total information content from the position frequency matrix. Prefer this for JASPAR questions naming a TF or matrix id.",
    inputSchema: {
      tfName: z.string().optional().describe("Transcription factor name to search in JASPAR (for example 'SPI1')."),
      matrixId: z.string().optional().describe("Optional exact JASPAR matrix id (for example 'MA0080.5')."),
      speciesTaxId: z.number().int().optional().default(9606).describe("NCBI taxonomy id to prefer when searching by TF name (default 9606 for human)."),
      collection: z.string().optional().default("CORE").describe("JASPAR collection to search (default CORE)."),
      taxGroup: z.string().optional().default("vertebrates").describe("JASPAR taxonomic group to search (default vertebrates)."),
    },
  },
  async ({ tfName = "", matrixId = "", speciesTaxId = 9606, collection = "CORE", taxGroup = "vertebrates" }) => {
    const cleanTfName = normalizeWhitespace(tfName || "");
    const cleanMatrixId = normalizeWhitespace(matrixId || "");
    const cleanCollection = normalizeWhitespace(collection || "CORE") || "CORE";
    const cleanTaxGroup = normalizeWhitespace(taxGroup || "vertebrates") || "vertebrates";
    const cleanSpeciesTaxId = toNonNegativeInt(speciesTaxId, 9606) || 9606;

    if (!cleanTfName && !cleanMatrixId) {
      return {
        content: [{ type: "text", text: "Provide either `tfName` or `matrixId` for JASPAR lookup." }],
      };
    }

    const sources = [];
    let selectedMatrix = null;

    if (cleanMatrixId) {
      const detailUrl = `${JASPAR_API}/matrix/${encodeURIComponent(cleanMatrixId)}/?format=json`;
      sources.push(detailUrl);
      try {
        selectedMatrix = await fetchJsonWithRetry(detailUrl, {
          headers: { Accept: "application/json" },
          retries: 1,
          timeoutMs: 15000,
          maxBackoffMs: 2500,
        });
      } catch (err) {
        return {
          content: [{ type: "text", text: renderStructuredResponse({
            summary: `JASPAR matrix lookup failed: ${err.message}`,
            keyFields: [`Matrix id: ${cleanMatrixId}`],
            sources,
            limitations: ["Check that the JASPAR matrix id exists and is publicly available."],
          }) }],
        };
      }
    } else {
      const params = new URLSearchParams({
        search: cleanTfName,
        collection: cleanCollection,
        tax_group: cleanTaxGroup,
        species: String(cleanSpeciesTaxId),
        format: "json",
      });
      const searchUrl = `${JASPAR_API}/matrix/?${params}`;
      sources.push(searchUrl);
      let searchData;
      try {
        searchData = await fetchJsonWithRetry(searchUrl, {
          headers: { Accept: "application/json" },
          retries: 1,
          timeoutMs: 15000,
          maxBackoffMs: 2500,
        });
      } catch (err) {
        return {
          content: [{ type: "text", text: renderStructuredResponse({
            summary: `JASPAR search failed: ${err.message}`,
            keyFields: [`TF name: ${cleanTfName}`],
            sources,
            limitations: ["The JASPAR API may be temporarily unavailable."],
          }) }],
        };
      }
      const results = Array.isArray(searchData?.results) ? searchData.results : [];
      if (results.length === 0) {
        return {
          content: [{ type: "text", text: renderStructuredResponse({
            summary: `No JASPAR motif matched ${cleanTfName}.`,
            keyFields: [
              `TF name: ${cleanTfName}`,
              `Species tax id: ${cleanSpeciesTaxId}`,
              `Collection: ${cleanCollection}`,
            ],
            sources,
            limitations: ["Try another TF synonym or query a specific matrix id if known."],
          }) }],
        };
      }

      const ranked = [];
      for (const result of results) {
        const url = normalizeWhitespace(result?.url || "");
        if (!url) continue;
        const detailUrl = `${url}${url.includes("?") ? "&" : "?"}format=json`;
        let detail;
        try {
          detail = await fetchJsonWithRetry(detailUrl, {
            headers: { Accept: "application/json" },
            retries: 1,
            timeoutMs: 15000,
            maxBackoffMs: 2500,
          });
        } catch (_) {
          continue;
        }
        const score = rankJasparMatrixCandidate(detail, cleanTfName, cleanSpeciesTaxId);
        ranked.push({ score, detail, detailUrl });
      }

      ranked.sort((a, b) => b.score - a.score);
      if (ranked.length === 0) {
        return {
          content: [{ type: "text", text: renderStructuredResponse({
            summary: `JASPAR search found candidate motifs for ${cleanTfName}, but none could be retrieved in detail.`,
            keyFields: [`TF name: ${cleanTfName}`],
            sources,
            limitations: ["The returned candidate matrix detail pages could not be fetched."],
          }) }],
        };
      }

      selectedMatrix = ranked[0].detail;
      sources.push(ranked[0].detailUrl);
    }

    const pfm = selectedMatrix?.pfm || {};
    const computed = computeJasparConsensusAndInformationContent(pfm);
    const totalBits = computed.totalInformationContent;
    const roundedBits = Number.isFinite(totalBits) ? totalBits.toFixed(2) : "unavailable";
    const speciesNames = (Array.isArray(selectedMatrix?.species) ? selectedMatrix.species : [])
      .map((entry) => normalizeWhitespace(entry?.name || ""))
      .filter(Boolean);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary:
            `JASPAR motif ${normalizeWhitespace(selectedMatrix?.matrix_id || cleanMatrixId || "") || "unavailable"} ` +
            `for ${normalizeWhitespace(selectedMatrix?.name || cleanTfName || "") || "the requested TF"} has consensus ${computed.consensus || "unavailable"} ` +
            `and total information content ${roundedBits} bits.`,
          keyFields: [
            `Matrix id: ${normalizeWhitespace(selectedMatrix?.matrix_id || cleanMatrixId || "") || "unavailable"}`,
            `TF name: ${normalizeWhitespace(selectedMatrix?.name || cleanTfName || "") || "unavailable"}`,
            speciesNames.length > 0 ? `Species: ${speciesNames.join(", ")}` : "",
            `Collection: ${normalizeWhitespace(selectedMatrix?.collection || cleanCollection || "") || "unavailable"}`,
            `Consensus recognition sequence: ${computed.consensus || "unavailable"}`,
            `Total information content (bits): ${roundedBits}`,
          ].filter(Boolean),
          sources: dedupeArray(sources),
          limitations: [
            "Consensus and total information content are computed directly from the JASPAR position frequency matrix returned by the public API.",
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
// Human Protein Atlas — protein expression, localization, and single-cell summaries
// ---------------------------------------------------------------------------

function normalizeHpaReleaseHost(release) {
  const raw = normalizeWhitespace(release || "").toLowerCase();
  if (!raw || raw === "latest" || raw === "current") {
    return HPA_API;
  }
  const versionMatch = raw.match(/^v?(\d{2})$/);
  if (versionMatch) {
    return `https://v${versionMatch[1]}.proteinatlas.org`;
  }
  if (/^https?:\/\//.test(raw)) {
    return raw.replace(/\/+$/, "");
  }
  return HPA_API;
}

function encodeHpaPathSegment(value) {
  return encodeURIComponent(normalizeWhitespace(value || "")).replace(/%20/g, "+");
}

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeHpaDatasetName(value) {
  const raw = normalizeWhitespace(value || "").toLowerCase().replace(/[\s-]+/g, "_");
  if (!raw) return "";
  if (["single_cell_type", "single_cell", "single_cell_non_tabula_sapiens", "non_tabula_sapiens", "non_tabula"].includes(raw)) {
    return "single_cell_type";
  }
  if (["ts_single_cell_type", "tabula_sapiens", "tabula"].includes(raw)) {
    return "ts_single_cell_type";
  }
  if (["tissue_cell_type", "tissue"].includes(raw)) {
    return "tissue_cell_type";
  }
  return raw;
}

function classifyHpaSingleCellSection(html, position) {
  const before = html.slice(0, Math.max(0, position));
  const anchors = [
    { name: "single_cell_type", patterns: ['id="single_cell_type"', "id='single_cell_type'"] },
    { name: "ts_single_cell_type", patterns: ['id="ts_single_cell_type"', "id='ts_single_cell_type'"] },
    { name: "tissue_cell_type", patterns: ['id="tissue_cell_type"', "id='tissue_cell_type'"] },
  ];

  let bestName = "";
  let bestIndex = -1;
  for (const anchor of anchors) {
    for (const pattern of anchor.patterns) {
      const idx = before.lastIndexOf(pattern);
      if (idx > bestIndex) {
        bestIndex = idx;
        bestName = anchor.name;
      }
    }
  }
  return bestName || "single_cell_type";
}

function normalizeHpaCellTypeLabel(value) {
  return normalizeWhitespace(String(value || "").replace(/\s+c-\d+\b/gi, "")).toLowerCase();
}

function scoreHpaCellTypeMatch(candidateLabel, requestedLabel) {
  const candidate = normalizeHpaCellTypeLabel(candidateLabel);
  const requested = normalizeHpaCellTypeLabel(requestedLabel);
  if (!candidate || !requested) return -1;
  if (candidate === requested) return 100;
  if (candidate.includes(requested) || requested.includes(candidate)) return 80;
  const candidateTokens = candidate.split(/\s+/).filter(Boolean);
  const requestedTokens = requested.split(/\s+/).filter(Boolean);
  const overlap = requestedTokens.filter((token) => candidateTokens.includes(token)).length;
  if (overlap === requestedTokens.length) return 70;
  if (overlap > 0) return overlap;
  return -1;
}

function groupHpaMarkerEntriesByCellType(entries) {
  const groups = new Map();
  for (const entry of Array.isArray(entries) ? entries : []) {
    const rawLabel = normalizeWhitespace(entry?.label || "");
    const baseLabel = normalizeWhitespace(rawLabel.replace(/\s+c-\d+\b/gi, ""));
    const value = toFiniteNumber(entry?.value, Number.NaN);
    if (!baseLabel || !Number.isFinite(value)) continue;
    const unitMatch = normalizeWhitespace(stripHtmlToText(entry?.tooltip || "")).match(/\b(n[CT]PM)\b/i);
    const cellCount = Math.max(0, toFiniteNumber(entry?.cell_count, 0));
    const group = groups.get(baseLabel) || {
      label: baseLabel,
      unit: unitMatch ? unitMatch[1] : "",
      entries: [],
      weightedSum: 0,
      weightTotal: 0,
      simpleSum: 0,
      valueCount: 0,
    };
    group.entries.push({
      label: rawLabel || baseLabel,
      value,
      cellCount,
      tooltip: normalizeWhitespace(stripHtmlToText(entry?.tooltip || "")),
      group: normalizeWhitespace(entry?.group || ""),
    });
    if (!group.unit && unitMatch) {
      group.unit = unitMatch[1];
    }
    if (cellCount > 0) {
      group.weightedSum += value * cellCount;
      group.weightTotal += cellCount;
    }
    group.simpleSum += value;
    group.valueCount += 1;
    groups.set(baseLabel, group);
  }

  return Array.from(groups.values()).map((group) => {
    const representativeValue = group.weightTotal > 0
      ? group.weightedSum / group.weightTotal
      : group.valueCount > 0
        ? group.simpleSum / group.valueCount
        : Number.NaN;
    return {
      label: group.label,
      unit: group.unit || "",
      representativeValue,
      entryCount: group.valueCount,
      weightedByCellCount: group.weightTotal > 0,
      totalCellCount: group.weightTotal,
      entries: group.entries,
    };
  });
}

function extractJsonAssignmentsFromHtml(html, variableName) {
  const text = String(html || "");
  const needle = `var ${String(variableName || "").trim()} = `;
  if (!text || !needle.trim()) return [];

  const assignments = [];
  let searchStart = 0;
  while (searchStart < text.length) {
    const statementStart = text.indexOf(needle, searchStart);
    if (statementStart === -1) break;
    const jsonStart = text.indexOf("{", statementStart + needle.length);
    if (jsonStart === -1) break;

    let depth = 0;
    let inString = false;
    let escaped = false;
    let jsonEnd = -1;

    for (let idx = jsonStart; idx < text.length; idx += 1) {
      const char = text[idx];
      if (inString) {
        if (escaped) {
          escaped = false;
        } else if (char === "\\") {
          escaped = true;
        } else if (char === "\"") {
          inString = false;
        }
        continue;
      }
      if (char === "\"") {
        inString = true;
        continue;
      }
      if (char === "{") {
        depth += 1;
      } else if (char === "}") {
        depth -= 1;
        if (depth === 0) {
          jsonEnd = idx;
          break;
        }
      }
    }

    if (jsonEnd === -1) break;
    try {
      assignments.push({
        start: statementStart,
        value: JSON.parse(text.slice(jsonStart, jsonEnd + 1)),
      });
    } catch (_) {
      // Ignore malformed blocks and continue scanning for later assignments.
    }
    searchStart = jsonEnd + 1;
  }

  return assignments;
}

function extractHpaCurrentGeneMarkerBlocks(html, geneSymbol) {
  const symbol = normalizeWhitespace(geneSymbol || "");
  if (!html || !symbol) return [];
  const blocks = [];
  for (const assignment of extractJsonAssignmentsFromHtml(html, "marker_tpm")) {
    try {
      const parsed = assignment.value && typeof assignment.value === "object" ? assignment.value : {};
      const payload = parsed[symbol] || null;
      if (!payload || normalizeWhitespace(payload?.cell_type_name || "").toLowerCase() !== "current gene") {
        continue;
      }
      blocks.push({
        section: classifyHpaSingleCellSection(html, assignment.start || 0),
        geneSymbol: symbol,
        entries: Array.isArray(payload?.data) ? payload.data : [],
        url: normalizeWhitespace(payload?.url || ""),
      });
    } catch (_) {
      continue;
    }
  }
  return blocks;
}

server.registerTool(
  "get_human_protein_atlas_gene",
  {
    description:
      "Retrieves a Human Protein Atlas gene summary including tissue specificity, single-cell specificity, " +
      "subcellular localization, protein class, disease involvement, and optional exact tissue/cell-type single-cell lookups. " +
      "Accepts a gene symbol or Ensembl gene ID.",
    inputSchema: {
      gene: z.string().optional().describe("Human gene symbol or alias to resolve via MyGene.info (e.g. 'TP53')."),
      ensemblGeneId: z.string().optional().describe("Ensembl gene ID (e.g. 'ENSG00000141510')."),
      singleCellTissue: z.string().optional().describe("Optional HPA single-cell tissue page to inspect exactly (for example 'prostate')."),
      singleCellCellType: z.string().optional().describe("Optional single-cell cell type label to resolve within the requested HPA tissue page (for example 'Basal prostatic cells')."),
      singleCellDataset: z.string().optional().describe("Optional HPA single-cell dataset selector: 'single_cell_type', 'tabula_sapiens', or 'tissue_cell_type'."),
      release: z.string().optional().describe("Optional Human Protein Atlas release such as 'v24' or 'latest'. Defaults to the current public release."),
    },
  },
  async ({ gene = "", ensemblGeneId = "", singleCellTissue = "", singleCellCellType = "", singleCellDataset = "", release = "" }) => {
    const normalizedEnsembl = normalizeWhitespace(ensemblGeneId || "");
    const normalizedGene = normalizeWhitespace(gene || "").toUpperCase();
    const normalizedSingleCellTissue = normalizeWhitespace(singleCellTissue || "");
    const normalizedSingleCellCellType = normalizeWhitespace(singleCellCellType || "");
    const normalizedSingleCellDataset = normalizeHpaDatasetName(singleCellDataset || "");
    const normalizedRelease = normalizeWhitespace(release || "");
    if (!normalizedEnsembl && !normalizedGene) {
      return { content: [{ type: "text", text: "Provide either `gene` or `ensemblGeneId`." }] };
    }

    let resolvedEnsembl = normalizedEnsembl;
    let resolvedSymbol = normalizedGene;
    const sources = [];
    const limitations = [];

    try {
      if (!resolvedEnsembl) {
        const resolved = await resolveGeneWithMyGene(normalizedGene, "human");
        const ids = normalizeMyGeneIds(resolved.bestHit || {});
        resolvedEnsembl = ids.ensemblGenes[0] || "";
        resolvedSymbol = ids.symbol || resolvedSymbol;
        sources.push(`${MYGENE_API}/query?q=${encodeURIComponent(normalizedGene)}&species=human`);
        if (!resolvedEnsembl) {
          return {
            content: [{
              type: "text",
              text: renderStructuredResponse({
                summary: `Could not resolve an Ensembl gene ID for "${normalizedGene}" via MyGene.info.`,
                keyFields: [`Query: ${normalizedGene}`],
                sources,
                limitations: ["Human Protein Atlas gene JSON endpoints are keyed by Ensembl gene ID."],
              }),
            }],
          };
        }
        limitations.push(`Resolved ${normalizedGene} to ${resolvedEnsembl} via MyGene.info.`);
      }

      const hpaHost = normalizeHpaReleaseHost(normalizedRelease);
      const dataUrl = `${hpaHost}/${encodeURIComponent(resolvedEnsembl)}.json`;
      const record = await fetchJsonWithRetry(dataUrl, { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 });
      const pageSymbol = normalizeWhitespace(record?.Gene || resolvedSymbol || "");
      const pageEnsembl = normalizeWhitespace(record?.Ensembl || resolvedEnsembl);
      const singleCellSpecific = record?.["RNA single cell type specific nCPM"];
      const brainSpecific = record?.["RNA single nuclei brain specific nCPM"];
      const topSingleCell = singleCellSpecific && typeof singleCellSpecific === "object"
        ? Object.entries(singleCellSpecific)
          .map(([label, value]) => [label, toFiniteNumber(value, Number.NaN)])
          .filter(([, value]) => Number.isFinite(value))
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([label, value]) => `${label} (${value.toFixed(1)} nCPM)`)
        : [];
      const topBrainCellTypes = brainSpecific && typeof brainSpecific === "object"
        ? Object.entries(brainSpecific)
          .map(([label, value]) => [label, toFiniteNumber(value, Number.NaN)])
          .filter(([, value]) => Number.isFinite(value))
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([label, value]) => `${label} (${value.toFixed(1)} nCPM)`)
        : [];

      const proteinClasses = Array.isArray(record?.["Protein class"]) ? record["Protein class"] : [];
      const diseaseInvolvement = Array.isArray(record?.["Disease involvement"]) ? record["Disease involvement"] : [];
      const biologicalProcess = Array.isArray(record?.["Biological process"]) ? record["Biological process"] : [];
      const molecularFunction = Array.isArray(record?.["Molecular function"]) ? record["Molecular function"] : [];
      const subcellularMain = Array.isArray(record?.["Subcellular main location"]) ? record["Subcellular main location"] : [];
      const subcellularAdditional = Array.isArray(record?.["Subcellular additional location"]) ? record["Subcellular additional location"] : [];

      const keyFields = [
        `Gene: ${pageSymbol || normalizedGene || resolvedEnsembl} (${pageEnsembl})`,
        `Release: ${normalizedRelease || "latest"}`,
        `Description: ${normalizeWhitespace(record?.["Gene description"] || "No description available.")}`,
        `Evidence level: ${normalizeWhitespace(record?.Evidence || record?.["HPA evidence"] || "unknown")}`,
        `RNA tissue specificity: ${normalizeWhitespace(record?.["RNA tissue specificity"] || "unknown")} | Distribution: ${normalizeWhitespace(record?.["RNA tissue distribution"] || "unknown")}`,
        `RNA single-cell specificity: ${normalizeWhitespace(record?.["RNA single cell type specificity"] || "unknown")} | Distribution: ${normalizeWhitespace(record?.["RNA single cell type distribution"] || "unknown")}`,
        `RNA brain nuclei specificity: ${normalizeWhitespace(record?.["RNA single nuclei brain specificity"] || "unknown")} | Distribution: ${normalizeWhitespace(record?.["RNA single nuclei brain distribution"] || "unknown")}`,
        `Subcellular main location: ${subcellularMain.length > 0 ? subcellularMain.join(", ") : "not listed"}`,
        `Subcellular additional location: ${subcellularAdditional.length > 0 ? subcellularAdditional.join(", ") : "not listed"}`,
      ];
      if (proteinClasses.length > 0) keyFields.push(`Protein classes: ${proteinClasses.slice(0, 8).join(", ")}`);
      if (diseaseInvolvement.length > 0) keyFields.push(`Disease involvement: ${diseaseInvolvement.slice(0, 8).join(", ")}`);
      if (biologicalProcess.length > 0) keyFields.push(`Biological process: ${biologicalProcess.slice(0, 8).join(", ")}`);
      if (molecularFunction.length > 0) keyFields.push(`Molecular function: ${molecularFunction.slice(0, 8).join(", ")}`);
      if (topSingleCell.length > 0) keyFields.push(`Top single-cell signals: ${topSingleCell.join(", ")}`);
      if (topBrainCellTypes.length > 0) keyFields.push(`Top brain cell-type signals: ${topBrainCellTypes.join(", ")}`);

      const hpaPageUrl = `${hpaHost}/${encodeURIComponent(pageEnsembl)}-${encodeURIComponent(pageSymbol || resolvedSymbol || "")}`;
      sources.push(dataUrl, hpaPageUrl);
      limitations.push("The MCP integration uses the Human Protein Atlas gene summary JSON endpoint, which emphasizes atlas-level summaries rather than per-image inspection.");

      let singleCellLookup = null;
      if (normalizedSingleCellTissue || normalizedSingleCellCellType || normalizedSingleCellDataset) {
        const detailPath = normalizedSingleCellTissue
          ? `/single+cell/${encodeHpaPathSegment(normalizedSingleCellTissue)}`
          : "/single+cell";
        const detailUrl = `${hpaPageUrl}${detailPath}`;
        try {
          const detailResponse = await fetchWithRetry(detailUrl, { retries: 1, timeoutMs: 20000, maxBackoffMs: 2500 });
          const detailHtml = await detailResponse.text();
          const markerBlocks = extractHpaCurrentGeneMarkerBlocks(detailHtml, pageSymbol || resolvedSymbol || normalizedGene);
          const matchingBlocks = normalizedSingleCellDataset
            ? markerBlocks.filter((block) => block.section === normalizedSingleCellDataset)
            : markerBlocks;
          const selectedBlock = matchingBlocks[0] || markerBlocks[0] || null;

          if (selectedBlock) {
            const groupedCellTypes = groupHpaMarkerEntriesByCellType(selectedBlock.entries);
            let matchedGroup = null;
            if (normalizedSingleCellCellType) {
              const rankedGroups = groupedCellTypes
                .map((group) => ({ group, score: scoreHpaCellTypeMatch(group.label, normalizedSingleCellCellType) }))
                .filter((item) => item.score >= 0)
                .sort((a, b) => b.score - a.score || b.group.entryCount - a.group.entryCount);
              matchedGroup = rankedGroups[0]?.group || null;
            }

            if (matchedGroup) {
              const displayValue = Number.isFinite(matchedGroup.representativeValue)
                ? matchedGroup.representativeValue.toFixed(1)
                : "unknown";
              const clusterSummaries = matchedGroup.entries
                .slice(0, 8)
                .map((entry) =>
                  `${entry.label} (${entry.value.toFixed(1)} ${matchedGroup.unit || "expression units"}${entry.cellCount > 0 ? `; n=${entry.cellCount}` : ""})`
                );
              keyFields.push(`Detailed single-cell tissue page: ${normalizedSingleCellTissue || "global single-cell"} | Dataset: ${selectedBlock.section}`);
              keyFields.push(`Matched cell type: ${matchedGroup.label}`);
              keyFields.push(`Exact single-cell expression: ${displayValue} ${matchedGroup.unit || "expression units"}`);
              if (clusterSummaries.length > 0) {
                keyFields.push(`Supporting clusters: ${clusterSummaries.join(", ")}`);
              }
              sources.push(detailUrl);
              limitations.push("Detailed single-cell values are parsed from the Human Protein Atlas single-cell tissue page rather than the summary JSON endpoint.");
              singleCellLookup = {
                tissue: normalizedSingleCellTissue || null,
                dataset: selectedBlock.section,
                cellTypeQuery: normalizedSingleCellCellType || null,
                matchedCellType: matchedGroup.label,
                value: displayValue,
                unit: matchedGroup.unit || null,
                weightedByCellCount: matchedGroup.weightedByCellCount,
                representativeValue: matchedGroup.representativeValue,
                clusters: matchedGroup.entries.map((entry) => ({
                  label: entry.label,
                  value: entry.value,
                  unit: matchedGroup.unit || null,
                  cellCount: entry.cellCount || null,
                })),
              };
            } else if (normalizedSingleCellCellType) {
              keyFields.push(`Detailed single-cell tissue page: ${normalizedSingleCellTissue || "global single-cell"} | Dataset: ${selectedBlock.section}`);
              keyFields.push(`No exact single-cell cell-type match found for: ${normalizedSingleCellCellType}`);
              const availableCellTypes = groupedCellTypes.slice(0, 8).map((group) => group.label);
              if (availableCellTypes.length > 0) {
                keyFields.push(`Available cell types on page: ${availableCellTypes.join(", ")}`);
              }
              sources.push(detailUrl);
              limitations.push("Requested exact single-cell cell type was not found on the selected Human Protein Atlas page.");
            }
          }
        } catch (detailError) {
          limitations.push(`Detailed single-cell page lookup failed: ${normalizeWhitespace(detailError?.message || String(detailError))}`);
        }
      }

      return {
        structuredContent: {
          identifier: {
            gene: pageSymbol || normalizedGene || resolvedEnsembl,
            ensemblGeneId: pageEnsembl,
          },
          release: normalizedRelease || "latest",
          singleCellLookup,
        },
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: singleCellLookup
              ? `Human Protein Atlas single-cell profile for ${pageSymbol || resolvedSymbol || resolvedEnsembl}: ${singleCellLookup.matchedCellType} in ${singleCellLookup.tissue || "the selected tissue"} is ${singleCellLookup.value} ${singleCellLookup.unit || "expression units"} (${singleCellLookup.dataset}).`
              : `Human Protein Atlas profile for ${pageSymbol || resolvedSymbol || resolvedEnsembl}.`,
            keyFields,
            sources,
            limitations,
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_human_protein_atlas_gene: ${error.message}` }] };
    }
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

server.registerTool(
  "get_tcga_project_data_availability",
  {
    description:
      "Count how many TCGA cases in a project have files for a specified data category, data type, or experimental strategy using the Genomic Data Commons (GDC) cases endpoint. Prefer this for TCGA project-level availability questions such as proteome profiling counts.",
    inputSchema: {
      projectId: z.string().describe("TCGA project id such as 'TCGA-BRCA'."),
      dataCategory: z.string().optional().default("Proteome Profiling").describe("GDC data category to require (default 'Proteome Profiling')."),
      dataType: z.string().optional().describe("Optional GDC data type filter such as 'Protein Expression Quantification'."),
      experimentalStrategy: z.string().optional().describe("Optional GDC experimental strategy such as 'Reverse Phase Protein Array'."),
    },
  },
  async ({ projectId, dataCategory = "Proteome Profiling", dataType = "", experimentalStrategy = "" }) => {
    const cleanProjectId = normalizeWhitespace(projectId || "").toUpperCase();
    const cleanDataCategory = normalizeWhitespace(dataCategory || "");
    const cleanDataType = normalizeWhitespace(dataType || "");
    const cleanExperimentalStrategy = normalizeWhitespace(experimentalStrategy || "");

    if (!cleanProjectId) {
      return { content: [{ type: "text", text: "Provide a TCGA project id such as `TCGA-BRCA`." }] };
    }

    const filterClauses = [
      {
        op: "in",
        content: {
          field: "project.project_id",
          value: [cleanProjectId],
        },
      },
    ];
    if (cleanDataCategory) {
      filterClauses.push({
        op: "in",
        content: {
          field: "files.data_category",
          value: [cleanDataCategory],
        },
      });
    }
    if (cleanDataType) {
      filterClauses.push({
        op: "in",
        content: {
          field: "files.data_type",
          value: [cleanDataType],
        },
      });
    }
    if (cleanExperimentalStrategy) {
      filterClauses.push({
        op: "in",
        content: {
          field: "files.experimental_strategy",
          value: [cleanExperimentalStrategy],
        },
      });
    }

    const filters = {
      op: "and",
      content: filterClauses,
    };
    const params = new URLSearchParams({
      filters: JSON.stringify(filters),
      format: "JSON",
      size: "0",
    });
    const url = `${GDC_API}/cases?${params}`;
    let data;
    try {
      data = await fetchJsonWithRetry(url, {
        headers: { Accept: "application/json" },
        retries: 1,
        timeoutMs: 20000,
        maxBackoffMs: 2500,
      });
    } catch (err) {
      return {
        content: [{ type: "text", text: renderStructuredResponse({
          summary: `GDC project-level availability query failed: ${err.message}`,
          keyFields: [
            `Project: ${cleanProjectId}`,
            cleanDataCategory ? `Data category: ${cleanDataCategory}` : "",
            cleanDataType ? `Data type: ${cleanDataType}` : "",
            cleanExperimentalStrategy ? `Experimental strategy: ${cleanExperimentalStrategy}` : "",
          ].filter(Boolean),
          sources: [url],
          limitations: ["The GDC API may be temporarily unavailable."],
        }) }],
      };
    }

    const totalCases = toNonNegativeInt(data?.data?.pagination?.total, 0);

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary:
            `${totalCases.toLocaleString()} ${cleanProjectId} cases have associated ${cleanDataCategory || "requested"} files` +
            `${cleanDataType ? ` (${cleanDataType})` : ""}${cleanExperimentalStrategy ? ` via ${cleanExperimentalStrategy}` : ""}.`,
          keyFields: [
            `Project: ${cleanProjectId}`,
            cleanDataCategory ? `Data category: ${cleanDataCategory}` : "",
            cleanDataType ? `Data type: ${cleanDataType}` : "",
            cleanExperimentalStrategy ? `Experimental strategy: ${cleanExperimentalStrategy}` : "",
            `Matching cases: ${totalCases.toLocaleString()}`,
          ].filter(Boolean),
          sources: [url, `https://portal.gdc.cancer.gov/projects/${encodeURIComponent(cleanProjectId)}`],
          limitations: [
            "Counts come from the GDC cases endpoint and represent cases with at least one matching file, not raw file totals.",
          ],
        }),
      }],
    };
  }
);

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
// DepMap — target dependency and predictive biomarkers
// ---------------------------------------------------------------------------

server.registerTool(
  "get_depmap_gene_dependency",
  {
    description:
      "Summarizes DepMap target dependency metrics for a gene using the latest public target-discovery release, " +
      "including CRISPR/RNAi dependency fractions, pan-dependency/selectivity flags, and top predictive features.",
    inputSchema: {
      geneSymbol: z.string().describe("Gene symbol (e.g. 'EGFR', 'KRAS', 'TP53')."),
      includePredictiveFeatures: z.boolean().optional().default(true).describe("Whether to include top predictive features from DepMap's predictive models."),
      predictiveFeatureLimit: z.number().optional().default(5).describe("Maximum predictive features to return (1-10)."),
    },
  },
  async ({ geneSymbol, includePredictiveFeatures = true, predictiveFeatureLimit = 5 }) => {
    const symbol = normalizeWhitespace(geneSymbol || "").toUpperCase();
    const boundedFeatureLimit = Math.max(1, Math.min(10, Math.round(predictiveFeatureLimit || 5)));
    if (!symbol) {
      return { content: [{ type: "text", text: "Provide a gene symbol (e.g. EGFR)." }] };
    }

    try {
      const [summaryCsv, catalogRows, characterizationRows] = await Promise.all([
        fetchDepMapSummaryTable(),
        fetchDepMapDownloadCatalog(),
        fetchJsonWithRetry(`${DEPMAP_PORTAL_API}/gene/gene_characterization_data/${encodeURIComponent(symbol)}`, {
          retries: 1,
          timeoutMs: 12000,
          maxBackoffMs: 2500,
        }).catch(() => []),
      ]);
      const summaryRow = findCsvRowByValue(summaryCsv, "symbol", symbol);
      if (!summaryRow) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Gene "${symbol}" was not present in the latest DepMap target-discovery summary table.`,
              keyFields: [`Gene: ${symbol}`],
              sources: [`${DEPMAP_PORTAL_API}/tda/table_download`],
              limitations: ["Some genes may be absent from the public target-discovery export or may require a different HGNC symbol."],
            }),
          }],
        };
      }

      const dependencyRelease = catalogRows.find((row) => row.filename === "CRISPRGeneDependency.csv")
        || catalogRows.find((row) => row.filename === "ScreenGeneDependency.csv")
        || null;
      const entityId = extractDepMapEntityId(characterizationRows);

      let predictivePayload = [];
      if (includePredictiveFeatures && entityId) {
        predictivePayload = await fetchJsonWithRetry(
          `${DEPMAP_PORTAL_API}/gene/api/predictive?entityId=${encodeURIComponent(entityId)}`,
          { retries: 1, timeoutMs: 15000, maxBackoffMs: 2500 }
        ).catch(() => []);
      }

      const predictiveFeatures = Array.isArray(predictivePayload)
        ? predictivePayload
          .flatMap((group) => Array.isArray(group?.modelsAndResults) ? group.modelsAndResults : [])
          .flatMap((group) => Array.isArray(group?.results) ? group.results : [])
          .map((row) => ({
            featureName: normalizeWhitespace(row?.featureName || ""),
            featureType: normalizeWhitespace(row?.featureType || ""),
            featureImportance: toFiniteNumber(row?.featureImportance, Number.NaN),
            correlation: toFiniteNumber(row?.correlation, Number.NaN),
            relatedType: normalizeWhitespace(row?.relatedType || ""),
            interactiveUrl: normalizeWhitespace(row?.interactiveUrl || ""),
          }))
          .filter((row) => row.featureName && Number.isFinite(row.featureImportance))
          .sort((a, b) => b.featureImportance - a.featureImportance)
        : [];
      const dedupedPredictiveFeatures = [];
      const seenPredictive = new Set();
      for (const row of predictiveFeatures) {
        const key = `${row.featureType}::${row.featureName}`;
        if (seenPredictive.has(key)) continue;
        seenPredictive.add(key);
        dedupedPredictiveFeatures.push(row);
        if (dedupedPredictiveFeatures.length >= boundedFeatureLimit) break;
      }

      const crisprDepFrac = toFiniteNumber(summaryRow.CRISPR_depCL_frac, Number.NaN);
      const crisprStrongCount = toNonNegativeInt(summaryRow.CRISPR_strong_depCL_count, 0);
      const crisprMinEffect = toFiniteNumber(summaryRow.CRISPR_min_gene_effect, Number.NaN);
      const crisprMinZ = toFiniteNumber(summaryRow.CRISPR_min_gene_effect_zscore, Number.NaN);
      const crisprPredictiveAccuracy = toFiniteNumber(summaryRow.CRISPR_Predictive_Accuracy, Number.NaN);
      const crisprLrt = toFiniteNumber(summaryRow.CRISPR_LRT, Number.NaN);
      const rnaiDepFrac = toFiniteNumber(summaryRow.RNAi_depCL_frac, Number.NaN);
      const rnaiStrongCount = toNonNegativeInt(summaryRow.RNAi_strong_depCL_count, 0);
      const isPanDependency = String(summaryRow.CRISPR_PanDependency || "").trim().toLowerCase() === "true";
      const isStronglySelective = String(summaryRow.CRISPR_StronglySelective || "").trim().toLowerCase() === "true";

      const keyFields = [
        `Gene: ${symbol} | Entrez: ${normalizeWhitespace(summaryRow.entrez_id || "unknown")}`,
        dependencyRelease ? `Release: ${dependencyRelease.release} (${dependencyRelease.release_date})` : "Release: latest public DepMap target-discovery export",
        Number.isFinite(crisprDepFrac) ? `CRISPR dependent cell-line fraction: ${formatPct(crisprDepFrac)}` : "CRISPR dependent cell-line fraction: not available",
        `CRISPR strongly dependent cell lines: ${crisprStrongCount.toLocaleString()}`,
        Number.isFinite(crisprMinEffect) ? `CRISPR minimum gene effect: ${crisprMinEffect.toFixed(3)}${Number.isFinite(crisprMinZ) ? ` (z-score ${crisprMinZ.toFixed(3)})` : ""}` : "CRISPR minimum gene effect: not available",
        `CRISPR pan-dependency: ${isPanDependency ? "yes" : "no"} | Strongly selective: ${isStronglySelective ? "yes" : "no"}`,
        Number.isFinite(crisprPredictiveAccuracy) ? `CRISPR predictive accuracy: ${crisprPredictiveAccuracy.toFixed(3)}` : "CRISPR predictive accuracy: not available",
        Number.isFinite(crisprLrt) ? `CRISPR LRT: ${crisprLrt.toFixed(3)}` : "CRISPR LRT: not available",
        Number.isFinite(rnaiDepFrac) ? `RNAi dependent cell-line fraction: ${formatPct(rnaiDepFrac)}` : "RNAi dependent cell-line fraction: not available",
        `RNAi strongly dependent cell lines: ${rnaiStrongCount.toLocaleString()}`,
      ];

      if (dedupedPredictiveFeatures.length > 0) {
        keyFields.push("Top predictive features:");
        keyFields.push(
          ...dedupedPredictiveFeatures.map((row, idx) =>
            `${idx + 1}. ${row.featureType || "Feature"}: ${row.featureName}` +
            ` | importance ${row.featureImportance.toFixed(3)}` +
            `${Number.isFinite(row.correlation) ? ` | corr ${row.correlation.toFixed(3)}` : ""}` +
            `${row.relatedType ? ` | relation ${row.relatedType}` : ""}`
          )
        );
      }

      const sources = [
        `${DEPMAP_PORTAL_API}/tda/table_download`,
        dependencyRelease?.url || "",
        `${DEPMAP_PORTAL_API}/gene/${encodeURIComponent(symbol)}`,
      ].filter(Boolean);
      if (entityId && includePredictiveFeatures) {
        sources.push(`${DEPMAP_PORTAL_API}/gene/api/predictive?entityId=${encodeURIComponent(entityId)}`);
      }

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary:
              `${symbol}: DepMap CRISPR dependency in ${Number.isFinite(crisprDepFrac) ? formatPct(crisprDepFrac) : "unknown fraction"} ` +
              `of profiled cell lines${isPanDependency ? " with pan-dependency signal" : ""}${isStronglySelective ? " and strong selectivity" : ""}.`,
            keyFields,
            sources,
            limitations: [
              "Target-discovery metrics summarize the latest public DepMap release and are optimized for prioritization, not causal proof.",
              "Dependency metrics come from cancer cell-line screens; translational relevance depends on disease context and lineage.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_depmap_gene_dependency: ${error.message}` }] };
    }
  }
);

server.registerTool(
  "get_depmap_expression_subset_mean",
  {
    description:
      "Computes the mean log2(TPM+1) expression for one gene across a named DepMap model subset or molecular subtype " +
      "from a public release (for example RB1Loss / RB1_LoF in DepMap Public 25Q3). " +
      "Uses the public SubtypeMatrix plus DepMap expression downloads instead of BigQuery.",
    inputSchema: {
      geneSymbol: z.string().describe("Gene symbol to summarize in the DepMap expression matrix (for example 'MT-CO2')."),
      subtype: z.string().describe("DepMap subtype code or human-readable alias (for example 'RB1Loss' or 'RB1_LoF')."),
      release: z.string().optional().default("25Q3").describe("DepMap public release tag or label (for example '25Q3' or 'DepMap Public 25Q3')."),
      expressionDataset: z.enum(["protein_coding_stranded", "protein_coding", "all_genes_stranded", "all_genes"]).optional().default("protein_coding_stranded")
        .describe("Which DepMap public expression matrix to query. Defaults to the stranded protein-coding log2(TPM+1) matrix."),
      defaultProfileOnly: z.boolean().optional().default(true)
        .describe("Whether to restrict to rows marked IsDefaultEntryForModel in the public expression matrix."),
    },
  },
  async ({
    geneSymbol,
    subtype,
    release = "25Q3",
    expressionDataset = "protein_coding_stranded",
    defaultProfileOnly = true,
  }) => {
    const normalizedGene = normalizeWhitespace(geneSymbol || "").toUpperCase();
    const normalizedSubtype = normalizeWhitespace(subtype || "");
    const normalizedRelease = normalizeDepMapReleaseQuery(release || "") || "25Q3";
    const selectedDataset = DEPMAP_EXPRESSION_DATASET_FILES[expressionDataset] ? expressionDataset : "protein_coding_stranded";

    if (!normalizedGene || !normalizedSubtype) {
      return {
        content: [{
          type: "text",
          text: "Provide both a DepMap gene symbol and a subtype or model-subset label (for example MT-CO2 and RB1Loss).",
        }],
      };
    }

    try {
      const [catalogRows, subtypeTree, subtypeMatrix] = await Promise.all([
        fetchDepMapDownloadCatalog(),
        fetchDepMapSubtypeTree(normalizedRelease),
        fetchDepMapSubtypeMatrix(normalizedRelease),
      ]);
      const resolvedSubtype = resolveDepMapSubtypeCode(
        normalizedSubtype,
        subtypeTree.rows,
        subtypeMatrix.header
      );
      const subtypeCode = resolvedSubtype.code;
      if (!subtypeCode) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `DepMap subtype "${normalizedSubtype}" could not be resolved in release ${subtypeMatrix.release}.`,
              keyFields: [
                `Requested subtype: ${normalizedSubtype}`,
                `Release: ${subtypeMatrix.release}`,
              ],
              sources: [subtypeTree.sourceUrl, subtypeMatrix.sourceUrl].filter(Boolean),
              limitations: [
                "Subtype resolution uses the public DepMap subtype tree and matrix aliases.",
                "Try the exact subtype code from SubtypeTree, for example RB1_LoF.",
              ],
            }),
          }],
        };
      }

      const subtypeIndex = subtypeMatrix.header.findIndex(
        (value) => normalizeWhitespace(value) === subtypeCode
      );
      if (subtypeIndex < 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Subtype ${subtypeCode} was not present in the DepMap subtype matrix for release ${subtypeMatrix.release}.`,
              keyFields: [
                `Requested subtype: ${normalizedSubtype}`,
                `Resolved subtype code: ${subtypeCode}`,
                `Release: ${subtypeMatrix.release}`,
              ],
              sources: [subtypeTree.sourceUrl, subtypeMatrix.sourceUrl].filter(Boolean),
              limitations: ["Subtype availability depends on the public DepMap subtype matrix for the selected release."],
            }),
          }],
        };
      }

      const modelIds = new Set();
      for (const line of subtypeMatrix.lines.slice(1)) {
        const cols = parseCsvLine(line);
        const modelId = normalizeWhitespace(cols[0] || "");
        const membership = normalizeWhitespace(cols[subtypeIndex] || "");
        if (!modelId) continue;
        if (membership === "1" || membership === "1.0" || membership.toLowerCase() === "true") {
          modelIds.add(modelId);
        }
      }

      if (modelIds.size === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `Subtype ${subtypeCode} resolved in DepMap ${subtypeMatrix.release}, but no model memberships were found in the public subtype matrix.`,
              keyFields: [
                `Requested subtype: ${normalizedSubtype}`,
                `Resolved subtype code: ${subtypeCode}`,
                `Release: ${subtypeMatrix.release}`,
              ],
              sources: [subtypeTree.sourceUrl, subtypeMatrix.sourceUrl].filter(Boolean),
              limitations: ["The public subtype matrix did not contain any models marked for the resolved subtype."],
            }),
          }],
        };
      }

      const cacheKey = buildDepMapExpressionSubsetMeanCacheKey({
        geneSymbol: normalizedGene,
        subtypeCode,
        release: normalizedRelease,
        expressionDataset: selectedDataset,
        defaultProfileOnly,
      });
      const cached = getFreshCacheValue(depMapExpressionSubsetMeanCache.get(cacheKey), 6 * 60 * 60 * 1000);
      if (cached) {
        return cached;
      }

      const expressionFilename = DEPMAP_EXPRESSION_DATASET_FILES[selectedDataset];
      const expressionCatalogRow = findDepMapCatalogRow(catalogRows, expressionFilename, normalizedRelease);
      if (!expressionCatalogRow?.url) {
        throw new Error(`No DepMap expression file "${expressionFilename}" was found for release "${normalizedRelease}".`);
      }

      const response = await fetchWithRetry(expressionCatalogRow.url, {
        retries: 1,
        timeoutMs: 120000,
        maxBackoffMs: 6000,
        headers: {
          Accept: "text/csv",
          "User-Agent": "Mozilla/5.0 research-mcp",
        },
      });
      const { geneColumn, valueCount, valueSum, matchedModelCount } =
        await computeDepMapExpressionSubsetMeanFromResponse(response, {
          modelIds,
          geneSymbol: normalizedGene,
          defaultProfileOnly,
        });

      if (valueCount === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `DepMap ${expressionCatalogRow.release || normalizedRelease}: no finite ${geneColumn.label} expression values were found for subtype ${subtypeCode}.`,
              keyFields: [
                `Gene: ${normalizedGene}`,
                `Resolved subtype code: ${subtypeCode}`,
                `Expression file: ${expressionFilename}`,
                `Default model profiles only: ${defaultProfileOnly ? "yes" : "no"}`,
              ],
              sources: [subtypeTree.sourceUrl, subtypeMatrix.sourceUrl, expressionCatalogRow.url].filter(Boolean),
              limitations: [
                "The selected release and expression matrix did not yield any finite values after applying the subtype/model filters.",
              ],
            }),
          }],
        };
      }

      const meanValue = valueSum / valueCount;
      const payload = {
        structuredContent: {
          gene: normalizedGene,
          matchedGeneColumn: geneColumn.label,
          subtypeQuery: normalizedSubtype,
          subtypeCode,
          release: normalizeWhitespace(expressionCatalogRow.release || subtypeMatrix.release || normalizedRelease),
          expressionDataset: selectedDataset,
          defaultProfileOnly,
          subsetModelCount: modelIds.size,
          matchedExpressionRows: valueCount,
          matchedModelCount,
          meanLog2TpmPlus1: meanValue,
        },
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary:
              `DepMap ${normalizeWhitespace(expressionCatalogRow.release || subtypeMatrix.release || normalizedRelease)}: ` +
              `${normalizedGene} mean log2(TPM+1) across ${matchedModelCount.toLocaleString()} ${subtypeCode} model(s) is ${meanValue.toFixed(5)}.`,
            keyFields: [
              `Gene: ${normalizedGene} | Matched expression column: ${geneColumn.label}`,
              `Requested subtype: ${normalizedSubtype} | Resolved subtype code: ${subtypeCode}`,
              `Release: ${normalizeWhitespace(expressionCatalogRow.release || subtypeMatrix.release || normalizedRelease)}`,
              `Expression file: ${expressionFilename}`,
              `Subtype matrix memberships: ${modelIds.size.toLocaleString()} model(s)`,
              `Matched expression rows: ${valueCount.toLocaleString()} | Matched models with values: ${matchedModelCount.toLocaleString()}`,
              `Mean log2(TPM+1): ${meanValue.toFixed(5)}`,
              `Default model profiles only: ${defaultProfileOnly ? "yes" : "no"}`,
            ],
            sources: [subtypeTree.sourceUrl, subtypeMatrix.sourceUrl, expressionCatalogRow.url].filter(Boolean),
            limitations: [
              "Subtype aliases are resolved against the public DepMap subtype tree; RB1Loss maps to RB1_LoF in the public 25Q3 subtype tree.",
              "Expression means are computed from the selected public expression matrix and may differ from internal portal aggregations if a different profile-selection rule is used.",
            ],
          }),
        }],
      };
      depMapExpressionSubsetMeanCache.set(cacheKey, storeCacheValue(null, payload));
      return payload;
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_depmap_expression_subset_mean: ${error.message}` }] };
    }
  }
);

// ---------------------------------------------------------------------------
// GDSC / CancerRxGene — drug sensitivity pharmacogenomics
// ---------------------------------------------------------------------------

server.registerTool(
  "get_gdsc_drug_sensitivity",
  {
    description:
      "Summarizes GDSC / CancerRxGene drug sensitivity profiles for a compound across cancer cell lines, " +
      "including screened datasets, tissue-level sensitivity patterns, and the most sensitive profiled cell lines.",
    inputSchema: {
      drugQuery: z.string().describe("Drug name, synonym, brand name, or GDSC drug ID (e.g. 'Sorafenib', 'Nexavar', '30')."),
      dataset: z.enum(["all", "GDSC1", "GDSC2"]).optional().default("all")
        .describe("Restrict to one GDSC screen or aggregate across both when available."),
      tissue: z.string().optional().describe("Optional tissue or disease filter (e.g. 'lung', 'melanoma', 'LAML')."),
      topCellLineLimit: z.number().optional().default(5).describe("Maximum sensitive cell lines to return (1-10)."),
      topTissueLimit: z.number().optional().default(5).describe("Maximum tissues to summarize (1-10)."),
    },
  },
  async ({ drugQuery, dataset = "all", tissue = "", topCellLineLimit = 5, topTissueLimit = 5 }) => {
    const query = normalizeWhitespace(drugQuery || "");
    const tissueFilter = normalizeWhitespace(tissue || "");
    const boundedCellLimit = Math.max(1, Math.min(10, Math.round(topCellLineLimit || 5)));
    const boundedTissueLimit = Math.max(1, Math.min(10, Math.round(topTissueLimit || 5)));

    if (!query) {
      return { content: [{ type: "text", text: "Provide a drug name, synonym, brand name, or GDSC drug ID (e.g. Sorafenib)." }] };
    }

    try {
      const catalogRows = await fetchGdscCompoundCatalog();
      const selection = resolveGdscCompoundSelection(catalogRows, query, dataset);

      if (selection.selectedRows.length === 0) {
        const datasetNote =
          dataset !== "all" && selection.availableDatasets.length > 0
            ? ` Available datasets for the matched compound: ${selection.availableDatasets.join(", ")}.`
            : "";
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `GDSC found no drug sensitivity profile for "${query}"${dataset === "all" ? "" : ` in ${dataset}`}.`,
              keyFields: [
                `Query: ${query}`,
                selection.alternates.length > 0 ? `Nearest compound names: ${selection.alternates.join(", ")}` : "Nearest compound names: none",
              ],
              sources: ["https://www.cancerrxgene.org/compounds"],
              limitations: [
                `Compound matching uses the public GDSC / CancerRxGene compound catalog.${datasetNote}`.trim(),
                "Some compounds appear under screening-specific IDs rather than a single canonical identifier.",
              ],
            }),
          }],
        };
      }

      const overviewPayloads = await Promise.all(
        selection.selectedRows.map(async (row) => {
          const records = await fetchGdscOverviewData(row.drug_id, row.dataset);
          return { row, records };
        })
      );

      const normalizedRecords = overviewPayloads
        .flatMap(({ row, records }) =>
          (Array.isArray(records) ? records : []).map((record) => ({
            dataset: row.dataset,
            drug_id: row.drug_id,
            compound_name: row.name,
            cell_name: normalizeWhitespace(record?.cell_name || ""),
            cell_id: normalizeWhitespace(record?.cell_id || record?.cosmic_id || ""),
            tcga: normalizeWhitespace(record?.tcga || ""),
            tissue_name: getGdscTissueLabel(record),
            gdsc_desc1: normalizeWhitespace(String(record?.gdsc_desc1 || "").replace(/_/g, " ")),
            gdsc_desc2: normalizeWhitespace(String(record?.gdsc_desc2 || "").replace(/_/g, " ")),
            ic50: toNullableNumber(record?.ic50),
            auc: toNullableNumber(record?.auc),
          }))
        )
        .filter((row) => row.cell_name);

      if (normalizedRecords.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${selection.matchedName}: GDSC located compound metadata, but no public sensitivity records were returned for the requested dataset selection.`,
              keyFields: [
                `Compound: ${selection.matchedName}`,
                `Datasets requested: ${selection.selectedRows.map((row) => row.dataset).join(", ")}`,
              ],
              sources: selection.selectedRows
                .map((row) => `https://www.cancerrxgene.org/compound/${encodeURIComponent(row.name)}/${row.drug_id}/by-tissue`)
                .slice(0, 4),
              limitations: [
                "Some compounds in the public catalog have sparse or retired screening payloads.",
                "Try a different dataset selection or a more common compound name if you expected results.",
              ],
            }),
          }],
        };
      }

      const filteredRecords =
        !tissueFilter
          ? normalizedRecords
          : normalizedRecords.filter((record) => {
            const tokens = tokenizeQuery(tissueFilter);
            const haystack = buildGdscRecordText(record);
            if (tokens.length > 0) {
              return matchesAllTokens(haystack, tokens);
            }
            return haystack.includes(tissueFilter.toLowerCase());
          });

      if (filteredRecords.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${selection.matchedName}: no GDSC sensitivity records matched tissue filter "${tissueFilter}".`,
              keyFields: [
                `Compound: ${selection.matchedName}`,
                `Datasets searched: ${selection.selectedRows.map((row) => `${row.dataset} (drug ID ${row.drug_id})`).join(", ")}`,
              ],
              sources: selection.selectedRows
                .map((row) => `https://www.cancerrxgene.org/compound/${encodeURIComponent(row.name)}/${row.drug_id}/by-tissue`)
                .slice(0, 4),
              limitations: [
                "Tissue matching is text-based over GDSC tissue labels and TCGA annotations.",
                "Try a broader tissue term (e.g. 'lung' instead of 'lung squamous').",
              ],
            }),
          }],
        };
      }

      const uniqueRecordMap = new Map();
      for (const record of filteredRecords) {
        const key = `${record.dataset}::${record.cell_id || record.cell_name}`;
        if (!uniqueRecordMap.has(key)) {
          uniqueRecordMap.set(key, record);
        }
      }
      const uniqueRecords = [...uniqueRecordMap.values()];

      const sortedSensitiveRecords = uniqueRecords
        .filter((record) => Number.isFinite(record.auc) || Number.isFinite(record.ic50))
        .sort((a, b) => {
          const aucA = Number.isFinite(a.auc) ? a.auc : Number.POSITIVE_INFINITY;
          const aucB = Number.isFinite(b.auc) ? b.auc : Number.POSITIVE_INFINITY;
          if (aucA !== aucB) return aucA - aucB;
          const ic50A = Number.isFinite(a.ic50) ? a.ic50 : Number.POSITIVE_INFINITY;
          const ic50B = Number.isFinite(b.ic50) ? b.ic50 : Number.POSITIVE_INFINITY;
          return ic50A - ic50B;
        });

      const byTissue = new Map();
      for (const record of uniqueRecords) {
        const key = record.tissue_name || "Unspecified tissue";
        if (!byTissue.has(key)) {
          byTissue.set(key, []);
        }
        byTissue.get(key).push(record);
      }

      const tissueEntries = [...byTissue.entries()]
        .map(([label, records]) => ({
          label,
          count: records.length,
          medianAuc: median(records.map((record) => record.auc)),
          medianIc50: median(records.map((record) => record.ic50)),
        }))
        .sort((a, b) => {
          const aucA = Number.isFinite(a.medianAuc) ? a.medianAuc : Number.POSITIVE_INFINITY;
          const aucB = Number.isFinite(b.medianAuc) ? b.medianAuc : Number.POSITIVE_INFINITY;
          if (aucA !== aucB) return aucA - aucB;
          const ic50A = Number.isFinite(a.medianIc50) ? a.medianIc50 : Number.POSITIVE_INFINITY;
          const ic50B = Number.isFinite(b.medianIc50) ? b.medianIc50 : Number.POSITIVE_INFINITY;
          return ic50A - ic50B;
        });

      const medianAucAll = median(uniqueRecords.map((record) => record.auc));
      const medianIc50All = median(uniqueRecords.map((record) => record.ic50));
      const datasetLines = selection.selectedRows.map((row) =>
        `${row.dataset}: drug ID ${row.drug_id}, ${row.cell_line_count.toLocaleString()} screened cell lines` +
        `${row.screening_site ? ` (${row.screening_site})` : ""}`
      );
      const sensitiveCellLines = sortedSensitiveRecords.slice(0, boundedCellLimit).map((record, idx) =>
        `${idx + 1}. ${record.cell_name} [${record.dataset}] | ${record.tissue_name}` +
        `${Number.isFinite(record.ic50) ? ` | IC50 ${record.ic50.toFixed(4)}` : ""}` +
        `${Number.isFinite(record.auc) ? ` | AUC ${record.auc.toFixed(4)}` : ""}`
      );
      const tissueLines = tissueEntries.slice(0, boundedTissueLimit).map((entry, idx) =>
        `${idx + 1}. ${entry.label} | n=${entry.count}` +
        `${Number.isFinite(entry.medianAuc) ? ` | median AUC ${entry.medianAuc.toFixed(4)}` : ""}` +
        `${Number.isFinite(entry.medianIc50) ? ` | median IC50 ${entry.medianIc50.toFixed(4)}` : ""}`
      );
      const bestRecord = sortedSensitiveRecords[0] || null;

      const keyFields = [
        `Compound: ${selection.matchedName}`,
        `Queried as: ${query}`,
        `Datasets: ${selection.selectedRows.map((row) => row.dataset).join(", ")}`,
        `Matched screening IDs: ${selection.selectedRows.map((row) => `${row.drug_id} (${row.dataset})`).join(", ")}`,
        `Unique profiled cell lines: ${uniqueRecords.length.toLocaleString()}`,
        Number.isFinite(medianAucAll) ? `Median AUC: ${medianAucAll.toFixed(4)}` : "Median AUC: not available",
        Number.isFinite(medianIc50All) ? `Median IC50: ${medianIc50All.toFixed(4)}` : "Median IC50: not available",
      ];
      if (tissueFilter) {
        keyFields.push(`Tissue filter: ${tissueFilter}`);
      }
      if (selection.selectedRows.some((row) => row.targets.length > 0)) {
        keyFields.push(
          `Targets: ${dedupeArray(selection.selectedRows.flatMap((row) => row.targets)).slice(0, 10).join(", ")}`
        );
      }
      if (selection.selectedRows.some((row) => row.target_pathway)) {
        keyFields.push(
          `Target pathway(s): ${dedupeArray(selection.selectedRows.map((row) => row.target_pathway).filter(Boolean)).join(", ")}`
        );
      }
      keyFields.push("Dataset coverage:");
      keyFields.push(...datasetLines);
      if (tissueLines.length > 0) {
        keyFields.push("Most sensitive tissues:");
        keyFields.push(...tissueLines);
      }
      if (sensitiveCellLines.length > 0) {
        keyFields.push("Most sensitive cell lines:");
        keyFields.push(...sensitiveCellLines);
      }
      if (selection.alternates.length > 0) {
        keyFields.push(`Other nearby compound names: ${selection.alternates.join(", ")}`);
      }

      const summary =
        `${selection.matchedName}: GDSC sensitivity data across ${uniqueRecords.length.toLocaleString()} profiled cell-line records` +
        `${tissueFilter ? ` matching "${tissueFilter}"` : ""} in ${selection.selectedRows.map((row) => row.dataset).join(", ")}.` +
        (bestRecord
          ? ` Most sensitive observed line: ${bestRecord.cell_name} [${bestRecord.dataset}]` +
            `${Number.isFinite(bestRecord.ic50) ? ` (IC50 ${bestRecord.ic50.toFixed(4)})` : ""}` +
            `${Number.isFinite(bestRecord.auc) ? `, AUC ${bestRecord.auc.toFixed(4)}` : ""}.`
          : "");

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary,
            keyFields,
            sources: selection.selectedRows
              .map((row) => `https://www.cancerrxgene.org/compound/${encodeURIComponent(row.name)}/${row.drug_id}/by-tissue`)
              .slice(0, 4),
            limitations: [
              "GDSC sensitivity is measured in cancer cell lines and does not directly estimate patient response.",
              "GDSC1 and GDSC2 use different screening designs and concentration ranges; cross-dataset comparisons are directional rather than perfectly harmonized.",
              "AUC and IC50 summarize in vitro response; mechanism, exposure, and resistance in vivo can differ substantially.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_gdsc_drug_sensitivity: ${error.message}` }] };
    }
  }
);

server.registerTool(
  "get_prism_repurposing_response",
  {
    description:
      "Summarizes Broad PRISM repurposing primary-screen response for a compound across pooled cancer cell lines, " +
      "including single-dose log2-fold-change viability patterns, top sensitive tissues, and the most sensitive profiled cell lines.",
    inputSchema: {
      drugQuery: z.string().describe("Drug name, synonym, or PRISM BRD identifier (e.g. 'Sorafenib', 'Paclitaxel', 'BRD:BRD-K23984367-075-15-2')."),
      screen: z.string().optional().describe("Optional PRISM screen filter such as 'REP.PRIMARY' or 'REP.300'."),
      tissue: z.string().optional().describe("Optional tissue filter (e.g. 'lung', 'skin', 'pancreas')."),
      topCellLineLimit: z.number().optional().default(5).describe("Maximum sensitive cell lines to return (1-10)."),
      topTissueLimit: z.number().optional().default(5).describe("Maximum tissues to summarize (1-10)."),
    },
  },
  async ({ drugQuery, screen = "", tissue = "", topCellLineLimit = 5, topTissueLimit = 5 }) => {
    const query = normalizeWhitespace(drugQuery || "");
    const screenFilter = normalizeWhitespace(screen || "");
    const tissueFilter = normalizeWhitespace(tissue || "");
    const boundedCellLimit = Math.max(1, Math.min(10, Math.round(topCellLineLimit || 5)));
    const boundedTissueLimit = Math.max(1, Math.min(10, Math.round(topTissueLimit || 5)));

    if (!query) {
      return { content: [{ type: "text", text: "Provide a drug name, synonym, or PRISM BRD identifier (e.g. Sorafenib)." }] };
    }

    try {
      const catalogRows = await fetchPrismCompoundCatalog();
      const selection = resolvePrismCompoundSelection(catalogRows, query, screenFilter);

      if (selection.selectedRows.length === 0) {
        const screenNote =
          screenFilter && selection.availableScreens.length > 0
            ? ` Available screens for the matched compound: ${selection.availableScreens.join(", ")}.`
            : "";
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `PRISM found no repurposing response profile for "${query}"${screenFilter ? ` in screen ${screenFilter}` : ""}.`,
              keyFields: [
                `Query: ${query}`,
                selection.alternates.length > 0 ? `Nearest compound names: ${selection.alternates.join(", ")}` : "Nearest compound names: none",
              ],
              sources: [PRISM_24Q2_FIGSHARE_ARTICLE],
              limitations: [
                `Compound matching uses the Broad PRISM Repurposing Public 24Q2 compound metadata.${screenNote}`.trim(),
                "Some compounds may appear under a BRD identifier or under screen-specific naming conventions.",
              ],
            }),
          }],
        };
      }

      const [matrixPayload, cellLineRows] = await Promise.all([
        fetchPrismPrimaryMatrix(),
        fetchPrismCellLineCatalog(),
      ]);
      const rowMap = extractPrismMatrixRows(matrixPayload, selection.selectedRows.map((row) => row.ids));

      if (rowMap.size === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${selection.matchedName}: PRISM matched compound metadata, but no matrix rows were recovered from the current 24Q2 primary-screen release.`,
              keyFields: [
                `Compound: ${selection.matchedName}`,
                `Matched perturbation IDs: ${selection.selectedRows.map((row) => row.ids).join(", ")}`,
              ],
              sources: [PRISM_24Q2_FIGSHARE_ARTICLE],
              limitations: [
                "The public matrix is a collapsed single-dose viability screen and may not include every metadata row returned by the companion file.",
                "Try a different synonym or a direct BRD identifier if you expected a hit.",
              ],
            }),
          }],
        };
      }

      const cellLineMap = new Map(
        cellLineRows.map((row) => [row.depmap_id, row])
      );
      const rawRecords = [];
      for (const selectedRow of selection.selectedRows) {
        const values = rowMap.get(selectedRow.ids);
        if (!Array.isArray(values)) continue;
        matrixPayload.depmapIds.forEach((depmapId, idx) => {
          const lfc = values[idx];
          if (!Number.isFinite(lfc)) return;
          const cellMeta = cellLineMap.get(depmapId) || {};
          rawRecords.push({
            ids: selectedRow.ids,
            compound_name: selectedRow.name,
            screen: selectedRow.screen,
            dose: toNullableNumber(selectedRow.dose),
            depmap_id: depmapId,
            ccle_name: normalizeWhitespace(cellMeta.ccle_name || depmapId),
            cell_name: normalizeWhitespace(cellMeta.cell_line_name || depmapId),
            tissue_name: normalizeWhitespace(cellMeta.tissue_name || "Unspecified tissue"),
            lfc,
          });
        });
      }

      if (rawRecords.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${selection.matchedName}: no PRISM response values were available after joining compound metadata to the 24Q2 primary matrix.`,
              keyFields: [
                `Compound: ${selection.matchedName}`,
                `Matched perturbation IDs: ${selection.selectedRows.map((row) => row.ids).join(", ")}`,
              ],
              sources: [PRISM_24Q2_FIGSHARE_ARTICLE],
              limitations: [
                "The PRISM primary matrix is large and only includes numeric response values for profiled compound-cell-line combinations.",
              ],
            }),
          }],
        };
      }

      const filteredRecords =
        !tissueFilter
          ? rawRecords
          : rawRecords.filter((record) => {
            const tokens = tokenizeQuery(tissueFilter);
            const haystack = buildPrismRecordText(record);
            if (tokens.length > 0) {
              return matchesAllTokens(haystack, tokens);
            }
            return haystack.includes(tissueFilter.toLowerCase());
          });

      if (filteredRecords.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${selection.matchedName}: no PRISM response records matched tissue filter "${tissueFilter}".`,
              keyFields: [
                `Compound: ${selection.matchedName}`,
                `Matched screens: ${dedupeArray(selection.selectedRows.map((row) => row.screen)).join(", ") || "none"}`,
              ],
              sources: [PRISM_24Q2_FIGSHARE_ARTICLE],
              limitations: [
                "Tissue matching is text-based over PRISM CCLE cell-line names and inferred tissue labels.",
                "Try a broader tissue term (for example 'lung' instead of a subtype).",
              ],
            }),
          }],
        };
      }

      const uniqueRecordMap = new Map();
      for (const record of filteredRecords) {
        const key = record.depmap_id || record.cell_name;
        if (!uniqueRecordMap.has(key)) {
          uniqueRecordMap.set(key, {
            ...record,
            lfcs: [],
            ids: new Set(),
            screens: new Set(),
          });
        }
        const entry = uniqueRecordMap.get(key);
        entry.lfcs.push(record.lfc);
        entry.ids.add(record.ids);
        entry.screens.add(record.screen);
      }

      const uniqueRecords = [...uniqueRecordMap.values()]
        .map((entry) => ({
          ...entry,
          median_lfc: median(entry.lfcs.filter(Number.isFinite)),
          ids: [...entry.ids],
          screens: [...entry.screens].filter(Boolean),
        }))
        .filter((entry) => Number.isFinite(entry.median_lfc))
        .sort((a, b) => a.median_lfc - b.median_lfc);

      if (uniqueRecords.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${selection.matchedName}: PRISM matched records, but no finite single-dose LFC values survived filtering.`,
              keyFields: [`Compound: ${selection.matchedName}`],
              sources: [PRISM_24Q2_FIGSHARE_ARTICLE],
              limitations: [
                "The PRISM primary screen reports collapsed single-dose log2-fold-change values; some compound-cell-line combinations may be missing or non-finite.",
              ],
            }),
          }],
        };
      }

      const byTissue = new Map();
      for (const record of uniqueRecords) {
        const key = record.tissue_name || "Unspecified tissue";
        if (!byTissue.has(key)) {
          byTissue.set(key, []);
        }
        byTissue.get(key).push(record);
      }
      const tissueEntries = [...byTissue.entries()]
        .map(([label, records]) => ({
          label,
          count: records.length,
          medianLfc: median(records.map((record) => record.median_lfc).filter(Number.isFinite)),
        }))
        .filter((entry) => Number.isFinite(entry.medianLfc))
        .sort((a, b) => a.medianLfc - b.medianLfc);

      const overallMedianLfc = median(uniqueRecords.map((record) => record.median_lfc).filter(Number.isFinite));
      const bestRecord = uniqueRecords[0] || null;
      const screenMap = new Map();
      for (const row of selection.selectedRows) {
        const key = row.screen || row.ids;
        if (!screenMap.has(key)) {
          screenMap.set(key, {
            screen: row.screen || "Unknown screen",
            dose: row.dose,
            ids: [],
          });
        }
        screenMap.get(key).ids.push(row.ids);
      }
      const screenEntries = [...screenMap.values()];

      const keyFields = [
        `Compound: ${selection.matchedName}`,
        `Queried as: ${query}`,
        `Matched PRISM perturbation IDs: ${selection.selectedRows.map((row) => row.ids).join(", ")}`,
        `PRISM screens: ${dedupeArray(selection.selectedRows.map((row) => row.screen)).join(", ") || "unknown"}`,
        `Unique profiled cell lines: ${uniqueRecords.length.toLocaleString()}`,
        Number.isFinite(overallMedianLfc) ? `Median single-dose LFC: ${overallMedianLfc.toFixed(3)}` : "Median single-dose LFC: not available",
      ];
      if (screenFilter) {
        keyFields.push(`Screen filter: ${screenFilter}`);
      }
      if (tissueFilter) {
        keyFields.push(`Tissue filter: ${tissueFilter}`);
      }
      const targets = dedupeArray(
        selection.selectedRows
          .flatMap((row) => normalizeWhitespace(row.repurposing_target || "").split(","))
          .map((value) => normalizeWhitespace(value))
          .filter(Boolean)
      ).slice(0, 12);
      if (targets.length > 0) {
        keyFields.push(`Repurposing targets: ${targets.join(", ")}`);
      }
      const moas = dedupeArray(selection.selectedRows.map((row) => normalizeWhitespace(row.moa || "")).filter(Boolean)).slice(0, 6);
      if (moas.length > 0) {
        keyFields.push(`Mechanism(s): ${moas.join(" | ")}`);
      }
      if (screenEntries.length > 0) {
        keyFields.push("Matched PRISM screen rows:");
        keyFields.push(
          ...screenEntries.slice(0, 6).map((entry, idx) =>
            `${idx + 1}. ${entry.screen}` +
            `${Number.isFinite(toNullableNumber(entry.dose)) ? ` | dose ${Number(entry.dose).toFixed(2)}` : ""}` +
            ` | perturbation IDs ${dedupeArray(entry.ids).slice(0, 4).join(", ")}`
          )
        );
      }
      if (tissueEntries.length > 0) {
        keyFields.push("Most sensitive tissues:");
        keyFields.push(
          ...tissueEntries.slice(0, boundedTissueLimit).map((entry, idx) =>
            `${idx + 1}. ${entry.label} | n=${entry.count} | median LFC ${entry.medianLfc.toFixed(3)}`
          )
        );
      }
      keyFields.push("Most sensitive cell lines:");
      keyFields.push(
        ...uniqueRecords.slice(0, boundedCellLimit).map((record, idx) =>
          `${idx + 1}. ${record.cell_name} | ${record.tissue_name}` +
          ` | median LFC ${record.median_lfc.toFixed(3)}` +
          `${record.screens.length > 0 ? ` | screens ${record.screens.join(", ")}` : ""}`
        )
      );
      if (selection.alternates.length > 0) {
        keyFields.push(`Other nearby compound names: ${selection.alternates.join(", ")}`);
      }

      const summary =
        `${selection.matchedName}: PRISM 24Q2 single-dose response across ${uniqueRecords.length.toLocaleString()} profiled cell lines` +
        `${tissueFilter ? ` matching "${tissueFilter}"` : ""}.` +
        (bestRecord
          ? ` Most negative median LFC: ${bestRecord.cell_name} (${bestRecord.tissue_name}) at ${bestRecord.median_lfc.toFixed(3)}.`
          : "");

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary,
            keyFields,
            sources: [
              PRISM_24Q2_FIGSHARE_ARTICLE,
              PRISM_24Q2_COMPOUND_LIST_URL,
              PRISM_24Q2_PRIMARY_MATRIX_URL,
            ],
            limitations: [
              "PRISM primary-screen values are collapsed single-dose log2-fold-change viability measurements rather than full dose-response curves.",
              "More negative LFC values indicate stronger viability reduction relative to negative-control wells.",
              "Tissue labels are inferred from PRISM/CCLE cell-line metadata and may be coarser than disease-specific histology labels.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_prism_repurposing_response: ${error.message}` }] };
    }
  }
);

server.registerTool(
  "get_pharmacodb_compound_response",
  {
    description:
      "Summarizes PharmacoDB compound-response evidence across multiple public pharmacogenomic datasets, " +
      "including dataset coverage, tissue-level sensitivity patterns, and the most sensitive profiled cell lines.",
    inputSchema: {
      drugQuery: z.string().describe("Drug name or PharmacoDB compound UID (e.g. 'Paclitaxel', 'Erlotinib', 'PDBC00058')."),
      dataset: z.string().optional().describe("Optional PharmacoDB dataset filter (e.g. 'GDSC2', 'PRISM', 'CTRPv2')."),
      tissue: z.string().optional().describe("Optional tissue filter (e.g. 'lung', 'breast', 'skin')."),
      topCellLineLimit: z.number().optional().default(5).describe("Maximum sensitive cell lines to return (1-10)."),
      topTissueLimit: z.number().optional().default(5).describe("Maximum tissues to summarize (1-10)."),
    },
  },
  async ({ drugQuery, dataset = "", tissue = "", topCellLineLimit = 5, topTissueLimit = 5 }) => {
    const query = normalizeWhitespace(drugQuery || "");
    const datasetFilter = normalizeWhitespace(dataset || "");
    const tissueFilter = normalizeWhitespace(tissue || "");
    const boundedCellLimit = Math.max(1, Math.min(10, Math.round(topCellLineLimit || 5)));
    const boundedTissueLimit = Math.max(1, Math.min(10, Math.round(topTissueLimit || 5)));

    if (!query) {
      return { content: [{ type: "text", text: "Provide a drug name or PharmacoDB compound UID (e.g. Paclitaxel)." }] };
    }

    try {
      const resolution = await resolvePharmacoDbCompound(query);
      const compound = resolution.compound;
      if (!compound) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `PharmacoDB found no compound profile for "${query}".`,
              keyFields: [
                `Query: ${query}`,
                resolution.alternates.length > 0 ? `Nearest compound names: ${resolution.alternates.join(", ")}` : "Nearest compound names: none",
              ],
              sources: ["https://pharmacodb.ca/", PHARMACODB_GRAPHQL_API],
              limitations: [
                "Compound matching relies on PharmacoDB's current compound search index.",
                "Brand-name or abbreviation queries may require the canonical compound name or a PharmacoDB UID.",
              ],
            }),
          }],
        };
      }

      let experiments =
        tissueFilter
          ? await fetchPharmacoDbExperiments(compound.id, tissueFilter).catch(() => [])
          : [];
      if (!tissueFilter || experiments.length === 0) {
        experiments = await fetchPharmacoDbExperiments(compound.id, "");
      }

      const normalizedRecords = (Array.isArray(experiments) ? experiments : [])
        .map((row) => ({
          dataset: normalizeWhitespace(row?.dataset?.name || ""),
          tissue_name: normalizeWhitespace(row?.tissue?.name || ""),
          cell_name: normalizeWhitespace(row?.cell_line?.name || ""),
          cell_uid: normalizeWhitespace(row?.cell_line?.uid || row?.cell_line?.name || ""),
          aac: toNullableNumber(row?.profile?.AAC),
          ic50: toNullableNumber(row?.profile?.IC50),
          ec50: toNullableNumber(row?.profile?.EC50),
          einf: toNullableNumber(row?.profile?.Einf),
          dss1: toNullableNumber(row?.profile?.DSS1),
          dss2: toNullableNumber(row?.profile?.DSS2),
          dss3: toNullableNumber(row?.profile?.DSS3),
          hs: toNullableNumber(row?.profile?.HS),
        }))
        .filter((row) => row.dataset && row.cell_name);

      if (normalizedRecords.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${compound.name}: PharmacoDB resolved the compound, but no experiment records were returned.`,
              keyFields: [
                `Compound: ${compound.name}`,
                `Compound UID: ${compound.uid}`,
              ],
              sources: [
                compound.uid ? `https://pharmacodb.ca/compounds/${encodeURIComponent(compound.uid)}` : "https://pharmacodb.ca/",
                PHARMACODB_GRAPHQL_API,
              ],
              limitations: [
                "Not every PharmacoDB compound has public experiment records exposed through the current GraphQL endpoint.",
              ],
            }),
          }],
        };
      }

      const datasetFiltered =
        !datasetFilter
          ? normalizedRecords
          : normalizedRecords.filter((record) => {
            const tokens = tokenizeQuery(datasetFilter);
            const haystack = normalizeWhitespace(record.dataset).toLowerCase();
            if (tokens.length > 0) {
              return matchesAllTokens(haystack, tokens);
            }
            return haystack.includes(datasetFilter.toLowerCase());
          });

      if (datasetFiltered.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${compound.name}: no PharmacoDB experiment records matched dataset filter "${datasetFilter}".`,
              keyFields: [
                `Compound: ${compound.name}`,
                `Available datasets: ${dedupeArray(normalizedRecords.map((record) => record.dataset)).slice(0, 12).join(", ")}`,
              ],
              sources: [
                compound.uid ? `https://pharmacodb.ca/compounds/${encodeURIComponent(compound.uid)}` : "https://pharmacodb.ca/",
                PHARMACODB_GRAPHQL_API,
              ],
              limitations: [
                "Dataset matching is text-based over PharmacoDB dataset names.",
              ],
            }),
          }],
        };
      }

      const filteredRecords =
        !tissueFilter
          ? datasetFiltered
          : datasetFiltered.filter((record) => {
            const tokens = tokenizeQuery(tissueFilter);
            const haystack = buildPharmacoDbRecordText(record);
            if (tokens.length > 0) {
              return matchesAllTokens(haystack, tokens);
            }
            return haystack.includes(tissueFilter.toLowerCase());
          });

      if (filteredRecords.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `${compound.name}: no PharmacoDB experiment records matched tissue filter "${tissueFilter}".`,
              keyFields: [
                `Compound: ${compound.name}`,
                `Datasets searched: ${dedupeArray(datasetFiltered.map((record) => record.dataset)).slice(0, 12).join(", ")}`,
              ],
              sources: [
                compound.uid ? `https://pharmacodb.ca/compounds/${encodeURIComponent(compound.uid)}` : "https://pharmacodb.ca/",
                PHARMACODB_GRAPHQL_API,
              ],
              limitations: [
                "Tissue matching is text-based over PharmacoDB tissue labels.",
                "Try a broader tissue term if you expected a match.",
              ],
            }),
          }],
        };
      }

      const uniqueRecordMap = new Map();
      for (const record of filteredRecords) {
        const key = `${record.dataset}::${record.cell_uid || record.cell_name}`;
        if (!uniqueRecordMap.has(key)) {
          uniqueRecordMap.set(key, {
            ...record,
            aacValues: [],
            ic50Values: [],
            ec50Values: [],
            einfValues: [],
            dss1Values: [],
            dss2Values: [],
            dss3Values: [],
            hsValues: [],
          });
        }
        const entry = uniqueRecordMap.get(key);
        if (Number.isFinite(record.aac)) entry.aacValues.push(record.aac);
        if (Number.isFinite(record.ic50)) entry.ic50Values.push(record.ic50);
        if (Number.isFinite(record.ec50)) entry.ec50Values.push(record.ec50);
        if (Number.isFinite(record.einf)) entry.einfValues.push(record.einf);
        if (Number.isFinite(record.dss1)) entry.dss1Values.push(record.dss1);
        if (Number.isFinite(record.dss2)) entry.dss2Values.push(record.dss2);
        if (Number.isFinite(record.dss3)) entry.dss3Values.push(record.dss3);
        if (Number.isFinite(record.hs)) entry.hsValues.push(record.hs);
      }

      const uniqueRecords = [...uniqueRecordMap.values()]
        .map((entry) => ({
          dataset: entry.dataset,
          tissue_name: entry.tissue_name,
          cell_name: entry.cell_name,
          cell_uid: entry.cell_uid,
          aac: median(entry.aacValues),
          ic50: median(entry.ic50Values),
          ec50: median(entry.ec50Values),
          einf: median(entry.einfValues),
          dss1: median(entry.dss1Values),
          dss2: median(entry.dss2Values),
          dss3: median(entry.dss3Values),
          hs: median(entry.hsValues),
        }))
        .filter((record) =>
          Number.isFinite(record.aac) ||
          Number.isFinite(record.ic50) ||
          Number.isFinite(record.ec50) ||
          Number.isFinite(record.dss1)
        )
        .sort((a, b) => {
          const aacA = Number.isFinite(a.aac) ? -a.aac : Number.POSITIVE_INFINITY;
          const aacB = Number.isFinite(b.aac) ? -b.aac : Number.POSITIVE_INFINITY;
          if (aacA !== aacB) return aacA - aacB;
          const dssA = Number.isFinite(a.dss1) ? -a.dss1 : Number.POSITIVE_INFINITY;
          const dssB = Number.isFinite(b.dss1) ? -b.dss1 : Number.POSITIVE_INFINITY;
          if (dssA !== dssB) return dssA - dssB;
          const ic50A = Number.isFinite(a.ic50) ? a.ic50 : Number.POSITIVE_INFINITY;
          const ic50B = Number.isFinite(b.ic50) ? b.ic50 : Number.POSITIVE_INFINITY;
          return ic50A - ic50B;
        });

      const byDataset = new Map();
      const byTissue = new Map();
      for (const record of uniqueRecords) {
        if (!byDataset.has(record.dataset)) byDataset.set(record.dataset, []);
        if (!byTissue.has(record.tissue_name || "Unspecified tissue")) byTissue.set(record.tissue_name || "Unspecified tissue", []);
        byDataset.get(record.dataset).push(record);
        byTissue.get(record.tissue_name || "Unspecified tissue").push(record);
      }

      const datasetEntries = [...byDataset.entries()]
        .map(([label, records]) => ({
          label,
          count: records.length,
          medianAac: median(records.map((record) => record.aac).filter(Number.isFinite)),
          medianIc50: median(records.map((record) => record.ic50).filter(Number.isFinite)),
        }))
        .sort((a, b) => {
          const aacA = Number.isFinite(a.medianAac) ? -a.medianAac : Number.POSITIVE_INFINITY;
          const aacB = Number.isFinite(b.medianAac) ? -b.medianAac : Number.POSITIVE_INFINITY;
          if (aacA !== aacB) return aacA - aacB;
          const ic50A = Number.isFinite(a.medianIc50) ? a.medianIc50 : Number.POSITIVE_INFINITY;
          const ic50B = Number.isFinite(b.medianIc50) ? b.medianIc50 : Number.POSITIVE_INFINITY;
          return ic50A - ic50B;
        });
      const tissueEntries = [...byTissue.entries()]
        .map(([label, records]) => ({
          label,
          count: records.length,
          medianAac: median(records.map((record) => record.aac).filter(Number.isFinite)),
          medianIc50: median(records.map((record) => record.ic50).filter(Number.isFinite)),
        }))
        .sort((a, b) => {
          const aacA = Number.isFinite(a.medianAac) ? -a.medianAac : Number.POSITIVE_INFINITY;
          const aacB = Number.isFinite(b.medianAac) ? -b.medianAac : Number.POSITIVE_INFINITY;
          if (aacA !== aacB) return aacA - aacB;
          const ic50A = Number.isFinite(a.medianIc50) ? a.medianIc50 : Number.POSITIVE_INFINITY;
          const ic50B = Number.isFinite(b.medianIc50) ? b.medianIc50 : Number.POSITIVE_INFINITY;
          return ic50A - ic50B;
        });

      const overallMedianAac = median(uniqueRecords.map((record) => record.aac).filter(Number.isFinite));
      const overallMedianIc50 = median(uniqueRecords.map((record) => record.ic50).filter(Number.isFinite));
      const bestRecord = uniqueRecords[0] || null;

      const keyFields = [
        `Compound: ${compound.name}`,
        `Queried as: ${query}`,
        `Compound UID: ${compound.uid}`,
        `Unique profiled cell-line records: ${uniqueRecords.length.toLocaleString()}`,
        `Datasets represented: ${datasetEntries.map((entry) => entry.label).join(", ")}`,
        Number.isFinite(overallMedianAac) ? `Median AAC: ${overallMedianAac.toFixed(4)}` : "Median AAC: not available",
        Number.isFinite(overallMedianIc50) ? `Median IC50: ${overallMedianIc50.toFixed(4)}` : "Median IC50: not available",
      ];
      if (datasetFilter) {
        keyFields.push(`Dataset filter: ${datasetFilter}`);
      }
      if (tissueFilter) {
        keyFields.push(`Tissue filter: ${tissueFilter}`);
      }
      if (normalizeWhitespace(compound?.annotation?.chembl || "")) {
        keyFields.push(`ChEMBL: ${normalizeWhitespace(compound.annotation.chembl)}`);
      }
      if (normalizeWhitespace(compound?.annotation?.pubchem || "")) {
        keyFields.push(`PubChem: ${normalizeWhitespace(compound.annotation.pubchem)}`);
      }
      if (normalizeWhitespace(compound?.annotation?.fda_status || "")) {
        keyFields.push(`FDA status: ${normalizeWhitespace(compound.annotation.fda_status)}`);
      }
      if (datasetEntries.length > 0) {
        keyFields.push("Dataset coverage:");
        keyFields.push(
          ...datasetEntries.slice(0, 10).map((entry, idx) =>
            `${idx + 1}. ${entry.label} | n=${entry.count}` +
            `${Number.isFinite(entry.medianAac) ? ` | median AAC ${entry.medianAac.toFixed(4)}` : ""}` +
            `${Number.isFinite(entry.medianIc50) ? ` | median IC50 ${entry.medianIc50.toFixed(4)}` : ""}`
          )
        );
      }
      if (tissueEntries.length > 0) {
        keyFields.push("Most sensitive tissues:");
        keyFields.push(
          ...tissueEntries.slice(0, boundedTissueLimit).map((entry, idx) =>
            `${idx + 1}. ${entry.label} | n=${entry.count}` +
            `${Number.isFinite(entry.medianAac) ? ` | median AAC ${entry.medianAac.toFixed(4)}` : ""}` +
            `${Number.isFinite(entry.medianIc50) ? ` | median IC50 ${entry.medianIc50.toFixed(4)}` : ""}`
          )
        );
      }
      keyFields.push("Most sensitive cell lines:");
      keyFields.push(
        ...uniqueRecords.slice(0, boundedCellLimit).map((record, idx) =>
          `${idx + 1}. ${record.cell_name} [${record.dataset}] | ${record.tissue_name}` +
          `${Number.isFinite(record.aac) ? ` | AAC ${record.aac.toFixed(4)}` : ""}` +
          `${Number.isFinite(record.ic50) ? ` | IC50 ${record.ic50.toFixed(4)}` : ""}`
        )
      );
      if (resolution.alternates.length > 0) {
        keyFields.push(`Other nearby compound names: ${resolution.alternates.join(", ")}`);
      }

      const summary =
        `${compound.name}: PharmacoDB response across ${uniqueRecords.length.toLocaleString()} profiled cell-line records` +
        `${datasetFilter ? ` in ${datasetFilter}` : ""}` +
        `${tissueFilter ? ` matching "${tissueFilter}"` : ""}.` +
        (bestRecord
          ? ` Highest ranked response: ${bestRecord.cell_name} [${bestRecord.dataset}]` +
            `${Number.isFinite(bestRecord.aac) ? ` with AAC ${bestRecord.aac.toFixed(4)}` : ""}` +
            `${Number.isFinite(bestRecord.ic50) ? ` and IC50 ${bestRecord.ic50.toFixed(4)}` : ""}.`
          : "");

      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary,
            keyFields,
            sources: [
              compound.uid ? `https://pharmacodb.ca/compounds/${encodeURIComponent(compound.uid)}` : "https://pharmacodb.ca/",
              PHARMACODB_GRAPHQL_API,
            ],
            limitations: [
              "PharmacoDB harmonizes multiple public pharmacogenomic datasets, but metrics are not perfectly interchangeable across source studies.",
              "AAC is typically interpreted with higher values indicating stronger response, while lower IC50 values indicate higher sensitivity.",
              "Some compounds include PRISM, GDSC, CTRPv2, or other upstream datasets, so cross-dataset comparisons should be treated as directional rather than perfectly matched.",
            ],
          }),
        }],
      };
    } catch (error) {
      return { content: [{ type: "text", text: `Error in get_pharmacodb_compound_response: ${error.message}` }] };
    }
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

/** Resolve a gene symbol (e.g. LRRK2) to the canonical ABA acronym (e.g. Lrrk2) when exact match fails. */
async function resolveAbaGeneAcronym(productId, geneAcronym) {
  const url =
    `${ABA_API}/data/Gene/query.json` +
    `?criteria=products[id$eq${productId}],[acronym$il'*${encodeURIComponent(geneAcronym)}*']` +
    `&num_rows=5`;
  const res = await fetchJsonWithRetry(url, { timeoutMs: 10000 }).catch(() => ({ msg: [] }));
  const genes = res.msg || [];
  for (const g of genes) {
    if (g?.acronym && String(g.acronym).toLowerCase() === String(geneAcronym).toLowerCase()) {
      return g.acronym;
    }
  }
  return genes[0]?.acronym || null;
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
      "Retrieves quantified gene expression data from the Allen Brain Atlas. Given a gene acronym or symbol (e.g. LRRK2, Pdyn, Gad1), returns expression energy, density, and intensity across brain structures. Human symbols are auto-resolved to organism-specific acronyms (e.g. LRRK2 -> Lrrk2 for mouse). Optionally filter to a specific structure or sort by expression metric.",
    inputSchema: {
      geneAcronym: z.string().describe("Gene acronym or symbol (e.g. 'LRRK2', 'Lrrk2', 'Pdyn', 'Gad1'). Human symbols like LRRK2 are auto-resolved to mouse Lrrk2 when organism is mouse."),
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

    let datasetRes = await fetchJsonWithRetry(datasetUrl, { timeoutMs: 15000 });
    let datasets = datasetRes.msg || [];
    let resolvedAcronym = geneAcronym;

    if (datasets.length === 0) {
      const canonical = await resolveAbaGeneAcronym(productId, geneAcronym);
      if (canonical && canonical !== geneAcronym) {
        const retryUrl =
          `${ABA_API}/data/SectionDataSet/query.json` +
          `?criteria=products[id$eq${productId}],genes[acronym$eq'${encodeURIComponent(canonical)}']${planeFilter}` +
          `&include=genes` +
          `&num_rows=1`;
        datasetRes = await fetchJsonWithRetry(retryUrl, { timeoutMs: 15000 });
        datasets = datasetRes.msg || [];
        if (datasets.length > 0) resolvedAcronym = canonical;
      }
    }

    let usedHumanRequestedMouseFallback = false;
    if (datasets.length === 0 && productId === ABA_PRODUCT_IDS.human) {
      const mouseCanonical = await resolveAbaGeneAcronym(ABA_PRODUCT_IDS.mouse, geneAcronym);
      const mouseAcronym = mouseCanonical || geneAcronym;
      const mouseUrl =
        `${ABA_API}/data/SectionDataSet/query.json` +
        `?criteria=products[id$eq${ABA_PRODUCT_IDS.mouse}],genes[acronym$eq'${encodeURIComponent(mouseAcronym)}']${planeFilter}` +
        `&include=genes` +
        `&num_rows=1`;
      const mouseRes = await fetchJsonWithRetry(mouseUrl, { timeoutMs: 15000 });
      const mouseDatasets = mouseRes.msg || [];
      if (mouseDatasets.length > 0) {
        datasetRes = mouseRes;
        datasets = mouseDatasets;
        resolvedAcronym = mouseAcronym;
        usedHumanRequestedMouseFallback = true;
      }
    }

    if (datasets.length === 0) {
      const humanHint =
        organism === "human"
          ? " The human Allen Brain Atlas has no ISH data for this gene. Try organism: 'mouse' for mouse ortholog expression (cross-species reference only)."
          : "";
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No ISH experiments found for gene "${geneAcronym}" in the Allen Brain Atlas (${organism || "mouse"}).`,
            keyFields: [`Gene: ${geneAcronym}`, `Product: ${organism || "mouse"}`],
            sources: [`${ABA_API}/data/SectionDataSet/query.json`],
            limitations: [
              "Ensure the gene acronym is valid and exists in the selected atlas product.",
              "For mouse atlas, use organism-specific acronym (e.g. Lrrk2 for LRRK2). Try search_aba_genes first to resolve the symbol.",
              ...(humanHint ? [humanHint.trim()] : []),
            ],
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
    const geneName = dataset.genes?.[0]?.name || resolvedAcronym;
    const resolvedNote =
      resolvedAcronym !== geneAcronym ? ` (${geneAcronym} resolved to ${resolvedAcronym})` : "";

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
          summary: `Allen Brain Atlas expression for ${resolvedAcronym} (${geneName})${resolvedNote}${usedHumanRequestedMouseFallback ? " [MOUSE — human atlas has no data]" : ""}: top ${records.length} structures by ${sortField} (${plane}, dataset ${datasetId}).`,
          keyFields: [
            `Gene: ${resolvedAcronym} (${geneName})${resolvedNote}${usedHumanRequestedMouseFallback ? " — MOUSE (human requested, human atlas has no data)" : ""}`,
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
            ...(usedHumanRequestedMouseFallback
              ? [
                  "Human Allen Brain Atlas has no ISH data for this gene; data shown is from mouse ortholog. Mouse expression patterns may differ from human — use as cross-species reference only.",
                ]
              : []),
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

async function fetchEbrainsKgSearch({ query, type, from, size }) {
  const url =
    `${EBRAINS_KG_SEARCH_API}/groups/public/search` +
    `?q=${encodeURIComponent(query)}&type=${encodeURIComponent(type)}&from=${from}&size=${size}`;

  const res = await fetchWithRetry(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
    timeoutMs: 20000,
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => "");
    const statusDetail = errorText ? `: ${compactErrorMessage(errorText, 220)}` : "";
    throw new Error(`HTTP ${res.status}${statusDetail}`);
  }

  return res.json();
}

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
    const requestedType = type && EBRAINS_KG_VALID_TYPES.has(type) ? type : null;

    let hits = [];
    let total = 0;
    let types = {};
    const limitations = [
      "Only publicly released (curated) resources are included in search results.",
      "Use get_ebrains_kg_document with the ID and type to retrieve full metadata, DOIs, authors, and file information.",
      "EBRAINS primarily hosts data from the Human Brain Project and European neuroscience initiatives.",
    ];

    if (requestedType) {
      const data = await fetchEbrainsKgSearch({
        query,
        type: requestedType,
        from: pageFrom,
        size: limit,
      });
      hits = data.hits || [];
      total = data.total || 0;
      types = data.types || {};
    } else {
      // The public search endpoint now fails with HTTP 500 when "type" is omitted.
      // Aggregate a best-effort cross-type search client-side instead.
      const searchWindow = Math.min(pageFrom + limit, 50);
      const searchTypes = Array.from(EBRAINS_KG_VALID_TYPES);
      const settled = await Promise.all(
        searchTypes.map(async (typeName) => {
          try {
            const data = await fetchEbrainsKgSearch({
              query,
              type: typeName,
              from: 0,
              size: searchWindow,
            });
            return { typeName, data, error: null };
          } catch (error) {
            return { typeName, data: null, error };
          }
        })
      );

      const successful = settled.filter((entry) => entry.data);
      const failed = settled
        .filter((entry) => entry.error)
        .map((entry) => `${entry.typeName}: ${compactErrorMessage(entry.error?.message || "unknown error", 160)}`);

      if (successful.length === 0) {
        return {
          content: [{
            type: "text",
            text: renderStructuredResponse({
              summary: `EBRAINS Knowledge Graph search failed for "${query}" across all resource types.`,
              keyFields: [
                `Query: ${query}`,
                "Type filter: all (aggregated)",
                failed.length > 0 ? `Failures: ${failed.join("; ")}` : "Failures: unknown",
              ],
              sources: ["https://search.kg.ebrains.eu/"],
              limitations: [
                "The EBRAINS public search API returns HTTP 500 when no type filter is supplied.",
              ],
            }),
          }],
        };
      }

      const combinedHits = [];
      const seenIds = new Set();
      for (const entry of successful) {
        for (const hit of entry.data.hits || []) {
          if (!hit?.id || seenIds.has(hit.id)) continue;
          seenIds.add(hit.id);
          combinedHits.push(hit);
        }
      }

      hits = combinedHits.slice(pageFrom, pageFrom + limit);
      total = successful.reduce((sum, entry) => sum + (entry.data.total || 0), 0);
      types = successful[0].data.types || {};

      limitations.push(
        "The EBRAINS public search API returns HTTP 500 when no type filter is supplied, so this tool aggregates per-type searches when type is omitted."
      );
      if (pageFrom + limit > 50) {
        limitations.push("Cross-type pagination is approximate beyond the first 50 results per resource type.");
      }
      if (failed.length > 0) {
        limitations.push(`Partial type failures: ${failed.join("; ")}`);
      }
    }

    if (hits.length === 0) {
      const typeCounts = Object.entries(types)
        .map(([t, v]) => `${t}: ${v.count}`)
        .join(", ");
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No EBRAINS Knowledge Graph results for "${query}"${requestedType ? ` (type: ${requestedType})` : ""}.${typeCounts ? ` Available counts across types: ${typeCounts}` : ""}`,
            keyFields: [`Query: ${query}`, requestedType ? `Type filter: ${requestedType}` : "Type: all (aggregated)"],
            sources: ["https://search.kg.ebrains.eu/"],
            limitations,
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
          summary: `EBRAINS Knowledge Graph: ${total} result(s) for "${query}"${requestedType ? ` (type: ${requestedType})` : ""}. Showing ${pageFrom + 1}–${pageFrom + hits.length}.`,
          keyFields: [
            `Query: ${query}`,
            requestedType ? `Type filter: ${requestedType}` : "Type: all (aggregated)",
            `Total: ${total}`,
            `Type breakdown: ${typeSummary}`,
            `\nResults:`,
            ...lines,
          ],
          sources: [
            `https://search.kg.ebrains.eu/?q=${encodeURIComponent(query)}${requestedType ? `&category=${encodeURIComponent(requestedType)}` : ""}`,
            "https://search.kg.ebrains.eu/",
          ],
          limitations,
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
      "Searches public CONP (Canadian Open Neuroscience Platform) dataset repositories using GitHub repository metadata.",
    inputSchema: {
      query: z.string().describe("Keyword for repository name, description, or topics."),
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
      "Fetches detailed metadata for a specific CONP dataset repository, including README preview, license, stars, topics, and links.",
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

const NEMAR_MODALITY_ALIASES = new Map([
  ["eeg", "EEG"],
  ["meg", "MEG"],
  ["ieeg", "iEEG"],
  ["emg", "EMG"],
]);

function parseNEMARQuery(rawQuery) {
  const raw = normalizeWhitespace(rawQuery || "");
  if (!raw) {
    return { rawQuery: "", keyword: "", modalities: [], modalityParam: "" };
  }
  const keywordTokens = [];
  const modalities = [];
  for (const token of raw.split(/\s+/)) {
    const normalizedToken = token.toLowerCase().replace(/[^a-z0-9]+/g, "");
    if (!normalizedToken || ["or", "and", "not"].includes(normalizedToken)) {
      continue;
    }
    const modality = NEMAR_MODALITY_ALIASES.get(normalizedToken);
    if (modality) {
      if (!modalities.includes(modality)) modalities.push(modality);
      continue;
    }
    keywordTokens.push(token);
  }
  const keyword = normalizeWhitespace(keywordTokens.join(" "));
  return {
    rawQuery: raw,
    keyword,
    modalities,
    modalityParam: modalities.length > 0 ? `OR ${modalities.join(" ")}` : "",
  };
}

function parseNEMARAgeRange(item) {
  const ageMin = Number(item?.age_min || 0);
  const ageMax = Number(item?.age_max || 0);
  if (ageMin > 0 && ageMax > 0) return `${ageMin}-${ageMax}`;
  if (ageMin > 0) return `${ageMin}+`;
  if (ageMax > 0) return `0-${ageMax}`;
  return "unknown";
}

function sortNEMARDatasets(items, mode) {
  const datasets = [...items];
  const newestValue = (item) => String(item?.latestSnapshot_created || item?.publishDate || item?.created || "");
  if (mode === "name") {
    datasets.sort((a, b) =>
      normalizeWhitespace(a?.description_name || a?.name || "").localeCompare(
        normalizeWhitespace(b?.description_name || b?.name || "")
      )
    );
    return datasets;
  }
  datasets.sort((a, b) => newestValue(b).localeCompare(newestValue(a)));
  return datasets;
}

function extractNEMARDetailField(html, label) {
  const escapedLabel = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = html.match(
    new RegExp(`<strong[^>]*>${escapedLabel}:?\\s*<\\/strong>\\s*<span[^>]*>([\\s\\S]*?)<\\/span>`, "i")
  );
  return match ? normalizeWhitespace(stripHtmlToText(match[1])) : "";
}

function parseNEMARDetailPage(html, datasetId) {
  const titleMatch = html.match(/<h2[^>]*>\s*([\s\S]*?)\s*<\/h2>/i);
  const title = titleMatch ? normalizeWhitespace(stripHtmlToText(titleMatch[1])) : "";
  const readmeMatch = html.match(/<div class="nemar-readme">([\s\S]*?)<\/div>\s*<\/div>/i);
  const readmeText = readmeMatch ? stripHtmlToText(readmeMatch[1], { preserveParagraphs: true }) : "";
  const openNeuroUrlMatch = html.match(/<a href="([^"]+)"[^>]*>\s*OpenNeuro\s*<\/a>/i);

  if (!title) return null;

  return {
    datasetId,
    title,
    participants: extractNEMARDetailField(html, "Participants"),
    eventFiles: extractNEMARDetailField(html, "Event files"),
    hedAnnotation: extractNEMARDetailField(html, "HED annotation"),
    bidsVersion: extractNEMARDetailField(html, "BIDS Version"),
    publishedDate: extractNEMARDetailField(html, "Published date"),
    tasks: extractNEMARDetailField(html, "Tasks"),
    modalities: extractNEMARDetailField(html, "Available modalities"),
    formats: extractNEMARDetailField(html, "Format(s)"),
    readmePreview: readmeText ? (readmeText.length > 1200 ? `${readmeText.slice(0, 1197)}...` : readmeText) : "",
    sources: [
      `${NEMAR_DATAEXPLORER_API}/detail?dataset_id=${encodeURIComponent(datasetId)}`,
      openNeuroUrlMatch?.[1] || "",
      "https://nemar.org/dataexplorer",
    ].filter(Boolean),
  };
}

server.registerTool(
  "search_nemar_datasets",
  {
    description:
      "Searches NEMAR (NeuroElectroMagnetic data Archive) for public EEG, MEG, iEEG, and related datasets using the public Data Explorer.",
    inputSchema: {
      query: z.string().optional().describe("Search term for dataset title or metadata. Omit to browse."),
      sortBy: z.enum(["updated", "stars", "name"]).optional().describe("Result ordering. Default 'updated'."),
      maxResults: z.number().optional().describe("Maximum results (default 20, max 50)."),
    },
  },
  async ({ query, sortBy, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const mode = String(sortBy || "updated").toLowerCase();
    const { rawQuery, keyword, modalities, modalityParam } = parseNEMARQuery(query || "");
    const params = new URLSearchParams({ file_format: "all" });
    if (keyword) params.set("search", keyword);
    if (modalityParam) params.set("modality", modalityParam);

    let data;
    try {
      data = await fetchJsonWithRetry(`${NEMAR_DATAEXPLORER_VIEW_API}?${params.toString()}`, {
        retries: 1,
        timeoutMs: 15000,
        headers: { Accept: "application/json", "User-Agent": "research-mcp" },
      });
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `NEMAR dataset search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [
              rawQuery ? `Query: ${rawQuery}` : "Browse: all",
              modalityParam ? `Modality filter: ${modalities.join(", ")}` : "Modality: all",
            ],
            sources: ["https://nemar.org/discover", "https://nemar.org/dataexplorer"],
            limitations: ["Results depend on the availability of NEMAR's public Data Explorer API."],
          }),
        }],
      };
    }

    const rawItems = Array.isArray(data) ? data : [];
    const total = rawItems.length;
    const items = sortNEMARDatasets(rawItems, mode === "stars" ? "updated" : mode).slice(0, limit);
    const limitations = [
      "NEMAR results are sourced from the public Data Explorer, which may rank matches using hidden metadata not visible in the summary card.",
      "Use get_nemar_dataset_details with a dataset id (for example ds005522) to inspect README-level task and anatomy details.",
    ];
    if (mode === "stars") {
      limitations.push("NEMAR's public explorer does not expose GitHub stars; results were ordered by recency instead.");
    }

    if (items.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No NEMAR datasets found${rawQuery ? ` for "${rawQuery}"` : ""}.`,
            keyFields: [
              rawQuery ? `Query: ${rawQuery}` : "Browse: all",
              modalityParam ? `Modality filter: ${modalities.join(", ")}` : "Modality: all",
            ],
            sources: ["https://nemar.org/discover", "https://nemar.org/dataexplorer"],
            limitations: ["Try modality-filtered queries like 'iEEG hippocampus' or 'MEG resting state'."],
          }),
        }],
      };
    }

    const lines = items.map((dataset, idx) => {
      const datasetId = dataset?.id || "unknown";
      const title = normalizeWhitespace(dataset?.description_name || dataset?.name || "Untitled dataset");
      const titlePreview = title.length > 100 ? `${title.slice(0, 97)}...` : title;
      const modalityText = normalizeWhitespace(dataset?.modalities || dataset?.primaryModality || "unknown");
      const participants = Number.isFinite(Number(dataset?.participants)) ? Number(dataset.participants) : "unknown";
      const ageRange = parseNEMARAgeRange(dataset);
      const formats = normalizeWhitespace(dataset?.file_formats || "unknown");
      const publishedAt = String(dataset?.publishDate || dataset?.latestSnapshot_created || dataset?.created || "unknown").slice(0, 10);
      return `  ${String(idx + 1).padStart(3)}. ${datasetId} — ${titlePreview} | modality: ${modalityText} | participants: ${participants} | ages: ${ageRange} | formats: ${formats} | published: ${publishedAt}`;
    });

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `NEMAR: ${items.length} dataset(s)${rawQuery ? ` matching "${rawQuery}"` : ""} from the public Data Explorer.`,
          keyFields: [
            rawQuery ? `Query: ${rawQuery}` : "Browse: all",
            keyword ? `Parsed text query: ${keyword}` : "Parsed text query: none",
            modalityParam ? `Modality filter: ${modalities.join(", ")}` : "Modality: all",
            `Total matches: ${total}`,
            "\nResults:",
            ...lines,
          ],
          sources: [
            "https://nemar.org/discover",
            "https://nemar.org/dataexplorer",
          ],
          limitations,
        }),
      }],
    };
  }
);

server.registerTool(
  "get_nemar_dataset_details",
  {
    description:
      "Fetches detailed metadata for a NEMAR dataset by dataset id or legacy repository id (for example ds005522 or nm000104). " +
      "Returns README-derived task details, modalities, formats, and links when available.",
    inputSchema: {
      repo: z.string().describe("NEMAR dataset id or legacy repository name from search_nemar_datasets (for example 'ds005522', 'nm000104', or 'nemarDatasets/nm000104')."),
    },
  },
  async ({ repo }) => {
    const rawRepo = normalizeWhitespace(repo || "");
    if (!rawRepo) {
      return { content: [{ type: "text", text: "Provide a NEMAR dataset id or repo name (for example ds005522 or nm000104)." }] };
    }
    const repoName = rawRepo.includes("/") ? rawRepo.split("/").pop() : rawRepo;
    if (!repoName) {
      return { content: [{ type: "text", text: "Unable to parse repository name." }] };
    }

    if (/^(?:ds|nm|on)\d+$/i.test(repoName)) {
      const detailUrl = `${NEMAR_DATAEXPLORER_API}/detail?dataset_id=${encodeURIComponent(repoName)}`;
      try {
        const detailResponse = await fetchWithRetry(detailUrl, {
          retries: 1,
          timeoutMs: 15000,
          headers: {
            Accept: "text/html,application/xhtml+xml",
            "User-Agent": "research-mcp",
          },
        });
        const html = await detailResponse.text();
        const parsed = parseNEMARDetailPage(html, repoName);
        if (parsed) {
          const keyFields = [
            `Dataset ID: ${parsed.datasetId}`,
            `Title: ${parsed.title}`,
          ];
          if (parsed.participants) keyFields.push(`Participants: ${parsed.participants}`);
          if (parsed.eventFiles) keyFields.push(`Event files: ${parsed.eventFiles}`);
          if (parsed.hedAnnotation) keyFields.push(`HED annotation: ${parsed.hedAnnotation}`);
          if (parsed.modalities) keyFields.push(`Modalities: ${parsed.modalities}`);
          if (parsed.formats) keyFields.push(`Formats: ${parsed.formats}`);
          if (parsed.tasks) keyFields.push(`Tasks: ${parsed.tasks}`);
          if (parsed.bidsVersion) keyFields.push(`BIDS version: ${parsed.bidsVersion}`);
          if (parsed.publishedDate) keyFields.push(`Published: ${parsed.publishedDate}`);
          if (parsed.readmePreview) keyFields.push(`\nREADME preview:\n${parsed.readmePreview}`);

          return {
            content: [{
              type: "text",
              text: renderStructuredResponse({
                summary: `NEMAR dataset: ${parsed.title} (${parsed.datasetId}).`,
                keyFields,
                sources: parsed.sources,
                limitations: [
                  "Metadata is parsed from NEMAR's public dataset detail page and README rendering.",
                  "Some anatomy or electrode-localization details may only be fully available in sidecar files or accompanying publications.",
                ],
              }),
            }],
          };
        }
      } catch (_) {
        // Fall through to GitHub repo lookup for legacy mirrored datasets.
      }
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
            summary: `NEMAR dataset not found in NEMAR detail pages or GitHub mirror: ${repoName}.`,
            keyFields: [`Identifier: ${repoName}`, `Error: ${compactErrorMessage(error?.message || "unknown error", 220)}`],
            sources: [`https://nemar.org/dataexplorer/detail?dataset_id=${encodeURIComponent(repoName)}`, `https://github.com/${NEMAR_GITHUB_ORG}/${repoName}`],
            limitations: ["The identifier may be incorrect, private, or not mirrored on GitHub."],
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
    } catch (_) {
      // README preview is best-effort only.
    }

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
          summary: `NEMAR GitHub mirror dataset: ${repoData?.full_name || `${NEMAR_GITHUB_ORG}/${repoName}`}.`,
          keyFields,
          sources: [
            repoData?.html_url || `https://github.com/${NEMAR_GITHUB_ORG}/${repoName}`,
            "https://nemar.org/discover",
            "https://nemar.org/",
          ],
          limitations: [
            "This is a legacy GitHub-mirror fallback. Prefer NEMAR dataset ids like ds005522 for live Data Explorer metadata.",
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
      "Searches Brain-CODE (Ontario Brain Institute) datasets mirrored through the CONP archive.",
    inputSchema: {
      query: z.string().optional().describe("Keyword to narrow results. Omit to list all mirrored Brain-CODE datasets."),
      maxResults: z.number().optional().describe("Maximum results (default 20, max 50)."),
    },
  },
  async ({ query, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const rawQuery = normalizeWhitespace(query || "");
    const searchTerms = splitArchiveSearchTerms(rawQuery);
    const queriesToRun = rawQuery ? searchTerms : [""];
    const items = [];
    const seen = new Set();
    const totals = [];

    try {
      for (const term of queriesToRun) {
        const q = term ? `${BRAINCODE_CONP_QUERY} ${term} org:${CONP_GITHUB_ORG}` : `${BRAINCODE_CONP_QUERY} org:${CONP_GITHUB_ORG}`;
        const params = new URLSearchParams({ q, per_page: String(limit), page: "1", sort: "updated", order: "desc" });
        const url = `${GITHUB_API}/search/repositories?${params.toString()}`;
        const data = await fetchJsonWithRetry(url, {
          retries: 1,
          timeoutMs: 12000,
          headers: { Accept: "application/vnd.github+json", "User-Agent": "research-mcp" },
        });
        totals.push(Number(data?.total_count || 0));
        for (const item of Array.isArray(data?.items) ? data.items : []) {
          const name = normalizeWhitespace(item?.name || "");
          if (!name || seen.has(name)) continue;
          seen.add(name);
          items.push(item);
          if (items.length >= limit) break;
        }
        if (items.length >= limit) break;
      }
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `Brain-CODE search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: buildArchiveSearchKeyFields(rawQuery, searchTerms, { browseLabel: "Browse: all Brain-CODE" }),
            sources: ["https://www.braincode.ca/", "https://github.com/conpdatasets"],
            limitations: ["GitHub API rate limits may apply."],
          }),
        }],
      };
    }

    const total = totals.length <= 1 ? (totals[0] || 0) : null;
    if (items.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No Brain-CODE datasets found${rawQuery ? ` for "${rawQuery}"` : ""}.`,
            keyFields: buildArchiveSearchKeyFields(rawQuery, searchTerms, { browseLabel: "Browse: all Brain-CODE" }),
            sources: ["https://www.braincode.ca/content/public-data-releases", "https://github.com/conpdatasets"],
          limitations: [
            "Brain-CODE datasets in CONP use braincode_* naming. Full catalog at braincode.ca.",
          ],
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
          summary: `Brain-CODE: ${items.length} dataset(s)${rawQuery ? ` matching "${rawQuery}"` : ""} in CONP.${total !== null ? ` Total: ${total}.` : ` Aggregated across ${searchTerms.length} parsed search term(s).`}`,
          keyFields: [
            ...buildArchiveSearchKeyFields(rawQuery, searchTerms, { browseLabel: "Browse: all Brain-CODE" }),
            ...(total !== null ? [`Total: ${total}`] : [`Showing unique results: ${items.length}`]),
            "\nResults:",
            ...lines,
          ],
          sources: [
            "https://www.braincode.ca/content/public-data-releases",
            "https://www.braincode.ca/",
            "https://github.com/conpdatasets",
          ],
          limitations: [
            "This tool indexes Brain-CODE datasets mirrored in CONP. For controlled releases and full catalog, see braincode.ca.",
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

const NEUROBAGEL_NIDM_TERMS = new Set([
  "T1Weighted", "T2Weighted", "DiffusionWeighted", "FlowWeighted",
  "ArterialSpinLabeling", "Electroencephalography", "Magnetoencephalography", "PositronEmissionTomography",
]);

/** Convert full URIs or plain terms to API-required prefix:term format (e.g. nidm:T1Weighted). */
function neurobagelNormalizeControlledTerm(value) {
  const raw = normalizeWhitespace(value || "");
  if (!raw) return "";
  if (/^[a-zA-Z]+:\S+$/.test(raw)) return raw;
  if (raw.startsWith("http://snomed.info/id/") || raw.startsWith("https://snomed.info/id/")) {
    const id = raw.replace(/^https?:\/\/snomed\.info\/id\//i, "").split(/[#?/]/)[0] || "";
    return id ? `snomed:${id}` : "";
  }
  if (raw.includes("nidash/nidm") || raw.includes("nidm#")) {
    const term = raw.replace(/^https?:\/\/[^#]*#?/i, "").split("/").pop() || "";
    return term ? `nidm:${term}` : "";
  }
  if (NEUROBAGEL_NIDM_TERMS.has(raw)) return `nidm:${raw}`;
  return "";
}

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
      "Queries public Neurobagel cohorts using structured demographic and imaging filters.",
    inputSchema: {
      minAge: z.number().optional().describe("Minimum participant age in years."),
      maxAge: z.number().optional().describe("Maximum participant age in years."),
      sex: z.string().optional().describe("SNOMED sex: snomed:248152002 (female) or snomed:248153007 (male)."),
      diagnosis: z.string().optional().describe("SNOMED diagnosis term."),
      minImagingSessions: z.number().optional().describe("Minimum number of imaging sessions per participant."),
      minPhenotypicSessions: z.number().optional().describe("Minimum number of phenotypic sessions per participant."),
      assessment: z.string().optional().describe("Assessment/tool term in prefix:term format."),
      imageModal: z.string().optional().describe("Imaging modality in nidm:term format, e.g. nidm:T1Weighted, nidm:Electroencephalography, nidm:Magnetoencephalography."),
      pipelineName: z.string().optional().describe("Pipeline name in prefix:term format from Neurobagel pipeline catalog."),
      pipelineVersion: z.string().optional().describe("Pipeline version string, e.g. 1.0.0."),
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
    const sexNorm = neurobagelNormalizeControlledTerm(sex) || normalizeWhitespace(sex);
    if (sexNorm) params.set("sex", sexNorm);
    const diagNorm = neurobagelNormalizeControlledTerm(diagnosis) || normalizeWhitespace(diagnosis);
    if (diagNorm) params.set("diagnosis", diagNorm);
    if (Number.isFinite(minImagingSessions)) params.set("min_num_imaging_sessions", String(minImagingSessions));
    if (Number.isFinite(minPhenotypicSessions)) params.set("min_num_phenotypic_sessions", String(minPhenotypicSessions));
    const assessNorm = neurobagelNormalizeControlledTerm(assessment) || normalizeWhitespace(assessment);
    if (assessNorm) params.set("assessment", assessNorm);
    const modalNorm = neurobagelNormalizeControlledTerm(imageModal) || normalizeWhitespace(imageModal);
    if (modalNorm) params.set("image_modal", modalNorm);
    const pipeNorm = neurobagelNormalizeControlledTerm(pipelineName) || normalizeWhitespace(pipelineName);
    if (pipeNorm) params.set("pipeline_name", pipeNorm);
    if (normalizeWhitespace(pipelineVersion)) params.set("pipeline_version", normalizeWhitespace(pipelineVersion));

    const limit = Math.min(Math.max(1, maxResults || 25), 100);
    const url = `${NEUROBAGEL_API}/query${params.toString() ? `?${params.toString()}` : ""}`;

    let rows;
    try {
      rows = await fetchJsonWithRetry(url, {
        retries: 2,
        timeoutMs: 35000,
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
      "Searches public OpenNeuro neuroimaging datasets by keyword and/or imaging modality. Returns dataset IDs, names, modalities, and latest snapshot tags.",
    inputSchema: {
      query: z.string().optional().describe("Keyword or phrase to match against public dataset metadata."),
      modality: z.string().optional().describe("Imaging modality: MRI, MEG, EEG, PET, iEEG, or behavioral. Omit to browse across all datasets."),
      after: z.string().optional().describe("Pagination cursor returned by a previous search_openneuro_datasets call. Use when continuing beyond the first page."),
      maxResults: z.number().optional().describe("Maximum results to return (default 20, max 50)."),
    },
  },
  async ({ query, modality, after, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const rawKeyword = normalizeWhitespace(query || "");
    const searchTerms = splitArchiveSearchTerms(rawKeyword);
    const afterCursor = normalizeWhitespace(after || "");
    const modArg = modality && OPENNEURO_MODALITIES.has(String(modality).trim().toUpperCase())
      ? String(modality).trim().toUpperCase()
      : null;

    const fetchOpenNeuroPage = async (pageLimit, cursor) => {
      const args = [`first: ${pageLimit}`];
      if (modArg) args.push(`modality: "${modArg}"`);
      if (cursor) args.push(`after: ${JSON.stringify(cursor)}`);
      const gql = `{ datasets(${args.join(", ")}) { edges { node { id name metadata { modalities dxStatus studyDomain datasetName } latestSnapshot { tag description { Name } } } } pageInfo { hasNextPage endCursor } } }`;

      const res = await fetchWithRetry(OPENNEURO_GRAPHQL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: gql }),
        timeoutMs: 15000,
        retries: 1,
      });
      return await res.json();
    };

    let data;
    try {
      if (rawKeyword) {
        const normalizeForMatch = (value) => String(value || "")
          .toLowerCase()
          .replace(/[^a-z0-9]+/g, " ")
          .replace(/\s+/g, " ")
          .trim();

        const needles = searchTerms.map((term) => normalizeForMatch(term)).filter(Boolean);
        let cursor = afterCursor || null;
        let scannedPages = 0;
        let hasNextPage = false;
        let endCursor = cursor;
        const matchedEdges = [];

        while (matchedEdges.length < limit && scannedPages < 40) {
          const page = await fetchOpenNeuroPage(50, cursor);
          const errs = page?.errors;
          if (errs && errs.length > 0) {
            data = page;
            break;
          }
          const pageEdges = page?.data?.datasets?.edges || [];
          const pageInfo = page?.data?.datasets?.pageInfo || {};
          for (const edge of pageEdges) {
            const node = edge?.node || {};
            const haystack = normalizeForMatch([
              node.id,
              node.name,
              node.metadata?.datasetName,
              node.metadata?.dxStatus,
              node.metadata?.studyDomain,
              node.latestSnapshot?.description?.Name,
            ].filter(Boolean).join(" "));
            if (needles.some((needle) => haystack.includes(needle))) {
              matchedEdges.push(edge);
            }
          }
          hasNextPage = Boolean(pageInfo?.hasNextPage);
          endCursor = pageInfo?.endCursor || null;
          if (matchedEdges.length >= limit) break;
          if (!hasNextPage) break;
          cursor = endCursor;
          scannedPages += 1;
        }

        data = {
          data: {
            datasets: {
              edges: matchedEdges.slice(0, limit),
              pageInfo: {
                hasNextPage,
                endCursor,
              },
            },
          },
        };
      } else {
        data = await fetchOpenNeuroPage(limit, afterCursor || null);
      }
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `OpenNeuro search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: [
              ...buildArchiveSearchKeyFields(rawKeyword, searchTerms, { browseLabel: "Keyword: none", queryLabel: "Keyword" }),
              modArg ? `Modality: ${modArg}` : "Modality: all",
            ],
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
            keyFields: [
              ...buildArchiveSearchKeyFields(rawKeyword, searchTerms, { browseLabel: "Keyword: none", queryLabel: "Keyword" }),
              modArg ? `Modality: ${modArg}` : "Modality: all",
            ],
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
            summary: `No OpenNeuro datasets found${rawKeyword ? ` for keyword '${rawKeyword}'` : ""}${modArg ? ` with modality ${modArg}` : ""}.`,
            keyFields: [
              ...buildArchiveSearchKeyFields(rawKeyword, searchTerms, { browseLabel: "Keyword: none", queryLabel: "Keyword" }),
              modArg ? `Modality: ${modArg}` : "Modality: all",
            ],
            sources: ["https://openneuro.org/datasets"],
            limitations: [
              "Valid modalities: MRI, MEG, EEG, PET, iEEG, behavioral.",
            ],
          }),
        }],
      };
    }

    const lines = edges.map((e, i) => {
      const node = e?.node || {};
      const id = node.id || "?";
      const name = node.metadata?.datasetName || node.latestSnapshot?.description?.Name || node.name || "Unnamed";
      const mods = Array.isArray(node.metadata?.modalities) ? node.metadata.modalities.join(", ") : "unknown";
      const tag = node.latestSnapshot?.tag || "—";
      const dx = node.metadata?.dxStatus ? ` dx: ${node.metadata.dxStatus}` : "";
      const domain = node.metadata?.studyDomain ? ` domain: ${node.metadata.studyDomain}` : "";
      return `  ${String(i + 1).padStart(3)}. ${id} — ${name} (${mods})${dx}${domain} snapshot: ${tag}`;
    });

    const hasMore = data?.data?.datasets?.pageInfo?.hasNextPage;
    const nextCursor = data?.data?.datasets?.pageInfo?.endCursor || null;

    return {
      content: [{
        type: "text",
        text: renderStructuredResponse({
          summary: `OpenNeuro: ${edges.length} dataset(s)${rawKeyword ? ` matching '${rawKeyword}'` : ""}${modArg ? ` (modality: ${modArg})` : ""}.${hasMore ? " More available via pagination." : ""}`,
          keyFields: [
            ...buildArchiveSearchKeyFields(rawKeyword, searchTerms, { browseLabel: "Keyword: none", queryLabel: "Keyword" }),
            modArg ? `Modality filter: ${modArg}` : "Modality: all",
            `Showing: ${edges.length}`,
            nextCursor ? `Next cursor: ${nextCursor}` : "Next cursor: none",
            "\nDatasets:",
            ...lines,
          ],
          sources: [
            "https://openneuro.org/datasets",
            `https://openneuro.org/datasets${modArg ? `?modality=${modArg.toLowerCase()}` : ""}`,
          ],
          limitations: [
            "Keyword matching is against public dataset metadata fields such as dataset name, dxStatus, and studyDomain; OpenNeuro does not expose a dedicated disorder filter in this tool.",
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
      "Returns dataset name, modalities, DOI, diagnosis/study fields when present, latest snapshot tag, tasks, and approximate subject counts from the public summary.",
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

    const query = `{ dataset(id: "${normalizedId}") { id name metadata { modalities dxStatus studyDomain studyDesign trialCount datasetName } latestSnapshot { tag description { Name DatasetDOI } summary { subjects tasks modalities } } } }`;

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
    const summary = ds.latestSnapshot?.summary || {};
    const name = desc.Name || ds.name || "Unnamed";
    const doi = desc.DatasetDOI || "";
    const tag = ds.latestSnapshot?.tag || "—";
    const subjectCount = Array.isArray(summary.subjects) ? summary.subjects.length : null;
    const tasks = Array.isArray(summary.tasks) && summary.tasks.length > 0 ? summary.tasks.join(", ") : null;

    const keyFields = [
      `Dataset: ${name}`,
      `ID: ${ds.id}`,
      `Modalities: ${mods}`,
      `Latest snapshot: ${tag}`,
    ];
    if (doi) keyFields.push(`DOI: ${doi}`);
    if (ds.metadata?.dxStatus) keyFields.push(`Diagnosis/status: ${ds.metadata.dxStatus}`);
    if (ds.metadata?.studyDomain) keyFields.push(`Study domain: ${ds.metadata.studyDomain}`);
    if (ds.metadata?.studyDesign) keyFields.push(`Study design: ${ds.metadata.studyDesign}`);
    if (subjectCount !== null) keyFields.push(`Approx. subjects in public summary: ${subjectCount}`);
    if (tasks) keyFields.push(`Tasks: ${tasks}`);
    if (Number.isFinite(ds.metadata?.trialCount)) keyFields.push(`Trial count: ${ds.metadata.trialCount}`);

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
            "Approximate subject counts are derived from the public summary subject list and may not reflect cohort splits or complete phenotype tables.",
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
      "Searches the DANDI Archive for neurophysiology datasets. Returns dandiset identifiers, names, asset counts, sizes, and embargo status.",
    inputSchema: {
      query: z.string().optional().describe("Search term. Omit to list recent dandisets."),
      maxResults: z.number().optional().describe("Maximum results (default 20, max 50)."),
    },
  },
  async ({ query, maxResults }) => {
    const limit = Math.min(Math.max(1, maxResults || 20), 50);
    const rawQuery = normalizeWhitespace(query || "");
    const searchTerms = splitArchiveSearchTerms(rawQuery);
    const queriesToRun = rawQuery ? searchTerms : [""];
    const results = [];
    const seen = new Set();
    const totals = [];

    try {
      for (const term of queriesToRun) {
        const params = new URLSearchParams({ page_size: String(limit), page: "1" });
        if (term) params.set("search", term);
        const url = `${DANDI_API}/dandisets/?${params.toString()}`;
        const data = await fetchJsonWithRetry(url, {
          retries: 1,
          timeoutMs: 15000,
          headers: { Accept: "application/json" },
        });
        totals.push(Number(data?.count ?? 0));
        for (const item of data?.results || []) {
          const ident = normalizeWhitespace(item?.identifier || "");
          if (!ident || seen.has(ident)) continue;
          seen.add(ident);
          results.push(item);
          if (results.length >= limit) break;
        }
        if (results.length >= limit) break;
      }
    } catch (error) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `DANDI search failed: ${compactErrorMessage(error?.message || "unknown error", 220)}.`,
            keyFields: buildArchiveSearchKeyFields(rawQuery, searchTerms, { browseLabel: "Query: (none)" }),
            sources: ["https://dandiarchive.org/", "https://api.dandiarchive.org/"],
            limitations: ["DANDI API may be temporarily unavailable."],
          }),
        }],
      };
    }

    const total = totals.length <= 1 ? (totals[0] || 0) : null;

    if (results.length === 0) {
      return {
        content: [{
          type: "text",
          text: renderStructuredResponse({
            summary: `No DANDI dandisets found${rawQuery ? ` for "${rawQuery}"` : ""}.`,
            keyFields: buildArchiveSearchKeyFields(rawQuery, searchTerms, { browseLabel: "Query: (none)" }),
            sources: ["https://dandiarchive.org/dandisets/"],
            limitations: [
              "Try broader search terms or omit the query to browse all dandisets.",
            ],
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
          summary: `DANDI: ${results.length} dandiset(s)${rawQuery ? ` matching "${rawQuery}"` : ""}.${total !== null ? ` Total in archive: ${total}.` : ` Aggregated across ${searchTerms.length} parsed search term(s).`}`,
          keyFields: [
            ...buildArchiveSearchKeyFields(rawQuery, searchTerms, { browseLabel: "Browse: recent dandisets", queryLabel: "Search" }),
            ...(total !== null ? [`Total: ${total}`] : [`Showing unique results: ${results.length}`]),
            "\nResults:",
            ...lines,
          ],
          sources: [
            rawQuery ? `https://dandiarchive.org/dandisets/?search=${encodeURIComponent(rawQuery)}` : "https://dandiarchive.org/dandisets/",
            "https://dandiarchive.org/",
          ],
          limitations: [
            "Use a dandiset identifier to inspect a specific dataset in detail.",
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
      "Returns name, version, asset count, size, embargo status, and contact.",
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
