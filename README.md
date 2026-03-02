# AI Co-Scientist

An agentic AI research assistant that synthesizes evidence across biomedical databases to guide pre-clinical decisions.

## What It Does

The AI Co-Scientist helps biomedical researchers evaluate therapeutic targets and research directions before human trials by:

- **Synthesizing evidence across 34 databases** — genomics, literature, clinical trials, neuroscience, protein structure, pathways, safety, and more
- **Generating query-specific execution plans** with explicit tool and data-source proposals
- **Requiring human plan approval or revision** before evidence tools run
- **Running iterative evidence-gathering loops** via a ReAct (Reason → Act → Observe) cycle
- **Producing structured final reports** with inline citations, limitations, and suggested next steps
- **Exporting reports as PDF** with full source attribution

### Databases

| Category | Sources |
|----------|---------|
| **Genomics & Variants** | Open Targets Platform, gnomAD, 1000 Genomes, Ensembl VEP, MyVariant.info |
| **Clinical Trials** | ClinicalTrials.gov |
| **Literature & Researchers** | PubMed, OpenAlex |
| **Protein Structure & Function** | AlphaFold, RCSB PDB, UniProt |
| **Pathways & Interactions** | Reactome, STRING |
| **Chemistry & Bioactivity** | ChEMBL, PubChem, SureChEMBL |
| **Safety & Regulatory** | FDA FAERS, RxNorm |
| **Immunology** | IEDB |
| **Perturbation Signatures** | LINCS L1000 |
| **Clinical Variant Interpretation** | CIViC, ClinVar |
| **Cancer Genomics** | cBioPortal |
| **Target Discovery & Druggability** | GWAS Catalog, DGIdb, GTEx |
| **Neuroscience Atlases & Knowledge Graphs** | Allen Brain Atlas, EBRAINS Knowledge Graph, CONP, Neurobagel, OpenNeuro, DANDI, NEMAR, Brain-CODE |

This stack combines live REST APIs (literature, trials, protein/pathway, neuroscience) with BigQuery public datasets (genomics, chemistry, safety, perturbation, and patent-derived chemistry).

## Architecture

```mermaid
flowchart TB
    U["Browser UI"] --> S["ui_server.py (FastAPI)"]
    S --> R["ADK Runner + Session State"]
    R --> W["SequentialAgent co_scientist_workflow<br/>planner → react_loop → report_synthesizer"]
    W --> M["MCPToolset → research-mcp/server.js"]
    M --> W
    W --> S
    S --> U
```

The custom web UI (`ui/index.html`, `app.js`, `styles.css`) communicates with `ui_server.py`, which manages conversations, run orchestration, and PDF export. Under the hood it uses the Google ADK `Runner` with `InMemorySessionService`.

## Dynamic Workflow

Source of truth: `adk-agent/co_scientist/workflow.py`.

```mermaid
flowchart TD
    U["User question"] --> P["planner (LlmAgent)<br/>builds plan JSON"]
    P --> H["User reviews plan"]
    H -->|approve| L
    H -->|revise| P

    subgraph L["react_loop (LoopAgent, max 25 iterations)"]
      direction TB
      S["Select next step"]
      R["Reason"]
      A["Act (call MCP tool)"]
      O["Observe"]
      C{"Step complete?"}
      M{"More steps?"}
      S --> R --> A --> O --> C
      C -->|no| R
      C -->|yes| M
      M -->|yes| S
    end

    M -->|all done| X{"Auto-synthesize?"}
    M -->|blocked/partial| W["User choice:<br/>continue or finalize"]
    W -->|continue| S
    W -->|finalize| Y
    X -->|yes| Y
    X -->|no| W

    Y["report_synthesizer (LlmAgent)<br/>final cited Markdown report"] --> Z["Report shown to user"]
    Z -->|follow-up question| P

    U -. command .-> HC["history / rollback / switch N"]
    HC --> HR["Archive current cycle<br/>restore selected cycle"]
    HR --> Z

    classDef llm fill:#eef2f7,stroke:#5f6b7a,stroke-width:1.5px,color:#1f2933;
    classDef guard fill:#f3f4f6,stroke:#6b7280,stroke-width:1.5px,color:#1f2933;
    classDef cmd fill:#fef9c3,stroke:#ca8a04,stroke-width:1.5px,color:#1f2933;
    classDef react fill:#e0f2fe,stroke:#0284c7,stroke-width:1.5px,color:#1f2933;
    class P,Y llm;
    class H,X,W,C,M guard;
    class HC,HR cmd;
    class S,R,A,O react;
```

### ReAct Execution Loop

Each plan step is executed as a **Reason → Act → Observe** cycle inside a `LoopAgent`:

1. **Reason** — the step executor reads the current step goal and decides what tool to call and why
2. **Act** — calls an MCP tool (e.g., `search_pubmed`, `run_bigquery_select_query`)
3. **Observe** — reviews the tool results; if insufficient, reasons again and retries with a different query or tool
4. **Conclude** — when the step's completion condition is met, returns a structured result with a `reasoning_trace`

The reasoning trace captures the full decision chain per step and is stored alongside step results. The synthesizer uses these traces to ground source citations in the final report.

**Error recovery:** if the executor returns invalid output, the loop retries the step (up to 3 attempts) with a corrective prompt before marking it blocked and advancing.

## Available Tools

### MCP Tools (Live APIs)

| Category | Tools | Source |
|----------|-------|--------|
| **Clinical Trials** | `search_clinical_trials`, `get_clinical_trial`, `summarize_clinical_trials_landscape` | ClinicalTrials.gov |
| **Literature** | `search_pubmed`, `search_pubmed_advanced`, `get_pubmed_abstract` | PubMed (NCBI E-utilities) |
| **Researcher Discovery** | `search_openalex_works`, `search_openalex_authors`, `rank_researchers_by_activity`, `get_researcher_contact_candidates` | OpenAlex |
| **Protein Annotations** | `search_uniprot_proteins`, `get_uniprot_protein_profile` | UniProt REST |
| **Pathways & Networks** | `search_reactome_pathways`, `get_string_interactions` | Reactome, STRING |
| **Variant Predictions** | `annotate_variants_vep` | Ensembl VEP (SIFT, PolyPhen, AlphaMissense) |
| **Variant Annotations** | `get_variant_annotations` | MyVariant.info (ClinVar, CADD, dbSNP, gnomAD, COSMIC) |
| **Clinical Variants** | `search_civic_variants`, `search_civic_genes` | CIViC (cancer variant interpretations) |
| **Protein Structures** | `get_alphafold_structure` | AlphaFold API (pLDDT confidence, PDB/CIF files) |
| **GWAS Associations** | `search_gwas_associations` | EBI GWAS Catalog (trait-variant associations, p-values, odds ratios) |
| **Drug-Gene Interactions** | `search_drug_gene_interactions` | DGIdb (druggability categories, approved/experimental drugs) |
| **Tissue Expression** | `get_gene_tissue_expression` | GTEx v8 (median TPM across 54 human tissues) |
| **Experimental Structures** | `search_protein_structures` | RCSB PDB (X-ray, cryo-EM structures, resolution, ligands) |
| **Cancer Mutations** | `get_cancer_mutation_profile` | cBioPortal (TCGA Pan-Cancer mutation frequencies, hotspots) |
| **Bioactivity** | `get_chembl_bioactivities` | ChEMBL API (IC50/Ki/Kd, target selectivity, assay metadata) |
| **Chemical Compounds** | `get_pubchem_compound` | PubChem (116M+ compounds, molecular properties, SMILES, drug-likeness) |
| **Safety Signals** | `search_fda_adverse_events` | openFDA FAERS (post-marketing adverse event reports) |
| **Brain Atlases** | `search_aba_genes`, `search_aba_structures`, `get_aba_gene_expression`, `search_aba_differential_expression` | Allen Brain Atlas (structure ontology, ISH expression, differential enrichment) |
| **Neuroscience Knowledge Graph** | `search_ebrains_kg`, `get_ebrains_kg_document` | EBRAINS KG (datasets, models, software, contributors, projects) |
| **Neuroscience Datasets (CONP)** | `search_conp_datasets`, `get_conp_dataset_details` | CONP datasets via `conpdatasets` GitHub catalog |
| **Cohort Discovery (Neurobagel)** | `query_neurobagel_cohorts` | Neurobagel public node API (harmonized cohort-level dataset discovery) |
| **Neuroimaging Datasets (OpenNeuro)** | `search_openneuro_datasets`, `get_openneuro_dataset` | OpenNeuro GraphQL (BIDS fMRI/MRI/MEG/EEG datasets by modality) |
| **Neurophysiology (DANDI)** | `search_dandi_datasets`, `get_dandi_dataset` | DANDI Archive REST API (electrophysiology, calcium imaging, NWB/BIDS) |
| **Neuroelectromagnetic (NEMAR)** | `search_nemar_datasets`, `get_nemar_dataset_details` | NEMAR (EEG/MEG/iEEG from OpenNeuro, nemarDatasets GitHub, BIDS, HED) |
| **Brain-CODE (OBI)** | `search_braincode_datasets`, `get_braincode_dataset_details` | Brain-CODE via CONP (Ontario Brain Institute: epilepsy, depression, neurodegeneration, CP, concussion) |
| **Benchmarks** | `benchmark_dataset_overview`, `check_gpqa_access` | Hugging Face Datasets |

### BigQuery Datasets

All accessed via `list_bigquery_tables` and `run_bigquery_select_query` with read-only row/bytes guardrails.

| Dataset | Contents |
|---------|----------|
| **open_targets_platform** | Disease-target associations, genetic evidence, drugs, tractability |
| **ebi_chembl** | Bioactive compounds, target bioactivity (IC50/Ki/EC50), mechanism of action |
| **gnomad** | Population variant frequencies across diverse ancestries |
| **human_genome_variants** | 1000 Genomes Phase 3 variants, Platinum Genomes, Simons Diversity |
| **human_variant_annotation** | ClinVar clinical significance classifications, variant-condition associations (hg19/hg38) |
| **immune_epitope_db** | Immune epitopes, B-cell assays, MHC ligand binding, T-cell receptor data |
| **nlm_rxnorm** | Drug nomenclature, ingredient relationships, clinical drug pathways |
| **fda_drug** | FAERS adverse event reports, drug labels, NDC listings, enforcement actions |
| **umiami_lincs** | L1000 perturbation signatures: cell lines, small molecules, readouts |
| **ebi_surechembl** | Chemical structures extracted from patents |

## User Commands

| Command | When | What it does |
|---------|------|-------------|
| `approve` / `yes` / `lgtm` / `go ahead` | Plan pending approval | Approve the plan and start execution |
| *(any other text while plan is pending)* | Plan pending approval | Treat as revision feedback — planner regenerates |
| `continue` / `next` / `go` | Execution paused | Resume executing remaining plan steps |
| `finalize` / `summarize now` | Any time after execution | Skip remaining steps and generate final report |
| `history` | Any time | List all archived + active research cycles |
| `rollback` | Any time | Archive current cycle and restore the most recent prior cycle |
| `switch N` | Any time | Archive current cycle and restore cycle number N |
| *(new question)* | After a report | Archives current cycle, starts fresh planning |

## Guardrails

- **HITL plan gate** — `before_agent_callback` blocks the ReAct loop and synthesizer until the plan is approved.
- **ReAct retry** — parse/validation errors trigger automatic retry (up to 3 attempts per step) before marking the step blocked.
- **Error callbacks** — `on_model_error_callback` and `on_tool_error_callback` surface rate-limit and tool failures to the user instead of silently crashing.
- **Step renumbering** — follow-up plans with non-sequential IDs are canonically renumbered to `S1, S2, ...`.
- **Source citations** — final reports cite human-readable database names (PubMed, ClinicalTrials.gov, etc.), never raw tool names or JSON URLs.
- **Research history** — up to 10 prior research cycles are archived with full state; rollback restores any previous cycle.
- **BigQuery guardrails** — read-only queries with configurable max rows (default 200, hard cap 1000) and bytes-billed limits.

## Example Queries

```
Evaluate LRRK2 as a drug target for Parkinson disease — what is the genetic evidence, druggability, and competitive landscape?
```
```
Is KRAS G12C structurally druggable? What do predicted protein structures and known interaction partners suggest about tractable binding sites?
```
```
What are the population-level variant frequencies for BRCA1, and which variants are classified as clinically significant?
```
```
What post-marketing safety signals exist for JAK inhibitors, and how selective are they across the kinase family?
```
```
What immune epitopes are known for PD-L1, and which signaling pathways does it participate in?
```
```
Who are the most active researchers working on CAR-T therapy for solid tumors, and what are the recent breakthroughs?
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Optional local auth: Google API key ([get one free](https://aistudio.google.com/apikey))
- Optional Vertex auth: `gcloud` CLI + Application Default Credentials

### Setup

```bash
# 1. Clone and install
git clone <repo-url>
cd <repo>

# 2. Create and activate a project virtualenv
python -m venv .venv
source .venv/bin/activate

# 3. Install MCP server dependencies
cd research-mcp
npm install

# 4. Install agent dependencies
cd ../adk-agent
pip install -r requirements.txt

# 5a. Local mode auth (AI Studio API key)
cp .env.local.example .env
# then edit .env and set GOOGLE_API_KEY
# optional for BigQuery tools (ADC):
# gcloud auth application-default login
# optional for gated Hugging Face datasets (e.g., GPQA):
# set HF_TOKEN in .env

# 5b. Vertex mode auth (project-backed)
cp .env.vertex.example .env
# then edit GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION
# and authenticate with:
# gcloud auth application-default login
```

Keep the same shell with `.venv` activated for all commands below.

### Run

**Web UI (primary):**

```bash
cd adk-agent
python ui_server.py
```

Opens the custom web interface at `http://localhost:8080` with conversation management, real-time activity tracking, report panel, and PDF export.

**ADK CLI / ADK Web UI (alternative):**

```bash
cd adk-agent
adk run co_scientist    # interactive terminal
adk web .               # ADK built-in web UI
```

**Standalone CLI wrapper:**

```bash
cd adk-agent
python agent.py
python agent.py --query "Evaluate LRRK2 as a drug target in Parkinson disease"
```

## Cloud Run Deployment

```bash
PROJECT_ID="your-project-id" \
REGION="us-central1" \
SERVICE_NAME="ai-co-scientist" \
bash scripts/deploy_cloud_run.sh
```

The deploy script builds a container image via Cloud Build, then deploys to Cloud Run with Vertex AI auth and the full BigQuery dataset allowlist pre-configured.

### Runtime endpoints (Cloud Run)
- `GET /healthz` — readiness and config status
- `POST /api/query` — submit a research question
- `GET /api/conversations` — list conversations
- `GET /api/conversations/{id}` — conversation detail with iterations
- `GET /api/tasks/{id}/report.pdf` — export report as PDF

## Project Structure

```
├── adk-agent/              # AI Co-Scientist Agent (Python)
│   ├── agent.py            # ADK-native CLI wrapper (interactive/single query)
│   ├── ui_server.py        # Custom web UI server (FastAPI, primary entrypoint)
│   ├── report_pdf.py       # PDF report generation
│   ├── server.py           # Minimal FastAPI HTTP wrapper (legacy)
│   ├── co_scientist/
│   │   ├── __init__.py     # Exports root_agent for `adk run` / `adk web`
│   │   └── workflow.py     # Workflow graph, HITL, history/rollback, callbacks
│   ├── ui/
│   │   ├── index.html      # Landing page and chat interface
│   │   ├── app.js          # Client-side application logic
│   │   └── styles.css      # UI styles
│   ├── .adk/               # ADK local sessions/artifacts (created at runtime)
│   └── test_*.py           # Regression tests
│
├── research-mcp/           # Research Tools Server (Node.js)
│   ├── server.js           # MCP tool server (Allen + EBRAINS + biomedical tools)
│   ├── data/               # Local datasets
│   └── test-tools.js       # Optional manual MCP tool test script
│
├── scripts/
│   └── deploy_cloud_run.sh # Build + deploy to Cloud Run with Vertex env
│
├── Dockerfile              # Cloud Run image (Python + Node runtime)
├── .dockerignore           # Build context guardrails
└── README.md               # This file
```

## Data Sources

### Live APIs
- **[ClinicalTrials.gov](https://clinicaltrials.gov/)** — Clinical trial registry and results
- **[PubMed / NCBI](https://pubmed.ncbi.nlm.nih.gov/)** — Biomedical literature and abstracts
- **[OpenAlex](https://openalex.org/)** — Scholarly works, authors, and citation data
- **[UniProt](https://www.uniprot.org/)** — Protein sequence, function, and annotation
- **[Reactome](https://reactome.org/)** — Curated biological pathway database
- **[STRING](https://string-db.org/)** — Protein-protein interaction networks
- **[MyVariant.info](https://myvariant.info/)** — Aggregated variant annotations across ClinVar/CADD/dbSNP/COSMIC
- **[GWAS Catalog](https://www.ebi.ac.uk/gwas/)** — Trait-variant associations with p-values and mapped genes
- **[DGIdb](https://dgidb.org/)** — Drug-gene interactions and druggability categories
- **[GTEx](https://gtexportal.org/)** — Tissue-level expression profiles across human tissues
- **[RCSB PDB](https://www.rcsb.org/)** — Experimentally resolved protein structures
- **[AlphaFold](https://alphafold.ebi.ac.uk/)** — Predicted protein structures and confidence scores
- **[cBioPortal](https://www.cbioportal.org/)** — Cancer mutation frequencies and hotspot profiles
- **[PubChem](https://pubchem.ncbi.nlm.nih.gov/)** — Compound structures and physicochemical metadata
- **[openFDA FAERS](https://open.fda.gov/apis/drug/event/)** — Post-marketing safety signal data
- **[Allen Brain Atlas](https://mouse.brain-map.org/)** — Brain structure ontology and ISH gene expression atlases
- **[EBRAINS Knowledge Graph](https://search.kg.ebrains.eu/)** — Neuroscience datasets, models, software, workflows, and contributors

### BigQuery Public Datasets
- **[Open Targets Platform](https://platform.opentargets.org/)** — Disease-target associations, genetic evidence, tractability
- **[ChEMBL](https://www.ebi.ac.uk/chembl/)** — Bioactive compound and target bioactivity data
- **[gnomAD](https://gnomad.broadinstitute.org/)** — Population variant frequencies
- **[1000 Genomes](https://www.internationalgenome.org/)** — Phase 3 variants, population structure
- **[ClinVar (BigQuery)](https://www.ncbi.nlm.nih.gov/clinvar/)** — Clinical significance labels and variant-condition mappings
- **[IEDB](https://www.iedb.org/)** — Immune epitope data, B-cell and T-cell assays
- **[RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/)** — Drug nomenclature and relationships
- **[FDA FAERS](https://open.fda.gov/data/faers/)** — Adverse event reports, drug labels, enforcement
- **[LINCS L1000](https://lincsproject.org/)** — Chemical and genetic perturbation signatures
- **[SureChEMBL](https://www.surechembl.org/)** — Chemical structures from patent literature

## Testing

```bash
cd adk-agent
../.venv/bin/python -m py_compile agent.py server.py ui_server.py report_pdf.py co_scientist/workflow.py
```

Notes:
- External network tests were removed from the default suite to keep CI/dev runs deterministic and fast.
- Generated artifacts in `adk-agent/reports/` are runtime outputs and can be safely deleted.
