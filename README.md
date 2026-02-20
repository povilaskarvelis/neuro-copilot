# AI Co-Scientist

An agentic AI co-investigator for scientific discovery and target evaluation.

## What It Does

The AI Co-Scientist helps researchers:
- **Discover drug targets** for diseases using Open Targets data
- **Evaluate druggability** of potential targets
- **Find clinical trial evidence** from ClinicalTrials.gov
- **Search scientific literature** via PubMed and Europe PMC preprints
- **Assess post-marketing safety and regulatory labels** via openFDA
- **Generate a query-specific execution plan** with explicit tool proposals
- **Require human plan approval or revision** before evidence tools run
- **Run iterative evidence refinement loops** with tool-use guardrails
- **Synthesize structured final reports** with citations, limitations, and next actions

## Architecture

```mermaid
flowchart TB
    U["User (ADK Web or CLI)"] --> R["ADK Runner + Session State"]
    R --> W["SequentialAgent workflow<br/>clarifier -> plan_approval_loop -> evidence_refinement_loop -> report_synthesizer"]
    W --> C["ADK tool confirmation<br/>request_plan_confirmation"]
    C --> U
    W --> M["MCPToolset -> research-mcp/server.js"]
```

## Dynamic Workflow (Detailed)

Source of truth: `adk-agent/co_scientist/workflow.py`.

```mermaid
flowchart TD
    U["User question"] --> C["clarifier (LlmAgent)<br/>normalize objective + constraints"]
    C --> P["planner (LlmAgent)<br/>emit PlanDraft JSON (steps + planned_tools)"]
    P --> R["plan_reviewer (LlmAgent)<br/>forced tool call: request_plan_confirmation"]
    R --> H{"Human decision"}
    H -->|revise| P
    H -->|approve| E["evidence_refinement_loop (LoopAgent)"]
    R -->|pending confirmation| B["Execution blocked until plan approval"]
    B --> H

    E --> X["evidence_executor (LlmAgent)<br/>run approved MCP tools only"]
    X --> Y{"evidence_critic (LlmAgent)<br/>sufficient evidence?"}
    Y -->|no| X
    Y -->|yes| S["report_synthesizer (LlmAgent)<br/>final structured answer"]

    classDef llm fill:#eef2f7,stroke:#5f6b7a,stroke-width:1.5px,color:#1f2933;
    classDef guard fill:#f3f4f6,stroke:#6b7280,stroke-width:1.5px,color:#1f2933;
    class C,P,R,X,Y,S llm;
    class H,B,E guard;
```

Guardrails currently enforced in code:
- `_prepare_plan_gate_for_new_query` resets approval state for each new query.
- `_block_until_plan_approved` blocks evidence and synthesis until approval.
- `_guard_evidence_tool_execution` blocks tools not present in `approved_tools`.

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Google API key ([get one free](https://aistudio.google.com/apikey))

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

# 5. Configure API key
echo 'GOOGLE_API_KEY="your-key-here"' > .env
```

Keep the same shell with `.venv` activated for all commands below.

### Run

Primary (ADK-native CLI):

```bash
cd adk-agent
adk run co_scientist
```

Primary (ADK-native Web UI):

```bash
cd adk-agent
adk web .
```

Optional lightweight wrapper (still ADK-native under the hood):

```bash
cd adk-agent
python agent.py
python agent.py --query "Evaluate LRRK2 as a drug target in Parkinson disease"
```

## Project Structure

```
├── adk-agent/              # AI Co-Scientist Agent (Python)
│   ├── agent.py            # ADK-native CLI wrapper (interactive/single query)
│   ├── co_scientist/
│   │   ├── __init__.py     # Exports root_agent used by `adk run co_scientist`
│   │   └── workflow.py    # Canonical ADK workflow graph + HITL gate
│   ├── workflow.py         # Legacy/compat helpers still present in repo
│   ├── task_state_store.py # State helpers used by compatibility paths
│   ├── .adk/               # ADK local sessions/artifacts (created at runtime)
│   └── test_*.py           # Lean core regression suite
│
├── research-mcp/           # Research Tools Server (Node.js)
│   ├── server.js           # MCP tool server
│   ├── data/               # Local datasets
│   └── README.md           # Tools documentation
│
└── README.md               # This file
```

## Available Tools

| Category | Tools | API |
|----------|-------|-----|
| **Disease & Targets** | `search_diseases`, `search_disease_targets`, `get_target_info`, `search_targets` | Open Targets GraphQL |
| **Druggability** | `check_druggability`, `get_target_drugs` | Open Targets GraphQL |
| **Clinical Trials** | `search_clinical_trials`, `get_clinical_trial`, `summarize_clinical_trials_landscape` | ClinicalTrials.gov |
| **Post-Marketing Safety & Labels** | `summarize_openfda_adverse_events`, `get_openfda_drug_label_summary` | openFDA (FAERS, Drug Label) |
| **Chemistry Evidence** | `search_chembl_compounds_for_target` | ChEMBL |
| **Expression & Cell Context** | `summarize_target_expression_context` | Open Targets GraphQL |
| **Genetic Direction-of-Effect** | `infer_genetic_effect_direction` | GWAS Catalog |
| **Competitive & Safety Intelligence** | `summarize_target_competitive_landscape`, `summarize_target_safety_liabilities` | Open Targets GraphQL |
| **Comparative Prioritization** | `compare_targets_multi_axis` (auto mode from goal text, preset, or custom axis weights) | Open Targets GraphQL |
| **Protein Annotations** | `search_uniprot_proteins`, `get_uniprot_protein_profile` | UniProt REST |
| **Literature** | `search_pubmed`, `search_europe_pmc_preprints`, `get_pubmed_abstract`, `search_pubmed_advanced`, `get_pubmed_paper_details`, `get_pubmed_author_profile` | PubMed E-utilities, Europe PMC |
| **Researcher Discovery** | `search_openalex_works`, `search_openalex_authors`, `rank_researchers_by_activity`, `get_researcher_contact_candidates` | OpenAlex |
| **Variants & Genomics** | `search_clinvar_variants`, `get_clinvar_variant_details`, `search_gwas_associations`, `get_gene_info` | NCBI ClinVar, GWAS Catalog, NCBI Gene |
| **Pathway & Networks** | `search_reactome_pathways`, `get_string_interactions` | Reactome, STRING |
| **Ontology Context** | `expand_disease_context` | OLS (EFO/MONDO) |
| **Local Data** | `list_local_datasets`, `read_local_dataset` | Local filesystem |

## Testing

Core regression tests (recommended default):

```bash
cd adk-agent
../.venv/bin/pytest test_adk_native_workflow.py test_task_state_store.py test_report_pdf.py -q
```

This suite covers:
- ADK-native workflow graph invariants
- task state persistence and report PDF generation

Notes:
- External network tests were removed from the default suite to keep CI/dev runs deterministic and fast.
- Generated artifacts in `adk-agent/reports/` are runtime outputs and can be safely deleted.

## Data Sources

- **[Open Targets Platform](https://platform.opentargets.org/)** - Disease-target associations, druggability data
- **[ClinicalTrials.gov](https://clinicaltrials.gov/)** - Clinical trial registry
- **[PubMed/NCBI](https://pubmed.ncbi.nlm.nih.gov/)** - Scientific literature, gene information
- **[Europe PMC](https://europepmc.org/)** - Biomedical literature and preprints (including medRxiv/bioRxiv)
- **[openFDA](https://open.fda.gov/apis/)** - FAERS adverse-event reports and FDA drug labeling data
- **[GWAS Catalog](https://www.ebi.ac.uk/gwas/)** - Variant-trait associations and effect direction signals
- **[ChEMBL](https://www.ebi.ac.uk/chembl/)** - Compound bioactivity and target chemical evidence
- **[OLS / EFO / MONDO](https://www.ebi.ac.uk/ols4/)** - Ontology expansion, synonyms, and hierarchy context
