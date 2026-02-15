# AI Co-Scientist

An AI-powered co-investigator for preclinical drug target discovery.

## What It Does

The AI Co-Scientist helps researchers:
- **Discover drug targets** for diseases using Open Targets data
- **Evaluate druggability** of potential targets
- **Find clinical trial evidence** from ClinicalTrials.gov
- **Search scientific literature** via PubMed
- **Run explicit multi-step workflows** (plan -> execute -> checkpoint -> synthesize)
- **Track workflow state** in terminal sessions with resumable task IDs
- **Synthesize findings** with citations and actionable recommendations

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                AI Co-Scientist Orchestrator                   │
│                (Google ADK + Gemini + Planner)                │
└──────────────────────────┬────────────────────────────────────┘
                           │ Step execution + HITL checkpoint
┌──────────────────────────▼────────────────────────────────────┐
│                Task Workflow State (JSON store)               │
│      • task_id • step status • checkpoint history • resume    │
└──────────────────────────┬────────────────────────────────────┘
                           │ MCP Protocol
┌──────────────────────────▼────────────────────────────────────┐
│                    Research MCP Server                         │
│  33 Tools: disease/target, trials, literature, omics, pathways │
└───────────────────────────────────────────────────────────────┘
```

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

# 2. Install MCP server dependencies
cd research-mcp
npm install

# 3. Install agent dependencies
cd ../adk-agent
pip install -r requirements.txt

# 4. Configure API key
echo 'GOOGLE_API_KEY="your-key-here"' > .env
```

### Run

```bash
cd adk-agent
python agent.py
```

Then ask questions like:
- *"Find promising drug targets for Parkinson's disease"*
- *"Evaluate LRRK2 as a drug target"*
- *"What clinical trials exist for Alzheimer's treatments?"*

In co-investigator mode, each request now:
1. prints a 2-3 step plan,
2. executes step 1,
3. pauses at a human-in-the-loop checkpoint (`continue`, `revise <scope>`, `stop`),
4. then produces a structured final report.
5. saves report artifacts to `adk-agent/reports/<task_id>.md` and `adk-agent/reports/<task_id>.pdf`.

PDF rendering behavior:
- Uses a high-fidelity HTML/CSS pipeline via headless Chrome/Chromium when available.
- Falls back to internal ReportLab rendering when Chrome/Chromium is unavailable.

General response contract used for broad request coverage:
- `Request Understanding`
- `Plan`
- `Execution Log`
- `Checkpoint History`
- `Findings`
- `Evidence`
- `Limitations & Risks`
- `Next Actions`

Terminal commands:
- `status`: show current or latest workflow status
- `resume [task_id]`: resume a saved task
- `history [task_id]`: show recent saved revisions for rollback
- `rollback <offset|revision_id> [task_id]`: restore a task to an earlier revision
- `help`: show command help

## Project Structure

```
├── adk-agent/              # AI Co-Scientist Agent (Python)
│   ├── agent.py            # Main agent with Gemini
│   ├── workflow.py         # Planner/task model/report rendering
│   ├── task_state_store.py # Lightweight JSON persistence for workflows
│   ├── co_scientist/       # Agent module for evaluation
│   ├── evals/              # ADK evaluation test cases
│   ├── test_accuracy.py    # Data validation tests
│   └── README.md           # Agent documentation
│
├── research-mcp/           # Research Tools Server (Node.js)
│   ├── server.js           # MCP server with 30 tools
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
| **Chemistry Evidence** | `search_chembl_compounds_for_target` | ChEMBL |
| **Expression & Cell Context** | `summarize_target_expression_context` | Open Targets GraphQL |
| **Genetic Direction-of-Effect** | `infer_genetic_effect_direction` | GWAS Catalog |
| **Competitive & Safety Intelligence** | `summarize_target_competitive_landscape`, `summarize_target_safety_liabilities` | Open Targets GraphQL |
| **Comparative Prioritization** | `compare_targets_multi_axis` (auto mode from goal text, preset, or custom axis weights) | Open Targets GraphQL |
| **Literature** | `search_pubmed`, `get_pubmed_abstract`, `search_pubmed_advanced`, `get_pubmed_paper_details`, `get_pubmed_author_profile` | PubMed E-utilities |
| **Researcher Discovery** | `search_openalex_works`, `search_openalex_authors`, `rank_researchers_by_activity`, `get_researcher_contact_candidates` | OpenAlex |
| **Variants & Genomics** | `search_clinvar_variants`, `get_clinvar_variant_details`, `search_gwas_associations`, `get_gene_info` | NCBI ClinVar, GWAS Catalog, NCBI Gene |
| **Pathway & Networks** | `search_reactome_pathways`, `get_string_interactions` | Reactome, STRING |
| **Ontology Context** | `expand_disease_context` | OLS (EFO/MONDO) |
| **Local Data** | `list_local_datasets`, `read_local_dataset` | Local filesystem |

## Testing

```bash
cd adk-agent

# Test data accuracy (API responses)
python test_accuracy.py

# Test MCP connection
python test_mcp.py

# Run agent evaluations
pytest test_agent_eval.py -v

# Run co-investigator workflow evaluation
pytest test_agent_eval.py::test_co_investigator_workflow -v

# Run generalized co-investigator contract evaluation
pytest test_agent_eval.py::test_co_investigator_general_contract -v

# Run researcher discovery workflow evaluation
pytest test_agent_eval.py::test_researcher_discovery -v

# Run variant + pathway workflow evaluation
pytest test_agent_eval.py::test_variant_pathway -v

# Run toolbox expansion evaluation (trial landscape + chemistry + ontology)
pytest test_agent_eval.py::test_toolbox_expansion -v

# Run Sprint 2 evaluation (expression/cell context + direction-of-effect)
pytest test_agent_eval.py::test_sprint2_context_direction -v

# Run Sprint 3 evaluation (competition + safety liabilities)
pytest test_agent_eval.py::test_sprint3_competition_safety -v

# Run Sprint 4 evaluation (multi-axis target ranking)
pytest test_agent_eval.py::test_sprint4_target_ranking -v

# Run Sprint 5 evaluation (custom ranking weights)
pytest test_agent_eval.py::test_sprint5_custom_weights -v

# Run Sprint 6 evaluation (auto mode selection + minimal clarification)
pytest test_agent_eval.py::test_sprint6_auto_mode_clarification -v
```

One-command acceptance demo (fixed challenge scenarios + deterministic scoreboard):

```bash
python adk-agent/run_acceptance_demo.py
```

Artifacts are written to:
- `adk-agent/acceptance/results/<run_id>/scoreboard.json`
- `adk-agent/acceptance/results/<run_id>/summary.md`
- `adk-agent/acceptance/results/<run_id>/summary.pdf`
- `adk-agent/acceptance/results/<run_id>/reports/*.md`
- `adk-agent/acceptance/results/<run_id>/reports/*.pdf`

Useful options:
- `--limit 1`: run only the first scenario for a quick smoke
- `--only-scenario <scenario_id>`: run specific scenario(s)
- `--no-pdf`: skip PDF artifact generation and emit markdown-only reports
- `--non-strict`: always exit 0 even when checks fail

CI automation:
- Workflow file: `.github/workflows/acceptance-demo.yml`
- Trigger: pushes to `main`/`master` and all pull requests
- Behavior: runs strict acceptance harness (no `--non-strict`) and fails build on any scenario/checkpoint failure
- Artifacts uploaded on every run (pass or fail): `scoreboard.json`, `summary.md`, `summary.pdf`, scenario markdown/PDF reports, HITL probe log
- Required repository secret: `GOOGLE_API_KEY`

Weighted rubric for general strategy quality:
- `adk-agent/evals/co_investigator_rubric.md`

## Data Sources

- **[Open Targets Platform](https://platform.opentargets.org/)** - Disease-target associations, druggability data
- **[ClinicalTrials.gov](https://clinicaltrials.gov/)** - Clinical trial registry
- **[PubMed/NCBI](https://pubmed.ncbi.nlm.nih.gov/)** - Scientific literature, gene information
- **[GWAS Catalog](https://www.ebi.ac.uk/gwas/)** - Variant-trait associations and effect direction signals
- **[ChEMBL](https://www.ebi.ac.uk/chembl/)** - Compound bioactivity and target chemical evidence
- **[OLS / EFO / MONDO](https://www.ebi.ac.uk/ols4/)** - Ontology expansion, synonyms, and hierarchy context
