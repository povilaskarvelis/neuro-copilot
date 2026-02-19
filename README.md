# AI Co-Scientist

An agentic AI co-investigator for scientific discovery and target evaluation.

## What It Does

The AI Co-Scientist helps researchers:
- **Discover drug targets** for diseases using Open Targets data
- **Evaluate druggability** of potential targets
- **Find clinical trial evidence** from ClinicalTrials.gov
- **Search scientific literature** via PubMed
- **Generate dynamic, query-specific plans** from model reasoning + available tools
- **Run stateful workflows** with resumable task IDs and HITL checkpoints
- **Synthesize final reports** directly from final-step model output

## Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                AI Co-Scientist Orchestrator                   │
│           (Google ADK + Gemini + dynamic planner)             │
└──────────────────────────┬────────────────────────────────────┘
                           │ Step execution + HITL checkpoint
┌──────────────────────────▼────────────────────────────────────┐
│                Task Workflow State (JSON store)               │
│      • task_id • step status • checkpoint history • resume    │
└──────────────────────────┬────────────────────────────────────┘
                           │ MCP Protocol
┌──────────────────────────▼────────────────────────────────────┐
│                    Research MCP Server                         │
│   Domain tools for targets, trials, literature, genomics, etc. │
└───────────────────────────────────────────────────────────────┘
```

## Dynamic Workflow (Detailed)

The flow below highlights where LLM reasoning is required versus deterministic runtime guardrails.

```mermaid
flowchart TD
    A[User submits a research question] --> B{{Model checks if the question is ambiguous or malformed (clarifier)}}
    B --> C1{Is clarification needed before running tools?}
    C1 -->|Yes| C2[Pause and ask the user a short clarification question (HITL clarification gate)]
    C2 --> C3[Attach the user's clarification to the objective (objective merge)]
    C1 -->|No| C
    C3 --> C
    C{{Hybrid router sets request type, intent tags, and normalized objective (heuristics + LLM)}}
    C --> D{{Planner drafts a query-specific subgoal graph from objective + available tools (dynamic planner)}}
    D --> E[Convert subgoals into executable workflow steps and save active plan metadata (WorkflowTask + PlanVersion)]
    E --> E1[Run the first planning/scope step (step_1 decomposition)]
    E1 --> F[Open the initial execution checkpoint before tool-heavy steps (pre_evidence_execution)]
    F --> G{User action at checkpoint}
    G -->|Start/Continue| H[Run the next planned step and persist task state]
    G -->|Revise once| I{{Parse user feedback into structured revision intent, then replan remaining steps (replan)}}
    I --> F
    G -->|Stop (CLI)| Z[Mark task as blocked/paused until resumed]

    H --> J{{Model executes the current step prompt and chooses from allowed tools (executor)}}
    J --> K[Run MCP tools, capture tool traces, and validate tool response contracts]
    K --> L[Recompute quality gates (coverage, citations, unresolved gaps, confidence)]
    L --> M{Open another checkpoint now? (phase boundary or pre-evidence gate)}
    M -->|Yes| F
    M -->|No| N{Are there remaining steps?}
    N -->|Yes| H
    N -->|No| O{{Model writes the final recommendation/report step (synthesis)}}
    O --> P[Persist final report markdown and make PDF export available (report artifacts)]
    P --> Q[Render the same report content in UI and CLI]

    classDef llm fill:#eef2f7,stroke:#5f6b7a,stroke-width:1.5px,color:#1f2933;
    classDef guard fill:#f3f4f6,stroke:#6b7280,stroke-width:1.5px,color:#1f2933;
    class B,C,D,I,J,O llm;
    class C1,C2,C3,E,E1,F,H,K,L,M,N,P,Q,Z guard;
```

Legend:
- **LLM nodes** = model-required reasoning/synthesis steps.
- **Guardrail nodes** = deterministic runtime orchestration, persistence, and HITL control flow.

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

Optional web UI on top of the same task store/CLI workflow:

```bash
cd adk-agent
python -m pip install -r requirements.txt
python ui_server.py
```

Then open `http://127.0.0.1:8080` in your browser.

Then ask questions like:
- *"Find promising drug targets for Parkinson's disease"*
- *"Evaluate LRRK2 as a drug target"*
- *"What clinical trials exist for Alzheimer's treatments?"*

In co-investigator mode, each request now:
1. builds a dynamic plan for the specific objective,
2. executes evidence steps with adaptive checkpointing,
3. supports one revision opportunity after initial plan creation,
4. runs straight through to synthesis/reporting (no forced checkpoint right before synthesis),
5. saves report artifacts to `adk-agent/reports/<task_id>.md` and `adk-agent/reports/<task_id>.pdf`.

Web UI thread model:
- The sidebar groups runs by conversation thread (not by single task only).
- Each follow-up creates a new iteration task in the same conversation.
- Follow-ups can branch from any selected prior report card.
- Every iteration has a separate research log in main chat.
- Suggested next actions are shown in main chat only (the report panel omits the `Next Actions` section).

PDF rendering behavior:
- Uses a high-fidelity HTML/CSS pipeline via headless Chrome/Chromium when available.
- Falls back to internal ReportLab rendering when Chrome/Chromium is unavailable.

Final report behavior:
- UI and terminal both use the same canonical report source.
- Final report is the model's final-step report markdown (rendered in UI, plain in terminal).

Terminal commands:
- `status`: show current or latest workflow status
- `resume [task_id]`: resume a saved task
- `history [task_id]`: show recent saved revisions for rollback
- `rollback <offset|revision_id> [task_id]`: restore a task to an earlier revision
- `help`: show command help

## Project Structure

```
├── adk-agent/              # AI Co-Scientist Agent (Python)
│   ├── agent.py            # Runtime orchestration + execution loop
│   ├── workflow.py         # Task model, planning helpers, report rendering
│   ├── ui_server.py        # FastAPI server for web UI
│   ├── ui/                 # Frontend app
│   ├── task_state_store.py # JSON persistence for workflow state
│   ├── co_scientist/       # Runtime/planning/domain modules
│   ├── reports/            # Runtime-generated report artifacts (safe to clear)
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

Core regression tests (recommended default):

```bash
cd adk-agent
pytest test_agentic_phase_flow.py test_ui_server_hitl.py test_agentic_workflow_eval.py test_agentic_invariants.py test_task_state_store.py test_report_pdf.py -q
```

This suite covers:
- agentic phase transitions and checkpoint behavior
- UI runtime/HITL flow and report availability
- dynamic planner/workflow invariants
- task state persistence and report PDF generation

Notes:
- External network/eval harness tests were removed from the default suite to keep CI/dev runs deterministic and fast.
- Generated artifacts in `adk-agent/reports/` are runtime outputs and can be safely deleted.

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
