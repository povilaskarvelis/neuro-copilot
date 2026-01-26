# AI Co-Scientist

An AI-powered research assistant for preclinical drug target discovery.

## What It Does

The AI Co-Scientist helps researchers:
- **Discover drug targets** for diseases using Open Targets data
- **Evaluate druggability** of potential targets
- **Find clinical trial evidence** from ClinicalTrials.gov
- **Search scientific literature** via PubMed
- **Synthesize findings** with citations and actionable recommendations

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 AI Co-Scientist Agent                        │
│                 (Google ADK + Gemini)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ MCP Protocol
┌─────────────────────────▼───────────────────────────────────┐
│                   Research MCP Server                        │
│                                                              │
│  13 Tools:                                                   │
│  • Disease/Target Discovery  • Clinical Trials              │
│  • Druggability Assessment   • Literature Search            │
│  • Gene Information          • Local Dataset Access         │
└─────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    External APIs                             │
│  • Open Targets  • ClinicalTrials.gov  • PubMed/NCBI        │
└─────────────────────────────────────────────────────────────┘
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

## Project Structure

```
├── adk-agent/              # AI Co-Scientist Agent (Python)
│   ├── agent.py            # Main agent with Gemini
│   ├── co_scientist/       # Agent module for evaluation
│   ├── evals/              # ADK evaluation test cases
│   ├── test_accuracy.py    # Data validation tests
│   └── README.md           # Agent documentation
│
├── research-mcp/           # Research Tools Server (Node.js)
│   ├── server.js           # MCP server with 13 tools
│   ├── data/               # Local datasets
│   └── README.md           # Tools documentation
│
└── README.md               # This file
```

## Available Tools

| Category | Tools |
|----------|-------|
| **Disease & Targets** | `search_diseases`, `search_disease_targets`, `get_target_info`, `search_targets` |
| **Druggability** | `check_druggability`, `get_target_drugs` |
| **Clinical Trials** | `search_clinical_trials`, `get_clinical_trial` |
| **Literature** | `search_pubmed`, `get_pubmed_abstract` |
| **Gene Info** | `get_gene_info` |
| **Local Data** | `list_local_datasets`, `read_local_dataset` |

## Testing

```bash
cd adk-agent

# Test data accuracy (API responses)
python test_accuracy.py

# Test MCP connection
python test_mcp.py

# Run agent evaluations
pytest test_agent_eval.py -v
```

## Data Sources

- **[Open Targets Platform](https://platform.opentargets.org/)** - Disease-target associations, druggability data
- **[ClinicalTrials.gov](https://clinicaltrials.gov/)** - Clinical trial registry
- **[PubMed/NCBI](https://pubmed.ncbi.nlm.nih.gov/)** - Scientific literature, gene information

