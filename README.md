# Compound LLM Agent Design Study — Reproducibility Artifact

Artifact for the ACM CAIS 2026 paper:

> **Compound LLM Agent Design in Adversarial POMDPs: A Cost-Performance Study of Context, Deliberation, and Hierarchy**

This repository contains the full agent implementation, experiment runner, and all YAML configuration snapshots needed to reproduce the three-axis ablation study reported in the paper (72 model–configuration pairs, 3,475 episodes).

---

## Overview

The system evaluates a compound LLM agent defending a network in the **CybORG CAGE-2** environment. The agent architecture is fully driven by declarative YAML definitions — no code changes are required to reproduce any experimental condition.

Three design axes are studied:

| Axis | Dimension | Configs |
|---|---|---|
| 1 | **Context** — what the agent sees | 6 variants (`obs`, `obs+hist`, `obs+hist+net`, `obs+net`, `network`, `hist+net`) |
| 2 | **Deliberation** — reasoning depth (cumulative) | 4 levels (`+question`, `+critique`, `+improve`, `+COT`) |
| 3 | **Hierarchy** — task decomposition | 2 configs (`hier-off`, `hier-on`) |

All 12 configuration snapshots are pre-built in `exp_configs/`. Switching between them requires changing one line in `experiment_agent_eval.yaml`.

---

## Repository Structure

```
.
├── agent_base/                        # Agent implementation
│   ├── agents/
│   │   └── prompts/
│   │       └── definitions/           # Live definitions (hist+net anchor)
│   ├── coordinators/                  # CybORG coordinator and agent coordinator
│   ├── utils/                         # Settings, logging utilities
│   └── run_cyborg_coordinator.py      # Entry point (runs inside Docker)
├── exp_configs/                       # All 12 paper ablation snapshots
│   ├── obs/                           # Axis 1: raw observation only
│   ├── obs_hist/                      # Axis 1: obs + history
│   ├── obs_hist_net/                  # Axis 1: obs + history + network_status
│   ├── obs_net/                       # Axis 1: obs + network_status
│   ├── network/                       # Axis 1: network_status only
│   ├── hist_net/                      # Axis 1: anchor (hist + network_status)
│   ├── delib_question/                # Axis 2: +question tool
│   ├── delib_critique/                # Axis 2: +question +critique
│   ├── delib_improve/                 # Axis 2: +question +critique +improve
│   ├── delib_cot/                     # Axis 2: all tools + COT instruction
│   ├── full_hierarchy/                # Axis 3: hier-off (delegation, no deliberation)
│   └── hier_on/                       # Axis 3: hier-on (delegation + deliberation)
├── experiment_agent_eval.yaml         # Experiment configuration (edit here)
├── run_experiment.py                  # Experiment runner (parallelises Docker instances)
├── Dockerfile                         # Container with CybORG + agent dependencies
└── .env                               # API keys (not committed — fill in before running)
```

---

## Prerequisites

- **Docker** — the agent and CybORG run entirely inside the container.
- **Python 3.9+** — only for `run_experiment.py` (the outer orchestrator); no extra packages beyond the standard library and `pyyaml`.
- **API keys** — add your LLM provider keys to `.env` (see `.env` template below).

### `.env` template

```bash
OPENROUTER_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here        # for Gemini models
```

---

## Quick Start

### 1. Build the Docker image

```bash
docker build -t cyborg-agent .
```

### 2. Configure the experiment

Open `experiment_agent_eval.yaml` and:
1. Uncomment the `definitions_source` line for the config you want to run.
2. Set the model under `agent_config`.

```yaml
# To reproduce the best-performing configuration (hier-off, Gemini-2.5-Flash-Lite):
definitions_source: "exp_configs/full_hierarchy"

agent_config:
  steps: 30
  provider: "openrouter"
  model: "google/gemini-2.5-flash-lite"
```

### 3. Run

```bash
python3 run_experiment.py --config experiment_agent_eval.yaml
```

Results are written to `experiments/<experiment_name>_<timestamp>/aggregated_logs/`:
- `evaluation_report.md` — per-instance reward table
- `summary.md` — aggregate statistics

---

## Reproducing Paper Results

Each row in Table 2 of the paper corresponds to one `exp_configs/` folder.

| Paper config | `definitions_source` |
|---|---|
| `obs` | `exp_configs/obs` |
| `obs+hist` | `exp_configs/obs_hist` |
| `obs+hist+net` | `exp_configs/obs_hist_net` |
| `obs+net` | `exp_configs/obs_net` |
| `network` | `exp_configs/network` |
| `hist+net` *(anchor)* | `exp_configs/hist_net` |
| `+question` | `exp_configs/delib_question` |
| `+critique` | `exp_configs/delib_critique` |
| `+improve` | `exp_configs/delib_improve` |
| `+COT` | `exp_configs/delib_cot` |
| `hier-off` | `exp_configs/full_hierarchy` |
| `hier-on` | `exp_configs/hier_on` |

The paper uses **10 instances × 5 runs = 50 episodes per config per model**. Set in `experiment_agent_eval.yaml`:

```yaml
num_instances: 10
max_parallel_workers: 10
num_evaluation_runs: 5
```

All models use deterministic decoding (`temperature: 0` or provider minimum). No per-model tuning is performed.

---

## Architecture

The key architectural invariant is the **engine–personality separation**: a shared ReAct execution engine (`run_cyborg_coordinator.py`) is paired with declarative YAML "personalities" in `definitions/`. Every experimental variant is a configuration change, not a code change.

Each agent (Planner, Analyst, ActionChooser) is defined by three YAML files:

| File | Purpose |
|---|---|
| `core.yaml` | Agent type, tool flags, deliberation toggles, system message |
| `initial_prompt.yaml` | Per-step prompt template with context placeholders |
| `persistent_knowledge.yaml` | Static domain knowledge (action glossary) |

Context injections into `initial_prompt.yaml`:

| Placeholder | Content |
|---|---|
| `{observation}` | Raw CybORG observation dictionary |
| `{network_status}` | Deterministic structured summary of non-baseline hosts |
| `{history}` | Compressed action log with smart step collapsing |

Deliberation tools toggled in `core.yaml`:

| Flag | Tool |
|---|---|
| `include_tool_raise_a_question` | Agent questions its own reasoning |
| `include_tool_critique_the_answer` | Agent critiques its response |
| `include_tool_improve_based_on_critique` | Agent revises in light of critique |
| `include_COT_instruction` | Injects explicit chain-of-thought instruction |

---

## License

See `LICENSE` for terms. The CybORG CAGE-2 environment is subject to its own license; see the CybORG repository for details.
