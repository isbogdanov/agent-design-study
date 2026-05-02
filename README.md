# Context, Reasoning, and Hierarchy — Reproducibility Artifact

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Artifact for the ACM CAIS 2026 paper:
> **Context, Reasoning, and Hierarchy: A Cost-Performance Study of Compound LLM Agent Design in an Adversarial POMDP**

This repository contains the full agent implementation, experiment runner, and all YAML configuration snapshots needed to reproduce the three-axis ablation study reported in the paper (72 model–configuration pairs, 3,475 episodes).

## Archived Artifact

Archived artifact: https://doi.org/10.5281/zenodo.19908100  
Development repository: https://github.com/isbogdanov/agent-design-study

---

## Overview

The system evaluates a compound LLM agent defending a network in the **CybORG CAGE-2** environment. The agent architecture is fully driven by declarative YAML definitions — no code changes are required to reproduce any experimental condition.

Three design axes are studied:

| Axis | Dimension | Configs |
|---|---|---|
| 1 | **Context** — what the agent sees | 6 variants (`obs`, `obs+hist`, `obs+hist+net`, `obs+net`, `network`, `hist+net`) |
| 2 | **Deliberation** — reasoning depth (cumulative) | 4 levels (`+question`, `+critique`, `+improve`, `+COT`) |
| 3 | **Hierarchy** — task decomposition | 2 configs (`hier-base`, `hier-delib`) |

All 12 configuration snapshots are pre-built in `exp_configs/`. Switching between them requires changing one line in `experiment_agent_eval.yaml`.

---

## Repository Structure

```
.
├── agent_base/                        # Agent implementation (mounted into Docker)
│   ├── run_cyborg_coordinator.py      # Per-instance entry point (runs inside Docker)
│   ├── agents/
│   │   └── prompts/definitions/       # Live definitions (hist+net anchor)
│   │       ├── planner/               # core.yaml, initial_prompt.yaml, persistent_knowledge.yaml
│   │       ├── analyst/
│   │       └── action_chooser/
│   ├── coordinators/                  # CybORG coordinator and agent coordinator
│   ├── llm-connector/                 # LLM provider abstraction
│   │   └── conf/                      # llm.yaml, logs.yaml, security.yaml
│   └── utils/                         # Settings, logging utilities
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
│   ├── full_hierarchy/                # Axis 3: hier-base (delegation, no deliberation)
│   └── hier_on/                       # Axis 3: hier-delib (delegation + deliberation on all agents)
├── experiment_agent_eval.yaml         # Experiment configuration (edit here)
├── run_experiment.py                  # Experiment runner (parallelises Docker instances)
├── Dockerfile                         # Container with CybORG + agent dependencies
├── container_requirements.txt         # Python dependencies (used by Docker)
├── .env.template                      # API key template — copy to .env
├── .gitignore
└── LICENSE
```

---

## Prerequisites

- **Docker** — the agent and CybORG run entirely inside the container.
- **Python 3.9+** — only for `run_experiment.py` (the outer orchestrator); standard library + `pyyaml` required, `rich` optional (see [Live progress dashboard](#live-progress-dashboard) below).
- **API keys** — at least one LLM provider key in `.env`.

### `.env` setup

```bash
cp .env.template .env
```

Open `.env` and fill in your key(s):

```bash
OPENROUTER_API_KEY=sk-or-v1-...   # recommended — single key, access to all models
GOOGLE_API_KEY=AIza...             # for direct Google AI Studio access
```

---

## Quick Start

### 1. Build the Docker image

```bash
docker build -t cyborg-agent .
```

Installs all Python dependencies and patches the CybORG CAGE-2 data files. Takes ~3–5 minutes on first build; subsequent builds are cached.

### 2. Configure the experiment

Open `experiment_agent_eval.yaml` and uncomment the `definitions_source` for the config you want to run:

```yaml
# To reproduce the best-performing configuration (hier-base):
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

### Live progress dashboard

Add `--progress` to display a live per-instance table that updates every 2 seconds while containers run:

```bash
python3 run_experiment.py --config experiment_agent_eval.yaml --progress
```

Requires the `rich` library (install once: `pip install rich`, or `pip install -r requirements.txt`).

**What is displayed:**

```
╭────────────┬──────────┬─────────────────────────────────────────┬────────────┬──────────────────╮
│  Instance  │   Run    │ Step Progress                           │    Elapsed │ Step Rew / Total │
├────────────┼──────────┼─────────────────────────────────────────┼────────────┼──────────────────┤
│     #1     │   2/5    │ [████████░░░░░░░░░░░░]  8✓  ▶  9/30    │    04m 21s │     -1.0 / -14.5 │
│     #2     │   1/5    │ [░░░░░░░░░░░░░░░░░░░░]   —  ▶  1/30    │    04m 18s │         — / —    │
│     #3     │   3/5    │ [████████████████████]  ✓ 30/30         │    04m 23s │         — / -4.3 │
╰────────────┴──────────┴─────────────────────────────────────────┴────────────┴──────────────────╯
```

| Column | Meaning |
|---|---|
| **Run** | Current evaluation run out of total (`2/5`) |
| **Step Progress** | Bar = completed steps; `8✓` = last finished step; `▶ 9/30` = step actively executing |
| **Step Rew / Total** | Per-step reward from last completed step / cumulative episode total so far |
| **Total Reward colour** | 🟢 green `> −50`, 🟡 yellow `> −100`, 🟠 orange `> −150`, 🔴 red `≤ −150` |

Without `--progress` the script behaves exactly as before (no change to output or behaviour).

---

## Expected Output

Results are written to `experiments/<experiment_name>_<timestamp>/aggregated_logs/`.

### `evaluation_report.md` (per-instance reward table)

```markdown
# Evaluation Report

## Summary Statistics
- Total Evaluation Runs: 50
- Average Reward: -24.0
- Min Reward: -71.2
- Max Reward: -3.1

## Per-Instance Evaluation Rewards

| Instance    | Run 1  | Run 2  | Run 3  | Run 4  | Run 5  | Avg    |
|-------------|--------|--------|--------|--------|--------|--------|
| instance_1  | -18.3  | -22.1  | -19.5  | -31.2  | -28.0  | -23.8  |
| instance_2  | -20.1  | -25.4  | -17.8  | -24.9  | -26.2  | -22.9  |
| ...         |        |        |        |        |        |        |
```

### `summary.md` (aggregate statistics)

```markdown
# Experiment Summary

## Overview
- Total Instances: 10
- Average Reward (all runs): -24.0

## Detailed Results
| Instance   | Eval Runs | Rewards            | Avg Reward |
|------------|-----------|--------------------|------------|
| instance_1 | 5         | -18.3, -22.1, ...  | -23.8      |
| instance_2 | 5         | -20.1, -25.4, ...  | -22.9      |
```

### Sanity-check values

Expected mean episode return from Table 3 of the paper (Gemini-2.5-Flash-Lite):

| Config | Expected mean return |
|---|---|
| `obs` (raw observation only) | ~ −215 |
| `hist+net` (anchor) | ~ −209 |
| `hier-base` *(best config)* | ~ −183 |
| `hier-delib` | ~ −186 |

> Returns are ≤ 0; closer to zero is better. Values vary by model — the table above is for Gemini-2.5-Flash-Lite specifically. See Table 3 of the paper for all six models.

---

## Reproducing Paper Results

Each row in Table 3 of the paper corresponds to one `exp_configs/` folder.

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
| `hier-base` | `exp_configs/full_hierarchy` |
| `hier-delib` | `exp_configs/hier_on` |

The paper uses **10 instances × 5 runs = 50 episodes per config per model**. Set in `experiment_agent_eval.yaml`:

```yaml
num_instances: 10
max_parallel_workers: 10
num_evaluation_runs: 5
```

All models use deterministic decoding (`temperature: 0` or provider minimum). No per-model tuning is performed.

> **Data availability.** The complete episode logs collected for the paper (raw console logs, token usage, per-step reward traces across all experiments and evaluated episodes) are not included in this repository due to size. They may be available upon request from the authors.

---

## Output Structure

```
experiments/<experiment_name>_<timestamp>/
├── experiment_config.yaml                  # Copy of the config used for this run
└── aggregated_logs/
    ├── evaluation_report.md                # Per-instance reward table
    ├── summary.md                          # Aggregate statistics
    └── instance_1/
        └── runs/evaluating/
            └── run_<timestamp>/
                ├── <timestamp>_console_mirror.log   # Full agent transcript
                └── connector/                       # LLM token usage logs
```

---

## Providers

API keys are read from `.env`. The `provider` value in the YAML determines which key is used:

| `provider` | Env var | Notes |
|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | Recommended — single key, all models |
| `google` | `GOOGLE_API_KEY` | Google AI Studio direct |
| `vertex` | `VERTEX_API_KEY` | Google Vertex AI |

### Switching models

```yaml
# Via OpenRouter (single key, all supported models):
provider: "openrouter"
model: "google/gemini-2.5-flash-lite"   # G2.5FL (paper default)
model: "x-ai/grok-4.1-fast"             # Grok
model: "meta-llama/llama-4-maverick"    # Llama
model: "qwen/qwen3-235b-a22b-2507"      # Qwen
model: "mistralai/devstral-2512"        # Devstral
model: "google/gemini-3-flash-preview"  # G3FP

# Direct Google AI Studio:
provider: "google"
model: "gemini-2.5-flash-lite"
```

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

## Environment: CybORG CAGE-2

Evaluated on [CybORG CAGE-2](https://github.com/cage-challenge/cage-challenge-2) — a simulated network-defense environment (13-host enterprise network, 30-step horizon, automated B-line red attacker). The Dockerfile automatically installs and patches the necessary data files.

---

## License

This artifact is released under the **Apache License 2.0** — see [`LICENSE`](LICENSE) for the full text.

The CybORG CAGE-2 environment is subject to its own license; see the [CybORG repository](https://github.com/cage-challenge/cage-challenge-2) for details.
