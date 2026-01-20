AGENTS.md — Repository Agent Rules

This file contains authoritative rules for all automated or human agents working in this repository.

⸻

Project Overview

Repository: https://github.com/curv-institute/curv-cot-harness

Purpose: Build a reproducible experimental harness to evaluate cross-model transfer of reasoning traces (e.g., chain-of-thought or short-step rationales), and to quantify when shared traces increase agreement, accuracy, or induce consistent-wrong collapse.

Primary hypothesis: A harmonized trace operator (multiple independent proposers followed by a selector) yields traces that improve cross-model agreement and agreement-on-correct relative to single-model traces.

Secondary hypothesis (risk): Optimizing for agreement alone increases the rate of consistent-wrong answers; explicit guardrails are required to detect and limit this failure mode.

⸻

Authoritative Rules

1. Determinism & Reproducibility

All implementations must:
	•	Be deterministic given a fixed seed and fixed model snapshot
	•	Record full configuration snapshots
	•	Write manifests for every run
	•	Isolate outputs per run (eval/results/<run_name>/)

Randomness must be:
	•	Explicitly seeded
	•	Logged
	•	Reproducible

Model determinism policy:
	•	If using API models, record: model name, snapshot/version (if available), temperature, top_p, max tokens, system prompt, and any sampling parameters.
	•	Default to temperature = 0 unless stochasticity is the explicit experiment axis.

2. Version Control Discipline (jj + git)

Use Jujutsu (jj) as the primary interface for local version control, with git used only for remote interoperability.

Required workflow:
	•	Use small, single-purpose commits
	•	One logical change per commit
	•	Commit messages must be imperative and descriptive

Push discipline (mandatory):
	•	After completing a task: jj commit then jj git push
	•	The canonical remote must always reflect the latest completed state
	•	No local-only commits unless explicitly instructed

History hygiene:
	•	Do not rewrite published history
	•	Do not squash commits unless explicitly instructed
	•	Prefer linear, readable history

3. Per-Run Output Isolation

Never overwrite results from previous runs. Always write to:

eval/results/<run_name>/
  manifest.json
  config.json
  prompts/
  traces/
  judgments/
  metrics.jsonl
  summary.json
  report.md
  figures/
  tables/
  manifests/

Required artifacts (minimum):
	•	prompts/: exact prompts used per item
	•	traces/: proposer traces and final selected trace
	•	judgments/: answers, correctness labels, verifier outputs
	•	metrics.jsonl: per-item metric records (one JSON object per item)
	•	summary.json: aggregated metrics with confidence intervals where applicable

3A. Experiment Naming Convention (Mandatory)

Run names must encode the git tag or version to ensure traceability.

Pattern: <tag>-<experiment>-<timestamp> or <experiment>_<tag>

Examples:
	•	v0.1.0-baseline-singletrace-20260120
	•	v0.1.0-harmonized-selector-20260120
	•	v0.1.1-ablation-noverifier-20260121
	•	baseline_v0.1.0

Enforcement:
	•	manifest.json must include run_name, git tag or commit hash, and the explicit experiment axis.

4. Parallel Subagent Usage
	•	Decompose tasks into independent modules wherever possible
	•	Use parallel subagents to reduce wall-clock time
	•	Serialize work only when interfaces require coordination
	•	Define interfaces and contracts first
	•	Require mergeable, unambiguous outputs from subagents

5. Scope Discipline

Must:
	•	Keep experiments single-axis per cycle
	•	Separate mechanism, measurement, and interpretation
	•	Treat agreement as a stability signal, not correctness

Must not:
	•	Overclaim causal mechanisms from metric shifts
	•	Broaden scope beyond cross-model trace transfer
	•	Turn this project into general reasoning optimization without explicit instruction

6. Python Implementation Rules

All Python code must:
	•	Be compatible with Python >= 3.12
	•	Use PEP 723 inline script headers for runnable scripts
	•	Be executable via uv run
	•	Avoid global mutable state
	•	Log seeds and configuration at runtime

No hidden dependencies:
	•	Every runnable script must declare dependencies in the PEP 723 header.

7. Metrics & Claims

Primary metrics (required):
	•	Pairwise cross-model agreement (answer-level)
	•	Agreement-on-correct
	•	Agreement-on-wrong (consistent-wrong rate)
	•	Transfer gain: Δ accuracy of target models with trace vs no-trace baseline

Secondary metrics (optional; must be pre-registered if used):
	•	Trace length and compression ratio
	•	Per-item collapse index (agreement ↑, accuracy ↓)
	•	Robustness under prompt perturbation (only if it is the experiment axis)

Claims discipline:
	•	Never conflate agreement with truth
	•	Any claim of improved generalization must report both agreement and correctness
	•	Negative results are first-class and must be reported
	•	Trade-offs must be stated explicitly

8. Prompt & Trace Archival (AGENT)

Any prompt longer than two lines must be saved as:

AGENT/<UNIXTIME>-in.md   # prompt
AGENT/<UNIXTIME>-out.md  # output

Experiment artifacts:
	•	Prompts used in runs must be copied into eval/results/<run_name>/prompts/
	•	Model outputs and traces must be copied into eval/results/<run_name>/traces/

All AGENT files must be committed and pushed.

9. Evaluation Integrity
	•	Benchmarks and datasets must be versioned and checksummed
	•	Public datasets require URL + checksum or commit hash
	•	Synthetic tasks require generator code and seed

No label leakage:
	•	Traces provided to target models must not contain gold answers
	•	Verifiers must be isolated from proposers unless explicitly tested

⸻

Non-Goals
	•	Do not introduce new learning objectives unless explicitly requested
	•	Do not claim general intelligence or reasoning breakthroughs
	•	Do not optimize benchmark scores without reporting failure modes

⸻

Guiding Principle

Agreement is a stability signal, not a truth guarantee.
