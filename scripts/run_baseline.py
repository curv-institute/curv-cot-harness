#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["litellm>=1.0.0"]
# ///
"""
Baseline Trace Transfer Experiment Runner (Experiment 001)

Establishes baseline measurements for cross-model trace transfer:
- Cross-model answer agreement
- Agreement-on-correct
- Agreement-on-wrong (consistent-wrong collapse)
- Transfer gain: Δ accuracy when target model receives trace vs no-trace

Uses litellm for unified API access to multiple model providers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import litellm

ROOT = Path(__file__).resolve().parents[1]

# Disable litellm telemetry and reduce verbosity
litellm.telemetry = False
litellm.set_verbose = False


def utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_git_info() -> dict[str, str]:
    """Get current git commit and tag info."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()[:12]
    except Exception:
        commit = "UNKNOWN"

    try:
        tag = subprocess.run(
            ["git", "describe", "--tags", "--always"],
            capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        tag = "UNKNOWN"

    return {"commit": commit, "tag": tag}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class ModelSpec:
    name: str       # logical name for manifests
    model: str      # litellm model identifier (e.g., "claude-3-5-haiku-latest", "gpt-4.1-mini")


def call_model(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Call a model and return the response text.

    For Claude models: uses Claude CLI (which has built-in auth)
    For GPT/OpenAI models: uses Codex CLI (with ChatGPT auth)
    For other models: uses litellm (requires API keys in environment)

    Model identifiers:
    - Claude CLI: "sonnet", "haiku", "opus" (or full names like "claude-3-5-sonnet-latest")
    - Codex CLI: "gpt-5-codex", or any model starting with "gpt-"
    - Other via litellm: requires OPENAI_API_KEY or other provider keys

    Environment variables:
    - CLAUDE_CMD: path to claude CLI (default: "claude")
    - CODEX_CMD: path to codex CLI (default: "/home/jwm/.bun/bin/codex")
    """
    # Use Claude CLI for Claude/Anthropic models
    if "claude" in model.lower() or model.lower() in ("sonnet", "haiku", "opus"):
        cmd = os.environ.get("CLAUDE_CMD", "claude")
        # Map short names to CLI model names
        model_arg = model
        if model.lower() == "sonnet":
            model_arg = "sonnet"
        elif model.lower() == "haiku":
            model_arg = "haiku"
        elif model.lower() == "opus":
            model_arg = "opus"

        args = [cmd, "-p", "--model", model_arg]
        proc = subprocess.run(
            args,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude CLI call failed (model={model})\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )
        return proc.stdout.strip()

    # Use Codex CLI for GPT/OpenAI models
    if model.lower().startswith("gpt-") or "codex" in model.lower():
        cmd = os.environ.get("CODEX_CMD", "/home/jwm/.bun/bin/codex")
        # For gpt-5-codex (default), don't pass model flag
        if model.lower() == "gpt-5-codex":
            args = [cmd, "exec", "-"]
        else:
            args = [cmd, "exec", "-m", model, "-"]

        proc = subprocess.run(
            args,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Codex CLI call failed (model={model})\n"
                f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
            )
        return proc.stdout.strip()

    # Use litellm for other models
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def extract_json(s: str) -> dict[str, Any]:
    """Extract JSON object from model response, handling markdown code blocks."""
    # Try direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    for pattern in patterns:
        match = re.search(pattern, s)
        if match:
            candidate = match.group(1) if "```" in pattern else match.group(0)
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract JSON object from response:\n{s}")


def proposer_prompt(template: str, prompt: str) -> str:
    return template.replace("{{PROMPT}}", prompt)


def target_prompt_with_trace(template: str, prompt: str, trace: str) -> str:
    return template.replace("{{PROMPT}}", prompt).replace("{{TRACE}}", trace)


def target_prompt_no_trace(template: str, prompt: str) -> str:
    return template.replace("{{PROMPT}}", prompt)


def pairwise_agreement(answers: list[str]) -> tuple[int, int]:
    """Returns (agree_pairs, total_pairs) for a list of normalized answers."""
    n = len(answers)
    total = 0
    agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if answers[i] == answers[j]:
                agree += 1
    return agree, total


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: scripts/run_baseline.py <run_name> <dataset_jsonl> <proposer_model> [<target_spec>...]\n\n"
            "Models use litellm identifiers:\n"
            "  - claude-3-5-sonnet-latest, claude-3-5-haiku-latest (Anthropic)\n"
            "  - gpt-4.1, gpt-4.1-mini, gpt-4o (OpenAI)\n\n"
            "Target spec format: <model>:<name> (name optional; defaults to model)\n"
            "Example: gpt-4.1-mini:gpt4mini\n\n"
            "Environment variables:\n"
            "  ANTHROPIC_API_KEY - for Claude models\n"
            "  OPENAI_API_KEY    - for OpenAI models"
        )
        return 2

    run_name = sys.argv[1]
    dataset_path = Path(sys.argv[2])
    proposer_model = sys.argv[3]

    targets: list[ModelSpec] = []
    for spec in sys.argv[4:]:
        parts = spec.split(":")
        model = parts[0]
        name = parts[1] if len(parts) >= 2 else model
        targets.append(ModelSpec(name=name, model=model))

    if not targets:
        raise ValueError("At least one target model is required.")

    outdir = ROOT / "eval" / "results" / run_name
    outdir.mkdir(parents=True, exist_ok=False)

    # Prompt templates (also saved verbatim)
    proposer_t = (
        "You will solve the task below.\n\n"
        "Return ONLY valid JSON with keys: \"trace\", \"final\".\n"
        "- \"trace\" is your reasoning trace.\n"
        "- \"final\" is your final answer.\n\n"
        "Task:\n{{PROMPT}}\n"
    )
    target_with_trace_t = (
        "You will solve the task below.\n\n"
        "A reasoning trace from another system is provided. Use it if helpful, but you must decide the final answer yourself.\n\n"
        "Return ONLY valid JSON with key: \"final\".\n\n"
        "Task:\n{{PROMPT}}\n\n"
        "Provided trace:\n{{TRACE}}\n"
    )
    target_no_trace_t = (
        "You will solve the task below.\n\n"
        "Return ONLY valid JSON with key: \"final\".\n\n"
        "Task:\n{{PROMPT}}\n"
    )

    (outdir / "prompts").mkdir(parents=True, exist_ok=True)
    (outdir / "prompts" / "proposer.txt").write_text(proposer_t, encoding="utf-8")
    (outdir / "prompts" / "target_with_trace.txt").write_text(target_with_trace_t, encoding="utf-8")
    (outdir / "prompts" / "target_no_trace.txt").write_text(target_no_trace_t, encoding="utf-8")

    items = read_jsonl(dataset_path)
    print(f"Loaded {len(items)} items from {dataset_path}")
    print(f"Proposer: {proposer_model}")
    print(f"Targets: {[t.name for t in targets]}")
    print()

    proposer_rows = []
    injected_rows = []
    answers_with_trace = []
    answers_no_trace = []
    correctness_rows = []
    metrics_rows = []

    for idx, it in enumerate(items):
        item_id = str(it["id"])
        q = str(it["prompt"])
        gold = str(it["answer"])

        print(f"[{idx+1}/{len(items)}] {item_id}...", end=" ", flush=True)

        # Proposer
        p_prompt = proposer_prompt(proposer_t, q)
        p_raw = call_model(proposer_model, p_prompt, temperature=0.0)
        p_obj = extract_json(p_raw)
        trace = str(p_obj.get("trace", ""))
        p_final = normalize_answer(str(p_obj.get("final", "")))

        proposer_rows.append({
            "id": item_id,
            "proposer_model": proposer_model,
            "trace": trace,
            "final": p_final,
            "raw": p_raw,
        })

        # Targets
        with_trace_finals: dict[str, str] = {}
        no_trace_finals: dict[str, str] = {}

        for t in targets:
            # with trace
            twt = target_prompt_with_trace(target_with_trace_t, q, trace)
            raw_w = call_model(t.model, twt, temperature=0.0)
            obj_w = extract_json(raw_w)
            fin_w = normalize_answer(str(obj_w.get("final", "")))
            with_trace_finals[t.name] = fin_w

            # no trace
            tnt = target_prompt_no_trace(target_no_trace_t, q)
            raw_n = call_model(t.model, tnt, temperature=0.0)
            obj_n = extract_json(raw_n)
            fin_n = normalize_answer(str(obj_n.get("final", "")))
            no_trace_finals[t.name] = fin_n

            answers_with_trace.append({"id": item_id, "model": t.name, "final": fin_w, "raw": raw_w})
            answers_no_trace.append({"id": item_id, "model": t.name, "final": fin_n, "raw": raw_n})

        # Correctness per model
        gold_n = normalize_answer(gold)
        corr_w = {m: (a == gold_n) for m, a in with_trace_finals.items()}
        corr_n = {m: (a == gold_n) for m, a in no_trace_finals.items()}
        correctness_rows.append({"id": item_id, "gold": gold_n, "with_trace": corr_w, "no_trace": corr_n})

        # Agreement metrics (targets only)
        agree_w, total_pairs = pairwise_agreement(list(with_trace_finals.values()))
        agree_n, _ = pairwise_agreement(list(no_trace_finals.values()))

        # Agreement-on-correct / wrong (targets only)
        all_agree_w = len(set(with_trace_finals.values())) == 1
        all_agree_n = len(set(no_trace_finals.values())) == 1

        aon_c_w = int(all_agree_w and (next(iter(with_trace_finals.values())) == gold_n))
        aon_w_w = int(all_agree_w and (next(iter(with_trace_finals.values())) != gold_n))
        aon_c_n = int(all_agree_n and (next(iter(no_trace_finals.values())) == gold_n))
        aon_w_n = int(all_agree_n and (next(iter(no_trace_finals.values())) != gold_n))

        metrics_rows.append({
            "id": item_id,
            "pairs_total": total_pairs,
            "agree_pairs_with_trace": agree_w,
            "agree_pairs_no_trace": agree_n,
            "all_agree_with_trace": all_agree_w,
            "all_agree_no_trace": all_agree_n,
            "agree_on_correct_with_trace": aon_c_w,
            "agree_on_wrong_with_trace": aon_w_w,
            "agree_on_correct_no_trace": aon_c_n,
            "agree_on_wrong_no_trace": aon_w_n,
        })

        injected_rows.append({"id": item_id, "trace": trace})

        # Quick status
        p_correct = "✓" if p_final == gold_n else "✗"
        print(f"proposer:{p_correct}", end="")
        for t in targets:
            tw = "✓" if with_trace_finals[t.name] == gold_n else "✗"
            tn = "✓" if no_trace_finals[t.name] == gold_n else "✗"
            print(f" {t.name}:w{tw}/n{tn}", end="")
        print()

    # Write artifacts
    write_jsonl(outdir / "traces" / "proposer.jsonl", proposer_rows)
    write_jsonl(outdir / "traces" / "injected.jsonl", injected_rows)
    write_jsonl(outdir / "judgments" / "answers_with_trace.jsonl", answers_with_trace)
    write_jsonl(outdir / "judgments" / "answers_no_trace.jsonl", answers_no_trace)
    write_jsonl(outdir / "judgments" / "correctness.jsonl", correctness_rows)
    write_jsonl(outdir / "metrics.jsonl", metrics_rows)

    # Aggregate summary
    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    agree_rate_w = mean([m["agree_pairs_with_trace"] / max(1, m["pairs_total"]) for m in metrics_rows])
    agree_rate_n = mean([m["agree_pairs_no_trace"] / max(1, m["pairs_total"]) for m in metrics_rows])

    aon_c_w = sum(m["agree_on_correct_with_trace"] for m in metrics_rows)
    aon_w_w = sum(m["agree_on_wrong_with_trace"] for m in metrics_rows)
    aon_c_n = sum(m["agree_on_correct_no_trace"] for m in metrics_rows)
    aon_w_n = sum(m["agree_on_wrong_no_trace"] for m in metrics_rows)

    # Accuracy per model
    acc_with = {}
    acc_no = {}
    for t in targets:
        cw = [r["with_trace"][t.name] for r in correctness_rows]
        cn = [r["no_trace"][t.name] for r in correctness_rows]
        acc_with[t.name] = sum(cw) / max(1, len(cw))
        acc_no[t.name] = sum(cn) / max(1, len(cn))

    # Proposer accuracy
    proposer_acc = sum(1 for r in proposer_rows if normalize_answer(r["final"]) == normalize_answer(
        next(c["gold"] for c in correctness_rows if c["id"] == r["id"])
    )) / max(1, len(proposer_rows))

    git_info = get_git_info()

    summary = {
        "run_name": run_name,
        "dataset": str(dataset_path),
        "n_items": len(items),
        "git": git_info,
        "proposer": {"model": proposer_model, "temperature": 0.0, "accuracy": proposer_acc},
        "targets": [{"name": t.name, "model": t.model} for t in targets],
        "agreement_rate": {"with_trace": agree_rate_w, "no_trace": agree_rate_n},
        "agree_all_counts": {
            "agree_on_correct_with_trace": aon_c_w,
            "agree_on_wrong_with_trace": aon_w_w,
            "agree_on_correct_no_trace": aon_c_n,
            "agree_on_wrong_no_trace": aon_w_n,
        },
        "accuracy": {
            "with_trace": acc_with,
            "no_trace": acc_no,
            "transfer_gain": {k: acc_with[k] - acc_no[k] for k in acc_no},
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Generate report
    report_lines = [
        f"# Report: {run_name}",
        "",
        f"**Dataset:** {dataset_path}",
        f"**Items:** {len(items)}",
        f"**Git:** {git_info['tag']} ({git_info['commit']})",
        "",
        "## Proposer",
        f"- Model: {proposer_model}",
        f"- Accuracy: {proposer_acc:.4f}",
        "",
        "## Agreement",
        f"- Pairwise agreement rate (with trace): {agree_rate_w:.4f}",
        f"- Pairwise agreement rate (no trace):  {agree_rate_n:.4f}",
        f"- Δ agreement: {agree_rate_w - agree_rate_n:+.4f}",
        "",
        "## Accuracy by Target Model",
        "",
        "| Model | No Trace | With Trace | Transfer Gain |",
        "|-------|----------|------------|---------------|",
    ]
    for m in acc_no:
        delta = acc_with[m] - acc_no[m]
        report_lines.append(f"| {m} | {acc_no[m]:.4f} | {acc_with[m]:.4f} | {delta:+.4f} |")

    report_lines.extend([
        "",
        "## All-Target Agreement Outcomes",
        "",
        "| Condition | Agree-on-Correct | Agree-on-Wrong |",
        "|-----------|------------------|----------------|",
        f"| With Trace | {aon_c_w} | {aon_w_w} |",
        f"| No Trace | {aon_c_n} | {aon_w_n} |",
        "",
        "## Summary Questions",
        "",
        f"1. **Did trace injection increase agreement?** {'Yes' if agree_rate_w > agree_rate_n else 'No'} (Δ = {agree_rate_w - agree_rate_n:+.4f})",
        "",
        "2. **Did trace injection increase accuracy?**",
    ])
    for m in acc_no:
        delta = acc_with[m] - acc_no[m]
        answer = "Yes" if delta > 0 else ("No change" if delta == 0 else "No (decreased)")
        report_lines.append(f"   - {m}: {answer} (Δ = {delta:+.4f})")

    consistent_wrong_change = aon_w_w - aon_w_n
    report_lines.extend([
        "",
        f"3. **Did consistent-wrong increase?** {'Yes' if consistent_wrong_change > 0 else 'No'} ({aon_w_n} → {aon_w_w}, Δ = {consistent_wrong_change:+d})",
    ])

    (outdir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    # Manifest
    manifest = {
        "run_name": run_name,
        "created_utc": utc_ts_compact(),
        "git": git_info,
        "experiment": "baseline-singletrace",
        "axis": "trace_injection",
        "conditions": ["with_trace", "no_trace"],
        "notes": "Baseline: single proposer trace injection vs no-trace control. No harmonization.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Config
    (outdir / "config.json").write_text(
        json.dumps({
            "dataset": str(dataset_path),
            "proposer": {"model": proposer_model, "temperature": 0.0},
            "targets": [{"name": t.name, "model": t.model, "temperature": 0.0} for t in targets],
        }, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 60)
    print(f"Run complete: {run_name}")
    print(f"Results: {outdir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
