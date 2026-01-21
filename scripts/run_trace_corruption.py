#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["litellm>=1.0.0"]
# ///
"""
Trace Corruption Stress Test (Experiment 004)

Tests whether injected traces can induce consistent-wrong collapse when
the trace is plausible but wrong.

Axis: trace quality (uncorrupted vs corrupted)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import litellm

ROOT = Path(__file__).resolve().parents[1]

litellm.telemetry = False
litellm.set_verbose = False

K_CORRUPT = 15  # Number of items to corrupt


def utc_ts_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_git_info() -> dict[str, str]:
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
            if line:
                items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def normalize_answer(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str


def call_model(model: str, prompt: str, temperature: float = 0.0) -> str:
    """Call a model via CLI or litellm."""
    if "claude" in model.lower() or model.lower() in ("sonnet", "haiku", "opus"):
        cmd = os.environ.get("CLAUDE_CMD", "claude")
        model_arg = model
        if model.lower() == "sonnet":
            model_arg = "sonnet"
        elif model.lower() == "haiku":
            model_arg = "haiku"
        elif model.lower() == "opus":
            model_arg = "opus"
        args = [cmd, "-p", "--model", model_arg]
        proc = subprocess.run(args, input=prompt, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Claude CLI failed: {proc.stderr}")
        return proc.stdout.strip()

    if model.lower().startswith("gpt-") or "codex" in model.lower():
        cmd = os.environ.get("CODEX_CMD", "/home/jwm/.bun/bin/codex")
        if model.lower() == "gpt-5-codex":
            args = [cmd, "exec", "-"]
        else:
            args = [cmd, "exec", "-m", model, "-"]
        proc = subprocess.run(args, input=prompt, text=True, capture_output=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Codex CLI failed: {proc.stderr}")
        return proc.stdout.strip()

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def extract_json(s: str) -> dict[str, Any]:
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    patterns = [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```", r"\{[\s\S]*\}"]
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
    raise ValueError(f"Could not extract JSON: {s[:200]}")


def select_corrupted_ids(item_ids: list[str], k: int = K_CORRUPT) -> list[str]:
    """Select k items for corruption using SHA256 hash (deterministic)."""
    hashed = [(hashlib.sha256(id_.encode()).hexdigest(), id_) for id_ in item_ids]
    hashed.sort(key=lambda x: x[0])
    return [id_ for _, id_ in hashed[:k]]


def corrupt_trace(trace: str, final: str, item_id: str) -> tuple[str, str]:
    """
    Corrupt a trace by introducing a single minimal arithmetic mistake.

    Strategy:
    1. Find the first number that appears to be a calculation result
    2. Modify it by ±1 (deterministic based on item_id hash)
    3. Update the final answer accordingly

    Returns (corrupted_trace, corrupted_final).
    """
    # Use hash to determine corruption direction (+1 or -1)
    h = hashlib.sha256(item_id.encode()).digest()
    delta = 1 if h[0] % 2 == 0 else -1

    # Find numbers in the trace that look like intermediate results
    # Pattern: "= X" where X is a number
    pattern = r'=\s*(\d+)'
    matches = list(re.finditer(pattern, trace))

    if not matches:
        # Fallback: just modify the final answer
        try:
            final_num = int(final)
            return trace, str(final_num + delta)
        except ValueError:
            return trace, final + " (corrupted)"

    # Take the first match and corrupt it
    first_match = matches[0]
    original_value = int(first_match.group(1))
    corrupted_value = original_value + delta

    # Replace in trace
    start, end = first_match.start(1), first_match.end(1)
    corrupted_trace = trace[:start] + str(corrupted_value) + trace[end:]

    # If there are subsequent numbers that depend on this, we need to propagate
    # For simplicity, we'll also modify the final answer by the same delta
    try:
        final_num = int(final)
        # The error propagates: if we changed an intermediate by delta,
        # the final should also change (in most arithmetic problems)
        corrupted_final = str(final_num + delta)
    except ValueError:
        corrupted_final = final

    return corrupted_trace, corrupted_final


def pairwise_agreement(answers: list[str]) -> tuple[int, int]:
    n = len(answers)
    total = agree = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if answers[i] == answers[j]:
                agree += 1
    return agree, total


def plurality_answer(answers: list[str]) -> tuple[str, int]:
    """Return (most common answer, count)."""
    counter = Counter(answers)
    if not counter:
        return "", 0
    most_common = counter.most_common(1)[0]
    return most_common


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: run_trace_corruption.py <run_name> <dataset> <proposer_model> [<target_spec>...]\n\n"
            "Target spec format: <model>:<name>\n"
            "Example: gpt-5-codex:gpt5_codex"
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
        raise ValueError("At least one target required.")

    outdir = ROOT / "eval" / "results" / run_name
    outdir.mkdir(parents=True, exist_ok=False)

    # Prompt templates
    proposer_t = (
        "You will solve the task below.\n\n"
        "Return ONLY valid JSON with keys: \"trace\", \"final\".\n"
        "- \"trace\" is your reasoning trace.\n"
        "- \"final\" is your final answer.\n\n"
        "Task:\n{{PROMPT}}\n"
    )
    target_with_trace_t = (
        "You will solve the task below.\n\n"
        "A reasoning trace from another system is provided. Use it if helpful, "
        "but you must decide the final answer yourself.\n\n"
        "Return ONLY valid JSON with key: \"final\".\n\n"
        "Task:\n{{PROMPT}}\n\n"
        "Provided trace:\n{{TRACE}}\n"
    )
    target_no_trace_t = (
        "You will solve the task below.\n\n"
        "Return ONLY valid JSON with key: \"final\".\n\n"
        "Task:\n{{PROMPT}}\n"
    )

    # Save prompts
    (outdir / "prompts").mkdir(parents=True, exist_ok=True)
    (outdir / "prompts" / "proposer.txt").write_text(proposer_t)
    (outdir / "prompts" / "target_with_trace.txt").write_text(target_with_trace_t)
    (outdir / "prompts" / "target_no_trace.txt").write_text(target_no_trace_t)

    corruption_spec = f"""Corruption Specification
========================

Selection: K={K_CORRUPT} items selected by sorting SHA256(item_id) ascending, taking first {K_CORRUPT}.

Corruption method:
1. Find first "= <number>" pattern in trace
2. Modify number by +1 or -1 (determined by SHA256(item_id)[0] % 2)
3. Modify final answer by same delta

Constraints:
- Deterministic given item_id
- Minimal change (single arithmetic error)
- Plausible (no obviously wrong text)
"""
    (outdir / "prompts" / "corruption_spec.txt").write_text(corruption_spec)

    items = read_jsonl(dataset_path)
    item_ids = [str(it["id"]) for it in items]
    corrupted_ids = set(select_corrupted_ids(item_ids, K_CORRUPT))

    # Save manifests
    (outdir / "manifests").mkdir(parents=True, exist_ok=True)
    (outdir / "manifests" / "corrupted_ids.json").write_text(
        json.dumps(sorted(corrupted_ids), indent=2)
    )
    subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False)
    (outdir / "manifests" / "claude_cli_version.txt").write_text(
        subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    )
    (outdir / "manifests" / "codex_cli_version.txt").write_text(
        subprocess.run(["/home/jwm/.bun/bin/codex", "--version"], capture_output=True, text=True).stdout.strip()
    )
    (outdir / "manifests" / "target_models_list.txt").write_text(
        "\n".join([f"{t.name}: {t.model}" for t in targets])
    )

    print(f"Loaded {len(items)} items from {dataset_path}")
    print(f"Proposer: {proposer_model}")
    print(f"Targets: {[t.name for t in targets]}")
    print(f"Corrupted items: {len(corrupted_ids)}")
    print()

    # Phase 1: Generate proposer traces
    print("=== Phase 1: Generating proposer traces ===")
    proposer_uncorrupted = []
    proposer_corrupted = []

    for idx, it in enumerate(items):
        item_id = str(it["id"])
        q = str(it["prompt"])
        gold = str(it["answer"])

        print(f"[{idx+1}/{len(items)}] {item_id}...", end=" ", flush=True)

        p_prompt = proposer_t.replace("{{PROMPT}}", q)
        p_raw = call_model(proposer_model, p_prompt, temperature=0.0)
        p_obj = extract_json(p_raw)
        trace = str(p_obj.get("trace", ""))
        p_final = normalize_answer(str(p_obj.get("final", "")))

        proposer_uncorrupted.append({
            "id": item_id,
            "trace": trace,
            "final": p_final,
            "raw": p_raw,
        })

        # Create corrupted version if selected
        if item_id in corrupted_ids:
            c_trace, c_final = corrupt_trace(trace, p_final, item_id)
            proposer_corrupted.append({
                "id": item_id,
                "trace_corrupted": c_trace,
                "final_corrupted": c_final,
                "original_trace": trace,
                "original_final": p_final,
            })
            print(f"corrupted ({p_final}→{c_final})")
        else:
            # Non-corrupted items: use original
            proposer_corrupted.append({
                "id": item_id,
                "trace_corrupted": trace,
                "final_corrupted": p_final,
                "original_trace": trace,
                "original_final": p_final,
            })
            print("ok")

    write_jsonl(outdir / "traces" / "proposer_uncorrupted.jsonl", proposer_uncorrupted)
    write_jsonl(outdir / "traces" / "proposer_corrupted.jsonl", proposer_corrupted)

    # Build lookup dicts
    uncorrupted_by_id = {r["id"]: r for r in proposer_uncorrupted}
    corrupted_by_id = {r["id"]: r for r in proposer_corrupted}

    # Phase 2: Run targets under all conditions
    print("\n=== Phase 2: Running targets ===")
    answers_no_trace = []
    answers_uncorrupted = []
    answers_corrupted = []
    correctness_rows = []
    metrics_rows = []

    for idx, it in enumerate(items):
        item_id = str(it["id"])
        q = str(it["prompt"])
        gold = normalize_answer(str(it["answer"]))
        is_corrupted = item_id in corrupted_ids

        trace_u = uncorrupted_by_id[item_id]["trace"]
        trace_c = corrupted_by_id[item_id]["trace_corrupted"]
        final_c = corrupted_by_id[item_id]["final_corrupted"]

        print(f"[{idx+1}/{len(items)}] {item_id}{'*' if is_corrupted else ''}...", end=" ", flush=True)

        no_trace_finals: dict[str, str] = {}
        uncorrupted_finals: dict[str, str] = {}
        corrupted_finals: dict[str, str] = {}

        for t in targets:
            # No trace
            tnt = target_no_trace_t.replace("{{PROMPT}}", q)
            raw_n = call_model(t.model, tnt, temperature=0.0)
            obj_n = extract_json(raw_n)
            fin_n = normalize_answer(str(obj_n.get("final", "")))
            no_trace_finals[t.name] = fin_n
            answers_no_trace.append({"id": item_id, "model": t.name, "final": fin_n, "raw": raw_n})

            # Uncorrupted trace
            twu = target_with_trace_t.replace("{{PROMPT}}", q).replace("{{TRACE}}", trace_u)
            raw_u = call_model(t.model, twu, temperature=0.0)
            obj_u = extract_json(raw_u)
            fin_u = normalize_answer(str(obj_u.get("final", "")))
            uncorrupted_finals[t.name] = fin_u
            answers_uncorrupted.append({"id": item_id, "model": t.name, "final": fin_u, "raw": raw_u})

            # Corrupted trace
            twc = target_with_trace_t.replace("{{PROMPT}}", q).replace("{{TRACE}}", trace_c)
            raw_c = call_model(t.model, twc, temperature=0.0)
            obj_c = extract_json(raw_c)
            fin_c = normalize_answer(str(obj_c.get("final", "")))
            corrupted_finals[t.name] = fin_c
            answers_corrupted.append({"id": item_id, "model": t.name, "final": fin_c, "raw": raw_c})

        # Correctness
        corr_n = {m: (a == gold) for m, a in no_trace_finals.items()}
        corr_u = {m: (a == gold) for m, a in uncorrupted_finals.items()}
        corr_c = {m: (a == gold) for m, a in corrupted_finals.items()}

        correctness_rows.append({
            "id": item_id,
            "gold": gold,
            "is_corrupted": is_corrupted,
            "corrupted_final": final_c if is_corrupted else None,
            "no_trace": corr_n,
            "uncorrupted_trace": corr_u,
            "corrupted_trace": corr_c,
        })

        # Agreement metrics
        agree_n, total_pairs = pairwise_agreement(list(no_trace_finals.values()))
        agree_u, _ = pairwise_agreement(list(uncorrupted_finals.values()))
        agree_c, _ = pairwise_agreement(list(corrupted_finals.values()))

        all_agree_n = len(set(no_trace_finals.values())) == 1
        all_agree_u = len(set(uncorrupted_finals.values())) == 1
        all_agree_c = len(set(corrupted_finals.values())) == 1

        aon_c_n = int(all_agree_n and next(iter(no_trace_finals.values())) == gold)
        aon_w_n = int(all_agree_n and next(iter(no_trace_finals.values())) != gold)
        aon_c_u = int(all_agree_u and next(iter(uncorrupted_finals.values())) == gold)
        aon_w_u = int(all_agree_u and next(iter(uncorrupted_finals.values())) != gold)
        aon_c_c = int(all_agree_c and next(iter(corrupted_finals.values())) == gold)
        aon_w_c = int(all_agree_c and next(iter(corrupted_finals.values())) != gold)

        # Corruption-specific metrics (only for corrupted items)
        follow_rates = {}
        plurality_ans, plurality_count = plurality_answer(list(corrupted_finals.values()))
        if is_corrupted:
            for t in targets:
                follow_rates[t.name] = int(corrupted_finals[t.name] == final_c)
            collapse_to_corrupted = int(plurality_ans == final_c)
            plurality_strength = plurality_count / len(targets)
        else:
            collapse_to_corrupted = None
            plurality_strength = None

        metrics_rows.append({
            "id": item_id,
            "is_corrupted": is_corrupted,
            "pairs_total": total_pairs,
            "agree_pairs_no_trace": agree_n,
            "agree_pairs_uncorrupted": agree_u,
            "agree_pairs_corrupted": agree_c,
            "all_agree_no_trace": all_agree_n,
            "all_agree_uncorrupted": all_agree_u,
            "all_agree_corrupted": all_agree_c,
            "aon_correct_no_trace": aon_c_n,
            "aon_wrong_no_trace": aon_w_n,
            "aon_correct_uncorrupted": aon_c_u,
            "aon_wrong_uncorrupted": aon_w_u,
            "aon_correct_corrupted": aon_c_c,
            "aon_wrong_corrupted": aon_w_c,
            "corruption_follow_rates": follow_rates if is_corrupted else None,
            "collapse_to_corrupted": collapse_to_corrupted,
            "plurality_strength": plurality_strength,
        })

        # Status
        print(f"n:{sum(corr_n.values())}/{len(targets)} u:{sum(corr_u.values())}/{len(targets)} c:{sum(corr_c.values())}/{len(targets)}")

    # Save judgments
    write_jsonl(outdir / "judgments" / "answers_no_trace.jsonl", answers_no_trace)
    write_jsonl(outdir / "judgments" / "answers_with_uncorrupted_trace.jsonl", answers_uncorrupted)
    write_jsonl(outdir / "judgments" / "answers_with_corrupted_trace.jsonl", answers_corrupted)
    write_jsonl(outdir / "judgments" / "correctness.jsonl", correctness_rows)
    write_jsonl(outdir / "metrics.jsonl", metrics_rows)

    # Also save injected traces
    injected_uncorrupted = [{"id": r["id"], "trace": r["trace"]} for r in proposer_uncorrupted]
    injected_corrupted = [{"id": r["id"], "trace": r["trace_corrupted"]} for r in proposer_corrupted]
    write_jsonl(outdir / "traces" / "injected_uncorrupted.jsonl", injected_uncorrupted)
    write_jsonl(outdir / "traces" / "injected_corrupted.jsonl", injected_corrupted)

    # Compute aggregates
    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    # Overall metrics
    agree_rate_n = mean([m["agree_pairs_no_trace"] / max(1, m["pairs_total"]) for m in metrics_rows])
    agree_rate_u = mean([m["agree_pairs_uncorrupted"] / max(1, m["pairs_total"]) for m in metrics_rows])
    agree_rate_c = mean([m["agree_pairs_corrupted"] / max(1, m["pairs_total"]) for m in metrics_rows])

    aon_c_n = sum(m["aon_correct_no_trace"] for m in metrics_rows)
    aon_w_n = sum(m["aon_wrong_no_trace"] for m in metrics_rows)
    aon_c_u = sum(m["aon_correct_uncorrupted"] for m in metrics_rows)
    aon_w_u = sum(m["aon_wrong_uncorrupted"] for m in metrics_rows)
    aon_c_c = sum(m["aon_correct_corrupted"] for m in metrics_rows)
    aon_w_c = sum(m["aon_wrong_corrupted"] for m in metrics_rows)

    # Per-target accuracy
    acc_n, acc_u, acc_c = {}, {}, {}
    for t in targets:
        acc_n[t.name] = mean([r["no_trace"][t.name] for r in correctness_rows])
        acc_u[t.name] = mean([r["uncorrupted_trace"][t.name] for r in correctness_rows])
        acc_c[t.name] = mean([r["corrupted_trace"][t.name] for r in correctness_rows])

    # Corruption-specific metrics (on corrupted items only)
    corrupted_metrics = [m for m in metrics_rows if m["is_corrupted"]]

    follow_rate_per_model = {}
    for t in targets:
        rates = [m["corruption_follow_rates"][t.name] for m in corrupted_metrics]
        follow_rate_per_model[t.name] = mean(rates)

    collapse_rate = mean([m["collapse_to_corrupted"] for m in corrupted_metrics])
    avg_plurality_strength = mean([m["plurality_strength"] for m in corrupted_metrics])

    # Agreement-on-wrong on corrupted items only
    aon_w_c_corrupted = sum(m["aon_wrong_corrupted"] for m in corrupted_metrics)
    aon_w_n_corrupted = sum(m["aon_wrong_no_trace"] for m in corrupted_metrics)

    git_info = get_git_info()

    summary = {
        "run_name": run_name,
        "dataset": str(dataset_path),
        "n_items": len(items),
        "n_corrupted": len(corrupted_ids),
        "git": git_info,
        "proposer": {"model": proposer_model, "temperature": 0.0},
        "targets": [{"name": t.name, "model": t.model} for t in targets],
        "agreement_rate": {
            "no_trace": agree_rate_n,
            "uncorrupted_trace": agree_rate_u,
            "corrupted_trace": agree_rate_c,
        },
        "agree_all_counts": {
            "no_trace": {"correct": aon_c_n, "wrong": aon_w_n},
            "uncorrupted_trace": {"correct": aon_c_u, "wrong": aon_w_u},
            "corrupted_trace": {"correct": aon_c_c, "wrong": aon_w_c},
        },
        "accuracy": {
            "no_trace": acc_n,
            "uncorrupted_trace": acc_u,
            "corrupted_trace": acc_c,
            "transfer_gain_uncorrupted": {k: acc_u[k] - acc_n[k] for k in acc_n},
            "transfer_gain_corrupted": {k: acc_c[k] - acc_n[k] for k in acc_n},
        },
        "corruption_metrics": {
            "follow_rate_per_model": follow_rate_per_model,
            "collapse_rate": collapse_rate,
            "avg_plurality_strength": avg_plurality_strength,
            "aon_wrong_corrupted_items_no_trace": aon_w_n_corrupted,
            "aon_wrong_corrupted_items_corrupted_trace": aon_w_c_corrupted,
        },
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Generate report
    report_lines = [
        f"# Report: {run_name}",
        "",
        f"**Dataset:** {dataset_path}",
        f"**Items:** {len(items)} (corrupted: {len(corrupted_ids)})",
        f"**Git:** {git_info['tag']} ({git_info['commit']})",
        "",
        "## Proposer",
        f"- Model: {proposer_model}",
        "",
        "## Agreement Rates",
        "",
        "| Condition | Pairwise Agreement |",
        "|-----------|-------------------|",
        f"| No trace | {agree_rate_n:.4f} |",
        f"| Uncorrupted trace | {agree_rate_u:.4f} |",
        f"| Corrupted trace | {agree_rate_c:.4f} |",
        "",
        "## Accuracy by Target Model",
        "",
        "| Model | No Trace | Uncorrupted | Corrupted | Gain (U) | Gain (C) |",
        "|-------|----------|-------------|-----------|----------|----------|",
    ]
    for m in acc_n:
        gain_u = acc_u[m] - acc_n[m]
        gain_c = acc_c[m] - acc_n[m]
        report_lines.append(f"| {m} | {acc_n[m]:.4f} | {acc_u[m]:.4f} | {acc_c[m]:.4f} | {gain_u:+.4f} | {gain_c:+.4f} |")

    report_lines.extend([
        "",
        "## All-Target Agreement Outcomes",
        "",
        "| Condition | Agree-on-Correct | Agree-on-Wrong |",
        "|-----------|------------------|----------------|",
        f"| No trace | {aon_c_n} | {aon_w_n} |",
        f"| Uncorrupted trace | {aon_c_u} | {aon_w_u} |",
        f"| Corrupted trace | {aon_c_c} | {aon_w_c} |",
        "",
        "## Corruption-Specific Metrics (K=15 corrupted items)",
        "",
        "### Corruption Follow Rate per Model",
        "",
        "| Model | Follow Rate |",
        "|-------|-------------|",
    ])
    for m, rate in sorted(follow_rate_per_model.items(), key=lambda x: -x[1]):
        report_lines.append(f"| {m} | {rate:.4f} |")

    report_lines.extend([
        "",
        f"### Collapse to Corrupted Answer: {collapse_rate:.4f}",
        f"### Average Plurality Strength: {avg_plurality_strength:.4f}",
        "",
        "### Agreement-on-Wrong (corrupted items only)",
        f"- No trace: {aon_w_n_corrupted}",
        f"- Corrupted trace: {aon_w_c_corrupted}",
        "",
        "## Summary Questions",
        "",
        f"1. **Does corrupted trace injection increase agreement-on-wrong on corrupted items?**",
        f"   {'Yes' if aon_w_c_corrupted > aon_w_n_corrupted else 'No'} ({aon_w_n_corrupted} → {aon_w_c_corrupted})",
        "",
        "2. **Which targets follow corrupted traces most/least?**",
        f"   - Most: {max(follow_rate_per_model.items(), key=lambda x: x[1])[0]} ({max(follow_rate_per_model.values()):.4f})",
        f"   - Least: {min(follow_rate_per_model.items(), key=lambda x: x[1])[0]} ({min(follow_rate_per_model.values()):.4f})",
        "",
        f"3. **Does corrupted trace injection increase pairwise agreement while decreasing accuracy?**",
    ])
    agreement_increased = agree_rate_c > agree_rate_n
    avg_acc_n = mean(list(acc_n.values()))
    avg_acc_c = mean(list(acc_c.values()))
    accuracy_decreased = avg_acc_c < avg_acc_n
    report_lines.append(f"   Agreement: {agree_rate_n:.4f} → {agree_rate_c:.4f} ({'increased' if agreement_increased else 'decreased'})")
    report_lines.append(f"   Accuracy: {avg_acc_n:.4f} → {avg_acc_c:.4f} ({'decreased' if accuracy_decreased else 'increased'})")
    report_lines.append(f"   **{'Yes' if agreement_increased and accuracy_decreased else 'No'}** - {'both conditions met' if agreement_increased and accuracy_decreased else 'conditions not met'}")

    (outdir / "report.md").write_text("\n".join(report_lines))

    # Manifest
    manifest = {
        "run_name": run_name,
        "created_utc": utc_ts_compact(),
        "git": git_info,
        "experiment": "trace-corruption-stress",
        "axis": "trace_quality",
        "conditions": ["no_trace", "uncorrupted_trace", "corrupted_trace"],
        "corruption_k": K_CORRUPT,
        "notes": "Stress test: plausible but wrong traces to induce consistent-wrong collapse.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Config
    (outdir / "config.json").write_text(json.dumps({
        "dataset": str(dataset_path),
        "proposer": {"model": proposer_model, "temperature": 0.0},
        "targets": [{"name": t.name, "model": t.model, "temperature": 0.0} for t in targets],
        "corruption": {"k": K_CORRUPT, "method": "arithmetic_flip_by_1"},
    }, indent=2))

    print()
    print("=" * 60)
    print(f"Run complete: {run_name}")
    print(f"Results: {outdir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
