#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["litellm>=1.0.0"]
# ///
"""
Causal Replication & Amplified Corruption (Experiment 006)

Replicates the single agreement-on-wrong event from Experiment 005 (causal family)
under amplified corruption: K=30 (all items) across 3 independent seeds.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import litellm

ROOT = Path(__file__).resolve().parents[1]

litellm.telemetry = False
litellm.set_verbose = False

SEEDS = [1, 2, 3]
DATASET = "exp005_causal.jsonl"


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
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(the answer is |answer: |option )", "", s)
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

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def extract_json(s: str) -> dict[str, Any]:
    """Extract JSON object from model response."""
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

    final_match = re.search(r'"final"\s*:\s*"([^"]*)"', s)
    if final_match:
        return {"final": final_match.group(1)}

    final_match = re.search(r'"final"\s*:\s*"([^"]*)', s)
    if final_match:
        return {"final": final_match.group(1)}

    return {"final": s.strip()[:500]}


def corrupt_semantic_trace(trace: str, final: str, item: dict, item_id: str, seed: int) -> tuple[str, str]:
    """
    Corrupt a trace with seed-dependent variation.

    The seed affects which alternative is chosen and how the trace is modified.
    All corruption remains minimal, plausible, fluent, and non-obvious.
    """
    # Seed-dependent hash: combining item_id and seed
    h = hashlib.sha256(f"{item_id}:seed={seed}".encode()).digest()
    alternatives = item.get("alternatives", [])

    if not alternatives:
        return trace, final + " (but uncertain)"

    # Seed determines which alternative
    alt_idx = h[0] % len(alternatives)
    corrupted_final = alternatives[alt_idx]

    if not trace:
        return f"After consideration, the answer is: {corrupted_final}", str(corrupted_final)

    corrupted_trace = trace.rstrip()

    # Seed determines corruption strategy
    strategy = h[1] % 3

    if strategy == 0:
        # Strategy 0: Replace conclusion after connective
        connectives = ["therefore", "thus", "so", "hence", "consequently", "this means"]
        for conn in connectives:
            if conn in corrupted_trace.lower():
                parts = re.split(f'({conn})', corrupted_trace, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) >= 2:
                    corrupted_trace = parts[0] + parts[1] + f" the answer should be: {corrupted_final}"
                    break
        else:
            corrupted_trace += f" Therefore, the answer is: {corrupted_final}"
    elif strategy == 1:
        # Strategy 1: Insert a misleading mid-trace assumption
        sentences = corrupted_trace.split(". ")
        if len(sentences) >= 2:
            insert_pos = h[2] % max(1, len(sentences) - 1)
            misleading = f"However, considering the context more carefully, this actually suggests {corrupted_final}"
            sentences.insert(insert_pos + 1, misleading)
            corrupted_trace = ". ".join(sentences)
        else:
            corrupted_trace += f" On reflection, this actually indicates: {corrupted_final}"
    else:
        # Strategy 2: Append a plausible reinterpretation
        corrupted_trace += (
            f" But reconsidering the causal chain, a more accurate conclusion is: {corrupted_final}"
        )

    return corrupted_trace, str(corrupted_final)


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
    counter = Counter(answers)
    if not counter:
        return "", 0
    return counter.most_common(1)[0]


def answers_match(a1: str, a2: str) -> bool:
    n1, n2 = normalize_answer(a1), normalize_answer(a2)
    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True
    if len(n1) == 1 and n1.isalpha() and n1 in n2:
        return True
    if len(n2) == 1 and n2.isalpha() and n2 in n1:
        return True
    return False


def run_seed(
    seed: int,
    items: list[dict],
    proposer_traces: list[dict],  # Shared uncorrupted traces
    proposer_model: str,
    targets: list[ModelSpec],
    outdir: Path,
) -> dict[str, Any]:
    """Run experiment for a single corruption seed."""

    seed_dir = outdir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # All items are corrupted
    corrupted_ids = set(str(it["id"]) for it in items)

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

    print(f"\n=== Seed {seed} (30 items, all corrupted) ===")

    # Build corrupted traces using seed
    uncorrupted_by_id = {r["id"]: r for r in proposer_traces}
    proposer_corrupted = []

    for it in items:
        item_id = str(it["id"])
        orig = uncorrupted_by_id[item_id]
        trace = orig["trace"]
        p_final = orig["final"]

        c_trace, c_final = corrupt_semantic_trace(trace, p_final, it, item_id, seed)
        proposer_corrupted.append({
            "id": item_id,
            "seed": seed,
            "trace_corrupted": c_trace,
            "final_corrupted": c_final,
            "original_trace": trace,
            "original_final": p_final,
        })

    write_jsonl(seed_dir / "traces_corrupted.jsonl", proposer_corrupted)
    corrupted_by_id = {r["id"]: r for r in proposer_corrupted}

    # Phase 2: Run targets
    answers_no_trace = []
    answers_uncorrupted = []
    answers_corrupted = []
    correctness_rows = []
    metrics_rows = []

    for idx, it in enumerate(items):
        item_id = str(it["id"])
        q = str(it["prompt"])
        gold = str(it["answer"])

        trace_u = uncorrupted_by_id[item_id]["trace"]
        trace_c = corrupted_by_id[item_id]["trace_corrupted"]
        final_c = corrupted_by_id[item_id]["final_corrupted"]

        print(f"  [{idx+1}/{len(items)}] {item_id} targets...", end=" ", flush=True)

        no_trace_finals: dict[str, str] = {}
        uncorrupted_finals: dict[str, str] = {}
        corrupted_finals: dict[str, str] = {}

        for t in targets:
            # No trace
            tnt = target_no_trace_t.replace("{{PROMPT}}", q)
            raw_n = call_model(t.model, tnt, temperature=0.0)
            obj_n = extract_json(raw_n)
            fin_n = str(obj_n.get("final", ""))
            no_trace_finals[t.name] = fin_n
            answers_no_trace.append({"id": item_id, "model": t.name, "final": fin_n, "raw": raw_n})

            # Uncorrupted trace
            twu = target_with_trace_t.replace("{{PROMPT}}", q).replace("{{TRACE}}", trace_u)
            raw_u = call_model(t.model, twu, temperature=0.0)
            obj_u = extract_json(raw_u)
            fin_u = str(obj_u.get("final", ""))
            uncorrupted_finals[t.name] = fin_u
            answers_uncorrupted.append({"id": item_id, "model": t.name, "final": fin_u, "raw": raw_u})

            # Corrupted trace (seed-specific)
            twc = target_with_trace_t.replace("{{PROMPT}}", q).replace("{{TRACE}}", trace_c)
            raw_c = call_model(t.model, twc, temperature=0.0)
            obj_c = extract_json(raw_c)
            fin_c = str(obj_c.get("final", ""))
            corrupted_finals[t.name] = fin_c
            answers_corrupted.append({"id": item_id, "model": t.name, "final": fin_c, "raw": raw_c})

        # Correctness
        corr_n = {m: answers_match(a, gold) for m, a in no_trace_finals.items()}
        corr_u = {m: answers_match(a, gold) for m, a in uncorrupted_finals.items()}
        corr_c = {m: answers_match(a, gold) for m, a in corrupted_finals.items()}

        correctness_rows.append({
            "id": item_id,
            "gold": gold,
            "corrupted_final": final_c,
            "no_trace": corr_n,
            "uncorrupted_trace": corr_u,
            "corrupted_trace": corr_c,
        })

        # Agreement metrics
        norm_n = [normalize_answer(a) for a in no_trace_finals.values()]
        norm_u = [normalize_answer(a) for a in uncorrupted_finals.values()]
        norm_c = [normalize_answer(a) for a in corrupted_finals.values()]

        agree_n, total_pairs = pairwise_agreement(norm_n)
        agree_u, _ = pairwise_agreement(norm_u)
        agree_c, _ = pairwise_agreement(norm_c)

        all_agree_n = len(set(norm_n)) == 1
        all_agree_u = len(set(norm_u)) == 1
        all_agree_c = len(set(norm_c)) == 1

        aon_c_n = int(all_agree_n and answers_match(norm_n[0], gold))
        aon_w_n = int(all_agree_n and not answers_match(norm_n[0], gold))
        aon_c_u = int(all_agree_u and answers_match(norm_u[0], gold))
        aon_w_u = int(all_agree_u and not answers_match(norm_u[0], gold))
        aon_c_c = int(all_agree_c and answers_match(norm_c[0], gold))
        aon_w_c = int(all_agree_c and not answers_match(norm_c[0], gold))

        # Plurality
        plur_n, plur_cnt_n = plurality_answer(norm_n)
        plur_u, plur_cnt_u = plurality_answer(norm_u)
        plur_c, plur_cnt_c = plurality_answer(norm_c)

        plur_wrong_n = int(not answers_match(plur_n, gold))
        plur_wrong_u = int(not answers_match(plur_u, gold))
        plur_wrong_c = int(not answers_match(plur_c, gold))

        # Corruption follow
        follow_rates = {}
        for t in targets:
            follow_rates[t.name] = int(answers_match(corrupted_finals[t.name], final_c))
        collapse_to_corrupted = int(answers_match(plur_c, final_c))

        metrics_rows.append({
            "id": item_id,
            "seed": seed,
            "pairs_total": total_pairs,
            "agree_pairs_no_trace": agree_n,
            "agree_pairs_uncorrupted": agree_u,
            "agree_pairs_corrupted": agree_c,
            "aon_correct_no_trace": aon_c_n,
            "aon_wrong_no_trace": aon_w_n,
            "aon_correct_uncorrupted": aon_c_u,
            "aon_wrong_uncorrupted": aon_w_u,
            "aon_correct_corrupted": aon_c_c,
            "aon_wrong_corrupted": aon_w_c,
            "plurality_wrong_no_trace": plur_wrong_n,
            "plurality_wrong_uncorrupted": plur_wrong_u,
            "plurality_wrong_corrupted": plur_wrong_c,
            "plurality_strength_no_trace": plur_cnt_n / len(targets),
            "plurality_strength_uncorrupted": plur_cnt_u / len(targets),
            "plurality_strength_corrupted": plur_cnt_c / len(targets) if plur_cnt_c else 0,
            "corruption_follow_rates": follow_rates,
            "collapse_to_corrupted": collapse_to_corrupted,
        })

        print(f"n:{sum(corr_n.values())}/{len(targets)} u:{sum(corr_u.values())}/{len(targets)} c:{sum(corr_c.values())}/{len(targets)}")

    # Save judgments
    write_jsonl(seed_dir / "answers_no_trace.jsonl", answers_no_trace)
    write_jsonl(seed_dir / "answers_with_uncorrupted_trace.jsonl", answers_uncorrupted)
    write_jsonl(seed_dir / "answers_with_corrupted_trace.jsonl", answers_corrupted)
    write_jsonl(seed_dir / "correctness.jsonl", correctness_rows)
    write_jsonl(seed_dir / "metrics.jsonl", metrics_rows)

    # Compute aggregates
    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    agree_rate_n = mean([m["agree_pairs_no_trace"] / max(1, m["pairs_total"]) for m in metrics_rows])
    agree_rate_u = mean([m["agree_pairs_uncorrupted"] / max(1, m["pairs_total"]) for m in metrics_rows])
    agree_rate_c = mean([m["agree_pairs_corrupted"] / max(1, m["pairs_total"]) for m in metrics_rows])

    aon_c_n_total = sum(m["aon_correct_no_trace"] for m in metrics_rows)
    aon_w_n_total = sum(m["aon_wrong_no_trace"] for m in metrics_rows)
    aon_c_u_total = sum(m["aon_correct_uncorrupted"] for m in metrics_rows)
    aon_w_u_total = sum(m["aon_wrong_uncorrupted"] for m in metrics_rows)
    aon_c_c_total = sum(m["aon_correct_corrupted"] for m in metrics_rows)
    aon_w_c_total = sum(m["aon_wrong_corrupted"] for m in metrics_rows)

    plur_wrong_n_total = sum(m["plurality_wrong_no_trace"] for m in metrics_rows)
    plur_wrong_u_total = sum(m["plurality_wrong_uncorrupted"] for m in metrics_rows)
    plur_wrong_c_total = sum(m["plurality_wrong_corrupted"] for m in metrics_rows)

    acc_n, acc_u, acc_c = {}, {}, {}
    for t in targets:
        acc_n[t.name] = mean([r["no_trace"][t.name] for r in correctness_rows])
        acc_u[t.name] = mean([r["uncorrupted_trace"][t.name] for r in correctness_rows])
        acc_c[t.name] = mean([r["corrupted_trace"][t.name] for r in correctness_rows])

    follow_rate_per_model = {}
    for t in targets:
        rates = [m["corruption_follow_rates"][t.name] for m in metrics_rows]
        follow_rate_per_model[t.name] = mean(rates)

    total_follow_events = sum(
        sum(m["corruption_follow_rates"].values()) for m in metrics_rows
    )

    summary = {
        "seed": seed,
        "n_items": len(items),
        "n_corrupted": len(items),  # All corrupted
        "agreement_rate": {
            "no_trace": agree_rate_n,
            "uncorrupted_trace": agree_rate_u,
            "corrupted_trace": agree_rate_c,
        },
        "agree_all_counts": {
            "no_trace": {"correct": aon_c_n_total, "wrong": aon_w_n_total},
            "uncorrupted_trace": {"correct": aon_c_u_total, "wrong": aon_w_u_total},
            "corrupted_trace": {"correct": aon_c_c_total, "wrong": aon_w_c_total},
        },
        "plurality_wrong": {
            "no_trace": plur_wrong_n_total,
            "uncorrupted_trace": plur_wrong_u_total,
            "corrupted_trace": plur_wrong_c_total,
        },
        "accuracy": {
            "no_trace": acc_n,
            "uncorrupted_trace": acc_u,
            "corrupted_trace": acc_c,
        },
        "corruption_metrics": {
            "follow_rate_per_model": follow_rate_per_model,
            "total_follow_events": total_follow_events,
        },
    }

    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: run_causal_replication.py <run_name> <proposer_model> [<target_spec>...]\n\n"
            "Target spec format: <model>:<name>"
        )
        return 2

    run_name = sys.argv[1]
    proposer_model = sys.argv[2]

    targets: list[ModelSpec] = []
    for spec in sys.argv[3:]:
        parts = spec.split(":")
        model = parts[0]
        name = parts[1] if len(parts) >= 2 else model
        targets.append(ModelSpec(name=name, model=model))

    if not targets:
        raise ValueError("At least one target required.")

    outdir = ROOT / "eval" / "results" / run_name
    resuming = outdir.exists()
    outdir.mkdir(parents=True, exist_ok=True)

    # Save manifests
    (outdir / "manifests").mkdir(parents=True, exist_ok=True)
    (outdir / "manifests" / "target_models_list.txt").write_text(
        "\n".join([f"{t.name}: {t.model}" for t in targets])
    )
    (outdir / "manifests" / "claude_cli_version.txt").write_text(
        subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    )

    # Load dataset
    dataset_path = ROOT / "eval" / "datasets" / DATASET
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}")
        return 1
    items = read_jsonl(dataset_path)

    print(f"Proposer: {proposer_model}")
    print(f"Targets: {[t.name for t in targets]}")
    print(f"Dataset: {DATASET} ({len(items)} items)")
    print(f"Seeds: {SEEDS}")
    print(f"Corruption: ALL items (K=30)")
    if resuming:
        print("RESUMING from existing run directory")

    # Phase 1: Generate proposer traces (shared across seeds)
    proposer_traces_path = outdir / "proposer_traces.jsonl"
    if resuming and proposer_traces_path.exists():
        print("\n=== Loading existing proposer traces ===")
        proposer_traces = read_jsonl(proposer_traces_path)
        print(f"Loaded {len(proposer_traces)} traces")
    else:
        proposer_t = (
            "You will solve the task below.\n\n"
            "Return ONLY valid JSON with keys: \"trace\", \"final\".\n"
            "- \"trace\" is your reasoning trace.\n"
            "- \"final\" is your final answer.\n\n"
            "Task:\n{{PROMPT}}\n"
        )

        print("\n=== Generating proposer traces (shared) ===")
        proposer_traces = []
        for idx, it in enumerate(items):
            item_id = str(it["id"])
            q = str(it["prompt"])
            print(f"[{idx+1}/{len(items)}] {item_id} proposer...", end=" ", flush=True)

            p_prompt = proposer_t.replace("{{PROMPT}}", q)
            p_raw = call_model(proposer_model, p_prompt, temperature=0.0)
            p_obj = extract_json(p_raw)
            trace = str(p_obj.get("trace", ""))
            p_final = str(p_obj.get("final", ""))

            proposer_traces.append({
                "id": item_id,
                "trace": trace,
                "final": p_final,
                "raw": p_raw,
            })
            print("ok")

        write_jsonl(proposer_traces_path, proposer_traces)

    # Phase 2: Run each seed
    seed_summaries = {}
    for seed in SEEDS:
        # Save corrupted ids (all 30)
        all_ids = sorted([str(it["id"]) for it in items])
        (outdir / "manifests" / f"corrupted_ids_seed_{seed}.json").write_text(
            json.dumps(all_ids, indent=2)
        )

        seed_summary_path = outdir / f"seed_{seed}" / "summary.json"
        if resuming and seed_summary_path.exists():
            print(f"\n=== Seed {seed}: ALREADY COMPLETE, skipping ===")
            seed_summaries[seed] = json.loads(seed_summary_path.read_text())
            continue

        # Clean up partial seed dir if resuming
        seed_dir = outdir / f"seed_{seed}"
        if seed_dir.exists():
            import shutil
            shutil.rmtree(seed_dir)

        summary = run_seed(seed, items, proposer_traces, proposer_model, targets, outdir)
        seed_summaries[seed] = summary

    # Aggregate replication metrics
    aow_values = [s["agree_all_counts"]["corrupted_trace"]["wrong"] for s in seed_summaries.values()]
    aow_mean = sum(aow_values) / len(aow_values)
    aow_std = math.sqrt(sum((v - aow_mean) ** 2 for v in aow_values) / len(aow_values))

    total_follow = sum(s["corruption_metrics"]["total_follow_events"] for s in seed_summaries.values())
    total_plur_wrong = sum(s["plurality_wrong"]["corrupted_trace"] for s in seed_summaries.values())

    replication = {
        "agreement_on_wrong_per_seed": {str(s): v for s, v in zip(SEEDS, aow_values)},
        "agreement_on_wrong_mean": aow_mean,
        "agreement_on_wrong_std": aow_std,
        "total_corruption_follow_events": total_follow,
        "total_plurality_wrong_events": total_plur_wrong,
    }

    # Global summary
    git_info = get_git_info()
    global_summary = {
        "run_name": run_name,
        "git": git_info,
        "proposer": {"model": proposer_model, "temperature": 0.0},
        "targets": [{"name": t.name, "model": t.model} for t in targets],
        "dataset": DATASET,
        "seeds": SEEDS,
        "corruption_k": len(items),
        "per_seed": {str(s): v for s, v in seed_summaries.items()},
        "replication": replication,
    }
    (outdir / "summary.json").write_text(json.dumps(global_summary, indent=2))

    # Config
    (outdir / "config.json").write_text(json.dumps({
        "proposer": {"model": proposer_model, "temperature": 0.0},
        "targets": [{"name": t.name, "model": t.model, "temperature": 0.0} for t in targets],
        "dataset": DATASET,
        "seeds": SEEDS,
        "corruption_k": len(items),
        "corruption_protocol": "All 30 items corrupted per seed. Seed affects alternative selection and trace modification strategy.",
    }, indent=2))

    # Manifest
    (outdir / "manifest.json").write_text(json.dumps({
        "run_name": run_name,
        "created_utc": utc_ts_compact(),
        "git": git_info,
        "experiment": "causal-replication-amplified",
        "axis": "corruption_coverage_seed_variation",
        "dataset": DATASET,
        "seeds": SEEDS,
        "conditions": ["no_trace", "uncorrupted_trace", "corrupted_trace"],
    }, indent=2))

    # Report
    report_lines = [
        f"# Report: {run_name}",
        "",
        f"**Git:** {git_info['tag']} ({git_info['commit']})",
        f"**Proposer:** {proposer_model}",
        f"**Dataset:** {DATASET} (30 items)",
        f"**Corruption:** K=30 (all items), 3 seeds",
        "",
        "## Per-Seed Results",
        "",
    ]

    for seed in SEEDS:
        s = seed_summaries[seed]
        acc_n = s["accuracy"]["no_trace"]
        acc_u = s["accuracy"]["uncorrupted_trace"]
        acc_c = s["accuracy"]["corrupted_trace"]
        avg_n = sum(acc_n.values()) / len(acc_n)
        avg_u = sum(acc_u.values()) / len(acc_u)
        avg_c = sum(acc_c.values()) / len(acc_c)

        report_lines.extend([
            f"### Seed {seed}",
            "",
            "| Condition | Agreement | Avg Accuracy | AoW | Plurality-Wrong |",
            "|-----------|-----------|-------------|-----|-----------------|",
            f"| No trace | {s['agreement_rate']['no_trace']:.4f} | {avg_n:.4f} | {s['agree_all_counts']['no_trace']['wrong']} | {s['plurality_wrong']['no_trace']} |",
            f"| Uncorrupted | {s['agreement_rate']['uncorrupted_trace']:.4f} | {avg_u:.4f} | {s['agree_all_counts']['uncorrupted_trace']['wrong']} | {s['plurality_wrong']['uncorrupted_trace']} |",
            f"| Corrupted | {s['agreement_rate']['corrupted_trace']:.4f} | {avg_c:.4f} | {s['agree_all_counts']['corrupted_trace']['wrong']} | {s['plurality_wrong']['corrupted_trace']} |",
            "",
            "**Per-model accuracy (corrupted):**",
        ])
        for t in targets:
            report_lines.append(f"- {t.name}: {acc_c[t.name]:.3f}")
        report_lines.extend([
            "",
            f"**Corruption follow rate:** {s['corruption_metrics']['follow_rate_per_model']}",
            f"**Total follow events:** {s['corruption_metrics']['total_follow_events']}",
            "",
        ])

    # Replication summary
    report_lines.extend([
        "## Replication Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| AoW per seed | {aow_values} |",
        f"| AoW mean | {aow_mean:.2f} |",
        f"| AoW std | {aow_std:.2f} |",
        f"| Total follow events | {total_follow} |",
        f"| Total plurality-wrong | {total_plur_wrong} |",
        "",
        "## Answers",
        "",
        "### 1. Does agreement-on-wrong replicate across seeds?",
        "",
    ])

    seeds_with_aow = [s for s, v in zip(SEEDS, aow_values) if v > 0]
    if len(seeds_with_aow) >= 2:
        report_lines.append(f"**Yes.** Agreement-on-wrong observed in seeds: {seeds_with_aow}")
    elif len(seeds_with_aow) == 1:
        report_lines.append(f"**Partial.** Agreement-on-wrong observed only in seed {seeds_with_aow[0]}.")
    else:
        report_lines.append("**No.** Agreement-on-wrong did not replicate in any seed.")

    report_lines.extend([
        "",
        "### 2. Does corruption-follow scale when K=30?",
        "",
        f"Total corruption-follow events across all seeds: {total_follow}",
        f"Expected if random: ~{len(items) * len(SEEDS) * len(targets) // len(items)} (baseline agreement rate)",
        "",
    ])

    if total_follow > 0:
        report_lines.append(f"**Yes**, {total_follow} follow events observed.")
    else:
        report_lines.append("**No**, zero corruption-follow events.")

    report_lines.extend([
        "",
        "### 3. Does plurality-wrong emerge under amplified corruption?",
        "",
        f"Total plurality-wrong under corrupted traces: {total_plur_wrong}",
        "",
    ])

    plur_wrong_no_trace = sum(s["plurality_wrong"]["no_trace"] for s in seed_summaries.values())
    if total_plur_wrong > plur_wrong_no_trace:
        report_lines.append(f"**Yes**, plurality-wrong increased from {plur_wrong_no_trace} (no trace) to {total_plur_wrong} (corrupted).")
    else:
        report_lines.append(f"**No**, plurality-wrong did not increase ({plur_wrong_no_trace} no trace vs {total_plur_wrong} corrupted).")

    report_lines.extend([
        "",
        "### 4. Which target is most sensitive to corruption?",
        "",
    ])

    # Aggregate follow rates across seeds
    agg_follow = {t.name: 0.0 for t in targets}
    for s in seed_summaries.values():
        for t in targets:
            agg_follow[t.name] += s["corruption_metrics"]["follow_rate_per_model"].get(t.name, 0.0)
    for t in targets:
        agg_follow[t.name] /= len(SEEDS)

    most_sensitive = max(agg_follow.items(), key=lambda x: x[1])
    least_sensitive = min(agg_follow.items(), key=lambda x: x[1])
    report_lines.extend([
        f"Mean corruption-follow rate per model (across seeds):",
    ])
    for t in targets:
        report_lines.append(f"- {t.name}: {agg_follow[t.name]:.4f}")
    report_lines.extend([
        "",
        f"**Most sensitive:** {most_sensitive[0]} ({most_sensitive[1]:.4f})",
        f"**Least sensitive:** {least_sensitive[0]} ({least_sensitive[1]:.4f})",
    ])

    (outdir / "report.md").write_text("\n".join(report_lines))

    print()
    print("=" * 60)
    print(f"Run complete: {run_name}")
    print(f"Results: {outdir}")
    print(f"Replication: AoW = {aow_values}, mean={aow_mean:.2f}, std={aow_std:.2f}")
    print(f"Total follow events: {total_follow}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
