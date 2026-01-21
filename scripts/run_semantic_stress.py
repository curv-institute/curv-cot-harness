#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = ["litellm>=1.0.0"]
# ///
"""
Semantic Ambiguity Stress Test (Experiment 005)

Tests whether trace injection induces agreement-on-wrong or collapse in tasks
where correctness is semantic, ambiguous, or non-computable.

Runs 4 task families: commonsense, causal, distractor, explanation
"""

from __future__ import annotations

import hashlib
import json
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

FAMILIES = ["commonsense", "causal", "distractor", "explanation"]
K_CORRUPT = 8  # Number of items to corrupt per family (out of 30)


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
    # Remove common prefixes for explanation answers
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

    if model.lower().startswith("gpt-") or "codex" in model.lower():
        cmd = os.environ.get("CODEX_CMD", "/home/jwm/.bun/bin/codex")
        if model.lower() == "gpt-5-codex":
            args = [cmd, "exec", "-"]
        else:
            args = [cmd, "exec", "-m", model, "-"]

        max_retries = 10
        for attempt in range(max_retries):
            proc = subprocess.run(args, input=prompt, text=True, capture_output=True, check=False)
            if proc.returncode == 0:
                return proc.stdout.strip()

            # Check for rate limit error
            stderr = proc.stderr
            if "usage_limit_reached" in stderr or "429" in stderr:
                # Try to extract reset time
                match = re.search(r'"resets_in_seconds":(\d+)', stderr)
                if match:
                    wait_secs = int(match.group(1)) + 5  # Add 5s buffer
                else:
                    wait_secs = 120  # Default 2 min wait
                print(f"\n[Rate limit hit, waiting {wait_secs}s for reset...]", file=sys.stderr)
                time.sleep(wait_secs)
                continue

            # Not a rate limit error - fail
            raise RuntimeError(f"Codex CLI failed: {proc.stderr}")

        raise RuntimeError(f"Codex CLI failed after {max_retries} rate limit retries")

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    return response.choices[0].message.content.strip()


def extract_json(s: str) -> dict[str, Any]:
    """Extract JSON object from model response, handling various formats and truncation."""
    # Try direct parse first
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
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

    # Try to extract just the "final" field if JSON is truncated
    final_match = re.search(r'"final"\s*:\s*"([^"]*)"', s)
    if final_match:
        return {"final": final_match.group(1)}

    # Try to extract from truncated JSON
    final_match = re.search(r'"final"\s*:\s*"([^"]*)', s)
    if final_match:
        return {"final": final_match.group(1)}

    # Last resort: return the whole response as final
    return {"final": s.strip()[:500]}


def select_corrupted_ids(item_ids: list[str], k: int = K_CORRUPT) -> list[str]:
    """Select k items for corruption using SHA256 hash (deterministic)."""
    hashed = [(hashlib.sha256(id_.encode()).hexdigest(), id_) for id_ in item_ids]
    hashed.sort(key=lambda x: x[0])
    return [id_ for _, id_ in hashed[:k]]


def corrupt_semantic_trace(trace: str, final: str, item: dict, item_id: str) -> tuple[str, str]:
    """
    Corrupt a trace by introducing a semantic error.

    For semantic tasks, we:
    1. Keep the trace structure but change the conclusion
    2. Use one of the plausible alternatives as the corrupted answer
    3. Modify the trace reasoning to support that alternative
    """
    # Use hash to select which alternative to use
    h = hashlib.sha256(item_id.encode()).digest()
    alternatives = item.get("alternatives", [])

    if not alternatives:
        # No alternatives, just modify the final slightly
        return trace, final + " (but uncertain)"

    # Select an alternative deterministically
    alt_idx = h[0] % len(alternatives)
    corrupted_final = alternatives[alt_idx]

    # For the trace, we'll append a misleading conclusion
    # This simulates a semantic error - same evidence, wrong conclusion
    if trace:
        # Find the last sentence and modify it
        corrupted_trace = trace.rstrip()
        if "therefore" in corrupted_trace.lower():
            # Replace conclusion
            parts = re.split(r'(therefore|thus|so|hence)', corrupted_trace, flags=re.IGNORECASE)
            if len(parts) >= 2:
                corrupted_trace = parts[0] + parts[1] + f" the answer should be: {corrupted_final}"
            else:
                corrupted_trace += f" However, on reflection, the answer is: {corrupted_final}"
        else:
            corrupted_trace += f" Therefore, the answer is: {corrupted_final}"
    else:
        corrupted_trace = f"After consideration, the answer is: {corrupted_final}"

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
    """Return (most common answer, count)."""
    counter = Counter(answers)
    if not counter:
        return "", 0
    return counter.most_common(1)[0]


def answers_match(a1: str, a2: str) -> bool:
    """Check if two answers match (normalized)."""
    n1, n2 = normalize_answer(a1), normalize_answer(a2)
    # Direct match
    if n1 == n2:
        return True
    # Check if one contains the other (for partial matches)
    if n1 in n2 or n2 in n1:
        return True
    # For choice answers, just check the letter
    if len(n1) == 1 and n1.isalpha() and n1 in n2:
        return True
    if len(n2) == 1 and n2.isalpha() and n2 in n1:
        return True
    return False


def run_family(
    family: str,
    items: list[dict],
    proposer_model: str,
    targets: list[ModelSpec],
    outdir: Path,
) -> dict[str, Any]:
    """Run experiment on a single family."""

    family_dir = outdir / family
    family_dir.mkdir(parents=True, exist_ok=True)

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
    (family_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (family_dir / "prompts" / "proposer.txt").write_text(proposer_t)
    (family_dir / "prompts" / "target_with_trace.txt").write_text(target_with_trace_t)
    (family_dir / "prompts" / "target_no_trace.txt").write_text(target_no_trace_t)

    # Select items for corruption
    item_ids = [str(it["id"]) for it in items]
    corrupted_ids = set(select_corrupted_ids(item_ids, K_CORRUPT))

    print(f"\n=== Family: {family} ({len(items)} items, {len(corrupted_ids)} corrupted) ===")

    # Phase 1: Generate proposer traces
    proposer_uncorrupted = []
    proposer_corrupted = []

    for idx, it in enumerate(items):
        item_id = str(it["id"])
        q = str(it["prompt"])

        print(f"[{idx+1}/{len(items)}] {item_id} proposer...", end=" ", flush=True)

        p_prompt = proposer_t.replace("{{PROMPT}}", q)
        p_raw = call_model(proposer_model, p_prompt, temperature=0.0)
        p_obj = extract_json(p_raw)
        trace = str(p_obj.get("trace", ""))
        p_final = str(p_obj.get("final", ""))

        proposer_uncorrupted.append({
            "id": item_id,
            "trace": trace,
            "final": p_final,
            "raw": p_raw,
        })

        if item_id in corrupted_ids:
            c_trace, c_final = corrupt_semantic_trace(trace, p_final, it, item_id)
            proposer_corrupted.append({
                "id": item_id,
                "trace_corrupted": c_trace,
                "final_corrupted": c_final,
                "original_trace": trace,
                "original_final": p_final,
            })
            print(f"corrupted")
        else:
            proposer_corrupted.append({
                "id": item_id,
                "trace_corrupted": trace,
                "final_corrupted": p_final,
                "original_trace": trace,
                "original_final": p_final,
            })
            print("ok")

    write_jsonl(family_dir / "traces" / "proposer_uncorrupted.jsonl", proposer_uncorrupted)
    write_jsonl(family_dir / "traces" / "proposer_corrupted.jsonl", proposer_corrupted)

    uncorrupted_by_id = {r["id"]: r for r in proposer_uncorrupted}
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
        is_corrupted = item_id in corrupted_ids

        trace_u = uncorrupted_by_id[item_id]["trace"]
        trace_c = corrupted_by_id[item_id]["trace_corrupted"]
        final_c = corrupted_by_id[item_id]["final_corrupted"]

        print(f"[{idx+1}/{len(items)}] {item_id}{'*' if is_corrupted else ''} targets...", end=" ", flush=True)

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

            # Corrupted trace
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
            "is_corrupted": is_corrupted,
            "corrupted_final": final_c if is_corrupted else None,
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

        gold_n = normalize_answer(gold)

        all_agree_n = len(set(norm_n)) == 1
        all_agree_u = len(set(norm_u)) == 1
        all_agree_c = len(set(norm_c)) == 1

        aon_c_n = int(all_agree_n and answers_match(norm_n[0], gold))
        aon_w_n = int(all_agree_n and not answers_match(norm_n[0], gold))
        aon_c_u = int(all_agree_u and answers_match(norm_u[0], gold))
        aon_w_u = int(all_agree_u and not answers_match(norm_u[0], gold))
        aon_c_c = int(all_agree_c and answers_match(norm_c[0], gold))
        aon_w_c = int(all_agree_c and not answers_match(norm_c[0], gold))

        # Plurality metrics
        plur_n, plur_cnt_n = plurality_answer(norm_n)
        plur_u, plur_cnt_u = plurality_answer(norm_u)
        plur_c, plur_cnt_c = plurality_answer(norm_c)

        plur_wrong_n = int(not answers_match(plur_n, gold))
        plur_wrong_u = int(not answers_match(plur_u, gold))
        plur_wrong_c = int(not answers_match(plur_c, gold))

        # Corruption-specific metrics
        follow_rates = {}
        if is_corrupted:
            for t in targets:
                follow_rates[t.name] = int(answers_match(corrupted_finals[t.name], final_c))
            collapse_to_corrupted = int(answers_match(plur_c, final_c))
            plurality_strength_c = plur_cnt_c / len(targets)
        else:
            collapse_to_corrupted = None
            plurality_strength_c = None

        # Answer diversity
        diversity_n = len(set(norm_n))
        diversity_u = len(set(norm_u))
        diversity_c = len(set(norm_c))

        metrics_rows.append({
            "id": item_id,
            "is_corrupted": is_corrupted,
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
            "diversity_no_trace": diversity_n,
            "diversity_uncorrupted": diversity_u,
            "diversity_corrupted": diversity_c,
            "corruption_follow_rates": follow_rates if is_corrupted else None,
            "collapse_to_corrupted": collapse_to_corrupted,
        })

        print(f"n:{sum(corr_n.values())}/{len(targets)} u:{sum(corr_u.values())}/{len(targets)} c:{sum(corr_c.values())}/{len(targets)}")

    # Save judgments
    write_jsonl(family_dir / "judgments" / "answers_no_trace.jsonl", answers_no_trace)
    write_jsonl(family_dir / "judgments" / "answers_with_uncorrupted_trace.jsonl", answers_uncorrupted)
    write_jsonl(family_dir / "judgments" / "answers_with_corrupted_trace.jsonl", answers_corrupted)
    write_jsonl(family_dir / "judgments" / "correctness.jsonl", correctness_rows)
    write_jsonl(family_dir / "metrics.jsonl", metrics_rows)

    # Compute aggregates
    def mean(xs: list[float]) -> float:
        return sum(xs) / max(1, len(xs))

    agree_rate_n = mean([m["agree_pairs_no_trace"] / max(1, m["pairs_total"]) for m in metrics_rows])
    agree_rate_u = mean([m["agree_pairs_uncorrupted"] / max(1, m["pairs_total"]) for m in metrics_rows])
    agree_rate_c = mean([m["agree_pairs_corrupted"] / max(1, m["pairs_total"]) for m in metrics_rows])

    aon_c_n = sum(m["aon_correct_no_trace"] for m in metrics_rows)
    aon_w_n = sum(m["aon_wrong_no_trace"] for m in metrics_rows)
    aon_c_u = sum(m["aon_correct_uncorrupted"] for m in metrics_rows)
    aon_w_u = sum(m["aon_wrong_uncorrupted"] for m in metrics_rows)
    aon_c_c = sum(m["aon_correct_corrupted"] for m in metrics_rows)
    aon_w_c = sum(m["aon_wrong_corrupted"] for m in metrics_rows)

    plur_wrong_n = sum(m["plurality_wrong_no_trace"] for m in metrics_rows)
    plur_wrong_u = sum(m["plurality_wrong_uncorrupted"] for m in metrics_rows)
    plur_wrong_c = sum(m["plurality_wrong_corrupted"] for m in metrics_rows)

    # Per-target accuracy
    acc_n, acc_u, acc_c = {}, {}, {}
    for t in targets:
        acc_n[t.name] = mean([r["no_trace"][t.name] for r in correctness_rows])
        acc_u[t.name] = mean([r["uncorrupted_trace"][t.name] for r in correctness_rows])
        acc_c[t.name] = mean([r["corrupted_trace"][t.name] for r in correctness_rows])

    # Corruption-specific aggregates
    corrupted_metrics = [m for m in metrics_rows if m["is_corrupted"]]

    follow_rate_per_model = {}
    if corrupted_metrics:
        for t in targets:
            rates = [m["corruption_follow_rates"][t.name] for m in corrupted_metrics]
            follow_rate_per_model[t.name] = mean(rates)
        collapse_rate = mean([m["collapse_to_corrupted"] for m in corrupted_metrics])
    else:
        collapse_rate = 0.0

    # Disagreement suppression (trace reduces diversity while decreasing accuracy)
    suppression_count = 0
    for m, c in zip(metrics_rows, correctness_rows):
        if m["diversity_uncorrupted"] < m["diversity_no_trace"]:
            acc_change = sum(c["uncorrupted_trace"].values()) - sum(c["no_trace"].values())
            if acc_change < 0:
                suppression_count += 1

    summary = {
        "family": family,
        "n_items": len(items),
        "n_corrupted": len(corrupted_ids),
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
        "plurality_wrong": {
            "no_trace": plur_wrong_n,
            "uncorrupted_trace": plur_wrong_u,
            "corrupted_trace": plur_wrong_c,
        },
        "accuracy": {
            "no_trace": acc_n,
            "uncorrupted_trace": acc_u,
            "corrupted_trace": acc_c,
        },
        "corruption_metrics": {
            "follow_rate_per_model": follow_rate_per_model,
            "collapse_rate": collapse_rate,
        },
        "disagreement_suppression_count": suppression_count,
    }

    (family_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Save corrupted IDs
    (outdir / "manifests" / f"corrupted_ids_{family}.json").write_text(
        json.dumps(sorted(corrupted_ids), indent=2)
    )

    return summary


def main() -> int:
    if len(sys.argv) < 4:
        print(
            "Usage: run_semantic_stress.py <run_name> <proposer_model> [<target_spec>...]\n\n"
            "Runs on all 4 semantic families: commonsense, causal, distractor, explanation\n"
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
    outdir.mkdir(parents=True, exist_ok=False)

    # Save manifests
    (outdir / "manifests").mkdir(parents=True, exist_ok=True)
    (outdir / "manifests" / "target_models_list.txt").write_text(
        "\n".join([f"{t.name}: {t.model}" for t in targets])
    )
    subprocess.run(["claude", "--version"], capture_output=True, text=True, check=False)
    (outdir / "manifests" / "claude_cli_version.txt").write_text(
        subprocess.run(["claude", "--version"], capture_output=True, text=True).stdout.strip()
    )
    (outdir / "manifests" / "codex_cli_version.txt").write_text(
        subprocess.run(["/home/jwm/.bun/bin/codex", "--version"], capture_output=True, text=True).stdout.strip()
    )

    print(f"Proposer: {proposer_model}")
    print(f"Targets: {[t.name for t in targets]}")
    print(f"Families: {FAMILIES}")

    # Run each family
    family_summaries = {}
    for family in FAMILIES:
        dataset_path = ROOT / "eval" / "datasets" / f"exp005_{family}.jsonl"
        if not dataset_path.exists():
            print(f"ERROR: Dataset not found: {dataset_path}")
            return 1

        items = read_jsonl(dataset_path)
        summary = run_family(family, items, proposer_model, targets, outdir)
        family_summaries[family] = summary

    # Global summary
    git_info = get_git_info()

    global_summary = {
        "run_name": run_name,
        "git": git_info,
        "proposer": {"model": proposer_model, "temperature": 0.0},
        "targets": [{"name": t.name, "model": t.model} for t in targets],
        "families": FAMILIES,
        "per_family": family_summaries,
    }
    (outdir / "summary.json").write_text(json.dumps(global_summary, indent=2))

    # Generate report
    report_lines = [
        f"# Report: {run_name}",
        "",
        f"**Git:** {git_info['tag']} ({git_info['commit']})",
        f"**Proposer:** {proposer_model}",
        f"**Families:** {', '.join(FAMILIES)}",
        f"**Items per family:** 30",
        "",
    ]

    for family in FAMILIES:
        s = family_summaries[family]
        acc_n = s["accuracy"]["no_trace"]
        acc_u = s["accuracy"]["uncorrupted_trace"]
        acc_c = s["accuracy"]["corrupted_trace"]
        avg_acc_n = sum(acc_n.values()) / len(acc_n)
        avg_acc_u = sum(acc_u.values()) / len(acc_u)
        avg_acc_c = sum(acc_c.values()) / len(acc_c)

        report_lines.extend([
            f"## {family.capitalize()}",
            "",
            f"### 1. Did trace injection increase agreement?",
            f"- No trace: {s['agreement_rate']['no_trace']:.4f}",
            f"- Uncorrupted: {s['agreement_rate']['uncorrupted_trace']:.4f}",
            f"- Corrupted: {s['agreement_rate']['corrupted_trace']:.4f}",
            f"- **{'Yes' if s['agreement_rate']['uncorrupted_trace'] > s['agreement_rate']['no_trace'] else 'No'}** (uncorrupted)",
            "",
            f"### 2. Did trace injection increase accuracy?",
            f"- No trace avg: {avg_acc_n:.4f}",
            f"- Uncorrupted avg: {avg_acc_u:.4f}",
            f"- Corrupted avg: {avg_acc_c:.4f}",
            f"- **{'Yes' if avg_acc_u > avg_acc_n else 'No'}** (uncorrupted)",
            "",
            f"### 3. Did agreement-on-wrong or plurality-wrong emerge?",
            f"- Agreement-on-wrong (no trace): {s['agree_all_counts']['no_trace']['wrong']}",
            f"- Agreement-on-wrong (uncorrupted): {s['agree_all_counts']['uncorrupted_trace']['wrong']}",
            f"- Agreement-on-wrong (corrupted): {s['agree_all_counts']['corrupted_trace']['wrong']}",
            f"- Plurality-wrong (no trace): {s['plurality_wrong']['no_trace']}",
            f"- Plurality-wrong (corrupted): {s['plurality_wrong']['corrupted_trace']}",
            "",
            f"### 4. Most/least affected models",
        ])

        # Find most/least affected by trace
        gain_u = {m: acc_u[m] - acc_n[m] for m in acc_n}
        most_helped = max(gain_u.items(), key=lambda x: x[1])
        least_helped = min(gain_u.items(), key=lambda x: x[1])
        report_lines.extend([
            f"- Most helped by trace: {most_helped[0]} ({most_helped[1]:+.4f})",
            f"- Least helped: {least_helped[0]} ({least_helped[1]:+.4f})",
            "",
        ])

    # Comparison table
    report_lines.extend([
        "## Cross-Family Comparison",
        "",
        "| Family | Agree (N) | Agree (U) | Agree (C) | Acc (N) | Acc (U) | Acc (C) | AoW (C) | Plur-W (C) |",
        "|--------|-----------|-----------|-----------|---------|---------|---------|---------|------------|",
    ])

    for family in FAMILIES:
        s = family_summaries[family]
        avg_acc_n = sum(s["accuracy"]["no_trace"].values()) / len(s["accuracy"]["no_trace"])
        avg_acc_u = sum(s["accuracy"]["uncorrupted_trace"].values()) / len(s["accuracy"]["uncorrupted_trace"])
        avg_acc_c = sum(s["accuracy"]["corrupted_trace"].values()) / len(s["accuracy"]["corrupted_trace"])
        report_lines.append(
            f"| {family} | {s['agreement_rate']['no_trace']:.3f} | {s['agreement_rate']['uncorrupted_trace']:.3f} | "
            f"{s['agreement_rate']['corrupted_trace']:.3f} | {avg_acc_n:.3f} | {avg_acc_u:.3f} | {avg_acc_c:.3f} | "
            f"{s['agree_all_counts']['corrupted_trace']['wrong']} | {s['plurality_wrong']['corrupted_trace']} |"
        )

    (outdir / "report.md").write_text("\n".join(report_lines))

    # Config
    (outdir / "config.json").write_text(json.dumps({
        "proposer": {"model": proposer_model, "temperature": 0.0},
        "targets": [{"name": t.name, "model": t.model, "temperature": 0.0} for t in targets],
        "families": FAMILIES,
        "corruption_k_per_family": K_CORRUPT,
    }, indent=2))

    # Manifest
    manifest = {
        "run_name": run_name,
        "created_utc": utc_ts_compact(),
        "git": git_info,
        "experiment": "semantic-ambiguity-stress",
        "axis": "task_verifiability",
        "families": FAMILIES,
        "conditions": ["no_trace", "uncorrupted_trace", "corrupted_trace"],
        "notes": "Semantic stress test: commonsense, causal, distractor, explanation tasks.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print("=" * 60)
    print(f"Run complete: {run_name}")
    print(f"Results: {outdir}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
