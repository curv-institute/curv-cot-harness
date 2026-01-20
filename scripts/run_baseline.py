#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]


def utc_ts_compact() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


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
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass(frozen=True)
class ModelSpec:
    name: str               # logical name used in manifests
    kind: str               # "claude" | "codex"
    model: str              # model id passed to CLI
    temperature: str = "0"


def run_cli(kind: str, model: str, prompt: str, temperature: str = "0") -> str:
    """Runs a model call through either Claude CLI or Codex CLI.

    Claude CLI:
    - Uses -p (print mode) for non-interactive single-response output
    - Uses --model to select model
    - Prompt passed via stdin
    - No direct temperature control (uses default settings)

    Codex CLI:
    - Uses 'exec' subcommand for non-interactive mode
    - Uses -m for model selection
    - Uses -c temperature=<val> for temperature control
    - Reads prompt from stdin when '-' is passed

    Configure command names via env:
    - CLAUDE_CMD (default: "claude")
    - CODEX_CMD  (default: "codex")
    """

    if kind == "claude":
        cmd = os.environ.get("CLAUDE_CMD", "claude")
        # Claude CLI: print mode, model selection, prompt via stdin
        args = [cmd, "-p", "--model", model]
    elif kind == "codex":
        cmd = os.environ.get("CODEX_CMD", "codex")
        # Codex CLI: exec subcommand, model, temperature config, read from stdin
        args = [cmd, "exec", "-m", model, "-c", f"temperature={temperature}", "-"]
    else:
        raise ValueError(f"Unknown kind: {kind}")

    proc = subprocess.run(
        args,
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"CLI call failed (kind={kind}, model={model})\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def must_json_obj(s: str) -> dict[str, Any]:
    try:
        obj = json.loads(s)
    except Exception as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{s}") from e
    if not isinstance(obj, dict):
        raise ValueError(f"Model JSON must be an object. Raw output:\n{s}")
    return obj


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
    if len(sys.argv) < 5:
        print(
            "Usage: scripts/run_baseline.py <run_name> <dataset_jsonl> <proposer_kind:claude|codex> <proposer_model> [<target_spec>...]\n\n"
            "Target spec format: <kind>:<model>:<name> (name optional; defaults to model)\n"
            "Example: codex:gpt-4.1-mini:codex-mini"
        )
        return 2

    run_name = sys.argv[1]
    dataset_path = Path(sys.argv[2])
    proposer_kind = sys.argv[3]
    proposer_model = sys.argv[4]

    targets: list[ModelSpec] = []
    for spec in sys.argv[5:]:
        parts = spec.split(":")
        if len(parts) < 2:
            raise ValueError(f"Bad target spec: {spec}")
        kind, model = parts[0], parts[1]
        name = parts[2] if len(parts) >= 3 else model
        targets.append(ModelSpec(name=name, kind=kind, model=model))

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

    proposer_rows = []
    injected_rows = []
    answers_with_trace = []
    answers_no_trace = []
    correctness_rows = []
    metrics_rows = []

    for it in items:
        item_id = str(it["id"])
        q = str(it["prompt"])
        gold = str(it["answer"])

        # Proposer
        p_prompt = proposer_prompt(proposer_t, q)
        p_raw = run_cli(proposer_kind, proposer_model, p_prompt, temperature="0")
        p_obj = must_json_obj(p_raw)
        trace = str(p_obj.get("trace", ""))
        p_final = normalize_answer(str(p_obj.get("final", "")))

        proposer_rows.append(
            {
                "id": item_id,
                "proposer_kind": proposer_kind,
                "proposer_model": proposer_model,
                "trace": trace,
                "final": p_final,
                "raw": p_raw,
            }
        )

        # Targets
        with_trace_finals: dict[str, str] = {}
        no_trace_finals: dict[str, str] = {}

        for t in targets:
            # with trace
            twt = target_prompt_with_trace(target_with_trace_t, q, trace)
            raw_w = run_cli(t.kind, t.model, twt, temperature=t.temperature)
            obj_w = must_json_obj(raw_w)
            fin_w = normalize_answer(str(obj_w.get("final", "")))
            with_trace_finals[t.name] = fin_w

            # no trace
            tnt = target_prompt_no_trace(target_no_trace_t, q)
            raw_n = run_cli(t.kind, t.model, tnt, temperature=t.temperature)
            obj_n = must_json_obj(raw_n)
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
        # Definition: count per-item whether ALL targets agree; then check correctness of agreed answer.
        all_agree_w = len(set(with_trace_finals.values())) == 1
        all_agree_n = len(set(no_trace_finals.values())) == 1

        aon_c_w = int(all_agree_w and (next(iter(with_trace_finals.values())) == gold_n))
        aon_w_w = int(all_agree_w and (next(iter(with_trace_finals.values())) != gold_n))
        aon_c_n = int(all_agree_n and (next(iter(no_trace_finals.values())) == gold_n))
        aon_w_n = int(all_agree_n and (next(iter(no_trace_finals.values())) != gold_n))

        metrics_rows.append(
            {
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
            }
        )

        injected_rows.append({"id": item_id, "trace": trace})

    # Write artifacts
    write_jsonl(outdir / "traces" / "proposer.jsonl", proposer_rows)
    write_jsonl(outdir / "traces" / "injected.jsonl", injected_rows)
    write_jsonl(outdir / "judgments" / "answers_with_trace.jsonl", answers_with_trace)
    write_jsonl(outdir / "judgments" / "answers_no_trace.jsonl", answers_no_trace)
    write_jsonl(outdir / "judgments" / "correctness.jsonl", correctness_rows)
    write_jsonl(outdir / "metrics.jsonl", metrics_rows)

    # Aggregate summary (simple; extend later)
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

    summary = {
        "run_name": run_name,
        "dataset": str(dataset_path),
        "n_items": len(items),
        "proposer": {"kind": proposer_kind, "model": proposer_model, "temperature": "0"},
        "targets": [t.__dict__ for t in targets],
        "agreement_rate": {"with_trace": agree_rate_w, "no_trace": agree_rate_n},
        "agree_all_counts": {
            "agree_on_correct_with_trace": aon_c_w,
            "agree_on_wrong_with_trace": aon_w_w,
            "agree_on_correct_no_trace": aon_c_n,
            "agree_on_wrong_no_trace": aon_w_n,
        },
        "accuracy": {"with_trace": acc_with, "no_trace": acc_no, "transfer_gain": {k: acc_with[k] - acc_no[k] for k in acc_no}},
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = (
        f"# Report: {run_name}\n\n"
        f"Items: {len(items)}\n\n"
        f"## Agreement\n"
        f"- Pairwise agreement rate (with trace): {agree_rate_w:.4f}\n"
        f"- Pairwise agreement rate (no trace):  {agree_rate_n:.4f}\n\n"
        f"## Accuracy\n"
        + "\n".join([f"- {m}: with={acc_with[m]:.4f} no={acc_no[m]:.4f} Δ={acc_with[m]-acc_no[m]:+.4f}" for m in acc_no])
        + "\n\n"
        f"## All-target agreement outcomes (counts)\n"
        f"- Agree-on-correct (with trace): {aon_c_w}\n"
        f"- Agree-on-wrong   (with trace): {aon_w_w}\n"
        f"- Agree-on-correct (no trace):  {aon_c_n}\n"
        f"- Agree-on-wrong   (no trace):  {aon_w_n}\n"
    )
    (outdir / "report.md").write_text(report, encoding="utf-8")

    # Minimal manifest (extend as needed)
    manifest = {
        "run_name": run_name,
        "created_utc": utc_ts_compact(),
        "git": {"commit": os.environ.get("GIT_COMMIT", "UNKNOWN"), "tag": os.environ.get("GIT_TAG", "UNKNOWN")},
        "cli": {
            "claude_cmd": os.environ.get("CLAUDE_CMD", "claude"),
            "codex_cmd": os.environ.get("CODEX_CMD", "codex"),
        },
        "notes": "Baseline: single proposer trace injection vs no-trace control. No harmonization.",
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (outdir / "config.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "proposer": {"kind": proposer_kind, "model": proposer_model, "temperature": "0"},
                "targets": [t.__dict__ for t in targets],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
