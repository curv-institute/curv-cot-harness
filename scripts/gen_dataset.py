#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Generate a simple arithmetic dataset with unambiguous ground truth."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def gen_arithmetic_items(n: int, seed: int = 42) -> list[dict]:
    """Generate arithmetic word problems with clear answers."""
    random.seed(seed)
    items = []
    item_id = 0

    templates = [
        # Addition
        (
            "A store has {a} apples and receives a shipment of {b} more apples. How many apples does the store have now?",
            lambda a, b: a + b,
        ),
        # Subtraction
        (
            "A library has {a} books. If {b} books are checked out, how many books remain in the library?",
            lambda a, b: a - b,
        ),
        # Multiplication
        (
            "A farmer has {a} rows of trees with {b} trees in each row. How many trees does the farmer have in total?",
            lambda a, b: a * b,
        ),
        # Division
        (
            "A teacher has {a} pencils to distribute equally among {b} students. How many pencils does each student get?",
            lambda a, b: a // b,
        ),
        # Two-step: add then multiply
        (
            "A bakery makes {a} cakes in the morning and {b} cakes in the afternoon. If each cake is cut into {c} slices, how many slices are there in total?",
            lambda a, b, c: (a + b) * c,
        ),
        # Two-step: multiply then subtract
        (
            "A box contains {a} packs of {b} cookies each. If {c} cookies are eaten, how many cookies remain?",
            lambda a, b, c: (a * b) - c,
        ),
    ]

    # Generate items ensuring valid operations
    while len(items) < n:
        tpl_idx = item_id % len(templates)
        template, fn = templates[tpl_idx]

        if tpl_idx == 1:  # subtraction: ensure a >= b
            a = random.randint(20, 100)
            b = random.randint(1, a)
            prompt = template.format(a=a, b=b)
            answer = fn(a, b)
        elif tpl_idx == 3:  # division: ensure clean division
            b = random.randint(2, 10)
            quotient = random.randint(5, 20)
            a = b * quotient
            prompt = template.format(a=a, b=b)
            answer = fn(a, b)
        elif tpl_idx in (4, 5):  # three-variable templates
            a = random.randint(5, 20)
            b = random.randint(5, 20)
            c = random.randint(2, 10)
            if tpl_idx == 5:  # ensure subtraction doesn't go negative
                total = a * b
                c = random.randint(1, min(total - 1, 30))
            prompt = template.format(a=a, b=b, c=c)
            answer = fn(a, b, c)
        else:  # simple two-variable
            a = random.randint(10, 50)
            b = random.randint(10, 50)
            prompt = template.format(a=a, b=b)
            answer = fn(a, b)

        items.append({
            "id": f"item-{item_id:04d}",
            "prompt": prompt,
            "answer": str(answer),
            "answer_type": "number",
        })
        item_id += 1

    return items


def main() -> int:
    items = gen_arithmetic_items(60, seed=20260120)

    out_path = ROOT / "eval" / "datasets" / "arithmetic-60.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(items)} items to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
