# Report: v0.1.2-baseline-multitarget-claude-plus-gpt52-20260120

**Dataset:** eval/datasets/arithmetic-60.jsonl
**Items:** 60
**Git:** e1a8603 (e1a8603c654c)

## Proposer
- Model: sonnet
- Accuracy: 1.0000

## Agreement
- Pairwise agreement rate (with trace): 0.9689
- Pairwise agreement rate (no trace):  0.8889
- Δ agreement: +0.0800

## Accuracy by Target Model

| Model | No Trace | With Trace | Transfer Gain |
|-------|----------|------------|---------------|
| opus | 1.0000 | 1.0000 | +0.0000 |
| sonnet_tgt | 1.0000 | 1.0000 | +0.0000 |
| haiku | 0.8833 | 0.9833 | +0.1000 |
| gpt5_codex | 0.9500 | 1.0000 | +0.0500 |
| gpt52_codex | 0.9500 | 0.9667 | +0.0167 |
| gpt52 | 0.8667 | 0.9500 | +0.0833 |

## All-Target Agreement Outcomes

| Condition | Agree-on-Correct | Agree-on-Wrong |
|-----------|------------------|----------------|
| With Trace | 55 | 0 |
| No Trace | 44 | 0 |

## Summary Questions

1. **Did trace injection increase agreement?** Yes (Δ = +0.0800)

2. **Did trace injection increase accuracy?**
   - opus: No change (Δ = +0.0000)
   - sonnet_tgt: No change (Δ = +0.0000)
   - haiku: Yes (Δ = +0.1000)
   - gpt5_codex: Yes (Δ = +0.0500)
   - gpt52_codex: Yes (Δ = +0.0167)
   - gpt52: Yes (Δ = +0.0833)

3. **Did consistent-wrong increase?** No (0 → 0, Δ = +0)