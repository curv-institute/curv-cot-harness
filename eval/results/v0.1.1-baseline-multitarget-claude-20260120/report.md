# Report: v0.1.1-baseline-multitarget-claude-20260120

**Dataset:** eval/datasets/arithmetic-60.jsonl
**Items:** 60
**Git:** 1c0a415 (1c0a4155b375)

## Proposer
- Model: sonnet
- Accuracy: 1.0000

## Agreement
- Pairwise agreement rate (with trace): 1.0000
- Pairwise agreement rate (no trace):  0.9667
- Δ agreement: +0.0333

## Accuracy by Target Model

| Model | No Trace | With Trace | Transfer Gain |
|-------|----------|------------|---------------|
| opus | 1.0000 | 1.0000 | +0.0000 |
| sonnet_tgt | 1.0000 | 1.0000 | +0.0000 |
| haiku | 0.9500 | 1.0000 | +0.0500 |

## All-Target Agreement Outcomes

| Condition | Agree-on-Correct | Agree-on-Wrong |
|-----------|------------------|----------------|
| With Trace | 60 | 0 |
| No Trace | 57 | 0 |

## Summary Questions

1. **Did trace injection increase agreement?** Yes (Δ = +0.0333)

2. **Did trace injection increase accuracy?**
   - opus: No change (Δ = +0.0000)
   - sonnet_tgt: No change (Δ = +0.0000)
   - haiku: Yes (Δ = +0.0500)

3. **Did consistent-wrong increase?** No (0 → 0, Δ = +0)