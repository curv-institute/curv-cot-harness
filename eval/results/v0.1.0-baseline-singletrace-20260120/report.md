# Report: v0.1.0-baseline-singletrace-20260120

**Dataset:** eval/datasets/arithmetic-60.jsonl
**Items:** 60
**Git:** 81d8a23 (81d8a23928fa)

## Proposer
- Model: sonnet
- Accuracy: 1.0000

## Agreement
- Pairwise agreement rate (with trace): 0.0000
- Pairwise agreement rate (no trace):  0.0000
- Δ agreement: +0.0000

## Accuracy by Target Model

| Model | No Trace | With Trace | Transfer Gain |
|-------|----------|------------|---------------|
| haiku | 0.8667 | 1.0000 | +0.1333 |

## All-Target Agreement Outcomes

| Condition | Agree-on-Correct | Agree-on-Wrong |
|-----------|------------------|----------------|
| With Trace | 60 | 0 |
| No Trace | 52 | 8 |

## Summary Questions

1. **Did trace injection increase agreement?** No (Δ = +0.0000)

2. **Did trace injection increase accuracy?**
   - haiku: Yes (Δ = +0.1333)

3. **Did consistent-wrong increase?** No (8 → 0, Δ = -8)