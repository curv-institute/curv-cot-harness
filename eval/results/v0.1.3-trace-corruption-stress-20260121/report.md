# Report: v0.1.3-trace-corruption-stress-20260121

**Dataset:** eval/datasets/arithmetic-60.jsonl
**Items:** 60 (corrupted: 15)
**Git:** a342407 (a3424072ab37)

## Proposer
- Model: sonnet

## Agreement Rates

| Condition | Pairwise Agreement |
|-----------|-------------------|
| No trace | 0.9022 |
| Uncorrupted trace | 0.9467 |
| Corrupted trace | 0.9533 |

## Accuracy by Target Model

| Model | No Trace | Uncorrupted | Corrupted | Gain (U) | Gain (C) |
|-------|----------|-------------|-----------|----------|----------|
| opus | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| sonnet_tgt | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| haiku | 0.8500 | 0.9833 | 0.9833 | +0.1333 | +0.1333 |
| gpt5_codex | 0.9500 | 0.9833 | 1.0000 | +0.0333 | +0.0500 |
| gpt52_codex | 0.9833 | 0.9167 | 0.9833 | -0.0667 | +0.0000 |
| gpt52 | 0.9167 | 0.9500 | 0.8833 | +0.0333 | -0.0333 |

## All-Target Agreement Outcomes

| Condition | Agree-on-Correct | Agree-on-Wrong |
|-----------|------------------|----------------|
| No trace | 44 | 0 |
| Uncorrupted trace | 51 | 0 |
| Corrupted trace | 53 | 0 |

## Corruption-Specific Metrics (K=15 corrupted items)

### Corruption Follow Rate per Model

| Model | Follow Rate |
|-------|-------------|
| haiku | 0.0667 |
| opus | 0.0000 |
| sonnet_tgt | 0.0000 |
| gpt5_codex | 0.0000 |
| gpt52_codex | 0.0000 |
| gpt52 | 0.0000 |

### Collapse to Corrupted Answer: 0.0000
### Average Plurality Strength: 0.9556

### Agreement-on-Wrong (corrupted items only)
- No trace: 0
- Corrupted trace: 0

## Summary Questions

1. **Does corrupted trace injection increase agreement-on-wrong on corrupted items?**
   No (0 → 0)

2. **Which targets follow corrupted traces most/least?**
   - Most: haiku (0.0667)
   - Least: opus (0.0000)

3. **Does corrupted trace injection increase pairwise agreement while decreasing accuracy?**
   Agreement: 0.9022 → 0.9533 (increased)
   Accuracy: 0.9500 → 0.9750 (increased)
   **No** - conditions not met