# Report: v0.1.5-causal-replication-amplified-20260212

**Git:** 44eb7f8 (44eb7f8ca424)
**Proposer:** sonnet
**Dataset:** exp005_causal.jsonl (30 items)
**Corruption:** K=30 (all items), 3 seeds

## Per-Seed Results

### Seed 1

| Condition | Agreement | Avg Accuracy | AoW | Plurality-Wrong |
|-----------|-----------|-------------|-----|-----------------|
| No trace | 0.0444 | 0.0889 | 0 | 27 |
| Uncorrupted | 0.0222 | 0.1556 | 0 | 26 |
| Corrupted | 0.0556 | 0.1333 | 0 | 26 |

**Per-model accuracy (corrupted):**
- opus: 0.133
- sonnet_tgt: 0.133
- haiku: 0.133

**Corruption follow rate:** {'opus': 0.13333333333333333, 'sonnet_tgt': 0.0, 'haiku': 0.0}
**Total follow events:** 4

### Seed 2

| Condition | Agreement | Avg Accuracy | AoW | Plurality-Wrong |
|-----------|-----------|-------------|-----|-----------------|
| No trace | 0.0000 | 0.0667 | 0 | 27 |
| Uncorrupted | 0.0111 | 0.1444 | 0 | 27 |
| Corrupted | 0.0556 | 0.1333 | 0 | 25 |

**Per-model accuracy (corrupted):**
- opus: 0.133
- sonnet_tgt: 0.133
- haiku: 0.133

**Corruption follow rate:** {'opus': 0.0, 'sonnet_tgt': 0.0, 'haiku': 0.06666666666666667}
**Total follow events:** 2

### Seed 3

| Condition | Agreement | Avg Accuracy | AoW | Plurality-Wrong |
|-----------|-----------|-------------|-----|-----------------|
| No trace | 0.0111 | 0.0667 | 0 | 27 |
| Uncorrupted | 0.0111 | 0.1444 | 0 | 27 |
| Corrupted | 0.0667 | 0.1444 | 1 | 26 |

**Per-model accuracy (corrupted):**
- opus: 0.100
- sonnet_tgt: 0.267
- haiku: 0.067

**Corruption follow rate:** {'opus': 0.1, 'sonnet_tgt': 0.03333333333333333, 'haiku': 0.0}
**Total follow events:** 4

## Replication Summary

| Metric | Value |
|--------|-------|
| AoW per seed | [0, 0, 1] |
| AoW mean | 0.33 |
| AoW std | 0.47 |
| Total follow events | 10 |
| Total plurality-wrong | 77 |

## Answers

### 1. Does agreement-on-wrong replicate across seeds?

**Partial.** Agreement-on-wrong observed only in seed 3.

### 2. Does corruption-follow scale when K=30?

Total corruption-follow events across all seeds: 10
Expected if random: ~9 (baseline agreement rate)

**Yes**, 10 follow events observed.

### 3. Does plurality-wrong emerge under amplified corruption?

Total plurality-wrong under corrupted traces: 77

**No**, plurality-wrong did not increase (81 no trace vs 77 corrupted).

### 4. Which target is most sensitive to corruption?

Mean corruption-follow rate per model (across seeds):
- opus: 0.0778
- sonnet_tgt: 0.0111
- haiku: 0.0222

**Most sensitive:** opus (0.0778)
**Least sensitive:** sonnet_tgt (0.0111)

## Closure Status

- Agreement-on-wrong does not replicate across seeds.
- Corruption-follow rate remains low and non-scaling.
- Plurality-wrong does not increase under amplified corruption.
- Stop condition satisfied.
- No harmonization, selection, or verification layers introduced.