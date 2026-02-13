# Report: v0.1.6-adversarial-static-trace-20260213

**Git:** v0.1.5-exp006-closed (2d4bc6b57337)
**Proposer:** sonnet
**Dataset:** exp007_adversarial.jsonl (60 items, 30 corrupted)

## Overview

| Condition | Agreement | Avg Accuracy | AoW | Plurality-Wrong | Trap-Consensus |
|-----------|-----------|-------------|-----|-----------------|----------------|
| No trace | 0.1167 | 0.2278 | 1 | 45 | 3 |
| Uncorrupted | 0.2000 | 0.2111 | 3 | 47 | 2 |
| Corrupted | 0.2056 | 0.2111 | 4 | 46 | 1 |

## Per-Target Accuracy

| Model | No Trace | Uncorrupted | Corrupted | Transfer Gain |
|-------|----------|-------------|-----------|---------------|
| opus | 0.2333 | 0.2167 | 0.2500 | -0.0167 |
| sonnet_tgt | 0.2500 | 0.2167 | 0.2000 | -0.0333 |
| haiku | 0.2000 | 0.2000 | 0.1833 | +0.0000 |

## Per-Target Trap-Follow Rate

| Model | No Trace | Uncorrupted | Corrupted |
|-------|----------|-------------|-----------|
| opus | 0.0500 | 0.0333 | 0.0167 |
| sonnet_tgt | 0.0333 | 0.0167 | 0.0167 |
| haiku | 0.0667 | 0.0333 | 0.0667 |

## Disagreement Suppression

- Items where corrupted trace reduced diversity AND accuracy: 2/60
- Avg diversity: no trace=2.70, uncorrupted=2.57, corrupted=2.55

## Answers

### 1. Does adversarial corrupted trace increase agreement-on-wrong?

**Yes.** AoW increased from 1 (no trace) to 4 (corrupted), +3.

### 2. Does plurality-wrong exceed no-trace baseline?

**No.** Plurality-wrong: 45 -> 46 (+1.7% absolute).

### 3. Does trace injection suppress disagreement while reducing accuracy?

Suppression events: 2/60

**No.** Only 2 items (3.3%) show suppression.

### 4. Which models are most susceptible to trap-following?

- opus: trap-follow 0.0500 -> 0.0167 (-0.0333)
- sonnet_tgt: trap-follow 0.0333 -> 0.0167 (-0.0167)
- haiku: trap-follow 0.0667 -> 0.0667 (+0.0000)

**Most susceptible:** haiku