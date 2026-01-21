# Report: v0.1.4-semantic-ambiguity-stress-20260121

**Git:** 6e242be (6e242bedea4d)
**Proposer:** sonnet
**Families:** commonsense, causal, distractor, explanation
**Items per family:** 30

## Commonsense

### 1. Did trace injection increase agreement?
- No trace: 0.0000
- Uncorrupted: 0.0889
- Corrupted: 0.1000
- **Yes** (uncorrupted)

### 2. Did trace injection increase accuracy?
- No trace avg: 0.2333
- Uncorrupted avg: 0.2444
- Corrupted avg: 0.2444
- **Yes** (uncorrupted)

### 3. Did agreement-on-wrong or plurality-wrong emerge?
- Agreement-on-wrong (no trace): 0
- Agreement-on-wrong (uncorrupted): 0
- Agreement-on-wrong (corrupted): 0
- Plurality-wrong (no trace): 21
- Plurality-wrong (corrupted): 22

### 4. Most/least affected models
- Most helped by trace: opus (+0.0333)
- Least helped: haiku (-0.0333)

## Causal

### 1. Did trace injection increase agreement?
- No trace: 0.0111
- Uncorrupted: 0.0444
- Corrupted: 0.0778
- **Yes** (uncorrupted)

### 2. Did trace injection increase accuracy?
- No trace avg: 0.0778
- Uncorrupted avg: 0.0778
- Corrupted avg: 0.1111
- **No** (uncorrupted)

### 3. Did agreement-on-wrong or plurality-wrong emerge?
- Agreement-on-wrong (no trace): 0
- Agreement-on-wrong (uncorrupted): 0
- Agreement-on-wrong (corrupted): 1
- Plurality-wrong (no trace): 28
- Plurality-wrong (corrupted): 26

### 4. Most/least affected models
- Most helped by trace: opus (+0.0667)
- Least helped: haiku (-0.1000)

## Distractor

### 1. Did trace injection increase agreement?
- No trace: 0.6444
- Uncorrupted: 0.9333
- Corrupted: 0.8556
- **Yes** (uncorrupted)

### 2. Did trace injection increase accuracy?
- No trace avg: 1.0000
- Uncorrupted avg: 1.0000
- Corrupted avg: 1.0000
- **No** (uncorrupted)

### 3. Did agreement-on-wrong or plurality-wrong emerge?
- Agreement-on-wrong (no trace): 0
- Agreement-on-wrong (uncorrupted): 0
- Agreement-on-wrong (corrupted): 0
- Plurality-wrong (no trace): 0
- Plurality-wrong (corrupted): 0

### 4. Most/least affected models
- Most helped by trace: opus (+0.0000)
- Least helped: opus (+0.0000)

## Explanation

### 1. Did trace injection increase agreement?
- No trace: 0.4222
- Uncorrupted: 0.5556
- Corrupted: 0.6000
- **Yes** (uncorrupted)

### 2. Did trace injection increase accuracy?
- No trace avg: 1.0000
- Uncorrupted avg: 1.0000
- Corrupted avg: 1.0000
- **No** (uncorrupted)

### 3. Did agreement-on-wrong or plurality-wrong emerge?
- Agreement-on-wrong (no trace): 0
- Agreement-on-wrong (uncorrupted): 0
- Agreement-on-wrong (corrupted): 0
- Plurality-wrong (no trace): 0
- Plurality-wrong (corrupted): 0

### 4. Most/least affected models
- Most helped by trace: opus (+0.0000)
- Least helped: opus (+0.0000)

## Cross-Family Comparison

| Family | Agree (N) | Agree (U) | Agree (C) | Acc (N) | Acc (U) | Acc (C) | AoW (C) | Plur-W (C) |
|--------|-----------|-----------|-----------|---------|---------|---------|---------|------------|
| commonsense | 0.000 | 0.089 | 0.100 | 0.233 | 0.244 | 0.244 | 0 | 22 |
| causal | 0.011 | 0.044 | 0.078 | 0.078 | 0.078 | 0.111 | 1 | 26 |
| distractor | 0.644 | 0.933 | 0.856 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| explanation | 0.422 | 0.556 | 0.600 | 1.000 | 1.000 | 1.000 | 0 | 0 |