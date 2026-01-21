# Experiment 005 Datasets

## Generation Method
These datasets were generated deterministically for the semantic ambiguity stress test (Experiment 005).

## Families

### commonsense (30 items)
Everyday physical and social reasoning tasks.
- Items test basic understanding of physical causation and social conventions
- Each has one correct answer and plausible alternatives

### causal (30 items)
Multi-step story reasoning with cause-effect chains.
- Items present narratives with clear causal progressions
- Tests ability to track consequences over multiple steps

### distractor (30 items)
Word problems with irrelevant details and distractors.
- Items include extraneous information to test focus
- Correct answers require filtering relevant from irrelevant details

### explanation (30 items)
Hypothesis selection tasks (multiple choice A/B/C).
- Items present phenomena and ask for best explanation
- Tests ability to evaluate competing explanations

## Format
Each line is a JSON object with:
- id: unique identifier (format: <family>-<num>)
- prompt: the question/task
- answer: the gold answer
- alternatives: plausible but wrong answers
- answer_type: "string" or "choice"

## Seed
Datasets are static and deterministic (no random generation).
