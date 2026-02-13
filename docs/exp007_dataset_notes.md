# Experiment 007 Dataset Notes

## Generation Method

Hand-crafted adversarial items in `scripts/gen_adversarial_dataset.py`. No LLM generation. No random seed.

## Dataset

`eval/datasets/exp007_adversarial.jsonl` — 60 items

## Adversarial Categories (15 items each)

### Plausible Wrong Dominant (adv-001 to adv-015)

Classic riddles and trick questions where the obvious/intuitive answer is wrong. The plausible_wrong answer is what most naive reasoning produces.

### Salience Trap (adv-016 to adv-030)

Problems with irrelevant but attention-grabbing details that distract from the actual question. Numeric distractors, misdirection, and frame-shifting.

### Causal Inversion (adv-031 to adv-045)

Correlation-vs-causation scenarios where the plausible_wrong answer assumes the presented correlation implies causation. Includes confounding variables, reverse causation, and selection bias.

### Normative Framing (adv-046 to adv-060)

Ethical/normative questions where a seemingly "reasonable" answer oversimplifies genuine philosophical disagreement. The plausible_wrong answer presents one framework's conclusion as universal truth.

## Item Schema

```json
{
  "id": "adv-NNN",
  "prompt": "question text",
  "answer": "correct/best answer",
  "plausible_wrong": "trap answer the corrupted trace steers toward",
  "category": "one of four categories above"
}
```
