# Perfect-Transcript SFT + Near-fix LoRA (Llama-3.1-8B)

**What was done**
- Started from Meta-Llama-3.1-8B-Instruct.
- Fine-tuned (SFT) on “perfect” CLEM transcripts (strict, rule-following examples).
- RG still sometimes printed extra text, so I trained a near-fix LoRA on RG (mix of near-miss + perfect) to force the one-line format: `Answer: first|second|third`.

**Why**
- Test if a small, targeted finetune improves rule-following, especially RG’s strict answer style.

**What’s included**
- `datasets/` – small JSONL files
- `scripts/` – training/mining code
- `models/` – not checked in
