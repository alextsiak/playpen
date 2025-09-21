# Perfect-Transcript SFT + Near-fix LoRA (Llama-3.1-8B)

**What I tried (short):**
- Start from Meta-Llama-3.1-8B-Instruct.
- Do an SFT using “perfect” CLEM transcripts (the ones that follow the rules).
- ReferenceGame still sometimes outputs more than one line, so I trained a  “near-fix” LoRA on RG examples (mix of near-miss + perfect) to push it toward the strict format.

**Why:**
- See if a little, targeted finetuning helps the model follow game rules better (especially RG’s strict “Answer: …” one-liner).

**What’s in this folder:**
- `datasets/` — small JSONL files I used (e.g., `rg_nearmiss.jsonl`).  
- `scripts/` — training/mining code used (e.g., `train_rg_nearfix_hf.py`).  
- `results/` — a small text summary if available.  
- `models/` — not checked in (see `.gitignore`).  
- `INFO.json` — tiny metadata about this snapshot.

