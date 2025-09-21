# ReferenceGame Near-Miss SFT — Meta-Llama-3.1-8B (Sept 2025)

**Idea/Motivation:** fine-tune the follower to answer with a *strict* single-line label (“Answer: first/second/third”) by training on:
1) canonical RG instance labels (`rg_from_benchmark_all.jsonl`) and  
2) mined *near-miss* examples from our own runs, plus perfect episodes when available.

**Implementation (high-level):**
- Mined instance-gold pairs from `clembench/referencegame/in/*.json`.
- Built near-miss additions from SFT/base RG runs.
- Merged to `rg_nearmiss_merged.jsonl` (540 rows in this snapshot).
- Trained LoRA SFT (`sft_train_tulu3_v2.py`) on Llama-3.1-8B-Instruct.
- Evaluated referencegame across temperatures.

**What’s here**
- `datasets/` — JSONL sources used to train (canonical + merged near-miss)
- `scripts/` — mining/build/training scripts used
- `results/` — per-temperature `results.csv`, `raw.csv` for RG
- `models/LOCATIONS.txt` — where the PEFT/merged weights live (not checked in)
- `INFO.json` — quick metadata

**Notes**
- If you re-run, ensure HF auth for the base model.
- The strict “Answer: …” behavior is enforced via data + prompts; toggle helper lives in `scripts/rg_prompt_toggle.py`.
