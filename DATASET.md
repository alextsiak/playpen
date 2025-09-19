# "Perfect episodes" dataset

**Source & license.** Episodes are derived from the public clembench-runs artifacts (CC-BY-4.0). We include only short text snippets necessary for supervised fine-tuning; attribution to CLEMbench is retained here.

**Selection criteria ("perfect filters").**
- Taboo: Main Score == 1.0 and (where detectable) first-guess success.
- Wordle & variants: solved in exactly one attempt.
- Drawing / Reference / Private-Shared: strict success (success == true or Main Score == 1.0). Where the logs expose attempts, we keep the first-attempt solution.

**Motivation.** We want a high-precision, low-noise SFT set that focuses the model on robust, instruction-following behavior observed in top runs.

**Statistics.**
- Total examples: 1,286
- Split: 1,157 train / 129 eval (random split at creation time)

**Record format (JSONL, one per line).**
Each line minimally contains:
{
  "input": "<system+user prompt that the model should respond to>",
  "output": "<target assistant message>",
  "meta": {
    "game": "taboo|referencegame|...",
    "split": "train|eval",
    "provenance": "clembench-runs/<run_id>/<episode_id>"
  }
}

**File included.**
- examples/trl/perfect_episodes.jsonl — concatenated, de-duplicated episodes that satisfy the filters above.

**Reproducibility notes.**
- We selected episodes from public run dumps using the rules above (success/score fields from each game’s scores.json and, where available, attempt counters).
- If the lab wants a fully scripted rebuild from upstream artifacts, we can add a small extractor that scans a local clone of clembench-runs and emits this JSONL. For now, we commit the exact JSONL used for SFT to guarantee byte-for-byte reproducibility.

**Attribution.**
- CLEMbench benchmark and run artifacts (CC-BY-4.0). Please cite the CLEMbench paper/website in derivative work.
