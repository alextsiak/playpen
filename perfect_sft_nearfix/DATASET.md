# "Perfect episodes" dataset

**Source & license.** Episodes are derived from the public clembench-runs artifacts (CC-BY-4.0). We include only short text snippets necessary for supervised fine-tuning.

**Selection criteria ("perfect filters").**
- Taboo: Main Score == 1.0 and (where detectable) first-guess success.
- Wordle & variants: solved in exactly one attempt.
- Drawing / Reference / Private-Shared: strict success (success == true or Main Score == 1.0). Where the logs expose attempts, we keep the first-attempt solution.

**Motivation.** We wanted a high-precision and low-noise SFT set that focuses the model on instruction-following behavior seen in top runs.

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

**Attribution.**
- CLEMbench benchmark and run artifacts (CC-BY-4.0).
