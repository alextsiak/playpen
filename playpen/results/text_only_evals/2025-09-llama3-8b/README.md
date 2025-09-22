# Text-only Evals — Meta-Llama-3.1-8B (Sept 2025)

**What:** CLEMbench text-only games run at temperatures t00/t02/t04 (+ partial t07).
**Why:** Establish a baseline and compare across decoding temps.

## Contents
- `clem_stats_by_temp_fixed.csv` — aggregated CLEMscore/STATscore per temperature
- `per_game_pivot_fixed.csv` — per-game CLEMscore across temps
- `grid_summary_base.csv` — game-level episode counts & timing
- `artifacts/` — per-temp minimal bundles (`results.csv`, `raw.csv`)
- `INFO.json` — quick summary of temps/games included

## Notes
- t07 is partial in this dump (some games missing; STATscore shows NaN).
- Use `results_*_tXX_*/results.csv` to drill into any particular game.
