#!/usr/bin/env bash
set -euo pipefail

STAMP=$(date +%Y-%m-%d_%H%M)
OUT="report/$STAMP"
mkdir -p "$OUT"/{summaries,results,rg_data,scripts}

# --- summaries you produced ---
cp runs/clem_stats_by_temp_fixed.csv "$OUT/summaries/clem_stats_by_temp.csv"
cp runs/per_game_pivot_fixed.csv     "$OUT/summaries/per_game_pivot.csv"
cp runs/grid_summary_base.csv        "$OUT/summaries/grid_summary_base.csv"

# --- RG datasets you trained on / merged ---
cp datasets/rg_nearmiss_merged.jsonl         "$OUT/rg_data/" || true
cp datasets/rg_from_benchmark_all.jsonl      "$OUT/rg_data/" || true

# --- per-game artifacts: copy only small top-level files (skip episode_* trees) ---
# finds every run dir that has raw.csv and pulls raw.csv/results.csv/results.html
while IFS= read -r -d '' raw; do
  d=$(dirname "$raw")
  rel="${d#./}"
  dest="$OUT/results/$rel"
  mkdir -p "$dest"
  for af in raw.csv results.csv results.html; do
    [ -f "$d/$af" ] && cp "$d/$af" "$dest/"
  done
done < <(find results_* -maxdepth 1 -type f -name raw.csv -print0)

# --- model registry (text) ---
[ -f model_registry.json ] && cp model_registry.json "$OUT/"

# --- the exact scripts you used (only if present) ---
for s in \
  scripts/sft_train_tulu3.py \
  scripts/sft_train_tulu3_v2.py \
  scripts/mine_rg_from_any.py \
  scripts/mine_rg_from_instances.py \
  scripts/build_rg_only.py \
  scripts/build_rg_from_rawcsv.py \
  scripts/rg_prompt_toggle.py
do
  [ -f "$s" ] && cp "$s" "$OUT/scripts/"
done

# --- lightweight README for the bundle ---
cat > "$OUT/README.md" <<'MD'
# What’s here

- `summaries/` — aggregated metrics (clem score by temp, per-game pivot, grid summary)
- `results/` — per-game results for each temperature (only raw.csv, results.csv, results.html)
- `rg_data/` — RG training sets used (merged near-miss and benchmark seed set)
- `scripts/` — exact training/mining/toggling scripts used
- `model_registry.json` — local entries used to run models

These are the artifacts referenced in the report and are small enough to live in git.
MD

echo "[ok] bundle at $OUT"
