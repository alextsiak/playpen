#!/bin/sh
set -eu

NAME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

STAMP=$(date +%Y-%m-%d_%H%M)
OUT="report/${NAME:-bundle}_${STAMP}"

echo "[i] creating bundle at: $OUT"
mkdir -p "$OUT/summaries" "$OUT/results" "$OUT/rg_data" "$OUT/scripts"

# Summaries
[ -f runs/clem_stats_by_temp_fixed.csv ] && cp runs/clem_stats_by_temp_fixed.csv "$OUT/summaries/clem_stats_by_temp.csv" || true
[ -f runs/per_game_pivot_fixed.csv ]     && cp runs/per_game_pivot_fixed.csv     "$OUT/summaries/per_game_pivot.csv"     || true
[ -f runs/grid_summary_base.csv ]        && cp runs/grid_summary_base.csv        "$OUT/summaries/grid_summary_base.csv"  || true

# Per-game results
find . -maxdepth 2 -type f -name raw.csv | while IFS= read -r raw; do
  d=$(dirname "$raw"); rel=${d#./}; dest="$OUT/results/$rel"
  mkdir -p "$dest"
  for af in raw.csv results.csv results.html; do
    [ -f "$d/$af" ] && cp "$d/$af" "$dest/"
  done
done

# Datasets
[ -f datasets/rg_nearmiss_merged.jsonl ]    && cp datasets/rg_nearmiss_merged.jsonl    "$OUT/rg_data/" || true
[ -f datasets/rg_from_benchmark_all.jsonl ] && cp datasets/rg_from_benchmark_all.jsonl "$OUT/rg_data/" || true

# Registry
[ -f model_registry.json ] && cp model_registry.json "$OUT/" || true

# Scripts (best effort)
for s in \
  scripts/sft_train_tulu3.py \
  scripts/sft_train_tulu3_v2.py \
  scripts/mine_rg_from_any.py \
  scripts/mine_rg_from_instances.py \
  scripts/build_rg_only.py \
  scripts/build_rg_from_rawcsv.py \
  scripts/rg_prompt_toggle.py \
  scripts/run_grid_posix.sh \
  scripts/run_grid_alltemps_posix.sh \
  scripts/make_clem_stats_tables.py
do
  [ -f "$s" ] && cp "$s" "$OUT/scripts/"
done

# README
cat > "$OUT/README.md" <<'MD'
# Results bundle

- Model: Meta-Llama-3.1-8B-Instruct (HF local)
- Temps covered: t00, t02, t04 (full), t07 (partial)
- Contents:
  - summaries/: clem_stats_by_temp.csv, per_game_pivot.csv, grid_summary_base.csv
  - results/: per-game result folders with raw.csv, results.csv, results.html
  - rg_data/: RG near-miss + seed datasets (if present)
  - scripts/: scripts used to produce these results
  - model_registry.json (if present)

Notes:
- `stat_avg` for t07 may be NaN if a stat game didn’t run.
- RG near-miss dataset size in your run: 540 lines.
MD

# INFO.json (if python is available)
if command -v python >/dev/null 2>&1; then
python <<'PY'
import os, re, json, glob
dirs=set(); temps=set(); games=set()
for raw in glob.glob("results_*/*/../raw.csv")+glob.glob("results_*/raw.csv"):
    d=os.path.dirname(raw)
    m=re.search(r"_t(\d{2})_", d)
    g=re.search(r"_t\d{2}_(.+)$", d)
    if m and g:
        temps.add(m.group(1)); games.add(g.group(1)); dirs.add(d)
out=os.environ.get("OUT","")
info={"temps":sorted(temps),"games":sorted(games),"num_result_dirs":len(dirs)}
with open(os.path.join(out,"INFO.json"),"w") as f: json.dump(info, f, indent=2)
print("[ok] INFO.json:", info)
PY
fi

echo "[ok] bundle ready at: $OUT"
du -sh "$OUT" 2>/dev/null || true
