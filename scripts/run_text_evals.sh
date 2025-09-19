#!/usr/bin/env bash
set -euo pipefail
export CLEM_MODEL_REGISTRY="$(pwd)/model_registry.json"

MODEL_SFT="llama3-8b-sft-playpen"
MODEL_BASE="Meta-Llama-3.1-8B-Instruct"

run_and_score () {
  local MODEL="$1"
  local TEMP="$2"
  local OUT="$3"
  clem run   -g taboo         -m "$MODEL" -t "$TEMP" -r "$OUT"
  clem score -g taboo         -r "$OUT"
  clem run   -g referencegame -m "$MODEL" -t "$TEMP" -r "$OUT"
  clem score -g referencegame -r "$OUT"
}

run_and_score "$MODEL_SFT"  0.0 results_sft_t00
run_and_score "$MODEL_SFT"  0.7 results_sft_t07
run_and_score "$MODEL_BASE" 0.0 results_base_t00
run_and_score "$MODEL_BASE" 0.7 results_base_t07

python3 scripts/summarize_text_results.py \
  results_sft_t00 results_sft_t07 results_base_t00 results_base_t07 \
  | tee results_min/text_only_summary.txt
