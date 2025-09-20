#!/bin/sh
set -eu

BASE_MODEL='Meta-Llama-3.1-8B-Instruct'
SFT_MODEL='llama3-8b-sft-tulu3-2025-09-merged'

# Allow overriding GAMES from the environment:
GAMES="${GAMES:-referencegame taboo matchit_ascii wordle wordle_withclue wordle_withcritic imagegame guesswhat cladder eqbench mmlu_pro textmapworld textmapworld_graphreasoning textmapworld_specificroom}"
TEMPS="${TEMPS:-0.0 0.2 0.4 0.7}"

model_key () { printf "%s" "$1" | tr -c 'A-Za-z0-9' '_' ; }
temp_suf  () { printf "%s" "$1" | tr -d '.' ; }

run_one () {
  m="$1"; t="$2"; g="$3"
  mk="$(model_key "$m")"; ts="$(temp_suf "$t")"
  out="results_${mk}_t${ts}_${g}"
  [ -f "${out}/run.json" ] || clem run -g "$g" -m "$m" -t "$t" -r "$out"
  clem score -g "$g" -r "$out"
  clem eval  -r "$out"
  echo "[ok] $out"
}

for M in "$BASE_MODEL" "$SFT_MODEL"; do
  for T in $TEMPS; do
    for G in $GAMES; do
      run_one "$M" "$T" "$G"
    done
  done
done
