#!/usr/bin/env python3
import os, json, random
from datasets import load_dataset
from playpen.curriculum import DIFFICULTY_BUCKETS
from clemcore.clemgame import GameRegistry

# where we’ll dump our 15–20 test files
OUT_DIR = "test_set"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) figure out which (game,experiment) combos actually exist
registry = GameRegistry()
valid_pairs = {
    (spec.meta["game"], spec.meta["experiment"])
    for spec in registry.get_game_specs()
}

# 2) for each difficulty, keep only the valid combos
filtered_buckets = {}
for lvl, bucket in DIFFICULTY_BUCKETS.items():
    fb = [(g, e) for (g, e) in bucket if (g, e) in valid_pairs]
    if len(fb) < len(bucket):
        skipped = set(bucket) - set(fb)
        for (g, e) in skipped:
            print(f"⚠️ skipping {g}/{e} in '{lvl}' — not in registry")
    filtered_buckets[lvl] = fb

# 3) pick up to 5 random experiments per difficulty
picked = []
for lvl, bucket in filtered_buckets.items():
    sample_n = min(5, len(bucket))
    for (g, e) in random.sample(bucket, sample_n):
        picked.append((lvl, g, e))

# 4) load all successful episodes once
ds = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")
ds = ds.filter(lambda ep: ep["meta"]["outcome"] == "success")

# 5) for each picked (lvl,game,exp), grab one random episode and dump it
count = 0
for lvl, game, exp in picked:
    sub = ds.filter(lambda ep, g=game, x=exp: ep["meta"]["game"] == g and ep["meta"]["experiment"] == x)
    if len(sub) == 0:
        print(f"⚠️  no successful episodes for {game}/{exp}, skipping")
        continue
    ep = sub.shuffle(seed=42)[0]
    fn = f"{lvl}__{game}__{exp}.json"
    with open(os.path.join(OUT_DIR, fn), "w") as f:
        json.dump(ep, f, indent=2)
    print(f"Wrote {fn}")
    count += 1

print(f"\n✅ Done — {count} files in {OUT_DIR}/")
