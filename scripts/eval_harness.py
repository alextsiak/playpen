#!/usr/bin/env python3
import os
import json
import argparse
from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
import subprocess

import os
os.environ["CLEM_GAME_REGISTRY"] = "/Users/yanaarva/PycharmProjects/playpen/game_registry.json"

import clemcore.cli
from clemcore.clemgame import GameRegistry
registry = GameRegistry()
specs = registry.get_game_specs()
print(f"Found {len(specs)} specs")
for spec in specs:
    print(spec)
    
from playpen import make_env
import torch

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

def find_spec(registry: GameRegistry, game_name: str, exp_name: str):
    # Try robust lookup using get_game_specs_that_unify_with
    try:
        specs = registry.get_game_specs_that_unify_with(game_name, exp_name)
        if specs:
            return specs[0]
    except Exception:
        pass
    # Fallback: manual scan
    for spec in registry.get_game_specs():
        m = spec.meta
        if m.get("game") == game_name and m.get("experiment") == exp_name:
            return spec
    raise ValueError(f"No GameSpec found for {game_name}/{exp_name}")

def load_test_specs(registry: GameRegistry, test_set_dir):
    specs = []
    for fn in os.listdir(test_set_dir):
        if not fn.endswith(".json"):
            continue
        parts = fn.split("__")
        if len(parts) != 3:
            continue
        _, game_name, exp_part = parts
        exp_name = exp_part[:-5]  # strip ".json"
        try:
            spec = find_spec(registry, game_name, exp_name)
            specs.append((spec, fn))
        except ValueError as e:
            print(f"⚠️  {e}, skipping")
    if not specs:
        raise RuntimeError("🔴 No specs loaded—did you run select_test_set.py and did it produce any JSON files?")
    return specs

def find_model_path(ckpt_dir):
    # Try common adapter/model file names
    for fname in ["adapter_model", "pytorch_model.bin"]:
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.isfile(fpath):
            return fpath
    # Try subdirectory (e.g., PEFT checkpoints)
    for sub in os.listdir(ckpt_dir):
        subdir = os.path.join(ckpt_dir, sub)
        if os.path.isdir(subdir):
            for fname in ["adapter_model", "pytorch_model.bin"]:
                fpath = os.path.join(subdir, fname)
                if os.path.isfile(fpath):
                    return fpath
    # Fallback: use the directory itself if it looks like a model dir
    if any(f.endswith(".bin") for f in os.listdir(ckpt_dir)):
        return ckpt_dir
    raise FileNotFoundError(f"No adapter/model file found in {ckpt_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on Playpen test set.")
    parser.add_argument("--test_set_dir", type=str, default="test_set", help="Directory with test set JSONs.")
    parser.add_argument("--checkpoints_dir", type=str, default="models/sft+lora", help="Directory with model checkpoints.")
    parser.add_argument("--plot", action="store_true", help="Plot a bar chart of results (requires matplotlib).")
    args = parser.parse_args()

    registry = GameRegistry()
    specs = load_test_specs(registry, args.test_set_dir)

    results = []
    for ckpt in sorted(os.listdir(args.checkpoints_dir)):
        ckpt_dir = os.path.join(args.checkpoints_dir, ckpt)
        if not os.path.isdir(ckpt_dir):
            continue
        try:
            model_path = find_model_path(ckpt_dir)
        except FileNotFoundError as e:
            print(f"⚠️  {e}, skipping {ckpt}")
            continue
        model = HuggingfaceLocalModel(model_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        wins = aborts = fails = total = 0
        for spec, fn in specs:
            instances_name = fn[:-5]  # strip ".json"
            with make_env(spec, [model], instances_name=instances_name) as env:
                outcome = env.play()  # or env.run()
                if outcome == "success":
                    wins += 1
                elif outcome == "aborted":
                    aborts += 1
                else:
                    fails += 1
                total += 1
        results.append({"ckpt": ckpt, "wins": wins, "aborts": aborts, "fails": fails, "total": total})

    # Print summary table
    print("\nCheckpoint                    Wins  Aborts  Fails  Total")
    print("------------------------------------------------------")
    for r in results:
        print(f"{r['ckpt']:<28} {r['wins']:>4}   {r['aborts']:>4}   {r['fails']:>4}   {r['total']:>4}")

    # Optional: plot bar chart
    if args.plot and HAS_MPL:
        import numpy as np
        labels = [r['ckpt'] for r in results]
        wins = [r['wins'] for r in results]
        aborts = [r['aborts'] for r in results]
        fails = [r['fails'] for r in results]
        x = np.arange(len(labels))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(labels)*1.2), 5))
        ax.bar(x - width, wins, width, label='Wins')
        ax.bar(x, aborts, width, label='Aborts')
        ax.bar(x + width, fails, width, label='Fails')
        ax.set_ylabel('Count')
        ax.set_title('Checkpoint Evaluation Results')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        plt.show()
    elif args.plot:
        print("matplotlib not installed; cannot plot.")

    # Example: List games and capture output
    result = subprocess.run(
        ["clem", "list", "games", "-v"],
        capture_output=True,
        text=True
    )
    print(result.stdout)

    # Example: Run your test set selector (if you wrap it as a CLI command)
    # subprocess.run(["clem", "your_custom_command", ...])

    # Optionally, parse the output for further processing
    # For example, save to a file:
    with open("cli_games_output.txt", "w") as f:
        f.write(result.stdout)

if __name__ == "__main__":
    main()
