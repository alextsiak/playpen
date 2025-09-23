import os, re, json, math
from pathlib import Path
from collections import defaultdict
from typing import Any, Iterable, Tuple

PROJ = Path(os.environ.get("PROJ", str(Path.home() / "playpen")))
RUNS = PROJ / "runs"
VERSION = os.environ.get("RUNS_VERSION", "v2.0")
OUT_DIR = PROJ / "examples" / "trl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT_DIR / "results.jsonl"
STATS_CSV = OUT_DIR / "results_stats.csv"

KNOWN_GAMES = (
    "wordle", "wordle_withclue", "wordle_withcritic",
    "taboo", "referencegame", "privateshared", "imagegame", "drawing"
)

MODEL_PAT = re.compile(os.environ.get(
    "MODEL_PAT",
    r"(o3[-_]?mini|gpt[-_]?4o|chatgpt[-_]?4o|claude[-_]?3(?:\.?5)?[-_]?sonnet)"
), re.I)

def read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def iter_kv(obj: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            yield from iter_kv(v, p)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = f"{prefix}[{i}]"
            yield from iter_kv(v, p)
    else:
        yield (prefix.lower(), obj)

def to_float(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    if isinstance(x, str):
        try:
            v = float(x.strip())
            if math.isnan(v) or math.isinf(v): return None
            return v
        except Exception:
            return None
    return None

def first_numeric(d: Any, keys: Iterable[str]) -> float | None:
    for k, v in iter_kv(d):
        if any(t in k for t in keys):
            f = to_float(v)
            if f is not None:
                return f
    return None

def first_bool(d: Any, keys: Iterable[str]) -> bool | None:
    for k, v in iter_kv(d):
        if any(t in k for t in keys):
            if isinstance(v, bool):
                return v
            f = to_float(v)
            if f is not None:
                return bool(abs(f - 1.0) < 1e-9 or abs(f - 100.0) < 1e-9)
    return None

def attempts_from_scores(s: Any) -> int | None:
    v = first_numeric(s, ["attempt", "attempts", "guess", "guesses", "turn", "turns", "try", "tries", "steps"])
    if v is None: return None
    try:
        return int(round(v))
    except Exception:
        return None

def success_from_scores(s: Any) -> bool | None:
    b = first_bool(s, ["success", "solved", "correct", "win", "exact", "match"])
    return b

def score_from_scores(s: Any) -> float | None:
    return first_numeric(s, ["score", "points", "f1", "accuracy", "acc"])

def is_perfect_for(game: str, scores: Any) -> bool:
    g = game.lower()
    attempts = attempts_from_scores(scores)
    success = success_from_scores(scores)
    score = score_from_scores(scores)

    if "taboo" in g:
        if score is not None and int(round(score)) == 100:
            return True
        return bool(success) and attempts == 1
    if "wordle" in g:
        if first_numeric(scores, ["solved_in"]) == 1:
            return True
        return bool(success) and attempts == 1
    # drawing/reference/private-shared/imagegame: strict success, prefer attempts==1 if present
    if success is True:
        return (attempts == 1) if attempts is not None else True
    if score is not None:
        return abs(score - 1.0) < 1e-9 or abs(score - 100.0) < 1e-9
    return False

def normalize_messages(msgs: Any) -> list[dict]:
    out: list[dict] = []
    for idx, m in enumerate(msgs or []):
        role = "user"
        text = ""
        if isinstance(m, dict):
            role = (m.get("role") or m.get("speaker") or m.get("name") or "").lower()
            text = m.get("content") or m.get("text") or m.get("message") or ""
        elif isinstance(m, str):
            line = m.strip()
            mobj = re.match(r'^\s*(gm|game[-_ ]?master|system|assistant|model|ai|user|player|human)\s*[:：]\s*(.*)$', line, re.I)
            if mobj:
                token = mobj.group(1).lower()
                text = mobj.group(2)
                if token in ("assistant", "model", "ai"):
                    role = "assistant"
                elif token in ("user", "player", "human"):
                    role = "user"
                elif token in ("gm", "game master", "game-master", "system"):
                    role = "system"
                else:
                    role = "user"
            else:
                # fallback: alternate roles
                role = "user" if idx % 2 == 0 else "assistant"
                text = line
        else:
            continue
        out.append({"role": role, "content": text})

    # collapse consecutive same-role
    merged: list[dict] = []
    for m in out:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(m)
    return merged

def iter_model_dirs():
    vdir = RUNS / VERSION
    if not vdir.is_dir():
        return
    for entry in vdir.iterdir():
        if not entry.is_dir(): 
            continue
        # "<modelA>--<modelB>"
        left = entry.name.split("--")[0]
        if MODEL_PAT.search(left):
            yield entry, left

def detect_game_from_path(p: Path) -> str:
    parts = [x.lower() for x in p.parts]
    for g in KNOWN_GAMES:
        if g in parts:
            return g
    return p.parent.name.lower()

def main():
    kept = []
    counts = defaultdict(int)
    considered_dirs = 0

    for model_dir, model_name in iter_model_dirs():
        considered_dirs += 1
        # Only descend into paths that contain known interactive games to keep things fast
        for root, dirs, files in os.walk(model_dir):
            rlow = root.lower()
            if not any(g in rlow for g in KNOWN_GAMES):
                continue
            if "scores.json" in files and "interactions.json" in files:
                ep_dir = Path(root)
                scores = read_json(ep_dir / "scores.json")
                if not isinstance(scores, (dict, list)):
                    continue
                game = detect_game_from_path(ep_dir)
                if not is_perfect_for(game, scores):
                    continue
                inter = read_json(ep_dir / "interactions.json")
                if isinstance(inter, dict) and "interactions" in inter:
                    msgs = inter.get("interactions")
                else:
                    msgs = inter
                messages = normalize_messages(msgs or [])
                if not any(m["role"] == "assistant" for m in messages):
                    continue
                iid = ep_dir.name
                item = {
                    "messages": messages,
                    "meta": {
                        "game": game,
                        "instance_id": iid,
                        "model": model_name,
                        "source_file": str(ep_dir)
                    }
                }
                kept.append(item)
                counts[game] += 1

    if not kept:
        print("No perfect episodes found for the chosen models under", RUNS / VERSION)
        return

    seen, deduped = set(), []
    for e in kept:
        gid = e["meta"]["game"]
        iid = str(e["meta"]["instance_id"])
        first_a = next((m["content"] for m in e["messages"] if m["role"] == "assistant"), "")[:160]
        key = (gid, iid, first_a)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for e in deduped:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    with open(STATS_CSV, "w", encoding="utf-8") as f:
        f.write("game,count\n")
        for g, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            f.write(f"{g},{c}\n")

    print(f"Wrote {len(deduped)} episodes → {OUT_JSONL}")
    print(f"Per-game counts → {STATS_CSV}")

if __name__ == "__main__":
    main()
