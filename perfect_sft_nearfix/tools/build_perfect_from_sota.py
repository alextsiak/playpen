import os, re, json, sys, math
from pathlib import Path
from typing import Any, List, Dict

PROJ = Path(os.environ.get("PROJ", str(Path.home()/ "playpen"))).expanduser()
RUNS_VERSION = os.environ.get("RUNS_VERSION", "v2.0")
RUNS = PROJ / "runs" / RUNS_VERSION
OUT = PROJ / "examples" / "trl"
OUT.mkdir(parents=True, exist_ok=True)
OUT_JSONL = OUT / "results.jsonl"
STATS = OUT / "results_stats.csv"

MODEL_PAT = re.compile(os.environ.get(
    "MODEL_PAT",
    r"(o3[-_]?mini|gpt[-_]?4o|claude.*3\.5.*sonnet)"
), re.I)

def read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def first_numeric(d: Any, keys: List[str]) -> float | None:
    if not isinstance(d, dict): return None
    pool = [d]
    es = d.get("episode scores") if isinstance(d.get("episode scores"), dict) else None
    if es: pool.append(es)
    for blob in pool:
        for k in keys:
            for kk in (k, k.lower(), k.title(), k.replace(" ", "_"), k.replace("_", " ")):
                v = blob.get(kk)
                if isinstance(v, (int, float)):
                    if isinstance(v, float) and math.isnan(v):
                        continue
                    return float(v)
                if isinstance(v, str):
                    try:
                        fv = float(v)
                        if math.isnan(fv):
                            continue
                        return fv
                    except Exception:
                        pass
    return None

def success_from_scores(scores: Any) -> bool | None:
    if not isinstance(scores, dict): return None
    pool = [scores]
    es = scores.get("episode scores") if isinstance(scores.get("episode scores"), dict) else None
    if es: pool.append(es)
    for blob in pool:
        for k in ("success","Success","game_successfully_finished"):
            v = blob.get(k)
            if isinstance(v, bool): return v
            if isinstance(v, (int, float)): return bool(v)
    return None

def score_from_scores(scores: Any) -> float | None:
    return first_numeric(scores, ["score","main score","main_score","Main Score"])

def attempts_from_scores(scores: Any) -> float | None:
    return first_numeric(scores, ["solved_in","solved-in","solved in","attempt","attempts","guesses","turns"])

def is_perfect_for(game: str, scores: Any) -> bool:
    g = (game or "").lower()
    def num(*keys): return first_numeric(scores, list(keys))
    suc = success_from_scores(scores)

    if "taboo" in g:
        ms = num("main score","main_score")
        return ms is not None and int(round(ms)) == 100

    if "wordle" in g:
        si = num("solved_in","solved-in","solved in")
        att = attempts_from_scores(scores)
        return (si is not None and int(round(si)) == 1) or (att is not None and int(round(att)) == 1)

    if "adventuregame" in g:
        ms = num("main score","main_score")
        return (ms is not None and int(round(ms)) == 100) or (suc is True)

    if suc is True: return True
    acc = num("accuracy","acc","f1")
    if acc is not None and abs(acc-1.0) < 1e-9: return True
    sc = score_from_scores(scores)
    if sc is not None and (abs(sc-1.0) < 1e-9 or int(round(sc)) == 100): return True
    return False

HDR_RE = re.compile(r"<\|start_header_id\|>(system|user|assistant)<\|end_header_id\|>\s*\n\n(.*?)<\|eot_id\|>", re.S | re.I)

def _extract_last_block(text: str, header: str) -> str | None:
    if not isinstance(text, str): return None
    hs = header.lower()
    out = None
    for m in HDR_RE.finditer(text):
        if m.group(1).lower() == hs:
            out = m.group(2).strip()
    return out

def _flatten_steps(x):
    if isinstance(x, dict):
        if "requests" in x: yield from _flatten_steps(x["requests"])
        else: yield x
    elif isinstance(x, (list, tuple)):
        for y in x: yield from _flatten_steps(y)

def turns_from_requests(ep_dir: Path) -> List[Dict] | None:
    rq = ep_dir / "requests.json"
    data = read_json(rq)
    steps = [st for st in _flatten_steps(data)] if data is not None else []
    steps = [st for st in steps if isinstance(st, dict)]
    if not steps: return None

    messages: List[Dict[str,str]] = []

    # optional system from the first step that has manipulated_prompt_obj
    for st in steps:
        mpo = st.get("manipulated_prompt_obj") if isinstance(st.get("manipulated_prompt_obj"), dict) else None
        if mpo:
            sys_txt = _extract_last_block(mpo.get("inputs",""), "system")
            if sys_txt:
                messages.append({"role":"system","content":sys_txt})
            break

    # user/assistant turns from each step
    for st in steps:
        mpo = st.get("manipulated_prompt_obj") if isinstance(st.get("manipulated_prompt_obj"), dict) else None
        if mpo:
            u = _extract_last_block(mpo.get("inputs",""), "user") or ""
            if u.strip(): messages.append({"role":"user","content":u})
        rr = st.get("raw_response_obj") if isinstance(st.get("raw_response_obj"), dict) else {}
        a = (rr.get("clem_player") or {}).get("response") \
            or _extract_last_block(rr.get("response",""), "assistant") \
            or ""
        if a.strip(): messages.append({"role":"assistant","content":a.strip()})

    # collapse consecutive same-role
    merged=[]
    for m in messages:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"]
        else:
            merged.append(m)
    return merged if any(m["role"]=="assistant" for m in merged) else None

## scan and build 

def detect_game_from_path(ep_dir: Path) -> str:
    try:
        return ep_dir.parent.parent.name.lower()
    except Exception:
        return ep_dir.parent.name.lower()

def model_from_modeldir(model_dirname: str) -> str:
    return (model_dirname.split("--",1)[0]).strip()

def main() -> None:
    if not RUNS.exists():
        print(f"Runs folder not found: {RUNS}", file=sys.stderr)
        sys.exit(1)

    written = 0
    counts: Dict[str,int] = {}

    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        for model_dir in sorted([p for p in RUNS.iterdir() if p.is_dir()]):
            if not MODEL_PAT.search(model_dir.name):
                continue
            for root, dirs, files in os.walk(model_dir):
                if not os.path.basename(root).startswith("episode_"):
                    continue
                ep_dir = Path(root)
                if not (ep_dir.joinpath("scores.json").exists() and ep_dir.joinpath("requests.json").exists()):
                    continue

                scores = read_json(ep_dir / "scores.json")
                game = detect_game_from_path(ep_dir)
                if not is_perfect_for(game, scores):
                    continue

                messages = turns_from_requests(ep_dir)
                if not messages:
                    continue

                meta = {
                    "game": game,
                    "instance_id": ep_dir.name,
                    "model": model_from_modeldir(model_dir.name),
                    "source_file": str(ep_dir),
                }
                obj = {"messages": messages, "meta": meta}
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                counts[game] = counts.get(game, 0) + 1

    with STATS.open("w", encoding="utf-8") as f:
        f.write("game,count\n")
        for g,c in sorted(counts.items(), key=lambda x:(-x[1], x[0])):
            f.write(f"{g},{c}\n")

    if written == 0:
        print("No perfect episodes found for the chosen models. Double-check runs/ content.")
    else:
        print(f"Wrote {written} episodes → {OUT_JSONL}")
        print(f"Per-game counts → {STATS}")

if __name__ == "__main__":
    main()
