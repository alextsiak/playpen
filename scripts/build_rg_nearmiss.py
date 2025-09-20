#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, re, hashlib, csv, sys
from pathlib import Path

RG_VALID = {"first","second","third"}
STRICT_NON_RG_ORDER = ["wordle_withcritic","wordle_withclue","wordle","private_shared","privateshared","matchit_ascii","matchit"]

def norm(s): return re.sub(r"\s+"," ", (s or "").strip().lower())
def hash_triplet(game, user, target):
    h = hashlib.sha256()
    h.update(norm(game).encode()); h.update(b"\x1f")
    h.update(norm(user).encode());  h.update(b"\x1f")
    h.update(norm(target).encode())
    return h.hexdigest()

def sniff_game(p: Path) -> str:
    s = str(p).lower()
    for g in ["referencegame","taboo","wordle_withcritic","wordle_withclue","wordle","private_shared","privateshared","matchit_ascii","matchit"]:
        if g in s: return g
    return "unknown"

def read_json(p: Path):
    try: return json.loads(p.read_text())
    except Exception: return {}

def extract_rg_gold(scores: dict):
    for k in ("gold","label","target","solution","reference","correct"):
        v = scores.get(k)
        if isinstance(v,str):
            vv = norm(v)
            for w in RG_VALID:
                if w in vv: return w
    for k in ("gt_index","target_index","solution_index","gold_index"):
        v = scores.get(k)
        try: i = int(v); return {0:"first",1:"second",2:"third"}.get(i)
        except Exception: pass
    for k in ("meta","details","score","metrics"):
        sub = scores.get(k, {})
        if isinstance(sub, dict):
            r = extract_rg_gold(sub)
            if r: return r
    return None

def rg_prompt_fallback():
    return ("You are Player B in CLEMbench Referencegame. "
            "Output exactly one line: Answer: first or Answer: second or Answer: third. "
            "Do not include anything else.")

def hunt_user_prompt(ep_dir: Path):
    for name in ["episode.json","episode_meta.json","play.jsonl","messages.jsonl","input.json","prompt.txt","log.jsonl"]:
        f = ep_dir / name
        if f.exists() and f.is_file():
            try:
                txt = f.read_text()
                m = re.search(r"Question:\s*Which grid.*?(first|second|third)", txt, re.I|re.S)
                if m:
                    start = max(0, m.start() - 1500); end = min(len(txt), m.end() + 1500)
                    return txt[start:end].strip()
            except Exception: pass
    return rg_prompt_fallback()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["results_sft_t00","results_sft_t07"])
    ap.add_argument("--out", default="datasets/rg_nearmiss_expanded.jsonl")
    ap.add_argument("--audit_csv", default="datasets/rg_nearmiss_audit.csv")
    ap.add_argument("--perfect", default="datasets/perfect_episodes.jsonl")
    ap.add_argument("--max_items", type=int, default=1000)
    ap.add_argument("--rg_min_fraction", type=float, default=0.70)
    args = ap.parse_args()

    perfect_hashes=set()
    P=Path(args.perfect)
    if P.exists():
        for line in P.open():
            try:
                ex=json.loads(line)
                u=ex.get("user") or ex.get("prompt",""); a=ex.get("assistant") or ex.get("completion","")
                perfect_hashes.add(hash_triplet(ex.get("game",""), u, a))
            except: pass

    eps=[]
    for root in args.roots:
        for s in Path(root).rglob("episode_*/scores.json"):
            game=sniff_game(s)
            sc=read_json(s)
            aborted=bool(sc.get("aborted", False))
            main_score=sc.get("main_score")
            try: wrong=(aborted or (main_score is not None and float(main_score)<1.0))
            except: wrong=True if aborted else False
            if not wrong: continue
            eps.append((game, s, sc))

    prio = {g:i for i,g in enumerate(["referencegame"]+STRICT_NON_RG_ORDER)}
    eps.sort(key=lambda x: (prio.get(x[0], 999), str(x[1])))

    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    aud=Path(args.audit_csv); aud.parent.mkdir(parents=True, exist_ok=True)
    seen=set(perfect_hashes)
    n_rg=n_other=written=0

    import csv
    with out.open("w") as jf, aud.open("w", newline="") as cf:
        w=csv.DictWriter(cf, fieldnames=["game","episode_dir","wrong_text","gold"])
        w.writeheader()
        for game, sp, sc in eps:
            if written>=args.max_items: break
            ep_dir=sp.parent
            if game=="referencegame":
                gold=extract_rg_gold(sc)
                if gold not in RG_VALID: continue
                user = hunt_user_prompt(ep_dir)
                target=f"Answer: {gold}"
            else:
                gold = sc.get("gold") or sc.get("target") or sc.get("solution")
                if not isinstance(gold, str): continue
                user = f"You are playing {game} in CLEMbench. Follow the exact output format required by the game rules."
                target = gold.strip()

            h=hash_triplet(game, user, target)
            if h in seen: continue
            seen.add(h)

            ex={"game":game, "user":user, "assistant":target, "source":str(ep_dir)}
            jf.write(json.dumps(ex, ensure_ascii=False)+"\n")
            wrong = sc.get("model_output") or sc.get("raw_output") or sc.get("answer") or ""
            w.writerow({"game":game,"episode_dir":str(ep_dir),"wrong_text":wrong,"gold":gold if isinstance(gold,str) else str(gold)})

            written+=1
            if game=="referencegame": n_rg+=1
            else: n_other+=1

    total=max(1, n_rg+n_other)
    frac=n_rg/total
    if frac<args.rg_min_fraction:
        sys.stderr.write(f"[warn] RG fraction {frac:.2f} < {args.rg_min_fraction:.2f}. Consider re-running with more RG sources.\n")

    ds_md = Path("datasets/DATASET.md")
    ds_md.parent.mkdir(parents=True, exist_ok=True)
    line = "- The **near-miss** set (`rg_nearmiss_expanded.jsonl`) is mined from my CLEMbench runs (SFT T=0.0/0.7) and inherits **CC-BY-4.0**.\n"
    if ds_md.exists():
        txt=ds_md.read_text()
        if "near-miss" not in txt: ds_md.write_text(txt.rstrip()+"\n\n"+line)
    else:
        ds_md.write_text("# Datasets\n\n"+line)

    print(f"[ok] near-miss → {out} (RG={n_rg}, other={n_other}, total={written})")
    print(f"[ok] audit     → {aud}")

if __name__=="__main__":
    main()
