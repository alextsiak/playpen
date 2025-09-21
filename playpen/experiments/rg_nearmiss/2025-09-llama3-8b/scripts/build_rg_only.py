#!/usr/bin/env python3
import argparse, json, re, hashlib, os
from pathlib import Path

RG_VALID={"first","second","third"}
INT2LBL={0:"first",1:"second",2:"third"}

def norm(s): return re.sub(r"\s+"," ", (s or "").strip().lower())

def deep_label_any(obj):
    if isinstance(obj, dict):
        for k,v in obj.items():
            kl=norm(k)
            if isinstance(v,int) and v in INT2LBL and any(x in kl for x in ["gold","target","solution","index","idx"]):
                return INT2LBL[v]
            if isinstance(v,str):
                vs=norm(v)
                for w in RG_VALID:
                    if vs==w or f" {w} " in f" {vs} ":
                        return w
            if isinstance(v,(dict,list)):
                r=deep_label_any(v)
                if r: return r
    elif isinstance(obj, list):
        for it in obj:
            r=deep_label_any(it)
            if r: return r
    return None

def read_json_safe(p: Path):
    try: return json.loads(p.read_text())
    except Exception: return None

def episode_stable_id(ep: Path):
    # Use experiment folder + episode name + tiny hash of episode.json/input.json if present
    parts = ep.parts
    exp = parts[-2] if len(parts)>=2 else "exp"
    epi = parts[-1]
    payload = ""
    for name in ("episode.json","input.json"):
        f = ep/name
        if f.exists():
            try:
                payload += json.dumps(json.loads(f.read_text()), sort_keys=True)
            except Exception:
                pass
    h = hashlib.sha1(payload.encode()).hexdigest()[:8] if payload else hashlib.sha1(str(ep).encode()).hexdigest()[:8]
    return f"{exp}/{epi}#{h}"

STRICT_LINE = "Output exactly one line in the form `Answer: first` or `Answer: second` or `Answer: third`. Do not include anything else."

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="datasets/rg_nearmiss_expanded.jsonl")
    ap.add_argument("--max_items", type=int, default=1000)
    args=ap.parse_args()

    # Collect episode dirs
    eps=[]
    for root in args.roots:
        rootp=Path(root)
        if not rootp.exists(): continue
        # Typical layout
        eps += [p for p in rootp.rglob("referencegame/*/episode_*") if p.is_dir()]
        # Fallback layout
        eps += [p for p in rootp.rglob("episode_*") if p.is_dir()]
    # De-dup episode dirs
    seen_dirs=set()
    uniq=[]
    for ep in sorted(eps):
        if ep in seen_dirs: continue
        seen_dirs.add(ep); uniq.append(ep)

    outp=Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    written=0
    with outp.open("w", encoding="utf-8") as wf:
        for ep in uniq:
            if written>=args.max_items: break
            # merge all JSONs to search for gold
            label=None
            for p in ep.glob("*.json"):
                j=read_json_safe(p)
                if j is None: continue
                r=deep_label_any(j)
                if r in RG_VALID:
                    label=r; break
            if label not in RG_VALID:
                continue
            eid=episode_stable_id(ep)
            user=f"{STRICT_LINE}\n[id: {eid}]"
            target=f"Answer: {label}"
            ex={"game":"referencegame","user":user,"assistant":target,"source":str(ep)}
            wf.write(json.dumps(ex, ensure_ascii=False)+"\n")
            written+=1

    print(f"[ok] RG dataset → {outp} (total={written})")

if __name__=="__main__":
    main()
