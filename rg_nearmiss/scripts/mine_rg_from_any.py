import argparse, json, re, hashlib, sys
from pathlib import Path

RG_WORDS = {"first","second","third"}
IDX2WORD = {0:"first",1:"second",2:"third","0":"first","1":"second","2":"third"}
STRICT_LINE = ("Output exactly one line in the form `Answer: first` or `Answer: second` or `Answer: third`. "
               "Do not include anything else.")

def deep_gold(obj):
    """Return 'first'/'second'/'third' if found anywhere in obj."""
    if isinstance(obj, dict):
        for k,v in obj.items():
            lk = str(k).lower()
            # 1) direct text with cluey key names
            if isinstance(v, str):
                s = v.strip().lower()
                m = re.search(r"\b(first|second|third)\b", s)
                if m and any(n in lk for n in ["gold","target","solution","label","answer","correct","index","idx"]):
                    return m.group(1)
            # 2) numeric index with cluey key names
            if isinstance(v, (int, float)) and int(v) in (0,1,2):
                if any(n in lk for n in ["gold","target","solution","index","idx","correct"]):
                    return IDX2WORD[int(v)]
            # 3) recurse
            if isinstance(v,(dict,list)):
                g = deep_gold(v)
                if g: return g
    elif isinstance(obj, list):
        for it in obj:
            g = deep_gold(it)
            if g: return g
    return None

def norm(s:str) -> str:
    import re as _re
    return _re.sub(r"\s+"," ", (s or "").strip().lower())

def hash_triplet(game, user, target):
    return hashlib.sha256((norm(game)+"\x1f"+norm(user)+"\x1f"+norm(target)).encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="results_* directories to scan")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--perfect", default="datasets/perfect_episodes.jsonl", help="optional dedup file")
    ap.add_argument("--max_items", type=int, default=100000)
    ap.add_argument("--debug_miss", action="store_true", help="print up to 20 episode dirs with no gold found")
    args = ap.parse_args()

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    P = Path(args.perfect)
    if P.exists():
        for line in P.open():
            try:
                ex = json.loads(line)
                user = ex.get("user") or ex.get("prompt","")
                asst = ex.get("assistant") or ex.get("completion","")
                seen.add(hash_triplet(ex.get("game",""), user, asst))
            except Exception:
                pass

    eps = []
    for root in args.roots:
        rp = Path(root)
        if not rp.exists(): continue
        for ep in rp.rglob("referencegame/*/episode_*"):
            if ep.is_dir():
                eps.append(ep)
    eps = sorted(eps)
    if not eps:
        print(f"[warn] No episodes found under roots: {args.roots}", file=sys.stderr)

    written = 0
    misses = []
    with outp.open("w", encoding="utf-8") as wf:
        for ep in eps:
            if written >= args.max_items:
                break
            gold = None
            for jf in sorted(ep.glob("*.json")):
                try:
                    J = json.loads(jf.read_text())
                    gold = deep_gold(J)
                    if gold in RG_WORDS:
                        break
                except Exception:
                    continue
            if gold not in RG_WORDS:
                misses.append(str(ep))
                continue
            rid = hashlib.sha1(str(ep).encode()).hexdigest()[:10]
            user = f"{STRICT_LINE}\n[id: ep:{rid}]"
            target = f"Answer: {gold}"
            k = hash_triplet("referencegame", user, target)
            if k in seen:
                continue
            seen.add(k)
            wf.write(json.dumps({
                "game":"referencegame",
                "user": user,
                "assistant": target,
                "source": str(ep)
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"[ok] RG-from-episodes → {outp} (total={written})")
    if args.debug_miss and misses:
        print("[debug] episodes with no detectable gold (showing up to 20):")
        for m in misses[:20]:
            print("  -", m)

if __name__ == "__main__":
    main()
