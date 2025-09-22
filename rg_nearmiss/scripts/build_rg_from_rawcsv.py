#!/usr/bin/env python3
import argparse, csv, json, re, hashlib
from pathlib import Path
RG_WORDS={"first","second","third"}
IDX2={"0":"first","1":"second","2":"third",0:"first",1:"second",2:"third"}
STRICT=("Output exactly one line in the form `Answer: first` or `Answer: second` or `Answer: third`. "
        "Do not include anything else.")
def norm(s): return re.sub(r"\s+"," ", (s or "").strip().lower())
def htrip(game,user,asst): return hashlib.sha256((norm(game)+"\x1f"+norm(user)+"\x1f"+norm(asst)).encode()).hexdigest()
def pick_label(row):
    lk={k.lower():k for k in row}
    def get(keys): 
        for k in keys:
            if k in lk: return row[lk[k]]
        return None
    txt=get(["gold","target","solution","label","answer","gold_label","target_label"])
    if txt:
        m=re.search(r"\b(first|second|third)\b", norm(str(txt)))
        if m: return m.group(1)
    idx=get(["gold_index","target_index","solution_index","solution_idx","target_idx","gold_idx","index","idx"])
    if idx is not None:
        s=str(idx).strip()
        if s in IDX2: return IDX2[s]
        try:
            i=int(float(s))
            if i in IDX2: return IDX2[i]
        except: pass
    if txt is not None and str(txt).strip() in IDX2: return IDX2[str(txt).strip()]
    return None
def is_rg(row):
    for k,v in row.items():
        if k.lower() in {"game","main_game","task","benchmark"} and "referencegame" in str(v).lower():
            return True
    return True
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--out", default="datasets/rg_from_csv.jsonl")
    ap.add_argument("--perfect", default="datasets/perfect_episodes.jsonl")
    ap.add_argument("--max_items", type=int, default=100000)
    a=ap.parse_args()
    seen=set()
    P=Path(a.perfect)
    if P.exists():
        for line in P.open():
            try:
                ex=json.loads(line)
                seen.add(htrip(ex.get("game",""), ex.get("user") or ex.get("prompt",""), ex.get("assistant") or ex.get("completion","")))
            except: pass
    raws=[]
    for r in a.roots:
        rp=Path(r)
        if not rp.exists(): continue
        raws+=sorted(rp.rglob("raw.csv"))
    outp=Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    n=0
    with outp.open("w", encoding="utf-8") as wf:
        for rcsv in raws:
            try:
                with rcsv.open(newline="") as f:
                    R=csv.DictReader(f)
                    for i,row in enumerate(R):
                        if n>=a.max_items: break
                        if not is_rg(row): continue
                        lb=pick_label(row)
                        if lb not in RG_WORDS: continue
                        rid=hashlib.sha1(f"{rcsv}:{i}".encode()).hexdigest()[:10]
                        user=f"{STRICT}\n[id: rawcsv:{rid}]"
                        asst=f"Answer: {lb}"
                        k=htrip("referencegame", user, asst)
                        if k in seen: continue
                        seen.add(k)
                        wf.write(json.dumps({"game":"referencegame","user":user,"assistant":asst,"source":str(rcsv)}, ensure_ascii=False)+"\n")
                        n+=1
            except Exception as e:
                print(f"[warn] {rcsv}: {e}")
    print(f"[ok] RG-from-CSV → {outp} (total={n})")
if __name__=="__main__": main()
