import json, pathlib, math, sys
def pct(x): return 100.0*x
def clamp01(x):
    try:
        x=float(x)
        if x>1.0: x/=100.0
        return min(max(x,0.0),1.0)
    except: return float('nan')

def summarize(rootdir):
    root = pathlib.Path(rootdir)
    files = list(root.rglob("*/episode_*/scores.json"))
    if not files:
        print(f"[WARN] no scores under {rootdir}"); return 0.0,0.0,0.0
    per_game={}
    for f in files:
        parts=f.parts
        try:
            ridx=parts.index(root.name)
            game=parts[ridx+2]
        except:
            continue
        data=json.loads(f.read_text())
        es = data.get("episode scores") or data.get("episode_scores") or {}
        aborted = bool(es.get("Aborted", 0))
        played  = not aborted
        ms = es.get("Main Score")
        if isinstance(ms,(int,float)): q = clamp01(ms)
        else: q = 1.0 if es.get("Success") else 0.0
        g=per_game.setdefault(game,{"played":0,"total":0,"q_sum":0.0,"q_cnt":0})
        g["total"]+=1
        if played:
            g["played"]+=1; g["q_sum"]+=q; g["q_cnt"]+=1

    mp_list=[]; mq_list=[]
    print(f"\n=== SUMMARY for {rootdir} ===")
    for game,d in sorted(per_game.items()):
        played_rate = (d["played"]/d["total"]) if d["total"] else 0.0
        qual = (d["q_sum"]/d["q_cnt"]) if d["q_cnt"] else 0.0
        mp_list.append(played_rate); mq_list.append(qual)
        print(f"{game:15s}  played {d['played']:3d}/{d['total']:3d} ({pct(played_rate):6.2f}%)  quality {pct(qual):6.2f}%")
    if mp_list:
        mp=sum(mp_list)/len(mp_list); mq=sum(mq_list)/len(mq_list)
        clemscore = pct(mp)*pct(mq)/100.0
        print("-"*64)
        print(f"Macro %played: {pct(mp):.2f}%")
        print(f"Macro quality: {pct(mq):.2f}%")
        print(f"**clemscore  : {clemscore:.2f}**")
        return pct(mp), pct(mq), clemscore
    return 0.0,0.0,0.0

if __name__ == "__main__":
    for rd in sys.argv[1:]:
        summarize(rd)
