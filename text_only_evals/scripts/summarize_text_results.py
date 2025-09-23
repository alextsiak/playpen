import sys, json, pathlib

def clamp01(x):
    try:
        x=float(x)
        if x>1.0: x/=100.0
        return min(max(x,0.0),1.0)
    except Exception:
        return float('nan')

def summarize_one(root):
    root = pathlib.Path(root)
    per_game={}
    for f in root.rglob("*/episode_*/scores.json"):
        parts=f.parts
        try:
            ridx = parts.index(root.name)
            game = parts[ridx+2]
        except ValueError:
            continue
        data = json.loads(f.read_text())
        es = data.get("episode scores") or data.get("episode_scores") or {}
        aborted = bool(es.get("Aborted", 0))
        played = not aborted
        ms = es.get("Main Score")
        q = clamp01(ms) if isinstance(ms,(int,float)) else (1.0 if es.get("Success") else 0.0)
        g = per_game.setdefault(game, {"played":0,"total":0,"q_sum":0.0,"q_cnt":0})
        g["total"]+=1
        if played:
            g["played"]+=1; g["q_sum"]+=q; g["q_cnt"]+=1
    per_game = {k:v for k,v in per_game.items() if k in {"taboo","referencegame"}}
    mp_list=[]; mq_list=[]
    lines=[]
    for game in sorted(per_game):
        d=per_game[game]
        played_rate = (d["played"]/d["total"]) if d["total"] else 0.0
        qual = (d["q_sum"]/d["q_cnt"]) if d["q_cnt"] else 0.0
        mp_list.append(played_rate); mq_list.append(qual)
        lines.append(f"{game:15s} played {d['played']:3d}/{d['total']:3d} ({played_rate*100:6.2f}%)  quality {qual*100:6.2f}%")
    mp = sum(mp_list)/len(mp_list) if mp_list else 0.0
    mq = sum(mq_list)/len(mq_list) if mq_list else 0.0
    clemscore = (mp*100)*(mq*100)/100.0
    return lines, mp*100, mq*100, clemscore

def main(args):
    for root in args:
        lines, mp, mq, cs = summarize_one(root)
        print(f"=== {root} (TEXT-ONLY) ===")
        for line in lines: print(line)
        print(f"-> Macro %played {mp:.2f}% | Macro quality {mq:.2f}% | clemscore {cs:.2f}\n")
    if len(args)>=4:
        _, mp_s0, mq_s0, cs_s0 = summarize_one(args[0])
        _, mp_s7, mq_s7, cs_s7 = summarize_one(args[1])
        _, mp_b0, mq_b0, cs_b0 = summarize_one(args[2])
        _, mp_b7, mq_b7, cs_b7 = summarize_one(args[3])
        print("=== DELTAS (SFT - BASE) ===")
        print(f"T=0.0: Δplayed={mp_s0-mp_b0:+.2f}pp  Δquality={mq_s0-mq_b0:+.2f}pp  Δclemscore={cs_s0-cs_b0:+.2f}")
        print(f"T=0.7: Δplayed={mp_s7-mp_b7:+.2f}pp  Δquality={mq_s7-mq_b7:+.2f}pp  Δclemscore={cs_s7-cs_b7:+.2f}")

if __name__=='__main__':
    main(sys.argv[1:] or ["results_sft_t00","results_sft_t07","results_base_t00","results_base_t07"])
