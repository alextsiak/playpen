#!/usr/bin/env python3
import argparse, json, re, hashlib, sys
from pathlib import Path

STRICT_LINE = ("Output exactly one line in the form `Answer: first` or "
               "`Answer: second` or `Answer: third`. Do not include anything else.")

def norm(s:str) -> str:
    return re.sub(r"\s+"," ", (s or "").strip().lower())

def key_hash(game,user,assistant):
    return hashlib.sha1((norm(game)+"\x1f"+norm(user)+"\x1f"+norm(assistant)).encode()).hexdigest()

def to_fst_snd_trd(val):
    if val is None: return None
    if isinstance(val, (int,float)):
        n=int(val)
        if n in (0,1,2): return ['first','second','third'][n]
        if n in (1,2,3): return ['first','second','third'][n-1]
        return None
    s=str(val).strip().lower()
    if s in {'first','1','a','i'}: return 'first'
    if s in {'second','2','b','ii'}: return 'second'
    if s in {'third','3','c','iii'}: return 'third'
    m=re.search(r'\b(first|second|third|[123]|[abc])\b', s)
    return to_fst_snd_trd(m.group(0)) if m else None

def extract_from_grouped_list(items, file_name):
    """
    items: list like [{'name': 'line_grids_rows', 'game_instances': [ {...}, ... ]}, ...]
    yield (src_id, gold)
    """
    total=0; ok=0
    for group_idx, group in enumerate(items):
        gi = group.get('game_instances') if isinstance(group, dict) else None
        group_name = group.get('name', f'group{group_idx}') if isinstance(group, dict) else f'group{group_idx}'
        if not isinstance(gi, list): continue
        for inst_idx, inst in enumerate(gi):
            gold = None
            # 1) explicit target_grid_name is usually ['first', 1] etc.
            tgn = inst.get('target_grid_name') if isinstance(inst, dict) else None
            if isinstance(tgn, list) and tgn:
                gold = to_fst_snd_trd(tgn[0])
            # 2) common index fields
            if gold is None:
                for k in ('target_grid_idx','target_grid_index','target_index','gold_index','answer_index'):
                    if k in inst:
                        gold = to_fst_snd_trd(inst[k]); break
            # 3) last resort: scan any value
            if gold is None:
                for v in inst.values():
                    g=to_fst_snd_trd(v)
                    if g: gold=g; break
            total += 1
            if gold in {'first','second','third'}:
                ok += 1
                src_id = f"{file_name}#{group_name}#{inst.get('game_id', inst_idx)}"
                yield src_id, gold
        # optional per-group progress to stderr
        print(f"[info] {file_name}:{group_name} -> {ok}/{total} extracted", file=sys.stderr)
    return

def iter_all_instances(file_path):
    """Return iterator of (src_id, gold) across schema variants."""
    data = json.loads(file_path.read_text(encoding='utf-8'))
    # Variant A: dict with 'instances' list
    if isinstance(data, dict):
        if isinstance(data.get('instances'), list):
            for i, it in enumerate(data['instances']):
                # try typical fields
                gold=None
                for k in ('target','gold','solution','answer','target_idx','target_index','gold_index','answer_index'):
                    if k in it:
                        gold = to_fst_snd_trd(it[k]); break
                if gold is None and 'target_grid_name' in it:
                    tg = it['target_grid_name']
                    gold = to_fst_snd_trd(tg[0] if isinstance(tg,list) else tg)
                if gold in {'first','second','third'}:
                    yield f"{file_path.name}#{i}", gold
            return
    # Variant B: top-level LIST of groups with 'game_instances' (v1.6/v2.0)
    if isinstance(data, list) and data and isinstance(data[0], dict) and 'game_instances' in data[0]:
        for src_id, gold in extract_from_grouped_list(data, file_path.name):
            yield src_id, gold
        return
    # Variant C: fallback — scan any list in dict
    if isinstance(data, dict):
        for k,v in data.items():
            if isinstance(v, list):
                if v and isinstance(v[0], dict) and 'game_instances' in v[0]:
                    for src_id, gold in extract_from_grouped_list(v, f"{file_path.name}:{k}"):
                        yield src_id, gold
                else:
                    for i, it in enumerate(v):
                        gold=None
                        for kk in ('target','gold','solution','answer','target_idx','target_index','gold_index','answer_index'):
                            if isinstance(it, dict) and kk in it:
                                gold = to_fst_snd_trd(it[kk]); break
                        if gold in {'first','second','third'}:
                            yield f"{file_path.name}:{k}#{i}", gold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench_root", default="clembench/referencegame", help="path to clembench/referencegame")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--perfect", default="datasets/perfect_episodes.jsonl", help="optional perfect set for dedup")
    ap.add_argument("--max_items", type=int, default=1000000)
    ap.add_argument("--include_non_en", action="store_true", help="also include non-English instances")
    args = ap.parse_args()

    in_dir = Path(args.bench_root) / "in"
    files = sorted(in_dir.glob("instances*.json"))
    if not args.include_non_en:
        files = [p for p in files if ("_en" in p.name) or (p.name == "instances.json")]
    if not files:
        print(f"[warn] no matching instances*.json in {in_dir}", file=sys.stderr)

    # dedup against perfect
    seen=set()
    P=Path(args.perfect)
    if P.exists():
        for line in P.open(encoding="utf-8"):
            try:
                ex=json.loads(line)
                seen.add(key_hash(ex.get("game",""), ex.get("user",""), ex.get("assistant","")))
            except: pass

    outp=Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    written=0
    with outp.open("w", encoding="utf-8") as wf:
        for fp in files:
            found_in_file=0
            for src_id, gold in iter_all_instances(fp):
                if written >= args.max_items: break
                user = f"{STRICT_LINE}\n[id: {src_id}]"
                target = f"Answer: {gold}"
                k = key_hash("referencegame", user, target)
                if k in seen: continue
                seen.add(k); found_in_file += 1; written += 1
                wf.write(json.dumps({"game":"referencegame","user":user,"assistant":target,"source":src_id}, ensure_ascii=False)+"\n")
            print(f"[info] {fp.name}: {found_in_file} items", file=sys.stderr)
            if written >= args.max_items: break
    print(f"[ok] RG-from-benchmark/in → {outp} (total={written}, files={len(files)})")

if __name__ == "__main__":
    main()
