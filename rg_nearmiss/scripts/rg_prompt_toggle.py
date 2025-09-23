import sys, shutil
from pathlib import Path

P = Path("clembench/referencegame/resources/initial_prompts/en")
ACTIVE = P/"player_b_prompt_header.template"
BACKUP = P/"player_b_prompt_header.template.orig"
STRICT = 'Output exactly one line in the form "Answer: first" or "Answer: second" or "Answer: third". Do not include anything else.\n'

def on():
    if not BACKUP.exists():
        shutil.copy2(ACTIVE, BACKUP)
    src = BACKUP.read_text(encoding="utf-8")
    if src.startswith(STRICT):
        content = src  # already strict in backup (unlikely)
    else:
        content = STRICT + src
    ACTIVE.write_text(content, encoding="utf-8")
    print("[strict=ON] wrote strict follower header:", ACTIVE)

def off():
    if BACKUP.exists():
        shutil.copy2(BACKUP, ACTIVE)
        print("[strict=OFF] restored original:", ACTIVE)
    else:
        print("[strict=OFF] no backup found; leaving current header as is")

def status():
    txt = ACTIVE.read_text(encoding="utf-8")
    print("ON" if txt.startswith(STRICT) else "OFF")

if __name__=="__main__":
    cmd = sys.argv[1] if len(sys.argv)>1 else "status"
    {"on":on, "off":off, "status":status}.get(cmd, status)()
