#!/usr/bin/env python3
import sys, platform
def v(m):
    try: return __import__(m).__version__
    except: return "N/A"
mods = ["torch","transformers","peft","accelerate","trl","datasets"]
print("python", sys.version.replace("\n"," "))
print("platform", platform.platform())
for m in mods: print(m, v(m))
try:
    import torch
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
except Exception:
    pass
