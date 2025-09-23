import argparse, json, random
from pathlib import Path
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

EXPLICIT_TARGETS = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

def load_jsonl(paths):
    rows=[]
    for p in paths:
        for line in Path(p).open():
            ex=json.loads(line)
            u=ex.get("user") or ex.get("prompt")
            a=ex.get("assistant") or ex.get("completion")
            if not u or not a: continue
            rows.append({"user":u,"assistant":a,"game":ex.get("game","")})
    return rows

def split(rows, seed=42, ratio=0.1):
    by={}
    for r in rows: by.setdefault(r["game"],[]).append(r)
    rnd=random.Random(seed)
    tr,va=[],[]
    for g,arr in by.items():
        rnd.shuffle(arr)
        k=max(1,int(round(len(arr)*ratio)))
        va+=arr[:k]; tr+=arr[k:]
    return tr,va

def chatfmt(tok, u, a):
    if hasattr(tok,"apply_chat_template"):
        msgs=[{"role":"user","content":u},{"role":"assistant","content":a}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return f"User: {u}\nAssistant: {a}"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--base", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--peft_out", default="runs/models/sft+tulu3-2025-09/peft")
    ap.add_argument("--merged_out", default="runs/models/sft+tulu3-2025-09/merged")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eff_batch", type=int, default=128)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    args=ap.parse_args()

    rows=load_jsonl(args.data)
    tr,va=split(rows, seed=args.seed, ratio=0.10)

    tok=AutoTokenizer.from_pretrained(args.base, use_fast=True)
    tok.pad_token = tok.eos_token

    def mk(rows):
        texts=[chatfmt(tok, r["user"], r["assistant"]) for r in rows]
        return Dataset.from_dict({"text":texts})
    ds_tr, ds_va = mk(tr), mk(va)

    torch_dtype=torch.bfloat16
    model=AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch_dtype, device_map="auto")
    model.config.use_cache=False
    model.gradient_checkpointing_enable()

    try:
        lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                        target_modules=EXPLICIT_TARGETS, task_type=TaskType.CAUSAL_LM)
        model=get_peft_model(model, lora)
    except Exception as e:
        print("[warn] explicit target_modules failed; falling back to all-linear:", e)
        lora=LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                        target_modules="all-linear", task_type=TaskType.CAUSAL_LM)
        model=get_peft_model(model, lora)

    pdt=args.per_device_train_batch_size
    gas=max(1, args.eff_batch // pdt)

    targs=TrainingArguments(
        output_dir=args.peft_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=pdt,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gas,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        max_grad_norm=1.0
    )

    trainer=SFTTrainer(
        model=model, tokenizer=tok,
        train_dataset=ds_tr, eval_dataset=ds_va,
        args=targs,
        max_seq_length=args.seq_len,
        packing=True, dataset_text_field="text"
    )
    trainer.train()

    Path(args.peft_out).mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.peft_out); tok.save_pretrained(args.peft_out)

    merged=trainer.model.merge_and_unload()
    Path(args.merged_out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.merged_out); tok.save_pretrained(args.merged_out)

    print(f"[ok] adapters → {args.peft_out}")
    print(f"[ok] merged   → {args.merged_out}")

if __name__=="__main__":
    main()
