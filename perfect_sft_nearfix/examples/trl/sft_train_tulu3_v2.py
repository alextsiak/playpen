#!/usr/bin/env python3
import os, json, random, argparse
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, PeftModel
import torch

def load_rows(p):
    rows=[]
    with open(p,"r",encoding="utf-8") as f:
        for ln in f:
            j=json.loads(ln)
            a=(j.get("assistant","") or "").strip()
            if a not in {"Answer: first","Answer: second","Answer: third"}:
                continue
            rows.append({"game": j.get("game","referencegame"),
                         "user": j["user"], "assistant": a})
    return rows

def make_text_field(rows, tok):
    out=[]
    for r in rows:
        msgs=[{"role":"user","content": r["user"]},
              {"role":"assistant","content": r["assistant"]}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        out.append({"game": r["game"], "text": txt})
    return out

def stratified_split(items, test_ratio=0.10, seed=42):
    random.seed(seed)
    bygame={}
    for x in items: bygame.setdefault(x["game"], []).append(x)
    train, test = [], []
    for g, xs in bygame.items():
        n=len(xs); k=max(1, int(round(n*test_ratio)))
        random.Random(seed).shuffle(xs)
        test.extend(xs[:k]); train.extend(xs[k:])
    return train, test

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--peft_out", required=True)
    ap.add_argument("--merged_out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--eff_batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    rows = load_rows(args.data)
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    # ---- critical: make sure we have a pad token ----
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    items = make_text_field(rows, tok)
    train_items, eval_items = stratified_split(items, 0.10, args.seed)
    print(f"[info] total={len(items)}  train={len(train_items)}  eval={len(eval_items)}")
    assert len(items) > 50, "Dataset too small; did filtering bite?"

    train_ds = Dataset.from_list(train_items)
    eval_ds  = Dataset.from_list(eval_items)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.config.pad_token_id = tok.pad_token_id
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM"
    )
    try:
        model = get_peft_model(model, lora)
    except Exception:
        lora.target_modules = "all-linear"
        model = get_peft_model(model, lora)

    world = int(os.environ.get("WORLD_SIZE", "1"))
    per_device_bs = 2
    accum = max(1, args.eff_batch // (per_device_bs * world))

    sft_cfg = SFTConfig(
        output_dir="runs/train_logs/sft_tulu3_v2",
        max_seq_length=args.seq_len,
        packing=False,
        dataset_text_field="text",
        eval_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        gradient_accumulation_steps=accum,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        bf16=True,
        lr_scheduler_type="linear",
        save_strategy="no",
        seed=args.seed
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg
    )
    trainer.train()

    Path(args.peft_out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.peft_out)
    tok.save_pretrained(args.peft_out)
    print(f"[ok] adapters → {args.peft_out}")

    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map="auto")
    base.config.pad_token_id = tok.pad_token_id
    merged = PeftModel.from_pretrained(base, args.peft_out).merge_and_unload()
    Path(args.merged_out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.merged_out)
    tok.save_pretrained(args.merged_out)
    print(f"[ok] merged   → {args.merged_out}")

if __name__ == "__main__":
    main()
