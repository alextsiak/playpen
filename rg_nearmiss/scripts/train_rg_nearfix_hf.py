import pathlib, json, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE = "runs/models/sft+lora/Meta-Llama-3.1-8B-Instruct/merged"
DATA = "datasets/rg_nearmiss.jsonl"
OUT  = "runs/models/sft+nearfix_hf"

assert pathlib.Path(BASE).exists(), f"Missing base: {BASE}"
assert pathlib.Path(DATA).exists(), f"Missing dataset: {DATA}"
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

print("Loading dataset…")
ds_all = load_dataset("json", data_files=DATA, split="train")

def to_text(example):
    # messages = [{"role":"user","content":...},{"role":"assistant","content":...}]
    msgs = example.get("messages", [])
    u = next((m["content"] for m in msgs if m.get("role")=="user"), "")
    a = next((m["content"] for m in msgs if m.get("role")=="assistant"), "")
    return f"User: {u}\nAssistant: {a}", f"User: {u}\nAssistant:"

n = len(ds_all)
val = max(1, int(0.1*n)) if n > 10 else 1
ds_train = ds_all.select(range(n - val))
ds_eval  = ds_all.select(range(n - val, n))
print(f"Train/Eval sizes: {len(ds_train)}/{len(ds_eval)}")

print("Loading tokenizer/model (QLoRA)…")
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base = AutoModelForCausalLM.from_pretrained(
    BASE, device_map="auto", quantization_config=bnb,
)
base = prepare_model_for_kbit_training(base)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(base, peft_cfg)

MAXLEN = 1024

def tokenize_with_mask(example):
    full, prefix = to_text(example)
    # tokenize without special tokens so prefix length aligns
    ids_full   = tok(full,   add_special_tokens=False, max_length=MAXLEN, truncation=True)["input_ids"]
    ids_prefix = tok(prefix, add_special_tokens=False)["input_ids"]

    # If truncation cut the assistant, cap prefix to available length
    pref_len = min(len(ids_prefix), len(ids_full))
    # labels: mask prefix with -100, learn only completion
    labels = [-100]*pref_len + ids_full[pref_len:]

    attn = [1]*len(ids_full)
    return {"input_ids": ids_full, "attention_mask": attn, "labels": labels}

ds_train_tok = ds_train.map(tokenize_with_mask, remove_columns=ds_train.column_names, desc="tokenize-train")
ds_eval_tok  = ds_eval.map(tokenize_with_mask,  remove_columns=ds_eval.column_names,  desc="tokenize-eval")

def collate(batch):
    # simple right-padding collator for causal LM with label masking
    maxlen = max(len(x["input_ids"]) for x in batch)
    def pad(seq, pad_id): return seq + [pad_id]*(maxlen - len(seq))
    input_ids      = [pad(x["input_ids"], tok.pad_token_id) for x in batch]
    attention_mask = [pad(x["attention_mask"], 0)           for x in batch]
    labels         = [pad(x["labels"], -100)                 for x in batch]
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels),
    }

args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=1,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,  # effective batch 32
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",           # use 'eval_strategy' for newer Transformers
    bf16=True,
    optim="paged_adamw_8bit",        # if this errors on your setup, switch to "adamw_torch"
    gradient_checkpointing=True,
    lr_scheduler_type="linear",
    warmup_ratio=0.03,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds_train_tok,
    eval_dataset=ds_eval_tok,
    data_collator=collate,
)

print("Starting training…")
trainer.train()
print("Training done. Saving adapters…")
trainer.save_model(OUT)  # saves LoRA adapters

print("Merging adapters into base and saving merged checkpoint…")
merged_dir = pathlib.Path(OUT) / "merged"
merged_dir.mkdir(parents=True, exist_ok=True)
merged = trainer.model.merge_and_unload()
merged.save_pretrained(str(merged_dir))
tok.save_pretrained(str(merged_dir))
print("All saved under:", OUT, "and", merged_dir)
