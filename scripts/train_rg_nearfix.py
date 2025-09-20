import pathlib, torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
)
from trl import SFTTrainer
from peft import LoraConfig

BASE = "runs/models/sft+lora/Meta-Llama-3.1-8B-Instruct/merged"
DATA = "datasets/rg_nearmiss.jsonl"
OUT  = "runs/models/sft+nearfix"

assert pathlib.Path(BASE).exists(), f"Missing base: {BASE}"
assert pathlib.Path(DATA).exists(), f"Missing dataset: {DATA}"
pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

print("Loading dataset…")
ds_all = load_dataset("json", data_files=DATA, split="train")

# Convert your {messages:[{role:..., content:...}, ...]} to plain text
def to_text(ex):
    msgs = ex.get("messages", [])
    u = next((m["content"] for m in msgs if m.get("role")=="user"), "")
    a = next((m["content"] for m in msgs if m.get("role")=="assistant"), "")
    return f"User: {u}\nAssistant: {a}"

n = len(ds_all)
val = max(1, int(0.1*n)) if n>10 else 1
ds_train = ds_all.select(range(n - val))
ds_eval  = ds_all.select(range(n - val, n))
print(f"Train/Eval sizes: {len(ds_train)}/{len(ds_eval)}")

print("Loading tokenizer/model (QLoRA)…")
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    device_map="auto",
    quantization_config=bnb,
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none", task_type="CAUSAL_LM",
)

# NOTE: new Transformers expects eval_strategy (not evaluation_strategy)
args = TrainingArguments(
    output_dir=OUT,
    num_train_epochs=1,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,   # effective batch 32
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    lr_scheduler_type="linear",
    warmup_ratio=0.03,
    report_to=[],
)

def formatting_func(batch):
    return [to_text(ex) for ex in batch]

print("Starting training…")
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    peft_config=peft_cfg,
    max_seq_length=1024,
    packing=False,
    formatting_func=formatting_func,
    args=args,
)

trainer.train()
print("Training done. Saving adapters…")
trainer.save_model(OUT)

print("Merging adapters into base and saving merged checkpoint…")
merged_dir = pathlib.Path(OUT)/"merged"
merged_dir.mkdir(parents=True, exist_ok=True)
merged = trainer.model.merge_and_unload()
merged.save_pretrained(str(merged_dir))
tok.save_pretrained(str(merged_dir))
print("All saved under:", OUT, "and", merged_dir)
