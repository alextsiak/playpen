from pathlib import Path
from datasets import load_dataset
from peft import LoraConfig
import trl

from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry
from playpen import BasePlayPen

DATA_JSONL = Path.home() / "playpen/examples/trl/results.jsonl"

def _infer_model_name(learner: HuggingfaceLocalModel) -> str:
    spec = getattr(learner, "model_spec", None) or {}
    for k in ("model_name", "huggingface_id", "id_or_path"):
        v = spec.get(k) if isinstance(spec, dict) else None
        if v:
            return str(v).split("/")[-1]
    name = getattr(getattr(learner, "model", None), "config", None)
    name = getattr(name, "_name_or_path", None)
    if name:
        return str(name).split("/")[-1]
    return "unknown-model"

class PeftSftTrainer(BasePlayPen):
    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)

    def learn(self, game_registry: GameRegistry):
        ds_all = load_dataset("json", data_files=str(DATA_JSONL))["train"]
        ds = ds_all.train_test_split(test_size=0.1, seed=42)

        model_name = _infer_model_name(self.learner)
        out_dir = Path("models") / "sft+lora" / model_name

        config = trl.SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=10,
            eval_strategy="epoch", 
            save_strategy="epoch",
            bf16=True,            
            max_length=2048,    
            packing=False         
        )


        tok = self.learner.tokenizer
        def formatting_func(example):
            return tok.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False  # learn to continue full dialogs
            )

        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            args=config,
            peft_config=LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_tokens"],
                task_type="CAUSAL_LM",
            ),
            formatting_func=formatting_func  # keep this; no dataset_text_field needed
        )

        trainer.train()

        trainer.model.save_pretrained(str(out_dir / "peft"))
        tok.save_pretrained(str(out_dir / "peft"))

        try:
            merged = trainer.model.merge_and_unload()
            merged_dir = out_dir / "merged"
            merged.save_pretrained(str(merged_dir))
            tok.save_pretrained(str(merged_dir))
            print(f"Merged model saved to: {merged_dir}")
        except Exception as e:
            print("Skipping merge (ok for first run):", e)

