from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from pathlib import Path

from playpen import BasePlayPen
from collections import Counter
import torch
import yaml
import wandb
import json


def load_config(config_path="training_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


config = load_config()


def load_task_difficulties(dataset, fail_file: Path):
    """split dataset tasks into easy/hard sets based on failed_instances.json"""
    with open(fail_file, "r") as f:
        fail_tasks = json.load(f)

    def to_key(x):
        return (x["game"], x["experiment"], int(x["task_id"]))

    all_keys = {
        (ep["meta"]["game"], ep["meta"]["experiment"], int(ep["meta"]["task_id"]))
        for ep in dataset
    }
    hard_keys = {to_key(x) for x in fail_tasks}
    easy_keys = all_keys - hard_keys

    print(f"by task_id: {len(easy_keys)} easy tasks, {len(hard_keys)} hard tasks")
    return {"easy": easy_keys, "hard": hard_keys}


def filter_by_tasks(dataset, keys):
    def keep(ep):
        m = ep["meta"]
        return (m["game"], m["experiment"], int(m["task_id"])) in keys
    return dataset.filter(keep)


class PeftSftTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)
    

    def learn(self, game_registry: GameRegistry):
        if config.get("wandb", {}).get("enable", False):
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"].get("name", None),
                config=config
            )

        full_dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")
        full_dataset = full_dataset.filter(lambda ep: ep["meta"]["outcome"] == "success")

        fail_file = Path("./failures_llama3-8b/failed_instances.json") 
        difficulties = load_task_difficulties(full_dataset, fail_file)

        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"],
            task_type="CAUSAL_LM"
        )
        self.learner.model = get_peft_model(self.learner.model, lora_config)
        print(self.learner.model.print_trainable_parameters())

        for stage_idx, (difficulty_lvl, keys) in enumerate(difficulties.items()):
            print(f"Currently training on {difficulty_lvl}")

            stage_dataset = filter_by_tasks(full_dataset, keys)
            print(f"Stage {difficulty_lvl}: {len(stage_dataset)} examples")

            if len(stage_dataset) == 0:
                print(f"Skipping {difficulty_lvl} stage (empty)")
                continue

            exp_counts = Counter(ep["meta"]["experiment"] for ep in stage_dataset)
            print(f"Experiment distribution in {difficulty_lvl}: {exp_counts}")

            stage_dataset = stage_dataset.train_test_split(0.2, shuffle=True, seed=42)

            print(f"{difficulty_lvl} train size: {len(stage_dataset['train'])}")
            print(f"{difficulty_lvl} test size: {len(stage_dataset['test'])}")

            print("Train sample:", stage_dataset["train"][0]["meta"])
            print("Test sample:", stage_dataset["test"][0]["meta"])

            model_name = self.learner.model.config._name_or_path.replace("/", "_")
            output_dir = f"models/sft+lora/{model_name}/stage_{stage_idx}_{difficulty_lvl}"

            config_trl = trl.SFTConfig(
                max_length=300,
                output_dir=output_dir,
                eval_strategy="epoch",
                max_steps=config["max_steps"],
                logging_steps=1,
                per_device_train_batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                report_to=["wandb"] if config.get("wandb", {}).get("enable", False) else []
            )

            trainer = trl.SFTTrainer(
                model=self.learner.model,
                train_dataset=stage_dataset["train"],
                eval_dataset=stage_dataset["test"],
                args=config_trl
            )

            trainer.train()

            self.learner.model = trainer.model

        if config.get("wandb", {}).get("enable", False):
            wandb.finish()

        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(f"models/sft+lora/{self.learner}")
