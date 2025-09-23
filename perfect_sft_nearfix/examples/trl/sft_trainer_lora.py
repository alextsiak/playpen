from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from datasets import concatenate_datasets

from playpen import BasePlayPen
from collections import Counter
import torch
import argparse
import yaml
import os
import wandb
import json

def load_config(config_path="examples/trl/training_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()


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


        dataset = load_dataset("json", data_files="./results_teachers/results.jsonl")
        dataset = dataset["train"]
        dataset_best = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")



        with open(config["failed_instances_path"], "r") as f:
            failed_instances = json.load(f)
        failed_set = {(x["game"], x["experiment"], x["task_id"]) for x in failed_instances}
        dataset_best = dataset_best.filter(
            lambda ep: (ep["meta"]["game"], ep["meta"]["experiment"], ep["meta"]["task_id"]) in failed_set)
        
        dataset_best = dataset_best.filter(lambda episode: episode["meta"]["model"] == "claude-3-5-sonnet-20241022" or episode["meta"]["model"] == "claude-3-5-sonnet-20250219" or episode["meta"]["model"] == "qwen-max")

        dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")
        dataset_best = dataset_best.filter(lambda episode: episode["meta"]["outcome"] == "success")

        dataset_best = dataset_best.cast(dataset.features)

        combined_dataset = concatenate_datasets([dataset, dataset_best])

        combined_dataset = combined_dataset.train_test_split(0.2, shuffle=True, seed=42)

        lora_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules="all-linear",
            modules_to_save=["lm_head", "embed_token"],
            task_type="CAUSAL_LM"
        )
        
        self.learner.model = get_peft_model(self.learner.model, lora_config)
    

        model_name = self.learner.model.config._name_or_path.replace("/", "_")


        config_trl = trl.SFTConfig(
                max_length=300,
                output_dir=f"models/sft+lora/{model_name}",
                eval_strategy="epoch",
                #max_steps=config["max_steps"],
                num_train_epochs = config["num_train_epochs"],
                logging_steps=10,
                per_device_train_batch_size=config["batch_size"],
                gradient_accumulation_steps=config["grad_accum_steps"],
                learning_rate=config["learning_rate"],
                report_to=["wandb"] if config.get("wandb", {}).get("enable", False) else []
            )


            
        trainer = trl.SFTTrainer(
                model=self.learner.model,
                train_dataset=combined_dataset["train"],
                eval_dataset=combined_dataset["test"],
                args=config_trl
                # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
            )

            
        trainer.train()

        if config.get("wandb", {}).get("enable", False):
            wandb.finish()

        save_path = f"models/sft+lora/{model_name}"
            
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        self.learner.tokenizer.save_pretrained(save_path)
