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
        # Note: We configure the proper chat template for the tokenizer already during model loading in the backend


    def learn(self, game_registry: GameRegistry):
        # Load a conversational dataset for SFT, that is, a list of "messages" -- basically tuples of role and content.
        # The role can be "user" or "assistant" and typically alternates within the list.
        # During training, everything up to the last assistant message becomes the prefix for prediction.
        # The loss is calculated based on the differences to the last assistant message.
        # Here we load the canonical training split as available in the huggingface playpen-data repository.
        # By default, the dataset is stored in ~/.cache/huggingface/datasets/ on your machine. This might take a while.


        if config.get("wandb", {}).get("enable", False):
            wandb.init(
                project=config["wandb"]["project"],
                name=config["wandb"].get("name", None),
                config=config
            )


        # local dataset
        dataset = load_dataset("json", data_files="./results_teachers/results.jsonl")
        dataset = dataset["train"]
        # add examples from best models (playpen-data dataset)
        dataset_best = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")



        # only get learner failed instances from dataset_best
        with open(config["failed_instances_path"], "r") as f:
            failed_instances = json.load(f)
        failed_set = {(x["game"], x["experiment"], x["task_id"]) for x in failed_instances}
        dataset_best = dataset_best.filter(
            lambda ep: (ep["meta"]["game"], ep["meta"]["experiment"], ep["meta"]["task_id"]) in failed_set)
        
        dataset_best = dataset_best.filter(lambda episode: episode["meta"]["model"] == "claude-3-5-sonnet-20241022" or episode["meta"]["model"] == "claude-3-5-sonnet-20250219" or episode["meta"]["model"] == "qwen-max")

        dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")
        dataset_best = dataset_best.filter(lambda episode: episode["meta"]["outcome"] == "success")

        #ensure same format for both datasets
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


        # Initialize training configuration
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


            
        # Initialize trainer context
        trainer = trl.SFTTrainer(
                model=self.learner.model,
                train_dataset=combined_dataset["train"],
                eval_dataset=combined_dataset["test"],
                args=config_trl
                # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
            )

            
        # Train on the dataset; this will save only the adapters to the checkpoints directory
        trainer.train()

        if config.get("wandb", {}).get("enable", False):
            wandb.finish()

        save_path = f"models/sft+lora/{model_name}"
            
        # Optional: Uncomment these lines to merge and save directly
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(save_path)
        self.learner.tokenizer.save_pretrained(save_path)