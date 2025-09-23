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

def load_config(config_path="training_config.yaml"):
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

        difficulties = {
            "easy": {
                "adventuregame": ["home_deliver_three_basic_easy", "home_deliver_three_basic_easy_invlimittwo"],
                "codenames": [
                    "easy", "none","unambiguous", "concrete"
                ],
                "guesswhat": ["Level_1", "Abs_Level_1"],
                "imagegame": ["compact_grids"],
                "matchit_ascii": ["same_grid"],
                "reference_game": ["letter_grids", "number_grids", "line_grids_rows"],
                "taboo": ["high_en"],
                "textmapworld": ["small"],
                "textmap_graphreasoning": ["small"],
                "textmap_specificroom": ["on"],
                "wordle": ["high_frequency_words_no_clue_no_critic"]
            },
            "medium": {
                "adventuregame": ["home_deliver_three_planning_easy", "home_deliver_three_planning_easy_invlimittwo"],
                "codenames": [
                    "high", "difficult", "low",
                    "ambiguous", "abstract"
                ],
                "guesswhat": ["Level_2", "Abs_Level_2"],
                "imagegame": ["random_grids"],
                "matchit_ascii": ["similar_grid_1", "similar_grid_2"],
                "privateshared": [
                    "travel-booking", "job-interview", "restaurant",
                    "things-places", "letter-number"
                ],
                "reference_game": ["line_grids_columns"],
                "taboo": ["medium_en"],
                "textmapworld": ["medium"],
                "textmap_graphreasoning": ["medium"],
                "textmap_specificroom": ["close"],
                "wordle": ["medium_frequency_words_no_clue_no_critic"],
                "wordle_withclue": ["high_frequency_words_clue_no_critic"],
                "wordle_withcritic": ["high_frequency_words_clue_with_critic"]
            },
            "hard": {
                "adventuregame": [
                    "home_deliver_three_basic_hard", "home_deliver_three_planning_hard",
                    "home_deliver_three_basic_hard_invlimittwo", "home_deliver_three_planning_hard_invlimittwo"
                ],
                "guesswhat": ["Level_3", "Abs_Level_3"],
                "matchit_ascii": ["different_grid"],
                "reference_game": ["random_grids"],
                "taboo": ["low_en"],
                "textmapworld": ["large", "medium_cycle", "large_cycle"],
                "textmap_graphreasoning": ["large"],
                "textmap_specificroom": ["far"],
                "wordle_withclue": ["medium_frequency_words_clue_no_critic"],
                "wordle_withcritic": ["medium_frequency_words_clue_with_critic"]
            }
        }

        full_dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")

        full_dataset = full_dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")

        
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
        for stage_idx, (difficulty_lvl, games) in enumerate(difficulties.items()):
            print(f"|||Currently training on {difficulty_lvl}")

            stage_dataset = full_dataset.filter(lambda episode: episode["meta"]["game"] in games and
                                                episode["meta"]["experiment"] in games[episode["meta"]["game"]])
            
            print(f"Stage: {difficulty_lvl}, #Examples: {len(stage_dataset)}")
            print(stage_dataset[0]["meta"])

            from collections import Counter

            exp_counts = Counter(ep["meta"]["experiment"] for ep in stage_dataset)
            print(f"Experiment distribution in {difficulty_lvl}: {exp_counts}")


            stage_dataset = stage_dataset.train_test_split(0.2, shuffle=True, seed=42)

            print(f"{difficulty_lvl} train size: {len(stage_dataset['train'])}")
            print(f"{difficulty_lvl} test size: {len(stage_dataset['test'])}")

            print("Train sample:", stage_dataset["train"][0]["meta"]["experiment"])
            print("Test sample:", stage_dataset["test"][0]["meta"]["experiment"])

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
