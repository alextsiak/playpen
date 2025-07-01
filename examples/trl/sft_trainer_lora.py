from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig
from datasets import load_dataset
from datasets import concatenate_datasets

from playpen import BasePlayPen
from collections import Counter
import torch

from playpen.curriculum import DIFFICULTY_BUCKETS

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

        full_dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")

        full_dataset = full_dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")


        for stage_idx, (difficulty_lvl, games) in enumerate(DIFFICULTY_BUCKETS.items()):
            print(f"|||Currently training on {difficulty_lvl}")

            stage_dataset = full_dataset.filter(
                lambda ep: (ep["meta"]["game"], ep["meta"]["experiment"])
                           in games
            )

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

            # Initialize training configuration
            config = trl.SFTConfig(  # inherits TrainingArguments
                max_length=300,
                output_dir=output_dir,
                eval_strategy="steps",
                max_steps=10,
                logging_steps=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4
            )


            
            # Initialize trainer context
            trainer = trl.SFTTrainer(
                model=self.learner.model,
                train_dataset=stage_dataset["train"],
                eval_dataset=stage_dataset["test"],
                args=config,
                # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
                peft_config=LoraConfig(
                    r=8, lora_alpha=16,
                    lora_dropout=0.05,
                    target_modules="all-linear",
                    # target_modules=["q_proj", "v_proj"],
                    modules_to_save=["lm_head", "embed_token"],
                    task_type="CAUSAL_LM"
                )
            )
            
            torch.cuda.empty_cache()
            # Train on the dataset; this will save only the adapters to the checkpoints directory
            trainer.train()

            # Update model with latest adapter weights before next stage
            self.learner.model = trainer.model


        # Optional: Uncomment these lines to merge and save directly
        merged_model = trainer.model.merge_and_unload()
        merged_model.save_pretrained(f"models/sft+lora/{self.learner.get_name()}")
