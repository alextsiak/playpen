from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
from peft import LoraConfig
from datasets import load_dataset

from playpen import BasePlayPen


class PeftSftTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)

    def learn(self, game_registry: GameRegistry):
        dataset = load_dataset("colab-potsdam/playpen-data", "interactions", split="train")

        dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")

        dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)

        config = trl.SFTConfig(  
            max_length=300,
            output_dir=f"models/sft+lora/{self.learner.get_name()}",
            eval_strategy="epoch"
        )

        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=config,
            # see https://huggingface.co/docs/trl/sft_trainer#training-adapters
            peft_config=LoraConfig(
                r=16, lora_alpha=32,
                lora_dropout=0.05,
                target_modules="all-linear",
                modules_to_save=["lm_head", "embed_token"],
                task_type="CAUSAL_LM",
            )
        )

        trainer.train()

