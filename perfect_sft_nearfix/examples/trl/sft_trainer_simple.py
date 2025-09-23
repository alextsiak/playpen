from clemcore.backends.huggingface_local_api import HuggingfaceLocalModel
from clemcore.clemgame import GameRegistry

import trl
import yaml
from datasets import load_dataset

from playpen import BasePlayPen



class SimpleSftTrainer(BasePlayPen):

    def __init__(self, learner: HuggingfaceLocalModel):
        super().__init__(learner)

    def learn(self, game_registry: GameRegistry):
        dataset = load_dataset(path="./results_Mistral-Small-24B-Instruct-2501/results.jsonl")

        dataset = dataset.filter(lambda episode: episode["meta"]["outcome"] == "success")

        dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)

        config = trl.SFTConfig(  # inherits TrainingArguments
            max_length=300,
            output_dir=f"models/sft/{self.learner.get_name()}",
            eval_strategy="epoch"
        )

        trainer = trl.SFTTrainer(
            model=self.learner.model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],  # Note: we use a subset of train as dev
            args=config
        )

        trainer.train()
