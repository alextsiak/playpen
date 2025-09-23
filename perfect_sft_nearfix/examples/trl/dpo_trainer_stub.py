from typing import List, Dict

from clemcore.backends import Model
from clemcore.clemgame import GameRegistry
from playpen import BasePlayPen, RolloutProgressCallback, GameRecordCallback, make_tree_env, \
    BranchingRolloutBuffer, StepRolloutBuffer, make_env, GameEnv
from datasets import Dataset


class DPORolloutBuffer(BranchingRolloutBuffer):

    def to_preference_dataset(self, perspective: Model, data_format="conversational") -> Dataset:
        """
        Transform the branching rollout buffer to a preference dataset for, e.g., DPO learning.

        preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}

        preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                              "chosen": [{"role": "assistant", "content": "It is blue."}],
                              "rejected": [{"role": "assistant", "content": "It is green."}]}

        :param perspective: of a model generating the responses
        :param data_format: conversational or standard
        :return: a preference dataset as described in https://huggingface.co/docs/trl/dataset_formats#preference
        """
        return Dataset.from_list([])


class DPOPlayPenTrainer(BasePlayPen):
    """
    Then, fine-tuning a language model via DPO consists of two steps and is easier than PPO:
    (1) Data collection: Gather a preference dataset with positive and negative pairs of generation, given a prompt.
    (2) Optimization: Maximize the log-likelihood of the DPO loss directly.

    DPO requires a preference dataset. The DPOTrainer supports both conversational and standard dataset formats.
    When provided with a conversational dataset, the trainer will automatically apply the chat template to the dataset.

    See https://huggingface.co/docs/trl/dpo_trainer
    """

    def __init__(self, learner: Model):
        super().__init__(learner)
        self.rollout_steps = 16
        self.add_callback(RolloutProgressCallback(self.rollout_steps))
        self.add_callback(GameRecordCallback())

    def learn(self, game_registry: GameRegistry):
        def branch_on_guesser(env: GameEnv):
            player = env.master.current_player
            return self.is_learner(player) and player.game_role == "WordGuesser"

        game_spec = game_registry.get_game_specs_that_unify_with("taboo")[0]
        with make_tree_env(game_spec, [self.learner],
                           branching_factor=2,
                           branching_criteria=branch_on_guesser) as game_env:
            rollout_buffer = DPORolloutBuffer(game_env)
            self._collect_rollouts(game_env, self.rollout_steps, rollout_buffer)
            self._train(rollout_buffer)
            rollout_buffer.reset()

    def _train(self, rollout_buffer):
        dataset = rollout_buffer.to_conversational_dataset(self.learner)
        if len(dataset) > 0:
            print(dataset[0])
        print(f"There are {len(dataset)} conversations in the dataset")
        rollout_buffer.to_preference_dataset(self.learner)
        ...
