import abc
from typing import List, Tuple, Callable, Union, Dict

from clemcore.clemgame import Player


class PlayPenEnv(abc.ABC):

    def __init__(self):
        self._done: bool = False

    @property
    def initial_prompts(self):
        return dict()

    def is_done(self) -> bool:
        return self._done

    @abc.abstractmethod
    def reset(self) -> None:
        pass

    @abc.abstractmethod
    def observe(self) -> Tuple[Player | Callable, Dict | List[Dict]]:
        pass

    @abc.abstractmethod
    def step(self, response: str | List) -> Tuple[bool | List, Dict | List]:
        pass

    @abc.abstractmethod
    def store_records(self, top_dir: str, rollout_dir: str, episode_dir: str):
        """
        Stores the records in a similar structure as for running clembench,
        so that transcribe and scoring commands can be applied.
        """
        pass
