"""Distributed self-play utilities using multiprocessing."""
from multiprocessing import Pool
from typing import Any

from .self_play import SelfPlay


def _run_hand(args: dict[str, Any]) -> list[Any]:
    sp: SelfPlay = args["self_play"]
    return sp.play_hand_for_training()


class DistributedSelfPlay:
    def __init__(self, cfr_trainer, game_engine_config: dict[str, Any]):
        self.cfr_trainer = cfr_trainer
        self.game_engine_config = game_engine_config

    def run(self, num_hands: int, num_workers: int = 2) -> list[list[Any]]:
        """Execute multiple hands in parallel and collect results."""
        hands_per_worker = [num_hands // num_workers for _ in range(num_workers)]
        for i in range(num_hands % num_workers):
            hands_per_worker[i] += 1
        tasks = []
        for count in hands_per_worker:
            sp = SelfPlay(self.cfr_trainer, self.game_engine_config)
            for _ in range(count):
                tasks.append({"self_play": sp})
        with Pool(processes=num_workers) as pool:
            results = pool.map(_run_hand, tasks)
        return results
