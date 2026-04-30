"""Distributed self-play utilities using multiprocessing."""

from __future__ import annotations

from multiprocessing import Pool
from typing import Any

from .self_play import SelfPlay


def _hand_result_summary(payload: Any) -> list[Any]:
    """Return a small pickle-safe summary for one generated hand.

    ``SelfPlay.play_hand_for_training`` returns the trainer replay buffer.  In
    normal trainers that buffer contains ``torch.Tensor`` instances.  Sending
    those tensors back from child processes forces PyTorch's shared-memory IPC
    path, which is not available in all sandboxed or CI environments.  The
    distributed runner only needs one result per requested hand, so return a
    compact summary instead of the raw buffer contents.
    """

    try:
        return ["replay_buffer_size", len(payload)]
    except TypeError:
        return ["result_type", type(payload).__name__]


def _run_multiple_hands(args: dict[str, Any]) -> list[list[Any]]:
    """Worker helper to play a batch of hands.

    Creating a :class:`SelfPlay` instance is relatively expensive because the
    underlying trainer, game engine configuration and replay buffers have to be
    serialised when passed to a worker process.  The previous implementation
    spawned one task per hand which resulted in the same objects being pickled
    and unpickled repeatedly.  Grouping all hands assigned to a worker together
    ensures that we only perform the expensive serialisation once per worker
    instead of once per hand.
    """

    cfr_trainer = args["cfr_trainer"]
    game_engine_config = args["game_engine_config"]
    training_config = args["training_config"]
    num_hands = args["num_hands"]

    if num_hands <= 0:
        return []

    sp = SelfPlay(
        cfr_trainer,
        game_engine_config,
        training_config=training_config,
    )
    return [
        _hand_result_summary(sp.play_hand_for_training(iteration=i + 1))
        for i in range(num_hands)
    ]


class DistributedSelfPlay:
    def __init__(
        self,
        cfr_trainer,
        game_engine_config: dict[str, Any],
        training_config: dict[str, Any] | None = None,
    ):
        self.cfr_trainer = cfr_trainer
        self.game_engine_config = game_engine_config
        self.training_config = training_config

    def run(self, num_hands: int, num_workers: int = 2) -> list[list[Any]]:
        """Execute multiple hands in parallel and collect results."""

        if num_hands <= 0:
            return []

        if num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        # Avoid spawning more worker processes than necessary; excess workers
        # would only increase scheduling overhead without producing additional
        # parallelism when ``num_hands`` is small.
        num_workers = min(num_workers, num_hands)

        base, remainder = divmod(num_hands, num_workers)
        hands_per_worker = [base + (1 if i < remainder else 0) for i in range(num_workers)]

        tasks = [
            {
                "cfr_trainer": self.cfr_trainer,
                "game_engine_config": self.game_engine_config,
                "training_config": self.training_config,
                "num_hands": count,
            }
            for count in hands_per_worker
            if count
        ]

        if not tasks:
            return []

        with Pool(processes=len(tasks)) as pool:
            worker_results = pool.map(_run_multiple_hands, tasks)

        results: list[list[Any]] = []
        for batch in worker_results:
            results.extend(batch)
        return results
