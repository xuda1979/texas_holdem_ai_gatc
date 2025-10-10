"""Implements External Sampling MCCFR for data generation as described in ``texas.tex``."""

# ruff: noqa

import copy
import inspect
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch

# Assuming these imports are correct relative to the project structure
from poker_ai.engine.texas_holdem import TexasHoldem, TexasHoldemRules
from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import (
    infer_normalization_scale,
    prepare_transformer_input,
)


@dataclass
class _GameStateSnapshot:
    """Lightweight snapshot for restoring ``TexasHoldem`` traversal state."""

    rules: TexasHoldemRules
    end_game_early: bool
    winner: Any
    pending_showdown: Any


class SelfPlay:
    """Orchestrates MCCFR traversals for training data generation."""

    def __init__(
        self,
        cfr_trainer: object,
        game_engine_config: dict[str, Any],
        training_config: dict[str, Any] | None = None,
        *,
        train_during_generation: bool | None = None,
    ) -> None:
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cfr_trainer = cfr_trainer
        self.starting_stack = game_engine_config.get("starting_stack", 1000)
        self.big_blind = game_engine_config.get("big_blind", 10)
        self.small_blind = game_engine_config.get("small_blind", 5)
        self.min_players = game_engine_config.get("min_players", 2)
        self.max_players = game_engine_config.get("max_players", 10)
        if training_config is None:
            cfg: dict[str, Any] = {}
        elif isinstance(training_config, dict):
            # Defensive copy to avoid mutating caller-owned dictionaries.  The
            # previous behaviour surprised unit tests that reused a shared
            # configuration fixture across multiple ``SelfPlay`` instances.
            cfg = copy.deepcopy(training_config)
        else:
            # Fallback to ``dict`` construction for mapping-like objects.
            cfg = dict(training_config)  # type: ignore[arg-type]
        self.training_config = cfg
        cfg_train_flag = None
        if isinstance(cfg, dict):
            cfg_train_flag = cfg.get("train_during_generation")
        if train_during_generation is None:
            train_during_generation = (
                bool(cfg_train_flag) if isinstance(cfg_train_flag, bool) else True
            )
        self.train_during_generation = bool(train_during_generation)
        if isinstance(cfg, dict):
            cfg["train_during_generation"] = self.train_during_generation
        min_buffer_raw = cfg.get("min_buffer_before_train", 256)
        try:
            min_buffer = int(min_buffer_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            min_buffer = 256
        self.min_buffer_before_train = max(1, min_buffer)
        self._model_reload_interval = self._determine_model_reload_interval(cfg)
        self._last_loaded_model_path: Path | None = None
        self._last_loaded_model_mtime: float | None = None
        self._maybe_refresh_model(iteration=0, force=True)
        self.logger.debug(
            "Initialized SelfPlay | stack=%s | blinds=(%s,%s) | players=%s-%s",
            self.starting_stack,
            self.small_blind,
            self.big_blind,
            self.min_players,
            self.max_players,
        )

    def _determine_model_reload_interval(self, cfg: dict[str, Any]) -> int:
        """Return the frequency (in hands) for refreshing model weights."""

        interval_raw = (
            cfg.get("reload_model_every_hands")
            or cfg.get("check_for_new_model_every_hands")
            or cfg.get("save_model_every_n_hands")
        )
        try:
            interval = int(interval_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            self.logger.warning(
                "Invalid reload interval %r; defaulting to refreshing every hand.",
                interval_raw,
            )
            interval = 1
        if interval <= 0:
            self.logger.warning(
                "Reload interval %s must be positive; defaulting to 1 hand.", interval
            )
            interval = 1
        return interval

    def _candidate_model_paths(self) -> list[Path]:
        """Return possible checkpoint files to inspect for refreshes."""

        candidates: list[Path] = []
        direct_paths: list[str] = []
        training_cfg = self.training_config if isinstance(self.training_config, dict) else {}
        direct_paths.extend(
            [
                training_cfg.get("latest_model_path"),
                training_cfg.get("save_model_path"),
            ]
        )

        trainer_cfg = getattr(self.cfr_trainer, "config", {})
        model_cfg: dict[str, Any] = {}
        if isinstance(trainer_cfg, dict):
            training_section = trainer_cfg.get("training")
            if isinstance(training_section, dict):
                direct_paths.append(training_section.get("save_model_path"))
            maybe_model_cfg = trainer_cfg.get("model")
            if isinstance(maybe_model_cfg, dict):
                model_cfg = maybe_model_cfg

        model_directory = training_cfg.get("model_directory") or model_cfg.get("directory")
        filename_prefix = training_cfg.get("model_filename_prefix") or model_cfg.get(
            "filename_prefix"
        )

        for path_str in direct_paths:
            if not path_str:
                continue
            candidate = Path(path_str).expanduser()
            if candidate.is_file():
                candidates.append(candidate)

        directories_to_search: list[Path] = []
        if isinstance(model_directory, str) and model_directory:
            directories_to_search.append(Path(model_directory).expanduser())
        directories_to_search.append(Path("models"))

        patterns: list[str] = ["*.pth"]
        if isinstance(filename_prefix, str) and filename_prefix:
            patterns.append(f"{filename_prefix}*.pth")

        seen_directories: set[str] = set()
        for directory in directories_to_search:
            try:
                resolved_dir = str(directory.resolve())
            except OSError:
                resolved_dir = str(directory)
            if resolved_dir in seen_directories:
                continue
            seen_directories.add(resolved_dir)
            try:
                if not directory.is_dir():
                    continue
            except PermissionError as exc:
                self.logger.warning(
                    "Skipping model directory %s due to permission error: %s",
                    directory,
                    exc,
                )
                continue
            except OSError:  # pragma: no cover - defensive
                continue
            for pattern in patterns:
                try:
                    iterator = directory.glob(pattern)
                except OSError:
                    continue
                for path in iterator:
                    try:
                        if path.is_file():
                            candidates.append(path)
                    except OSError:
                        continue

        unique: list[Path] = []
        seen: set[str] = set()
        for path in candidates:
            try:
                resolved = str(path.resolve())
            except OSError:  # pragma: no cover - defensive
                resolved = str(path)
            if resolved in seen:
                continue
            seen.add(resolved)
            unique.append(path)
        return unique

    def _find_latest_model_checkpoint(self) -> Path | None:
        """Return the most recently modified checkpoint file if available."""

        candidates = self._candidate_model_paths()
        if not candidates:
            return None

        latest_path = None
        latest_mtime = float("-inf")
        for path in candidates:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = path
        return latest_path

    def _load_model_from_path(self, path: Path) -> bool:
        """Invoke the trainer's load routine for ``path`` if possible."""

        load_attr = getattr(self.cfr_trainer, "load_model", None)
        if load_attr is None:
            return False

        try:
            load_attr(str(path))
        except TypeError:
            load_attr()
        except FileNotFoundError:
            logging.warning("Checkpoint %s disappeared before it could be loaded.", path)
            return False
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error("Failed to load model from %s: %s", path, exc)
            return False
        return True

    def _maybe_refresh_model(self, iteration: int, *, force: bool = False) -> None:
        """Reload the latest checkpoint when due or when forced."""

        if not force:
            if self._model_reload_interval <= 0:
                return
            if iteration <= 0:
                return
            if iteration % self._model_reload_interval != 0:
                return

        latest_path = self._find_latest_model_checkpoint()
        if latest_path is None:
            return

        try:
            latest_mtime = latest_path.stat().st_mtime
        except OSError as exc:
            self.logger.warning(
                "Unable to stat potential checkpoint %s: %s", latest_path, exc
            )
            return

        if (
            not force
            and self._last_loaded_model_path is not None
            and self._last_loaded_model_mtime is not None
        ):
            try:
                same_path = latest_path.resolve() == self._last_loaded_model_path
            except OSError:  # pragma: no cover - defensive
                same_path = False
            if same_path and latest_mtime <= self._last_loaded_model_mtime:
                return

        if not self._load_model_from_path(latest_path):
            return

        try:
            resolved = latest_path.resolve()
        except OSError as exc:  # pragma: no cover - defensive
            self.logger.warning(
                "Failed to resolve checkpoint path %s: %s", latest_path, exc
            )
            resolved = latest_path
        self._last_loaded_model_path = resolved
        self._last_loaded_model_mtime = latest_mtime
 

    def _normalization_scale_for_game(self, game: TexasHoldem) -> float:
        """Determine the chip normalization scale for ``game``."""

        config_obj = getattr(self.cfr_trainer, "config", {})
        preferred_scale = None
        if isinstance(config_obj, dict):
            for key in ("normalization_scale", "chip_normalization", "starting_stack"):
                value = config_obj.get(key)
                if isinstance(value, (int, float)):
                    preferred_scale = float(value)
                    break

        if preferred_scale is None:
            preferred_scale = float(self.starting_stack)

        return infer_normalization_scale(game, preferred_scale)
 

    def play_hand_for_training(self, iteration: int = 0) -> list[Any]:
        """Run one full MCCFR traversal for a new hand."""
        self._maybe_refresh_model(iteration)
        # 1. Initialize a new hand with a random number of players
        num_players = random.randint(self.min_players, self.max_players)
        constructor = TexasHoldem
        try:
            signature = inspect.signature(constructor)
        except (TypeError, ValueError):  # pragma: no cover - dynamic classes
            signature = None

        kwargs = {"num_players": num_players, "starting_stack": self.starting_stack}
        if signature is None or "verbose" in signature.parameters:
            kwargs["verbose"] = False

        try:
            game = constructor(**kwargs)
        except TypeError:
            # Some lightweight test doubles only accept positional arguments.
            game = constructor(num_players, self.starting_stack)
        game.rules.big_blind = self.big_blind
        game.rules.small_blind = self.small_blind
        game.initialize_game()
        self.logger.debug(
            "Iteration %s | initialized game with %s players", iteration, num_players
        )

        # 2. Perform a traversal for each player in the hand
        base_reach = [1.0] * num_players
        for traverser_id in range(num_players):
            snapshot = self._snapshot_state(game)
            self._traverse_mccfr(game, traverser_id, iteration, base_reach.copy(), [])
            self._restore_state(game, snapshot)

        # 3. After the traversals, run a training step on the collected data

        buffer_length = len(self.cfr_trainer.replay_buffer)
        if self.train_during_generation:
            if buffer_length >= self.min_buffer_before_train:
                loss = self.cfr_trainer.train(batch_size=self.min_buffer_before_train)
                if loss is not None:
                    self.logger.info(
                        "Iteration %s | training step complete | loss=%.6f | buffer=%s",
                        iteration,
                        float(loss),
                        buffer_length,
                    )
            else:
                self.logger.debug(
                    "Iteration %s | buffer below threshold (%s/%s)",
                    iteration,
                    buffer_length,
                    self.min_buffer_before_train,
                )

        return self.cfr_trainer.replay_buffer

    def _get_policy(
        self,
        game: TexasHoldem,
        player_id: int,
        normalization_scale: float | None = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return the regret-matched policy for ``player_id``.

        Parameters
        ----------
        game:
            Active :class:`TexasHoldem` instance.
        player_id:
            Index of the acting player.
        normalization_scale:
            Optional chip scale used when constructing the infoset features.  Passing
            a value avoids recomputing the preferred scale when the caller already
            inferred it for the current node.
        return_mask:
            When ``True`` the boolean mask of legal abstract actions (on CPU) is
            returned alongside the policy.  This allows callers to reuse the mask
            without recomputing it, reducing repeated environment queries.
        """

        model_config = self.cfr_trainer.config.get("model", {})
        max_seq_len = model_config.get("max_seq_len", 256)
        d_raw_feature = model_config.get("d_raw_feature", 18)

        if normalization_scale is None:
            normalization_scale = self._normalization_scale_for_game(game)

        hole, community, history_tensor = prepare_transformer_input(
            game,
            player_id,
            max_seq_len,
            d_raw_feature,
            normalization_scale=normalization_scale,
        )

        try:
            advantages = self.cfr_trainer.get_advantages(hole, community, history_tensor)
        except TypeError:  # pragma: no cover - backwards compat
            advantages = self.cfr_trainer.get_advantages(history_tensor)

        legal_actions_mask_cpu = get_legal_actions_mask(
            game, player_id, self.cfr_trainer.num_actions
        )
        legal_actions_mask = legal_actions_mask_cpu.to(advantages.device)

        positive_advantages = torch.clamp(advantages, min=0.0)
        positive_advantages = torch.where(
            legal_actions_mask,
            positive_advantages,
            torch.zeros_like(positive_advantages, device=advantages.device),
        )

        sum_positive = positive_advantages.sum()
        policy = torch.zeros_like(advantages, device=advantages.device)
        if sum_positive.item() > 0:
            policy[legal_actions_mask] = positive_advantages[legal_actions_mask] / sum_positive
        else:
            num_legal = int(legal_actions_mask.sum().item())
            if num_legal > 0:
                policy[legal_actions_mask] = 1.0 / num_legal

        if return_mask:
            return policy, legal_actions_mask_cpu
        return policy

    def _is_betting_round_over(self, game: TexasHoldem) -> bool:
        """Return ``True`` when the current street has finished for traversal purposes."""

        rules = game.rules
        active_non_allin = [
            i
            for i in range(rules.num_players)
            if rules.active_players[i] and rules.player_chips[i] > 0
        ]
        if not active_non_allin:
            return True

        all_settled = all(
            rules.bets[i] == rules.current_bet for i in active_non_allin
        )
        actions_this_round = getattr(rules, "actions_this_round", 0)
        return all_settled and actions_this_round >= len(active_non_allin)

    def _advance_street(self, game: TexasHoldem) -> None:
        """Advance the existing game to the next street without allocating a clone."""

        cleanup = getattr(game.rules, "end_betting_round_cleanup", None)
        if callable(cleanup):
            cleanup()

        community_cards = game.rules.community_cards
        if len(community_cards) == 0:
            game.play_stage("flop")
        elif len(community_cards) == 3:
            game.play_stage("turn")
        elif len(community_cards) == 4:
            game.play_stage("river")

        # First player to act post-flop is directly left of the dealer button.
        start_player = (game.rules.dealer_button + 1) % game.rules.num_players
        current = start_player
        for _ in range(game.rules.num_players):
            if game.rules.active_players[current] and game.rules.player_chips[current] > 0:
                break
            current = (current + 1) % game.rules.num_players
        game.rules.current_player = current

    def _snapshot_state(self, game: TexasHoldem) -> _GameStateSnapshot:
        """Capture the current mutable state so it can be restored later."""

        rules = game.rules
        if hasattr(rules, "clone") and callable(getattr(rules, "clone")):
            rules_snapshot = rules.clone()
        else:
            rules_snapshot = copy.deepcopy(rules)

        end_game_early = bool(getattr(game, "end_game_early", False))
        winner = copy.deepcopy(getattr(game, "winner", None))

        return _GameStateSnapshot(
            rules=rules_snapshot,
            end_game_early=end_game_early,
            winner=winner,
            pending_showdown=copy.deepcopy(getattr(game, "_pending_showdown_winnings", None)),
        )

    def _restore_state(self, game: TexasHoldem, snapshot: _GameStateSnapshot) -> None:
        """Restore ``game`` to a previously captured snapshot."""

        game.rules = snapshot.rules
        game.end_game_early = snapshot.end_game_early
        game.winner = snapshot.winner
        if hasattr(game, "_pending_showdown_winnings"):
            game._pending_showdown_winnings = snapshot.pending_showdown

    def _apply_action_in_place(
        self, game: TexasHoldem, player_id: int, action_idx: int
    ) -> None:
        """Apply an indexed action to ``game`` without cloning."""

        action = get_action_from_index(action_idx, game, player_id=player_id)
        action_str, amount = action_to_tuple(action)

        if action_str == "raise" and amount is not None:
            current_bet = getattr(getattr(game, "rules", game), "current_bet", 0)
            if current_bet > 0:
                amount = max(0, amount - current_bet)

        game.process_action(player_id, action_str, amount)
        game.rules.advance_turn()

    def _traverse_mccfr(  # noqa: C901
        self,
        game: TexasHoldem,
        traverser_id: int,
        iteration: int,
        reach_probs: list[float],
        undo_stack: Optional[List[_GameStateSnapshot]] = None,
    ) -> float:
        """Recursive function to perform an External Sampling MCCFR traversal."""

        if undo_stack is None:
            undo_stack = []

        # --- Terminal Node ---
        if game.is_hand_over():
            return game.get_payoff(traverser_id)

        # --- Chance Node ---
        if self._is_betting_round_over(game):
            undo_stack.append(self._snapshot_state(game))
            self._advance_street(game)
            value = self._traverse_mccfr(
                game, traverser_id, iteration, reach_probs, undo_stack
            )
            snapshot = undo_stack.pop()
            self._restore_state(game, snapshot)
            return value

        # --- Decision Node ---
        current_player = game.rules.current_player
        normalization_scale = self._normalization_scale_for_game(game)
        policy_result = self._get_policy(
            game,
            current_player,
            normalization_scale=normalization_scale,
            return_mask=True,
        )
        if isinstance(policy_result, tuple):
            policy, legal_actions_mask_cpu = policy_result
        else:  # pragma: no cover - defensive fallback for mocked tests
            policy = policy_result
            legal_actions_mask_cpu = get_legal_actions_mask(
                game, current_player, self.cfr_trainer.num_actions
            )

        if current_player == traverser_id:
            action_utilities = torch.zeros_like(policy)
            legal_actions_mask = legal_actions_mask_cpu
            for action_idx in range(self.cfr_trainer.num_actions):
                if not legal_actions_mask[action_idx].item():
                    continue

                undo_stack.append(self._snapshot_state(game))
                self._apply_action_in_place(game, current_player, action_idx)
                action_utilities[action_idx] = self._traverse_mccfr(
                    game, traverser_id, iteration, reach_probs, undo_stack
                )
                snapshot = undo_stack.pop()
                self._restore_state(game, snapshot)

            node_value = (action_utilities * policy).sum().item()
            regrets = action_utilities - node_value

            opponent_reach = 1.0
            for idx, prob in enumerate(reach_probs):
                if idx != traverser_id:
                    opponent_reach *= prob
            weighted_regrets = regrets * opponent_reach

            model_config = self.cfr_trainer.config.get("model", {})
            max_seq_len = model_config.get("max_seq_len", 256)
            d_raw_feature = model_config.get("d_raw_feature", 18)
            hole_s, community_s, state_tensor = prepare_transformer_input(
                game,
                traverser_id,
                max_seq_len,
                d_raw_feature,
                normalization_scale=normalization_scale,
            )
            if hasattr(self.cfr_trainer, "add_experience"):
                self.cfr_trainer.add_experience(
                    hole_s,
                    community_s,
                    state_tensor,
                    action_values=action_utilities.detach().clone(),
                    regrets=weighted_regrets.detach().clone(),
                    legal_mask=legal_actions_mask,
                    opponent_reach=opponent_reach,
                    iteration=iteration,
                )
            elif hasattr(self.cfr_trainer, "replay_buffer"):
                try:
                    self.cfr_trainer.replay_buffer.push(
                        hole_s,
                        community_s,
                        state_tensor,
                        action_utilities.detach().clone(),
                        legal_mask=legal_actions_mask,
                        opponent_reach=opponent_reach,
                        iteration=iteration,
                    )
                except TypeError:
                    self.cfr_trainer.replay_buffer.push(
                        hole_s,
                        community_s,
                        state_tensor,
                        weighted_regrets,
                        iteration,
                    )

            return node_value

        action_idx = torch.multinomial(policy, 1).item()
        undo_stack.append(self._snapshot_state(game))
        self._apply_action_in_place(game, current_player, action_idx)
        new_reach = reach_probs.copy()
        new_reach[current_player] *= policy[action_idx].item()
        value = self._traverse_mccfr(game, traverser_id, iteration, new_reach, undo_stack)
        snapshot = undo_stack.pop()
        self._restore_state(game, snapshot)
        return value
