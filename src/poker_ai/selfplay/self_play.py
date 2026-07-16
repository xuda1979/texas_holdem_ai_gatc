from __future__ import annotations

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
from poker_ai.model_storage import remote_checkpoint_dir
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
        # Epsilon-greedy exploration during MCCFR self-play.  The config may
        # put this under ``self_play.epsilon`` (matching the YAML schema) or
        # inline in the training section.  Default to 0.0 (pure regret
        # matching) to preserve prior behaviour when the key is absent.
        epsilon_raw = 0.0
        if isinstance(cfg, dict):
            epsilon_raw = cfg.get("epsilon", cfg.get("self_play_epsilon", 0.0))
        try:
            epsilon = float(epsilon_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            epsilon = 0.0
        # Clamp to [0, 1] — values outside this range are nonsensical for an
        # epsilon-greedy mixture weight.
        self.epsilon = max(0.0, min(1.0, epsilon))
        if self.epsilon > 0:
            self.logger.info(
                "SelfPlay epsilon-greedy exploration enabled | epsilon=%.3f",
                self.epsilon,
            )
        # Opponent-pool self-play (fictitious-play style).  With probability
        # ``opponent_mix_probability`` per hand, one non-traverser is designated
        # to play a fixed baseline strategy (default ``calling_station``) instead
        # of the model policy.  This gives the advantage network a direct
        # exploitation signal against that opponent, so the learned equilibrium
        # robustly beats the baseline rather than only transiently peaking
        # against it.  See SESSION_20260713_TRAJECTORY.md for the regression
        # analysis that motivated this.
        opp_mix_raw = 0.0
        opp_strategy_name = "calling_station"
        if isinstance(cfg, dict):
            opp_cfg = cfg.get("opponent_mix", None)
            if isinstance(opp_cfg, dict):
                opp_mix_raw = opp_cfg.get("probability", opp_cfg.get("mix_probability", 0.0))
                opp_strategy_name = opp_cfg.get("strategy", opp_cfg.get("name", "calling_station"))
            else:
                opp_mix_raw = cfg.get("opponent_mix_probability", 0.0)
                opp_strategy_name = cfg.get("opponent_mix_strategy", "calling_station")
        try:
            opp_mix = float(opp_mix_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            opp_mix = 0.0
        self.opponent_mix_probability = max(0.0, min(1.0, opp_mix))
        self.opponent_mix_strategy = str(opp_strategy_name).strip() or "calling_station"
        self._opponent_strategy = self._make_fixed_opponent_strategy(self.opponent_mix_strategy)
        if self.opponent_mix_probability > 0.0 and self._opponent_strategy is not None:
            self.logger.info(
                "SelfPlay opponent-mix enabled | probability=%.3f | strategy=%s",
                self.opponent_mix_probability,
                self.opponent_mix_strategy,
            )
        # Per-hand state: when a hand is drawn as a "mixed opponent" hand, this
        # holds the player id that plays the fixed strategy for this hand, or
        # ``None`` when the hand uses pure self-play.
        self._current_fixed_opponent_id: int | None = None
        # Cumulative counters for opponent-mix engagement diagnostics.  Logged
        # periodically so run7 results can be verified to actually engage the
        # fixed-opponent strategy (see SESSION_20260716_AUTH_BLOCKER.md).
        self._opponent_mix_hands: int = 0
        self._opponent_mix_total_hands: int = 0
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
        directories_to_search.append(remote_checkpoint_dir())

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

    # ------------------------------------------------------------------
    # Opponent-pool self-play helpers (fictitious-play style).
    # ------------------------------------------------------------------
    def _make_fixed_opponent_strategy(self, name: str) -> Any | None:
        """Return a fixed-strategy opponent by name, or ``None`` if disabled.

        Currently supports ``calling_station`` (check/call every decision) and
        ``always_fold`` (fold to any bet, otherwise check).  Unknown names log a
        warning and return ``None`` (pure self-play).
        """
        normalized = (name or "").strip().lower()
        if normalized in ("", "none", "off", "disabled"):
            return None
        try:
            from poker_ai.evaluation.head_to_head import (
                AlwaysFoldStrategy,
                CallingStationStrategy,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(
                "Could not import baseline strategies for opponent-mix: %s", exc
            )
            return None
        if normalized == "calling_station":
            return CallingStationStrategy()
        if normalized == "always_fold":
            return AlwaysFoldStrategy()
        self.logger.warning(
            "Unknown opponent_mix strategy %r; opponent-mix disabled.", name
        )
        return None

    def _fixed_opponent_action_index(
        self, game: TexasHoldem, player_id: int, legal_actions_mask
    ) -> int | None:
        """Map a fixed-strategy opponent's decision to an action index.

        Returns the action index compatible with ``_apply_action_in_place`` that
        best matches the fixed strategy's ``choose_action`` result, or ``None``
        if no legal mapping exists (the caller should fall back to the model
        policy in that case).  Index mapping: 0=FOLD, 1=CHECK/CALL, 2+=RAISE
        buckets (see ``utils.action_mapping.get_action_from_index``).
        """
        if self._opponent_strategy is None:
            return None
        try:
            action_str, amount = self._opponent_strategy.choose_action(game, player_id)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug(
                "Fixed-opponent strategy raised %s for player %s; falling back to model.",
                exc,
                player_id,
            )
            return None
        # ``legal_actions_mask`` is a torch.BoolTensor aligned with the action
        # index space used by ``_get_policy``.  We map the strategy's action
        # string to the first legal index that implements it.
        import torch as _torch  # local import to avoid module-level dep in tests

        if not isinstance(legal_actions_mask, _torch.Tensor):
            mask_list = list(legal_actions_mask)
        else:
            mask_list = legal_actions_mask.detach().cpu().tolist()
        # Index 1 is always CHECK/CALL in the canonical mapping.  If it is legal
        # and the strategy wants check/call, use it.
        if action_str in ("check", "call"):
            if len(mask_list) > 1 and bool(mask_list[1]):
                return 1
            # If index 1 is illegal (shouldn't happen for check/call in normal
            # play), scan for the first legal index — but that would be a fold,
            # which contradicts calling_station, so return None to fall back.
            return None
        if action_str == "fold":
            if len(mask_list) > 0 and bool(mask_list[0]):
                return 0
            return None
        # Raises / bets: the fixed strategies we support never choose these, so
        # this branch is unreachable for calling_station/always_fold.  Return
        # None to let the caller fall back to the model policy.
        return None

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

        deck_manager = getattr(getattr(game, "rules", None), "deck_manager", None)
        if deck_manager is not None and hasattr(deck_manager, "rng"):
            try:
                deck_seed = random.randrange(1 << 63)
            except ValueError:  # pragma: no cover - defensive safeguard
                deck_seed = random.randint(0, 2**31 - 1)
            deck_manager.rng.seed(deck_seed)

        game.rules.big_blind = self.big_blind
        game.rules.small_blind = self.small_blind
        game.initialize_game()
        self.logger.debug(
            "Iteration %s | initialized game with %s players", iteration, num_players
        )

        # 2. Decide whether this hand uses an opponent-pool (fictitious-play)
        # opponent.  When enabled, one non-traverser seat plays a fixed
        # baseline strategy (e.g. calling_station) so the advantage network
        # receives direct exploitation signal.  The designation is per-hand and
        # applies to all traversers in that hand.
        self._current_fixed_opponent_id = None
        is_mixed_opponent_hand = False
        if (
            self.opponent_mix_probability > 0.0
            and self._opponent_strategy is not None
            and num_players >= 2
            and random.random() < self.opponent_mix_probability
        ):
            # Pick a non-traverser seat.  The traverser is the "learner" and the
            # fixed opponent must be a different seat.  We pick seat 1 as the
            # fixed opponent for 2-player hands (seat 0 is the first traverser);
            # for >2 players pick a random non-traverser seat per traversal.
            # The actual designation is resolved per-traverser inside the loop
            # below so the fixed opponent is always a non-traverser seat.
            self._current_fixed_opponent_id = -1  # sentinel: resolve per-traverser
            is_mixed_opponent_hand = True

        # Track opponent-mix engagement for diagnostics.
        self._opponent_mix_total_hands += 1
        if is_mixed_opponent_hand:
            self._opponent_mix_hands += 1
        # Log engagement every 500 hands so run7 results can be verified to
        # actually use the fixed-opponent strategy.
        if (
            self.opponent_mix_probability > 0.0
            and self._opponent_mix_total_hands % 500 == 0
        ):
            rate = (
                self._opponent_mix_hands / self._opponent_mix_total_hands
                if self._opponent_mix_total_hands > 0
                else 0.0
            )
            self.logger.info(
                "opponent-mix engagement | hands=%d | mixed=%d | rate=%.3f (target=%.3f)",
                self._opponent_mix_total_hands,
                self._opponent_mix_hands,
                rate,
                self.opponent_mix_probability,
            )

        # 3. Perform a traversal for each player in the hand
        base_reach = [1.0] * num_players
        for traverser_id in range(num_players):
            # Resolve the fixed-opponent seat for this traverser: any seat
            # other than the traverser.
            if self._current_fixed_opponent_id == -1:
                others = [p for p in range(num_players) if p != traverser_id]
                self._current_fixed_opponent_id = random.choice(others) if others else None
            snapshot = self._snapshot_state(game)
            self._traverse_mccfr(game, traverser_id, iteration, base_reach.copy(), [])
            self._restore_state(game, snapshot)
            # Reset for the next traverser.
            if self.opponent_mix_probability > 0.0:
                self._current_fixed_opponent_id = -1 if (
                    self._opponent_strategy is not None and num_players >= 2
                ) else None
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

                strategy_buffer = getattr(self.cfr_trainer, "strategy_buffer", None)
                train_policy = getattr(self.cfr_trainer, "train_policy", None)
                strategy_buffer_length = (
                    len(strategy_buffer) if hasattr(strategy_buffer, "__len__") else 0
                )
                if (
                    callable(train_policy)
                    and strategy_buffer_length >= self.min_buffer_before_train
                ):
                    policy_loss = train_policy(batch_size=self.min_buffer_before_train)
                    if policy_loss is not None:
                        self.logger.info(
                            "Iteration %s | policy training step complete | loss=%.6f | buffer=%s",
                            iteration,
                            float(policy_loss),
                            strategy_buffer_length,
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

        # Epsilon-greedy exploration: mix the regret-matched policy with a
        # uniform distribution over legal actions.  This prevents the
        # strategy buffer from collapsing to a single dominant line (which
        # produces a policy network that exploits one baseline but loses to
        # another).  The mixture weight is read from the ``self_play.epsilon``
        # config key (default 0.0 preserves prior behaviour).
        if self.epsilon > 0.0:
            num_legal = int(legal_actions_mask.sum().item())
            if num_legal > 0:
                uniform = torch.zeros_like(policy)
                uniform[legal_actions_mask] = 1.0 / num_legal
                policy = (1.0 - self.epsilon) * policy + self.epsilon * uniform

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

            # External Sampling MCCFR (Lanctot et al., 2009; Brown et al., 2019):
            # opponent actions are *sampled* from their current policy, so the
            # sampled instantaneous regret u(a) - v is already an unbiased
            # estimator of the opponent-reach-weighted counterfactual regret.
            # Multiplying by the realized opponent reach again would weight the
            # targets by pi_{-i}^2 and bias the regret estimates, breaking the
            # convergence guarantee.  The reach is therefore NOT applied here.
            #
            # Targets are expressed in units of the chip normalization scale so
            # they match the magnitude of the (normalized) network inputs.
            # Regret matching is invariant to positive rescaling, so this does
            # not change the induced policies -- only the conditioning of the
            # regression problem.
            target_scale = max(float(normalization_scale), 1e-12)
            scaled_regrets = regrets / target_scale
            scaled_action_values = action_utilities / target_scale

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
                    action_values=scaled_action_values.detach().clone(),
                    regrets=scaled_regrets.detach().clone(),
                    legal_mask=legal_actions_mask,
                    opponent_reach=1.0,
                    iteration=iteration,
                )

            if not hasattr(self.cfr_trainer, "add_experience") and hasattr(self.cfr_trainer, "replay_buffer"):
                try:
                    self.cfr_trainer.replay_buffer.push(
                        hole_s,
                        community_s,
                        state_tensor,
                        scaled_action_values.detach().clone(),
                        legal_mask=legal_actions_mask,
                        opponent_reach=1.0,
                        iteration=iteration,
                    )
                except TypeError:
                    self.cfr_trainer.replay_buffer.push(
                        hole_s,
                        community_s,
                        state_tensor,
                        scaled_regrets.detach().clone(),
                        iteration,
                    )

            return node_value

        # --- Opponent Node ---
        # Deep CFR (Brown et al., 2019, Algorithm 1) collects average-strategy
        # samples at the *opponent's* decision nodes.  Because the opponent's
        # actions along the sampled trajectory are drawn from their own current
        # policy, infosets are visited with probability proportional to the
        # acting player's reach, which yields the correct reach weighting of
        # the average strategy without explicit multiplication.
        #
        # Opponent-pool extension: when this hand was designated a
        # "mixed-opponent" hand and ``current_player`` is the fixed-opponent
        # seat, the opponent plays the baseline strategy (e.g. calling_station)
        # instead of the model policy.  We still collect the strategy-buffer
        # sample using the *model's* policy (so the policy network learns to
        # imitate the model's own average strategy at this infoset), but the
        # *action taken along the trajectory* is the fixed strategy's action.
        # This keeps the Deep CFR strategy-buffer semantics intact while giving
        # the advantage network direct exploitation signal: the traverser's
        # regret targets reflect "what's the value of each action when the
        # opponent plays calling_station?".
        use_fixed_opponent = (
            self._current_fixed_opponent_id is not None
            and self._current_fixed_opponent_id >= 0
            and current_player == self._current_fixed_opponent_id
            and self._opponent_strategy is not None
        )
        if hasattr(self.cfr_trainer, "add_strategy_experience"):
            model_config = self.cfr_trainer.config.get("model", {})
            max_seq_len = model_config.get("max_seq_len", 256)
            d_raw_feature = model_config.get("d_raw_feature", 18)
            hole_s, community_s, state_tensor = prepare_transformer_input(
                game,
                current_player,
                max_seq_len,
                d_raw_feature,
                normalization_scale=normalization_scale,
            )
            self.cfr_trainer.add_strategy_experience(
                hole_s,
                community_s,
                state_tensor,
                strategy=policy.detach().clone(),
                iteration=iteration,
            )

        if use_fixed_opponent:
            fixed_idx = self._fixed_opponent_action_index(
                game, current_player, legal_actions_mask_cpu
            )
            if fixed_idx is not None:
                action_idx = fixed_idx
                # Use the model's policy probability at that index for the
                # reach update so External Sampling's reach-weighting remains
                # consistent with the on-policy trajectory.
                undo_stack.append(self._snapshot_state(game))
                self._apply_action_in_place(game, current_player, action_idx)
                new_reach = reach_probs.copy()
                new_reach[current_player] *= max(float(policy[action_idx].item()), 1e-12)
                value = self._traverse_mccfr(game, traverser_id, iteration, new_reach, undo_stack)
                snapshot = undo_stack.pop()
                self._restore_state(game, snapshot)
                return value
            # If the fixed strategy had no legal mapping (rare edge case),
            # fall through to the model-policy sampling below.

        action_idx = torch.multinomial(policy, 1).item()
        undo_stack.append(self._snapshot_state(game))
        self._apply_action_in_place(game, current_player, action_idx)
        new_reach = reach_probs.copy()
        new_reach[current_player] *= policy[action_idx].item()
        value = self._traverse_mccfr(game, traverser_id, iteration, new_reach, undo_stack)
        snapshot = undo_stack.pop()
        self._restore_state(game, snapshot)
        return value
