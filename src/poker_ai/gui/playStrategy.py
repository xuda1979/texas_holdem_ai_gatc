# playStrategy.py

# ruff: noqa: N999,ANN001,ANN201,ANN204
import random

import torch
from typing import TYPE_CHECKING

from poker_ai.utils.action_mapping import (
    action_to_tuple,
    get_action_from_index,
    get_legal_actions_mask,
)
from poker_ai.utils.state_representation import prepare_transformer_input

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from poker_ai.ai.models.transformer import AdvantageNetwork

# Import the new AI GTO display function lazily inside HumanStrategy to avoid
# pulling heavy GUI dependencies when simply importing this module.


class PlayerStrategy:
    @property
    def is_human(self):
        raise NotImplementedError

    def choose_action(self, game, player_index):
        raise NotImplementedError


class RandomAIStrategy(PlayerStrategy):
    @property
    def is_human(self):
        return False

    def choose_action(self, game, player_index):
        actions = game.get_valid_actions(player_index)
        amount_to_call = game.rules.current_bet - game.rules.bets[player_index]

        action = random.choice(actions)

        if action in ["raise", "bet"]:
            min_raise = game.get_min_raise_amount(player_index)
            max_raise = game.get_max_raise_amount(player_index)
            if max_raise < min_raise:
                if amount_to_call > 0:
                    return "call", None  # Can't raise; must call or fold
                else:
                    return "check", None  # Can't raise; must check
            # AI decides on a raise amount within the allowed range
            raise_amount = random.randint(min_raise, min(max_raise, min_raise + 100))
            return action, raise_amount
        return action, None


class ModelAIStrategy(PlayerStrategy):
    """Strategy driven by a trained :class:`AdvantageNetwork`."""

    def __init__(self, model: "AdvantageNetwork", config: dict, device: torch.device):
        # Import locally to avoid circular dependency during module import
        from poker_ai.ai.models.transformer import AdvantageNetwork

        if not isinstance(model, AdvantageNetwork):  # pragma: no cover - simple type check
            raise TypeError("model must be an AdvantageNetwork instance")

        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.device = device

    @property
    def is_human(self) -> bool:
        return False

    @torch.no_grad()
    def choose_action(self, game, player_index):
        max_seq_len = self.config.get("max_seq_len", 256)
        d_raw_feature = self.config.get(
            "d_raw_feature",
            self.config.get(
                "input_feature_dim",
                self.config.get("history_feature_dim", 18),
            ),
        )
        normalization_scale = getattr(
            game.rules, "starting_stack", getattr(game, "starting_stack", 1.0)
        )
        hole, community, history = prepare_transformer_input(
            game,
            player_index,
            max_seq_len,
            d_raw_feature,
            normalization_scale=normalization_scale,
        )
        advantages = (
            self.model(
                hole.unsqueeze(0).to(self.device),
                community.unsqueeze(0).to(self.device),
                history.unsqueeze(0).to(self.device),
            )
            .squeeze(0)
            .cpu()
        )
        num_actions = self.config.get("num_actions", self.model.num_actions)
        legal_mask = get_legal_actions_mask(game, player_index, num_actions)

        advantages[~legal_mask] = -float("inf")
        positive = torch.clamp(advantages, min=0) * legal_mask.float()
        if positive.sum() > 0:
            policy = positive / positive.sum()
        else:
            policy = legal_mask.float() / legal_mask.sum()

        action_idx = torch.multinomial(policy, 1).item()
        action = get_action_from_index(action_idx, game, player_index)
        return action_to_tuple(action)


class HumanStrategy(PlayerStrategy):
    @property
    def is_human(self):
        return True

    def choose_action(self, game, player_index):  # noqa: C901
        # Display AI-derived GTO stats before prompting for action.
        # The import is delayed to keep GUI dependencies optional.
        # ``display_ai_gto_stats`` lives in the optional evaluation package and
        # may be unavailable in lightweight environments.  Import through the
        # package namespace so the function is ``None`` rather than raising a
        # ``ModuleNotFoundError`` when the dependency tree is incomplete.
        from poker_ai.evaluation import display_ai_gto_stats

        if display_ai_gto_stats:
            display_ai_gto_stats(game, player_index)

        while True:
            print(f"\n--- Player {player_index + 1}'s Turn ---")
            print(f"Current pot: {game.rules.pot} chips")
            print(f"Your chips: {game.rules.player_chips[player_index]} chips")
            amount_to_call = game.rules.current_bet - game.rules.bets[player_index]
            if amount_to_call > 0:
                print(f"Amount to call: {amount_to_call} chips")
            else:
                print("You can check.")

            valid_actions = game.get_valid_actions(player_index)

            action = input(f"Choose your action ({', '.join(valid_actions)}): ").lower()
            if action in ["raise", "bet"]:
                min_raise = game.get_min_raise_amount(player_index)
                max_raise = game.get_max_raise_amount(player_index)
                if max_raise < min_raise:
                    if amount_to_call > 0:
                        print("You don't have enough chips to raise. You can only call or fold.")
                    else:
                        print("You don't have enough chips to raise. You can only check.")
                    continue
                while True:
                    try:
                        raise_amount = int(
                            input(f"Enter raise amount (minimum {min_raise} chips): ")
                        )
                        if raise_amount < min_raise:
                            print(f"Raise amount must be at least {min_raise} chips.")
                        elif raise_amount > max_raise:
                            print(
                                f"Raise amount cannot exceed your available chips "
                                f"({max_raise} chips)."
                            )
                        else:
                            break
                    except ValueError:
                        print("Invalid input. Please enter a numeric value.")
                return action, raise_amount
            elif action in ["call", "fold", "check"]:
                return action, None
            else:
                print("Invalid action. Please try again.")
