import os
from typing import Any

# Optional dependency: PyYAML.  The analyzer is rarely used in tests, so we
# allow the module to load even if the package is missing.
try:  # pragma: no cover - executed when PyYAML is present
    import yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML missing
    yaml = None

from poker_ai.utils.action_mapping import get_action_from_index

from poker_ai.ai.trainers.ai_cfr_trainer import AICFRTrainer as CFRTrainer

# We need access to game_engine.texas_holdem.TexasHoldem for type hinting
# if game_state is passed directly. However, CFRTrainer.encode_state and
# get_action_from_index expect specific attributes from ``game_state`` or
# ``game_state.rules``. For now, let's assume ``game_state`` has a
# ``rules`` attribute and player chips can be accessed.

MODEL_CONFIG_PATH = "config.yaml"
DEFAULT_MODEL_FILENAME = "cfr_model.pth"  # From config.yaml training.save_model_path

# Global variable to cache the loaded model and trainer
# This is to avoid reloading the model on every call if the analyzer is used multiple times.
_loaded_cfr_trainer = None
_trainer_config = None
_full_config = None


def load_cfr_model_and_config(
    model_path: str | None = None,
) -> tuple[CFRTrainer | None, dict | None]:
    """Load the CFRTrainer model and configuration.

    If a model is already loaded, it returns the cached one unless a
    different ``model_path`` is specified.
    """
    global _loaded_cfr_trainer, _trainer_config, _full_config

    if _trainer_config is None or _full_config is None:
        try:
            if yaml is not None:
                with open(MODEL_CONFIG_PATH) as f:
                    _full_config = yaml.safe_load(f)  # type: ignore[arg-type]
            else:
                raise FileNotFoundError
            # Extract relevant parts for CFRTrainer.
            # CFRTrainer expects keys like 'num_actions', 'input_shape', 'learning_rate'.
            # The 'model' section in config.yaml has 'num_actions', 'hidden_dim', 'd_raw_feature'.
            # Determine ``input_shape`` for the trainer based on game state
            # representation. Example layouts include (height, width, channels)
            # or (depth, height, width). ``TexasHoldem.get_initial_state``
            # returns (num_players + 5, len(RANKS), len(SUITS)); treat these as
            # (depth, height, width) for the CNN. The trainer's build_model uses
            # ``c, h, w = self.input_shape[2], self.input_shape[0],
            # self.input_shape[1]`` implying ``input_shape`` should be
            # ordered as (h, w, c) in the config. For now we hardcode a
            # plausible shape based on typical card representations. A more
            # robust solution would derive these from engine constants, e.g.:
            # ``num_players = full_config.get('game_engine', {}).get('num_players', 2)``,
            # ``num_ranks = 13`` and ``num_suits = 4``. Depth becomes
            # ``num_players + 5`` (player hands + community cards), giving
            # ``input_shape_for_trainer_config = (num_ranks, num_suits, depth)``.

            # The CFRTrainer's encode_state uses np.resize(arr, self.input_shape).
            # And its CNN expects (N, C, H, W) after permute, from (N, H, W, C)
            # So self.input_shape should be (H, W, C)
            # Let H = num_ranks, W = num_suits, C = num_players + 5
            # This needs to be consistent with how get_initial_state() output is interpreted.
            # The default ``get_initial_state()`` is ``(num_players + 5,
            # len(RANKS), len(SUITS))`` i.e. (C, H, W). Let's use placeholders
            # for ``input_shape`` and ``num_actions`` from ``config.yaml``; these
            # should align with the saved model's architecture.

            # A simplified config for CFRTrainer based on what it uses:
            _trainer_config = {
                "num_actions": _full_config.get("model", {}).get("num_actions", 10),
                # input_shape: (height, width, channels/depth).
                # game.get_initial_state() produces (depth, height, width)
                # cfr_trainer.encode_state resizes this to self.input_shape
                # Then cnn permutes from (N,H,W,C) to (N,C,H,W)
                # So self.input_shape for trainer needs to be (H,W,C)
                # Use placeholder values if the exact training configuration is
                # unclear. These matter only if they affect layer sizes that are
                # not obvious from the model's ``state_dict``. Usually,
                # ``num_actions`` is most critical.
                "input_shape": _full_config.get("model", {}).get(
                    "input_shape", (13, 4, 7)
                ),  # (Ranks, Suits, Depth_placeholder)
                "learning_rate": _full_config.get("training", {}).get(
                    "learning_rate", 0.001
                ),  # Default if not in config
            }
        except Exception as e:
            print(f"Error loading or parsing {MODEL_CONFIG_PATH}: {e}")
            return None, None

    current_model_path = model_path
    if current_model_path is None:
        _trainer_config.get(
            "model_directory", _full_config.get("model", {}).get("directory", "trained_models")
        )
        actual_model_save_path = _full_config.get("training", {}).get(
            "save_model_path", os.path.join("trained_models", DEFAULT_MODEL_FILENAME)
        )
        current_model_path = actual_model_save_path

    os.makedirs(os.path.dirname(current_model_path), exist_ok=True)

    if _loaded_cfr_trainer is not None and _loaded_cfr_trainer.model_path == current_model_path:
        return _loaded_cfr_trainer, _trainer_config

    try:
        trainer = CFRTrainer(config=_trainer_config)
        if os.path.exists(current_model_path):
            if trainer.load_model(current_model_path):
                trainer.model.eval()  # Set to evaluation mode
                _loaded_cfr_trainer = trainer
                _loaded_cfr_trainer.model_path = current_model_path  # Store path for caching check
                print(f"Successfully loaded AI model from: {current_model_path}")
                return _loaded_cfr_trainer, _trainer_config
            else:
                print(f"Failed to load AI model from: {current_model_path}")
                _loaded_cfr_trainer = None
                return None, _trainer_config
        else:
            print(f"AI model file not found at: {current_model_path}. Cannot display AI GTO stats.")
            _loaded_cfr_trainer = None  # Ensure cache is cleared if load fails
            return None, _trainer_config  # Return config for potential re-attempt or partial use

    except Exception as e:
        print(f"Error initializing CFRTrainer or loading model from {current_model_path}: {e}")
        _loaded_cfr_trainer = None  # Ensure cache is cleared
        return None, _trainer_config


def display_ai_gto_stats(  # noqa: C901
    game: Any,  # noqa: ANN401
    player_index: int,
    model_path: str | None = None,
) -> None:
    """Display GTO-related statistics for the current player.

    Args:
        game: The current game object (e.g., an instance of ``TexasHoldem``).
        player_index: The index of the human player.
        model_path: Optional path to a specific model file.
    """
    print("\n--- AI Model Strategy Analysis ---")

    trainer, trainer_config = load_cfr_model_and_config(model_path)

    if trainer is None or trainer_config is None:
        print("  Could not load AI model or configuration. GTO stats unavailable.")
        print("------------------------------------")
        return

    try:
        # The game state for encode_state should be the game object itself if
        # it has ``get_initial_state``. Alternatively, it could be
        # ``game.rules`` if that's where ``get_initial_state`` resides.
        # ``CFRTrainer.encode_state`` expects an object with this method, and
        # the ``TexasHoldem`` class provides it.
        state_representation = trainer.encode_state(game)  # Pass the main game object

        # Get strategy from the AI model
        # This returns a list of probabilities for num_actions
        ai_strategy_probabilities = trainer.get_strategy(state_representation)

        if len(ai_strategy_probabilities) != trainer_config["num_actions"]:
            print(
                f"  Error: AI model returned strategy of length {len(ai_strategy_probabilities)}, "
                f"but config expects {trainer_config['num_actions']}."
            )
            print("------------------------------------")
            return

        print("The AI model suggests the following probabilities for your actions:")

        game_rules = game.rules  # Assumes game object has a .rules attribute like TexasHoldem
        player_chips = game_rules.player_chips[player_index]

        amount_to_call_for_player = game_rules.current_bet - game_rules.bets[player_index]

        for i in range(trainer_config["num_actions"]):
            prob = ai_strategy_probabilities[i]
            action_str, action_amount = get_action_from_index(i, game_rules, player_chips)

            display_action = ""
            if action_str == "fold":
                display_action = f"Fold: {prob*100:.1f}%"
            elif action_str == "check":
                display_action = f"Check: {prob*100:.1f}%"
            elif action_str == "call":
                # get_action_from_index for call (index 2) uses game_rules.current_bet for amount.
                # This might not be the "actual amount to call" for the current player.
                # For display, it's better to show the player's actual call amount.
                if amount_to_call_for_player > 0:
                    display_action = f"Call {amount_to_call_for_player} chips: {prob*100:.1f}%"
                else:  # Player can check, so "Call 0"
                    display_action = f"Call 0 chips (Check): {prob*100:.1f}%"
            elif action_str == "raise":
                # Determine the description for the raise based on index
                # This is for display only, get_action_from_index gives the amount
                descriptions = {
                    3: "25% pot",
                    4: "50% pot",
                    5: "75% pot",
                    6: "100% pot",
                    7: "150% pot",
                    8: "200% pot",
                    9: "All-in",  # Index 9 is all-in
                }
                desc = descriptions.get(i, "")
                if i == 9:  # All-in
                    display_action = f"All-in ({action_amount} chips): {prob*100:.1f}%"
                else:
                    display_action = f"Raise to {action_amount} ({desc}): {prob*100:.1f}%"
            else:
                display_action = f"Action {i} ({action_str}, {action_amount}): {prob*100:.1f}%"

            print(f"  {i+1}. {display_action}")

        print("\nNotes:")
        print("- Legality of actions (e.g., 'Check' if facing a bet) is determined by game rules.")
        print(
            "- Raise amounts shown are calculated based on the current pot size and player stack."
        )

    except Exception as e:
        print(f"  Error during AI GTO analysis: {e}")
        import traceback

        traceback.print_exc()  # For debugging

    print("------------------------------------")


if __name__ == "__main__":
    # This is a placeholder for testing.
    # To run this, you'd need a mock game object and a trained model + config.
    print("AI GTO Analyzer Module")
    print(
        "To test, integrate with human_vs_ai.py and ensure a trained model "
        "and config.yaml are available."
    )

    # Example of how it might be called (conceptual)
    # from game_engine.texas_holdem import TexasHoldem
    # from playStrategy import HumanStrategy # To get a game instance setup

    # # 1. Setup a game
    # # This setup is a bit circular for standalone testing, as TexasHoldem needs strategies.
    # # For a real test, you'd run human_vs_ai.py

    # # Mocking enough of the game and rules for display_ai_gto_stats to run
    # class MockRules:
    #     def __init__(
    #         self,
    #         num_players,
    #         pot,
    #         current_bet,
    #         player_bets,
    #         player_chips_list,
    #         active_players_list,
    #     ):
    #         self.num_players = num_players
    #         self.pot = pot
    #         self.current_bet = current_bet # Max bet this round
    #         self.bets = player_bets # Bets this round by each player
    #         self.player_chips = player_chips_list
    #         self.active_players = active_players_list
    #         self.hands = [['Ah', 'Kh'], ['Qd', 'Js']] # Example
    #         self.community_cards = ['Th', 'Jh', '2c'] # Example

    # class MockGame:
    #     def __init__(self, rules):
    #         self.rules = rules
    #         self.num_players = rules.num_players

    #     def get_initial_state(self): # Expected by CFRTrainer.encode_state
    #         # Return a dummy numpy array of a shape that encode_state can handle
    #         # This shape needs to be compatible with the loaded model's input_shape config
    #         # Example: (depth, height, width) = (7, 13, 4)
    #         # Based on config: (13,4,7) for trainer, so input is (7,13,4)
    #         # For now, let's assume a dummy shape that might work with a default config.
    #         # The actual shape comes from TexasHoldem.get_initial_state()
    #         # num_players = self.num_players
    #         # RANKS_LEN = 13
    #         # SUITS_LEN = 4
    #         # return np.zeros((num_players + 5, RANKS_LEN, SUITS_LEN))
    #         # The CFRTrainer's encode_state will resize this.
    #         # Let's use what CFRTrainer might expect as self.input_shape (H, W, C)
    #         # So, if trainer_config['input_shape'] is (13,4,7), we provide that.
    #         # No, encode_state takes the raw game state output and *resizes* it.
    #         # So we provide something like the true game output.
    #         # For this test, the content doesn't matter as much as the types.
    #         # We need a real model to test `get_strategy`.
    #         # This test block is more for checking `display_ai_gto_stats` structure.
    #         return np.random.rand(7, 13, 4) # Dummy state (Channels, Height, Width) like

    # print("\nAttempting conceptual test of display_ai_gto_stats (requires model):")
    # # Setup mock parameters
    # mock_player_bets = [20, 0] # Player 0 (BB), Player 1 (SB) hasn't acted on BB
    # mock_player_chips = [980, 990]
    # mock_active_players = [True, True]
    # mock_rules_obj = MockRules(num_players=2, pot=30, current_bet=20,
    #                            player_bets=mock_player_bets,
    #                            player_chips_list=mock_player_chips,
    #                            active_players_list=mock_active_players)
    # mock_game_obj = MockGame(mock_rules_obj)
    # human_player_idx = 1 # Player 1 to act

    # # This will try to load the model as specified in config.yaml
    # # Ensure 'trained_models/cfr_model.pth' exists and config.yaml is correct for it.
    # # display_ai_gto_stats(mock_game_obj, human_player_idx)

    # print(
    #     "Conceptual test structure is present. Full test requires running"
    #     " human_vs_ai.py with a model."
    # )
    pass  # End of main guard
