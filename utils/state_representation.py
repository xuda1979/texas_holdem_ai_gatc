"""
Handles the conversion of game state into a numerical representation suitable for a Transformer model.
"""
import torch
from typing import List, Tuple, Any

# Assuming GameState and Player will be importable from these paths
# from game_engine.game_state import GameState
# from game_engine.player import Player

# --- Mock classes for development and testing ---
class MockPlayer:
    def __init__(self, player_id: str, hand: List[str], stack: int):
        self.player_id = player_id
        self.hand = hand
        self.stack = stack
        self.current_bet_in_round = 0 # Player's current contribution in the betting round

class MockGameState:
    def __init__(self, players: List[MockPlayer], community_cards: List[str],
                 pot: int, current_bet: int, betting_round: str,
                 betting_history: List[Tuple[str, Tuple[str, int | None]]],
                 player_order: List[str] | None = None): # player_order can be passed if specific order matters
        self.players_map = {p.player_id: p for p in players} # Renamed from self.players to avoid confusion
        if player_order:
            self.player_order = player_order
        else:
            self.player_order = [p.player_id for p in players] # Default order if not specified
        
        self.community_cards = community_cards
        self.pot = pot
        self.current_bet = current_bet 
        self.betting_round = betting_round
        self.betting_history = betting_history

    def get_player(self, player_id: str) -> MockPlayer | None:
        return self.players_map.get(player_id)

# --- End Mock classes ---

# Use actual classes when available
GameState = MockGameState
Player = MockPlayer


RANK_TO_NUM = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
SUIT_TO_NUM = {'s': 1, 'h': 2, 'd': 3, 'c': 4} # Spades, Hearts, Diamonds, Clubs

ACTION_TO_ID = {'fold': 0, 'check': 1, 'call': 2, 'bet': 3, 'raise': 4}
ROUND_TO_ID = {'pre-flop': 0, 'flop': 1, 'turn': 2, 'river': 3}

# For feature construction as per prompt's examples for d_raw_feature=3:
# Card features: [encoded_card_value, 0, 0] (type implied by value, or could use a specific ID like 0 for card type)
# Action features: [player_id_numeric, action_id_numeric, amount_normalized] (type implied)
# Pot feature: [pot_value_normalized, 0, 1] (type indicator 1)
# Current Bet feature: [bet_value_normalized, 0, 2] (type indicator 2)
# Player Stack feature: [stack_value_normalized, 0, 3] (type indicator 3)
# Round feature: [round_id, 0, 4] (type indicator 4)

# Let's define these type indicators explicitly
TYPE_ID_CARD = 0.0 # Default type for cards, if needed in the third position.
TYPE_ID_POT = 1.0
TYPE_ID_CURRENT_BET = 2.0
TYPE_ID_PLAYER_STACK = 3.0
TYPE_ID_ROUND = 4.0
# Betting actions don't use a type_id in the third position in the prompt's example for d_raw_feature=3,
# as all three positions are used for player_id, action_id, amount.

# Normalization constants (placeholders, ideally should be more dynamic or configurable)
NORM_AMOUNT = 100.0 # e.g. divide amounts by a typical big blind or average pot
NORM_STACK_POT = 100.0 # For stack and pot sizes

def _encode_card(card_str: str) -> float:
    """
    Encodes a card string (e.g., 'Js', 'Td') into a numeric representation.
    Rank + Suit/10. e.g. Jack of Spades (Js) -> 11.1
    """
    if len(card_str) != 2:
        raise ValueError(f"Invalid card string: {card_str}")
    rank, suit = card_str[0].upper(), card_str[1].lower() # Normalize case
    if rank not in RANK_TO_NUM or suit not in SUIT_TO_NUM:
        raise ValueError(f"Invalid card components: {rank}, {suit}")
    return float(RANK_TO_NUM[rank] + SUIT_TO_NUM[suit] / 10.0)

def _get_numeric_player_id(player_id_str: str, all_player_ids_in_order: List[str]) -> int:
    """
    Converts a string player_id to its index in the ordered list of all players.
    """
    try:
        return all_player_ids_in_order.index(player_id_str)
    except ValueError:
        # This error means a player_id from betting history or current_player_id
        # was not found in the game_state's official player list.
        raise ValueError(f"Player ID '{player_id_str}' not found in the game's ordered player list.")


def prepare_transformer_input(
    game_state: GameState,
    current_player_id: str,
    # players_list is not strictly needed if game_state.get_player() and game_state.player_order are robust
    max_seq_len: int,
    d_raw_feature: int
) -> "torch.Tensor": # Use string literal for type hint to avoid import error if torch is not fully functional
    """
    Prepares a game state for input to a Transformer model.
    """
    raw_sequence: List[List[float]] = []
    
    # Ensure d_raw_feature is at least 3 for the specific type indicator strategy
    if d_raw_feature < 3:
        # This specific encoding strategy might not work well.
        # Fallback or error, for now, let's print a warning.
        print(f"Warning: d_raw_feature is {d_raw_feature}, which is less than 3. "
              "The suggested encoding uses 3 features for some items.")
        # The _create_feature_vector will still try its best to fit.

    all_player_ids_ordered = game_state.player_order

    # Helper to create a feature vector of fixed dimension d_raw_feature
    def _create_feature(value1: float, value2: float = 0.0, value3_or_type_id: float = 0.0) -> List[float]:
        """
        Creates a feature vector of size d_raw_feature.
        - For d_raw_feature=3: [value1, value2, value3_or_type_id]
        - For d_raw_feature > 3: [value1, value2, value3_or_type_id, 0, ..., 0]
        - For d_raw_feature < 3: Truncates. e.g. d_raw_feature=1 -> [value1]
        """
        base = [value1, value2, value3_or_type_id]
        if d_raw_feature == 3:
            return base
        elif d_raw_feature > 3:
            return (base + [0.0] * (d_raw_feature - 3))[:d_raw_feature]
        else: # d_raw_feature < 3
            return base[:d_raw_feature]

    # 1. Card Encoding
    current_player_obj = game_state.get_player(current_player_id)
    if not current_player_obj:
        raise ValueError(f"Player {current_player_id} not found in game_state.")

    # Player's hand (2 cards)
    # Uses [encoded_card_value, 0, TYPE_ID_CARD (or 0)]
    for card_str in current_player_obj.hand:
        encoded_card = _encode_card(card_str)
        raw_sequence.append(_create_feature(encoded_card, 0.0, TYPE_ID_CARD))

    # Community cards (0 to 5 cards)
    # Uses [encoded_card_value, 0, TYPE_ID_CARD (or 0)]
    for card_str in game_state.community_cards:
        encoded_card = _encode_card(card_str)
        raw_sequence.append(_create_feature(encoded_card, 0.0, TYPE_ID_CARD))
        
    # 2. Betting History Encoding
    # Uses [player_id_numeric, action_id_numeric, amount_normalized]
    for p_id_str, action_tuple in game_state.betting_history:
        action_name, amount_val = action_tuple
        
        numeric_p_id = float(_get_numeric_player_id(p_id_str, all_player_ids_ordered))
        action_id = float(ACTION_TO_ID.get(action_name.lower(), -1)) # -1 for unknown
        
        normalized_amount = float(amount_val / NORM_AMOUNT if amount_val is not None else 0.0)
        
        raw_sequence.append(_create_feature(numeric_p_id, action_id, normalized_amount))

    # 3. Other Game State Features
    # Pot size: [pot_value_normalized, 0, TYPE_ID_POT]
    normalized_pot = float(game_state.pot / NORM_STACK_POT)
    raw_sequence.append(_create_feature(normalized_pot, 0.0, TYPE_ID_POT))
    
    # Current bet faced by player: [bet_value_normalized, 0, TYPE_ID_CURRENT_BET]
    # This is the additional amount the player needs to call.
    player_bet_in_round = current_player_obj.current_bet_in_round
    effective_bet_faced = max(0, game_state.current_bet - player_bet_in_round)
    normalized_bet_faced = float(effective_bet_faced / NORM_AMOUNT)
    raw_sequence.append(_create_feature(normalized_bet_faced, 0.0, TYPE_ID_CURRENT_BET))

    # Player's current stack size: [stack_value_normalized, 0, TYPE_ID_PLAYER_STACK]
    normalized_stack = float(current_player_obj.stack / NORM_STACK_POT)
    raw_sequence.append(_create_feature(normalized_stack, 0.0, TYPE_ID_PLAYER_STACK))

    # Current betting round: [round_id, 0, TYPE_ID_ROUND]
    round_id_numeric = float(ROUND_TO_ID.get(game_state.betting_round.lower(), -1)) # -1 for unknown
    raw_sequence.append(_create_feature(round_id_numeric, 0.0, TYPE_ID_ROUND))
    
    # 4. Assembling the Sequence (already done by appending to raw_sequence)

    # 5. Padding and Tensor Conversion
    final_sequence: List[List[float]] = []
    for i in range(max_seq_len):
        if i < len(raw_sequence):
            final_sequence.append(raw_sequence[i])
        else:
            final_sequence.append([0.0] * d_raw_feature) # Pad with zero vectors
            
    state_tensor = torch.tensor(final_sequence, dtype=torch.float32)

    if state_tensor.shape != (max_seq_len, d_raw_feature):
        # This error indicates a problem with padding or _create_feature logic
        raise ValueError(f"Final tensor shape is {state_tensor.shape}, expected ({max_seq_len}, {d_raw_feature}).")

    return state_tensor


if __name__ == '__main__':
    D_RAW_FEATURE_EXAMPLE = 3 

    # Mock players
    p0 = MockPlayer(player_id="player_0", hand=['Ah', 'Ks'], stack=1000)
    p1 = MockPlayer(player_id="player_1", hand=['Qc', 'Jd'], stack=800)
    p2 = MockPlayer(player_id="player_2", hand=['7h', '2s'], stack=1200)
    
    # Define player order for consistency in numeric IDs
    player_order_mock = [p0.player_id, p1.player_id, p2.player_id]
    
    # Mock GameState
    # Player p0 is current player. p1 bet 50, p2 raised to 100.
    # So, game_state.current_bet is 100.
    # p0's current_bet_in_round is 0 (e.g. p0 is UTG or hasn't acted).
    # p0 faces 100.
    # p1's current_bet_in_round would be 50. If it was p1's turn, they'd face 100-50=50.
    
    game_state_mock_flop = MockGameState(
        players=[p0, p1, p2], # List of player objects
        player_order=player_order_mock,
        community_cards=['Th', '9d', '8c'], # 3 community cards
        # Initial pot value was 150, then overridden to 500 based on a comment. Using 500 directly.
        pot=500,
        current_bet=100, # Highest bet on table.
        betting_round="flop",
        betting_history=[
            ("player_1", ("bet", 50)),    # p1 (id 1) bets 50. current_bet becomes 50.
            ("player_2", ("raise", 100)), # p2 (id 2) raises to 100. (total bet is 100). current_bet becomes 100.
        ]
    )
    # Update player contributions for calculating effective_bet_faced
    p0.current_bet_in_round = 0 # Current player (p0) has 0 in pot this round. Faces 100.
    p1.current_bet_in_round = 50 # p1 has put in 50.
    p2.current_bet_in_round = 100 # p2 has put in 100.


    current_player_id_mock = "player_0"
    max_seq_len_mock = 20 

    print(f"--- Test with d_raw_feature = {D_RAW_FEATURE_EXAMPLE} (Flop) ---")
    try:
        input_tensor = prepare_transformer_input(
            game_state_mock_flop,
            current_player_id_mock,
            max_seq_len_mock,
            D_RAW_FEATURE_EXAMPLE
        )
        print(f"Output tensor shape: {input_tensor.shape}")
        print("Output tensor:")
        print(input_tensor)
        
        # Expected sequence for player_0 (id 0):
        # Hand: Ah (14.2), Ks (13.1)
        # Comm: Th (10.2), 9d (9.3), 8c (8.4)
        # History: (p1, bet, 50), (p2, raise, 100)
        # Pot: 500, Current Bet faced: 100, Stack: 1000, Round: flop (1)
        # Normalization: amounts/100, stacks_pots/100
        #
        # Tokens (d_raw_feature=3):
        # 1. Card Ah: [14.2, 0.0, 0.0 (TYPE_ID_CARD)]
        # 2. Card Ks: [13.1, 0.0, 0.0 (TYPE_ID_CARD)]
        # 3. Comm Th: [10.2, 0.0, 0.0 (TYPE_ID_CARD)]
        # 4. Comm 9d: [ 9.3, 0.0, 0.0 (TYPE_ID_CARD)]
        # 5. Comm 8c: [ 8.4, 0.0, 0.0 (TYPE_ID_CARD)]
        # 6. Hist p1 bet 50: [1.0 (p1_id), 3.0 (bet_id), 0.5 (50/100)]
        # 7. Hist p2 raise 100: [2.0 (p2_id), 4.0 (raise_id), 1.0 (100/100)]
        # 8. Pot: [5.0 (500/100), 0.0, 1.0 (TYPE_ID_POT)]
        # 9. Bet Faced: [1.0 (100/100), 0.0, 2.0 (TYPE_ID_CURRENT_BET)] (p0 has 0 in pot, faces 100)
        # 10. Stack: [10.0 (1000/100), 0.0, 3.0 (TYPE_ID_PLAYER_STACK)]
        # 11. Round: [1.0 (flop_id), 0.0, 4.0 (TYPE_ID_ROUND)]
        # Total 11 tokens. Rest are padding [0.0, 0.0, 0.0] up to max_seq_len=20.

        print("\nVerifying selected token values (approximated for clarity):")
        print(f"Token 0 (Player Card Ah): {input_tensor[0].tolist()}") # Expected: [14.2, 0.0, 0.0]
        print(f"Token 5 (Hist p1 bet 50): {input_tensor[5].tolist()}") # Expected: [1.0, 3.0, 0.5]
        print(f"Token 7 (Pot): {input_tensor[7].tolist()}")          # Expected: [5.0, 0.0, 1.0]
        print(f"Token 8 (Bet Faced): {input_tensor[8].tolist()}")    # Expected: [1.0, 0.0, 2.0]
        print(f"Token 10 (Round): {input_tensor[10].tolist()}")      # Expected: [1.0, 0.0, 4.0]
        print(f"Token 11 (Padding): {input_tensor[11].tolist()}")    # Expected: [0.0, 0.0, 0.0]


        # Test with d_raw_feature = 1
        D_RAW_FEATURE_SMALL = 1
        print(f"\n--- Test with d_raw_feature = {D_RAW_FEATURE_SMALL} ---")
        input_tensor_small = prepare_transformer_input(
            game_state_mock_flop, current_player_id_mock, max_seq_len_mock, D_RAW_FEATURE_SMALL
        )
        print(f"Output tensor shape: {input_tensor_small.shape}")
        # print(input_tensor_small) # Will be very truncated

        # Test with d_raw_feature = 5
        D_RAW_FEATURE_LARGE = 5
        print(f"\n--- Test with d_raw_feature = {D_RAW_FEATURE_LARGE} ---")
        input_tensor_large = prepare_transformer_input(
            game_state_mock_flop, current_player_id_mock, max_seq_len_mock, D_RAW_FEATURE_LARGE
        )
        print(f"Output tensor shape: {input_tensor_large.shape}")
        print(f"Token 0 (Player Card Ah, d=5): {input_tensor_large[0].tolist()}") # Expected [14.2, 0.0, 0.0, 0.0, 0.0]
        print(f"Token 7 (Pot, d=5): {input_tensor_large[7].tolist()}")          # Expected [5.0, 0.0, 1.0, 0.0, 0.0]


        # Test pre-flop, empty community cards, no betting history (only blinds)
        p0_preflop = MockPlayer(player_id="player_0", hand=['Ac', 'Ad'], stack=1000)
        p1_preflop = MockPlayer(player_id="player_1", hand=['Kh', 'Kd'], stack=1000) # SB
        p2_preflop = MockPlayer(player_id="player_2", hand=['Qh', 'Qd'], stack=1000) # BB
        
        player_order_preflop = [p0_preflop.player_id, p1_preflop.player_id, p2_preflop.player_id]
        # Assume p0 is UTG, p1 SB, p2 BB. Action is on p0.
        # SB posts 5, BB posts 10. Pot = 15. Current bet = 10.
        p0_preflop.current_bet_in_round = 0
        p1_preflop.current_bet_in_round = 5 # Small Blind
        p1_preflop.stack -=5
        p2_preflop.current_bet_in_round = 10 # Big Blind
        p2_preflop.stack -=10

        game_state_preflop = MockGameState(
            players=[p0_preflop, p1_preflop, p2_preflop],
            player_order=player_order_preflop,
            community_cards=[],
            pot=15, # SB + BB
            current_bet=10, # BB is the current bet
            betting_round="pre-flop",
            betting_history=[ # History might include blind postings
                ("player_1", ("bet", 5)), # SB, often "bet" or "blind"
                ("player_2", ("bet", 10)),# BB
            ]
        )
        current_player_preflop = "player_0"
        print(f"\n--- Test Pre-flop, d_raw_feature = {D_RAW_FEATURE_EXAMPLE} ---")
        input_tensor_preflop = prepare_transformer_input(
            game_state_preflop, current_player_preflop, max_seq_len_mock, D_RAW_FEATURE_EXAMPLE
        )
        print(f"Output tensor shape: {input_tensor_preflop.shape}")
        print(input_tensor_preflop[0:7]) # Print first few tokens
        # Expected for player_0:
        # Hand: Ac (14.4), Ad (14.3)
        # Comm: None
        # History: (p1, bet, 5), (p2, bet, 10)
        # Pot: 15, Current Bet faced: 10, Stack: 1000, Round: pre-flop (0)
        # Tokens:
        # 1. Card Ac: [14.4, 0.0, 0.0]
        # 2. Card Ad: [14.3, 0.0, 0.0]
        # 3. Hist p1 bet 5: [1.0 (p1_id), 3.0 (bet_id), 0.05 (5/100)]
        # 4. Hist p2 bet 10: [2.0 (p2_id), 3.0 (bet_id), 0.1 (10/100)]
        # 5. Pot: [0.15 (15/100), 0.0, 1.0]
        # 6. Bet Faced: [0.1 (10/100), 0.0, 2.0] (p0 has 0 in pot, faces 10)
        # 7. Stack: [10.0 (1000/100), 0.0, 3.0]
        # 8. Round: [0.0 (preflop_id), 0.0, 4.0]
        # Total 8 tokens.

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing helper _encode_card ---")
    print(f"Encoding 'As': {_encode_card('As')}") # Expected: 14.1
    print(f"Encoding '2c': {_encode_card('2C')}") # Expected: 2.4 (handles mixed case)

    print("\n--- Testing helper _get_numeric_player_id ---")
    print(f"ID for 'player_1' in {player_order_mock}: {_get_numeric_player_id('player_1', player_order_mock)}") # Expected: 1
    try:
        _get_numeric_player_id("player_x", player_order_mock)
    except ValueError as e:
        print(f"Correctly caught error for 'player_x': {e}")
