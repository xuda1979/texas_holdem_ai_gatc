# gto_analysis.py

from treys import Card, Evaluator

def calculate_pot_odds(pot_size: int, bet_to_call: int) -> float | None:
    """
    Calculates pot odds.
    Pot Odds = (Bet to Call) / (Pot Size + Bet to Call)
    Returns the odds as a percentage, or None if bet_to_call is 0.
    """
    if bet_to_call <= 0:
        return None
    total_pot_if_call = pot_size + bet_to_call
    # The odds are often expressed as "X:1", meaning you need to win 1 out of (X+1) times.
    # Here, we calculate the required equity: bet_to_call / (pot_size_before_call + bet_to_call + your_call)
    # Or, more simply: amount_to_win / amount_risked
    # Amount risked = bet_to_call
    # Amount can win = pot_size + bet_to_call (from opponent, if they made the current bet)
    # Simplified: bet_to_call / (pot_size + bet_to_call + bet_to_call) if pot_size is what's ALREADY in there
    # Correct: Required equity = Cost of Call / (Total Pot After Call)
    # Total Pot After Call = Current Pot + All Bets in Current Round (including the bet you are calling)

    # Let's use the common definition: Pot Odds = Pot Size / Bet to Call
    # This gives a ratio like 3:1, meaning you need to win 1 out of 4 times (25% equity).
    # Or, required equity = Bet to Call / (Pot Size + Bet to Call)

    # Simpler: what percentage of the final pot your call will represent
    # If pot is 100, opponent bets 50 (pot becomes 150). You call 50. Final pot 200. Your call is 50.
    # You risk 50 to win 150. Odds are 150:50 or 3:1. Equity needed = 50 / (150 + 50) = 50 / 200 = 25%.

    # Let pot_size be the size of the pot *before* the bet_to_call was made by opponent.
    # So, if current pot shows P, and opponent bets B, the pot is now P+B.
    # You need to call B.
    # You are risking B to win (P+B).
    # Your equity needed = B / ( (P+B) + B ) = B / (P + 2B)

    # Let's redefine arguments for clarity based on game state:
    # pot_size_before_current_bet_round: The pot accumulated from previous streets.
    # current_pot_total: The pot including all bets from the current street so far.
    # bet_to_call: The amount the current player needs to call.

    # If `pot_size` is the total money in the pot *including* the opponent's bet you are considering calling.
    # And `bet_to_call` is the amount you must put in.
    # Then you are risking `bet_to_call` to win `pot_size`.
    # The equity you need is `bet_to_call / (pot_size + bet_to_call)`
    if (pot_size + bet_to_call) == 0: # Avoid division by zero if pot is 0 and bet is 0 (though caught by bet_to_call <=0)
        return None

    required_equity = bet_to_call / (pot_size + bet_to_call)
    return required_equity * 100 # As percentage


def calculate_mdf(pot_size_before_bet: int, bet_size: int) -> float | None:
    """
    Calculates Minimum Defense Frequency (MDF).
    MDF = Pot Size / (Pot Size + Bet Size)
    This is the percentage of hands you should continue with to prevent opponent from profitably bluffing.
    Returns MDF as a percentage, or None if bet_size is 0.
    """
    if bet_size <= 0:
        return None
    if (pot_size_before_bet + bet_size) == 0: # Avoid division by zero
        return None
    mdf = pot_size_before_bet / (pot_size_before_bet + bet_size)
    return mdf * 100 # As percentage


def calculate_hand_equity(player_hand: list[str], community_cards: list[str], num_active_opponents: int) -> str:
    """
    Estimates hand equity against a number of random opponent hands.
    This is a simplified version. Full equity calculation can be computationally intensive.
    """
    if not player_hand or len(player_hand) != 2:
        return "Hand equity: Not enough cards to calculate."

    evaluator = Evaluator()

    # Convert player hand and community cards to treys format
    try:
        player_hole_cards = [Card.new(c) for c in player_hand]
        board = [Card.new(c) for c in community_cards]
    except ValueError as e: # Catches Card.new errors for invalid card strings
        # print(f"Debug: Invalid card format for equity calculation. Hand: {player_hand}, Board: {community_cards}. Error: {e}")
        return "Hand equity: Invalid card data."

    if len(board) < 3 and len(board) > 0: # Pre-flop or invalid board state for simple eval
        return "Hand equity: Calculation available post-flop (3+ community cards)."
    if len(board) == 0: # Pre-flop, can give pre-flop equity estimate
        # Pre-flop equity is very complex against multiple opponents and ranges.
        # For simplicity, we'll return a placeholder or use a very basic lookup if available.
        # This part requires a more sophisticated approach (e.g., Monte Carlo simulation or pre-computed tables)
        # For now, let's just acknowledge it's pre-flop
        return "Hand equity: Pre-flop (requires simulation or lookup table)."

    # Simplified Monte Carlo for post-flop equity
    # This is computationally expensive if not careful. Let's do a small number of simulations.
    # For a real GTO tool, this would be more robust.

    # Create a deck and remove known cards
    deck = Card.get_deck_int()
    known_cards_int = player_hole_cards + board

    # Check for duplicates in known_cards_int which Card.new would not catch if strings are e.g. "AS" and "As"
    if len(set(known_cards_int)) != len(known_cards_int):
        # print(f"Debug: Duplicate cards found. Player hand: {player_hand}, Community: {community_cards}")
        return "Hand equity: Duplicate cards detected."

    for card_int in known_cards_int:
        if card_int in deck:
            deck.remove(card_int)
        else:
            # This case should ideally not happen if card conversion and deck are correct
            # print(f"Debug: Card {Card.int_to_str(card_int)} not found in deck for removal.")
            return "Hand equity: Card consistency error."


    wins = 0
    simulations = 1000 # Number of simulations. Increase for accuracy, decrease for speed.

    if num_active_opponents == 0 : # Heads up against one implicit opponent for calculation
        num_active_opponents = 1

    for _ in range(simulations):
        try:
            deck_sample = list(deck) # Make a copy for this simulation run
            random.shuffle(deck_sample)

            # Deal remaining community cards
            remaining_community = 5 - len(board)
            current_board_sim = list(board) # make a copy

            if len(deck_sample) < remaining_community + (2 * num_active_opponents) :
                # Not enough cards in deck_sample for all players and board completion
                # This can happen if num_active_opponents is high and many cards are already out.
                # print(f"Debug: Not enough cards in deck for simulation. Deck: {len(deck_sample)}, Need: {remaining_community + (2 * num_active_opponents)}")
                continue # Skip this simulation if not enough cards

            sim_community_cards = deck_sample[:remaining_community]
            current_board_sim.extend(sim_community_cards)

            # Opponent hands
            opponent_hands_sim = []
            cards_for_opponents = deck_sample[remaining_community : remaining_community + (2 * num_active_opponents)]

            for i in range(num_active_opponents):
                opponent_hands_sim.append(cards_for_opponents[i*2 : (i*2)+2])

            # Evaluate player's hand
            player_score = evaluator.evaluate(current_board_sim, player_hole_cards)

            # Evaluate opponents' hands
            opponent_best_score = float('inf') # Lower score is better in treys
            for opp_hand_int in opponent_hands_sim:
                if len(opp_hand_int) != 2: # Should always be 2
                    # print(f"Debug: Opponent hand generated with {len(opp_hand_int)} cards.")
                    continue
                opp_score = evaluator.evaluate(current_board_sim, opp_hand_int)
                if opp_score < opponent_best_score:
                    opponent_best_score = opp_score

            if player_score < opponent_best_score:
                wins += 1
            elif player_score == opponent_best_score: # Tie
                wins += 0.5 # Count ties as half a win for equity

        except Exception as e:
            # print(f"Error during equity simulation: {e}")
            # This can happen with card conflicts if deck management isn't perfect,
            # or if treys encounters an issue. For robustness, skip faulty simulation.
            continue

    if simulations == 0: return "Hand equity: N/A (simulations failed or not run)"
    equity = (wins / simulations) * 100
    return f"Hand equity (vs {num_active_opponents} random): {equity:.2f}% (based on {simulations} sims)"


def display_gto_stats(game_state, player_index: int):
    """
    Calculates and displays GTO-related statistics for the current player.
    """
    print("\n--- GTO Analysis ---")

    rules = game_state.rules
    pot_total_all_bets = rules.pot # This 'pot' in TexasHoldemRules is the total sum of money from all bets so far.

    player_current_bet_this_round = rules.bets[player_index]
    current_bet_to_match = rules.current_bet # The highest bet made in the current round by any player.

    bet_to_call = current_bet_to_match - player_current_bet_this_round

    # Pot Odds Calculation
    # Pot odds are typically: (amount you stand to win) / (amount you have to call)
    # Amount you stand to win = pot_total_all_bets (which includes opponent's bet if they raised)
    # Amount you have to call = bet_to_call
    # Required equity = bet_to_call / (pot_total_all_bets + bet_to_call)
    if bet_to_call > 0:
        # pot_size_for_odds = pot_total_all_bets (this is the pot *including* the last aggressor's bet)
        # You risk 'bet_to_call' to win 'pot_size_for_odds'.
        # So, equity needed = bet_to_call / (pot_size_for_odds + bet_to_call)
        # Example: Pot = 100. Opponent bets 50. Pot is now 150. You call 50. Final pot = 200.
        # You risk 50 to win 150. Odds = 150:50 = 3:1. Equity needed = 50 / (150+50) = 25%.

        # `pot_total_all_bets` from `game_state.rules.pot` already includes the bet that the current player might be calling.
        # So, the amount to win is `pot_total_all_bets`. The cost is `bet_to_call`.
        # Required equity = `bet_to_call` / (`pot_total_all_bets` + `bet_to_call`)

        # Let's use the definition for `calculate_pot_odds(pot_size_reward, amount_to_risk)`
        # pot_size_reward is current pot size *before* our call, but *after* opponent's bet.
        # amount_to_risk is bet_to_call

        # pot_odds_value = calculate_pot_odds(pot_total_all_bets, bet_to_call)
        # Re-check definition of calculate_pot_odds: pot_size is current total pot, bet_to_call is what we add.
        # If pot is 100 (before opponent's bet), opponent bets 50. Game state pot is 150. Bet to call is 50.
        # Pot odds for player: risk 50 to win (100 + 50) = 150. Odds = 150:50 = 3:1.
        # Required equity = 50 / (150 + 50) = 25%.
        # So, pass `pot_total_all_bets` as `pot_size` to `calculate_pot_odds`.

        pot_odds_equity_needed = calculate_pot_odds(pot_total_all_bets, bet_to_call)
        if pot_odds_equity_needed is not None:
            print(f"  Pot Odds: You need {pot_odds_equity_needed:.2f}% equity to profitably call {bet_to_call}.")
            print(f"            (Risking {bet_to_call} to win pot of {pot_total_all_bets})")
        else:
            print(f"  Pot Odds: N/A (no bet to call)")
    else:
        print(f"  Pot Odds: N/A (no bet to call)")

    # MDF Calculation
    # MDF applies when facing a bet. `bet_size` is the size of the bet the opponent just made.
    # `pot_size_before_bet` is the pot *before* the opponent made that bet.

    # If current_bet_to_match > 0, it means an opponent made a bet/raise.
    # The size of that bet/raise was `rules.previous_raise_amount` if it was a raise,
    # or `current_bet_to_match` if it was an opening bet.
    # The pot *before* this last bet was `pot_total_all_bets - effective_bet_size_faced`.

    effective_bet_faced_by_player = 0
    if current_bet_to_match > 0 : # Facing some bet
        # This is tricky. `rules.previous_raise_amount` is the *last raise amount*.
        # If it was an opening bet, current_bet_to_match is the bet size.
        # If player A bets 10, pot was P. Pot is P+10. Player B raises to 30 (raise of 20). Pot is P+10+20.
        # Player C facing this: current_bet_to_match = 30. previous_raise_amount = 20.
        # The "bet" player C is facing is the full 30 (or the part they need to call).
        # For MDF, the "bet_size" is the last aggressive action's size relative to the pot before it.

        # Let's assume the "bet_size" for MDF is simply `bet_to_call` if we are simplifying.
        # And `pot_size_before_bet` is `pot_total_all_bets - bet_to_call` (pot before opponent's last bet if bet_to_call is that bet)
        # This is only true if bet_to_call *is* the opponent's bet size.
        # More accurate: the bet MDF is calculated against is the opponent's last bet/raise size.

        # If an opponent made a bet of X into a pot of P:
        # MDF applies against X. Pot before bet = P. Pot after bet = P+X.
        # `bet_size` for MDF is X. `pot_size_before_bet` is P.

        # `rules.current_bet` is the total bet amount player needs to match for current street.
        # `rules.previous_raise_amount` is the size of the *last raise*.
        # If `rules.current_bet > 0` and `rules.previous_raise_amount > 0`, it was a raise.
        #   The size of this raise was `rules.previous_raise_amount`.
        #   The pot before this raise was `pot_total_all_bets - rules.previous_raise_amount`.
        # If `rules.current_bet > 0` and `rules.previous_raise_amount == 0` (or contextually an opening bet),
        #   The size of this bet was `rules.current_bet`.
        #   The pot before this bet was `pot_total_all_bets - rules.current_bet`.

        mdf_bet_size = 0
        mdf_pot_before_bet = 0

        if bet_to_call > 0: # Only calculate MDF if facing a bet/raise we haven't matched
            # Heuristic: The "bet" we are facing is effectively `bet_to_call` from our perspective of what we'd lose to a bluff.
            # The pot, if we fold, remains `pot_total_all_bets`. The opponent risked `bet_to_call` (from their stack) to win this.
            # No, this isn't quite right. MDF is about opponent's bet sizing.
            # Let's assume the "bet" is the most recent aggressive action.

            # If `rules.current_bet` > `player_current_bet_this_round`, someone else bet/raised.
            # What was the size of *that* bet/raise?
            # This is `rules.current_bet - (sum of other players' bets that constituted the previous total bet level)`.
            # This is getting complex. `rules.previous_raise_amount` is the most direct measure of the last aggressor's sizing.
            # If `rules.previous_raise_amount` is 0, it means it was an opening bet, equal to `rules.current_bet`.

            last_aggressive_action_size = rules.previous_raise_amount
            if last_aggressive_action_size == 0 and rules.current_bet > 0 : # It was likely an opening bet
                last_aggressive_action_size = rules.current_bet

            if last_aggressive_action_size > 0:
                mdf_bet_size = last_aggressive_action_size
                # Pot before this last aggressive action:
                mdf_pot_before_bet = pot_total_all_bets - mdf_bet_size
                                    # (assuming pot_total_all_bets includes this last aggressive action)
                mdf_value = calculate_mdf(mdf_pot_before_bet, mdf_bet_size)
                if mdf_value is not None:
                    print(f"  Min Defense Freq (MDF): ~{mdf_value:.2f}% (vs opponent's bet of {mdf_bet_size} into pot of {mdf_pot_before_bet})")
                else:
                    print(f"  Min Defense Freq (MDF): N/A (cannot determine opponent's bet size for MDF)")
            else:
                print(f"  Min Defense Freq (MDF): N/A (no bet faced or bet size unclear)")
        else: # Not facing a bet that requires a call (e.g. can check or open bet)
            print(f"  Min Defense Freq (MDF): N/A (not facing a bet)")


    # Hand Equity
    player_hand = rules.hands[player_index]
    community_cards = rules.community_cards

    num_active_opponents = 0
    for i in range(rules.num_players):
        if i != player_index and rules.active_players[i]:
            # Consider only opponents who are still in the hand and have chips to act or are all-in
            if rules.player_chips[i] > 0 or (rules.player_chips[i] == 0 and rules.bets[i] > 0):
                 num_active_opponents += 1

    equity_message = calculate_hand_equity(player_hand, community_cards, num_active_opponents)
    print(f"  {equity_message}")

    print("----------------------")

# Example usage (for testing standalone)
if __name__ == '__main__':
    # Mock game_state and player_index for testing
    class MockPlayer:
        def __init__(self, chips, bet_this_round, active=True):
            self.chips = chips
            self.bet_this_round = bet_this_round # How much this player has put in *this round*
            self.active = active

    class MockRules:
        def __init__(self, num_players, pot, current_bet, previous_raise_amount, small_blind, big_blind):
            self.num_players = num_players
            self.pot = pot # Total pot currently
            self.current_bet = current_bet # Highest bet level to match on current street
            self.previous_raise_amount = previous_raise_amount # Size of the last raise delta
            self.small_blind = small_blind
            self.big_blind = big_blind
            self.hands = [[] for _ in range(num_players)]
            self.community_cards = []
            self.bets = [0] * num_players # Total amount each player has bet *this round*
            self.active_players = [True] * num_players
            self.player_chips = [1000] * num_players


    class MockGameState:
        def __init__(self, rules):
            self.rules = rules

    # Test Pot Odds
    # Scenario 1: Pot is 100. Opponent bets 50. Pot becomes 150. We need to call 50.
    # We risk 50 to win 150. Odds 3:1. Equity needed = 50 / (150 + 50) = 25%.
    print("Pot Odds Tests:")
    print(f"Test 1 (150 pot, 50 to call): Expected 25% -> {calculate_pot_odds(150, 50)}")
    # Scenario 2: Pot is 10. Opponent bets 10. Pot becomes 20. We need to call 10.
    # We risk 10 to win 20. Odds 2:1. Equity needed = 10 / (20 + 10) = 33.33%.
    print(f"Test 2 (20 pot, 10 to call): Expected 33.33% -> {calculate_pot_odds(20, 10)}")
    # Scenario 3: No bet to call
    print(f"Test 3 (100 pot, 0 to call): Expected None -> {calculate_pot_odds(100, 0)}")

    # Test MDF
    # Scenario 1: Pot is 100. Opponent bets 50 (half pot).
    # MDF = 100 / (100 + 50) = 100 / 150 = 66.67%.
    print("\nMDF Tests:")
    print(f"Test 1 (Pot 100, Bet 50): Expected 66.67% -> {calculate_mdf(100, 50)}")
    # Scenario 2: Pot is 100. Opponent bets 100 (pot size bet).
    # MDF = 100 / (100 + 100) = 100 / 200 = 50%.
    print(f"Test 2 (Pot 100, Bet 100): Expected 50% -> {calculate_mdf(100, 100)}")
    # Scenario 3: No bet
    print(f"Test 3 (Pot 100, Bet 0): Expected None -> {calculate_mdf(100, 0)}")

    # Test display_gto_stats
    print("\nDisplay GTO Stats Test:")
    mock_rules = MockRules(num_players=3, pot=150, current_bet=50, previous_raise_amount=50, small_blind=5, big_blind=10)
    mock_rules.hands = [['Ah', 'Kh'], ['Qd', 'Js'], ['7c', '2h']] # Player 0, 1, 2
    mock_rules.community_cards = ['Th', 'Jh', '2c']
    mock_rules.bets = [0, 50, 0] # Player 0 needs to call 50, P1 made the bet, P2 folded or hasn't acted
    mock_rules.active_players = [True, True, True] # Assume all active for equity calc simplicity
    mock_rules.player_chips = [950, 950, 1000]

    game = MockGameState(mock_rules)

    # Player 0's turn, facing a bet of 50. Current bet is 50. Player 0 has 0 in pot this round.
    # Pot is 150 (includes the 50 from player 1).
    # Bet to call for player 0 is 50.
    print("\n--- Player 0's Turn (facing bet) ---")
    display_gto_stats(game, 0)

    # Player 2's turn, option to check or bet. (current_bet = 0)
    mock_rules_check_option = MockRules(num_players=3, pot=20, current_bet=0, previous_raise_amount=0, small_blind=5, big_blind=10)
    mock_rules_check_option.hands = [['Ah', 'Kh'], ['Qd', 'Js'], ['7c', '2h']]
    mock_rules_check_option.community_cards = ['Th', 'Jh', '2c']
    mock_rules_check_option.bets = [10, 10, 0] # P0 (SB), P1 (BB) - BB is 10. P2 to act. current_bet could be BB if preflop.
                                            # Let's make it post-flop, current_bet is 0.
    mock_rules_check_option.bets = [0,0,0] # Post-flop, no bets yet.
    mock_rules_check_option.active_players = [True, True, True]
    mock_rules_check_option.player_chips = [990,990,1000]


    game_check_option = MockGameState(mock_rules_check_option)
    print("\n--- Player 2's Turn (can check) ---")
    display_gto_stats(game_check_option, 2)

    # Test pre-flop equity (should give placeholder)
    print("\n--- Player 0's Turn (Pre-flop equity) ---")
    mock_rules_preflop = MockRules(num_players=2, pot=15, current_bet=10, previous_raise_amount=5, small_blind=5, big_blind=10)
    mock_rules_preflop.hands = [['As', 'Ks'], ['Td', 'Tc']]
    mock_rules_preflop.community_cards = [] # Pre-flop
    mock_rules_preflop.bets = [5, 10] # SB, BB
    mock_rules_preflop.active_players = [True, True]
    mock_rules_preflop.player_chips = [995, 990]
    game_preflop = MockGameState(mock_rules_preflop)
    # Player 0 (SB) to act, facing BB of 10. Player 0 has 5 in. Needs to call 5 more.
    # Pot is 15. Bet to call is 5.
    display_gto_stats(game_preflop, 0)

    # Test equity with specific cards that might cause issues
    print("\n--- Equity Test (Specific Cards) ---")
    player_h = ["Ac", "Ad"] # Pocket Aces
    comm_c = ["Ah", "As", "Kc"] # Trips Aces on board
    # This should be fine, but good to test card representation
    print(calculate_hand_equity(player_h, comm_c, 1))

    # Test with fewer than 3 community cards (but not zero)
    print("\n--- Equity Test (1-2 Community Cards) ---")
    comm_c_one = ["2d"]
    print(calculate_hand_equity(player_h, comm_c_one, 1)) # Should say needs 3+

    # Test with invalid card string
    print("\n--- Equity Test (Invalid Card String) ---")
    player_h_invalid = ["Xx", "Yy"]
    print(calculate_hand_equity(player_h_invalid, comm_c,1))

    # Test equity with duplicate cards between hand and board (conceptual test, Card.new might catch some)
    print("\n--- Equity Test (Duplicate conceptual cards) ---")
    player_h_dup = ["Ac", "Ks"]
    comm_c_dup = ["Ac", "Qh", "Js"] # Ac duplicated
    # Treys Card.new('Ac') will be the same object if called multiple times with same string.
    # The issue is if the *source lists* have duplicates that would make combined list invalid.
    # My logic for deck removal should catch actual int value duplicates.
    print(calculate_hand_equity(player_h_dup, comm_c_dup, 1))

    print("\n--- Equity Test (Empty player hand) ---")
    print(calculate_hand_equity([], comm_c_dup, 1))

import random
