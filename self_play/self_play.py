from game_engine.texas_holdem import TexasHoldem, TexasHoldemRules
from game_engine.game_state import GameState as AI_GameState # AI_GameState uses structured betting_history
from game_engine.player import Player as AI_Player
from utils.state_representation import prepare_transformer_input
from utils.action_mapping import get_action_from_index
import torch
import copy
import random # Added import
from typing import List, Tuple, Dict, Any

class DummyStrategy:
    def choose_action(self, game_rules_obj: TexasHoldemRules, player_index: int) -> Tuple[str, int | None]:
        return 'fold', None

class SelfPlay:
    def __init__(self, cfr_trainer, game_engine_config: Dict[str, Any], num_ai_players: int = 1):
        self.cfr_trainer = cfr_trainer 
        self.num_ai_players = num_ai_players
        self.ai_player_idx = 0 

        num_total_players = game_engine_config.get('num_players', 2)
        self.game_engine_base_config = {
            'num_players': num_total_players,
            'starting_stack': game_engine_config.get('starting_stack', 0),
            'player_strategies': [DummyStrategy() for _ in range(num_total_players)],
        }
        self.game_engine = TexasHoldem(**self.game_engine_base_config)
        # Apply blind settings after creation to avoid unexpected kwargs
        self.game_engine.rules.big_blind = game_engine_config.get('big_blind', 10)
        self.game_engine.rules.small_blind = game_engine_config.get('small_blind', 5)

    def _is_action_valid(self, engine_rules: TexasHoldemRules, player_idx: int, action_str: str, amount: int | None) -> bool:
        player_chips = engine_rules.player_chips[player_idx]
        player_current_bet_in_round = engine_rules.bets[player_idx]
        game_current_bet = engine_rules.current_bet
        amount_to_call = game_current_bet - player_current_bet_in_round

        if action_str == 'fold':
            return True
        
        if action_str == 'check':
            return amount_to_call == 0
        
        if action_str == 'call':
            if amount_to_call <= 0: return False 
            return player_chips >= amount_to_call 
            
        if action_str == 'bet':
            if game_current_bet != 0: return False
            if amount is None or amount <= 0: return False
            # If all-in, it's valid even if less than big blind (as long as it's > 0, checked above)
            if amount < engine_rules.big_blind and amount != player_chips: 
                return False 
            return player_chips >= amount

        if action_str == 'raise':
            if game_current_bet == 0: return False 
            if amount is None or amount <= game_current_bet: return False # Total bet amount must be greater
            
            raise_amount_committed = amount - player_current_bet_in_round # The actual chips player needs to add to the pot
            if player_chips < raise_amount_committed: return False # Cannot afford the raise

            # Check if the raise increment is valid
            actual_raise_increment = amount - game_current_bet
            min_raise_increment = engine_rules.previous_raise_amount if engine_rules.previous_raise_amount > 0 else engine_rules.big_blind
            
            if actual_raise_increment < min_raise_increment:
                # If the raise increment is too small, it's only valid if the player is going all-in
                # and this all-in amount constitutes a raise (i.e. amount > game_current_bet, checked above)
                if raise_amount_committed == player_chips: # Player is going all-in
                    return True # All-in is a valid under-raise
                return False # Increment too small and not an all-in
            return True # Raise increment is valid
        
        return False

    def _populate_ai_gamestate(self, engine_rules: TexasHoldemRules, current_game_ai_state: AI_GameState) -> AI_GameState:
        ai_gs = current_game_ai_state 
        ai_gs.pot = engine_rules.pot
        ai_gs.community_cards = list(engine_rules.community_cards)
        ai_gs.current_bet = engine_rules.current_bet
        num_community_cards = len(engine_rules.community_cards)
        if num_community_cards == 0: ai_gs.current_round = 'pre-flop'
        elif num_community_cards == 3: ai_gs.current_round = 'flop'
        elif num_community_cards == 4: ai_gs.current_round = 'turn'
        elif num_community_cards == 5: ai_gs.current_round = 'river'
        else: ai_gs.current_round = 'unknown' 
        ai_gs_players_list: List[AI_Player] = []
        for i in range(engine_rules.num_players):
            player_id_str = str(i)
            stack_size = engine_rules.player_chips[i]
            ai_player_obj = AI_Player(player_id=player_id_str, stack_size=stack_size)
            if i == self.ai_player_idx:
                if engine_rules.hands and i < len(engine_rules.hands):
                     ai_player_obj.hand = list(engine_rules.hands[i])
                else: ai_player_obj.hand = [] 
            ai_player_obj.current_bet_in_round = engine_rules.bets[i]
            ai_gs_players_list.append(ai_player_obj)
        ai_gs.players = ai_gs_players_list
        return ai_gs

    def _get_opponent_action(self, engine_rules_obj: TexasHoldemRules, player_index: int) -> Tuple[str, int | None]:
        player_chips = engine_rules_obj.player_chips[player_index]
        player_current_bet_in_round = engine_rules_obj.bets[player_index]
        game_current_bet = engine_rules_obj.current_bet
        amount_to_call = game_current_bet - player_current_bet_in_round

        chosen_action_str = 'fold' # Default safe action
        chosen_amount_for_engine = None

        if amount_to_call == 0: # Can check or bet
            rand_val = random.random()
            if rand_val < 0.7: # 70% chance to check
                chosen_action_str, chosen_amount_for_engine = 'check', None
            else: # 30% chance to bet
                bet_total_amount = int(engine_rules_obj.pot * 0.5)
                bet_total_amount = max(bet_total_amount, engine_rules_obj.big_blind)
                bet_total_amount = min(bet_total_amount, player_chips) # Cap at player's stack

                if self._is_action_valid(engine_rules_obj, player_index, 'bet', bet_total_amount):
                    chosen_action_str, chosen_amount_for_engine = 'bet', bet_total_amount
                else: # Bet is invalid (e.g. player_chips is 0, or calculated bet is 0 and invalid)
                    chosen_action_str, chosen_amount_for_engine = 'check', None # Fallback to check
        else: # Facing a bet/raise (amount_to_call > 0)
            rand_val = random.random()
            if rand_val < 0.1: # 10% chance to fold
                chosen_action_str, chosen_amount_for_engine = 'fold', None
            elif rand_val < 0.3: # 20% chance to attempt raise (cumulative 0.1 + 0.2 = 0.3)
                # Try to raise: e.g., raise by 75% of current pot size, min increment is big blind
                raise_increment = int(engine_rules_obj.pot * 0.75)
                raise_increment = max(raise_increment, engine_rules_obj.big_blind)
                raise_increment = min(raise_increment, player_chips - amount_to_call) # Cap increment if it makes player all-in

                if raise_increment <=0 : # cannot make a positive raise increment (e.g. already all in to call)
                    # Fallback to call if possible
                    if self._is_action_valid(engine_rules_obj, player_index, 'call', game_current_bet):
                         chosen_action_str, chosen_amount_for_engine = 'call', None # Engine calculates call amount
                    else: # Cannot call
                         chosen_action_str, chosen_amount_for_engine = 'fold', None
                else:
                    # Validate the proposed raise. `_is_action_valid` for 'raise' expects the *total* bet amount.
                    proposed_total_bet_for_validation = game_current_bet + raise_increment
                    if self._is_action_valid(engine_rules_obj, player_index, 'raise', proposed_total_bet_for_validation):
                        chosen_action_str, chosen_amount_for_engine = 'raise', raise_increment
                    else: # Raise is invalid or unaffordable, try to call
                        if self._is_action_valid(engine_rules_obj, player_index, 'call', game_current_bet):
                            chosen_action_str, chosen_amount_for_engine = 'call', None
                        else: # Cannot call
                            chosen_action_str, chosen_amount_for_engine = 'fold', None
            else: # 70% chance to call
                if self._is_action_valid(engine_rules_obj, player_index, 'call', game_current_bet):
                    chosen_action_str, chosen_amount_for_engine = 'call', None
                else: # Cannot call
                    chosen_action_str, chosen_amount_for_engine = 'fold', None
        
        # Final safety net: if chosen action is somehow still invalid, default to fold or check.
        # This logic should ideally be covered by the decision paths above.
        # For 'call', _is_action_valid needs the total bet amount (game_current_bet)
        # For 'bet', _is_action_valid needs the total bet amount
        # For 'raise', _is_action_valid needs the total bet amount
        # The `chosen_amount_for_engine` is what process_action expects (None for call, total for bet, increment for raise)
        
        # Re-construct the 'amount' argument for _is_action_valid based on action type
        amount_for_validation = None
        if chosen_action_str == 'bet':
            amount_for_validation = chosen_amount_for_engine
        elif chosen_action_str == 'raise':
            # Must calculate total bet for validation if chosen_amount_for_engine is the increment
            if chosen_amount_for_engine is not None:
                 amount_for_validation = game_current_bet + chosen_amount_for_engine
            else: # Should not happen if raise is chosen with None amount
                 pass # Keep it None, will likely fail validation or be fold
        elif chosen_action_str == 'call':
            amount_for_validation = game_current_bet # Call is to match current game bet

        if not self._is_action_valid(engine_rules_obj, player_index, chosen_action_str, amount_for_validation):
            # print(f"Warning: Opponent action {chosen_action_str}, {chosen_amount_for_engine} (valid. amount: {amount_for_validation}) was chosen but is invalid. Fallback.")
            if amount_to_call == 0: # Original situation was check/bet
                chosen_action_str, chosen_amount_for_engine = 'check', None
            # Check if player can call the original amount_to_call
            elif self._is_action_valid(engine_rules_obj, player_index, 'call', game_current_bet):
                chosen_action_str, chosen_amount_for_engine = 'call', None
            else: # Must fold
                chosen_action_str, chosen_amount_for_engine = 'fold', None
                
        return chosen_action_str, chosen_amount_for_engine

    def _simulate_hand_outcome(self, temp_game_engine_state: TexasHoldemRules, ai_player_idx_for_payoff: int) -> float:
        sim_engine = TexasHoldem(**self.game_engine_base_config)
        sim_engine.rules = copy.deepcopy(temp_game_engine_state)
        num_active_in_sim_rules = sum(1 for active in sim_engine.rules.active_players if active)
        sim_engine.end_game_early = num_active_in_sim_rules < 2
        initial_chips = sim_engine.rules.player_chips[ai_player_idx_for_payoff]
        ALL_STAGES = ["pre-flop", "flop", "turn", "river"]
        num_community_cards = len(sim_engine.rules.community_cards)
        current_stage_index = 0
        if num_community_cards == 0: current_stage_index = 0
        elif num_community_cards == 3: current_stage_index = 1
        elif num_community_cards == 4: current_stage_index = 2
        elif num_community_cards == 5: current_stage_index = 3
        else: current_stage_index = 4 
        for i in range(current_stage_index, len(ALL_STAGES)):
            stage_name = ALL_STAGES[i]
            if sim_engine.end_game_early: break
            if stage_name == "flop" and len(sim_engine.rules.community_cards) == 0: sim_engine.play_stage("flop")
            elif stage_name == "turn" and len(sim_engine.rules.community_cards) == 3: sim_engine.play_stage("turn")
            elif stage_name == "river" and len(sim_engine.rules.community_cards) == 4: sim_engine.play_stage("river")
            if sim_engine.rules.actions_this_round == 0: 
                if stage_name != "pre-flop": 
                    sim_engine.rules.current_player = (sim_engine.rules.dealer_button + 1) % sim_engine.rules.num_players
                    for _ in range(sim_engine.rules.num_players): 
                        if sim_engine.rules.active_players[sim_engine.rules.current_player] and \
                           sim_engine.rules.player_chips[sim_engine.rules.current_player] > 0: break
                        sim_engine.rules.current_player = (sim_engine.rules.current_player + 1) % sim_engine.rules.num_players
            active_betting_round = True
            if sim_engine.rules.betting_round_is_over(): active_betting_round = False
            while active_betting_round and not sim_engine.end_game_early:
                current_player_sub_sim = sim_engine.rules.current_player
                is_player_able_to_act = sim_engine.rules.active_players[current_player_sub_sim] and \
                                        sim_engine.rules.player_chips[current_player_sub_sim] > 0
                if not is_player_able_to_act:
                    sim_engine.rules.advance_turn()
                    if sim_engine.rules.betting_round_is_over(): active_betting_round = False
                    continue 
                action_str, amount_val = self._get_opponent_action(sim_engine.rules, current_player_sub_sim)
                sim_engine.process_action(current_player_sub_sim, action_str, amount_val)
                sim_engine.rules.actions_this_round += 1
                num_active_in_sim_rules = sum(1 for active_p in sim_engine.rules.active_players if active_p)
                if num_active_in_sim_rules < 2: sim_engine.end_game_early = True
                if sim_engine.rules.betting_round_is_over(): active_betting_round = False
            if not sim_engine.end_game_early:
                sim_engine.rules.end_betting_round_cleanup() 
        if not sim_engine.end_game_early:
            sim_engine.perform_showdown() 
        final_chips = sim_engine.rules.player_chips[ai_player_idx_for_payoff]
        payoff = float(final_chips - initial_chips)
        return payoff

    def play_hand_for_training(self):
        training_data_for_hand = []
        self.game_engine.initialize_game()
        main_ai_gs = AI_GameState()

        # Incorporate initial blind actions into main_ai_gs.betting_history
        initial_actions = self.game_engine.current_hand_initial_actions
        if initial_actions: # Ensure it's not None or empty if that's possible
            print(f"Recording initial blind actions: {initial_actions}")
            for player_id_str, action_detail_tuple in initial_actions:
                main_ai_gs.record_action(player_id_str, action_detail_tuple)
        
        # Now populate the rest of the game state. Betting history will be preserved and include blinds.
        main_ai_gs = self._populate_ai_gamestate(self.game_engine.rules, main_ai_gs)
        stages = ["pre-flop", "flop", "turn", "river"]
        for stage_name in stages:
            print(f"\n--- Starting Stage: {stage_name.upper()} ---")
            if self.game_engine.end_game_early: 
                print(f"Game ended early before {stage_name} stage.")
                break
            if stage_name != "pre-flop": self.game_engine.play_stage(stage_name) 
            main_ai_gs = self._populate_ai_gamestate(self.game_engine.rules, main_ai_gs)
            print(f"Community cards: {main_ai_gs.community_cards}")
            print(f"Pot: {main_ai_gs.pot}, Current Bet by engine: {main_ai_gs.current_bet}")
            if stage_name != "pre-flop": 
                 self.game_engine.rules.current_player = (self.game_engine.rules.dealer_button + 1) % self.game_engine.rules.num_players
                 for _ in range(self.game_engine.rules.num_players):
                    if self.game_engine.rules.active_players[self.game_engine.rules.current_player] and \
                       self.game_engine.rules.player_chips[self.game_engine.rules.current_player] > 0: break
                    self.game_engine.rules.current_player = (self.game_engine.rules.current_player + 1) % self.game_engine.rules.num_players
            print(f"Betting round for {stage_name} starts. Player to act: {self.game_engine.rules.current_player}")
            active_betting_round_main_loop = True
            if self.game_engine.rules.betting_round_is_over(): active_betting_round_main_loop = False
            
            while active_betting_round_main_loop and not self.game_engine.end_game_early:
                current_player_engine_idx = self.game_engine.rules.current_player
                is_player_able_to_act = self.game_engine.rules.active_players[current_player_engine_idx] and \
                                        self.game_engine.rules.player_chips[current_player_engine_idx] > 0
                if not is_player_able_to_act:
                    self.game_engine.rules.advance_turn() 
                    if self.game_engine.rules.betting_round_is_over(): active_betting_round_main_loop = False
                    continue 
                
                action_tuple = None
                action_to_engine_str = None
                amount_for_engine = None

                if current_player_engine_idx == self.ai_player_idx:
                    current_ai_view_gs = self._populate_ai_gamestate(self.game_engine.rules, main_ai_gs)
                    if not hasattr(self.cfr_trainer, 'config') or not hasattr(self.cfr_trainer, 'model') or \
                       not hasattr(self.cfr_trainer.model, 'num_actions'): 
                        raise AttributeError("cfr_trainer missing 'config', 'model', or model.num_actions attributes.")
                    
                    model_config = self.cfr_trainer.config.get('model', {})
                    max_seq_len = model_config.get('max_seq_len', 20) 
                    d_raw_feature = model_config.get('d_raw_feature', 3)
                    state_tensor = prepare_transformer_input(current_ai_view_gs, str(self.ai_player_idx), 
                                                             current_ai_view_gs.players, max_seq_len, d_raw_feature)
                    strategy_probs_tensor = self.cfr_trainer.model(state_tensor.unsqueeze(0)).squeeze(0)
                    if not torch.all(strategy_probs_tensor >= 0):
                        print(f"Warning: strategy_probs_tensor has negative values: {strategy_probs_tensor}. Using uniform.")
                        strategy_probs_tensor = torch.ones_like(strategy_probs_tensor) / strategy_probs_tensor.numel() 
                    if not torch.isclose(torch.sum(strategy_probs_tensor), torch.tensor(1.0), atol=1e-6):
                         print(f"Warning: strategy_probs_tensor does not sum to 1: {torch.sum(strategy_probs_tensor)}. Normalizing.")
                         strategy_probs_tensor = strategy_probs_tensor / torch.sum(strategy_probs_tensor)

                    counterfactual_payoffs = torch.zeros(self.cfr_trainer.model.num_actions)
                    for k_action_idx in range(self.cfr_trainer.model.num_actions):
                        temp_engine_rules_for_k = copy.deepcopy(self.game_engine.rules)
                        temp_ai_gs_for_k = self._populate_ai_gamestate(temp_engine_rules_for_k, main_ai_gs)
                        action_str_k, amount_k = get_action_from_index(
                            k_action_idx, temp_ai_gs_for_k, temp_engine_rules_for_k.player_chips[self.ai_player_idx]
                        )
                        is_valid_k = self._is_action_valid(
                            temp_engine_rules_for_k, self.ai_player_idx, action_str_k, amount_k
                        )
                        if is_valid_k:
                            sim_engine_for_k = TexasHoldem(**self.game_engine_base_config)
                            sim_engine_for_k.rules = temp_engine_rules_for_k 
                            num_active_k = sum(1 for active in sim_engine_for_k.rules.active_players if active)
                            sim_engine_for_k.end_game_early = num_active_k < 2
                            
                            sim_engine_for_k.process_action(self.ai_player_idx, action_str_k, amount_k)
                            sim_engine_for_k.rules.actions_this_round += 1 
                            
                            num_active_after_k_action = sum(1 for active_p in sim_engine_for_k.rules.active_players if active_p)
                            if num_active_after_k_action < 2:
                                sim_engine_for_k.end_game_early = True

                            payoff_for_k = self._simulate_hand_outcome(sim_engine_for_k.rules, self.ai_player_idx)
                            counterfactual_payoffs[k_action_idx] = payoff_for_k
                        else:
                            counterfactual_payoffs[k_action_idx] = -1e9 
                    
                    actual_action_idx = torch.multinomial(strategy_probs_tensor.cpu(), 1).item()
                    if counterfactual_payoffs[actual_action_idx].item() < -1e8 : 
                        print(f"Warning: AI chose actual_action_idx {actual_action_idx} which was invalid. Picking best valid option.")
                        valid_mask = counterfactual_payoffs > -1e9
                        if torch.any(valid_mask):
                            actual_action_idx = torch.argmax(counterfactual_payoffs + (~valid_mask * -1e10)).item() 
                        else: 
                            print("Error: All actions for AI are invalid. Defaulting to fold action index if possible.")
                            fold_action_idx_fallback = 0 
                            for i_fold_check in range(self.cfr_trainer.model.num_actions):
                                temp_act_str, _ = get_action_from_index(i_fold_check, current_ai_view_gs, self.game_engine.rules.player_chips[self.ai_player_idx])
                                if temp_act_str == 'fold':
                                    fold_action_idx_fallback = i_fold_check
                                    break
                            actual_action_idx = fold_action_idx_fallback
                    
                    actual_payoff = counterfactual_payoffs[actual_action_idx].item()
                    training_data_for_hand.append((
                        state_tensor.clone(), actual_action_idx, actual_payoff, counterfactual_payoffs.clone()
                    ))
                    action_tuple = get_action_from_index(actual_action_idx, current_ai_view_gs, 
                                                         self.game_engine.rules.player_chips[self.ai_player_idx])
                    print(f"AI Player {self.ai_player_idx} (Engine Idx {current_player_engine_idx}) ACTUALLY takes action_idx {actual_action_idx}: {action_tuple}")
                    action_to_engine_str, amount_for_engine = action_tuple
                else: 
                    action_tuple = self._get_opponent_action(self.game_engine.rules, current_player_engine_idx)
                    print(f"Opponent Player {current_player_engine_idx} action: {action_tuple}")
                    action_to_engine_str, amount_for_engine = action_tuple
                
                self.game_engine.process_action(current_player_engine_idx, action_to_engine_str, amount_for_engine)
                main_ai_gs.record_action(str(current_player_engine_idx), action_tuple)
                self.game_engine.rules.actions_this_round += 1 
                num_active_after_action = sum(1 for active_p in self.game_engine.rules.active_players if active_p)
                if num_active_after_action < 2: self.game_engine.end_game_early = True
                main_ai_gs = self._populate_ai_gamestate(self.game_engine.rules, main_ai_gs)
                if self.game_engine.rules.betting_round_is_over(): active_betting_round_main_loop = False
            
            print(f"Betting round for {stage_name} ended. Total actions in round: {self.game_engine.rules.actions_this_round}")
            if not self.game_engine.end_game_early: 
                 self.game_engine.rules.end_betting_round_cleanup()

        if not self.game_engine.end_game_early:
            print("\n--- Performing Showdown ---")
            self.game_engine.perform_showdown() 
            main_ai_gs = self._populate_ai_gamestate(self.game_engine.rules, main_ai_gs)
            print(f"Final pot: {main_ai_gs.pot}, Community cards: {main_ai_gs.community_cards}")
            for p_idx in range(self.game_engine.rules.num_players):
                if self.game_engine.rules.active_players[p_idx]: 
                    hand_display = self.game_engine.format_hand_display(self.game_engine.rules.hands[p_idx])
                    print(f"Player {p_idx} hand: {hand_display}, Chips: {self.game_engine.rules.player_chips[p_idx]}")
        
        print("\n--- Training AI Model (Post-Hand) ---")
        if not training_data_for_hand:
            print("No training data (AI decision points) collected for this hand.")
        for state_tensor_data, _, _, cf_payoffs_data in training_data_for_hand:
            print(f"Calling cfr_trainer.train with state_tensor shape: {state_tensor_data.shape}, cf_payoffs: {cf_payoffs_data.tolist()}")
            self.cfr_trainer.train(state_tensor_data, cf_payoffs_data)
        
        print("\nHand complete.")
        return training_data_for_hand

    def run_parallel_hands(self, num_hands: int, num_workers: int = 2):
        """Run multiple hands in parallel processes and aggregate training data."""
        import multiprocessing as mp

        def worker(_: int, queue: mp.Queue):
            data = self.play_hand_for_training()
            queue.put(data)

        queue: mp.Queue = mp.Queue()
        processes = [mp.Process(target=worker, args=(i, queue)) for i in range(num_workers)]
        for p in processes:
            p.start()
        results = []
        for _ in range(num_workers):
            results.append(queue.get())
        for p in processes:
            p.join()
        return results


if __name__ == '__main__':
    class MockModel(torch.nn.Module):
        def __init__(self, num_actions=10): 
            super().__init__()
            self.num_actions = num_actions 
        def forward(self, x):
            probs = torch.ones(1, self.num_actions) / self.num_actions
            return probs

    class MockCFRTrainer:
        def __init__(self, num_actions=10):
            self.config = { 'model': { 'max_seq_len': 20, 'd_raw_feature': 3 } }
            self.model = MockModel(num_actions=num_actions)
            self.num_actions = num_actions 

        def train(self, state_tensor: torch.Tensor, all_counterfactual_payoffs: torch.Tensor):
            print(f"MockCFRTrainer.train called:")
            print(f"  State Tensor Shape: {state_tensor.shape}")
            print(f"  Counterfactual Payoffs: {all_counterfactual_payoffs.tolist()}")

    mock_cfr_trainer = MockCFRTrainer(num_actions=10) 
    
    game_config = {
        'num_players': 2, 'starting_stack': 1000, 'big_blind': 10, 'small_blind': 5,
    }
    print("Initializing SelfPlay environment...")
    self_play_env = SelfPlay(cfr_trainer=mock_cfr_trainer, game_engine_config=game_config)
    
    print("\nStarting to play a single hand for training (with counterfactuals and trainer integration)...")
    training_data = self_play_env.play_hand_for_training()
    print("\n--- Training Data Log (from play_hand_for_training) ---")
    if training_data:
        for i, data_point in enumerate(training_data):
            state_t, action_idx, actual_p, cf_payoffs = data_point
            print(f"Data point {i}: Actual Action Idx: {action_idx}, Actual Payoff: {actual_p:.2f}, CF Payoffs: {[f'{p:.2f}' for p in cf_payoffs.tolist()]}")
    else:
        print("No training data was collected.")

    print("\nSelfPlay with counterfactual logic and trainer integration test finished.")
