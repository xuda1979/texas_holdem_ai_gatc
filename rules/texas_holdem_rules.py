import random
from itertools import combinations

class TexasHoldem:
    def __init__(self, num_players=2):
        self.num_players = num_players
        self.deck = self._create_deck()
        self.hands = [[] for _ in range(num_players)]
        self.community_cards = []
        self.pot = 0
        self.bets = [0] * num_players
        self.current_player = 0
        self.active_players = [True] * num_players
        self.player_chips = [1000] * num_players  # Example starting chips
        self.small_blind = 10
        self.big_blind = 20
        self.current_bet = self.big_blind
        self.dealer_button = 0

    def _create_deck(self):
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        return [(rank, suit) for suit in suits for rank in ranks]

    def shuffle_deck(self):
        random.shuffle(self.deck)

    def deal(self):
        self.shuffle_deck()
        for _ in range(2):  # Deal 2 cards to each player
            for i in range(self.num_players):
                self.hands[i].append(self.deck.pop())

    def post_blinds(self):
        small_blind_player = (self.dealer_button + 1) % self.num_players
        big_blind_player = (self.dealer_button + 2) % self.num_players

        self.player_chips[small_blind_player] -= self.small_blind
        self.bets[small_blind_player] = self.small_blind
        self.pot += self.small_blind

        self.player_chips[big_blind_player] -= self.big_blind
        self.bets[big_blind_player] = self.big_blind
        self.pot += self.big_blind

        self.current_bet = self.big_blind
        self.current_player = (self.dealer_button + 3) % self.num_players

    def bet(self, player_index, amount):
        if amount < self.current_bet - self.bets[player_index]:
            raise ValueError("Bet amount is less than the current bet.")
        
        bet_difference = amount - self.bets[player_index]
        self.player_chips[player_index] -= bet_difference
        self.bets[player_index] += bet_difference
        self.pot += bet_difference

        self.current_bet = max(self.current_bet, amount)
        self.current_player = (player_index + 1) % self.num_players

    def fold(self, player_index):
        self.active_players[player_index] = False

    def call(self, player_index):
        call_amount = self.current_bet - self.bets[player_index]
        self.bet(player_index, call_amount)

    def check(self, player_index):
        if self.bets[player_index] != self.current_bet:
            raise ValueError("Cannot check unless current bet is matched.")
        self.current_player = (player_index + 1) % self.num_players

    def reveal_community_cards(self, num_cards):
        for _ in range(num_cards):
            self.community_cards.append(self.deck.pop())

    def reset_bets(self):
        self.bets = [0] * self.num_players
        self.current_bet = 0

    def play_round(self):
        self.post_blinds()
        self.deal()
        self.betting_round()

        # Flop
        self.reveal_community_cards(3)
        self.betting_round()

        # Turn
        self.reveal_community_cards(1)
        self.betting_round()

        # River
        self.reveal_community_cards(1)
        self.betting_round()

        winner = self.get_winner()
        print("Winner:", winner)
        self.award_pot(winner)

    def betting_round(self):
        while True:
            player_index = self.current_player
            if not self.active_players[player_index]:
                self.current_player = (self.current_player + 1) % self.num_players
                continue

            # Example decision logic (can be replaced with actual player/AI logic)
            if self.player_chips[player_index] > self.current_bet:
                action = random.choice(['bet', 'call', 'fold', 'check'])
            else:
                action = 'call' if self.player_chips[player_index] >= self.current_bet else 'fold'

            if action == 'bet':
                self.bet(player_index, self.current_bet + 10)
            elif action == 'call':
                self.call(player_index)
            elif action == 'fold':
                self.fold(player_index)
            elif action == 'check':
                self.check(player_index)

            # Move to next player
            if self.current_player == self.dealer_button:
                break

    def get_winner(self):
        active_players = [i for i in range(self.num_players) if self.active_players[i]]
        best_hand = None
        winner = None

        for player in active_players:
            player_hand = self.hands[player] + self.community_cards
            hand_strength = self.evaluate_hand(player_hand)
            if best_hand is None or hand_strength > best_hand:
                best_hand = hand_strength
                winner = player

        return winner

    def evaluate_hand(self, cards):
        # Simplified hand evaluation (to be replaced with full poker hand evaluation logic)
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return sum(rank_values[rank] for rank, suit in cards)

    def award_pot(self, winner):
        self.player_chips[winner] += self.pot
        self.pot = 0

    def rotate_dealer(self):
        self.dealer_button = (self.dealer_button + 1) % self.num_players

if __name__ == "__main__":
    game = TexasHoldem(num_players=4)
    game.play_round()
