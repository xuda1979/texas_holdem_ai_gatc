
class GameState:
    def __init__(self):
        self.pot = 0
        self.community_cards = []
        self.player_hands = {}
        self.betting_history = []
        self.current_bet = 0
        self.current_round = 'pre-flop'
        self.players = []

    def set_players(self, players):
        self.players = players

    def set_player_hand(self, player_id, hand):
        self.player_hands[player_id] = hand

    def add_community_cards(self, cards):
        self.community_cards.extend(cards)

    def record_action(self, player_id, action):
        self.betting_history.append((player_id, action))

    def set_current_bet(self, bet):
        self.current_bet = bet
