class GameState:
    def __init__(self):
        self.pot = 0
        self.community_cards = []
        self.player_hands = {}
        self.betting_history = []
        self.current_bet = 0
        self.current_round = 'pre-flop'
        self.players = []
        # Keep track of player ordering for feature encoding
        self.player_order = []
        # Some utilities expect `betting_round` attribute
        self.betting_round = self.current_round

    def set_players(self, players):
        self.players = players
        # Assume players is a list of objects with a player_id attribute
        try:
            self.player_order = [str(p.player_id) for p in players]
        except AttributeError:
            # Fallback if players are provided as IDs or lacking attribute
            self.player_order = [str(p) for p in players]

    def get_player(self, player_id):
        """Return player object by id if available."""
        for p in self.players:
            if getattr(p, 'player_id', None) == player_id or str(p) == str(player_id):
                return p
        return None

    def set_player_hand(self, player_id, hand):
        self.player_hands[player_id] = hand

    def add_community_cards(self, cards):
        self.community_cards.extend(cards)

    def record_action(self, player_id, action_details: tuple):
        """
        Records a player's action in the betting history.

        Args:
            player_id: The ID of the player performing the action.
            action_details: A tuple representing the action.
                Expected format: (action_name_str, amount_int_or_None)
                Examples:
                    ('fold', None)
                    ('check', None)
                    ('call', 100)
                    ('bet', 200)
                    ('raise', 500)
        """
        self.betting_history.append((player_id, action_details))

    def set_current_bet(self, bet):
        self.current_bet = bet
