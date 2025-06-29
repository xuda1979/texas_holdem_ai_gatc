
class Player:
    def __init__(self, player_id, stack_size):
        self.player_id = player_id
        self.stack_size = stack_size
        self.hand = None
        self.current_bet = 0

    def set_hand(self, hand):
        self.hand = hand

    def bet(self, amount):
        if amount > self.stack_size:
            amount = self.stack_size
        self.stack_size -= amount
        self.current_bet += amount
        return amount
