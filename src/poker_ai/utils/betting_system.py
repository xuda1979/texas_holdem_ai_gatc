
class BettingSystem:
    def __init__(self):
        pass
    
    def resolve_betting(self, players):
        for player in players:
            if player.current_bet < 0:
                return False
        return True
