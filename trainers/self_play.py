import torch
from ai_cfr_trainer import AICFRTrainer
from texas_holdem_rules import TexasHoldem

def simulate_self_play(trainer: AICFRTrainer, num_games: int):
    for _ in range(num_games):
        game = TexasHoldem()
        game_state_sequence = game.start()
        # Simulate the game using the AI model and update strategies
        trainer.train(game_state_sequence)

if __name__ == "__main__":
    trainer = AICFRTrainer()
    simulate_self_play(trainer, num_games=10000)
    trainer.save_model()
