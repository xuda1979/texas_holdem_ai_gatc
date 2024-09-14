import os
import time
import json
from datetime import datetime
from cfr_trainer import AICFRTrainer
from rules.texas_holdem_rules import TexasHoldemRules

MODEL_DIR = 'models'
DATA_DIR = 'data/simulated_data'
SIMULATION_INTERVAL = 3600  # 1 hour

def load_best_model():
    weights_path = os.path.join(MODEL_DIR, 'model_best_weights.h5')
    config_path = os.path.join(MODEL_DIR, 'model_best_config.json')
    
    if not os.path.exists(weights_path) or not os.path.exists(config_path):
        raise FileNotFoundError("Best model files not found.")
    
    trainer = AICFRTrainer()
    trainer.load_model(weights_path, config_path)
    return trainer

def simulate_game(trainer):
    rules = TexasHoldemRules()
    gui = PokerGameGUI(trainer, rules)
    game_data = gui.start_game()
    return game_data

def save_simulated_game(game_data):
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    file_path = os.path.join(DATA_DIR, f'simulated_game_{timestamp}.json')
    with open(file_path, 'w') as f:
        json.dump(game_data, f)

def main():
    while True:
        trainer = load_best_model()
        game_data = simulate_game(trainer)
        save_simulated_game(game_data)
        time.sleep(SIMULATION_INTERVAL)

if __name__ == "__main__":
    main()