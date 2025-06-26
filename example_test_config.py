import yaml

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("✓ Config loaded successfully!")
    print(f"Game engine players: {config['game_engine']['num_players']}")
    print(f"Model actions: {config['model']['num_actions']}")
    print(f"Training hands: {config['training']['num_training_hands']}")
    print(f"Model directory: {config['model']['directory']}")
    
except Exception as e:
    print(f"✗ Error loading config: {e}")
