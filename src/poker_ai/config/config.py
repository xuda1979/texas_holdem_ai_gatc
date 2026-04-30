import os

from poker_ai.model_storage import remote_checkpoint_dir

# Define the base data directory
BASE_DATA_DIR = "data"

# Directories for saving models and simulated data
MODEL_DIR = str(remote_checkpoint_dir())
SIMULATED_DATA_DIR = os.path.join(BASE_DATA_DIR, "simulated_data")

# Ensure local data directories exist
os.makedirs(SIMULATED_DATA_DIR, exist_ok=True)
