
import os

# Define the base data directory
BASE_DATA_DIR = 'data'

# Directories for saving models and simulated data
MODEL_DIR = os.path.join(BASE_DATA_DIR, 'models')
SIMULATED_DATA_DIR = os.path.join(BASE_DATA_DIR, 'simulated_data')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SIMULATED_DATA_DIR, exist_ok=True)
