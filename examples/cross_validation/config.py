# Training parameters
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

# Data settings
import os
self_folder = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = f'{self_folder}/data'
DATASET_NAME = 'MNIST'

# Process pool settings
MAX_WORKERS = None
