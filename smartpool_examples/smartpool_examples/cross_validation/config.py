import os

self_folder = os.path.dirname(os.path.abspath(__file__))

# Training parameters
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001

# Data settings
DATA_ROOT = f'{self_folder}/data'
DATASET_NAME = 'ESC-50'
N_FOLDS = 5
N_CLASSES = 50

# Audio / feature extraction settings
SAMPLE_RATE = 44100
N_MELS = 64
MEL_WIDTH = 64
NPERSEG = 1024
HOP_LENGTH = 512

# ESC-50 parquet mirror
ESC50_URLS = [
    {
        "train": "https://hf-mirror.com/datasets/mskov/ESC50/resolve/main/data/train-00000-of-00001-cd782ca55710a2e6.parquet",
        "test": "https://hf-mirror.com/datasets/mskov/ESC50/resolve/main/data/test-00000-of-00001-f167dc83a7b3449c.parquet",
    }
]
ESC50_SIZE = 386534912 + 386788474  # bytes for train + test parquet
FEATURES_DIR = f'{DATA_ROOT}/esc50_features'
