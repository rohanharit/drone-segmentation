import os
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "archive", "dataset", "semantic_drone_dataset")
IMAGE_DIR = os.path.join(DATA_DIR, "original_images")
MASK_DIR = os.path.join(DATA_DIR, "label_images_semantic")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

NUM_CLASSES = 5
INPUT_CHANNELS = 3
IMAGE_SIZE = 512

EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
TEST_SIZE = 0.2
RANDOM_STATE = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["obstacles", "water", "nature", "moving", "landable"]

GROUPED_CLASSES = {
    0: {0, 6, 10, 11, 12, 13, 14, 21, 22, 23},
    1: {5, 7},
    2: {2, 3, 8, 19, 20},
    3: {15, 16, 17, 18},
    4: {1, 4, 9},
}

COLOR_MAP = {
    0: [155, 38, 182],
    1: [14, 135, 204],
    2: [124, 252, 0],
    3: [255, 20, 147],
    4: [169, 169, 169],
}
