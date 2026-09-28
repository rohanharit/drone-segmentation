import os
import torch

# ─── Paths ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "archive", "dataset", "semantic_drone_dataset")
IMAGE_DIR = os.path.join(DATA_DIR, "original_images")
MASK_DIR = os.path.join(DATA_DIR, "label_images_semantic")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# ─── Model ───────────────────────────────────────────────────
NUM_CLASSES = 5
INPUT_CHANNELS = 3
IMAGE_SIZE = 512

# ─── Training ────────────────────────────────────────────────
EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ─── Device ──────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Class Mapping ───────────────────────────────────────────
# Original 24 classes → 5 grouped classes for drone landing
CLASS_NAMES = ["obstacles", "water", "nature", "moving", "landable"]

GROUPED_CLASSES = {
    0: {0, 6, 10, 11, 12, 13, 14, 21, 22, 23},  # obstacles
    1: {5, 7},                                     # water
    2: {2, 3, 8, 19, 20},                          # nature
    3: {15, 16, 17, 18},                           # moving
    4: {1, 4, 9},                                  # landable
}

# ─── Visualization Colors (RGB) ─────────────────────────────
COLOR_MAP = {
    0: [155, 38, 182],    # obstacles → purple
    1: [14, 135, 204],    # water     → blue
    2: [124, 252, 0],     # nature    → green
    3: [255, 20, 147],    # moving    → pink
    4: [169, 169, 169],   # landable  → gray
}
