import torch
import os

# === Device ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Paths ===
DATA_ROOT = "data"
SWI_INPUT_DIR = os.path.join(DATA_ROOT, "samsung_data", "swi")
ROI_INPUT_DIR = os.path.join(DATA_ROOT, "samsung_data", "roi")
SWI_OUTPUT_DIR = os.path.join(DATA_ROOT, "output_images", "swi")
ROI_OUTPUT_DIR = os.path.join(DATA_ROOT, "output_images", "roi")
LMDB_DIR = os.path.join(DATA_ROOT, "lmdb")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
BBOX_JSON_PATH = os.path.join(DATA_ROOT, "bboxes", "all_bboxes.json")
RESULTS_DIR = "results"
WEIGHTS_DIR = "weights"

# === Data Params ===
BATCH_SIZE = 40
NUM_WORKERS = 8
SPLIT_RATIO = 0.2

# === Model / Training Params ===
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-4
SEED = 42
K_FOLDS = 10
USE_K_FOLD = True

# Loss Configuration
ALPHA = 0.99
GAMMA = 2.0
NEG_POS_RATIO = 3

# === Training Params ===
TRAIN_IOU_THRESH = 0.35

# === Evaluation Params ===
CONF_THRESH = 0.1
TEST_IOU_THRESH = 0.1
NMS_THRESH = 0.45