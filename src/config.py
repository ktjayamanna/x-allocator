import torch
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

VOCAB_SIZE = 256
N_EMBD = 256
N_LAYER = 1 # number of transformer blocks
N_HEAD = 4
MAX_SEQ_LEN = 128

BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_NAME = "tiny_shakespeare"
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

INFERENCE_MAX_LENGTH = 200
INFERENCE_TEMPERATURE = 0.8
INFERENCE_TOP_K = 40
