import torch
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TMP_DIR = os.path.join(DATA_DIR, "tmp")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Custom GPT Model config - Small and interpretable (~10-15M parameters)
# This is MUCH smaller than GPT-2's 124M parameters
VOCAB_SIZE = 256  # Character-level tokenization (ASCII)
N_EMBD = 256      # Embedding dimension
N_LAYER = 6       # Number of transformer layers
N_HEAD = 4        # Number of attention heads
MAX_LENGTH = 128  # Maximum sequence length
DROPOUT = 0.1     # Dropout rate

# Training config - Optimized for RTX 2070 (8GB VRAM)
BATCH_SIZE = 32   # Larger batch size since model is smaller
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size = 64
LEARNING_RATE = 3e-4  # Higher learning rate for training from scratch
NUM_EPOCHS = 5    # More epochs since we're training from scratch
MAX_TRAIN_SAMPLES = None  # Use full dataset
MAX_EVAL_SAMPLES = None   # Use full validation set

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset config - Using TinyShakespeare for fast training and interpretability
DATASET_NAME = "tiny_shakespeare"  # Small, character-level dataset
DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Inference config
INFERENCE_MAX_LENGTH = 200
INFERENCE_TEMPERATURE = 0.8
INFERENCE_TOP_K = 40
