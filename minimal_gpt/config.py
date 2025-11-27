import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 128
    N_EMBD = 256
    N_LAYER = 6
    N_HEAD = 4
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 3

