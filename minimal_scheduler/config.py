import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    MAX_SEQ_LEN = 128
    VOCAB_SIZE = 256
    N_EMBD = 256
    N_LAYER = 6
    N_HEAD = 4
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 5
