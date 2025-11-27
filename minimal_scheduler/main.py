import torch
from torch.utils.data import DataLoader
from model import MinimalGPT
from config import Config
from dataset import MinimalDataset
from train import train_epoch
from tqdm import tqdm

with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()
dataset = MinimalDataset(text, Config.MAX_SEQ_LEN)
loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
# Overfit to a single batch for fast debugging: reuse the same batch every epoch
loader = [next(iter(loader))]
model = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

for epoch in tqdm(range(Config.NUM_EPOCHS), total=Config.NUM_EPOCHS, desc="Epochs"):
    train_epoch(model, loader, optimizer)