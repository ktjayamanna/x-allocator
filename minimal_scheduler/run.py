import torch
from model import MinimalGPT
from config import Config
from dataset import get_dataloader
from train import train_epoch

loader, dataset = get_dataloader("data/tiny_shakespeare.txt", Config.MAX_SEQ_LEN, Config.BATCH_SIZE)
model = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

for epoch in range(Config.NUM_EPOCHS):
    train_epoch(model, loader, optimizer)