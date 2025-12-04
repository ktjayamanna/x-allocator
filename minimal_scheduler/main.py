import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .model import MinimalGPT
from .config import Config
from .dataset import MinimalDataset
from .scheduler import PrefetchScheduler

with open("data/tiny_shakespeare.txt") as f:
    text = f.read()
dataset = MinimalDataset(text, Config.MAX_SEQ_LEN)
loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True)
model = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

def train_baseline():
    for x, y in tqdm(loader):
        x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
        loss = F.cross_entropy(model(x).view(-1, model(x).size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def train_prefetch():
    scheduler = PrefetchScheduler(Config.DEVICE)
    def step(x, y):
        loss = F.cross_entropy(model(x).view(-1, model(x).size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.run_epoch(loader, step)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        from .profile import profile_bottleneck, profile_costs
        subset = torch.utils.data.Subset(dataset, range(320))
        small_loader = DataLoader(subset, batch_size=Config.BATCH_SIZE, pin_memory=True)
        bottleneck = profile_bottleneck(model, optimizer, small_loader, Config.DEVICE)
        costs = profile_costs(model, optimizer, small_loader, Config.DEVICE)
        print(f"Bottleneck: transfer={bottleneck['transfer_ms']:.1f}ms, compute={bottleneck['compute_ms']:.1f}ms, overhead={bottleneck['overhead_%']:.1f}%")
        print(f"Costs: prefetch={costs['prefetch_ms']:.1f}ms, compute={costs['compute_ms']:.1f}ms")
        print(f"Can overlap: {costs['prefetch_ms'] < costs['compute_ms']}")
    else:
        for epoch in range(Config.NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
            train_prefetch()