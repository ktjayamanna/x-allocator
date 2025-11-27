import torch.nn.functional as F
from config import Config

def train_epoch(model, loader, optimizer):
    model.train()
    for x, y in loader:
        x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


