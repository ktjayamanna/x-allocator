import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model, loader, optimizer):
    model.train()
    device = next(model.parameters()).device
    for x, y in tqdm(loader, total=len(loader)):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()  # Clear old gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update parameters (gradient decent)


