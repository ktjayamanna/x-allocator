import torch.nn.functional as F
from tqdm import tqdm
from .scheduler import PrefetchScheduler


def train_epoch(model, loader, optimizer, use_prefetch=False):
    """
    Train for one epoch.

    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: PyTorch optimizer
        use_prefetch: If True, use prefetch scheduler for speedup

    Returns:
        Average loss for the epoch
    """
    device = next(model.parameters()).device
    model.train()

    if use_prefetch:
        # Use prefetch scheduler (overlaps I/O with compute)
        return train_epoch_with_prefetch(model, loader, optimizer, device)
    else:
        # Baseline (no prefetching)
        return train_epoch_baseline(model, loader, optimizer, device)


def train_epoch_baseline(model, loader, optimizer, device):
    """
    Baseline training (no prefetching).

    Sequential: load → compute → load → compute → ...
    """
    total_loss = 0.0
    num_batches = 0

    for x, y in tqdm(loader, total=len(loader), desc="Training"):
        # Load batch (blocking)
        x, y = x.to(device), y.to(device)

        # Training step
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def train_epoch_with_prefetch(model, loader, optimizer, device):
    """
    Training with prefetch scheduler (overlaps I/O with compute).

    Pattern: compute(N) + prefetch(N+1) → faster!
    """
    scheduler = PrefetchScheduler(device)

    # Define training step (what to do with each batch)
    def train_step(x, y):
        """Single training step - called by scheduler"""
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    # Run epoch with prefetching
    return scheduler.run_epoch(loader, train_step, show_progress=True)


