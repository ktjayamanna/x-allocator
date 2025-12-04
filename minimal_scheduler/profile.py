"""Profiling - measure costs and find bottleneck"""
import torch
import time
import torch.nn.functional as F

def profile_bottleneck(model, optimizer, loader, device):
    """Measure isolated vs real training loop to find overhead"""
    x, y = next(iter(loader))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    x, y = x.to(device), y.to(device)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_transfer = time.time() - t0

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    loss = F.cross_entropy(model(x).view(-1, model(x).size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_compute = time.time() - t0

    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x).view(-1, model(x).size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    t_real = time.time() - t0

    overhead = (t_real - (t_transfer + t_compute) * len(loader)) / t_real
    return {"transfer_ms": t_transfer*1000, "compute_ms": t_compute*1000, "overhead_%": overhead*100}

def profile_costs(model, optimizer, loader, device):
    """Measure prefetch and compute costs"""
    x, y = next(iter(loader))

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    x_gpu = x.to(device, non_blocking=True)
    y_gpu = y.to(device, non_blocking=True)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_prefetch = time.time() - t0

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    loss = F.cross_entropy(model(x_gpu).view(-1, model(x_gpu).size(-1)), y_gpu.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_compute = time.time() - t0

    return {"prefetch_ms": t_prefetch*1000, "compute_ms": t_compute*1000}

