"""
Cost Profiling - Step 3

Purpose: Measure HOW EXPENSIVE each operation in the vocabulary is
Output: Guides schedule optimization decisions

This answers:
- How long does each operation take?
- Can operations overlap?
- Is the optimization worth it?

Vocabulary (for "make training faster" goal):
- prefetch(batch_id): Async load batch to GPU
- wait(batch_id): Block until prefetch completes
- compute(batch_id): Run training step
- release(batch_id): Free GPU memory
"""

import torch
import time
from .model import MinimalGPT
from .config import Config
from .dataset import MinimalDataset
from torch.utils.data import DataLoader

# Setup
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()
dataset = MinimalDataset(text, Config.MAX_SEQ_LEN)
loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, pin_memory=True)
model = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

print("=" * 60)
print("COST PROFILING")
print("=" * 60)
print("\nGoal: Measure cost of each operation in vocabulary\n")

# Get batches for testing
batch_iter = iter(loader)
x1, y1 = next(batch_iter)
x2, y2 = next(batch_iter)

print("Vocabulary: {prefetch, wait, compute, release}")
print("\n--- Measuring Operation Costs ---\n")

# 1. Measure PREFETCH (async transfer to GPU)
print("1. prefetch(batch_id): Async load batch to GPU")
if torch.cuda.is_available():
    torch.cuda.synchronize()
    start = time.time()
    x_gpu = x1.to(Config.DEVICE, non_blocking=True)
    y_gpu = y1.to(Config.DEVICE, non_blocking=True)
    # Don't synchronize - that's the point of async!
    prefetch_launch_time = time.time() - start

    # Now measure how long until it's actually done
    torch.cuda.synchronize()
    prefetch_complete_time = time.time() - start

    print(f"   Launch time:   {prefetch_launch_time*1000:.3f}ms (returns immediately)")
    print(f"   Complete time: {prefetch_complete_time*1000:.1f}ms (actual transfer)")
    prefetch_time = prefetch_complete_time
else:
    # CPU: transfer is always blocking
    start = time.time()
    x_gpu = x1.to(Config.DEVICE)
    y_gpu = y1.to(Config.DEVICE)
    prefetch_time = time.time() - start
    print(f"   Time: {prefetch_time*1000:.1f}ms (CPU: always blocking)")

# 2. Measure WAIT (sync point)
print("\n2. wait(batch_id): Block until prefetch completes")
if torch.cuda.is_available():
    # Start a new async transfer
    x_gpu = x2.to(Config.DEVICE, non_blocking=True)
    y_gpu = y2.to(Config.DEVICE, non_blocking=True)

    # Measure wait time
    torch.cuda.synchronize()
    start = time.time()
    torch.cuda.synchronize()  # This is the "wait"
    wait_time = time.time() - start
    print(f"   Time: {wait_time*1000:.3f}ms (if transfer already complete)")
    print(f"   Note: If called immediately after prefetch, waits ~{prefetch_time*1000:.1f}ms")
else:
    wait_time = 0
    print(f"   Time: {wait_time*1000:.3f}ms (CPU: no async operations)")

# 3. Measure COMPUTE (forward + backward + optimizer)
print("\n3. compute(batch_id): Run training step")
x_gpu, y_gpu = x1.to(Config.DEVICE), y1.to(Config.DEVICE)

torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()

logits = model(x_gpu)
loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y_gpu.view(-1))
optimizer.zero_grad()
loss.backward()
optimizer.step()

torch.cuda.synchronize() if torch.cuda.is_available() else None
compute_time = time.time() - start
print(f"   Time: {compute_time*1000:.1f}ms")

# 4. Measure RELEASE (free GPU memory)
print("\n4. release(batch_id): Free GPU memory")
torch.cuda.synchronize() if torch.cuda.is_available() else None
start = time.time()
del x_gpu, y_gpu
torch.cuda.synchronize() if torch.cuda.is_available() else None
release_time = time.time() - start
print(f"   Time: {release_time*1000:.3f}ms (usually negligible)")

# Summary
print("\n" + "=" * 60)
print("COST SUMMARY")
print("=" * 60)
print(f"\nprefetch: {prefetch_time*1000:6.1f}ms (async transfer)")
print(f"wait:     {wait_time*1000:6.3f}ms (if already complete)")
print(f"compute:  {compute_time*1000:6.1f}ms (forward+backward+optimizer)")
print(f"release:  {release_time*1000:6.3f}ms (free memory)")

# Analysis: Can we overlap?
print("\n" + "=" * 60)
print("OPTIMIZATION ANALYSIS")
print("=" * 60)

if prefetch_time < compute_time:
    speedup = prefetch_time / (prefetch_time + compute_time) * 100
    print(f"\n✓ Prefetch time ({prefetch_time*1000:.1f}ms) < Compute time ({compute_time*1000:.1f}ms)")
    print(f"  → Can HIDE prefetch latency by overlapping with compute!")
    print(f"  → Potential speedup: ~{speedup:.1f}% per step")
    print(f"\n  Baseline schedule (sequential):")
    print(f"    prefetch(N) + wait(N) + compute(N) = {(prefetch_time + compute_time)*1000:.1f}ms")
    print(f"\n  Optimized schedule (overlapped):")
    print(f"    compute(N) || prefetch(N+1) = {compute_time*1000:.1f}ms")
    print(f"    (prefetch happens during compute, effectively free!)")
else:
    print(f"\n⚠️  Prefetch time ({prefetch_time*1000:.1f}ms) >= Compute time ({compute_time*1000:.1f}ms)")
    print(f"  → Cannot fully hide prefetch latency")
    print(f"  → Optimization still helps, but limited benefit")

# Check if pinned memory is enabled
if not loader.pin_memory and torch.cuda.is_available():
    print(f"\n💡 TIP: DataLoader has pin_memory=False")
    print(f"   Set pin_memory=True for faster async transfers!")

print("\n" + "=" * 60)
