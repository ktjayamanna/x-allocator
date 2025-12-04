"""
Bottleneck Profiling - Step 0.5 (COMPLETE VERSION)

Purpose: Identify WHERE time is being spent and WHAT the bottleneck is
Output: Guides vocabulary choice

This measures BOTH:
1. Isolated operations (pure I/O, pure compute)
2. Real training loop (with DataLoader, tqdm, Python overhead)

Key insight: Real training has overhead that isolated profiling misses!
"""

import torch
import time
import torch.nn.functional as F
from .model import MinimalGPT
from .config import Config
from .dataset import MinimalDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

print("=" * 70)
print("BOTTLENECK PROFILING (COMPLETE)")
print("=" * 70)
print("\nGoal: Identify where time is spent in REAL training\n")

# Setup
with open("data/tiny_shakespeare.txt", "r") as f:
    text = f.read()
dataset = MinimalDataset(text, Config.MAX_SEQ_LEN)

# Use subset for faster profiling
subset = torch.utils.data.Subset(dataset, range(min(320, len(dataset))))

loader = DataLoader(
    subset,
    batch_size=Config.BATCH_SIZE,
    shuffle=False,
    pin_memory=True if torch.cuda.is_available() else False
)

model = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

# Count parameters
params = sum(p.numel() for p in model.parameters())
param_bytes = params * 4  # 4 bytes per float32

print(f"Model size:  {params/1e6:.2f}M parameters ({param_bytes/1e6:.1f} MB)")
print(f"Device:      {Config.DEVICE}")
print(f"Batches:     {len(loader)}")
print(f"Pin memory:  {loader.pin_memory}")

# ============================================================================
# PART 1: ISOLATED OPERATIONS (What we measured before)
# ============================================================================

print("\n" + "=" * 70)
print("PART 1: ISOLATED OPERATIONS")
print("=" * 70)
print("(Measuring pure operations without overhead)")
print()

# Get one batch
x, y = next(iter(loader))

# 1. Pure data transfer
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
x_gpu, y_gpu = x.to(Config.DEVICE), y.to(Config.DEVICE)
if torch.cuda.is_available():
    torch.cuda.synchronize()
pure_transfer_time = time.time() - start

# 2. Pure compute (forward + backward + optimizer)
if torch.cuda.is_available():
    torch.cuda.synchronize()
start = time.time()
logits = model(x_gpu)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_gpu.view(-1))
optimizer.zero_grad()
loss.backward()
optimizer.step()
if torch.cuda.is_available():
    torch.cuda.synchronize()
pure_compute_time = time.time() - start

print(f"Pure transfer:  {pure_transfer_time*1000:6.1f}ms")
print(f"Pure compute:   {pure_compute_time*1000:6.1f}ms")
print(f"Total:          {(pure_transfer_time + pure_compute_time)*1000:6.1f}ms")
print()
print(f"I/O percentage: {pure_transfer_time/(pure_transfer_time + pure_compute_time)*100:.1f}%")

# ============================================================================
# PART 2: REAL TRAINING LOOP (What actually happens)
# ============================================================================

print("\n" + "=" * 70)
print("PART 2: REAL TRAINING LOOP")
print("=" * 70)
print("(Measuring with DataLoader iteration + tqdm)")
print()

# Measure real training loop WITH tqdm (exactly as in train.py)
model2 = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=Config.LEARNING_RATE)
model2.train()

print("Running real training loop (with tqdm, matching train.py)...")

# First: Measure WITHOUT detailed timing (real scenario)
overall_start = time.time()

for x, y in tqdm(loader, desc="Training", total=len(loader)):
    x_gpu = x.to(Config.DEVICE)
    y_gpu = y.to(Config.DEVICE)
    logits = model2(x_gpu)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_gpu.view(-1))
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

real_training_time = time.time() - overall_start

print()
print(f"Real training time:   {real_training_time*1000:6.1f}ms")
print(f"Per batch:            {real_training_time/len(loader)*1000:6.1f}ms")
print()

# Second: Measure WITH detailed timing (for breakdown)
model3 = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
optimizer3 = torch.optim.AdamW(model3.parameters(), lr=Config.LEARNING_RATE)
model3.train()

total_transfer_time = 0
total_compute_time = 0
num_batches = 0

print("Running detailed profiling (without tqdm)...")

for x, y in loader:
    # Measure transfer
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    transfer_start = time.time()
    x_gpu = x.to(Config.DEVICE)
    y_gpu = y.to(Config.DEVICE)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    transfer_time = time.time() - transfer_start

    # Measure compute
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compute_start = time.time()
    logits = model3(x_gpu)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_gpu.view(-1))
    optimizer3.zero_grad()
    loss.backward()
    optimizer3.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    compute_time = time.time() - compute_start

    total_transfer_time += transfer_time
    total_compute_time += compute_time
    num_batches += 1

pure_training_time = total_transfer_time + total_compute_time

# Calculate overhead
avg_transfer = total_transfer_time / num_batches
avg_compute = total_compute_time / num_batches
avg_real = real_training_time / num_batches
avg_pure = pure_training_time / num_batches
total_overhead = real_training_time - pure_training_time
avg_overhead = total_overhead / num_batches

print()
print(f"Pure training time:   {pure_training_time*1000:6.1f}ms (no tqdm)")
print(f"Real training time:   {real_training_time*1000:6.1f}ms (with tqdm)")
print(f"Overhead:             {total_overhead*1000:6.1f}ms")
print()
print(f"Per batch average:")
print(f"  Transfer:           {avg_transfer*1000:6.1f}ms ({avg_transfer/avg_real*100:5.1f}%)")
print(f"  Compute:            {avg_compute*1000:6.1f}ms ({avg_compute/avg_real*100:5.1f}%)")
print(f"  Overhead:           {avg_overhead*1000:6.1f}ms ({avg_overhead/avg_real*100:5.1f}%)")
print(f"  Total (real):       {avg_real*1000:6.1f}ms")

# ============================================================================
# PART 3: COMPARISON & ANALYSIS
# ============================================================================

print("\n" + "=" * 70)
print("PART 3: COMPARISON & BOTTLENECK ANALYSIS")
print("=" * 70)
print()

print("ISOLATED OPERATIONS:")
print(f"  Pure transfer:      {pure_transfer_time*1000:6.1f}ms ({pure_transfer_time/(pure_transfer_time+pure_compute_time)*100:5.1f}%)")
print(f"  Pure compute:       {pure_compute_time*1000:6.1f}ms ({pure_compute_time/(pure_transfer_time+pure_compute_time)*100:5.1f}%)")
print(f"  Total:              {(pure_transfer_time+pure_compute_time)*1000:6.1f}ms")
print()

print("REAL TRAINING LOOP:")
print(f"  Transfer:           {avg_transfer*1000:6.1f}ms ({avg_transfer/avg_real*100:5.1f}%)")
print(f"  Compute:            {avg_compute*1000:6.1f}ms ({avg_compute/avg_real*100:5.1f}%)")
print(f"  Overhead:           {avg_overhead*1000:6.1f}ms ({avg_overhead/avg_real*100:5.1f}%)")
print(f"  Total per batch:    {avg_real*1000:6.1f}ms")
print()

overhead_pct = avg_overhead / avg_real * 100

print("OVERHEAD BREAKDOWN:")
print(f"  tqdm progress bar:  ~{overhead_pct:.1f}% of total time")
print(f"  DataLoader iteration")
print(f"  Python function calls")
print(f"  Memory allocation")
print()

# Identify REAL bottleneck
print("=" * 70)
print("BOTTLENECK IDENTIFICATION")
print("=" * 70)
print()

if overhead_pct > 10:
    print(f"⚠️  BOTTLENECK IDENTIFIED: Python/DataLoader overhead ({overhead_pct:.1f}%)")
    print()
    print(f"   Real training has {overhead_pct:.1f}% overhead from:")
    print(f"   - tqdm progress bar")
    print(f"   - DataLoader iteration")
    print(f"   - Python function calls")
    print()
    print(f"   → Opportunity: Prefetching can hide this overhead!")
    print(f"   → Strategy: Pre-load batches, use tqdm on range()")
    print(f"   → Expected speedup: ~{overhead_pct:.0f}%")
    print(f"   → Vocabulary: {{prefetch, wait, compute, release}}")
elif avg_transfer / avg_real > 0.05:
    print(f"⚠️  BOTTLENECK IDENTIFIED: Data transfer ({avg_transfer/avg_real*100:.1f}%)")
    print()
    print(f"   → Opportunity: Overlap I/O with compute")
    print(f"   → Vocabulary: {{prefetch, wait, compute, release}}")
else:
    print(f"✓  I/O is not a bottleneck ({avg_transfer/avg_real*100:.1f}% of time)")
    print(f"   Overhead is minimal ({overhead_pct:.1f}%)")
    print(f"   → Focus on compute optimization instead")
    print()
    print(f"   Note: Even with minimal overhead, prefetching may still help")
    print(f"   by hiding DataLoader iteration and Python overhead.")

print()
print("=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print()
print("Isolated profiling showed:")
print(f"  I/O: {pure_transfer_time/(pure_transfer_time+pure_compute_time)*100:.1f}% (misleading!)")
print()
print("Real training loop shows:")
print(f"  I/O: {avg_transfer/avg_real*100:.1f}%")
print(f"  Overhead: {overhead_pct:.1f}%")
print()
if overhead_pct > 5:
    print(f"Prefetching can hide this {overhead_pct:.1f}% overhead!")
else:
    print("Overhead is minimal, but prefetching may still provide small gains.")
print()
print("LESSON: Always profile the REAL training loop, not isolated operations!")
print("=" * 70)
