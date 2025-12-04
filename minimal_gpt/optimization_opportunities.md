# Blocking Events as Optimization Opportunities

This document identifies the 8 major blocking events in the minimal_gpt training loop and describes optimization opportunities for each.

## Overview

In the unoptimized training loop, CPU and GPU are **never working simultaneously**. When one is busy, the other is idle, creating significant performance bottlenecks.

```
CPU: [Load batch] → [.to(device) BLOCKS] → [Launch kernels] → [BLOCKS waiting for GPU] → [Load next batch]
                         ↓                         ↓                      ↓
GPU:     [IDLE]    →  [Receive data]  →  [Compute forward+backward]  →  [IDLE]
```

---

## Blocking Event #1: Data Transfer Blocks CPU

### Current Behavior
```python
x, y = x.to(device), y.to(device)  # Synchronous blocking call
```

- **What blocks**: CPU thread executing the training loop
- **Waiting for**: PCIe DMA transfer to complete (RAM → VRAM)
- **Duration**: ~2-10 microseconds per batch
- **Who's idle**: CPU waits, GPU is idle during transfer

### Optimization Opportunities

1. **Use Asynchronous Transfers**
   - Use `tensor.to(device, non_blocking=True)` with pinned memory
   - CPU can continue preparing next batch while transfer happens
   - Requires pinned memory allocation

2. **Pinned Memory in DataLoader**
   ```python
   DataLoader(dataset, pin_memory=True)
   ```
   - Enables faster DMA transfers (no page-locking overhead)
   - Allows non-blocking transfers

3. **CUDA Streams**
   - Use separate CUDA streams for data transfer and computation
   - Overlap transfer of batch N+1 with computation of batch N

---

## Blocking Event #2: GPU Blocks on Data Transfer

### Current Behavior

- **What blocks**: GPU/SMs waiting for input data
- **Waiting for**: The `.to(device)` transfer to complete before forward pass can start
- **Duration**: ~2-10 microseconds
- **Who's idle**: All SMs are idle while waiting for batch data

### Optimization Opportunities

1. **Prefetching with CudaPrefetcher**
   - Transfer batch N+1 to GPU while processing batch N
   - GPU always has data ready when it finishes previous batch
   - Eliminates GPU idle time waiting for data

2. **Double Buffering**
   - Maintain two buffers in VRAM
   - Alternate between buffers for consecutive batches
   - One buffer computes while other receives new data

---

## Blocking Event #3: CPU Blocks on GPU Computation

### Current Behavior
```python
x, y = x.to(device), y.to(device)
logits = model(x)
loss.backward()
optimizer.step()
# Next iteration implicitly synchronizes - CPU BLOCKS HERE
```

- **What blocks**: CPU when it tries to access the next batch
- **Waiting for**: GPU to finish entire forward + backward + optimizer step
- **Duration**: ~10-50 milliseconds (most of the iteration time)
- **Who's idle**: CPU waits for GPU to finish before preparing next batch

### Optimization Opportunities

1. **Asynchronous Execution**
   - Use non-blocking CUDA operations
   - CPU prepares next batch while GPU computes current batch
   - Only synchronize when absolutely necessary

2. **Multi-threaded Data Loading**
   ```python
   DataLoader(dataset, num_workers=4)
   ```
   - Background threads prepare batches while GPU computes
   - Main thread only coordinates, doesn't do heavy lifting

3. **Prefetcher Pattern**
   - Dedicated prefetcher loads and transfers next batch
   - Main loop only consumes pre-loaded batches
   - CPU and GPU work in parallel

---

## Blocking Event #4: GPU Blocks on Sequential Kernel Launches

### Current Behavior
```python
tok_emb = self.token_emb(input_ids)  # Kernel 1
pos_emb = self.pos_emb(pos)          # Kernel 2 waits for Kernel 1
x = tok_emb + pos_emb                # Kernel 3 waits for Kernel 2
```

- **What blocks**: Each CUDA kernel waits for previous kernel to complete
- **Waiting for**: Previous kernel's output to be written to VRAM
- **Duration**: ~5-10 microseconds per kernel launch
- **Who's idle**: SMs may be partially idle between kernel launches

### Optimization Opportunities

1. **Kernel Fusion**
   - Combine multiple operations into single kernel
   - Reduce kernel launch overhead
   - Keep intermediate results in registers/shared memory

2. **Multiple CUDA Streams**
   - Launch independent operations in parallel streams
   - Example: Different attention heads in parallel
   - Requires careful dependency management

3. **Operator Fusion (torch.compile)**
   - Use PyTorch 2.0's `torch.compile()` for automatic fusion
   - Reduces kernel launches and memory traffic
   - Keeps data in cache/registers longer

---

## Blocking Event #5: Memory-Bound Operations Block Compute

### Current Behavior
```python
tok_emb = self.token_emb(input_ids)  # Memory-bound: embedding lookup
```

- **What blocks**: SM compute units waiting for data
- **Waiting for**: VRAM reads to complete (embedding table lookups)
- **Duration**: Variable, depends on cache hit rate
- **Who's idle**: SM ALUs idle while waiting for memory subsystem
- **Bandwidth**: Limited by VRAM bandwidth (~448 GB/s on RTX 2070)

### Optimization Opportunities

1. **Tensor Layout Optimization**
   - Arrange data in memory for coalesced access patterns
   - Contiguous memory layouts improve cache utilization
   - Row-major vs column-major considerations

2. **Memory Hierarchy Awareness**
   - Keep frequently accessed data in L2 cache
   - Use shared memory for data reuse within thread blocks
   - Optimize for cache line sizes (128 bytes on modern GPUs)

3. **Quantization**
   - Use FP16 or INT8 instead of FP32
   - Reduces memory bandwidth requirements by 2-4x
   - Increases effective throughput

4. **Gradient Checkpointing**
   - Trade compute for memory bandwidth
   - Recompute activations instead of storing them
   - Reduces memory traffic during backward pass

---

## Blocking Event #6: Compute-Bound Operations Block on Dependencies

### Current Behavior
```python
qkv = self.qkv(x)  # Matrix multiply: compute-bound
# Next operation must wait for result
```

- **What blocks**: Later operations waiting for matmul result
- **Waiting for**: cuBLAS to finish computing the matrix multiplication
- **Duration**: Variable, depends on matrix sizes
- **Who's idle**: Rest of GPU pipeline waits for matmul to complete

### Optimization Opportunities

1. **Tensor Cores (Mixed Precision)**
   - Use FP16 with Tensor Cores for 8x speedup on matmul
   - Automatic mixed precision (AMP) in PyTorch
   - Maintains FP32 accuracy with FP16 speed

2. **Flash Attention**
   - Fused attention kernel that reduces memory I/O
   - Computes attention without materializing full attention matrix
   - 2-4x speedup on attention operations

3. **Parallelism Across Layers**
   - Pipeline parallelism: Process different batches in different layers
   - Requires careful scheduling and buffering
   - Keeps all layers busy simultaneously

---

## Blocking Event #7: Backward Pass Blocks on Saved Activations

### Current Behavior
```python
loss.backward()  # Needs activations saved during forward
```

- **What blocks**: Gradient computation kernels
- **Waiting for**: Reading saved activations from VRAM
- **Duration**: Variable, depends on activation sizes
- **Who's idle**: SMs wait for VRAM reads of activation tensors

### Optimization Opportunities

1. **Activation Checkpointing**
   - Save only subset of activations (e.g., every Nth layer)
   - Recompute intermediate activations during backward
   - Trades 33% more compute for 75% less memory

2. **Activation Compression**
   - Store activations in lower precision (FP16 or INT8)
   - Decompress on-the-fly during backward pass
   - Reduces memory bandwidth requirements

3. **Fused Backward Kernels**
   - Combine gradient computation with activation reads
   - Keep activations in cache/registers longer
   - Reduces round-trips to VRAM

---

## Blocking Event #8: Optimizer Blocks on Gradient Computation

### Current Behavior
```python
loss.backward()        # Must complete first
optimizer.step()       # Blocks until all gradients computed
```

- **What blocks**: Optimizer update kernels
- **Waiting for**: All backward pass gradients to be computed
- **Duration**: Waits for entire backward pass
- **Who's idle**: Optimizer kernels wait for backward pass to finish

### Optimization Opportunities

1. **Gradient Accumulation**
   - Accumulate gradients over multiple batches
   - Update parameters less frequently
   - Amortizes optimizer overhead

2. **Asynchronous Optimizer Updates**
   - Start updating early layers while late layers still compute gradients
   - Requires careful synchronization
   - Can overlap optimizer with backward pass

3. **Fused Optimizer Kernels**
   - Combine gradient computation with parameter updates
   - Single kernel does backward + optimizer step
   - Reduces memory traffic and kernel launches

4. **ZeRO Optimizer (Distributed)**
   - Partition optimizer states across GPUs
   - Reduces memory footprint per GPU
   - Enables larger models/batches

---

## Summary Table

| **Blocking Event** | **Who Waits** | **Duration** | **Best Optimization** |
|-------------------|---------------|--------------|----------------------|
| 1. Data transfer blocks CPU | CPU | ~2-10 μs | Async transfers + pinned memory |
| 2. GPU waits for data | GPU/SMs | ~2-10 μs | CudaPrefetcher + double buffering |
| 3. CPU waits for GPU | CPU | ~10-50 ms | Multi-threaded DataLoader + prefetcher |
| 4. Sequential kernels | Each kernel | ~5-10 μs | Kernel fusion + torch.compile |
| 5. Memory-bound ops | SM ALUs | Variable | Tensor layout optimization + FP16 |
| 6. Compute-bound ops | Pipeline | Variable | Tensor Cores + Flash Attention |
| 7. Backward on activations | Gradient kernels | Variable | Activation checkpointing |
| 8. Optimizer waits | Update kernels | Waits for backward | Fused optimizer kernels |

---

## Implementation Priority

### High Impact (Implement First)
1. **CudaPrefetcher** (Event #2, #3) - Overlaps data transfer with computation
2. **Pinned Memory + Async Transfers** (Event #1) - Enables prefetching
3. **Mixed Precision Training** (Event #5, #6) - 2-4x speedup with minimal code change

### Medium Impact
4. **Multi-threaded DataLoader** (Event #3) - Better CPU utilization
5. **Gradient Accumulation** (Event #8) - Larger effective batch sizes
6. **Activation Checkpointing** (Event #7) - Enables larger models

### Advanced Optimizations
7. **Kernel Fusion / torch.compile** (Event #4) - Requires PyTorch 2.0+
8. **Flash Attention** (Event #6) - Specialized attention implementation
9. **Custom CUDA Kernels** (All events) - Maximum performance, high complexity

---

## References

- **Prefetching**: See `src/deployer/cuda_prefetcher.py` in this codebase
- **Layout Optimization**: See `src/deployer/layout_aware_module.py` in this codebase
- **PyTorch Performance Tuning**: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

