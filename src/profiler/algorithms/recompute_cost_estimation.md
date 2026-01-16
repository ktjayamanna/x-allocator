# Non-Contiguous Tensor Conversion Cost Measurement

### Measurement Function
The `measure_conversion_cost()` function preserves exact memory layouts:

```python
def measure_conversion_cost(t: torch.Tensor, is_cuda: bool) -> Optional[float]:
    """
    Measure conversion cost on tensor with same memory layout as original.
    Implemented to replace inaccurate torch.empty_like() approach.
    """
    if t.is_contiguous() or t.numel() == 0:
        return None
    
    # Create tensor with identical strides and memory pattern
    copy_with_layout = torch.empty_strided(
        t.shape, 
        t.stride(), 
        dtype=t.dtype, 
        device=t.device
    )
    copy_with_layout.copy_(t)  # Copy data into the non-contiguous layout
    
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    _ = copy_with_layout.contiguous()  # Convert the copy
    if is_cuda:
        torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) * 1000.0  # Convert seconds to milliseconds
```

## Deployment Details

### Location
The function has been deployed in `src/profiler/tensor_utils.py`.

### Integration with Training Loop
- **Real-time Measurement**: Conversion costs are now measured immediately upon discovery of non-contiguous tensors during forward hooks
- **No Batch Delays**: Measurements occur as non-contiguous tensors are detected, no need for wait-for-all approach (which doesn't simulate the environment in the training loop well)
- **Minimal Overhead**: Only non-contiguous tensors are copied.
- **Averaging**: Measurements are averaged across 2 batches for reliable results.

### Current Workflow
1. During forward pass execution
2. Forward hooks detect non-contiguous tensors
3. Immediate measurement using replica with identical memory layout
4. Results logged and averaged across batches
