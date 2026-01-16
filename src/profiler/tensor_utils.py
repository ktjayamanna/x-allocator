import time
from typing import Optional

import torch

from .data_types import TensorLayoutInfo


def iter_tensors(obj):
    """Recursively extract all tensors from nested structures."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            yield from iter_tensors(x)
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from iter_tensors(x)


def get_tensor_layout_info(t: torch.Tensor) -> TensorLayoutInfo:
    """Extract layout information from a tensor."""
    return TensorLayoutInfo(
        shape=tuple(t.shape),
        strides=tuple(t.stride()),
        numel=t.numel(),
        ndim=t.ndim,
        dtype=str(t.dtype).replace("torch.", ""),
        device=str(t.device),
        is_contiguous=t.is_contiguous(),
        tensor_id=id(t),
    )


def measure_conversion_cost(t: torch.Tensor, is_cuda: bool) -> Optional[float]:
    """
    Measure conversion cost on tensor with same memory layout as original.

    Creates a replica that preserves the exact memory layout (strides) of the
    original tensor, then measures the cost of converting it to contiguous.
    This gives accurate conversion cost estimates for the actual tensor layouts
    flowing through the model.
    """
    if t.is_contiguous() or t.numel() == 0:
        return None

    with torch.no_grad():
        try:
            # Create tensor with identical strides and memory pattern
            copy_with_layout = torch.empty_strided(
                t.shape,
                t.stride(),
                dtype=t.dtype,
                device=t.device
            )
            copy_with_layout.copy_(t)  # Copy data into the non-contiguous layout
        except RuntimeError:
            # Handle cases where strides cause overlapping memory (e.g., from expand())
            # Fall back to cloning the original tensor which preserves the view
            copy_with_layout = t.clone()
            if copy_with_layout.is_contiguous():
                # If clone made it contiguous, we can't measure - use as_strided
                # to recreate a similar non-contiguous layout
                copy_with_layout = torch.as_strided(
                    t.clone().contiguous(),
                    t.shape,
                    t.stride(),
                    storage_offset=0
                )

        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = copy_with_layout.contiguous()  # Convert the copy
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) * 1000.0  # Converting seconds to milliseconds

