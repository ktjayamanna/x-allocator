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
    """Measure cost of converting a non-contiguous tensor to contiguous."""
    if t.numel() == 0:
        return None

    with torch.no_grad():
        dummy = torch.empty_like(t)

        if dummy.ndim >= 2:
            nc = dummy.transpose(-1, -2).contiguous().transpose(-1, -2)
        else:
            nc = dummy[::2]
            if nc.numel() == 0:
                nc = dummy

        if nc.is_contiguous():
            return None

        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = nc.contiguous()
        if is_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()

        return (end - start) * 1000.0

