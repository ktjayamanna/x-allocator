"""
CUDA-aware data prefetcher for overlapping data loading with computation.

This module implements asynchronous data loading using CUDA streams and events
to hide data transfer latency during training on single-GPU setups.
"""

from typing import Optional, Iterator, Any
import torch
from torch.utils.data import DataLoader


class CudaPrefetcher:
    """
    Asynchronous data prefetcher using CUDA streams for single-GPU training.
    
    This prefetcher overlaps:
    - Host-to-device data transfers (on prefetch stream)
    - Optional tensor layout conversions (on conversion stream)
    - Model computation (on default stream)
    
    The key insight is that while the GPU is busy computing step N, we can
    simultaneously transfer data for step N+1 on a separate CUDA stream.
    
    Args:
        loader: PyTorch DataLoader to prefetch from
        device: Target CUDA device (e.g., "cuda:0")
        convert_inputs: Whether to apply .contiguous() during prefetch
        use_pinned_memory: Whether to use pinned memory for faster transfers
    
    Example:
        >>> loader = DataLoader(dataset, batch_size=32, pin_memory=True)
        >>> prefetcher = CudaPrefetcher(loader, device="cuda")
        >>> for batch in prefetcher:
        ...     outputs = model(**batch)  # Data already on GPU
    """
    
    def __init__(
        self,
        loader: DataLoader,
        device: str = "cuda",
        convert_inputs: bool = False,
        use_pinned_memory: bool = True,
    ):
        self.loader = loader
        self.device = torch.device(device)
        self.convert_inputs = convert_inputs
        
        # Verify CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CudaPrefetcher requires CUDA")
        
        # Create dedicated CUDA streams for async operations
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.conversion_stream = torch.cuda.Stream(device=self.device) if convert_inputs else None
        
        # Warn if pinned memory is not enabled
        if use_pinned_memory and not loader.pin_memory:
            print("Warning: DataLoader.pin_memory=False. "
                  "Set pin_memory=True for faster transfers.")
        
        # State
        self.iterator: Optional[Iterator] = None
        self.next_batch: Optional[Any] = None
        self.next_event: Optional[torch.cuda.Event] = None
    
    def __iter__(self):
        """Initialize iterator and prefetch first batch."""
        self.iterator = iter(self.loader)
        self._prefetch_next()
        return self
    
    def __next__(self):
        """
        Get next batch (already on GPU) and prefetch the following batch.
        
        This is where the magic happens:
        1. Wait for the prefetched batch to be ready
        2. Return it to the caller (who will use it for computation)
        3. While computation happens, prefetch the next batch in parallel
        """
        # Wait for current batch to be ready
        if self.next_event is not None:
            self.next_event.wait()
        
        # Get the batch that was prefetched
        batch = self.next_batch
        
        if batch is None:
            raise StopIteration
        
        # Prefetch next batch while caller processes current batch
        self._prefetch_next()
        
        return batch
    
    def _prefetch_next(self):
        """
        Asynchronously prefetch next batch to GPU.
        
        This runs on a separate CUDA stream, allowing it to overlap with
        computation on the default stream.
        """
        try:
            # Get next batch from CPU loader
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            self.next_event = None
            return
        
        # Create event to signal when prefetch is complete
        self.next_event = torch.cuda.Event()
        
        # Perform async transfer on prefetch stream
        with torch.cuda.stream(self.prefetch_stream):
            # Transfer batch to GPU (non-blocking if pinned memory)
            batch = self._move_to_device(batch, non_blocking=True)
            
            # Optional: Apply contiguous conversion on separate stream
            if self.convert_inputs and self.conversion_stream is not None:
                with torch.cuda.stream(self.conversion_stream):
                    batch = self._convert_batch(batch)
                # Wait for conversion to complete
                self.prefetch_stream.wait_stream(self.conversion_stream)
            
            # Record event to signal completion
            self.prefetch_stream.record_event(self.next_event)
        
        self.next_batch = batch
    
    def _move_to_device(self, batch: Any, non_blocking: bool = True) -> Any:
        """Recursively move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=non_blocking)
        elif isinstance(batch, dict):
            return {k: self._move_to_device(v, non_blocking) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._move_to_device(x, non_blocking) for x in batch)
        else:
            return batch
    
    def _convert_batch(self, batch: Any) -> Any:
        """Apply .contiguous() to all tensors in batch."""
        if isinstance(batch, torch.Tensor):
            return batch.contiguous() if not batch.is_contiguous() else batch
        elif isinstance(batch, dict):
            return {k: self._convert_batch(v) for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(self._convert_batch(x) for x in batch)
        else:
            return batch
    
    def __len__(self):
        """Return length of underlying loader."""
        return len(self.loader)

