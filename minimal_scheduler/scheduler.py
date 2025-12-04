"""
Runtime Scheduler - Step 7

Production-level scheduler that executes the greedy pattern at runtime.

Key differences from static schedule:
- No pre-generated schedule (constant memory)
- Executes pattern dynamically (flexible)
- Handles arbitrary number of batches (scalable)

Pattern:
  First batch:      prefetch(0) → wait(0) → compute(0) → release(0)
  Remaining batches: compute(N) + prefetch(N+1) → release(N) → wait(N+1)
"""

import torch
from tqdm import tqdm


class PrefetchScheduler:
    """
    Production scheduler that overlaps data loading with compute.

    Implements greedy strategy: start prefetch as early as possible.

    This is a WRAPPER around your existing training loop - it just handles
    the prefetching pattern, and delegates the actual training to a callback.
    """

    def __init__(self, device):
        """
        Args:
            device: Device to run on (cuda/cpu)
        """
        self.device = device

        # Track current and next batch
        self.current_batch = None
        self.next_batch = None

    def run_epoch(self, dataloader, train_step_fn, show_progress=True):
        """
        Run one training epoch with prefetching.

        Args:
            dataloader: PyTorch DataLoader
            train_step_fn: Function that takes (x, y) and performs one training step.
                          Should return loss value (float).
                          Example: lambda x, y: train_step(model, x, y, optimizer)
            show_progress: Whether to show tqdm progress bar

        Returns:
            Average loss for the epoch
        """
        total_loss = 0.0
        num_batches = 0

        # Convert to list to know total length
        batches = list(dataloader)

        # First batch: Must prefetch and wait (can't overlap)
        if len(batches) == 0:
            return 0.0

        self.current_batch = self._prefetch(batches[0])
        self.current_batch = self._wait(self.current_batch)

        # Main loop: Execute greedy pattern
        iterator = tqdm(range(len(batches)), desc="Training") if show_progress else range(len(batches))

        for i in iterator:
            # Prefetch next batch (if exists) - starts immediately
            if i < len(batches) - 1:
                self.next_batch = self._prefetch(batches[i + 1])

            # Compute current batch (overlaps with prefetch!)
            x, y = self.current_batch
            loss = train_step_fn(x, y)
            total_loss += loss
            num_batches += 1

            # Release current batch
            self._release(self.current_batch)

            # Wait for next batch (should already be done if prefetch < compute)
            if i < len(batches) - 1:
                self.current_batch = self._wait(self.next_batch)

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _prefetch(self, batch):
        """
        Async transfer batch to GPU.

        Args:
            batch: (x, y) tuple from dataloader

        Returns:
            (x_gpu, y_gpu) tuple (transfer may still be in progress)
        """
        x, y = batch

        # non_blocking=True enables async transfer
        x_gpu = x.to(self.device, non_blocking=True)
        y_gpu = y.to(self.device, non_blocking=True)

        return (x_gpu, y_gpu)

    def _wait(self, batch):
        """
        Wait for async transfer to complete.

        Args:
            batch: (x_gpu, y_gpu) tuple from _prefetch

        Returns:
            Same batch (now guaranteed to be on GPU)
        """
        # Synchronize to ensure transfer is complete
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return batch

    def _release(self, batch):
        """
        Free GPU memory for batch.

        Args:
            batch: (x, y) tuple to release
        """
        # Python's garbage collector will handle this
        # Explicit del helps free memory immediately
        del batch
