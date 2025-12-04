"""
Prefetch Scheduler - overlaps I/O with compute

Pattern (greedy strategy):
  First:     prefetch(0) → wait(0) → compute(0) → release(0)
  Rest:      prefetch(i+1) + compute(i) → release(i) → wait(i+1)

_compute is implemented by calling the user-provided train_step_fn(x, y)
because the scheduler doesn't need to know HOW to train - it just orchestrates WHEN.

_release is implicit via Python GC when reassigning self.current_batch.

"""
import torch
from tqdm import tqdm

class PrefetchScheduler:
    def __init__(self, device):
        self.device = device
        self.current_batch = None
        self.next_batch = None

    def run_epoch(self, dataloader, train_step_fn):
        batches = list(dataloader)
        if not batches:
            return

        # First batch: prefetch → wait
        self.current_batch = self._prefetch(batches[0])
        self.current_batch = self._wait(self.current_batch)

        # Main loop: overlap prefetch(i+1) with compute(i)
        for i in tqdm(range(len(batches))):
            if i < len(batches) - 1:
                self.next_batch = self._prefetch(batches[i + 1])
            x, y = self.current_batch
            train_step_fn(x, y)
            if i < len(batches) - 1:
                self.current_batch = self._wait(self.next_batch)

    def _prefetch(self, batch):
        x, y = batch
        return (x.to(self.device, non_blocking=True),
                y.to(self.device, non_blocking=True))

    def _wait(self, batch):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return batch
