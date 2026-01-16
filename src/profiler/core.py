from typing import Any, Callable, List, Optional
import time

import torch
import torch.nn as nn

from .exporter import ProfileExporter
from .hooks import HookManager
from .data_types import IdleEventRecord


class ContiguityProfiler:
    """
    Profiles full training loop to detect non-contiguous tensors and GPU idle time.

    Tracks:
    - Non-contiguous tensors in model forward pass
    - Conversion costs for .contiguous() calls
    - GPU idle time during data transfer (CPU → GPU)
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        measure_conversion_cost: bool = True,
        sample_conversion_for_all_shapes: bool = True,
    ):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.measure_conversion_cost = measure_conversion_cost
        self.sample_conversion_for_all_shapes = sample_conversion_for_all_shapes

        self._is_cuda = self.device.type == "cuda"
        self._hook_manager = HookManager(
            model, self._is_cuda, measure_conversion_cost, sample_conversion_for_all_shapes
        )

        self.idle_events: List[IdleEventRecord] = []
        self.num_ops_per_iter: Optional[int] = None

    @property
    def records(self):
        return self._hook_manager.records

    @property
    def conversion_cost_table(self):
        return self._hook_manager.conversion_cost_table

    @property
    def tensor_producers(self):
        return self._hook_manager.tensor_producers

    @property
    def tensor_consumers(self):
        return self._hook_manager.tensor_consumers

    @property
    def tensor_info(self):
        return self._hook_manager.tensor_info

    @property
    def tensor_persistence(self):
        """Maps anchor_name -> 'persistent' | 'transient'."""
        return self._hook_manager.tensor_persistence

    @property
    def anchor_tensor_info(self):
        """Maps anchor_name -> (shape, is_contiguous, conv_cost)."""
        return self._hook_manager.anchor_tensor_info

    def profile(
        self,
        dataloader,
        train_step_fn: Callable[[Any, Any], None],
        warmup: int = 1,
        iters: int = 1,
    ):
        """
        Profile full training loop including data transfer and GPU idle time.

        Args:
            dataloader: PyTorch DataLoader providing batches
            train_step_fn: Function that takes (x, y) and performs training step
            warmup: Number of warmup iterations
            iters: Number of profiling iterations
        """
        self._hook_manager.register_hooks()

        data_iter = iter(dataloader)

        # Warmup
        for _ in range(warmup):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            train_step_fn(x, y)

        # Clear warmup data
        self._hook_manager.clear_records()
        self.idle_events.clear()

        # Profile iterations
        num_ops_per_iter = None
        for iter_idx in range(iters):
            # Set current iteration for persistence tracking
            self._hook_manager.set_current_iteration(iter_idx)

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            x, y = batch

            # Measure data transfer time (GPU idle time)
            if self._is_cuda:
                torch.cuda.synchronize()
            transfer_start = time.perf_counter()
            x_gpu, y_gpu = x.to(self.device), y.to(self.device)
            if self._is_cuda:
                torch.cuda.synchronize()
            transfer_end = time.perf_counter()
            transfer_ms = (transfer_end - transfer_start) * 1000.0

            # Determine op_ids for this idle event
            # Idle event happens BEFORE the forward pass of this iteration
            if num_ops_per_iter is None:
                # First iteration - we don't know how many ops yet
                before_op_id = None
                after_op_id = 0  # Will execute op 0 next
            else:
                # Subsequent iterations
                before_op_id = (iter_idx - 1) * num_ops_per_iter + (num_ops_per_iter - 1)
                after_op_id = iter_idx * num_ops_per_iter

            # Record idle event
            idle_event = IdleEventRecord(
                event_name="data_transfer",
                event_type="cpu_to_gpu_transfer",
                duration_ms=transfer_ms,
                tensor_shapes=[tuple(x_gpu.shape), tuple(y_gpu.shape)],
                tensor_dtypes=[str(x_gpu.dtype), str(y_gpu.dtype)],
                before_op_id=before_op_id,
                after_op_id=after_op_id,
                extra={"batch_size": x_gpu.shape[0] if x_gpu.ndim > 0 else 1, "iteration": iter_idx},
            )
            self.idle_events.append(idle_event)

            # Run training step (forward, loss, backward, optimizer)
            records_before = len(self.records)
            train_step_fn(x_gpu, y_gpu)
            records_after = len(self.records)

            # After first iteration, we know how many ops per iteration
            if num_ops_per_iter is None:
                num_ops_per_iter = records_after - records_before
                self.num_ops_per_iter = num_ops_per_iter

        # Compute persistence classification after profiling iterations
        self._hook_manager.compute_persistence()

        self._hook_manager.remove_hooks()

    def summarize(self, top_k: int = 20):
        """Print human-readable summary of profiling results."""
        ProfileExporter.summarize(self.records, self.conversion_cost_table, top_k)

        if self.idle_events:
            print("\n=== GPU Idle Events Summary ===\n")
            total_idle_ms = sum(e.duration_ms for e in self.idle_events)
            avg_idle_ms = total_idle_ms / len(self.idle_events)
            print(f"Total idle events: {len(self.idle_events)}")
            print(f"Total idle time: {total_idle_ms:.2f} ms")
            print(f"Average idle time per event: {avg_idle_ms:.2f} ms")
            print()

    def export_json(self, path: str):
        """Export profiling data to JSON file."""
        ProfileExporter.export_json(
            self.records,
            self.conversion_cost_table,
            self.idle_events,
            self.tensor_producers,
            self.tensor_consumers,
            self.tensor_info,
            self.num_ops_per_iter,
            path,
            self.tensor_persistence,
            self.anchor_tensor_info,
        )

