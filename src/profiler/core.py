from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .exporter import ProfileExporter
from .hooks import HookManager


class ContiguityProfiler:
    """
    Profiles PyTorch models to detect non-contiguous tensors and measure conversion costs.

    Attaches forward hooks to all modules to record tensor layouts, execution times,
    and conversion costs. Intended for offline profiling before training.
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

    @property
    def records(self):
        return self._hook_manager.records

    @property
    def conversion_cost_table(self):
        return self._hook_manager.conversion_cost_table

    def profile(
        self,
        example_inputs: Tuple[Any, ...],
        example_kwargs: Optional[Dict[str, Any]] = None,
        warmup: int = 1,
        iters: int = 1,
    ):
        """Run model with profiling hooks attached."""
        example_kwargs = example_kwargs or {}

        self._hook_manager.register_hooks()

        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(*example_inputs, **example_kwargs)

        self._hook_manager.clear_records()

        for _ in range(iters):
            if self._is_cuda:
                torch.cuda.synchronize()
            with torch.no_grad():
                _ = self.model(*example_inputs, **example_kwargs)
            if self._is_cuda:
                torch.cuda.synchronize()

        self._hook_manager.remove_hooks()

    def summarize(self, top_k: int = 20):
        """Print human-readable summary of profiling results."""
        ProfileExporter.summarize(self.records, self.conversion_cost_table, top_k)

    def export_json(self, path: str):
        """Export profiling data to JSON file."""
        ProfileExporter.export_json(self.records, self.conversion_cost_table, path)

