import inspect
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .data_types import OpProfileRecord
from .tensor_utils import iter_tensors, get_tensor_layout_info, measure_conversion_cost


class HookManager:
    """Manages forward hooks for profiling module execution."""

    def __init__(
        self,
        model: nn.Module,
        is_cuda: bool,
        measure_conversion_cost: bool,
        sample_conversion_for_all_shapes: bool,
    ):
        self.model = model
        self.is_cuda = is_cuda
        self.measure_conversion_cost = measure_conversion_cost
        self.sample_conversion_for_all_shapes = sample_conversion_for_all_shapes

        self._forward_start: Dict[int, float] = {}
        self._handles: List[Any] = []
        self.records: List[OpProfileRecord] = []
        self.conversion_cost_table: Dict[Tuple[int, ...], List[float]] = {}

    def register_hooks(self):
        """Attach forward hooks to all submodules."""
        for name, module in self.model.named_modules():
            if name == "":
                continue
            self._attach_hooks_to_module(name, module)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear_records(self):
        """Clear all collected profiling data."""
        self.records.clear()
        self.conversion_cost_table.clear()

    def _get_call_site(self) -> Optional[Tuple[str, int]]:
        """Get the file path and line number where the module was called from."""
        try:
            # Walk up the stack to find the first frame outside of PyTorch internals
            for frame_info in inspect.stack():
                filename = frame_info.filename
                # Skip PyTorch internal files and this profiler code
                if (
                    'torch' not in filename
                    and 'profiler' not in filename
                    and '<' not in filename  # Skip <string>, <stdin>, etc.
                ):
                    return (filename, frame_info.lineno)
        except Exception:
            pass
        return None

    def _attach_hooks_to_module(self, name: str, module: nn.Module):
        """Attach pre and forward hooks to a specific module."""

        def pre_hook(mod, inputs):
            if self.is_cuda:
                torch.cuda.synchronize()
            self._forward_start[id(mod)] = time.perf_counter()

        def fwd_hook(mod, inputs, output):
            if self.is_cuda:
                torch.cuda.synchronize()
            start = self._forward_start.pop(id(mod), None)
            if start is None:
                start = time.perf_counter()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0

            input_tensors = list(iter_tensors(inputs))
            output_tensors = list(iter_tensors(output))

            input_layouts = [get_tensor_layout_info(t) for t in input_tensors]
            output_layouts = [get_tensor_layout_info(t) for t in output_tensors]

            has_nc_in = any(not l.is_contiguous for l in input_layouts)
            has_nc_out = any(not l.is_contiguous for l in output_layouts)

            raw_conv_samples = []
            est_conv_cost_ms = None

            if self.measure_conversion_cost:
                targets = []
                if has_nc_in:
                    targets.extend([t for t in input_tensors if not t.is_contiguous()])
                if self.sample_conversion_for_all_shapes and input_tensors:
                    targets.append(input_tensors[0])

                for t in targets:
                    cost_ms = measure_conversion_cost(t, self.is_cuda)
                    if cost_ms is not None:
                        raw_conv_samples.append(cost_ms)
                        key = tuple(t.shape)
                        self.conversion_cost_table.setdefault(key, []).append(cost_ms)

                if raw_conv_samples:
                    est_conv_cost_ms = sum(raw_conv_samples) / len(raw_conv_samples)

            # Capture call site information
            call_site = self._get_call_site()
            extra = {}
            if call_site:
                extra["call_site_file"] = call_site[0]
                extra["call_site_line"] = call_site[1]

            record = OpProfileRecord(
                module_name=name,
                module_type=mod.__class__.__name__,
                phase="forward",
                has_noncontig_input=has_nc_in,
                has_noncontig_output=has_nc_out,
                input_layouts=input_layouts,
                output_layouts=output_layouts,
                forward_time_ms=elapsed_ms,
                estimated_conversion_cost_ms=est_conv_cost_ms,
                raw_conversion_samples_ms=raw_conv_samples,
                extra=extra,
            )

            self.records.append(record)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(fwd_hook))

