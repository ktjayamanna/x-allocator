import inspect
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .data_types import OpProfileRecord, TensorFingerprint
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

        # Legacy tensor flow tracking (kept for backward compatibility)
        self.tensor_producers: Dict[int, int] = {}  # tensor_id -> op_id that produced it
        self.tensor_consumers: Dict[int, List[int]] = {}  # tensor_id -> [op_ids that consumed it]
        self.tensor_info: Dict[int, Tuple[Tuple[int, ...], bool, Optional[float]]] = {}  # tensor_id -> (shape, is_contiguous, measured_conv_cost_ms)

        # Fingerprint-based tensor flow tracking (robust identification)
        # Maps fingerprint_key -> (TensorFingerprint, op_id) for producers
        self.fingerprint_producers: Dict[str, Tuple[TensorFingerprint, int]] = {}
        # Maps fingerprint_key -> [(TensorFingerprint, op_id)] for consumers
        self.fingerprint_consumers: Dict[str, List[Tuple[TensorFingerprint, int]]] = {}
        # Maps (object_id, data_ptr) -> fingerprint_key for cross-referencing
        self._tensor_to_fingerprint: Dict[Tuple[int, int], str] = {}

        # Three-field persistence tracking (Explicit Naming Contract)
        # Maps anchor_name -> TensorFingerprint for iteration 1
        self._iteration_1_fingerprints: Dict[str, TensorFingerprint] = {}
        # Maps anchor_name -> TensorFingerprint for iteration 2
        self._iteration_2_fingerprints: Dict[str, TensorFingerprint] = {}
        # Current iteration index (0-based, set during profiling)
        self._current_iteration: int = 0
        # Final persistence classification: anchor_name -> "persistent" | "transient"
        self.tensor_persistence: Dict[str, str] = {}
        # Maps anchor_name -> tensor info for export
        self.anchor_tensor_info: Dict[str, Tuple[Tuple[int, ...], bool, Optional[float]]] = {}

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
        self.tensor_producers.clear()
        self.tensor_consumers.clear()
        self.tensor_info.clear()
        self.fingerprint_producers.clear()
        self.fingerprint_consumers.clear()
        self._tensor_to_fingerprint.clear()
        self._iteration_1_fingerprints.clear()
        self._iteration_2_fingerprints.clear()
        self._current_iteration = 0
        self.tensor_persistence.clear()
        self.anchor_tensor_info.clear()

    def set_current_iteration(self, iteration: int):
        """Set the current iteration index for persistence tracking."""
        self._current_iteration = iteration

    def compute_persistence(self):
        """
        Compare fingerprints from iteration 1 vs iteration 2 to classify persistence.

        CASE A: Persistent (Target for Optimization)
            - Name Matches AND Object ID Matches AND Data Pointer Matches
            - Conclusion: The tensor is stable. If non-contiguous, we fix it.

        CASE B: Transient (Ignore)
            - Name Matches BUT Object ID Changed OR Data Pointer Changed
            - Conclusion: Tensor was recreated or moved. Do not optimize.
        """
        all_anchor_names = set(self._iteration_1_fingerprints.keys()) | set(self._iteration_2_fingerprints.keys())

        for anchor_name in all_anchor_names:
            fp1 = self._iteration_1_fingerprints.get(anchor_name)
            fp2 = self._iteration_2_fingerprints.get(anchor_name)

            if fp1 is None or fp2 is None:
                # Only appeared in one iteration - classify as transient
                self.tensor_persistence[anchor_name] = "transient"
            elif fp1.object_id == fp2.object_id and fp1.data_ptr == fp2.data_ptr:
                # CASE A: All three fields match - persistent
                self.tensor_persistence[anchor_name] = "persistent"
            else:
                # CASE B: Name matches but object_id or data_ptr changed - transient
                self.tensor_persistence[anchor_name] = "transient"

    def _extract_explicit_name(self, inputs: Tuple) -> Optional[str]:
        """
        Extract explicit name string from inputs (for Mark module).

        The Mark module passes (tensor, name: str) as inputs.
        Returns the string if found, None otherwise.
        """
        for item in inputs:
            if isinstance(item, str):
                return item
        return None

    def get_fingerprint_flow(self) -> Dict[str, Any]:
        """
        Build tensor flow graph using TensorFingerprint-based tracking.

        Returns a dictionary containing:
        - producers: Maps fingerprint_key -> {fingerprint_info, producer_op_id}
        - consumers: Maps fingerprint_key -> [{fingerprint_info, consumer_op_id}, ...]
        - edges: List of (producer_op_id, consumer_op_id, fingerprint_info) tuples
        """
        producers = {}
        consumers = {}
        edges = []

        # Build producer info
        for fp_key, (fingerprint, op_id) in self.fingerprint_producers.items():
            producers[fp_key] = {
                "anchor_name": fingerprint.anchor_name,
                "object_id": fingerprint.object_id,
                "data_ptr": fingerprint.data_ptr,
                "producer_op_id": op_id,
            }

        # Build consumer info and edges
        for fp_key, consumer_list in self.fingerprint_consumers.items():
            consumers[fp_key] = []
            producer_op_id = None
            if fp_key in self.fingerprint_producers:
                _, producer_op_id = self.fingerprint_producers[fp_key]

            for fingerprint, consumer_op_id in consumer_list:
                consumers[fp_key].append({
                    "anchor_name": fingerprint.anchor_name,
                    "object_id": fingerprint.object_id,
                    "data_ptr": fingerprint.data_ptr,
                    "consumer_op_id": consumer_op_id,
                })
                if producer_op_id is not None:
                    edges.append({
                        "producer_op_id": producer_op_id,
                        "consumer_op_id": consumer_op_id,
                        "fingerprint_key": fp_key,
                        "anchor_name": fingerprint.anchor_name,
                    })

        return {
            "producers": producers,
            "consumers": consumers,
            "edges": edges,
        }

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

            # Extract tensor IDs
            input_tensor_ids = [id(t) for t in input_tensors]
            output_tensor_ids = [id(t) for t in output_tensors]

            # Hybrid Naming Strategy:
            # 1. Explicit Names (Priority): Check for string argument from Mark module
            # 2. Implicit Names (Fallback): Use module's registry name
            explicit_name = self._extract_explicit_name(inputs)
            anchor_key = explicit_name if explicit_name is not None else name

            # Current op_id
            current_op_id = len(self.records)

            # Legacy tracking (backward compatibility)
            for tid in input_tensor_ids:
                self.tensor_consumers.setdefault(tid, []).append(current_op_id)
            for tid in output_tensor_ids:
                self.tensor_producers[tid] = current_op_id

            raw_conv_samples = []
            est_conv_cost_ms = None
            tensor_conv_costs: Dict[int, Optional[float]] = {}  # tensor_id -> measured conversion cost

            if self.measure_conversion_cost:
                # Measure conversion cost for each tensor individually
                all_tensors = input_tensors + output_tensors
                for t in all_tensors:
                    tid = id(t)
                    if tid not in tensor_conv_costs:
                        cost_ms = measure_conversion_cost(t, self.is_cuda)
                        tensor_conv_costs[tid] = cost_ms
                        if cost_ms is not None:
                            raw_conv_samples.append(cost_ms)
                            key = tuple(t.shape)
                            self.conversion_cost_table.setdefault(key, []).append(cost_ms)

                if raw_conv_samples:
                    est_conv_cost_ms = sum(raw_conv_samples) / len(raw_conv_samples)

            # Store tensor info for building flow graph later (legacy)
            for t in input_tensors:
                tid = id(t)
                if tid not in self.tensor_info:
                    conv_cost = tensor_conv_costs.get(tid)
                    self.tensor_info[tid] = (tuple(t.shape), t.is_contiguous(), conv_cost)

            for t in output_tensors:
                tid = id(t)
                if tid not in self.tensor_info:
                    conv_cost = tensor_conv_costs.get(tid)
                    self.tensor_info[tid] = (tuple(t.shape), t.is_contiguous(), conv_cost)

            # Fingerprint-based producer/consumer tracking
            # Track input tensors as consumers of this operation
            for idx, t in enumerate(input_tensors):
                tensor_key = (id(t), t.data_ptr())
                # Look up if this tensor was produced by a previous op
                if tensor_key in self._tensor_to_fingerprint:
                    fp_key = self._tensor_to_fingerprint[tensor_key]
                    # Get the fingerprint from the producer
                    if fp_key in self.fingerprint_producers:
                        producer_fp, _ = self.fingerprint_producers[fp_key]
                        self.fingerprint_consumers.setdefault(fp_key, []).append(
                            (producer_fp, current_op_id)
                        )

            # Track output tensors as produced by this operation
            for idx, t in enumerate(output_tensors):
                tensor_anchor = f"{anchor_key}:out_{idx}" if len(output_tensors) > 1 else f"{anchor_key}:out"
                conv_cost = tensor_conv_costs.get(id(t))

                fingerprint = TensorFingerprint(
                    anchor_name=tensor_anchor,
                    object_id=id(t),
                    data_ptr=t.data_ptr(),
                )

                fp_key = f"{tensor_anchor}|{id(t)}|{t.data_ptr()}"
                self.fingerprint_producers[fp_key] = (fingerprint, current_op_id)
                self._tensor_to_fingerprint[(id(t), t.data_ptr())] = fp_key

            # Three-Field Persistence Tracking (with Early Exit optimization):
            # Only track non-contiguous tensors to keep profiler lightweight
            all_tensors = input_tensors + output_tensors
            for idx, t in enumerate(all_tensors):
                # Early Exit: Skip contiguous tensors - we only care about
                # tensors that currently have a memory layout problem
                if t.is_contiguous():
                    continue

                # Create unique anchor name per tensor if multiple tensors
                tensor_anchor = f"{anchor_key}:tensor_{idx}" if len(all_tensors) > 1 else anchor_key

                fingerprint = TensorFingerprint(
                    anchor_name=tensor_anchor,
                    object_id=id(t),
                    data_ptr=t.data_ptr(),
                )

                # Record fingerprint for current iteration
                if self._current_iteration == 0:
                    self._iteration_1_fingerprints[tensor_anchor] = fingerprint
                else:
                    self._iteration_2_fingerprints[tensor_anchor] = fingerprint

                # Store tensor info by anchor name for export
                conv_cost = tensor_conv_costs.get(id(t))
                self.anchor_tensor_info[tensor_anchor] = (tuple(t.shape), t.is_contiguous(), conv_cost)

            # Capture call site information
            call_site = self._get_call_site()
            extra = {}
            if call_site:
                extra["call_site_file"] = call_site[0]
                extra["call_site_line"] = call_site[1]

            # Add anchor key to extra for tracking
            extra["anchor_key"] = anchor_key
            if explicit_name is not None:
                extra["explicit_name"] = explicit_name

            record = OpProfileRecord(
                module_name=name,
                module_type=mod.__class__.__name__,
                phase="forward",
                has_noncontig_input=has_nc_in,
                has_noncontig_output=has_nc_out,
                input_layouts=input_layouts,
                output_layouts=output_layouts,
                input_tensor_ids=input_tensor_ids,
                output_tensor_ids=output_tensor_ids,
                forward_time_ms=elapsed_ms,
                estimated_conversion_cost_ms=est_conv_cost_ms,
                raw_conversion_samples_ms=raw_conv_samples,
                extra=extra,
            )

            self.records.append(record)

        self._handles.append(module.register_forward_pre_hook(pre_hook))
        self._handles.append(module.register_forward_hook(fwd_hook))

