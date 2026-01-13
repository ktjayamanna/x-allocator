import json
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from .data_types import OpProfileRecord, TensorLayoutInfo, IdleEventRecord


class ProfileExporter:
    """Handles exporting profiling data to various formats."""

    @staticmethod
    def export_json(
        records: List[OpProfileRecord],
        conversion_cost_table: Dict[Tuple[int, ...], List[float]],
        idle_events: List[IdleEventRecord],
        tensor_producers: Dict[int, int],
        tensor_consumers: Dict[int, List[int]],
        tensor_info: Dict[int, Tuple[Tuple[int, ...], bool, Any]],
        num_ops_per_iter: Optional[int],
        path: str,
    ):
        """Export raw profiling data to JSON file (profile.json)."""
        data = {
            "records": [ProfileExporter._record_to_dict(r) for r in records],
            "conversion_cost_table": {
                str(shape): samples for shape, samples in conversion_cost_table.items()
            },
            "gpu_idle_events": [ProfileExporter._idle_event_to_dict(e) for e in idle_events],
            "tensor_flow": ProfileExporter._build_tensor_flow_graph(
                tensor_producers, tensor_consumers, tensor_info, num_ops_per_iter
            ),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Exported profiling data to {path}")

    @staticmethod
    def summarize(
        records: List[OpProfileRecord],
        conversion_cost_table: Dict[Tuple[int, ...], List[float]],
        top_k: int = 20,
    ):
        """Print human-readable summary of profiling results."""
        print("\n=== Contiguity Profiler Summary ===\n")

        problematic = [
            r for r in records if r.has_noncontig_input or r.has_noncontig_output
        ]
        problematic = sorted(
            problematic, key=lambda r: r.forward_time_ms, reverse=True
        )

        print(">> Top ops with non-contiguous tensors (by forward time):")
        for rec in problematic[:top_k]:
            conv_cost_str = (
                f"est_conv_cost={rec.estimated_conversion_cost_ms:.3f} ms "
                if rec.estimated_conversion_cost_ms is not None
                else ""
            )
            print(
                f"- {rec.module_name} ({rec.module_type}): "
                f"time={rec.forward_time_ms:.3f} ms, "
                f"noncontig_in={rec.has_noncontig_input}, "
                f"noncontig_out={rec.has_noncontig_output}, "
                f"{conv_cost_str}"
            )

        print("\n>> Conversion cost model (shape -> avg contiguous() cost, ms):")
        for shape, samples in conversion_cost_table.items():
            avg = sum(samples) / len(samples)
            print(f"  shape={shape}, avg_cost={avg:.4f} ms, samples={len(samples)}")

        print("\n=== End of Summary ===\n")

    @staticmethod
    def _build_tensor_flow_graph(
        tensor_producers: Dict[int, int],
        tensor_consumers: Dict[int, List[int]],
        tensor_info: Dict[int, Tuple[Tuple[int, ...], bool, Any]],
        num_ops_per_iter: Optional[int],
    ) -> Dict[str, Any]:
        """Build tensor flow graph from producer-consumer relationships."""
        tensor_flow = {}

        # Get all unique tensor IDs
        all_tensor_ids = set(tensor_producers.keys()) | set(tensor_consumers.keys())

        for tid in all_tensor_ids:
            shape, is_contiguous, measured_conv_cost = tensor_info.get(tid, ((), True, None))

            # Compute tensor lifetime
            lifetime = ProfileExporter._compute_tensor_lifetime(
                tid, tensor_producers, tensor_consumers, num_ops_per_iter
            )

            tensor_flow[str(tid)] = {
                "tensor_id": tid,
                "shape": list(shape),
                "is_contiguous": is_contiguous,
                "produced_by": tensor_producers.get(tid),
                "consumed_by": tensor_consumers.get(tid, []),
                "measured_conv_cost_ms": measured_conv_cost,
                "lifetime": lifetime,
            }

        return tensor_flow

    @staticmethod
    def _record_to_dict(r: OpProfileRecord) -> Dict[str, Any]:
        """Convert OpProfileRecord to dictionary."""

        def layout_to_dict(l: TensorLayoutInfo) -> Dict[str, Any]:
            return asdict(l)

        return {
            "module_name": r.module_name,
            "module_type": r.module_type,
            "phase": r.phase,
            "has_noncontig_input": r.has_noncontig_input,
            "has_noncontig_output": r.has_noncontig_output,
            "input_layouts": [layout_to_dict(l) for l in r.input_layouts],
            "output_layouts": [layout_to_dict(l) for l in r.output_layouts],
            "input_tensor_ids": r.input_tensor_ids,
            "output_tensor_ids": r.output_tensor_ids,
            "forward_time_ms": r.forward_time_ms,
            "estimated_conversion_cost_ms": r.estimated_conversion_cost_ms,
            "raw_conversion_samples_ms": r.raw_conversion_samples_ms,
            "extra": r.extra,
        }

    @staticmethod
    def _idle_event_to_dict(e: IdleEventRecord) -> Dict[str, Any]:
        """Convert IdleEventRecord to dictionary."""
        return {
            "event_name": e.event_name,
            "event_type": e.event_type,
            "duration_ms": e.duration_ms,
            "tensor_shapes": [list(shape) for shape in e.tensor_shapes],
            "tensor_dtypes": e.tensor_dtypes,
            "before_op_id": e.before_op_id,
            "after_op_id": e.after_op_id,
            "extra": e.extra,
        }

    @staticmethod
    def _compute_tensor_lifetime(
        tensor_id: int,
        tensor_producers: Dict[int, int],
        tensor_consumers: Dict[int, List[int]],
        num_ops_per_iter: Optional[int],
    ) -> str:
        """
        Compute tensor lifetime based on which iterations it appears in.

        Returns:
            "persistent" - tensor appears in multiple iterations (worth converting during idle time)
            "batch_specific" - tensor only appears in one iteration (not worth converting)
            "unknown" - not enough profiling data to determine (need iters >= 2)
        """
        if num_ops_per_iter is None or num_ops_per_iter == 0:
            return "unknown"

        # Collect all op_ids where this tensor appears (as input or output)
        all_op_ids = []

        if tensor_id in tensor_producers:
            all_op_ids.append(tensor_producers[tensor_id])

        if tensor_id in tensor_consumers:
            all_op_ids.extend(tensor_consumers[tensor_id])

        if not all_op_ids:
            return "unknown"

        # Convert op_ids to iteration numbers
        iterations = {op_id // num_ops_per_iter for op_id in all_op_ids}

        # If tensor appears in multiple iterations, it's persistent
        if len(iterations) > 1:
            return "persistent"
        else:
            return "batch_specific"

    @staticmethod
    def build_schedule_input(profile: Dict[str, Any]) -> Dict[str, Any]:
        """Build scheduler-ready input from profiler data using measured tensor costs."""
        ops = []
        op_id = 0
        tensor_flow = profile.get("tensor_flow", {})

        for rec in profile.get("records", []):
            fwd = float(rec.get("forward_time_ms", 0.0))
            has_nc_in = bool(rec.get("has_noncontig_input", False))
            has_nc_out = bool(rec.get("has_noncontig_output", False))

            # Get tensor IDs for this op
            input_tensor_ids = rec.get("input_tensor_ids", [])
            output_tensor_ids = rec.get("output_tensor_ids", [])

            layouts = rec.get("input_layouts", []) + rec.get("output_layouts", [])
            if layouts:
                main_layout = max(
                    layouts,
                    key=lambda l: int(math.prod(l["shape"])) if l.get("shape") else 0
                )
                shape = tuple(main_layout["shape"])
            else:
                shape = ()

            # Get measured conversion cost from tensor flow graph
            # Use the first input tensor's measured cost if available
            measured_conv_cost = None
            for tid in input_tensor_ids:
                tensor_info = tensor_flow.get(str(tid), {})
                if tensor_info.get("measured_conv_cost_ms") is not None:
                    measured_conv_cost = tensor_info.get("measured_conv_cost_ms")
                    break

            # Extract call site information from extra field
            extra = rec.get("extra", {})
            call_site_file = extra.get("call_site_file")
            call_site_line = extra.get("call_site_line")

            op_dict = {
                "op_id": op_id,
                "module_name": rec.get("module_name"),
                "module_type": rec.get("module_type"),
                "forward_time_ms": fwd,
                "main_tensor_shape": list(shape),
                "has_noncontig_input": has_nc_in,
                "has_noncontig_output": has_nc_out,
                "measured_conv_cost_ms": measured_conv_cost,
                "input_tensor_ids": input_tensor_ids,
                "output_tensor_ids": output_tensor_ids,
            }

            # Add call site info if available
            if call_site_file is not None:
                op_dict["call_site_file"] = call_site_file
            if call_site_line is not None:
                op_dict["call_site_line"] = call_site_line

            ops.append(op_dict)

            op_id += 1

        return {"ops": ops}

    @staticmethod
    def export_schedule_json(profile_path: str, output_path: str):
        """
        Export schedule.json for compiler.

        Contains:
        - ops: scheduler-ready operations list with tensor IDs
        - gpu_idle_events: GPU idle events with op references
        - tensor_flow: tensor-level data flow graph with measured conversion costs

        Args:
            profile_path: Path to input profile.json
            output_path: Path to save schedule.json
        """
        with open(profile_path, "r") as f:
            profile = json.load(f)

        schedule_data = {
            "ops": ProfileExporter.build_schedule_input(profile)["ops"],
            "gpu_idle_events": profile.get("gpu_idle_events", []),
            "tensor_flow": profile.get("tensor_flow", {})
        }

        with open(output_path, "w") as f:
            json.dump(schedule_data, f, indent=2)

        print(f"Exported schedule data to {output_path}")

