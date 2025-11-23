import json
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from .data_types import OpProfileRecord, TensorLayoutInfo


class ProfileExporter:
    """Handles exporting profiling data to various formats."""

    @staticmethod
    def export_json(
        records: List[OpProfileRecord],
        conversion_cost_table: Dict[Tuple[int, ...], List[float]],
        path: str,
    ):
        """Export profiling data to JSON file."""
        data = {
            "records": [ProfileExporter._record_to_dict(r) for r in records],
            "conversion_cost_table": {
                str(shape): samples for shape, samples in conversion_cost_table.items()
            },
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
            print(
                f"- {rec.module_name} ({rec.module_type}): "
                f"time={rec.forward_time_ms:.3f} ms, "
                f"noncontig_in={rec.has_noncontig_input}, "
                f"noncontig_out={rec.has_noncontig_output}, "
                f"est_conv_cost={rec.estimated_conversion_cost_ms:.3f} ms "
                if rec.estimated_conversion_cost_ms is not None
                else ""
            )

        print("\n>> Conversion cost model (shape -> avg contiguous() cost, ms):")
        for shape, samples in conversion_cost_table.items():
            avg = sum(samples) / len(samples)
            print(f"  shape={shape}, avg_cost={avg:.4f} ms, samples={len(samples)}")

        print("\n=== End of Summary ===\n")

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
            "forward_time_ms": r.forward_time_ms,
            "estimated_conversion_cost_ms": r.estimated_conversion_cost_ms,
            "raw_conversion_samples_ms": r.raw_conversion_samples_ms,
            "extra": r.extra,
        }

