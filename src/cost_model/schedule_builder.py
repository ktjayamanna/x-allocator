import math
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ConversionCostModel


class ScheduleInputBuilder:
    """Builds schedule_input.json from profiler data and cost model."""

    @staticmethod
    def build(
        profile: Dict[str, Any],
        cost_model: "ConversionCostModel"
    ) -> Dict[str, Any]:
        """Build scheduler-ready input from profiler data."""
        ops = []
        op_id = 0

        for rec in profile.get("records", []):
            fwd = float(rec.get("forward_time_ms", 0.0))
            has_nc_in = bool(rec.get("has_noncontig_input", False))
            has_nc_out = bool(rec.get("has_noncontig_output", False))

            layouts = rec.get("input_layouts", []) + rec.get("output_layouts", [])
            if layouts:
                main_layout = max(
                    layouts,
                    key=lambda l: int(math.prod(l["shape"])) if l.get("shape") else 0
                )
                shape = tuple(main_layout["shape"])
            else:
                shape = ()

            est_cost = cost_model.estimate(shape) if shape else None

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
                "estimated_conv_cost_ms": est_cost
            }

            # Add call site info if available
            if call_site_file is not None:
                op_dict["call_site_file"] = call_site_file
            if call_site_line is not None:
                op_dict["call_site_line"] = call_site_line

            ops.append(op_dict)

            op_id += 1

        return {"ops": ops}

