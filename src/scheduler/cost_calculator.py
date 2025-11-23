from typing import List

from .data_types import ScheduledOp


class CostCalculator:
    """Calculates total estimated latency from scheduled operations."""

    @staticmethod
    def calculate_total_latency(scheduled_ops: List[ScheduledOp]) -> float:
        """Calculate total estimated latency in milliseconds."""
        total = 0.0
        
        for op in scheduled_ops:
            forward_ms = op.estimated_forward_ms or 0.0
            conversion_ms = op.estimated_conversion_ms or 0.0
            total += forward_ms + conversion_ms
        
        return total

