from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ScheduledOp:
    """Represents a scheduled operation with layout action."""
    op_id: int
    module_name: str
    module_type: str
    selected_layout_action: str
    estimated_forward_ms: Optional[float]
    estimated_conversion_ms: Optional[float]
    main_tensor_shape: Optional[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "module_name": self.module_name,
            "module_type": self.module_type,
            "selected_layout_action": self.selected_layout_action,
            "estimated_forward_ms": self.estimated_forward_ms,
            "estimated_conversion_ms": self.estimated_conversion_ms,
            "main_tensor_shape": self.main_tensor_shape,
        }

