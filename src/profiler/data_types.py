from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TensorLayoutInfo:
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    numel: int
    ndim: int
    dtype: str
    device: str
    is_contiguous: bool


@dataclass
class OpProfileRecord:
    module_name: str
    module_type: str
    phase: str
    has_noncontig_input: bool
    has_noncontig_output: bool
    input_layouts: List[TensorLayoutInfo]
    output_layouts: List[TensorLayoutInfo]
    forward_time_ms: float
    estimated_conversion_cost_ms: Optional[float]
    raw_conversion_samples_ms: List[float]
    extra: Dict[str, Any]

