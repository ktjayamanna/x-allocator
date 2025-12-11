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
    tensor_id: int  # Python id() of the tensor object


@dataclass
class OpProfileRecord:
    module_name: str
    module_type: str
    phase: str
    has_noncontig_input: bool
    has_noncontig_output: bool
    input_layouts: List[TensorLayoutInfo]
    output_layouts: List[TensorLayoutInfo]
    input_tensor_ids: List[int]  # IDs of input tensors
    output_tensor_ids: List[int]  # IDs of output tensors
    forward_time_ms: float
    estimated_conversion_cost_ms: Optional[float]
    raw_conversion_samples_ms: List[float]
    extra: Dict[str, Any]


@dataclass
class IdleEventRecord:
    """Record for GPU idle events (e.g., data transfer from CPU to GPU)."""
    event_name: str
    event_type: str  # "data_transfer", "cpu_preprocessing", etc.
    duration_ms: float
    tensor_shapes: List[Tuple[int, ...]]
    tensor_dtypes: List[str]
    before_op_id: Optional[int]  # op_id that executed before this idle event (None if first)
    after_op_id: Optional[int]  # op_id that will execute after this idle event (None if last)
    extra: Dict[str, Any]


@dataclass
class TensorFlowNode:
    """Represents a tensor in the data flow graph."""
    tensor_id: int
    shape: Tuple[int, ...]
    is_contiguous: bool
    produced_by: Optional[int]  # op_id that produced this tensor (None if input)
    consumed_by: List[int]  # op_ids that consumed this tensor
    estimated_conv_cost_ms: Optional[float]  # Cost to convert to contiguous

