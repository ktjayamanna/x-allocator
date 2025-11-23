"""
Layout-aware module wrapper that applies tensor conversions based on schedule.
"""

from typing import Any, Optional
import torch
import torch.nn as nn

from .layout_actions import LayoutAction


class LayoutAwareModule(nn.Module):
    """
    Wrapper around a PyTorch module that applies layout conversions to inputs
    based on the scheduled action.
    
    This is the core runtime component that enforces the offline schedule's decisions.
    It can optionally use CUDA streams for async conversion when beneficial.
    
    Args:
        inner: The original PyTorch module to wrap
        action: The layout action to apply (from schedule)
        module_name: Name of the module (for debugging/logging)
        use_async_conversion: Whether to use CUDA streams for async conversion
        conversion_stream: Optional CUDA stream for async conversions
    """
    
    def __init__(
        self,
        inner: nn.Module,
        action: LayoutAction,
        module_name: str = "unknown",
        use_async_conversion: bool = False,
        conversion_stream: Optional[torch.cuda.Stream] = None,
    ):
        super().__init__()
        self.inner = inner
        self.action = action
        self.module_name = module_name
        self.use_async_conversion = use_async_conversion
        self.conversion_stream = conversion_stream
        
        # Statistics for monitoring
        self.conversion_count = 0
        self.forward_count = 0
    
    def _convert_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply contiguous conversion to a single tensor."""
        if not isinstance(tensor, torch.Tensor):
            return tensor
        
        if tensor.is_contiguous():
            return tensor
        
        self.conversion_count += 1
        
        # Async conversion using CUDA streams (if enabled and on CUDA)
        if self.use_async_conversion and tensor.is_cuda and self.conversion_stream is not None:
            with torch.cuda.stream(self.conversion_stream):
                converted = tensor.contiguous()
            # Synchronize to ensure conversion is complete before use
            torch.cuda.current_stream().wait_stream(self.conversion_stream)
            return converted
        else:
            # Synchronous conversion
            return tensor.contiguous()
    
    def _convert_inputs(self, *args, **kwargs):
        """Apply conversion to all tensor inputs."""
        # Convert positional args
        converted_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                converted_args.append(self._convert_tensor(arg))
            elif isinstance(arg, (list, tuple)):
                # Handle nested structures
                converted_args.append(type(arg)(
                    self._convert_tensor(x) if isinstance(x, torch.Tensor) else x
                    for x in arg
                ))
            else:
                converted_args.append(arg)
        
        # Convert keyword args
        converted_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                converted_kwargs[key] = self._convert_tensor(value)
            elif isinstance(value, (list, tuple)):
                converted_kwargs[key] = type(value)(
                    self._convert_tensor(x) if isinstance(x, torch.Tensor) else x
                    for x in value
                )
            else:
                converted_kwargs[key] = value
        
        return tuple(converted_args), converted_kwargs
    
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass with optional layout conversion based on action.
        """
        self.forward_count += 1
        
        if self.action == LayoutAction.CONVERT_TO_CONTIGUOUS:
            # Apply conversion to all tensor inputs
            args, kwargs = self._convert_inputs(*args, **kwargs)
        elif self.action in (LayoutAction.NO_ACTION, LayoutAction.SKIP_CONVERSION, 
                             LayoutAction.ASSUME_CONTIGUOUS):
            # Pass through without conversion
            pass
        else:
            # Unknown action - log warning but proceed
            if self.forward_count == 1:  # Only warn once
                print(f"Warning: Unknown action {self.action} for {self.module_name}")
        
        # Call the wrapped module
        return self.inner(*args, **kwargs)
    
    def get_stats(self) -> dict:
        """Get conversion statistics for monitoring."""
        return {
            "module_name": self.module_name,
            "action": str(self.action),
            "forward_count": self.forward_count,
            "conversion_count": self.conversion_count,
            "conversion_rate": self.conversion_count / max(1, self.forward_count),
        }
    
    def __repr__(self) -> str:
        return (f"LayoutAwareModule(module={self.module_name}, "
                f"action={self.action}, inner={self.inner.__class__.__name__})")

