"""
Layout action enum for controlling tensor memory layout conversions.
"""

from enum import Enum


class LayoutAction(Enum):
    """
    Actions the deployer may take for each module based on the schedule.
    
    These actions control whether and how tensor layout conversions are applied
    to module inputs during forward passes.
    """
    
    NO_ACTION = "NO_ACTION"
    """Do not insert any conversion. Use tensors as-is."""
    
    CONVERT_TO_CONTIGUOUS = "CONVERT_TO_CONTIGUOUS"
    """Insert x.contiguous() before the module's forward pass."""
    
    SKIP_CONVERSION = "SKIP_CONVERSION"
    """Explicitly avoid conversion even if non-contiguous (module can handle it)."""
    
    ASSUME_CONTIGUOUS = "ASSUME_CONTIGUOUS"
    """Scheduler predicts layout is already contiguous, no action needed."""
    
    UNKNOWN = "UNKNOWN"
    """Placeholder for debugging or uninitialized state."""
    
    @classmethod
    def from_string(cls, action_str: str) -> "LayoutAction":
        """Convert string to LayoutAction enum."""
        try:
            return cls(action_str)
        except ValueError:
            return cls.UNKNOWN
    
    def __str__(self) -> str:
        return self.value

