from enum import Enum


class LayoutAction(Enum):
    """Actions the scheduler may choose for each operator."""
    
    NO_ACTION = "NO_ACTION"
    CONVERT_TO_CONTIGUOUS = "CONVERT_TO_CONTIGUOUS"
    SKIP_CONVERSION = "SKIP_CONVERSION"
    ASSUME_CONTIGUOUS = "ASSUME_CONTIGUOUS"
    UNKNOWN = "UNKNOWN"

