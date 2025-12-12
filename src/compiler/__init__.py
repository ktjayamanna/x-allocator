"""
X-Allocator Compiler
Transforms PyTorch code by inserting .contiguous() calls at optimal locations
based on profiling data from schedule.json.
"""

from .core import compile_project

__all__ = ["compile_project"]

