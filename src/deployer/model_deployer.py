"""
Model deployer - applies offline schedule to PyTorch models.
"""

import json
from typing import Dict, Optional, Any
import torch
import torch.nn as nn

from .layout_actions import LayoutAction
from .layout_module import LayoutAwareModule


def load_schedule(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load schedule from JSON file and convert to module_name -> entry mapping.

    Args:
        path: Path to schedule JSON file (e.g., "data/tmp/optimal_schedule.json")

    Returns:
        Dictionary mapping module_name to schedule entry
    """
    with open(path, "r") as f:
        schedule_data = json.load(f)
    
    # Convert list of ops to dict keyed by module_name
    schedule_dict = {}
    for entry in schedule_data.get("schedule", []):
        module_name = entry.get("module_name")
        if module_name:
            schedule_dict[module_name] = entry
    
    return schedule_dict


def apply_layout_schedule(
    model: nn.Module,
    schedule_path: str,
    use_async_conversion: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Apply layout schedule to a PyTorch model by wrapping targeted modules.

    Walks through the model and wraps modules that appear in the schedule
    with LayoutAwareModule to apply the scheduled layout actions.

    Args:
        model: PyTorch model to apply schedule to
        schedule_path: Path to schedule JSON file
        use_async_conversion: Whether to use CUDA streams for async conversion
        verbose: Whether to print deployment information

    Returns:
        Modified model with wrapped modules
    """
    # Load schedule
    schedule = load_schedule(schedule_path)

    if verbose:
        print(f"Loaded schedule with {len(schedule)} entries")

    # Create CUDA stream for async conversions if needed
    conversion_stream = None
    if use_async_conversion and torch.cuda.is_available():
        conversion_stream = torch.cuda.Stream()
        if verbose:
            print(f"Created CUDA stream for async conversions")
    
    # Track wrapped modules
    wrapped_count = 0
    action_counts = {action: 0 for action in LayoutAction}
    
    # Walk through all named modules and wrap those in schedule
    for name, module in list(model.named_modules()):
        if name in schedule:
            entry = schedule[name]
            action_str = entry.get("selected_layout_action", "NO_ACTION")
            action = LayoutAction.from_string(action_str)
            
            # Only wrap if action requires it
            if action == LayoutAction.CONVERT_TO_CONTIGUOUS:
                # Get parent module and attribute name
                parent, attr = _get_parent_and_attr(model, name)
                
                if parent is not None and attr is not None:
                    # Wrap the module
                    wrapped = LayoutAwareModule(
                        inner=module,
                        action=action,
                        module_name=name,
                        use_async_conversion=use_async_conversion,
                        conversion_stream=conversion_stream,
                    )
                    
                    # Replace in parent
                    setattr(parent, attr, wrapped)
                    wrapped_count += 1
                    action_counts[action] += 1
                    
                    if verbose:
                        print(f"  Wrapped: {name} (action={action})")
            else:
                # Track but don't wrap
                action_counts[action] += 1
    
    if verbose:
        print(f"\nDeployment complete: {wrapped_count} modules wrapped")
        for action, count in action_counts.items():
            if count > 0:
                print(f"  {action.value}: {count}")
    
    return model


def _get_parent_and_attr(model: nn.Module, module_name: str):
    """
    Get parent module and attribute name for a given module path.
    
    Args:
        model: Root model
        module_name: Dot-separated module path (e.g., "blocks.0.attn.qkv")
    
    Returns:
        Tuple of (parent_module, attribute_name) or (None, None) if not found
    """
    if not module_name:
        return None, None
    
    parts = module_name.split(".")
    
    # Handle root module
    if len(parts) == 1:
        return model, parts[0]
    
    # Navigate to parent
    parent = model
    for part in parts[:-1]:
        if hasattr(parent, part):
            parent = getattr(parent, part)
        else:
            return None, None
    
    attr_name = parts[-1]
    
    # Verify the attribute exists
    if hasattr(parent, attr_name):
        return parent, attr_name
    else:
        return None, None

