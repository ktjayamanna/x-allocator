"""
Core compiler logic for x-allocator.

Phase 1: Read schedule.json and build IR (list of insertions to make)
Phase 2: Apply insertions in reverse order to preserve line numbers
"""

import json
import os
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class ContiguousInsertion:
    """Represents a .contiguous() call to insert."""
    file_path: str
    line_number: int
    insertion_type: str  # 'sync', 'async_at_idle', 'at_data_transfer'
    op_id: int
    module_name: str
    tensor_id: Optional[int] = None


def load_schedule(schedule_path: str) -> Dict[str, Any]:
    """Load schedule.json."""
    with open(schedule_path, 'r') as f:
        return json.load(f)


def find_gpu_idle_between_ops(
    schedule: Dict[str, Any],
    after_op_id: int,
    before_op_id: int
) -> Optional[Dict[str, Any]]:
    """Check if there's a GPU idle event between two ops."""
    for event in schedule.get("gpu_idle_events", []):
        event_after = event.get("after_op_id")
        event_before = event.get("before_op_id")
        # Idle event is between after_op_id and before_op_id
        if event_after is not None and event_before is not None:
            if event_after >= after_op_id and event_before <= before_op_id:
                return event
    return None


def build_ir(schedule: Dict[str, Any]) -> List[ContiguousInsertion]:
    """
    Phase 1: Analyze schedule and build IR of insertions needed.
    
    Decision logic (from compiler.excalidraw):
    1. Is tensor contiguous? -> Yes: skip
    2. No: Does it show up again (has consumers)?
       - No: Is it persistent? -> Yes: convert at data transfer
       - Yes: Is there GPU idle before next occurrence?
         -> Yes: convert async at that window
         -> No: convert synchronously
    """
    insertions = []
    ops = schedule.get("ops", [])
    tensor_flow = schedule.get("tensor_flow", {})
    
    for op in ops:
        # Check if input is non-contiguous
        if not op.get("has_noncontig_input", False):
            continue  # Already contiguous, skip
        
        # Get input tensor info
        input_tensor_ids = op.get("input_tensor_ids", [])
        if not input_tensor_ids:
            continue
        
        for tensor_id in input_tensor_ids:
            tensor_info = tensor_flow.get(str(tensor_id), {})
            
            # Check if tensor shows up again (has consumers)
            consumers = tensor_info.get("consumed_by", [])
            
            if not consumers:
                # No consumers - check if persistent
                lifetime = tensor_info.get("lifetime", "batch_specific")
                if lifetime == "persistent":
                    # Convert at data transfer
                    insertions.append(ContiguousInsertion(
                        file_path=op.get("call_site_file", ""),
                        line_number=op.get("call_site_line", 0),
                        insertion_type="at_data_transfer",
                        op_id=op.get("op_id", -1),
                        module_name=op.get("module_name", ""),
                        tensor_id=tensor_id
                    ))
                # else: not persistent, skip
            else:
                # Has consumers - check for GPU idle before next occurrence
                next_consumer_op_id = min(consumers)
                idle_event = find_gpu_idle_between_ops(
                    schedule,
                    op.get("op_id", -1),
                    next_consumer_op_id
                )
                
                if idle_event:
                    # Convert async at that window
                    insertions.append(ContiguousInsertion(
                        file_path=op.get("call_site_file", ""),
                        line_number=op.get("call_site_line", 0),
                        insertion_type="async_at_idle",
                        op_id=op.get("op_id", -1),
                        module_name=op.get("module_name", ""),
                        tensor_id=tensor_id
                    ))
                else:
                    # Convert synchronously
                    insertions.append(ContiguousInsertion(
                        file_path=op.get("call_site_file", ""),
                        line_number=op.get("call_site_line", 0),
                        insertion_type="sync",
                        op_id=op.get("op_id", -1),
                        module_name=op.get("module_name", ""),
                        tensor_id=tensor_id
                    ))
    
    return insertions


def apply_insertions(source_path: str, insertions: List[ContiguousInsertion]) -> str:
    """
    Phase 2: Apply .contiguous() insertions to source code.
    Uses reverse order to preserve line numbers.
    """
    with open(source_path, 'r') as f:
        lines = f.readlines()
    
    # Filter insertions for this file, deduplicate by line number, sort descending
    file_insertions = [
        ins for ins in insertions
        if ins.file_path.endswith(os.path.basename(source_path))
    ]
    # Deduplicate by line number (keep first occurrence)
    seen_lines = set()
    unique_insertions = []
    for ins in file_insertions:
        if ins.line_number not in seen_lines:
            seen_lines.add(ins.line_number)
            unique_insertions.append(ins)
    unique_insertions.sort(key=lambda x: x.line_number, reverse=True)

    for insertion in unique_insertions:
        line_idx = insertion.line_number - 1  # Convert to 0-indexed
        if 0 <= line_idx < len(lines):
            line = lines[line_idx]
            # Insert .contiguous() - simple approach for now
            # Add a comment indicating the insertion type
            comment = f"  # x-allocator: {insertion.insertion_type}"

            # Strip existing comments to find the code part
            code_part = line.split('#')[0].rstrip()
            existing_comment = '#' + '#'.join(line.split('#')[1:]) if '#' in line else ''
            existing_comment = existing_comment.rstrip()

            if code_part.endswith(')'):
                # Insert .contiguous() before the closing paren
                new_code = code_part[:-1] + ".contiguous())"
                lines[line_idx] = new_code + existing_comment + comment + "\n"
            else:
                # Fallback: just add comment
                lines[line_idx] = line.rstrip() + comment + "\n"
    
    return ''.join(lines)


def compile_project(
    src_dir: str,
    schedule_path: str,
    output_dir: str
) -> List[ContiguousInsertion]:
    """
    Main entry point: compile a project with contiguity optimizations.

    Args:
        src_dir: Source directory with config.py, model.py, etc.
        schedule_path: Path to schedule.json
        output_dir: Output directory for optimized code (default: data/build)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load schedule and build IR
    schedule = load_schedule(schedule_path)
    insertions = build_ir(schedule)
    
    # Export IR for debugging
    ir_path = os.path.join("data", "tmp", "ir.json")
    os.makedirs(os.path.dirname(ir_path), exist_ok=True)
    export_ir_json(insertions, ir_path)
    
    # Files to process
    files = ["config.py", "dataset.py", "model.py", "train.py", "utils.py"]
    
    for filename in files:
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(src_path):
            continue
        
        # Check if this file has any insertions
        file_insertions = [
            ins for ins in insertions
            if ins.file_path.endswith(filename)
        ]
        
        if file_insertions:
            # Apply transformations
            transformed = apply_insertions(src_path, insertions)
            with open(dst_path, 'w') as f:
                f.write(transformed)
        else:
            # Copy unchanged
            shutil.copy2(src_path, dst_path)
    
    return insertions


def export_ir_json(insertions: List[ContiguousInsertion], output_path: str):
    """Export IR (insertions) to JSON for debugging purposes."""
    ir_data = {
        "insertions": [
            {
                "file_path": ins.file_path,
                "line_number": ins.line_number,
                "insertion_type": ins.insertion_type,
                "op_id": ins.op_id,
                "module_name": ins.module_name,
                "tensor_id": ins.tensor_id
            }
            for ins in insertions
        ],
        "total_insertions": len(insertions),
        "insertions_by_type": {
            "sync": len([ins for ins in insertions if ins.insertion_type == "sync"]),
            "async_at_idle": len([ins for ins in insertions if ins.insertion_type == "async_at_idle"]),
            "at_data_transfer": len([ins for ins in insertions if ins.insertion_type == "at_data_transfer"])
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(ir_data, f, indent=2)
    
    print(f"Exported IR to {output_path}")

