#!/usr/bin/env python3
"""
CLI script to run the x-allocator compiler.

Usage:
    python scripts/compile.py --schedule data/tmp/schedule.json --src src --output data/tmp/build
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from compiler import compile_project


def main():
    parser = argparse.ArgumentParser(
        description="X-Allocator Compiler: Insert .contiguous() calls at optimal locations"
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="data/tmp/schedule.json",
        help="Path to schedule.json from profiler"
    )
    parser.add_argument(
        "--src",
        type=str,
        default="src",
        help="Source directory containing model.py, config.py, etc."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/tmp/build",
        help="Output directory for optimized code"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output"
    )
    
    args = parser.parse_args()
    
    print(f"X-Allocator Compiler")
    print(f"====================")
    print(f"Schedule: {args.schedule}")
    print(f"Source:   {args.src}")
    print(f"Output:   {args.output}")
    print()
    
    # Run compiler
    insertions = compile_project(
        src_dir=args.src,
        schedule_path=args.schedule,
        output_dir=args.output
    )
    
    print(f"Compilation complete!")
    print(f"Total insertions: {len(insertions)}")
    
    if args.verbose and insertions:
        print("\nInsertions made:")
        for ins in insertions:
            print(f"  - {ins.file_path}:{ins.line_number} ({ins.insertion_type})")
    
    print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()

