#!/bin/bash
# Generate all JSON schedule files needed for training demo

set -e  # Exit on error

echo "========================================================================"
echo "Generating Schedule Files"
echo "========================================================================"
echo ""
echo "This script generates the following files in data/tmp/:"
echo "  - profiling_results.json"
echo "  - schedule_input.json"
echo "  - optimal_schedule.json"
echo "  - test_schedule.json"
echo ""
echo "Expected runtime: ~30 seconds"
echo "========================================================================"
echo ""

python scripts/generate_schedules.py

echo ""
echo "Schedule generation complete!"
echo ""

