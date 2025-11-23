#!/bin/bash
# Demo script to show the isolated impact of contiguity optimization
# This runs training WITH and WITHOUT the Model Deployer (no prefetcher in either case)

set -e  # Exit on error

echo "========================================================================"
echo "xAllocator Demo: Contiguity Optimization Impact"
echo "========================================================================"
echo ""
echo "This demo compares training performance:"
echo "  - WITH Model Deployer (contiguity optimization)"
echo "  - WITHOUT Model Deployer (baseline)"
echo ""
echo "Both runs use the prefetcher to show that contiguity optimizations"
echo "remain impactful even when data loading is fast."
echo ""
echo "Expected runtime: ~7 minutes total"
echo ""
echo "Note: Run scripts/generate_schedules.sh first if you haven't already"
echo "      or if you want to regenerate the schedule files."
echo "========================================================================"
echo ""

# Check if schedule files exist
if [ ! -f "data/tmp/test_schedule.json" ]; then
    echo "Error: Schedule files not found in data/tmp/"
    echo "Please run: bash scripts/generate_schedules.sh"
    exit 1
fi

# Create temporary files for outputs
DEPLOYER_OUTPUT=$(mktemp)
BASELINE_OUTPUT=$(mktemp)

# Cleanup on exit
trap "rm -f $DEPLOYER_OUTPUT $BASELINE_OUTPUT" EXIT

echo "Step 1: Training WITH Model Deployer (contiguity optimization)"
echo "----------------------------------------------------------------"
echo ""

# Run training with progress monitoring and prefetcher
python scripts/demo_train.py --schedule data/tmp/test_schedule.json --output $DEPLOYER_OUTPUT --label "With Deployer" --prefetch

echo ""
echo " Training with Model Deployer complete!"
echo ""

echo "Step 2: Training WITHOUT Model Deployer (baseline)"
echo "----------------------------------------------------------------"
echo ""

# Run training with progress monitoring, prefetcher, and comparison
python scripts/demo_train.py --output $BASELINE_OUTPUT --label "Baseline" --prefetch --compare $DEPLOYER_OUTPUT

echo ""
echo "Demo complete!"
echo ""

