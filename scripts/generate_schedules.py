#!/usr/bin/env python3
"""
Generate all JSON files needed for training demo.

This script:
1. Profiles the model to generate profiling_results.json
2. Builds cost model and generates schedule_input.json
3. Runs scheduler to generate optimal_schedule.json and test_schedule.json

All files are saved to data/tmp/
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from dataset import load_and_prepare_data, get_tokenizer
from model import get_model
from profiler import ContiguityProfiler
from cost_model import ConversionCostModel
from scheduler import Scheduler


def main():
    print("=" * 80)
    print("Generating Schedule JSON Files")
    print("=" * 80)
    print()
    
    # Ensure tmp directory exists
    os.makedirs(config.TMP_DIR, exist_ok=True)
    
    print(f"Using device: {config.DEVICE}")
    print(f"Output directory: {config.TMP_DIR}")
    print()
    
    # Step 1: Profile the model
    print("Step 1: Profiling model...")
    print("-" * 80)
    
    # Load model
    model = get_model(from_pretrained=False)
    model.eval()
    
    # Load a small batch of data for profiling
    tokenizer = get_tokenizer()
    train_dataset, _ = load_and_prepare_data(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    
    # Get one batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(config.DEVICE)
    attention_mask = batch["attention_mask"].to(config.DEVICE)
    labels = batch["labels"].to(config.DEVICE)
    
    # Profile
    profiler = ContiguityProfiler(model, device=config.DEVICE)

    # Run profiling
    profiler.profile(
        example_inputs=(),
        example_kwargs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        },
        warmup=1,
        iters=1
    )
    
    # Export profiling results
    profiling_path = os.path.join(config.TMP_DIR, "profiling_results.json")
    profiler.export_json(profiling_path)
    print(f"Saved profiling results to {profiling_path}")
    print()

    # Step 2: Build cost model and schedule input
    print("Step 2: Building cost model and schedule input...")
    print("-" * 80)

    cost_model = ConversionCostModel.from_profile_json(profiling_path)

    # Load profile data
    with open(profiling_path, "r") as f:
        profile_data = json.load(f)

    # Build schedule input
    schedule_input = cost_model.build_schedule_input(profile_data)

    # Save schedule input
    schedule_input_path = os.path.join(config.TMP_DIR, "schedule_input.json")
    with open(schedule_input_path, "w") as f:
        json.dump(schedule_input, f, indent=2)
    print(f"Saved schedule input to {schedule_input_path}")
    print()

    # Step 3: Generate schedules
    print("Step 3: Generating schedules...")
    print("-" * 80)

    scheduler = Scheduler.from_json(schedule_input_path)

    # Generate optimal schedule
    optimal_schedule_path = os.path.join(config.TMP_DIR, "optimal_schedule.json")
    scheduler.save(optimal_schedule_path)
    print(f"Saved optimal schedule to {optimal_schedule_path}")

    # Generate test schedule (same as optimal for now)
    test_schedule_path = os.path.join(config.TMP_DIR, "test_schedule.json")
    scheduler.save(test_schedule_path)
    print(f"Saved test schedule to {test_schedule_path}")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Generated {len(profiler.records)} profiling records")
    print(f"Cost model fitted: {cost_model.fitted}")
    if cost_model.fitted:
        print(f"  α (numel coef): {cost_model.alpha:.3e}")
        print(f"  β (ndim coef): {cost_model.beta:.3e}")
        print(f"  γ (constant): {cost_model.gamma:.3e}")
    print(f"Schedule contains {len(schedule_input['ops'])} operations")
    print()
    print("All files saved to:", config.TMP_DIR)
    print("  - profiling_results.json")
    print("  - schedule_input.json")
    print("  - optimal_schedule.json")
    print("  - test_schedule.json")
    print()
    print("Done!")


if __name__ == "__main__":
    main()

