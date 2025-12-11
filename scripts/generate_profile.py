#!/usr/bin/env python3
"""
Generate profiling and cost model outputs:
- profile.json: Raw profiling data (records, conversion_cost_table, gpu_idle_events)
- cost.json: Cost model data for debugging (conversion_cost_table, cost_model coefficients)
- schedule.json: Compiler input (ops, gpu_idle_events with op references)
"""
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from utils import load_and_prepare_data, get_model
from profiler import ContiguityProfiler
from cost_model import ConversionCostModel


def main():
    os.makedirs(config.TMP_DIR, exist_ok=True)

    train_dataset, _ = load_and_prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = get_model(vocab_size=train_dataset.vocab_size)
    model.train()

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    def train_step(x, y):
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Step 1: Profile the model
    print("Step 1: Profiling model...")
    profiler = ContiguityProfiler(model, device=config.DEVICE)
    profiler.profile(dataloader=train_loader, train_step_fn=train_step, warmup=1, iters=1)

    # Export profile.json (raw profiling data)
    profile_path = os.path.join(config.TMP_DIR, "profile.json")
    profiler.export_json(profile_path)
    print(f"profile.json exported")

    # Step 2: Train cost model
    print("\nStep 2: Training cost model...")
    cost_model = ConversionCostModel.from_profile_json(profile_path)

    # Export cost.json (cost model data for debugging)
    cost_path = os.path.join(config.TMP_DIR, "cost.json")
    cost_model.export_cost_json(profile_path, cost_path)
    print(f"cost.json exported")

    # Export schedule.json (compiler input)
    schedule_path = os.path.join(config.TMP_DIR, "schedule.json")
    cost_model.export_schedule_json(profile_path, schedule_path)
    print(f"schedule.json exported")

    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80)
    print(f"\nGenerated files in {config.TMP_DIR}:")
    print(f"\n  profile.json  - Raw profiling data")
    print(f"                  (records, conversion_cost_table, gpu_idle_events)")
    print(f"\n  cost.json     - Cost model for debugging")
    print(f"                  (conversion_cost_table, cost_model coefficients)")
    print(f"\n  schedule.json - Compiler input")
    print(f"                  (ops, gpu_idle_events with op references)")
    print("="*80)


if __name__ == "__main__":
    main()

