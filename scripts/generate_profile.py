#!/usr/bin/env python3
"""
Generate profiling outputs:
- profile.json: Raw profiling data (records, conversion_cost_table, gpu_idle_events, tensor_flow)
- schedule.json: Compiler input (ops, gpu_idle_events, tensor_flow with measured conversion costs)
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
from profiler.exporter import ProfileExporter


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
    # Use iters=2 to detect persistent vs batch-specific tensors
    profiler.profile(dataloader=train_loader, train_step_fn=train_step, warmup=1, iters=2)

    # Export profile.json (raw profiling data with tensor-level conversion costs)
    profile_path = os.path.join(config.TMP_DIR, "profile.json")
    profiler.export_json(profile_path)
    print(f"profile.json exported")

    # Step 2: Export schedule.json (compiler input)
    print("\nStep 2: Exporting schedule...")
    schedule_path = os.path.join(config.TMP_DIR, "schedule.json")
    ProfileExporter.export_schedule_json(profile_path, schedule_path)
    print(f"schedule.json exported")

    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80)
    print(f"\nGenerated files in {config.TMP_DIR}:")
    print(f"\n  profile.json  - Raw profiling data")
    print(f"                  (records, conversion_cost_table, gpu_idle_events, tensor_flow)")
    print(f"\n  schedule.json - Compiler input")
    print(f"                  (ops, gpu_idle_events, tensor_flow with measured conversion costs)")
    print("="*80)


if __name__ == "__main__":
    main()

