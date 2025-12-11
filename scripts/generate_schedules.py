#!/usr/bin/env python3
import os
import sys
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from utils import load_and_prepare_data, get_model
from profiler import ContiguityProfiler
from cost_model import ConversionCostModel
from scheduler import Scheduler


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

    profiler = ContiguityProfiler(model, device=config.DEVICE)
    profiler.profile(dataloader=train_loader, train_step_fn=train_step, warmup=1, iters=1)

    profiling_path = os.path.join(config.TMP_DIR, "profiling_results.json")
    profiler.export_json(profiling_path)

    cost_model = ConversionCostModel.from_profile_json(profiling_path)

    with open(profiling_path, "r") as f:
        profile_data = json.load(f)

    schedule_input = cost_model.build_schedule_input(profile_data)

    schedule_input_path = os.path.join(config.TMP_DIR, "schedule_input.json")
    with open(schedule_input_path, "w") as f:
        json.dump(schedule_input, f, indent=2)

    scheduler = Scheduler.from_json(schedule_input_path)

    optimal_schedule_path = os.path.join(config.TMP_DIR, "optimal_schedule.json")
    scheduler.save(optimal_schedule_path)

    test_schedule_path = os.path.join(config.TMP_DIR, "test_schedule.json")
    scheduler.save(test_schedule_path)


if __name__ == "__main__":
    main()

