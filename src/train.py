"""
Usage:
    # Standard training (no optimization)
    python src/train.py

    # With Model Deployer
    python src/train.py --schedule data/tmp/optimal_schedule.json

    # With CudaPrefetcher
    python src/train.py --schedule data/tmp/test_schedule.json --use-prefetcher

    # Full optimization
    python src/train.py --schedule data/tmp/test_schedule.json --use-prefetcher --async-conversion
"""

import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import time
import config
from dataset import load_and_prepare_data, get_tokenizer
from model import get_model, check_tensor_contiguity


def train_epoch_with_prefetcher(model, prefetcher, optimizer, epoch):
    """Train for one epoch using CudaPrefetcher"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(prefetcher, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Batch is already on device thanks to prefetcher
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        # Gradient accumulation
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({"loss": loss.item() * config.GRADIENT_ACCUMULATION_STEPS})

    return total_loss / len(prefetcher)


def train_epoch_standard(model, train_loader, optimizer, epoch):
    """Train for one epoch (standard approach)"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        labels = batch["labels"].to(config.DEVICE)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        # Gradient accumulation
        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({"loss": loss.item() * config.GRADIENT_ACCUMULATION_STEPS})

    return total_loss / len(train_loader)


def evaluate(model, eval_loader):
    """Evaluate the model"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(eval_loader)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train with optional Model Deployer")
    parser.add_argument("--schedule", type=str, default=None,
                        help="Path to schedule JSON (e.g., data/tmp/optimal_schedule.json)")
    parser.add_argument("--use-prefetcher", action="store_true",
                        help="Use CudaPrefetcher for async data loading")
    parser.add_argument("--async-conversion", action="store_true",
                        help="Use async conversion in LayoutAwareModule")
    args = parser.parse_args()

    print(f"Using device: {config.DEVICE}")
    print(f"Data directory: {config.DATA_DIR}")

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load and prepare data
    train_dataset, eval_dataset = load_and_prepare_data(
        tokenizer,
        max_train_samples=config.MAX_TRAIN_SAMPLES,
        max_eval_samples=config.MAX_EVAL_SAMPLES
    )

    # Create data loaders
    pin_memory = args.use_prefetcher and config.DEVICE == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=pin_memory,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load model
    model = get_model(from_pretrained=False)

    # Apply schedule if provided
    if args.schedule and os.path.exists(args.schedule):
        print(f"\nApplying Model Deployer with schedule: {args.schedule}")

        from deployer import apply_layout_schedule

        model = apply_layout_schedule(
            model,
            schedule_path=args.schedule,
            use_async_conversion=args.async_conversion,
            verbose=True
        )
    else:
        print("\nNo schedule provided - training without Model Deployer")
        if args.schedule:
            print(f"  (Schedule file not found: {args.schedule})")

    # Check tensor contiguity
    print("\nChecking tensor contiguity...")
    check_tensor_contiguity(model)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Setup prefetcher if requested
    if args.use_prefetcher and config.DEVICE == "cuda":
        print("\nUsing CudaPrefetcher for async data loading")
        from deployer import CudaPrefetcher

        prefetcher = CudaPrefetcher(
            train_loader,
            device=config.DEVICE,
            convert_inputs=False,  # Model handles conversions
        )
        train_fn = lambda m, l, o, e: train_epoch_with_prefetcher(m, prefetcher, o, e)
    else:
        train_fn = train_epoch_standard

    # Training loop
    print(f"\nStarting training for {config.NUM_EPOCHS} epoch(s)...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()

        train_loss = train_fn(model, train_loader, optimizer, epoch)
        eval_loss = evaluate(model, eval_loader)

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Eval Loss: {eval_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")

    # Save model
    model_path = f"{config.DATA_DIR}/custom_gpt_trained.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
