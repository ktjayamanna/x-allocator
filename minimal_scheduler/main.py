import torch
import argparse
from torch.utils.data import DataLoader
from .model import MinimalGPT
from .config import Config
from .dataset import MinimalDataset
from .train import train_epoch


def main(use_prefetch=True):
    """
    Run training.

    Args:
        use_prefetch: If True, use prefetch scheduler for speedup
    """
    # Setup
    with open("data/tiny_shakespeare.txt", "r") as f:
        text = f.read()

    dataset = MinimalDataset(text, Config.MAX_SEQ_LEN)

    # Use pin_memory for faster async transfers (essential for prefetching!)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        pin_memory=True if torch.cuda.is_available() else False
    )

    model = MinimalGPT(dataset.vocab_size).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Print config
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Device:       {Config.DEVICE}")
    print(f"Batch size:   {Config.BATCH_SIZE}")
    print(f"Epochs:       {Config.NUM_EPOCHS}")
    print(f"Pin memory:   {loader.pin_memory}")
    print(f"Prefetching:  {'ENABLED' if use_prefetch else 'DISABLED'}")
    print("=" * 60)
    print()

    # Training loop
    for epoch in range(Config.NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{Config.NUM_EPOCHS}")
        avg_loss = train_epoch(model, loader, optimizer, use_prefetch=use_prefetch)
        print(f"Average loss: {avg_loss:.4f}\n")

    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train minimal GPT")
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable prefetch scheduler (use baseline)"
    )
    args = parser.parse_args()

    main(use_prefetch=not args.no_prefetch)