import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import time
import config
from utils import load_and_prepare_data, get_model


def train_epoch(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, (x, y) in enumerate(progress_bar):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
        loss.backward()

        if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
        progress_bar.set_postfix({"loss": loss.item() * config.GRADIENT_ACCUMULATION_STEPS})

    return total_loss / len(train_loader)


def evaluate(model, eval_loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in tqdm(eval_loader, desc="Evaluating"):
            x, y = x.to(config.DEVICE), y.to(config.DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()

    return total_loss / len(eval_loader)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=str, default=None)
    parser.add_argument("--async-conversion", action="store_true")
    args = parser.parse_args()

    train_dataset, eval_dataset = load_and_prepare_data()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = get_model(vocab_size=train_dataset.vocab_size)

    if args.schedule and os.path.exists(args.schedule):
        from deployer import apply_layout_schedule
        model = apply_layout_schedule(
            model,
            schedule_path=args.schedule,
            use_async_conversion=args.async_conversion,
            verbose=True
        )

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, epoch)
        eval_loss = evaluate(model, eval_loader)
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch}: Train={train_loss:.4f}, Eval={eval_loss:.4f}, Time={epoch_time:.2f}s")

    model_path = f"{config.DATA_DIR}/model.pt"
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
