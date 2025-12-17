import os
import requests
import config
import torch.nn as nn


def download_tiny_shakespeare():
    filepath = os.path.join(config.DATA_DIR, "tiny_shakespeare.txt")

    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    response = requests.get(config.DATASET_URL)
    response.raise_for_status()
    text = response.text

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

    return text


def get_model(vocab_size=None):
    from model import MinimalGPT

    if vocab_size is None:
        vocab_size = config.VOCAB_SIZE

    model = MinimalGPT(vocab_size=vocab_size)
    model.to(config.DEVICE)

    return model


def check_tensor_contiguity(model):
    non_contiguous = []
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            non_contiguous.append(name)
    return non_contiguous


def load_and_prepare_data():
    from dataset import MinimalDataset

    text = download_tiny_shakespeare()

    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    train_dataset = MinimalDataset(train_text, config.MAX_SEQ_LEN)
    eval_dataset = MinimalDataset(val_text, config.MAX_SEQ_LEN)

    return train_dataset, eval_dataset

class Mark(nn.Module):
    """Marker module - makes tensor visible to profiler hooks."""
    def forward(self, x):
        return x

