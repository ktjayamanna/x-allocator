import torch
from torch.utils.data import Dataset
import config
import os
import requests


class CharTokenizer:
    """Simple character-level tokenizer for interpretability"""

    def __init__(self):
        # Use ASCII characters (0-255)
        self.vocab_size = 256
        self.pad_token_id = 0
        self.eos_token_id = 0

    def encode(self, text):
        """Convert text to list of character codes"""
        return [min(ord(c), 255) for c in text]

    def decode(self, tokens):
        """Convert list of character codes back to text"""
        return ''.join([chr(t) if 0 <= t < 256 else '?' for t in tokens])

    def __call__(self, text):
        """Make tokenizer callable like HuggingFace tokenizers"""
        if isinstance(text, str):
            return {"input_ids": self.encode(text)}
        elif isinstance(text, list):
            return {"input_ids": [self.encode(t) for t in text]}
        else:
            raise ValueError(f"Unsupported input type: {type(text)}")


class CharDataset(Dataset):
    """Character-level dataset for language modeling"""

    def __init__(self, text, max_length):
        self.text = text
        self.max_length = max_length
        self.tokenizer = CharTokenizer()

        # Encode entire text
        self.data = self.tokenizer.encode(text)

        # Calculate number of samples
        self.num_samples = len(self.data) // max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get a chunk of text
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length

        chunk = self.data[start_idx:end_idx]

        # Pad if necessary
        if len(chunk) < self.max_length:
            chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))

        input_ids = torch.tensor(chunk, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids,
        }


def download_tiny_shakespeare():
    """Download TinyShakespeare dataset"""
    filepath = os.path.join(config.DATA_DIR, "tiny_shakespeare.txt")

    if os.path.exists(filepath):
        print(f"TinyShakespeare already downloaded at {filepath}")
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    print(f"Downloading TinyShakespeare from {config.DATASET_URL}...")
    response = requests.get(config.DATASET_URL)
    response.raise_for_status()

    text = response.text

    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Downloaded {len(text)} characters")
    return text


def load_and_prepare_data(tokenizer=None, max_train_samples=None, max_eval_samples=None):
    """
    Load TinyShakespeare dataset and prepare for training

    Args:
        tokenizer: Ignored (we use character-level tokenization)
        max_train_samples: Ignored (we use the full dataset)
        max_eval_samples: Ignored (we use the full dataset)
    """
    print(f"Loading TinyShakespeare dataset to {config.DATA_DIR}...")

    # Download dataset
    text = download_tiny_shakespeare()

    # Split into train/validation (90/10 split)
    split_idx = int(len(text) * 0.9)
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"Train characters: {len(train_text):,}")
    print(f"Validation characters: {len(val_text):,}")

    # Create datasets
    train_dataset = CharDataset(train_text, config.MAX_LENGTH)
    eval_dataset = CharDataset(val_text, config.MAX_LENGTH)

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(eval_dataset):,}")

    return train_dataset, eval_dataset


def get_tokenizer():
    """Get character-level tokenizer"""
    return CharTokenizer()
