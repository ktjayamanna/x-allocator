import torch
from torch.utils.data import Dataset, DataLoader

class MinimalDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len
        self.chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.text[idx:idx + self.seq_len + 1]
        indices = [self.char_to_idx[ch] for ch in chunk]
        x = torch.tensor(indices[:-1], dtype=torch.long)
        y = torch.tensor(indices[1:], dtype=torch.long)
        return x, y


def get_dataloader(file_path, seq_len, batch_size):
    with open(file_path, 'r') as f:
        text = f.read()
    dataset = MinimalDataset(text, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset
