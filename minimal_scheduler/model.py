import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Query, Key, Value projections for all heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim (num of features)

        # Calculate Q, K, V for all heads in batch
        qkv = self.qkv(x) # map the last dimension C to 3 * C
        q, k, v = qkv.split(self.n_embd, dim=2) # split the last dimension 3 * C to 3 different tensors.

        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # query you are asking every other token
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # key is the answer to the query in every other token
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2) # value is the amount of attention to a toekn w.r.t to the similarity between the query and the key.

        # Scaled dot-product attention
        qk_similarity = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, n_head, T, T) A.K.A (B, n_head, Q position, K position)

        # Apply causal mask (prevent attending to future tokens)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        qk_similarity = qk_similarity.masked_fill(mask == 0, float('-inf'))
        qk_similarity = F.softmax(qk_similarity, dim=-1)

        # Apply attention to values using matrix multiplication; not dot product.
        y = qk_similarity @ v  # (B, n_head, T, head_dim)
        # move the n_head dimension back to the second dimension and then merge using view
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Merged heads back by n_heads x head_dim = C ->(B, T, C)

        # Output projection
        return self.proj(y) # (B, T, C)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x)) # residual connection from input to the output of the multi-head attention
        x = x + self.ffn(self.ln2(x)) # residual connection from input to the output of the feedforward network
        return x


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        n_embd = Config.N_EMBD
        n_layer = Config.N_LAYER
        n_head = Config.N_HEAD
        max_seq_len = Config.MAX_SEQ_LEN

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        
        tok_emb = self.token_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.head(x) # Final Output Shape: (B, T, vocab_size) 
        


