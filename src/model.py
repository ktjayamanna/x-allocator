import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config
from utils import Mark


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.mark = Mark()

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = self.mark(q.view(B, T, self.n_head, self.head_dim).transpose(1, 2), "attn_q")  # @noncontig
        k = self.mark(k.view(B, T, self.n_head, self.head_dim).transpose(1, 2), "attn_k")  # @noncontig
        v = self.mark(v.view(B, T, self.n_head, self.head_dim).transpose(1, 2), "attn_v")  # @noncontig

        qk_similarity = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        qk_similarity = self.mark(qk_similarity, "attn_qk_similarity")  # trap for the compiler
        mask = torch.tril(torch.ones(T, T, device=x.device))
        qk_similarity = qk_similarity.masked_fill(mask == 0, float('-inf'))
        qk_similarity = F.softmax(qk_similarity, dim=-1)
        y = qk_similarity @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # @noncontigHandledByUser
        return self.proj(y)


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
        self.mark = Mark()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        # my_weird_tensor = self.mark(torch.randn(1, 2, 3).transpose(1, 2)) # @noncontig
        # more_weird_stuff = self.mark(5 * my_weird_tensor)
        return x


class MinimalGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        n_embd = config.N_EMBD
        n_layer = config.N_LAYER
        n_head = config.N_HEAD
        max_seq_len = config.MAX_SEQ_LEN

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.mark = Mark()

    def forward(self, input_ids):
        B, T = input_ids.shape
        tok_emb = self.token_emb(input_ids)

        pos = torch.arange(T, device=input_ids.device)
        pos_emb = self.pos_emb(pos)
        pos_emb = self.mark(pos_emb.unsqueeze(0).expand(B, -1, -1), "pos_emb_expanded")  # @noncontig

        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
        return input_ids



