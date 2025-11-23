import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head

        # Query, Key, Value projections for all heads
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        # Calculate Q, K, V for all heads in batch
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply causal mask (prevent attending to future tokens)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(causal_mask == 0, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            att = att.masked_fill(attention_mask.view(B, 1, 1, T) == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Apply attention to values
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection
        y = self.resid_dropout(self.proj(y))
        return y


class FeedForward(nn.Module):
    """Position-wise feed-forward network"""

    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and feed-forward"""

    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x, attention_mask=None):
        # Pre-norm architecture (more stable for training)
        x = x + self.attn(self.ln1(x), attention_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class CustomGPT(nn.Module):
    """
    Custom GPT-style language model for interpretability research.

    Architecture:
    - Small size (~10-15M parameters vs GPT-2's 124M)
    - Character-level or simple tokenization
    - Transparent attention mechanisms for interpretability
    """

    def __init__(self, vocab_size, n_embd=256, n_layer=6, n_head=4,
                 max_seq_len=128, dropout=0.1):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq_len, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying (share embeddings with output layer)
        self.head.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape

        # Get token and position embeddings
        tok_emb = self.token_emb(input_ids)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)  # (1, T)
        pos_emb = self.pos_emb(pos)  # (1, T, n_embd)

        x = self.drop(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        # Final layer norm and projection to vocabulary
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten for cross-entropy
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        # Return in HuggingFace-compatible format
        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        return Output(loss, logits)

    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """Generate new tokens autoregressively"""
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]

            # Forward pass
            outputs = self(idx_cond)
            logits = outputs.logits

            # Get logits for last token and apply temperature
            logits = logits[:, -1, :] / temperature

            # Optionally crop to top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)

        return input_ids


def get_model(from_pretrained=False):
    """
    Get custom GPT model

    Args:
        from_pretrained: Ignored for custom model (always trains from scratch)
    """
    print(f"Initializing custom GPT model from scratch...")

    model = CustomGPT(
        vocab_size=config.VOCAB_SIZE,
        n_embd=config.N_EMBD,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        max_seq_len=config.MAX_LENGTH,
        dropout=config.DROPOUT
    )

    model.to(config.DEVICE)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024**2:.2f} MB (fp32)")

    return model


def check_tensor_contiguity(model):
    """
    Utility function to check tensor contiguity in model
    Useful for performance bottleneck experiments
    """
    non_contiguous = []
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            non_contiguous.append(name)

    if non_contiguous:
        print(f"Non-contiguous tensors found: {non_contiguous}")
    else:
        print("All model tensors are contiguous")

    return non_contiguous
