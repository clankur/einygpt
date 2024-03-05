import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional
from common import GptConfig, KVCacheType, BlocksKVCacheType

torch.manual_seed(1337)


class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""

    def __init__(self, hyperparameters: GptConfig, num_heads: int, head_size: int) -> None:
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)

        self.num_heads = num_heads
        self.head_size = head_size

        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.proj = nn.Linear(self.n_embd, self.n_embd)
        self.dropout = nn.Dropout(self.dropout)

        self.register_buffer('tril', torch.tril(
            torch.ones(1, 1, self.block_size, self.block_size)))

    def forward(self, x: torch.Tensor, use_cache: bool, kvcache: Optional[KVCacheType]) -> torch.Tensor:
        """
        performs a forward pass of the model

        Parameters:
        - x: a [B, T, C] tensor of floats representing the input sequence

        Returns:
        - out: a [B, T, C] tensor of floats representing the output sequence
        """
        B, T, C = x.shape
        k, v, q = self.key(x), self.value(x), self.query(x)  # [B, T, C]

        # C = n * h where n is the number of heads, h is the head dimension, C is the model dimension
        k, v, q = [t.reshape(B, T, self.num_heads, self.head_size)
                   for t in (k, v, q)]  # [B, T, C] -> [B, T, n, h]
        k, v, q = [torch.transpose(x, 1, 2) for x in (
            k, v, q)]  # [B, T, n, h] -> [B, n, T, h]
        if use_cache:
            if kvcache:
                prev_k, prev_v = kvcache
                prev_k, prev_v = prev_k[:, :, -self.block_size -
                                        1:, :], prev_v[:, :, -self.block_size-1:, :]
                # [B, n, K, h] -> [B, n, K+T, h]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            kvcache = (k, v)

        # [B, n, Q, h] @ [B, n, K, h] -> [B, n, Q, K]
        att_wei = torch.einsum('bnqh,bnkh->bnqk', q, k) * \
            (self.head_size**-0.5)
        # casual masking
        att_wei = att_wei.masked_fill(
            self.tril[:, :, :T, :T] == 0, float('-inf'))
        # don't really get the dimensions defined in self.tril

        att_wei = F.softmax(att_wei, dim=-1)
        att_wei = self.dropout(att_wei)
        # [B, n, Q, K] @ [B, n, K, h] -> [B, n, Q, h]
        out = torch.einsum('bnqk,bnkh->bnqh', att_wei, v)

        out = torch.transpose(out, 1, 2)  # [B, n, Q, h] -> [B, Q, n, h]
        out = out.reshape(B, T, C)  # [B, T, n, h] -> [B, T, C]
        # what is the purpose of this? allow heads to communicate
        out = self.proj(out)
        out = self.dropout(out)  # apply dropout
        return out, kvcache


class FeedForward(nn.Module):
    """ simple linear layer followed by a non-linearity and another linear layer"""

    def __init__(self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(self.dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass of the model

        Parameters:
        - x: a [B, T, C] tensor of floats representing the input sequence

        Returns:
        - a [B, T, C] tensor of floats representing the output sequence
        """
        return self.net(x)


class Block (nn.Module):
    """ a transformer block: intersperses communication with computation"""

    def __init__(self, n_embd: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(
            num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor, use_cache: bool, kvcache: Optional[BlocksKVCacheType]) -> torch.Tensor:
        """
        performs a forward pass of the model

        Parameters:
        - x: a [B, T, C] tensor of floats representing the input sequence

        Returns:
        - out: a [B, T, C] tensor of floats representing the output sequence
        """
        # we also perform layer normalization before being fed into the heads and ffwd
        heads_out, kvcache = self.sa_heads(
            self.ln1(x), use_cache=use_cache, kvcache=kvcache)
        x = x + heads_out  # residual connection adding to sa heads
        x = x + self.ffwd(self.ln2(x))  # residual connection adding to ffwd
        return x, kvcache


class NanoGPTLanguageModel(nn.Module):
    def __init__(self, hyperparameters: GptConfig) -> None:
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(
            self.block_size, self.n_embd)
        self.blocks = nn.ModuleList(
            [Block(self.n_embd, self.n_head) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, use_cache: bool = False, blocks_kvcache: Optional[BlocksKVCacheType] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[BlocksKVCacheType]]:
        """
        performs a forward pass of the model

        Parameters:
        - idx: a [B, T] tensor of integers representing the input sequence
        - targets: a [B, T] tensor of integers representing the output sequence

        Returns:
        - logits: a [B*T, C] tensor of non-normalized scores over the vocabulary
        - loss: a scalar loss value if targets is not None
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        history_length = 0 if not blocks_kvcache[0] else blocks_kvcache[0][0].shape[2]
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device) + history_length)

        x = tok_emb + pos_emb  # [B, T, C]
        new_kvcaches = []
        for block, kvcache in zip(self.blocks, blocks_kvcache):
            x, new_cache = block(x, use_cache=use_cache,
                                 kvcache=kvcache)  # [B, T, C]
            new_kvcaches.append(new_cache)
        x = self.ln_f(x)  # [B, T, C]
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # reshape logits and targets to [B*T, C] and [B*T] respectively
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        if use_cache:
            return logits, loss, new_kvcaches
        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        generates the next `max_token_len` tokens given an input sequence 

        Parameters:
        - idx: a [B, T] tensor of integers representing the input sequence
        - max_token_len: the maximum number of tokens to generate
        """
        curr_idx = idx
        blocks_kvcache = [None] * self.n_layer
        for _ in range(max_new_tokens):
            # get the predictions for the next token
            logits, loss, blocks_kvcache = self.forward(
                curr_idx, use_cache=True, blocks_kvcache=blocks_kvcache)
            # focus on the last token
            logits = logits[:, -1, :]  # this becomes [B, C]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # a [B, C] tensor
            # sample and get the next token
            idx_next = torch.multinomial(
                probs, num_samples=1)  # this is a [B, 1] tensor
            curr_idx = idx_next
            idx = torch.cat([idx, idx_next], dim=-1)  # becomes [B, T+1]
        return idx
