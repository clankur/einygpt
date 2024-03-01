import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass
from einops import rearrange
import torch

KVCacheType = Tuple[torch.Tensor, torch.Tensor]
BlocksKVCacheType = List[Optional[KVCacheType]]
torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


@dataclass
class GptConfig:
    """hyperparameters for GptLanguageModel"""

    batch_size: int = 64
    block_size: int = 256
    max_epochs: int = 5000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters: int = 200
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2
    vocab_size: int = len(set(text))


class GptLanguageModel (nn.Module):

    def __init__(self, hyperparameters: GptConfig) -> None:
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)
        print(hyperparameters)
        self.token_emb_table = nn.Parameter(torch.randn((self.vocab_size, self.n_embd)))
        self.position_emb_table = nn.Parameter(torch.randn((self.block_size, self.n_embd)))

        # MLP projection matrices
        self.w_in = nn.Parameter(torch.randn((self.n_embd, 4 * self.n_embd)) / self.n_embd ** 0.5)
        self.w_out = nn.Parameter(torch.randn((4 * self.n_embd, self.n_embd)) / (4 * self.n_embd) ** 0.5)

        # projection matrices for attention
        self.head_dim = self.n_embd // self.n_head

        self.attention_k = nn.Parameter(torch.randn((self.n_embd, self.n_head, self.head_dim)) / self.head_dim ** 0.5)  # [C, h, d]
        self.attention_q = nn.Parameter(torch.randn((self.n_embd, self.n_head, self.head_dim)) / self.head_dim ** 0.5)
        self.attention_v = nn.Parameter(torch.randn((self.n_embd, self.n_head, self.head_dim)) / self.head_dim ** 0.5)

        # for communication between attention heads
        self.out_proj = nn.Parameter(torch.randn((self.n_head, self.head_dim, self.n_embd)) / self.head_dim ** 0.5)  # [h, d, C]

        self.register_buffer('tril', torch.tril(torch.ones(1, 1, self.block_size, self.block_size)))

        self.lm_head = nn.Parameter(torch.randn((self.n_embd, self.vocab_size)) / self.n_embd ** 0.5)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, blocks_kvcache: Optional[BlocksKVCacheType] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[BlocksKVCacheType]]:
        """
        performs a forward pass of the model
        """

        tok_emb = self.token_emb_table[idx]
        B, T, C = tok_emb.shape

        history_length = 0 if not blocks_kvcache or not blocks_kvcache[
            0] else blocks_kvcache[0][0].shape[2]
        print(T, history_length)
        pos_emb = self.position_emb_table[torch.arange(T, device=self.device) + history_length]


        x = tok_emb + pos_emb  # [B, T, C]

        for layer in range(self.n_layer):
            x = F.layer_norm(x, (B, T, C))
            kv_cache = blocks_kvcache[layer] if blocks_kvcache else None

            projections = torch.stack(
                [self.attention_k, self.attention_v, self.attention_q], dim=0)  # [3, C, h, d]

            # [B, T, C] @ [3, C, h, d] -> [3, B, h, T, d], S = 3
            k, v, q = torch.einsum('btc,schd->sbhtd', x, projections)
            # will reduce on C dimension

            if blocks_kvcache:  # not None if we are using cache
                if kv_cache:
                    prev_k, prev_v = [cache[:, :, -self.block_size - 1:, :]
                                      for cache in kv_cache]  # truncate the first token

                    # [B, h, K, d] -> [B, h, K+T, d]
                    k, v = [torch.cat([prev_elmnt, elmnt], dim=2)
                            for prev_elmnt, elmnt in [(prev_k, k), (prev_v, v)]]

                blocks_kvcache[layer] = (k, v)

            # shape of k, v are [B, h, K, d] and for q it's [B, h, Q, d]
            att_wei = torch.einsum('bhkd,bhqd->bhqk', q,
                                   k) * (self.head_dim ** -0.5)
            # casual masking
            att_wei = att_wei.masked_fill(
                self.tril[:, :, :T, :T] == 0, float('-inf')
            )

            att_wei = F.softmax(att_wei, dim=-1)
            att_wei = F.dropout(att_wei, p=self.dropout,
                                training=self.training)

            # [B, h, Q, K] @ [B, h, K, d] -> [B, h, Q, d]
            out = torch.einsum('bhqk,bhkd->bhqd', att_wei, v)

            # [B, h, Q, d] @ [h, d, C] -> [B, Q, C]
            out = torch.einsum('bhqd,hdc->bqc', out, self.out_proj)
            out = F.dropout(out, p=self.dropout, training=self.training)

            x = F.layer_norm(x + out, (B, T, C))
            x = x + out  # residual connection

            # MLP block
            # [B, T, C] @ [C, 4C] -> [B, T, 4C]
            mlp_hidden = torch.einsum('btc,cd->btd', x, self.w_in)
            mlp_hidden = F.relu(mlp_hidden)

            # [B, T, 4C] @ [4C, C] -> [B, T, C]
            mlp_out = torch.einsum('btd,dc->btc', mlp_hidden, self.w_out)
            mlp_out = F.dropout(x, p=self.dropout, training=self.training)
            x = x + mlp_out  # residual connection

        x = F.layer_norm(x, (B, T, C))
        logits = torch.einsum('btc,cd->btd', x, self.lm_head)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = rearrange(logits, 'b t c -> (b t) c')
            targets = rearrange(targets, 'b t -> (b t)')
            loss = F.cross_entropy(logits, targets)

        return x, loss, kv_cache

    def generate(self, idx: str, max_new_tokens: int) -> str:
        """
        generates a sequence of text
        """
        blocks_kvcache = [None] * self.n_layer
        curr_idx = idx
        for _ in range(max_new_tokens):
            logits, loss, blocks_kvcache = self.forward(
                curr_idx, blocks_kvcache=blocks_kvcache
            )
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            curr_idx = next_idx
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
