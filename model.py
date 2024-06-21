import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from typing import List, Tuple, Optional
from common import GptConfig, KVCacheType, BlocksKVCacheType
from mup import MuReadout


class Block(nn.Module):
    def __init__(self, hyperparameters: GptConfig):
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)

        self.head_dim = self.n_embd // self.n_head

        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.dropout_param = nn.Dropout(self.dropout)
        self.register_buffer(
            "tril", torch.tril(torch.ones(1, 1, self.block_size, self.block_size))
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * self.n_embd, self.n_embd, bias=False),
            nn.Dropout(self.dropout),
        )
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)

    def forward(self, x: torch.Tensor, kvcache: KVCacheType) -> Tuple[torch.Tensor]:
        x = self.ln1(x)

        # perform MH attention
        B, T, C = x.shape

        k, v, q = self.key(x), self.value(x), self.query(x)  # [B, T, C]
        # C = n * h where n is the number of heads, h is the head dimension, C is the model dimension
        k, v, q = [
            t.reshape(B, T, self.n_head, self.head_dim) for t in (k, v, q)
        ]  # [B, T, C] -> [B, T, n, h]
        k, v, q = [
            torch.transpose(x, 1, 2) for x in (k, v, q)
        ]  # [B, T, n, h] -> [B, n, T, h]

        if kvcache is not None:
            if len(kvcache) != 0:
                prev_k, prev_v = kvcache
                prev_k, prev_v = (
                    prev_k[:, :, -self.block_size - 1 :, :],
                    prev_v[:, :, -self.block_size - 1 :, :],
                )
                # [B, n, K, h] -> [B, n, K+T, h]
                k = torch.cat([prev_k, k], dim=2)
                v = torch.cat([prev_v, v], dim=2)
            kvcache = (k, v)

        # [B, n, Q, h] @ [B, n, K, h] -> [B, n, Q, K]
        att_wei = torch.einsum("bnqh,bnkh->bnqk", q, k) * (self.head_dim)

        # casual masking
        att_wei = att_wei.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        # don't really get the dimensions defined in self.tril

        att_wei = F.softmax(att_wei, dim=-1)
        att_wei = self.dropout_param(att_wei)
        # [B, n, Q, K] @ [B, n, K, h] -> [B, n, Q, h]
        out = torch.einsum("bnqk,bnkh->bnqh", att_wei, v)
        out = rearrange(out, "b n q h -> b q (n h)")

        out = self.out_proj(out)  # mix the heads
        heads_out = self.dropout_param(out)  # apply dropout

        x = x + heads_out  # residual connection adding to the heads
        x = mlp_out = self.ln2(x)

        for layer in self.mlp:
            mlp_out = layer(mlp_out)
        x += mlp_out  # residual connection adding to ffwd

        return x, kvcache


class GptLanguageModel(nn.Module):
    def __init__(self, hyperparameters: GptConfig) -> None:
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(self.block_size, self.n_embd)
        self.blocks = nn.ModuleList(
            [
                Block(hyperparameters)
                for _ in range(self.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(self.n_embd)
        # self.lm_head = MuReadout(self.n_embd, self.vocab_size, bias=False)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kvcache: Optional[BlocksKVCacheType] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[BlocksKVCacheType]]:
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        history_length = 0 if not kvcache or not kvcache[0] else kvcache[0][0].shape[2]
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device) + history_length
        )

        x = tok_emb + pos_emb
        new_kvcaches = []
        for layer, block in enumerate(self.blocks):
            layer_kvcache = kvcache[layer] if kvcache else None
            x, new_cache = block(x, kvcache=layer_kvcache)  # [B, T, C]
            new_kvcaches.append(new_cache)

        x = self.ln_f(x)  # [B, T, C]

        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # reshape logits and targets to [B*T, C] and [B*T] respectively
            B, T, C = logits.shape

            logits = rearrange(logits, 'b t c -> (b t) c')
            targets = rearrange(targets, 'b t -> (b t)')

            loss = F.cross_entropy(logits, targets)

        return logits, loss, new_kvcaches

    def generate(self, idx: torch.Tensor, max_new_tokens:int) -> torch.Tensor:
        curr_idx = idx
        blocks_kvcache = [()] * self.n_layer
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