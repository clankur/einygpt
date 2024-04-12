import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, einsum
from typing import List, Tuple, Optional
from common import GptConfig, KVCacheType, BlocksKVCacheType

torch.manual_seed(1337)

class GptLanguageModel (nn.Module):

    def __init__(self, hyperparameters: GptConfig) -> None:
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)
        self.token_embedding_table = nn.Parameter(
            torch.randn((self.vocab_size, self.n_embd)))
        self.position_embedding_table = nn.Parameter(
            torch.randn((self.block_size, self.n_embd)))

        # MLP projection matrices

        self.w_in = nn.Parameter(torch.randn(
            (self.n_layer, self.n_embd, 4*self.n_embd)) / self.n_embd ** 0.5)
        self.w_out = nn.Parameter(torch.randn(
            (self.n_layer, 4*self.n_embd, self.n_embd)) / (4 * self.n_embd) ** 0.5)

        # projection matrices for attention
        self.head_dim = self.n_embd // self.n_head

        self.attention_kvq = nn.Parameter(torch.randn(
            (self.n_layer, 3, self.n_embd, self.n_head, self.head_dim)) / self.head_dim ** 0.5)  # [L, 3, C, h, d]

        # for communication between attention heads
        self.out_proj = nn.Parameter(torch.randn(
            (self.n_layer, self.n_embd, self.n_embd)) / self.head_dim ** 0.5)  # [L, h, d, C]

        self.register_buffer('tril', torch.tril(
            torch.ones(1, 1, self.block_size, self.block_size)))

        self.lm_head = nn.Parameter(torch.randn(
            (self.n_embd, self.vocab_size)) / self.n_embd ** 0.5)

        self.scale = nn.Parameter(torch.ones(self.n_layer, 3, self.n_embd))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, blocks_kvcache: Optional[BlocksKVCacheType] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[BlocksKVCacheType]]:
        """
        performs a forward pass of the model
        """

        tok_emb = self.token_embedding_table[idx]
        B, T, C = tok_emb.shape

        history_length = 0 if not blocks_kvcache or not blocks_kvcache[
            0] else blocks_kvcache[0][0].shape[2]
        pos_emb = self.position_embedding_table[torch.arange(
            T, device=self.device) + history_length]

        x = tok_emb + pos_emb  # [B, T, C]
        for layer, (projections, layer_w_in, layer_w_out, mha_proj, layer_scale) in enumerate(zip(self.attention_kvq, self.w_in, self.w_out, self.out_proj, self.scale)):
            x = F.layer_norm(x, (C,), weight=layer_scale[0])

            # [B, T, C] @ [3, C, h, d] -> [3, B, h, T, d], S = 3
            k, v, q = torch.einsum('btc,schd->sbhtd', x, projections)
            # will reduce on C dimension

            if blocks_kvcache:  # not None if we are using cache
                kv_cache = blocks_kvcache[layer]
                if kv_cache:
                    prev_k, prev_v = kv_cache
                    # [B, h, K, d] -> [B, h, K+T, d]
                    k = torch.cat([prev_k, k], dim=2)
                    v = torch.cat([prev_v, v], dim=2)

                blocks_kvcache[layer] = [cache[:, :, -self.block_size:, :] # [B, h, K, d]
                                      for cache in (k, v)]  # truncate to conist of the last block_size tokens

            # shape of k, v are [B, h, K, d] and for q it's [B, h, Q, d]
            att_wei = torch.einsum('bhqd,bhkd->bhqk', q,
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
            out = rearrange(out, 'b h q d -> b q (h d)')
            out = torch.einsum('bqe,ce->bqc', out, mha_proj)
            out = F.dropout(out, p=self.dropout, training=self.training)

            x = x + out
            x = F.layer_norm(x, [C], weight=layer_scale[1])

            # MLP block
            # [B, T, C] @ [C, 4C] -> [B, T, 4C]
            mlp_hidden = torch.einsum('btc,cd->btd', x, layer_w_in)
            mlp_hidden = F.relu(mlp_hidden)
            # [B, T, 4C] @ [4C, C] -> [B, T, C]
            mlp_out = torch.einsum('btd,dc->btc', mlp_hidden, layer_w_out)
            mlp_out = F.dropout(mlp_out, p=self.dropout, training=self.training)

            x = x + mlp_out  # residual connection

        x = F.layer_norm(x, [C], weight=layer_scale[2])
        
        logits = torch.einsum('btc,cv->btv', x, self.lm_head)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = rearrange(logits, 'b t c -> (b t) c')
            targets = rearrange(targets, 'b t -> (b t)')
            loss = F.cross_entropy(logits, targets)

        return logits, loss, blocks_kvcache

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
