import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange, einsum
from typing import List, Tuple, Optional
from common import GptConfig, KVCacheType, BlocksKVCacheType
from torch.utils.checkpoint import checkpoint

class GptLanguageModel (nn.Module):

    def __init__(self, hyperparameters: GptConfig) -> None:
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)

        torch.manual_seed(self.seed)

        self.token_embedding_table = nn.Parameter(
            torch.randn((self.vocab_size, self.n_embd)))
        self.position_embedding_table = nn.Parameter(
            torch.randn((self.block_size, self.n_embd)))

        # MLP projection matrices
        self.fc_in = nn.Parameter(torch.randn(
            (self.n_layer, self.n_embd, 4*self.n_embd)) / self.n_embd ** 0.5)
        self.fc_out = nn.Parameter(torch.randn(
            (self.n_layer, 4*self.n_embd, self.n_embd)) / (4 * self.n_embd) ** 0.5)

        # projection matrices for attention
        self.head_dim = self.n_embd // self.n_head
        self.num_kv_heads = self.n_head // self.n_groups # h = g * num_kv_heads
        
        self.q_proj = nn.Parameter(torch.randn(
            (self.n_layer, self.n_embd, self.n_groups, self.num_kv_heads, self.head_dim)) / self.n_embd ** 0.5) # [L, C, g, num_kv_heads, d] 
        
        self.kv_proj =  nn.Parameter(torch.randn(
            (self.n_layer, 2, self.n_embd, self.num_kv_heads, self.head_dim)) / self.n_embd ** 0.5) # [L, 2, C, num_kv_heads, d]
        
        # mixes the head outputs 
        self.out_proj = nn.Parameter(torch.randn(
            (self.n_layer, self.n_embd, self.n_embd)) / (self.head_dim * self.n_head) ** 0.5)  # [L, C, C]

        self.register_buffer('tril', torch.tril(
            torch.ones(1, 1, self.block_size, self.block_size)))

        self.lm_head = nn.Parameter(torch.randn(
            (self.n_embd, self.vocab_size)) / self.n_embd ** 0.5)

        self.scale = nn.Parameter(torch.ones(self.n_layer, 2, self.n_embd))
        
        self.out_scale = nn.Parameter(torch.ones(self.n_embd))

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
        for layer, (w_kv, w_q, w_out_proj, fc1, fc2, layer_scale) in enumerate(zip(self.kv_proj, self.q_proj, self.out_proj, self.fc_in, self.fc_out, self.scale)):

            x = F.layer_norm(x, (C,), weight=layer_scale[0])

            q = einsum(x, w_q, 'b t c, c g num_kv d -> b g num_kv t d') # [B, g, num_kv_heads, T, d]
            
            # [B, T, C] @ [2, C, num_kv_heads, d] -> [2, B, T, num_kv_heads, d]
            k, v = einsum(x, w_kv, 'b t c, s c num_kv d -> s b num_kv t d') # 2 [B, num_kv_heads, T, d]
            # will reduce on C dimension

            if blocks_kvcache:  # not None if we are using cache
                kv_cache = blocks_kvcache[layer]
                if kv_cache:
                    prev_k, prev_v = kv_cache
                    # [B, h, K, d] -> [B, h, K+T, d]
                    k = torch.cat([prev_k, k], dim=2)
                    v = torch.cat([prev_v, v], dim=2)

                blocks_kvcache[layer] = [cache[:, :, -self.block_size:, :]  # [B, h, K, d]
                                         for cache in (k, v)]  # truncate to conist of the last block_size tokens

            # shape of k, v are [B, num_kv_heads, K, d] and for q it's [B, g, num_kv_heads, Q, d]
            # compute qK^T for logits
            att_wei = einsum(q, k, 'b g num_kv q d, b num_kv k d -> b g num_kv q k') * (self.head_dim ** -0.5)

            # casual masking
            att_wei = att_wei.masked_fill(
                self.tril[:, :, :T, :T] == 0, float('-inf')
            )
            
            att_wei = F.softmax(att_wei, dim=-1) # normalized attention logits
            att_wei = F.dropout(att_wei, p=self.dropout,
                                training=self.training)

            # [B, g, num_kv_heads, Q, K] @ [B, num_kv_heads, K, d] -> [B, g, num_kv_heads, Q, d]
            out = einsum(att_wei, v, 'b g num_kv q k, b num_kv k d -> b g num_kv q d')

            # mixing the heads outputs amongst each other
            # [B, g, num_kv_heads, Q, d] -> [B, Q, C]
            out = rearrange(out, 'b g num_kv Q d -> b Q (g num_kv d)')
            # [B, Q, C] @ [C, C] -> [B, Q, C]
            out = einsum(out, w_out_proj, 'b Q C1, C1 C2 -> b Q C2')

            out = F.dropout(out, p=self.dropout, training=self.training)

            x = x + out
            x = F.layer_norm(x, [C], weight=layer_scale[1])
            # switching back to referencing Q as T, so out = [B, T, C]

            # MLP block
            # [B, T, C] @ [C, 4C] -> [B, T, 4C]
            mlp_hidden = einsum(x, fc1, 'b t c, c upscale_c -> b t upscale_c')
            mlp_hidden = F.relu(mlp_hidden)
            # [B, T, 4C] @ [4C, C] -> [B, T, C]
            mlp_out = einsum(mlp_hidden, fc2, 'b t c, c downscale_c -> b t downscale_c')
            mlp_out = F.dropout(mlp_out, p=self.dropout, training=self.training)

            x = x + mlp_out  # residual connection

        x = F.layer_norm(x, [C], weight=self.out_scale)
        
        logits = einsum(x, self.lm_head, 'b t c, c v -> b t v')
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = rearrange(logits, 'b t c -> (b t) c')
            targets = rearrange(targets, 'b t -> (b t)')
            loss = F.cross_entropy(logits, targets, reduction='none')

            # zero out the loss for the padding tokens
            padding_mask = (targets == self.tokenizer.pad_token_id)
            loss = loss * ~padding_mask
            loss = loss.sum() / (~padding_mask).sum()

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
