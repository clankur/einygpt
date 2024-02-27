import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional
from dataclasses import dataclass

KVCacheType = Tuple[torch.Tensor, torch.Tensor]
BlocksKVCacheType = List[Optional[KVCacheType]]
torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


@dataclass
class GptConfig:
    """hyperparameters for GptLanguageModel"""

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

        self.token_emb_table = torch.ones((self.vocab_size, self.n_embd))
        self.position_emb_table = torch.ones((self.block_size, self.n_embd))

        # MLP projection matrices
        self.w_in = torch.ones((self.n_embd, 4 * self.n_embd))
        self.w_out = torch.ones((4 * self.n_embd, self.n_embd))

        # projection matrices for attention
        self.head_dim = self.n_embd // self.n_head

        self.attention_k = torch.ones(
            (self.n_embd, self.n_head, self.head_dim))  # [n, h]
        self.attention_q = torch.ones((self.n_embd, self.n_head, self.head_dim))
        self.attention_v = torch.ones((self.n_embd, self.n_head, self.head_dim))

        # for communication between attention heads
        self.mha_proj = torch.ones((self.n_embd, self.n_head, self.head_dim))

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, blocks_kvcache: Optional[BlocksKVCacheType] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        performs a forward pass of the model
        """

        tok_emb = self.token_emb_table[idx]
        B, T, C = tok_emb.shape

        history_length = 0 if not blocks_kvcache or not blocks_kvcache[
            0] else blocks_kvcache[0][0].shape[2]
        pos_emb = self.position_emb_table[history_length:history_length + T]

        x = tok_emb + pos_emb  # [B, T, C]

        for layer in range(self.n_layer):
            kv_cache = blocks_kvcache[layer] if blocks_kvcache else None
            k, v, q = [torch.einsum('btc,cnh->btnh', x, proj)
                       for proj in (self.attention_k, self.attention_q, self.attention_v)]
            
            # will reduce on c dimension
            if blocks_kvcache:  # not None if we are using cache
                if kv_cache:
                    prev_k, prev_v = [cache[:, :, -self.block_size - 1:, :] for cache in kv_cache] # truncate the first token
                    
                    # [B, n, K, h] -> [B, n, K+T, h]
                    k, v = [torch.cat([prev_x, x], dim=2)
                            for prev_x, x in [(prev_k, k), (prev_v, v)]]

                blocks_kvcache[layer] = (k, v)
            
            # need to transpose to [B, n, T, h]
            k, v, q = [x.transpose(1, 2) for x in (k, v, q)]
            att_wei = torch.einsum('bnqh,bnkh->bnqk', q, k) * (self.head_dim ** -0.5)

    def generate(self, idx: str, max_new_tokens: int) -> str:
        """
        generates a sequence of text
        """
        blocks_kvcache = [None] * self.n_layer
