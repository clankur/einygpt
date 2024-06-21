import torch
import torch.nn as nn
from einops import rearrange 
from torch.nn import functional as F
from typing import List, Tuple, Optional
from common import GptConfig, KVCacheType, BlocksKVCacheType
from mup import MuReadout

class Block (nn.Module) :
    def __init__ (self, hyperparameters: GptConfig): 
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)
        
        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.query = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.dropout = nn.Dropout(self.dropout)
        self.register_buffer('tril', torch.tril(
            torch.ones(1, 1, self.block_size, self.block_size)))

        self.net = nn.Sequential(
            nn.Linear(self.n_embd, 4 * self.n_embd, bias=False),
            nn.ReLU(),
            nn.Linear(4 * self.n_embd, self.n_embd, bias=False),
            nn.Dropout(self.dropout)
        )
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)

    def forward (self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        pass
    
class GPTLanguageModel(nn.Module):
    def __init__(self, hyperparameters: GptConfig) -> None:
        super().__init__()
        for k, v in hyperparameters.__dict__.items():
            setattr(self, k, v)

        self.token_embedding_table = nn.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = nn.Embedding(
            self.block_size, self.n_embd)
        self.blocks = nn.ModuleList(
            [Block(hyperparameters, self.n_embd, self.n_head) for _ in range(self.n_layer)])
        self.ln_f = nn.LayerNorm(self.n_embd)
        self.lm_head = MuReadout(self.n_embd, self.vocab_size, bias=False)

    def forward (self, x: torch.Tensor):
        pass

    def generate (self): 
        pass

