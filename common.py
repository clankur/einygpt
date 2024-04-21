import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional
from datasets import load_dataset
from transformers import LlamaTokenizer
KVCacheType = Tuple[torch.Tensor, torch.Tensor]
BlocksKVCacheType = List[Optional[KVCacheType]]


tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
dataset = load_dataset("roneneldan/TinyStories")

# making a mapping from character to integers and vice versa
def encode(s): return tokenizer.encode(s).ids
def decode(l): return tokenizer.decode(l)

train_data, val_data = dataset['train'], dataset['validation']

@dataclass
class GptConfig:
    """hyperparameters for GptLanguageModel"""

    batch_size: int = 64
    block_size: int = 256
    max_epochs: int = 5000
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd: int = 384
    n_head: int = 6
    n_groups: int = 3
    n_layer: int = 6
    dropout: float = 0.2
    vocab_size: int = tokenizer.vocab_size

lp_hyperparameters = GptConfig(
    batch_size=32,
    block_size=8,
    max_epochs=5000,
    learning_rate=1e-3,
    n_embd=32,
    n_layer=3,
    n_head=4,
    dropout=0.2
)