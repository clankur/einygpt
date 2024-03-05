import torch
from dataclasses import dataclass
from typing import List, Tuple, Optional

KVCacheType = Tuple[torch.Tensor, torch.Tensor]
BlocksKVCacheType = List[Optional[KVCacheType]]

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))

# making a mapping from character to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

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
    vocab_size: int = len(chars)