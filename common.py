import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from transformers import LlamaTokenizer
KVCacheType = Tuple[torch.Tensor, torch.Tensor]
BlocksKVCacheType = List[Optional[KVCacheType]]

tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
dataset = load_dataset("roneneldan/TinyStories")

encode = tokenizer.encode
decode = tokenizer.decode

def tokenize_function(examples: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """
    Tokenizes text examples.

    Args:
        examples (Dict[str, str]): A dictionary containing text examples.

    Returns:
        Dict[str, torch.Tensor]: A dictionary with tokenized tensors for input_ids, etc.
    """
    return tokenizer(examples["text"], truncation=True, padding="max_length")

train_data = dataset["train"].map(tokenize_function, batched=True)
val_data = dataset["validation"].map(tokenize_function, batched=True)

def get_batch(split: str, block_size:int, batch_size:int, device:str="cpu")-> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    # will return batch_size random numbers that are offsets of the data set
    ix = torch.randint(len(data) - block_size,
                       size=(batch_size,))
    # builds a stack of tensors of size blocksize for each random number in ix
    x = data["input_ids"][ix]
    y = data["input_ids"][ix + 1] # Shift by 1 for target prediction
    x, y = x.to(device), y.to(device)
    return x, y

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