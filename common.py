import torch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from tiny_tokenizer import TinyTokenizer
from torch.utils.data import DataLoader
import functools

KVCacheType = Tuple[torch.Tensor, torch.Tensor]
BlocksKVCacheType = List[Optional[KVCacheType]]

def get_gpt2_tokenizer() -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '<PAD>'})
    return tokenizer

@dataclass
class GptConfig:
    """hyperparameters for GptLanguageModel"""

    batch_size: int = 64
    block_size: int = 256
    max_steps: int = 300000
    learning_rate: float = 3.0e-3
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd: int = 64
    n_head: int = 8
    n_groups: int = 4
    n_layer: int = 12
    dropout: float = 0.2
    seed: int = 42
    warmup_steps: int = .1 * max_steps 
    tokenizer: TinyTokenizer | PreTrainedTokenizer =  get_gpt2_tokenizer() 
    vocab_size: int = tokenizer.vocab_size + 1
    top_k: int = 5

hyperparameters = GptConfig()

class TinyStoriesLoader:

    def __init__(self, config: GptConfig, split:str='train') -> None:
        batch_size, max_length, num_workers = config.batch_size, config.block_size, 0
        self.tokenizer = config.tokenizer

        tokenize = functools.partial(
            self.tokenizer, 
            padding="max_length",
            truncation=True, 
            max_length=max_length,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors="pt"
        )

        dataset = load_dataset("roneneldan/TinyStories", streaming=True, split=split)
        dataset = dataset.shuffle(seed=config.seed)

        tokenized = dataset.map(tokenize, batched=True, input_columns=['text'], remove_columns=["text"])

        self.dataloader = DataLoader(
            tokenized,
            num_workers=num_workers,
            collate_fn=self.collate,
            drop_last=True,
            batch_size=batch_size
        )
        self.iterator = iter(self.dataloader)
    
    def collate(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(item['input_ids']) for item in batch)

        # Pad to the right manually
        padded_input_ids = [torch.cat([
            item['input_ids'],
            torch.full((max_len - len(item['input_ids']),), self.tokenizer.pad_token_id, dtype=torch.long)
        ]) for item in batch]

        input_ids = torch.stack(padded_input_ids)

        return {
            'input_ids': input_ids
        }

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            raise StopIteration

def compute_perplexity (model: torch.nn.Module, dataset: TinyStoriesLoader): 
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataset:
            input_ids = batch['input_ids'].to(model.device)
            xb, yb = input_ids[:, :-1], input_ids[:, 1:]
            _, loss, _ = model(xb, yb)

            total_loss += loss
            total_tokens += input_ids.numel()

    return torch.exp(-total_loss / total_tokens)


    