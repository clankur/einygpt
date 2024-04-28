import torch
from dataclasses import dataclass
from typing import Iterator, List, Dict, Tuple, Optional
import itertools
from types import SimpleNamespace
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer
from torch.utils.data import Dataset, DataLoader

KVCacheType = Tuple[torch.Tensor, torch.Tensor]
BlocksKVCacheType = List[Optional[KVCacheType]]

@dataclass
class GptConfig:
    """hyperparameters for GptLanguageModel"""

    batch_size: int = 64
    block_size: int = 128
    max_epochs: int = 5000
    learning_rate: float = 3e-4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_embd: int = 384
    n_head: int = 6
    n_groups: int = 3
    n_layer: int = 6
    dropout: float = 0.2
    seed: int = 42

hyperparameters = GptConfig(
    max_epochs=1
)

class Collator:
    def __init__ (self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(examples, return_attention_mask=False, return_tensors='pt')

        return {
            "inputs": batch['input_ids'][:, :-1],
            "inputs_position": batch['position'][:, :-1],
            "inputs_sequence": batch['sequences'][:, :-1],
            "targets": batch['input_ids'][:, 1:],
            "targets_position": batch['position'][:, 1:],
            "targets_sequence": batch['sequences'][:, 1:]
        }
    
def package_data(dataset: Dataset, tokenizer: PreTrainedTokenizer, config: GptConfig):
    """
    Tokenize the dataset and package it into chunks of block_size
    dataset: dataset to be tokenized
    tokenizer: tokenizer to be applied
    config: hyperparameters for model
    """

    def tokenize(examples: Dict[str, str]) -> PreTrainedTokenizer:
        return tokenizer(
            examples['text'],
            padding=False, # we will pad with the next sequence
            truncation=False, 
            max_length=None, 
            add_special_tokens=False,
            return_token_type_ids=False, # ?
            return_attention_mask=False,
        )
    
    def prep_example (examples: List[Dict[str, torch.Tensor]]) -> Iterator[torch.Tensor]:
        for example in examples:
            yield tokenizer.bos_token_id
            yield from example
    
    def new_chunk() -> SimpleNamespace:
        """
        Create a chunk - a SimpleNamespace with the following attributes:
            ids: token ids of the chunk
            positions: position embeddings of the chunk
            sequences: all sequences of the chunk storing the idx of each sequence
            sequence_idx: current sequence idx
            position: current position
        """

        return SimpleNamespace(
            ids = [], 
            positions = [], 
            sequences=[], 
            sequence_idx=0, 
            position=0
        )

    def chunk(examples: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Package the tokenized examples into chunks of block_size adding the position and sequence embeddings
        and changing the input_ids to a list of lists each containing block_size tokens
        """

        chunk = new_chunk()
        chunk_len = config.block_size
        global_ids = []
        global_positions = []
        global_sequences = []

        for token_id in prep_example(examples['input_ids']):
            if token_id == tokenizer.bos_token_id:
                chunk.sequence_idx += 1
                chunk.position = 0
            chunk.ids.append(token_id)
            chunk.positions.append(chunk.position)
            chunk.sequences.append(chunk.sequence_idx) 
            chunk.position += 1

            if len(chunk.ids) == chunk_len:
                global_ids.append(chunk.ids)
                global_positions.append(chunk.positions)
                global_sequences.append(chunk.sequences)
                chunk = new_chunk()
        
        if not global_ids: # we didn't package any chunks
            return []
        
        examples['input_ids'] = global_ids
        examples['position'] = global_positions
        examples['sequences'] = global_sequences

        return examples


    return dataset.map(
        tokenize, 
        batched=True, 
        remove_columns=['text']
    ).map(chunk, batched=True, batch_size=config.batch_size)

def make_iterator(config: GptConfig, dataset: Dataset, tokenizer: PreTrainedTokenizer) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    batch_size = config.batch_size
    og_dataset = dataset.shuffle(config.seed)
    for epoch in itertools.count():
        rdm_dataset = og_dataset.shuffle(config.seed + epoch)
        rdm_dataset = package_data(rdm_dataset, tokenizer, config)
        dataloader = iter(DataLoader(rdm_dataset, batch_size=batch_size, collate_fn=Collator(tokenizer)))
        for data in dataloader:
            yield data['inputs'], data['targets']

def get_iterator_tokenizer_and_config (config: GptConfig) -> Tuple[Iterator, PreTrainedTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
    dataset = load_dataset("roneneldan/TinyStories", streaming=True, split='train')
    iterator = make_iterator(config, dataset, tokenizer)
    config.vocab_size = tokenizer.vocab_size

    return iterator, tokenizer, config