from tokenizers import Tokenizer 
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from typing import Any, List, Optional
import torch

class TinyTokenizer:
    def __init__(self, tokenizer_path: Optional[str]=None) -> None:
        if tokenizer_path is None:
            def filter_non_ascii(examples: List[str]) -> List[str]:
                return { "text" : [''.join(char for char in text if ord(char) < 128 ) for text in examples] }

            dataset = load_dataset("roneneldan/TinyStories", split='train')
            dataset = dataset.map(filter_non_ascii, input_columns='text', batched=True, num_proc=4)

            self.trainer = BpeTrainer(special_tokens=["[UNK]", "[BOS]", "[SEP]", "[PAD]", "[MASK]"])
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = Whitespace()
            self.tokenizer.train_from_iterator(dataset['text'], trainer=self.trainer, length=5000)
            self.tokenizer.save("tiny_tokenizer.json")
        else:
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # variables needed for the model also defined by tokenizers fetched by AutoTokenizer
        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str, *args: Any, **kwargs: Any) -> List[int]:
        return self.tokenizer.encode(text, *args, **kwargs).ids
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, examples: List[str] | str, padding="max_length", truncation:bool=True, max_length:Optional[int]=None, add_special_tokens:bool=True, return_tensors:str="pt", *args: Any, **kwds: Any):
        if padding == "max_length":
            self.tokenizer.enable_padding(length=max_length)
        if truncation:
            self.tokenizer.enable_truncation(max_length=max_length)
        if isinstance(examples, str):
            return self.encode(examples, add_special_tokens=add_special_tokens)
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        tokenized_examples  = [self.encode(text, add_special_tokens=add_special_tokens) for text in examples]
        return {"input_ids": torch.tensor(tokenized_examples)}