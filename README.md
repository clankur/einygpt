# einygpt

An implementation of a transformer primarily using einops and trained on the TinyStories dataset. Contains improvements to improve memory bandwidth such as the implmentation of a KVCache and usage of GQA (grouped query attention).

When training a 6.9 million parameter model on a RTX4090 with the GPT2Tokenizer, it achieves results inline with the findings in the [TinyStories paper](https://arxiv.org/pdf/2305.07759).
