# einygpt

An implementation of a transformer-based language model primarily using `einops` and trained over the TinyStories dataset. It incorporates techniques to improve memory bandwidth, such as a KVCache implementation and the usage of GQA (grouped query attention).

Training a 6.9 million parameter model on a RTX4090 with the GPT2Tokenizer achieves results inline with the findings from the [TinyStories paper](https://arxiv.org/pdf/2305.07759) and gets a perplexity of 0.996 over the validation set. Additionally a 4.3 million parameter model with its [own Byte-Pair Encoding tokenizer](tiny_tokenizer.py) trained on the TinyStories dataset achieves a perplexity of 0.999.
