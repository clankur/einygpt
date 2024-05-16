# einygpt

An implementation of a transformer-based language model primarily using `einops` and trained over the TinyStories dataset. It incorporates techniques to improve memory bandwidth, such as a KVCache implementation and the usage of GQA (grouped query attention).

Training a 6.9 million parameter model on a RTX4090 with the GPT2Tokenizer achieves results inline with the findings from the [TinyStories paper](https://arxiv.org/pdf/2305.07759) and gets a perplexity of 1.0001195 over the validation set. Additionally a 4.3 million parameter model with its [own Byte-Pair Encoding tokenizer](tiny_tokenizer.py) trained on the TinyStories dataset achieves a slightly lower perplexity of 1.0000896.

Both models produce stories that have a logical flow and have a good grasp of grammar. The custom tokenizer model does have it's drawbacks, despite the lower perplexity - it generates less text within the same context length and also treat's punctuation as seperate tokens leading to whitespace between it - both being a result of the custom tokenizer. You can compare their outputs side by side in this [notebook](perplexity.ipynb).