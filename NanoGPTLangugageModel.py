import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Optional

# hyperparameters
batch_size = 64
block_size = 256
max_epochs = 600
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # every head is = 384 / 6 = 64 dims, C = 64?
n_head = 1
n_layer = 6
dropout = 0.2
# -----

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# making a mapping from character to integers and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split: str) -> (torch.Tensor, torch.Tensor):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,)) # will return batch_size random numbers that are offsets of the data set 
    x = torch.stack([data[i:i+block_size] for i in ix]) # builds a stack of tensors of size blocksize for each random number in ix
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 stack of tensors
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss() -> dict:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # define tril as a buffer so it is not a parameter of the model
        self.dropout = nn.Dropout(dropout)

    def forward (self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass of the model

        Parameters:
        - x: a [B, T, C] tensor of floats representing the input sequence

        Returns:
        - out: a [B, T, C] tensor of floats representing the output sequence
        """
        # compute the keys, queries and values
        B, T, C = x.shape
        k = self.key(x) # [B, T, C]
        q = self.query(x) # [B, T, C]

        # computing the affiniities aka the attention scores
        # (C**-0.5) is a scaling factor to normalize the dot product
        wei = q @ k.transpose(-2, -1) * (C**-0.5) # [B, T, C] @ [B, C, T] -> [B, T, T]
        # decoder block
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # [B, T, T]
        wei = F.softmax(wei, dim=-1) # [B, T, T]
        wei = self.dropout(wei)
        # perform the weight aggregation of vals
        v = self.value(x) # [B, T, C]
        out = wei @ v # [B, T, T] @ [B, T, C] -> [B, T, C]
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self attention in parallel"""
    def __init__(self, num_heads:int, head_size:int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, use_cache:bool = False, kvcache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        performs a forward pass of the model

        Parameters:
        - x: a [B, T, C] tensor of floats representing the input sequence

        Returns:
        - out: a [B, T, C] tensor of floats representing the output sequence
        """
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x) # [B, T, C]
       
        # C = n * d where n is the number of heads, d is the head dimension, C is the model dimension
        k, v, q = [t.reshape(B, T, self.num_heads, self.head_size) for t in (k, v, q)] # [B, T, C] -> [B, T, n, d]

        if use_cache and kvcache is not None:
            prev_k, prev_v = kvcache
            k = torch.cat([prev_k, k], dim=1) # [B, K, n, d] -> [B, K+T, n, d]
            v = torch.cat([prev_v, v], dim=1)
            
<<<<<<< HEAD
        att_wei = torch.einsum('bqnd,bknd->bqkn', q, k) * (C**-0.5) # [B, Q, n, d] @ [B, K, n, d] -> [B, Q, K, n]
=======
        att_wei = torch.einsum('bqnd,bknd->bqkn', q, k) * (self.head_size**-0.5) # [B, Q, n, d] @ [B, K, n, d] -> [B, Q, K, n]
        # casual masking
        att_wei = att_wei.masked_fill(torch.tril(torch.ones(T, T, device=device)) == 0, float('-inf')) # [B, Q, K, n]
        # idk how this line of code here works besides its gonna give a tril hopefully

>>>>>>> eaef454 (forgot to apply casual mask, added it)
        att_wei = F.softmax(att_wei, dim=-2)
        att_wei = self.dropout(att_wei)
        out = torch.einsum('bqkn,bknd->bqnd', att_wei, v) # [B, Q, K, n] @ [B, K, n, d] -> [B, Q, n, d]

        out = out.reshape(B, T, C) # [B, T, n, d] -> [B, T, C] 
        out = self.proj(out) # what is the purpose of this? allow heads to communicate
        out = self.dropout(out) # apply dropout
        return out

class FeedForward(nn.Module):
    """ simple linear layer followed by a non-linearity and another linear layer"""
    def __init__ (self, n_embd) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass of the model

        Parameters:
        - x: a [B, T, C] tensor of floats representing the input sequence

        Returns:
        - a [B, T, C] tensor of floats representing the output sequence
        """
        return self.net(x)
    
class Block (nn.Module):
    """ a transformer block: intersperses communication with computation"""

    def __init__(self, n_embd:int, n_head: int) -> None:
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd) 
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        performs a forward pass of the model

        Parameters:
        - x: a [B, T, C] tensor of floats representing the input sequence

        Returns:
        - out: a [B, T, C] tensor of floats representing the output sequence
        """
        # we also perform layer normalization before being fed into the heads and ffwd
        x = x + self.sa_heads(self.ln1(x)) # residual connection adding to sa heads
        x = x + self.ffwd(self.ln2(x)) # residual connection adding to ffwd
        return x

class NanoGPTLanguageModel(nn.Module):
    def __init__(self) -> None:   
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx: torch.Tensor, targets:torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        performs a forward pass of the model

        Parameters:
        - idx: a [B, T] tensor of integers representing the input sequence
        - targets: a [B, T] tensor of integers representing the output sequence

        Returns:
        - logits: a [B*T, C] tensor of non-normalized scores over the vocabulary
        - loss: a scalar loss value if targets is not None
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb # [B, T, C]
        x = self.blocks(x) # [B, T, C] 
        x = self.ln_f(x) # [B, T, C]
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else: 
            # reshape logits and targets to [B*T, C] and [B*T] respectively
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

            loss = F.cross_entropy(logits, targets)
        return logits, loss        
    
    def generate (self, idx: torch.Tensor, max_new_tokens:int) -> torch.Tensor:
        """
        generates the next `max_token_len` tokens given an input sequence 

        Parameters:
        - idx: a [B, T] tensor of integers representing the input sequence
        - max_token_len: the maximum number of tokens to generate
        """
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens since pos embs runs out scope
            idx_cond = idx[:, -block_size:]
            # get the predictions for the next token
            logits, loss = self.forward(idx_cond)
            # focus on the last token
            logits = logits[:, -1, :] # this becomes [B, C]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # a [B, C] tensor
            # sample and get the next token
            idx_next = torch.multinomial(probs, num_samples=1) # this is a [B, 1] tensor
            idx = torch.cat([idx, idx_next], dim=-1) # becomes [B, T+1]
        return idx
    
if __name__ == "__main__":
    model = NanoGPTLanguageModel()
    m = model.to(device)

    # create a pytorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    # training the model
    for steps in range(max_epochs):
        # every once in a while eval loss on train and val sets
        if steps % eval_interval == 0:
            losses = estimate_loss()
            print(f"Step: {steps}, Train loss: {losses['train']:.2f}, Val loss: {losses['val']:.2f}")   
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = m(xb, yb) 
        optimizer.zero_grad(set_to_none=True) # clear the gradients
        loss.backward() # compute gradients
        optimizer.step() # update parameters

    print(loss.item())

    torch.save(m.state_dict(), 'model_weights.pth')

    start_str = "\n"
    idx = torch.tensor(encode(start_str), dtype=torch.long, device=device).unsqueeze(0)
    print(decode(m.generate(idx = idx, max_new_tokens=500)[0].tolist()))