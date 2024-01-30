import torch
from torch import nn
from torch.nn import functional as F
from NanoGPTLangugageModel import NanoGPTLanguageModel, encode, decode, train_data, val_data, block_size, device

batch_size = 64
learning_rate = 3e-4
max_epochs = 5000
eval_interval = 500
eval_iters = 200

def get_batch(split: str) -> (torch.Tensor, torch.Tensor):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, size=(batch_size,)) # will return batch_size random numbers that are offsets of the data set 
    x = torch.stack([data[i:i+block_size] for i in ix]) # builds a stack of tensors of size blocksize for each random number in ix
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # offset by 1 stack of tensors
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model) -> dict:
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

if __name__ == "__main__":
    model = NanoGPTLanguageModel()
    m = model.to(device)

    # create a pytorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    # training the model
    for steps in range(max_epochs):
        # every once in a while eval loss on train and val sets
        if steps % eval_interval == 0:
            losses = estimate_loss(m)
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
    print(decode(m.generate(idx = idx, max_new_tokens=block_size)[0].tolist()))