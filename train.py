import torch
from torch import nn
from torch.nn import functional as F
from clearml import Task
from datetime import datetime
import torch
from typing import Tuple

from  common import GptConfig, encode, decode, train_data, val_data
from NanoGPTLangugageModel import NanoGPTLanguageModel
from model import GptLanguageModel


low_power_hyperparameters = GptConfig(
    batch_size=32,
    block_size=8,
    max_epochs=5000,
    eval_interval=500,
    learning_rate=1e-3,
    eval_iters=200,
    n_embd=32,
    n_layer=3,
    n_head=4,
    dropout=0.2
)
hyperparameters = GptConfig(
    max_epochs=1000
)


def get_batch(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == 'train' else val_data
    # will return batch_size random numbers that are offsets of the data set
    ix = torch.randint(len(data) - hyperparameters.block_size,
                       size=(hyperparameters.batch_size,))
    # builds a stack of tensors of size blocksize for each random number in ix
    x = torch.stack([data[i:i+hyperparameters.block_size] for i in ix])
    y = torch.stack([data[i+1:i+hyperparameters.block_size+1]
                    for i in ix])  # offset by 1 stack of tensors
    x, y = x.to(hyperparameters.device), y.to(hyperparameters.device)
    return x, y


@torch.no_grad()
def estimate_loss(model) -> dict:
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hyperparameters.eval_iters)
        for k in range(hyperparameters.eval_iters):
            X, Y = get_batch(split)
            _, loss, _ = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":

    # Get the current date and time
    current_date_time = datetime.now()

    # Format the date and time in a string
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")

    # import hydra

    task = Task.init(project_name='nanogpt', task_name=formatted_date_time)
    task.execute_remotely('default', clone=False, exit_process=True)
    logger = task.get_logger()

    model = NanoGPTLanguageModel(hyperparameters)
    m = model.to(hyperparameters.device)

    # create a pytorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=model.learning_rate)

    # training the model
    for steps in range(model.max_epochs):
        # every once in a while eval loss on train and val sets
        if steps % model.eval_interval == 0:
            losses = estimate_loss(m)
            print(
                f"Step: {steps}, Train loss: {losses['train']:.2f}, Val loss: {losses['val']:.2f}")
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss, _ = m(xb, yb)

        logger.report_scalar(title="Train Loss", series="Train Loss",
                             value=loss.item(), iteration=steps)

        optimizer.zero_grad(set_to_none=True)  # clear the gradients
        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

    print(loss.item())

    torch.save(m.state_dict(), 'model_weights.pth')

    start_str = "\n"
    idx = torch.tensor(encode(start_str), dtype=torch.long,
                       device=hyperparameters.device).unsqueeze(0)
    print(decode(m.generate(
        idx=idx, max_new_tokens=hyperparameters.block_size)[0].tolist()))
