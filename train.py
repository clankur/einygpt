import torch
from torch import nn
from clearml import Task
from datetime import datetime
from typing import Tuple, List, Dict
from collections import OrderedDict
from einops import rearrange
from torch.optim.lr_scheduler import CosineAnnealingLR
from common import hyperparameters, get_iterator_tokenizer_and_config
from model import GptLanguageModel

remote = True

if __name__ == "__main__":

    # Get the current date and time
    current_date_time = datetime.now()

    # Format the date and time in a string
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")

    # import hydra

    task = Task.init(project_name='nanogpt', task_name=formatted_date_time)
    if remote:
        task.execute_remotely('default', clone=False, exit_process=True)
    logger = task.get_logger()

    data_iterator, tokenizer, hyperparameters = get_iterator_tokenizer_and_config(hyperparameters)
    einops_model = GptLanguageModel(hyperparameters)
    m = einops_model.to(einops_model.device)

    # register hook
    # register_hook(m)
    # use a torch scheduler
    # linear warm up and cosine decay
    # try setting up warm up and decay for learning rate

    # create a pytorch optimizer and scheduler
    optimizer = torch.optim.AdamW(m.parameters(), lr=m.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=m.max_epochs, eta_min=1e-6)

    # training the model
    for steps in range(m.max_epochs):
        # sample a batch of data
        xb, yb = next(data_iterator)

        # evaluate the loss
        logits, loss, _ = m(xb.to(m.device), yb.to(m.device))

        logger.report_scalar(title="Train Loss", series="Train Loss",
                             value=loss.item(), iteration=steps)

        optimizer.zero_grad(set_to_none=True)  # clear the gradients
        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

        # apply linear warm up for learning rate
        if steps < m.warmup_steps:
            lr_scale = min(1.0, float(steps + 1) / m.warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = m.learning_rate * lr_scale

        # apply cosine decay for learning rate
        scheduler.step()

    print(loss.item())

    torch.save(m.state_dict(), 'model_weights.pth')

    start_str = "\n"
    curr_token = tokenizer.encode(start_str)
    idx = torch.tensor(curr_token, dtype=torch.long,
                       device=hyperparameters.device).unsqueeze(0)
    # get length of current token

    print(tokenizer.decode(m.generate(
        idx=idx, max_new_tokens=hyperparameters.block_size-len(curr_token))[0].tolist()))
