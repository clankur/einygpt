import torch
from torch import nn
from clearml import Task
from datetime import datetime
from typing import Tuple, List, Dict
from collections import OrderedDict
from einops import rearrange

from common import GptConfig, encode, decode, get_batch, train_data, val_data, lp_hyperparameters
from model import GptLanguageModel

remote = True

hyperparameters = GptConfig()

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

    
    einops_model = GptLanguageModel(hyperparameters)
    
    m = einops_model.to(hyperparameters.device)
    block_layers = {
        f"blocks.{i}": f"blocks.{i}" for i in range(hyperparameters.n_layer)
    }

    return_layers = {
        "token_embedding_table": "token_embedding_table",
        "position_embedding_table": "position_embedding_table",
        "lm_head": "lm_head",
        **block_layers
    }


    # register hook
    # register_hook(m)

    # create a pytorch optimizer
    optimizer = torch.optim.AdamW(m.parameters(), lr=m.learning_rate)

    # training the model
    for steps in range(m.max_epochs):
        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        # mid_getter = MidGetter(m, return_layers=return_layers)
        # mid_outputs, model_output = mid_getter(xb, yb)
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
