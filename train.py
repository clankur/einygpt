import torch
from torch import nn
from clearml import Task
from datetime import datetime
from typing import Tuple, List, Dict
from collections import OrderedDict
from einops import rearrange

from common import GptConfig, encode, decode, train_data, val_data, lp_hyperparameters
from NanoGPTLangugageModel import NanoGPTLanguageModel
from model import GptLanguageModel

remote = False 

hyperparameters = GptConfig()

def get_batch(split: str)-> Tuple[torch.Tensor, torch.Tensor]:
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

def convert_state_dict(einops_model, nano_model):
    gpt_state_dict = einops_model.state_dict()
    nano_state_dict = nano_model.state_dict()

    n_head = hyperparameters.n_head
    # Copy over the parameters that don't need to be transformed
    for param in ['token_embedding_table.weight', 'position_embedding_table.weight', 'lm_head.weight']:
        field = param.split('.')[0]
        if param == 'lm_head.weight':
            gpt_state_dict[field] = nano_state_dict[param].T
        else:
            gpt_state_dict[field] = nano_state_dict[param]

    # Transform the parameters for the blocks
    att_kvq = torch.zeros_like(gpt_state_dict['attention_kvq'])
    out_proj = torch.zeros_like(gpt_state_dict['out_proj'])
    w_in = torch.zeros_like(gpt_state_dict['w_in'])
    w_out = torch.zeros_like(gpt_state_dict['w_out'])

    for i in range(hyperparameters.n_layer):
        block_prefix = f'blocks.{i}.'
        # Attention weights
        key_weight = nano_state_dict[block_prefix + 'sa_heads.key.weight'].T

        value_weight = nano_state_dict[block_prefix + 'sa_heads.value.weight'].T
        query_weight = nano_state_dict[block_prefix + 'sa_heads.query.weight'].T
        projection = rearrange(torch.stack([key_weight, value_weight, query_weight], dim=0), "s c (h d) -> s c h d", h=n_head)
        att_kvq[i] = projection

        out_proj_weight = nano_state_dict[block_prefix + 'sa_heads.proj.weight']
        out_proj[i] = out_proj_weight # rearrange(out_proj_weight, "(h d) c -> h d c", h=hyperparameters.n_head)

        # Feedforward network weights and biases
        w_in[i] = nano_state_dict[block_prefix + 'ffwd.net.0.weight'].T
        w_out[i]= nano_state_dict[block_prefix + 'ffwd.net.2.weight'].T

        # Projection weights
    gpt_state_dict['attention_kvq'] = att_kvq
    gpt_state_dict['out_proj'] = out_proj
    gpt_state_dict['w_in'] = w_in
    gpt_state_dict['w_out'] = w_out

    # print(gpt_state_dict)
    return gpt_state_dict

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

    
    pytorch_model = NanoGPTLanguageModel(hyperparameters)
    einops_model = GptLanguageModel(hyperparameters)
    einops_model.load_state_dict(convert_state_dict(einops_model, pytorch_model))
    
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
