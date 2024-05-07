import torch
from clearml import Task
from datetime import datetime
from torch.optim.lr_scheduler import CosineAnnealingLR
from common import hyperparameters, TinyStoriesLoader
from model import GptLanguageModel

remote = False
load_last_checkpoint = False

if __name__ == "__main__":

    # Get the current date and time
    current_date_time = datetime.now()

    # Format the date and time in a string
    formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")

    task = Task.init(project_name='nanogpt', task_name=formatted_date_time)
    if remote:
        task.execute_remotely('default', clone=False, exit_process=True)
    logger = task.get_logger()
    task.connect(vars(hyperparameters))
    dataloader = TinyStoriesLoader(hyperparameters, split='train')
    einops_model = GptLanguageModel(hyperparameters)

    if load_last_checkpoint:
        try:
            einops_model.load_state_dict(torch.load('model_weights.pth'))
        except:
            pass
        
    m = einops_model.to(einops_model.device)
    tokenizer = m.tokenizer

    # create a pytorch optimizer and scheduler
    optimizer = torch.optim.AdamW(m.parameters(), lr=m.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=m.max_steps, eta_min=m.learning_rate * .1)
    # training the model

    epochs = 0
    for steps in range(m.max_steps):
        # sample a batch of data
        try:
            inputs = next(dataloader)
        except StopIteration:
            epochs += 1
            hyperparameters.seed += 1
            dataloader = TinyStoriesLoader(hyperparameters, split='train')
            inputs = next(dataloader)

        xb, yb = inputs['input_ids'][:, :-1], inputs['input_ids'][:, 1:]

        # switch to chain schedulers after this
        # apply linear warm up for learning rate
        if steps < m.warmup_steps:
            lr_scale = min(1.0, float(steps + 1) / m.warmup_steps)
            for group in optimizer.param_groups:
                group['lr'] = m.learning_rate * lr_scale
            logger.report_scalar(title="LR Scale", series="LR Scale",
                                value=lr_scale, iteration=steps)

        # evaluate the loss
        logits, loss, _ = m(xb.to(m.device), yb.to(m.device))
        
        logger.report_scalar(title="Train Loss", series="Train Loss",
                             value=loss.item(), iteration=steps)
        logger.report_scalar(title="Learning Rate", series="Learning Rate",
                                value=optimizer.param_groups[0]['lr'], iteration=steps)
        logger.report_scalar(title="Epochs", series="Epochs",
                                value=epochs, iteration=steps)
        
        optimizer.zero_grad(set_to_none=True)  # clear the gradients

        loss.backward()  # compute gradients
        optimizer.step()  # update parameters

        # apply cosine decay for learning rate
        scheduler.step()

    print(loss.item())

    torch.save(m.state_dict(), 'model_weights.pth')
    start_str = "\n"
    curr_token = tokenizer.encode(start_str)
    idx = torch.tensor(curr_token, dtype=torch.long,
                       device=hyperparameters.device).unsqueeze(0)
    print(tokenizer.decode(m.generate(
        idx=idx, max_new_tokens=hyperparameters.block_size-len(curr_token))[0].tolist()))
