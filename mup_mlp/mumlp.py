# %%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from mup import MuSGD, get_shapes, set_base_shapes, make_base_shapes, MuReadout
# %%
nonlin = F.relu
criterion = F.cross_entropy
batch_size = 32
# %%
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
class MLP(nn.Module):
    def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(MLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(784, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = nn.Linear(width, num_classes, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult ** 0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in') 
        nn.init.zeros_(self.fc_3.weight)
        
    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult ** 0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out) * self.output_mult

class muMLP(nn.Module):
    def __init__(self, width=128, num_classes=10, nonlin=F.relu, output_mult=1.0, input_mult=1.0):
        super(muMLP, self).__init__()
        self.nonlin = nonlin
        self.input_mult = input_mult
        self.output_mult = output_mult
        self.fc_1 = nn.Linear(784, width, bias=False)
        self.fc_2 = nn.Linear(width, width, bias=False)
        self.fc_3 = MuReadout(width, num_classes, bias=False, output_mult=output_mult)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.fc_1.weight, a=1, mode='fan_in')
        self.fc_1.weight.data /= self.input_mult ** 0.5
        nn.init.kaiming_normal_(self.fc_2.weight, a=1, mode='fan_in') 
        nn.init.zeros_(self.fc_3.weight)
        
    def forward(self, x):
        out = self.nonlin(self.fc_1(x) * self.input_mult ** 0.5)
        out = self.nonlin(self.fc_2(out))
        return self.fc_3(out)

# %%
base_shapes_path = './demo_width256.bsh'
data_dir = './tmp'

# %%
base_shapes = get_shapes(MLP(width=256, nonlin=nonlin))
delta_shapes = get_shapes(
    # just need to change whatever dimension(s) we are scaling
    MLP(width=256+1, nonlin=nonlin)
)
make_base_shapes(base_shapes, delta_shapes, savefile=base_shapes_path)
# %%
batch_size = 64
epochs = 20
log_interval = 300
# optimal HPs
output_mult = 32
input_mult = 0.00390625

# %%
# train procedure
from datetime import datetime
from clearml import Task

current_date_time = datetime.now()

# Format the date and time in a string
formatted_date_time = current_date_time.strftime("%Y-%m-%d %H:%M:%S")

task = Task.init(project_name='mupmlp', task_name=formatted_date_time)
task.execute_remotely('default', clone=False, exit_process=True)
logger = task.get_logger()

def train (model, device, train_loader, optimizer, epoch, scheduler=None, criterion=criterion):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
    
        loss = criterion(output, target)
        loss.backward()
        train_loss += loss.item() * data.shape[0]
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            logger.report_scalar(title="Train Loss", series="Train Loss",
                             value=loss.item(), iteration=epoch)
            accuracy = correct / len(train_loader.dataset) * 100.
            logger.report_scalar(title="Train Accuracy", series="Train Accuracy",
                                value=accuracy, iteration=epoch)
            
            print(f"Progress: {100. * batch_idx / len(train_loader):.2f}%")
        
        if scheduler is not None:
            scheduler.step()

    train_loss /= len(train_loader.dataset)
    print(f'Train set: Average loss: {train_loss:.4f}, correct {correct}, total {len(train_loader.dataset)}') 
    return train_loss

# %%
# maximal update parameterization training
from torchvision import datasets, transforms
transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root=data_dir, train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2)
testset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=False, num_workers=2)
# %%
import math
logs = []

for width in [256, 512, 1024, 2048, 4096, 8192]:
    for log2lr in np.linspace(-8, 0, 20):
        torch.manual_seed(1)
        mynet = muMLP(width=width, nonlin=nonlin, output_mult=output_mult, input_mult=input_mult).to(device)
        print(f'loading base shapes from {base_shapes_path}')
        set_base_shapes(mynet, base_shapes_path)
        print('done')
        optimizer = MuSGD(mynet.parameters(), lr=2**log2lr)
        for epoch in range(1, epochs+1):
            train_loss = train(mynet, device, train_loader, optimizer, epoch, criterion=criterion)
            logs.append(dict(
                epoch=epoch,
                model_type='muP MLP',
                log2lr=log2lr,
                train_loss=train_loss,
                width=width,
            ))
            if math.isnan(train_loss):
                break
# %%
np
# %%
