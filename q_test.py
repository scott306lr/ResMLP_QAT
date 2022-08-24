import wandb

import os
import shutil
import time
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.utils import *

from torchsummary import summary
from src.model_analysis.utils import same_output
from src.model_analysis.visualize import simulate_input

model = resmlp_24(pretrained=True)
qmodel = q_test(model)

# x = simulate_input()

# for name, m in qmodel.named_modules():
#     setattr(m, 'regular', True)
# print(model(x))
# print(qmodel(x))

set_training(qmodel, False)
print(qmodel)

# print(same_output(model, qmodel, eps=1e-6))
# print(qmodel)

# for n, m in qmodel.named_modules():
#     # mod = getattr(model, attr)
#     print(n, type(m))

