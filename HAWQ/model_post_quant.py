import torchvision.datasets as datasets
from bit_config import *
from utils import *
import torch
from torchsummary import summary
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os, time

model = resmlp_24(pretrained=True)
qmodel = q_resmlp24(model, full_precision_flag=True)

def get_linear_layers(model):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            linear_layers.append((name, module))
    return linear_layers

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def calibrate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time],
        prefix='Calibrate: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

    print('Calibration done.')

def find_layers_dist(linear_layers):
    layers_dist = {}
    momentum = 0.1
    def get_std_mean(name):
        def hook(model, input, output):
            new_std_mean = output.detach().torch.std_mean(x, dim=[0, 1], unbiased=False)
            if name in layers_dist:
                layers_dist[name][0] = (layers_dist[name][0]*momentum) + (new_std_mean[0]*(1 - momentum))
                layers_dist[name][1] = (layers_dist[name][1]*momentum) + (new_std_mean[1]*(1 - momentum))
            else:
                layers_dist[name] = new_std_mean
        return hook

    # register hook
    for i, (n, m) in enumerate(linear_layers):
        m.register_forward_hook(get_std_mean(f'l{i}-{n}'))

    # access small batch of validation data
    data_loc = "/mnt/disk1/imagenet"
    valdir = os.path.join(data_loc, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop(train_resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    data_percentage = 0.0001
    dataset_length = int(len(val_dataset) * data_percentage)
    partial_train_dataset, _ = torch.utils.data.random_split(val_dataset,
                                                                [dataset_length, len(val_dataset) - dataset_length])
    val_loader = torch.utils.data.DataLoader(
        partial_train_dataset, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    calibrate(val_loader, model)

    return layers_dist
    

def high_bias_absorption(linear_layers, layers_dist):
    for idx in range(1, len(linear_layers)):
        (prev_name, prev), (curr_name, curr) = linear_layers[idx-1], linear_layers[idx]
        gamma, beta = layers_dist[idx-1]
        # torch.std_mean(a, dim=1, unbiased=False)[0]

def resmlp_bias_absorb(model):
    linear_layers = []
    for i in range(0, 24):
        todo_layer = getattr(model, f'layer{i}')
        linear_layers += get_linear_layers(todo_layer)[3:] # cross-channel sublayer only
    print(len(linear_layers))
    layers_dist = find_layers_dist(linear_layers)
    high_bias_absorption(linear_layers, layers_dist)

resmlp_bias_absorb(qmodel)
# getattr(qmodel, f'layer0')