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

def get_linear_layers(model, prefix=""):
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((f'{prefix}{name}', module))
    return linear_layers

class StdMean(object):
    def __init__(self, std=0, mean=0):
        self.std = std
        self.mean = mean

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
    momentum = 0.99
    def get_std_mean(name):
        def hook(model, input, output):
            new_std_mean = torch.std_mean(output.detach(), dim=[0, 1], unbiased=False)
            if name in layers_dist:
                layers_dist[name].std = (layers_dist[name].std*momentum) + (new_std_mean[0]*(1 - momentum))
                layers_dist[name].mean = (layers_dist[name].mean*momentum) + (new_std_mean[1]*(1 - momentum))
            else:
                layers_dist[name] = StdMean(new_std_mean[0], new_std_mean[1])

        return hook

    # register hook
    for n, m in linear_layers:
        m.register_forward_hook(get_std_mean(n))

    # access small batch of validation data
    print("Loading a small piece of validation data...")
    data_loc = "/mnt/disk1/imagenet"
    valdir = os.path.join(data_loc, 'val')
    train_resolution = 224
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
    
    data_percentage = 0.01
    dataset_length = int(len(val_dataset) * data_percentage)
    partial_val_dataset, _ = torch.utils.data.random_split(val_dataset,
                                                                [dataset_length, len(val_dataset) - dataset_length])
    val_loader = torch.utils.data.DataLoader(
        partial_val_dataset, batch_size=32, shuffle=True,
        num_workers=4, pin_memory=True, sampler=None)

    calibrate(val_loader, model)

    return layers_dist
    

def high_bias_absorption(linear_layers, layers_dist):
    for idx in range(1, len(linear_layers)):
        (prev_name, prev), (curr_name, curr) = linear_layers[idx-1], linear_layers[idx]
        
        gamma, beta = layers_dist[prev_name].std, layers_dist[prev_name].mean
        
        c = (beta - 3 * torch.abs(gamma)).clamp_(min = 0)
        print(prev_name, prev.weight.shape, prev.bias.shape)
        print(curr_name, curr.weight.shape, curr.bias.shape)
        print("c", c.shape)
        print()
        prev.bias.data.add_(-c)
        w_mul = curr.weight.data.matmul(c)
        curr.bias.data.add_(w_mul)

def cross_layer_equalization(linear_layers):
    '''
    Perform Cross Layer Scaling :
    Iterate modules until scale value is converged up to 1e-8 magnitude
    '''
    S_history = dict()
    eps = 1e-8
    converged = [False] * (len(linear_layers)-1)
    with torch.no_grad(): 
        while not np.all(converged):
            for idx in range(1, len(linear_layers)):
                (prev_name, prev), (curr_name, curr) = linear_layers[idx-1], linear_layers[idx]
                
                range_1 = 2.*torch.abs(prev.weight).max(axis = 1)[0] # abs max of each row * 2
                range_2 = 2.*torch.abs(curr.weight).max(axis = 0)[0] # abs max of each col * 2
                S = torch.sqrt(range_1 * range_2) / range_2

                if idx in S_history:
                    prev_s = S_history[idx]
                    if torch.allclose(S, prev_s, atol=eps):
                        converged[idx-1] = True
                        continue
                    else:
                        converged[idx-1] = False

                # div S for each row
                prev.weight.data.div_(S.view(-1, 1))
                if prev.bias is not None:
                    prev.bias.data.div_(S)
                
                # mul S for each col
                curr.weight.data.mul_(S)
                    
                S_history[idx] = S
    return linear_layers

def cle_for_resmlp(model):
    for i in range(0, 24):
        todo_layer = model.blocks[i]
        linear_layers = get_linear_layers(todo_layer)[3:] # cross-channel sublayer only
        cross_layer_equalization(linear_layers)

def resmlp_bias_absorb(model):
    linear_layers = []
    for i in range(0, 24):
        todo_layer = model.blocks[i]
        linear_layers += get_linear_layers(todo_layer, f'{i}-')[3:6] # cross-channel sublayer only
    print("Linear layers to track:", len(linear_layers))
    layers_dist = find_layers_dist(linear_layers)

    for i in range(0, 24):
        todo_layer = model.blocks[i]
        todo_layers = get_linear_layers(todo_layer, f'{i}-')[3:6] # cross-channel sublayer only
        high_bias_absorption(todo_layers, layers_dist)

def cross_layer_equalization(linear_layers):
    '''
    Perform Cross Layer Scaling :
    Iterate modules until scale value is converged up to 1e-8 magnitude
    '''
    S_history = dict()
    eps = 1e-8
    converged = [False] * (len(linear_layers)-1)
    with torch.no_grad(): 
        while not np.all(converged):
            for idx in range(1, len(linear_layers)):
                (prev_name, prev), (curr_name, curr) = linear_layers[idx-1], linear_layers[idx]
                
                range_1 = 2.*torch.abs(prev.weight).max(axis = 1)[0] # abs max of each row * 2
                range_2 = 2.*torch.abs(curr.weight).max(axis = 0)[0] # abs max of each col * 2
                S = torch.sqrt(range_1 * range_2) / range_2

                if idx in S_history:
                    prev_s = S_history[idx]
                    if torch.allclose(S, prev_s, atol=eps):
                        converged[idx-1] = True
                        continue
                    else:
                        converged[idx-1] = False

                # div S for each row
                prev.weight.data.div_(S.view(-1, 1))
                if prev.bias is not None:
                    prev.bias.div_(S)
                
                # mul S for each col
                curr.weight.mul_(S)
                    
                S_history[idx] = S
    return linear_layers

def cle_for_resmlp(model):
    for i in range(0, 24):
        todo_layer = model.blocks[i]
        linear_layers = get_linear_layers(todo_layer) # cross-channel sublayer only
        # cross_patch_cle(linear_layers[:3])
        cross_layer_equalization(linear_layers[3:])

# model = resmlp_24(pretrained=True)

# model.cpu()
# bias_dist_layer(model, 0, 23, "before")

# model.cuda()
# cle_for_resmlp(model)

# model.cpu()
# bias_dist_layer(model, 0, 23, "mid")

# model.cuda()
# resmlp_bias_absorb(model)

# model.cpu()
# bias_dist_layer(model, 0, 23, "after")