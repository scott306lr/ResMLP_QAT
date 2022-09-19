import torch

from ..quantization.utils import signed_max_bits
from .utils import HookHandler
from ..data_utils import *

class StdMean(object):
    def __init__(self, std=0, mean=0):
        self.std = std
        self.mean = mean

def std_mean_func(org_val, output):
    momentum = 0.99
    new_std_mean = torch.std_mean(output.detach(), dim=[0, 1], unbiased=False)
    if org_val is None:
        return {
            "std":  new_std_mean[0], 
            "mean": new_std_mean[1]
        }
    else:
        return {
            "std":  org_val["std"] *momentum + new_std_mean[0]*(1 - momentum), 
            "mean": org_val["mean"]*momentum + new_std_mean[1]*(1 - momentum)
        }

def find_layer_dist(model, model_layers):
    layers_dist = {}

    print("Creating hooks...")
    hook_handler = HookHandler()
    hook_handler.create_apply_hook(std_mean_func, layers_dist, model_layers)

    print("Loading a small piece of training data...")
    data_loader = getTrainData(dataset='imagenet', path="E:\datasets\imagenet", batch_size=64, data_percentage=0.001)
    
    print("Calibrating...")
    calibrate(data_loader, model)
    
    print('Removing hooks...')
    hook_handler.remove_hook()

    return layers_dist

def calc_quant_scale(min_val, max_val, bits):
        sat_val = torch.max(min_val.abs(), max_val.abs())
        n = signed_max_bits(bits)
        if sat_val == 0 : 
            sat_val = n 
        scale = torch.clamp(sat_val, min=1e-8) / n
        return scale

def scale_func(org_val, output):
    averaging_constant = 0.1
    x = output.detach().cpu()
    x = x.to(torch.float32)

    if org_val is None:
         min_val, max_val = torch.aminmax(x)
    else:
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = org_val["min_val"] + averaging_constant * (min_val_cur - org_val["min_val"])
        max_val = org_val["max_val"] + averaging_constant * (max_val_cur - org_val["max_val"])
    return {
        "min_val": min_val,
        "max_val": max_val,
        "scale"  : calc_quant_scale(min_val, max_val, 8)
    }

def find_layers_scale(model, model_layers):
    layers_scale = {}

    print("Creating hooks...")
    hook_handler = HookHandler()
    hook_handler.create_apply_hook(scale_func, layers_scale, model_layers)

    print("Loading a small piece of training data...")
    data_loader = getTrainData(dataset='imagenet', path="E:\datasets\imagenet", batch_size=32, data_percentage=0.001)
    
    print("Calibrating...")
    calibrate(data_loader, model)
    
    print('Removing hooks...')
    hook_handler.remove_hook()

    return layers_scale