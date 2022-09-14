import torch
from .utils import HookHandler
from ..data_utils import *

class StdMean(object):
    def __init__(self, std=0, mean=0):
        self.std = std
        self.mean = mean

def std_mean_func(org_val, output):
    momentum = 0.99
    new_std_mean = torch.std_mean(output.detach(), dim=[0, 1], unbiased=False)
    if org_val is not None:
        return {
            "std":  org_val*momentum + new_std_mean[0]*(1 - momentum), 
            "mean": org_val*momentum + new_std_mean[1]*(1 - momentum)
        }
    else:
        return {
            "std":  new_std_mean[0], 
            "mean": new_std_mean[1]
        }

def find_layer_dist(model, model_layers):
    layers_dist = {}

    print("Creating hooks...")
    hook_handler = HookHandler()
    hook_handler.create_apply_hook(std_mean_func, layers_dist, model_layers)

    print("Loading a small piece of training data...")
    data_loader = getTrainData(dataset='imagenet', path="/mnt/disk1/imagenet", batch_size=64, data_percentage=0.001)
    
    print("Calibrating...")
    calibrate(data_loader, model)
    
    print('Removing hooks...')
    hook_handler.remove_hook()

    return layers_dist