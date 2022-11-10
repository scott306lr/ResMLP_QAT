from ..quantization.data_utils import calibrate, getTrainData
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

from ..post_quant.find_distribution import find_layer_dist, find_layers_scale
from .utils import simulate_input
from ..post_quant.utils import HookHandler, get_linear_layers 

def add_value_labels(ax, skip_cnt, precision=2):
    line = ax.lines[0]
    skip = 0
    for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
        if skip % skip_cnt == 0:
            label = format(y_value, f'.{precision}f')
            ax.annotate(label, (x_value, y_value), xytext=(0, 5), fontsize=14, 
                textcoords="offset points", ha='center', va='bottom')
            skip = 1
        else: 
            skip += 1

def my_boxplot(data, labels, name, ax, total=None, sep_interval=None):
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(20, 5))
        
    ax.boxplot(data, labels=labels)
    ymin, ymax = ax.get_ylim()
    ax.set_title(name, size=30)
    ax.tick_params(labelrotation=30)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)

    if sep_interval is not None:
        ax.vlines(x=[ x+0.5 for x in range(sep_interval, total, sep_interval) ], ymin=ymin, ymax=ymax, ls='--', lw=1, color='b')

def layer_dist(model, start, end, show_layers=None, type="weight", name="Distribution of Each Layer", ax=None):
    model_layers = []
    for i in range(start, end+1):
        todo_layer = model.blocks[i]
        model_layers.append(get_linear_layers(todo_layer, specify_names=show_layers, prefix=f"{i}-"))
    
    data = []
    labels = []
    for layer in model_layers:
        for n, m in layer:
            if type == "weight":
                val = m.weight
            elif type == "bias":
                val = m.bias
                if val is None:
                    continue
            else: raise NameError(f'Type:{type} not found')

            data.append(val.detach().numpy().flatten())
            labels.append(n)

    my_boxplot(data, labels, name, ax, total=len(labels), sep_interval=(len(labels) // (end-start+1)))

def flat_act_func(org_val, output):
    return output.cpu().detach().numpy().flatten()

def act_dist(model, start, end, show_layers=None, name="Activation Distribution from Gamma_1/Gamma_2 for Each Layer", ax=None, real_sim=False):
    model_layers = []
    model.eval()
    for i in range(start, end+1):
        todo_layer = model.blocks[i]
        model_layers.append(get_linear_layers(todo_layer, specify_names=show_layers, prefix=f"{i}-"))

    activations = {}
    hook_handler = HookHandler()
    hook_handler.create_apply_hook(flat_act_func, activations, model_layers)
    if real_sim:
        print("Loading a small piece of training data...")
        data_loader = getTrainData(dataset='imagenet', path="E:\datasets\imagenet", batch_size=64, data_percentage=0.001)
        print("Calibrating...")
        calibrate(data_loader, model)
    else:
        model(simulate_input().cuda())

    hook_handler.remove_hook()
    
    data = []
    labels = []
    for layer in model_layers:
        for n, m in layer:
            data.append(activations[n])
            labels.append(n)
  
    my_boxplot(data, labels, name, ax, total=len(labels), sep_interval=(len(labels) // (end-start+1)))

def scale_plot(model, start, end, show_layers=None, name="Activation Distribution from Gamma_1/Gamma_2 for Each Layer", ax=None):
    model_layers = []
    for i in range(start, 24):
        todo_layer = model.blocks[i]
        model_layers.append(get_linear_layers(todo_layer, specify_names=show_layers, prefix=f'{i}-')) # cross-channel sublayer only
    
    layers_scale = find_layers_scale(model, model_layers)
    
    data = []
    labels = []
    for layer in model_layers:
        for n, m in layer:
            data.append(layers_scale[n]["scale"])
            labels.append(n)
    

    ax.set_title(name, size=30)
    ax.tick_params(labelrotation=30)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(16)
    ax.plot(labels, data)

    return labels, data
    # my_boxplot(data, labels, name, ax, total=len(labels), sep_interval=(len(labels) // (end-start+1)))