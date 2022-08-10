import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from .utils import simulate_input
from ..post_quant.utils import HookHandler, get_linear_layers 

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
            else: raise NameError(f'Type:{type} not found')

            data.append(val.detach().numpy().flatten())
            labels.append(n)

    my_boxplot(data, labels, name, ax, total=len(labels), sep_interval=(len(labels) // (end-start+1)))

def flat_act_func(org_val, output):
    return output.detach().numpy().flatten()

def act_dist(model, start, end, show_layers=None, name="Activation Distribution from Gamma_1/Gamma_2 for Each Layer", ax=None):
    model_layers = []
    for i in range(start, end+1):
        todo_layer = model.blocks[i] # todo_layer = getattr(model, f"layer{i}")
        model_layers.append(get_linear_layers(todo_layer, specify_names=show_layers, prefix=f"{i}-"))

    activations = {}
    hook_handler = HookHandler()
    hook_handler.create_apply_hook(flat_act_func, activations, model_layers)
    model(simulate_input())
    hook_handler.remove_hook()
    
    data = []
    labels = []
    for layer in model_layers:
        for n, m in layer:
            data.append(activations[n])
            labels.append(n)
  
    my_boxplot(data, labels, name, ax, total=len(labels), sep_interval=(len(labels) // (end-start+1)))

def act_dist(model, start, end, show_layers=None, name="Activation Distribution from Gamma_1/Gamma_2 for Each Layer", ax=None):
    model_layers = []
    for i in range(start, end+1):
        todo_layer = model.blocks[i]
        model_layers.append(get_linear_layers(todo_layer, specify_names=show_layers, prefix=f"{i}-"))

    activations = {}
    hook_handler = HookHandler()
    hook_handler.create_apply_hook(flat_act_func, activations, model_layers)
    model(simulate_input())
    hook_handler.remove_hook()
    
    data = []
    labels = []
    for layer in model_layers:
        for n, m in layer:
            data.append(activations[n])
            labels.append(n)
  
    my_boxplot(data, labels, name, ax, total=len(labels), sep_interval=(len(labels) // (end-start+1)))