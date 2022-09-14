import torch
from .utils import get_linear_layers
from .find_distribution import find_layer_dist

def high_bias_absorption(linear_layers, layers_dist):
    for idx in range(1, len(linear_layers)):
        (prev_name, prev), (curr_name, curr) = linear_layers[idx-1], linear_layers[idx]
        
        gamma, beta = layers_dist[prev_name]["std"], layers_dist[prev_name]["mean"]
        
        c = (beta - 3 * torch.abs(gamma)).clamp_(min = 0)
        # print(prev_name, prev.weight.shape, prev.bias.shape)
        # print(curr_name, curr.weight.shape, curr.bias.shape)
        # print("c", c.max())
        # print()
        prev.bias.data.add_(-c)
        w_mul = curr.weight.data.matmul(c)
        curr.bias.data.add_(w_mul)

def ba_for_resmlp(model):
    model_layers = []
    for i in range(0, 24):
        todo_layer = model.blocks[i]
        model_layers.append(get_linear_layers(todo_layer, prefix=f'{i}-')[3:6]) # cross-channel sublayer only
    layers_dist = find_layer_dist(model, model_layers)

    for i in range(0, 24):
        todo_layer = model.blocks[i]
        todo_layers = get_linear_layers(todo_layer, f'{i}-')[3:6] # cross-channel sublayer only
        high_bias_absorption(todo_layers, layers_dist)