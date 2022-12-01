import torch
import numpy as np
from .utils import get_linear_layers

def cross_layer_equalization(linear_layer_pairs):
    eps = 1e-8
    converged = [False] * (len(linear_layer_pairs))
    with torch.no_grad(): 
        while not np.all(converged):
            for idx, linear_pair in enumerate(linear_layer_pairs):
                (n1, m1), (n2, m2) = linear_pair
                range_1 = 2.*torch.abs(m1.weight).max(axis = 1)[0] # abs max of each row * 2
                range_2 = 2.*torch.abs(m2.weight).max(axis = 0)[0] # abs max of each col * 2
                S = torch.sqrt(range_1 * range_2) / range_2

                # check convergence
                if torch.allclose(S, torch.ones_like(S), atol=eps):
                    converged[idx] = True
                    continue
                else:
                    converged[idx] = False

                m1.weight.data.div_(S.view(-1, 1))
                if m1.bias is not None:
                    m1.bias.div_(S)
                m2.weight.mul_(S)

    return linear_layer_pairs

def cle_for_resmlp(model):
    # todo_layer = model.blocks[0]
    linear_layers = get_linear_layers(model.blocks) # 7 * 24
    for i in range(0, 24):
        cross_layer_equalization([
            (linear_layers[3 + i*7], linear_layers[4 + i*7]),
            (linear_layers[4 + i*7], linear_layers[5 + i*7]),
            (linear_layers[5 + i*7], linear_layers[6 + i*7]),
        ])


def cle_for_resmlp_v3(model):
    # todo_layer = model.blocks[0]
    linear_layers = get_linear_layers(model.blocks) # 7 * 24
    for i in range(0, 24):
        cross_layer_equalization([
            (linear_layers[2 + i*4], linear_layers[3 + i*4]),
        ])

def cle_for_resmlp_v4(model):
    # todo_layer = model.blocks[0]
    linear_layers = get_linear_layers(model.blocks) # 7 * 24
    for i in range(0, 24):
        # print(linear_layers)
        cross_layer_equalization([
            (linear_layers[2 + i*6], linear_layers[3 + i*6]),
            (linear_layers[3 + i*6], linear_layers[4 + i*6]),
            (linear_layers[4 + i*6], linear_layers[5 + i*6]),
        ])

#! Need an additional Linear layer after conv, for it to work on ResMLP
# def res_cle(left_layers, right_layers): 
#     eps=1e-8
#     converged = False
#     with torch.no_grad(): 
#         while not converged:
#             range_1s = []
#             range_2s = []
#             for n, m in left_layers:
#                 range_1s.append( 2.*torch.abs(m.weight).max(axis = 1)[0].unsqueeze(0) )
#             for n, m in right_layers:
#                 range_2s.append( 2.*torch.abs(m.weight).max(axis = 0)[0].unsqueeze(0)  )
#             range_1s = torch.cat(range_1s, axis=0)
#             range_2s = torch.cat(range_2s, axis=0)

#             # find the largest ranges of each channels on both sides
#             range_1 = range_1s.max(axis = 0)[0] # abs max of each row * 2
#             range_2 = range_2s.max(axis = 0)[0] # abs max of each col * 2
            
#             # check convergence
#             S = torch.sqrt(range_1 * range_2) / range_2
#             if torch.allclose(S, torch.ones_like(S), atol=eps):
#                 converged = True
#                 continue
#             else:
#                 converged = False

#             # the res layers have no bias to scale
#             for n, m in left_layers:
#                 m.weight.data.div_(S.view(-1, 1))
            
#             for n, m in right_layers:
#                 m.weight.mul_(S)

#     return left_layers, right_layers

# #! Need an additional Linear layer after conv, for it to work on ResMLP
# def res_cle_for_resmlp(model):
#     todo_layer = model.blocks[0]
#     linear_layers = get_linear_layers(model.blocks) # 7 * 24

#     l_layers = []
#     r_layers = []
#     for i in range(0, 23):
#         l_layers.append(linear_layers[2 + i*7])
#         r_layers.append(linear_layers[3 + i*7])
#         l_layers.append(linear_layers[6 + i*7])
#         r_layers.append(linear_layers[7 + i*7])
#     l_layers.append(linear_layers[2 + 23*7])
#     r_layers.append(linear_layers[3 + 23*7])
#     l_layers.append(linear_layers[6 + 23*7])
#     r_layers.append(('24-norm', model.norm))
#     res_cle(l_layers, r_layers)