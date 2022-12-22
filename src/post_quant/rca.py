
import torch
import numpy as np
from .utils import get_quant_layers
from ..data_utils import *
from ..quantization.quantizer.lsq import init_scale_counter, set_training

def res_add_align(prev, prev_act, cur, add):
    #if add.align_int < 50: return
    factor = 10#add.align_int.data# / 50

    cur.weight.data.mul_(factor)
    if cur.bias is not None:
        cur.bias.data.mul_(factor)
    # prev.weight.data.div_(factor)
    # prev.bias.data.div_(factor)

    # prev.observer.scale.data.div_(factor)
    # prev_act.observer.scale.data.div_(factor)
    cur.observer.scale.data.mul_(factor)
    # add.observer.scale.data.div_(factor)

    # prev.observer.init_scale_counter()
    # prev_act.observer.init_scale_counter()
    # cur.observer.init_scale_counter()
    # add.observer.init_scale_counter()

# #-1, #5:  res act

# #3,  #11: prev linear
# #4,  #12: cur linear
# #5,  #13: add

def rca_for_resmlp(model):
    set_training(model, True)
    data_loader = getTrainData(dataset='imagenet', path="E:\datasets\imagenet", batch_size=16, data_percentage=0.001)
    calibrate(data_loader, model, eval=False)

    linear_layers = get_quant_layers(model.blocks) # 7 * 24
    for i in range(0, 24):
        print(linear_layers[2 + i*14][0], linear_layers[3 + i*14][0], linear_layers[4 + i*14][0], linear_layers[5 + i*14][0])
        print(linear_layers[10 + i*14][0], linear_layers[11 + i*14][0], linear_layers[12 + i*14][0], linear_layers[13 + i*14][0])
        res_add_align(linear_layers[2 + i*14][1], linear_layers[3 + i*14][1], linear_layers[4 + i*14][1], linear_layers[5 + i*14][1])
        res_add_align(linear_layers[10 + i*14][1], linear_layers[11 + i*14][1], linear_layers[12 + i*14][1], linear_layers[13 + i*14][1])
    
    # init_scale_counter(model)