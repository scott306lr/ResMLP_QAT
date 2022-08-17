from re import X
import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Module, Parameter
from .quant_utils import *


def update_ema(biased_ema, value, decay, step):
    biased_ema = biased_ema * decay + (1 - decay) * value
    unbiased_ema = biased_ema / (1 - decay ** step)  # Bias correction
    return biased_ema, unbiased_ema

class QAct(Module):
    def __init__(self,
                 num_bit=8,
                 ema_decay=0.999, 
                 training=True,
                 ReLU_clip=True
                 ):
        super(QAct, self).__init__()
        self.num_bit = num_bit
        self.training = training
        self.ReLU_clip = ReLU_clip

        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max_biased', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))

        self.register_buffer('org_scale', torch.ones(1))
        self.register_buffer('mult', torch.ones(1))
        self.register_buffer('shift', torch.ones(1))

    def __repr__(self):
        s = super(QAct, self).__repr__()
        s = f"({s} weight_bit={self.weight_bit}, training={self.training}, ReLU_clip={self.ReLU_clip})"
        return s

    def set_training(self, set=True):
        self.train = set

    def forward(self, input, a_s=None):
        if self.ReLU_clip:
            input = F.relu(input)
        
        if self.training:
            with torch.no_grad():
                current_min, current_max = input.min(), input.max() 
                self.iter_count += 1
                self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data, current_min, self.ema_decay, self.iter_count)
                self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data, current_max, self.ema_decay, self.iter_count)
        
                max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
                self.scale = symmetric_linear_quantization_params(self.num_bits, max_abs)
                self.mult.data, self.shift.data = get_scale_approximation_params(a_s / self.scale, self.num_bits)
        
            x_int32 = LinearQuantizeSTE.apply(input, a_s)
            x_int8 = FloorSTE.apply((x_int32 * self.mult) / (2 ** self.shift))
            x_fp  = LinearDequantizeSTE.apply(x_int8, self.scale)

            return x_fp, self.scale
        
        else: #input: int8 instead
            with torch.no_grad():
                return FloorSTE.apply((input * self.mult) / (2 ** self.shift))

class QLinear(Module):
    def __init__(self,
                 weight_bit=8,
                 bias_bit=None,
                 full_precision_flag=False,
                 training = True
                 ):
        super(QLinear, self).__init__()
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.full_precision_flag = full_precision_flag
        self.training = training

    def __repr__(self):
        s = super(QLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, full_precision_flag={})".format(
            self.weight_bit, self.bias_bit, self.full_precision_flag)
        return s

    def set_param(self, linear):
        self.register_buffer('w_s', torch.zeros(1))
        self.register_buffer('w_int', torch.zeros_like(linear.weight))
        if linear.bias is not None:
            self.register_buffer('b_int', torch.zeros_like(linear.bias))
        else:
            self.bias = None

        self.linear = linear

    def set_training(self, set=True):
        self.training = set

    def forward(self, input, a_s=None):
        if self.training:
            with torch.no_grad():
                w_transform = self.linear.weight
                w_min = w_transform.min()
                w_max = w_transform.max()
                self.w_s = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max)
                # self.w_s = approx_scale_as_mult_and_shift(self.org_scale, self.num_bits)
                # self.mult, self.shift = get_scale_approximation_params(1. / self.org_scale, self.num_bits)
            
            self.w_int = LinearQuantizeSTE.apply(self.linear.weight, self.w_s)
            b_s = self.w_s * a_s

            if self.linear.bias is not None:
                self.b_int = LinearQuantizeSTE.apply(self.linear.bias, b_s)
            else:
                self.b_int = None
            
            x_int8 = LinearQuantizeSTE.apply(input, a_s)

            out_int32 = RoundSTE.apply(
                F.linear(x_int8, weight=self.w_int, bias=self.b_int)
            )
            out_fp = LinearDequantizeSTE.apply(out_int32, b_s)
            return out_fp, self.b_s

        else: #input: int8 instead
            with torch.no_grad():
                return F.linear(input, weight=self.w_int, bias=self.b_int)

class QResAct(Module):
    def __init__(self,
                 num_bit=8,
                 ema_decay=0.999, 
                 training=True,
                 ):
        super(QResAct, self).__init__()
        self.num_bit = num_bit
        self.training = training

        self.register_buffer('ema_decay', torch.tensor(ema_decay))
        self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max_biased', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))
        
        self.register_buffer('res_mult', torch.ones(1))
        self.register_buffer('res_shift', torch.ones(1))
        self.register_buffer('org_scale', torch.ones(1))
        self.register_buffer('mult', torch.ones(1))
        self.register_buffer('shift', torch.ones(1))

    def __repr__(self):
        s = super(QResAct, self).__repr__()
        s = f"({s} weight_bit={self.weight_bit}, training={self.training}, ReLU_clip={self.ReLU_clip})"
        return s

    def set_training(self, set=True):
        self.train = set

    def forward(self, input, wb_s=None, res_fp=None, res_a_s=None):
        if self.training:
            mix_fp = input + res_fp
            with torch.no_grad():
                self.res_mult, self.res_shift = get_scale_approximation_params(res_a_s / wb_s, self.num_bits)

                current_min, current_max = mix_fp.min(), mix_fp.max() 
                self.iter_count += 1
                self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data, current_min, self.ema_decay, self.iter_count)
                self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data, current_max, self.ema_decay, self.iter_count)
    
                max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
                self.org_scale = symmetric_linear_quantization_params(self.num_bits, max_abs)
                self.mult, self.shift = get_scale_approximation_params(res_a_s / self.org_scale, self.num_bits)
        
            x_int32 = LinearQuantizeSTE.apply(input, wb_s)
            res_x_int8 = LinearQuantizeSTE.apply(res_fp, res_a_s)
            res_x_int32 = FloorSTE.apply((res_x_int8 * self.res_mult) / (2 ** self.res_shift))

            mix_int32 = x_int32 + res_x_int32
            out_int8 = FloorSTE.apply((mix_int32 * self.mult) / (2 ** self.shift))

            out_fp  = LinearDequantizeSTE.apply(out_int8, self.org_scale)
            return out_fp, self.org_scale

        else: #input: int8 instead
            res_x_int32 = FloorSTE.apply((res_x_int8 * self.res_mult) / (2 ** self.res_shift))
            mix_int32 = input + res_x_int32

            return FloorSTE.apply((mix_int32 * self.mult) / (2 ** self.shift))

def set_training(model, set=True):
    if type(model) == QLinear or type(model) == QAct or type(model) == QResAct:
        model.set_training(set)

    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and attr != 'norm':
                set_training(mod, set)