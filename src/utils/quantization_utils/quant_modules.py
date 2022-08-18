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

        self.register_buffer('mult', torch.ones(1))
        self.register_buffer('shift', torch.ones(1))

    def __repr__(self):
        s = super(QAct, self).__repr__()
        s = f"({s} num_bit={self.num_bit}, training={self.training}, ReLU_clip={self.ReLU_clip})"
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
                self.tracked_min_biased, self.tracked_min = update_ema(self.tracked_min_biased, current_min, self.ema_decay, self.iter_count)
                self.tracked_max_biased, self.tracked_max = update_ema(self.tracked_max_biased, current_max, self.ema_decay, self.iter_count)
        
                max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
                org_scale = get_quantization_scale(self.num_bit, max_abs)
                if a_s == None:
                    self.mult, self.shift = get_scale_approximation_params(1. / org_scale, self.num_bit)
                else:
                    self.mult, self.shift = get_scale_approximation_params(a_s / org_scale, self.num_bit)

            if a_s == None:
                x_int8 = FloorSTE.apply((input * self.mult) / (2 ** self.shift))
                x_fp  = linear_dequantization(x_int8, org_scale)
                return x_fp, org_scale
                
            else :
                x_int32 = linear_quantization(input, a_s, self.num_bit)
                x_int8 = FloorSTE.apply((x_int32 * self.mult) / (2 ** self.shift))
                x_fp  = linear_dequantization(x_int8, org_scale)

            return x_fp, org_scale
        
        else: #input: int8 instead
            if a_s != None: raise ValueError('Should not have value during Validation!')

            with torch.no_grad():
                return FloorSTE.apply((input * self.mult) / (2 ** self.shift)), None

class QLinear(Module):
    def __init__(self,
            linear,
            weight_bit=8,
            bias_bit=32,
            full_precision_flag=False,
            training=True
            ):
        super(QLinear, self).__init__()
        self.has_bias = (linear.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.full_precision_flag = full_precision_flag
        self.training = training
        self.set_param(linear)

    def __repr__(self):
        s = super(QLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={}, training={})".format(
            self.weight_bit, self.bias_bit, self.has_bias, self.training)
        return s

    def set_param(self, linear):
        self.register_buffer('w_s', torch.zeros(1)) #not needed, just for analyzing purpose
        self.register_buffer('w_int', torch.zeros_like(linear.weight))
        if self.has_bias:
            self.register_buffer('b_int', torch.zeros_like(linear.bias))
        else:
            self.b_int = None

        self.linear = linear

    def set_training(self, set=True):
        self.training = set

    def forward(self, input, a_s=None):
        if self.training:
            if a_s == None: raise ValueError('Should not be None during QAT!')

            with torch.no_grad():
                w_transform = self.linear.weight
                w_min = w_transform.min()
                w_max = w_transform.max()
                max_abs = max(abs(w_min), abs(w_max))
                self.w_s = get_quantization_scale(self.weight_bit, max_abs)
                b_s = self.w_s * a_s
            
            x_int8 = linear_quantization(input, a_s, self.weight_bit)
            self.w_int = linear_quantization(self.linear.weight, self.w_s, self.weight_bit)
            if self.has_bias:
                self.b_int = linear_quantization(self.linear.bias, b_s, self.bias_bit)
            else:
                self.b_int = None

            out_int32 = F.linear(x_int8, weight=self.w_int, bias=self.b_int)
            out_fp = linear_dequantization(out_int32, b_s)
            return out_fp, b_s

        else: #input: int8 instead
            if a_s != None: raise ValueError('Should not have value during Validation!')

            with torch.no_grad():
                return F.linear(input, weight=self.w_int, bias=self.b_int), None

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
        self.register_buffer('mult', torch.ones(1))
        self.register_buffer('shift', torch.ones(1))

    def __repr__(self):
        s = super(QResAct, self).__repr__()
        s = f"({s} num_bit={self.num_bit}, training={self.training})"
        return s

    def set_training(self, set=True):
        self.train = set

    def forward(self, input, wb_s=None, res_fp=None, res_a_s=None):
        if self.training:
            if wb_s == None or res_fp == None or res_a_s == None : raise ValueError('Should not be None during QAT!')

            mix_fp = input + res_fp
            with torch.no_grad():
                self.res_mult, self.res_shift = get_scale_approximation_params(res_a_s / wb_s, self.num_bit)

                current_min, current_max = mix_fp.min(), mix_fp.max() 
                self.iter_count += 1
                self.tracked_min_biased, self.tracked_min = update_ema(self.tracked_min_biased, current_min, self.ema_decay, self.iter_count)
                self.tracked_max_biased, self.tracked_max = update_ema(self.tracked_max_biased, current_max, self.ema_decay, self.iter_count)
    
                max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
                org_scale = get_quantization_scale(self.num_bit, max_abs)
                self.mult, self.shift = get_scale_approximation_params(res_a_s / org_scale, self.num_bit)
        
            x_int32 = linear_quantization(input, wb_s, self.num_bit)
            res_x_int8 = linear_quantization(res_fp, res_a_s, self.num_bit)
            res_x_int32 = FloorSTE.apply((res_x_int8 * self.res_mult) / (2 ** self.res_shift))

            mix_int32 = x_int32 + res_x_int32
            out_int8 = FloorSTE.apply((mix_int32 * self.mult) / (2 ** self.shift))

            out_fp  = linear_dequantization(out_int8, org_scale)
            return out_fp, org_scale

        else: #input: int8 instead
            if wb_s != None or res_fp != None or res_a_s != None : raise ValueError('Should not have value during Validation!')

            res_x_int32 = FloorSTE.apply((res_x_int8 * self.res_mult) / (2 ** self.res_shift))
            mix_int32 = input + res_x_int32

            return FloorSTE.apply((mix_int32 * self.mult) / (2 ** self.shift)), None

class QConv2d(Module):
    def __init__(self,
                 conv,
                 weight_bit=8,
                 bias_bit=32,
                 training=True,
                 ):
        super(QConv2d, self).__init__()
        self.has_bias = (conv.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.set_param(conv)
        
    def __repr__(self):
        s = super(QConv2d, self).__repr__()
        s = f"({s} weight_bit={self.weight_bit}, bias_bit={self.bias_bit}, has_bias={self.has_bias}, training={self.training})"
        return s

    def set_param(self, conv):
        self.register_buffer('w_s', torch.zeros(1))
        self.register_buffer('w_int', torch.zeros_like(conv.weight))
        self.conv = conv

    def set_training(self, set=True):
        self.train = set

    def forward(self, input, a_s=None):
        if a_s == None: raise ValueError('Should not have value during QAT!')

        if self.training:
            with torch.no_grad():
                w_transform = self.conv.weight
                w_min = w_transform.min()
                w_max = w_transform.max()
                max_abs = max(abs(w_min), abs(w_max))
                self.w_s = get_quantization_scale(self.weight_bit, max_abs)                

            x_int8 = linear_quantization(input, a_s, self.weight_bit)
            self.w_int = linear_quantization(self.conv.weight, self.w_s, self.weight_bit)
            b_s = self.w_s * a_s
            if self.has_bias:
                self.b_int = linear_quantization(self.conv.bias, b_s, self.bias_bit)
            else:
                self.b_int = None

            if self.conv.bias is None:
                return (F.conv2d(x_int8, self.w_int, torch.ze))

            out_int32 = RoundSTE.apply(F.conv2d(x_int8, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups))
            out_fp = linear_dequantization(out_int32, b_s)
            return out_fp, b_s
        
        else:
            if a_s != None: raise ValueError('Should be None during validation!')

            return F.conv2d(x_int8, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)



def set_training(model, set=True):
    if type(model) in [QLinear, QAct, QResAct]:
        model.set_training(set)

    else:
        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and attr != 'norm':
                set_training(mod, set)