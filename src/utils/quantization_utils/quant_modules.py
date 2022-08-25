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

class MinMaxObserver(nn.Module):
    def __init__(self, num_bits, remember_old=True):
        super(MinMaxObserver, self).__init__()
        self.num_bits = num_bits
        self.register_buffer('min_val', torch.tensor([float("inf")], requires_grad=False))
        self.register_buffer('max_val', torch.tensor([float("-inf")], requires_grad=False))

    def get_min_max(self):
        return self.min_val, self.max_val

    def get_quant_scale(self, sat_val):
        if any (sat_val < 0):
            raise ValueError('Saturation value must be >= 0')

        n = signed_max_bits(self.num_bits)
        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        sat_val[sat_val == 0] = n
        # scale = torch.clamp(sat_val, min=1e-8) / n
        scale = sat_val / n
        return scale

    def forward(self, x_orig):
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        
        if self.remember_old:
            min_val = torch.min(min_val_cur, self.min_val)
            max_val = torch.max(max_val_cur, self.max_val)
            self.min_val.copy_(min_val)
            self.max_val.copy_(max_val)
            abs_max_val = torch.max(min_val.abs(), max_val.abs())
        else:
            abs_max_val = torch.max(min_val_cur.abs(), max_val_cur.abs())

        return self.get_quant_scale(abs_max_val)
        
 

class QLinear(Module):
    def __init__(self,
            linear,
            weight_bit=8,
            bias_bit=32,
            training=True,
            regular=True, #! Should be False in QAT, turned on to debug QAct
            ):
        super(QLinear, self).__init__()
        self.has_bias = (linear.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.regular = regular
        self.observer = MinMaxObserver(weight_bit, remember_old=True)
        self.set_param(linear)

    def __repr__(self):
        s = super(QLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={}, regular={})".format(
            self.weight_bit, self.bias_bit, self.has_bias, self.regular)
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
        if self.regular:
            if self.training:
                return F.linear(input, weight=self.linear.weight, bias=self.linear.bias), None#self.linear(input), None
            else:
                with torch.no_grad():
                    return F.linear(input, weight=self.linear.weight, bias=self.linear.bias), None
            
        if self.training:
            if a_s == None: raise ValueError('Should not be None during QAT!')

            self.w_s = self.observer(self.linear.weight, self.weight_bit)
            
            x_int8 = linear_quant(input, a_s, self.weight_bit)
            self.w_int = linear_quant(self.linear.weight, self.w_s, self.weight_bit)

            b_s = a_s * self.w_s
            # print("scales: ", a_s, self.w_s, b_s)
            self.b_int = linear_quant(self.linear.bias, b_s, self.bias_bit) if self.has_bias else None

            out_int32 = RoundSTE.apply(F.linear(x_int8, weight=self.w_int, bias=self.b_int))
            # out_fp = linear_dequant(out_int32, b_s)
            return out_int32 * b_s, b_s

        else: #input: int8 instead
            if a_s != None: raise ValueError('Should not have value during Validation!')

            with torch.no_grad():
                return F.linear(input, weight=self.w_int, bias=self.b_int), None

class QAct(Module):
    def __init__(self,
                 from_bit=32,
                 to_bit=8,
                 mult_bit=16,
                 ema_decay=0.999, 
                 ReLU_clip=False,
                 training=True,
                 regular=False,
                 ):
        super(QAct, self).__init__()
        self.from_bit = from_bit
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.ReLU_clip = ReLU_clip
        self.regular = regular
        self.observer = MinMaxObserver(to_bit, remember_old=True)

        self.register_buffer('mult', torch.ones(1))
        self.register_buffer('shift', torch.ones(1))

    def __repr__(self):
        s = super(QAct, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, training={self.training}, ReLU_clip={self.ReLU_clip}, regular={self.regular})"
        return s

    def set_training(self, set=True):
        self.training = set

    def forward(self, input, a_s=None):
        if self.ReLU_clip:
            input = F.relu(input)

        if self.regular:
            return input, None
        
        if self.training:
            org_scale = self.observer(input)
            # print("min, max: ", self.observer.get_min_max())
            # print("OBS: ", org_scale)
   
            if a_s == None:
                # scale = (1. / org_scale.type(torch.double)).type(torch.float)
                # self.mult, self.shift = get_scale_approx(scale, self.mult_bit)
                # print("mult, shift: ", self.mult, self.shift)

                # x_int8 = DyadicQuantizeSTE.apply(input, self.mult, self.shift, self.to_bit)
                # x_int8 = LinearQuantizeSTE.apply(input, org_scale, self.to_bit, True)
                x_int8 = LinearQuantizeSTE.apply(input, org_scale, self.to_bit)
                return x_int8*org_scale, org_scale
                
            else :
                print("hi")
                scale = (a_s.type(torch.double) / org_scale.type(torch.double)).type(torch.float)
                self.mult, self.shift = get_scale_approx(scale, self.mult_bit)
                
                input_int32 = LinearQuantizeSTE.apply(input, a_s, self.from_bit, False)
                x_int8 = DyadicQuantizeSTE.apply(input_int32, self.mult, self.shift, self.to_bit)
                return x_int8 * org_scale, org_scale
        
        else: #input: int8 instead
            if a_s != None: raise ValueError('Should not have value during Validation!')

            # print(self.mult, self.shift) // good!!!!!!!!!!!!!
            with torch.no_grad():
                return torch.bitwise_right_shift(input.type(torch.int64) * self.mult.int(), self.shift.int()).type(torch.float), None

class QResAct(Module):
    def __init__(self,
                 num_bit=8,
                 prev_bit=32,
                 ema_decay=0.999, 
                 training=True,
                 regular=True
                 ):
        super(QResAct, self).__init__()
        self.num_bit = num_bit
        self.prev_bit = prev_bit
        self.training = training
        self.regular = regular
        # self.observer = MinMaxObserver(to_bit, remember_old=True)

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
        self.training = set

    def forward(self, input, wb_s=None, res_fp=None, res_a_s=None):
        if self.regular:
            return input + res_fp, None

        if self.training:
            if wb_s == None or res_a_s == None : raise ValueError('Should not be None during QAT!')

            with torch.no_grad():
                mix_fp = input + res_fp
                current_min, current_max = mix_fp.min(), mix_fp.max()
                # current_min, current_max = input.min(), input.max() 
                # self.iter_count += 1
                # self.tracked_min_biased, self.tracked_min = update_ema(self.tracked_min_biased, current_min, self.ema_decay, self.iter_count)
                # self.tracked_max_biased, self.tracked_max = update_ema(self.tracked_max_biased, current_max, self.ema_decay, self.iter_count)

                # max_abs = max(abs(self.tracked_min), abs(self.tracked_max))

                max_abs = max(abs(current_min), abs(current_max))
                org_scale = get_quant_scale(self.num_bit, max_abs)
                self.res_mult, self.res_shift = get_scale_approx(wb_s / res_a_s, self.num_bit)
                self.mult, self.shift = get_scale_approx(res_a_s / org_scale, self.num_bit)
        
            x_int32 = linear_quant(input, wb_s, self.prev_bit)
            res_x_int8 = linear_quant(res_fp, res_a_s, self.num_bit)
            res_x_int32 = DN_apply(res_x_int8, self.res_mult, self.res_shift, self.num_bit)
            
            mix_int32 = x_int32 + res_x_int32
            out_int8 = DN_apply(mix_int32, self.mult, self.shift, self.num_bit)

            out_fp  = linear_dequant(out_int8, org_scale)
            return out_fp, org_scale

        else: #input: int8 instead
            if wb_s != None or res_a_s != None : raise ValueError('Should not have value during Validation!')

            res_x_int32 = torch.bitwise_right_shift(res_fp.int() * self.res_mult.int(), self.res_shift.int())
            mix_int32 = input.int() + res_x_int32
            return torch.bitwise_right_shift(mix_int32 * self.mult.int(), self.shift.int()).float(), None

class QConv2d(Module):
    def __init__(self,
                 conv,
                 weight_bit=8,
                 bias_bit=32,
                 training=True,
                 regular=True
                 ):
        super(QConv2d, self).__init__()
        self.has_bias = (conv.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.regular = regular
        self.set_param(conv)
        
    def __repr__(self):
        s = super(QConv2d, self).__repr__()
        s = f"({s} weight_bit={self.weight_bit}, bias_bit={self.bias_bit}, has_bias={self.has_bias}, training={self.training}, regular={self.regular})"
        return s

    def set_param(self, conv):
        self.register_buffer('w_s', torch.zeros(1))
        self.register_buffer('w_int', torch.zeros_like(conv.weight))
        self.conv = conv

    def set_training(self, set=True):
        self.training = set

    def forward(self, input, a_s=None):
        if self.regular:
            return F.conv2d(input, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups), None
            # return self.conv(input), None

        if self.training:
            if a_s == None: raise ValueError('Should not have value during QAT!')

            with torch.no_grad():
                w_transform = self.conv.weight
                w_min = w_transform.min()
                w_max = w_transform.max()
                max_abs = max(abs(w_min), abs(w_max))
                self.w_s = get_quant_scale(self.weight_bit, max_abs)                

            x_int8 = linear_quant(input, a_s, self.weight_bit)
            self.w_int = linear_quant(self.conv.weight, self.w_s, self.weight_bit)
            b_s = self.w_s * a_s
            if self.has_bias:
                self.b_int = linear_quant(self.conv.bias, b_s, self.bias_bit)
            else:
                self.b_int = None

            if self.conv.bias is None:
                return (F.conv2d(x_int8, self.w_int, torch.ze))

            out_int32 = RoundSTE.apply(F.conv2d(x_int8, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups))
            out_fp = linear_dequant(out_int32, b_s)
            return out_fp, b_s
        
        else:
            if a_s != None: raise ValueError('Should be None during validation!')

            return F.conv2d(input, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups), None



def set_training(model, set=True):
    for n, m in model.named_modules():
        if type(m) in [QLinear, QAct, QResAct]:
            print(n)
            m.set_training(set)