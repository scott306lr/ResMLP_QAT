import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Module, Parameter
from .quant_utils import *

class MinMaxObserver(nn.Module):
    def __init__(self, num_bits, remember_old=True):
        super(MinMaxObserver, self).__init__()
        self.num_bits = num_bits
        self.remember_old = remember_old
        self.register_buffer('min_val', torch.tensor([float("inf")], requires_grad=False))
        self.register_buffer('max_val', torch.tensor([float("-inf")], requires_grad=False))
        self.register_buffer('scale', torch.tensor(torch.zeros(1), requires_grad=False))

    def get_min_max(self):
        return self.min_val, self.max_val

    def get_scale(self):
        return self.scale

    def calc_quant_scale(self):
        sat_val = torch.max(self.min_val.abs(), self.max_val.abs())
        
        n = signed_max_bits(self.num_bits)
        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        if sat_val == 0 : 
            sat_val = n 

        scale = torch.clamp(sat_val, min=1e-8) / n
        return scale

    def forward(self, x_orig):
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val, max_val = torch.aminmax(x)
        min_val, max_val = min_val.unsqueeze(0), max_val.unsqueeze(0)
        
        if self.remember_old:
            min_val = torch.min(min_val, self.min_val)
            max_val = torch.max(max_val, self.max_val)

        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self.scale = self.calc_quant_scale()
        return self.scale

class MovingAverageMinMaxObserver(nn.Module):
    def __init__(self, num_bits, remember_old=True, averaging_constant=0.01):
        super(MovingAverageMinMaxObserver, self).__init__()
        self.num_bits = num_bits
        self.remember_old = remember_old
        self.averaging_constant = averaging_constant
        self.register_buffer('min_val', torch.tensor(float("inf"), requires_grad=False))
        self.register_buffer('max_val', torch.tensor(float("-inf"), requires_grad=False))
        self.register_buffer('scale', torch.tensor(torch.zeros(1), requires_grad=False))

    def get_min_max(self):
        return self.min_val, self.max_val

    def get_scale(self):
        return self.scale

    def calc_quant_scale(self):
        sat_val = torch.max(self.min_val.abs(), self.max_val.abs())

        n = signed_max_bits(self.num_bits)
        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        if sat_val == 0 : 
            sat_val = n 

        scale = torch.clamp(sat_val, min=1e-8) / n
        return scale

    def forward(self, x_orig): # update the min and max values
        x = x_orig.detach()
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val

        if min_val.item() == float("inf") and max_val.item() == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self.scale = self.calc_quant_scale()
        return self.scale

class QLinear(Module):
    def __init__(self,
            linear,
            weight_bit=8,
            bias_bit=32,
            training=True,
            regular=False, #! Should be False in QAT, turned on to debug QAct
            ):
        super(QLinear, self).__init__()
        self.has_bias = (linear.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.regular = regular
        self.observer = MinMaxObserver(weight_bit, remember_old=True) # seems better to remember_old even for quantizing the weights?
        self.set_param(linear)

    def __repr__(self):
        s = super(QLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={}, regular={})".format(
            self.weight_bit, self.bias_bit, self.has_bias, self.regular)
        return s

    def set_param(self, linear):
        # self.register_buffer('w_s', torch.zeros(1)) #not needed, just for analyzing purpose
        self.register_buffer('w_int', torch.zeros_like(linear.weight, requires_grad=False))
        if self.has_bias:
            self.register_buffer('b_int', torch.zeros_like(linear.bias, requires_grad=False))
        else:
            self.b_int = None

        self.linear = linear

    def set_training(self, set=True):
        self.training = set

    def forward(self, input, a_s=None):
        # To ensure model is connected correctly
        if self.regular:
            if self.training:
                return F.linear(input, weight=self.linear.weight, bias=self.linear.bias), torch.ones(1).to(self.linear.weight.device)#None
            else:
                with torch.no_grad():
                    return F.linear(input, weight=self.linear.weight, bias=self.linear.bias), None
        
        # On Training (inputs are dequantized values with its scale to quantize back)
        if self.training:
            if a_s == None: raise ValueError('Should not be None during QAT!')

            # step 1: quantize back the weights
            x_int8 = input / a_s
            
            # step 2: obtain scale to quantize weight/bias
            w_s = self.observer(self.linear.weight)
            b_s = a_s * w_s
            
            # step 3: quantize weight/bias
            self.w_int = LinearQuantizeSTE.apply(self.linear.weight, w_s, self.weight_bit)
            self.b_int = LinearQuantizeSTE.apply(self.linear.bias, b_s, self.bias_bit) if self.has_bias else None

            # step 4: perform linear operation with quantized values
            out_int32 = RoundSTE.apply(F.linear(x_int8, weight=self.w_int, bias=self.b_int))

            # step 5: dequantize output and return
            return out_int32 * b_s, b_s

        # On Validation (inputs are quantized values)
        else: 
            with torch.no_grad():
                return F.linear(input, weight=self.w_int, bias=self.b_int), None


class QAct(Module):
    def __init__(self,
                 to_bit=8,
                 mult_bit=16,
                 ReLU_clip=False,
                 training=True,
                 regular=False,
                 from_fp32=False,
                 ):
        super(QAct, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.ReLU_clip = ReLU_clip
        self.regular = regular
        self.from_fp32 = from_fp32
        self.observer = MovingAverageMinMaxObserver(to_bit)

        self.register_buffer('mult', torch.ones(1, requires_grad=False))
        self.register_buffer('shift', torch.ones(1, requires_grad=False))

    def __repr__(self):
        s = super(QAct, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, from_fp32={self.from_fp32}, training={self.training}, ReLU_clip={self.ReLU_clip}, regular={self.regular})"
        return s

    def set_training(self, set=True):
        self.training = set

    def forward(self, input, a_s=None):
        if self.ReLU_clip:
            input = F.relu(input)

        # To ensure model is connected correctly
        if self.regular:
            return input, None
        
        # On Training (inputs are dequantized values with its scale to quantize back)
        if self.training:
            # step 1: quantize back the input if there is
            if a_s == None or self.from_fp32:
                if a_s != None: raise ValueError('"a_s" should be None when "from_fp32" is set to true!')
                input_int32 = input
            else:
                input_int32 = input / a_s
            
            # step 2: obtain org_scale to quantize fp32 input
            org_scale = self.observer(input)

            # step 3: rescale input to int8
            if a_s == None or self.from_fp32: # "a_s == None" means to rescale fp32 directly using org_scale, used only on the first input layer
                x_int8 = LinearQuantizeSTE.apply(input_int32, org_scale, self.to_bit)
            
            else: 
                # find real scale from org_scale to directly rescale int32 down to int8 (approximated with Dyadic Number)
                scale = (a_s.type(torch.double) / org_scale.type(torch.double)).type(torch.float)
                self.mult, self.shift = get_scale_approx(scale, self.mult_bit)
                
                x_int8 = DyadicQuantizeSTE.apply(input_int32, self.mult, self.shift, self.to_bit)

            return x_int8 * org_scale, org_scale
        
        # On Validation (inputs are quantized values)
        else: 
            if a_s != None: raise ValueError('Should not have value during Validation!')
            
            with torch.no_grad():
                # return torch.bitwise_right_shift(input.type(torch.int64) * self.mult.int(), self.shift.int()).type(torch.float), None
                
                if self.from_fp32:
                    scale = self.observer.get_scale()
                    return LinearQuantizeSTE.apply(input, scale, self.to_bit), None
                else:
                    # return torch.bitwise_right_shift(input.type(torch.int64) * self.mult.int(), self.shift.int()).type(torch.float), None
                    return DyadicQuantizeSTE.apply(input, self.mult, self.shift, self.to_bit), None


class QResAct(Module):
    def __init__(self,
                 to_bit=16,
                 mult_bit=16,
                 training=True,
                 regular=False,
                 to_fp32=False
                 ):
        super(QResAct, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.regular = regular
        self.to_fp32 = to_fp32
        self.observer = MovingAverageMinMaxObserver(to_bit)
        
        self.register_buffer('res_mult', torch.ones(1, requires_grad=False))
        self.register_buffer('res_shift', torch.ones(1, requires_grad=False))
        self.register_buffer('mult', torch.ones(1, requires_grad=False))
        self.register_buffer('shift', torch.ones(1, requires_grad=False))

    def __repr__(self):
        s = super(QResAct, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, to_fp32={self.to_fp32}, training={self.training})"
        return s

    def set_training(self, set=True):
        self.training = set

    def forward(self, input, a_s=None, res_fp=None, res_a_s=None):
        if self.regular:
            #return input + res_fp, None
            mix_fp = input + res_fp
            org_scale = self.observer(mix_fp)
            self.mult, self.shift = get_scale_approx(org_scale, self.mult_bit)
            #x_int8 = DyadicQuantizeSTE.apply(mix_fp, self.mult, self.shift, self.to_bit)
            x_int8 = LinearQuantizeSTE.apply(mix_fp, org_scale, self.to_bit)
            return x_int8 * org_scale, org_scale

        # On Training (inputs are two pairs of dequantized values with its scale to quantize back)
        if self.training:
            if a_s == None or res_a_s == None : raise ValueError('Should not be None during QAT!')

            # step 1: quantize back the inputs
            x_int32     = input / a_s
            res_x_int8  = res_fp / res_a_s
            
            # step 2: obtain org_scale to quantize fp32 input
            org_scale = self.observer(input)

            # step 3: reduction of scales to a common denominator (for RecAccel aware) (approximated with Dyadic Number)
            scale0 = (res_a_s.type(torch.double) / a_s.type(torch.double)).type(torch.float)
            scale  = (a_s.type(torch.double) / org_scale.type(torch.double)).type(torch.float)
            self.res_mult, self.res_shift = get_scale_approx(scale0, self.mult_bit)
            self.mult, self.shift = get_scale_approx(scale, self.mult_bit)

            # step 4: rescale residual input down, add up inputs, then rescale again
            res_x_int32 = DyadicQuantizeSTE.apply(res_x_int8, self.res_mult, self.res_shift, self.to_bit)
            mix_int32 = x_int32 + res_x_int32
            out_int8 = DyadicQuantizeSTE.apply(mix_int32, self.mult, self.shift, self.to_bit)

            return out_int8*org_scale, org_scale

        # On Validation (inputs are quantized values)
        else:
            if a_s != None or res_a_s != None : raise ValueError('Should not have value during Validation!')

            with torch.no_grad():
                res_x_int32 = DyadicQuantizeSTE.apply(res_fp, self.res_mult, self.res_shift, self.to_bit)
                mix_int32 = input + res_x_int32
                out = DyadicQuantizeSTE.apply(mix_int32, self.mult, self.shift, self.to_bit)
                # res_x_int32 = torch.bitwise_right_shift(res_fp.type(torch.int64) * self.res_mult.int(), self.res_shift.int())
                # mix_int32 = input.type(torch.int64) + res_x_int32
                # out = torch.bitwise_right_shift(mix_int32 * self.mult.int(), self.shift.int()).type(torch.float)
                if self.to_fp32:
                    scale = self.observer.get_scale()
                    return out * scale, scale
                else:
                    return out, None            

#! Not used, hasn't correctly implemented yet
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
            if self.conv.bias is None:
                return F.conv2d(input, self.conv.weight, torch.zeros_like(self.conv.bias),
                                self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups), None
            else:
                return F.conv2d(input, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding,
                                self.conv.dilation, self.conv.groups), None

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
            m.set_training(set)
            # print(n)
            


                
# class QFP(Module):
#     def __init__(self,
#                  training=True,
#                  regular=False,
#                  ):
#         super(QFP, self).__init__()
#         self.training = training
#         self.regular = regular

#     def __repr__(self):
#         s = super(QAct, self).__repr__()
#         s = f"({s} training={self.training}, regular={self.regular})"
#         return s

#     def set_training(self, set=True):
#         self.training = set

#     def forward(self, input, a_s=None):
#         if self.regular:
#             return input, None
        
#         if (not self.regular) or self.training:
#             if a_s == None: raise ValueError('Should have value during Training!')
#             return input / a_s, None
        
#         else: #input: int8 instead
#             if a_s != None: raise ValueError('Should not have value during Validation!')
#             return input, None