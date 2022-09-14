import torch
import time
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.nn import Module, Parameter
from .utils import *
from .quantizer.func import *
from .quantizer.lsq import *
from .quantizer.observer import *

class QLinear(Module):
    def __init__(self,
            linear,
            weight_bit=8,
            bias_bit=32,
            training=True,
            regular=False,
            ):
        super(QLinear, self).__init__()
        self.has_bias = (linear.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit

        self.training = training
        self.regular = regular
        self.wquantizer = LSQWeight(weight_bit)
        # self.bquantizer = STEQuantizer(bias_bit, linear.bias.numel()) if self.has_bias else None
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
                return F.linear(input, weight=self.linear.weight, bias=self.linear.bias), None #torch.ones(1).to(self.linear.weight.device)
            else:
                with torch.no_grad():
                    return F.linear(input, weight=self.linear.weight, bias=self.linear.bias), None
        
        # On Training (inputs are dequantized values with its scale to quantize back)
        if self.training:
            if a_s == None: raise ValueError('Should not be None during QAT!')

            ### step 1: quantize back the weights
            x_int8 = torch.div(input, a_s)
            
            ### step 2: quantize weight/bias and backpropagate gradients for weight/bias, scale
            self.w_int = self.wquantizer(self.linear.weight)

            b_s = a_s * self.wquantizer.scale
            # self.b_int = self.bquantizer(self.linear.bias, b_s) if self.has_bias else None
            self.b_int = STEQuantizer.apply(self.linear.bias, b_s, self.bias_bit) if self.has_bias else None

            # step 3: perform linear operation with quantized values
            out_int32 = F.linear(x_int8, weight=self.w_int, bias=self.b_int)

            # step 4: dequantize output and return
            return out_int32 * b_s, b_s

        # On Validation (inputs are quantized values)
        else: 
            with torch.no_grad():
                return F.linear(input, weight=self.w_int, bias=self.b_int), None

class QInput(Module): # fp
    def __init__(self,
                 to_bit=8,
                 training=True,
                 regular=False,
                 ):
        super(QInput, self).__init__()
        self.to_bit = to_bit
        self.training = training
        self.regular = regular
        self.quantizer = LSQAct(to_bit)

    def __repr__(self):
        s = super(QInput, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, from_fp32={self.from_fp32}, training={self.training}, regular={self.regular})"
        return s

    def set_training(self, set=True):
        self.training = set
    
    def get_scale(self, dyadic=False):
        return [self.quantizer.scale]

    def forward(self, input, a_s=None):
        # To ensure model is connected correctly
        if self.regular:
            return input, None
        
        # On Training (inputs are dequantized values with its scale to quantize back)
        if self.training:
            org_scale = self.quantizer.scale
            x_int8 = self.quantizer(input)
            
            return x_int8 * org_scale, org_scale
        
        # On Validation (inputs are quantized values)
        else: 
            if a_s != None: raise ValueError('Should not have value during Validation!')
            
            with torch.no_grad():
                return self.quantizer(input), None

class QAct(Module):
    def __init__(self,
                 to_bit=8,
                 mult_bit=16,
                 training=True,
                 regular=False,
                 ):
        super(QAct, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.regular = regular
        # self.quantizer = DyadicLSQAct(to_bit, mult_bit)
        self.quantizer = LSQAct(to_bit)

    def __repr__(self):
        s = super(QAct, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, from_fp32={self.from_fp32}, training={self.training}, ReLU_clip={self.ReLU_clip}, regular={self.regular})"
        return s

    def set_training(self, set=True):
        self.training = set
    
    def get_scale(self):
        return self.quantizer.get_scale()

    def forward(self, input, a_s=None):
        # To ensure model is connected correctly
        if self.regular:
            return input, None
        
        # On Training (inputs are dequantized values with its scale to quantize back)
        if self.training:
            # step 1: quantize back the input if there is
            org_scale = self.quantizer.scale
            x_int8 = self.quantizer(input, a_s)

            return x_int8 * org_scale, org_scale
        
        # On Validation (inputs are quantized values)
        else: 
            if a_s != None: raise ValueError('Should not have value during Validation!')
            
            with torch.no_grad():
                # return torch.bitwise_right_shift(input.type(torch.int64) * self.mult.int(), self.shift.int()).type(torch.float), None
                return self.quantizer(input, None), None


class QResAct(Module):
    def __init__(self,
                 to_bit=8,
                 mult_bit=16,
                 training=True,
                 regular=False,
                 ):
        super(QResAct, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.regular = regular
        # self.quantizer = DyadicLSQResAct(to_bit, mult_bit)
        self.quantizer = LSQResAct(to_bit)

    def __repr__(self):
        s = super(QResAct, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, training={self.training})"
        return s

    def set_training(self, set=True):
        self.training = set
    
    def get_scale(self, dyadic=False):
        if dyadic:
            # return [(self.res_mult, self.res_shift), (self.mult, self.shift)]
            align = (self.res_mult[0].type(torch.double) / (2.0 ** self.res_shift[0]).type(torch.double)).type(torch.float)
            scale = (self.mult[0].type(torch.double) / (2.0 ** self.shift[0]).type(torch.double)).type(torch.float)
            return [(align, scale)]
        else:
            return [self.observer.scale]

    def forward(self, input, a_s=None, res_fp=None, res_a_s=None):
        # To ensure model is connected correctly
        if self.regular:
            return input + res_fp, None

        if self.training:
            if a_s == None or res_a_s == None : raise ValueError('Should not be None during QAT!')

            org_scale = self.quantizer.scale
            out_int8 = self.quantizer(input, a_s, res_fp, res_a_s)

            return out_int8*org_scale, org_scale

        # On Validation (inputs are quantized values)
        else:
            if a_s != None or res_a_s != None : raise ValueError('Should not have value during Validation!')

            with torch.no_grad():
                return self.quantizer(input, a_s, res_fp, res_a_s), None

class QResOutput(Module):
    def __init__(self,
                 to_bit=8,
                 mult_bit=16,
                 training=True,
                 regular=False,
                 ):
        super(QResOutput, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.regular = regular
        # self.quantizer = DyadicLSQResAct(to_bit, mult_bit)
        self.quantizer = LSQResAct(to_bit)

    def __repr__(self):
        s = super(QResAct, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, training={self.training})"
        return s

    def set_training(self, set=True):
        self.training = set
    
    def get_scale(self, dyadic=False):
        if dyadic:
            # return [(self.res_mult, self.res_shift), (self.mult, self.shift)]
            align = (self.res_mult[0].type(torch.double) / (2.0 ** self.res_shift[0]).type(torch.double)).type(torch.float)
            scale = (self.mult[0].type(torch.double) / (2.0 ** self.shift[0]).type(torch.double)).type(torch.float)
            return [(align, scale)]
        else:
            return [self.observer.scale]

    def forward(self, input, a_s=None, res_fp=None, res_a_s=None):
        # To ensure model is connected correctly
        if self.regular:
            return input + res_fp, None

        if self.training:
            if a_s == None or res_a_s == None : raise ValueError('Should not be None during QAT!')

            org_scale = self.quantizer.scale
            out_int8 = self.quantizer(input, a_s, res_fp, res_a_s)

            return out_int8*org_scale, org_scale

        # On Validation (inputs are quantized values)
        else:
            if a_s != None or res_a_s != None : raise ValueError('Should not have value during Validation!')

            with torch.no_grad():
                org_scale = self.quantizer.scale
                return self.quantizer(input, a_s, res_fp, res_a_s)*org_scale, None

#! Not used, hasn't correctly implemented yet
class QConv2d(Module):
    def __init__(self,
                 conv,
                 weight_bit=8,
                 bias_bit=32,
                 training=True,
                 regular=False
                 ):
        super(QConv2d, self).__init__()
        self.has_bias = (conv.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.regular = regular
        # self.observer = MovingAverageMinMaxObserver(weight_bit)

        self.wquantizer = LSQWeight(weight_bit)
        # self.bquantizer = STEQuantizer(bias_bit, linear.bias.numel()) if self.has_bias else None
        self.set_param(conv)
        
    def __repr__(self):
        s = super(QConv2d, self).__repr__()
        s = f"({s} weight_bit={self.weight_bit}, bias_bit={self.bias_bit}, has_bias={self.has_bias}, training={self.training}, regular={self.regular})"
        return s

    def set_param(self, conv):
        self.register_buffer('w_int', torch.zeros_like(conv.weight))
        if self.has_bias:
            self.register_buffer('b_int', torch.zeros_like(conv.bias, requires_grad=False))
        else:
            self.b_int = None
            
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

        # On Training (inputs are dequantized values with its scale to quantize back)
        if self.training:
            if a_s == None: raise ValueError('Should not be None during QAT!')
            ### step 1: quantize back the weights
            x_int8 = torch.div(input, a_s)
            
            ### step 2: quantize weight/bias and backpropagate gradients for weight/bias, scale
            self.w_int = self.wquantizer(self.conv.weight)

            b_s = a_s * self.wquantizer.scale
            # self.b_int = self.bquantizer(self.linear.bias, b_s) if self.has_bias else None
            self.b_int = STEQuantizer.apply(self.conv.bias, b_s, self.bias_bit) if self.has_bias else None

            # step 3: perform linear operation with quantized values
            out_int32 = F.conv2d(x_int8, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

            # step 4: dequantize output and return
            return out_int32 * b_s, b_s

        # On Validation (inputs are quantized values)
        else: 
            if a_s != None: raise ValueError('Should be None during validation!')

            with torch.no_grad():
                return F.conv2d(input, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups), None

def set_training(model, set=True):
    for n, m in model.named_modules():
        if type(m) in [QLinear, QAct, QResAct, QInput, QResOutput, QConv2d]:
            m.set_training(set)
            # print(n)