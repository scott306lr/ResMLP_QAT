import torch
import torch.nn as nn
from ..utils import *

class MinMaxObserver(nn.Module):
    def __init__(self, num_bits, remember_old=True):
        super(MinMaxObserver, self).__init__()
        self.num_bits = num_bits
        self.remember_old = remember_old
        self.register_buffer('min_val', torch.tensor([float("inf")], requires_grad=False))
        self.register_buffer('max_val', torch.tensor([float("-inf")], requires_grad=False))
        self.register_buffer('scale', torch.zeros(1, requires_grad=False))

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
        self.register_buffer('min_val', torch.tensor([float("inf")], requires_grad=False))
        self.register_buffer('max_val', torch.tensor([float("-inf")], requires_grad=False))
        self.register_buffer('scale', torch.zeros(1, requires_grad=False))

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
            min_val, max_val = min_val.unsqueeze(0), max_val.unsqueeze(0)
        else:
            min_val_cur, max_val_cur = torch.aminmax(x)
            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)

        self.scale = self.calc_quant_scale()
        return self.scale