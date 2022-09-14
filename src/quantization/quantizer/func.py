import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from ..utils import *

class DyadicSTEQuantizer(autograd.Function):
    @staticmethod
    def forward(self, input, mult, shift, num_bits, grad_scale=None):
        n = signed_max_bits(num_bits)
        scale = (mult.type(torch.double) / (2.0 ** shift).type(torch.double)).type(torch.float)
        x_scale = input * scale
        x_clamp = torch.clamp(x_scale, -n, n)
        x_round = torch.round(x_clamp) #torch.floor(input * scale)
        # output = torch.bitwise_right_shift(input.type(torch.int64)*mult.type(torch.int64), shift.type(torch.int64)).type(torch.float)
        
        self.save_for_backward(grad_scale)
        return x_round
    
    @staticmethod
    def backward(self, grad_output):
        grad_scale = self.saved_tensors[0]
        if grad_scale is None:
            return grad_output, None, None, None, None

        return grad_output * grad_scale, None, None, None, None

class STEQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(self, x, scale, num_bits):
        n = signed_max_bits(num_bits)
        x_scale = torch.div(x, scale)
        x_clip = F.hardtanh(x_scale, min_val=-n, max_val=n)
        x_round = torch.round(x_clip)
        self.save_for_backward(scale)
        return x_round

    @staticmethod
    def backward(self, grad_output):
        # gradient for weight
        scale = self.saved_tensors[0]
        grad_weight = grad_output / scale

        return grad_weight, None, None