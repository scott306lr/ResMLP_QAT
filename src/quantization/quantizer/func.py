import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from ..utils import *

def linear_quantize(x, a_s, num_bits):
    n = signed_max_bits(num_bits)
    x_scale = torch.div(x, a_s)
    x_clamp = torch.clamp(x_scale, -n, n)
    x_round = torch.round(x_clamp)
    return x_round

def dyadic_quant(x, mult, shift, num_bits):
    n = signed_max_bits(num_bits)
    scale = dyadic_to_scale(mult,shift)
    x_scale = x * scale
    x_clamp = torch.clamp(x_scale, -n, n)
    x_round = torch.round(x_clamp)
    return x_round

class STEQuantizer(autograd.Function):
    @staticmethod
    def forward(ctx, x, a_s, num_bits, dyadic):
        n = signed_max_bits(num_bits)
        ctx.save_for_backward(a_s)
        ctx.other = -n, n
        
        if dyadic:
            mult, shift = scale_to_dyadic(1 / a_s, num_bits)
            scale = 1 / dyadic_to_scale(mult,shift)
        else:
            scale = a_s

        x_scale = torch.div(x, scale)
        x_clamp = torch.clamp(x_scale, -n, n)
        x_round = torch.round(x_clamp) #torch.floor(input * scale)
        # output = torch.bitwise_right_shift(input.type(torch.int64)*mult.type(torch.int64), shift.type(torch.int64)).type(torch.float)
        
        if dyadic:
            return x_round, (mult, shift)
        
        return x_round, a_s
    
    @staticmethod
    def backward(ctx, grad_output, grad_s):
        a_s = ctx.saved_tensors[0]
        Qn, Qp = ctx.other
        return torch.clamp(grad_output / a_s, Qn, Qp), None, None, None

# class STEQuantizer(torch.autograd.Function):
#     @staticmethod
#     def forward(self, x, scale, num_bits):
#         n = signed_max_bits(num_bits)
#         x_scale = torch.div(x, scale)
#         x_clip = F.hardtanh(x_scale, min_val=-n, max_val=n)
#         x_round = torch.round(x_clip)
#         self.save_for_backward(scale)
#         return x_round

#     @staticmethod
#     def backward(self, grad_output):
#         # gradient for weight
#         scale = self.saved_tensors[0]
#         grad_weight = grad_output / scale

#         return grad_weight, None, None