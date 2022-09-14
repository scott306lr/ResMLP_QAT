import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module
from decimal import Decimal

def unsigned_max_bits(b):
    return (1 << b) - 1

def signed_max_bits(b):
    return (1 << (b-1)) - 1

def get_scale_approx(fp32_scale: torch.Tensor, mult_bits, limit_bits=False):
    m, e = torch.frexp(fp32_scale)
    m = torch.round(m * signed_max_bits(mult_bits+1)) # unsigned has 1 bit more space than signed
    new_e = mult_bits - e.type(torch.float) # right shift instead of left

    if (new_e < 0) : raise ValueError(f'Shift value is negative! e: {new_e}, org_e: {-e}')
    return m, new_e

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        output = torch.floor(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input