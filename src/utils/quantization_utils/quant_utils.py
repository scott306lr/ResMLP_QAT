import torch
import torch.nn as nn
from torch.nn import Module

def get_scale_approximation_shift_bits(fp32_scale, mult_bits, limit=False):
    shift_bits = torch.log2((2 ** mult_bits - 1) / fp32_scale).floor()
    if limit: # not sure
        shift_bits = min(mult_bits, shift_bits)
    return shift_bits


def get_scale_approximation_mult(fp32_scale, shift_bits):
    return (fp32_scale * (2 ** shift_bits)).floor()


def get_scale_approximation_params(fp32_scale, mult_bits, limit=False):
    shift_bits = get_scale_approximation_shift_bits(fp32_scale, mult_bits, limit=limit)
    multiplier = get_scale_approximation_mult(fp32_scale, shift_bits)
    return multiplier, shift_bits


def approx_scale_as_mult_and_shift(fp32_scale, mult_bits, limit=False):
    multiplier, shift_bits = get_scale_approximation_params(fp32_scale, mult_bits, limit=limit)
    return multiplier / (2 ** shift_bits)


def get_quantization_scale(num_bits, abs_max):
    # these computation do not require any gradients, to enforce this, we use torch.no_grad()
    with torch.no_grad():
        n = 2 ** (num_bits - 1) - 1
        scale = torch.clamp(abs_max, min=1e-8) / n
    return scale

class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y

class FloorSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y

def linear_quantization(weight, scale, num_bits):
    q_levels_per_sign = 2**(num_bits-1)
    min_level = -q_levels_per_sign
    max_level = q_levels_per_sign - 1
    return torch.clamp(RoundSTE.apply(weight / scale), min_level, max_level)

def linear_dequantization(weight, scale, num_bits=None):
    return weight * scale