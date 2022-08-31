import torch
import numpy as np
import torch.nn as nn
from torch.nn import Module
from decimal import Decimal


def signed_max_bits(b):
    return (1 << (b-1)) - 1

def get_scale_approx(fp32_scale: torch.Tensor, mult_bits, limit_bits=False):
    # fp64_scale = fp32_scale.type(torch.double)
    # shift_bits = get_scale_approx_shift(fp64_scale, mult_bits)
    # multiplier = get_scale_approx_mult(fp64_scale, shift_bits)
    # return multiplier.type(torch.float), shift_bits.type(torch.float)
    m, e = torch.frexp(fp32_scale)
    m = torch.round(m * signed_max_bits(mult_bits+1)) # unsigned has 1 bit more space than signed
    new_e = mult_bits - e.type(torch.float) # right shift instead of left

    if (new_e < 0) : raise ValueError(f'Shift value is negative! e: {new_e}, org_e: {-e}')
    return m, new_e

# def dyadic_approx_quant(input, mult, shift, rescale_bits):
#     bit_range = signed_max_bits(rescale_bits) # 127
#     output = input.type(torch.double) * mult.type(torch.double)
#     output = torch.round(output / (2.0 ** shift))
#     # output_int = torch.clamp(output.type(torch.float), -bit_range, bit_range)
#     # output_fp = torch.clamp(output.type(torch.float), -bit_range, bit_range)
#     return torch.clamp(output.type(torch.float), -bit_range, bit_range)

class DyadicQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mult, shift, rescale_bits):
        bit_range = signed_max_bits(rescale_bits) #127

        # output = torch.round(input * scale)
        # to_shift = (2.0 ** shift).type(torch.double)
        # output = torch.bitwise_right_shift(input.type(torch.int64)*mult.type(torch.int64), shift.type(torch.int64)).type(torch.float)
        # output = torch.round(output.type(torch.double) / to_shift).type(torch.float)
        scale = (mult.type(torch.double) / (2.0 ** shift).type(torch.double)).type(torch.float)
        output = torch.round(input * scale) #torch.floor(input * scale)
        # output = torch.bitwise_right_shift(input.type(torch.int64)*mult.type(torch.int64), shift.type(torch.int64)).type(torch.float)
        # print("Compare:")
        # print("1:")
        # print(torch.floor(input * scale))
        # print("2:")
        # print(output)
        
        
        ctx.save_for_backward(scale)
        return torch.clamp(output, -bit_range, bit_range)
    
    @staticmethod
    def backward(ctx, grad_output):
        scale, = ctx.saved_tensors
        return grad_output * scale, None, None, None


class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, num_bits):
        output = linear_quant(input, scale)
        ctx.save_for_backward(scale)

        bit_range = signed_max_bits(num_bits)
        return torch.clamp(output, -bit_range, bit_range)

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        scale, = ctx.saved_tensors
        return grad_output / scale, None, None


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

def linear_quant(input, scale): # divide
    return torch.round(input / scale)

def linear_dequant(input, scale):
    return input * scale


def get_quant_scale_tt(sat_val, num_bits):
        if any (sat_val < 0):
            raise ValueError('Saturation value must be >= 0')

        n = signed_max_bits(num_bits)
        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        sat_val[sat_val == 0] = n
        scale = torch.clamp(sat_val, min=1e-8) / n
        return scale