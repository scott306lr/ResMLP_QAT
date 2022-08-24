import torch
import torch.nn as nn
from torch.nn import Module

def get_scale_approx_shift(fp32_scale, mult_bits, limit=False):
    shift_bits = torch.log2((2 ** mult_bits - 1) / fp32_scale).floor()
    if limit: # not sure
        shift_bits = min(mult_bits, shift_bits)
    return shift_bits

def get_scale_approx_mult(fp32_scale, shift_bits):
    return (fp32_scale * (2 ** shift_bits)).floor()

def get_scale_approx(fp32_scale, mult_bits, limit=False):
    shift_bits = get_scale_approx_shift(fp32_scale, mult_bits, limit=limit)
    multiplier = get_scale_approx_mult(fp32_scale, shift_bits)
    return multiplier, shift_bits



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

def DN_apply(x, mult, shift, num_bits): # multiply
    with torch.no_grad():
        scale = mult / 2**shift
        bit_range = 2**(num_bits-1) -1 # 127
        
    return torch.clamp(RoundSTE.apply(x * scale), -bit_range, bit_range)

def DN_reverse(x, mult, shift, num_bits=None): # multiply
    with torch.no_grad():
        scale = 2**shift / mult

    return x * scale

def linear_quant(input, scale, num_bits): # divide
    bit_range = 2**(num_bits-1) - 1
    # with torch.no_grad():
    #     bit_range = 2**(num_bits-1) - 1 # 127
    #     print("input.shape:", input.shape)
    #     if len(input.shape) == 4:
    #         scale = scale.view(-1, 1, 1, 1)
    #     elif len(input.shape) == 3: # linear layer
    #         scale = transfer_fc_size(scale)
    #     else: # act layer
    #         scale = scale.view(-1)

    # return RoundSTE.apply(input / scale)
    return torch.clamp(RoundSTE.apply(input / scale), -bit_range, bit_range)

def linear_dequant(input, scale, num_bits=None):
    # if len(input.shape) == 2: # linear layer
    #     scale = transfer_fc_size(scale)
    # else: # act layer
    #     scale = scale.view(-1)

    return input * scale

def generate_scale_minmax(x, num_bits):
    current_min, current_max = x.min(), x.max()
    abs_max = max(abs(current_min), abs(current_max))

    return get_quant_scale(abs_max, num_bits)

# def transfer_fc_size(input_tensor):
#     return input_tensor.view(-1, 1, 1)

def get_quant_scale(saturation_val, num_bits: int):
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)

    # print("is_scalar, sat_val", is_scalar, sat_val)

    if any (sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    n = torch.tensor(2 ** (num_bits - 1) - 1, dtype=float)
    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = torch.clamp(sat_val, min=1e-8) / n

    # If input was scalar, return scalars
    if is_scalar:
        return scale.item()
    
    return scale

def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, num_bits, dequantize):
        output = linear_quant(input, scale, num_bits)
        if dequantize:
            output = linear_dequant(output, scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None
