import torch

def unsigned_max_bits(b):
    return (1 << b) - 1

def signed_max_bits(b):
    return (1 << (b-1)) - 1

def signed_minmax(b):
    n = signed_max_bits(b)
    return -n, n

def scale_to_dyadic(fp32_scale: torch.Tensor, mult_bits: int, limit_bits=False):
    m, e = torch.frexp(fp32_scale)
    m = torch.round(m * unsigned_max_bits(mult_bits)) # unsigned has 1 bit more space than signed
    new_e = mult_bits - e.type(torch.float) # right shift instead of left

    if (new_e < 0) : raise ValueError(f'Shift value is negative! e: {new_e}, org_e: {-e}')
    return (m.type(torch.int64), new_e.type(torch.int64))

def dyadic_to_scale(mult: torch.Tensor, shift: torch.Tensor):
    scale = (mult.type(torch.double) / (2.0 ** shift).type(torch.double)).type(torch.float)
    return scale

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