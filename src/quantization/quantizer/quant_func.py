import torch
import math

# bit range calculation functions
def unsigned_max_bits(b):
    return (1 << b) - 1

def signed_max_bits(b):
    return (1 << (b-1)) - 1

def signed_minmax(b):
    n = signed_max_bits(b)
    return -n, n

# STE (Straight Through Estimator)
def round_pass(x: torch.Tensor):
    y = (x).round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

# Apprximating scale to dyadic value  fp32 -> int8 / 2^int8  (s ~= m >> e)
def scale_to_dyadic(fp32_scale: torch.Tensor, mult_bits: int, limit_bits=False):
    m, e = torch.frexp(fp32_scale)
    m = torch.round(m * unsigned_max_bits(mult_bits)) # unsigned has 1 bit more space than signed
    new_e = - e.type(torch.float) + mult_bits  # right shift instead of left

    if (new_e < 0) : raise ValueError(f'Shift value is negative! e: {new_e}, org_e: {-e}')
    return (m.type(torch.int64), new_e.type(torch.int64))

def dyadic_to_scale(mult: torch.Tensor, shift: torch.Tensor):
    scale = (mult.type(torch.double) / (2.0 ** shift).type(torch.double)).type(torch.float)
    return scale

def dyadic_scale(scale: torch.Tensor, mult_bit):
    m, e = scale_to_dyadic(1 / scale, mult_bit)
    d_scale = 1 / dyadic_to_scale(m, e)
    return d_scale.detach() - scale.detach() + scale, (m, e)

# Give scale a additional gradient (used in LSQ)
def grad_scale(x: torch.Tensor, scale: torch.Tensor):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def get_scale_func(mode, Qn, Qp):
    if mode == "minmax":
        def observer(x: torch.Tensor):
            y = x.detach().abs()
            return torch.max(y) / Qp
        return observer

    elif mode == "lsq":
        def observer(x: torch.Tensor):
            y = x.detach().abs()
            mean = torch.mean(y[y.nonzero(as_tuple=True)])
            return 2*mean / math.sqrt(Qp)
        return observer

    elif mode == "lsq+":
        def observer(x: torch.Tensor):
            y = x.detach().abs()
            std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
            return torch.max(torch.abs(mean - 3*std), torch.abs(mean + 3*std))/Qp
        return observer

    elif mode == "lab":
        def observer(x: torch.Tensor):
            y = x.detach()
            std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
            return torch.max(torch.abs(mean - 4*std), torch.abs(mean + 4*std))/Qp
        return observer

    else:
        raise NotImplementedError

#Currently NOT used, use round_pass instead
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        output = torch.round(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input