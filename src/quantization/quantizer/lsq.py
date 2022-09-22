import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math

from ..utils import signed_max_bits, scale_to_dyadic, dyadic_to_scale

class LSQFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a_s, num_bits, dyadic):
        # assert scale > 0, 'scale = {}'.format(scale)
        if dyadic:
            mult, shift = scale_to_dyadic(1 / a_s, num_bits)
            scale = 1 / dyadic_to_scale(mult,shift)
        else:
            scale = a_s

        n = signed_max_bits(num_bits)
        x_scale = torch.div(x, scale)
        x_clamp = torch.clamp(x_scale, -n, n)
        x_round = torch.round(x_clamp)

        ctx.save_for_backward(x_scale, a_s)
        ctx.other = x.numel(), -n, n

        if dyadic:
            return x_round, (mult, shift)
        return x_round, a_s

    @staticmethod
    def backward(ctx, grad_output, grad_s):
        x_scale, a_s = ctx.saved_tensors
        x_num, Qn, Qp = ctx.other
        ind_under = (x_scale <= Qn).float()
        ind_over  = (x_scale >= Qp).float()
        ind_mid = 1.0 - ind_under - ind_over

        # gradient for weight
        grad_weight = ind_mid * (grad_output / a_s)

        # gradient for scale
        grad = (ind_under * Qn + ind_over * Qp + ind_mid * (-x_scale + torch.round(x_scale)))
        g = 1.0/math.sqrt(x_num * Qp)
        step_size = grad_output * g
        grad_scale = grad * step_size

        return grad_weight, grad_scale, None, None, None, None

class LSQWeight(nn.Module):
    def __init__(self, num_bits):
        super(LSQWeight, self).__init__()
        self.num_bits = num_bits
        n = signed_max_bits(num_bits)
        self.Qn = -n
        self.Qp = n
        scale_init = torch.tensor([float("inf")])
        self.scale = nn.Parameter(scale_init)

    def get_scales(self):
        return [self.scale]

    def forward(self, x):
        if self.scale.item() == float("inf"):
            with torch.no_grad():
                min_val, max_val = torch.aminmax(x)
                # std, mean = torch.std_mean(x.abs())
                # init_scale = max(abs(mean - 3*std), abs(mean + 3*std)) / (2 ** (self.num_bits-1))
                init_scale = (max_val - min_val)/(self.Qp-self.Qn)
                # print("scale comparison: ",  org_init_scale, init_scale)
                
                # print("Weight: ", init_scale)
                self.scale.copy_(init_scale.unsqueeze(0))

        return LSQFun.apply(x, self.scale, self.num_bits, False)

class LSQAct(nn.Module):
    def __init__(self, rescale_bits, dyadic=False):
        super(LSQAct, self).__init__()
        self.rescale_bits = rescale_bits
        n = signed_max_bits(rescale_bits)
        self.Qn = -n
        self.Qp = n
        scale_init = torch.tensor([float("inf")])
        self.scale = nn.Parameter(scale_init)
        self.dyadic = dyadic
    
    def get_scales(self):
        return [self.scale]

    def initialize_scale(self, x):
        with torch.no_grad():
            min_val, max_val = torch.aminmax(x)
            init_scale = (max_val - min_val)/(self.Qp-self.Qn)
            self.scale.copy_(init_scale.unsqueeze(0))

    def forward(self, x, a_s=None):
        if a_s is None:
            return LSQFun.apply(x, self.scale, self.rescale_bits, self.dyadic)
        return LSQFun.apply(x, self.scale/a_s, self.rescale_bits, self.dyadic)

# class LSQResAct(nn.Module):
#     def __init__(self, rescale_bits, scale_init=None):
#         super(LSQResAct, self).__init__()
#         n = signed_max_bits(rescale_bits)
#         self.Qn = -n
#         self.Qp = n
#         scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
#         self.scale = nn.Parameter(scale_init)
#         self.register_buffer('a_s', torch.ones(1, requires_grad=False))
#         self.register_buffer('res_a_s', torch.ones(1, requires_grad=False))

#     def get_scales(self):
#         return [self.scale]

#     def forward(self, x, a_s=None, res_x=None, res_a_s=None):
#         if self.scale.item() == float("inf"):
#             with torch.no_grad():
#                 mix = x + res_x
#                 min_val, max_val = torch.aminmax(mix)
#                 init_scale = (max_val - min_val)/(self.Qp-self.Qn)
#                 self.scale.copy_(init_scale.unsqueeze(0))

#         g = 1.0/math.sqrt(x.numel() * self.Qp)

#         if a_s != None or res_a_s != None:
#             self.a_s = a_s
#             self.res_a_s = res_a_s
#             mix = x + res_x
#             out_int8 = LSQFun.apply(mix, self.scale, g, self.Qn, self.Qp)
        
#         else:
#             mix = x*self.a_s + res_x*self.res_a_s
#             out_int8 = LSQFun.apply(mix, self.scale, g, self.Qn, self.Qp)
        
#         return out_int8

# class DyadicLSQFun(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, s_grad, mult, shift, g, Qn, Qp):
#         scale = dyadic_to_scale(mult, shift)
#         x_scale = x * scale
#         x_clamp = torch.clamp(x_scale, Qn, Qp)
#         x_round = torch.round(x_clamp)

#         ctx.save_for_backward(x_scale, s_grad)
#         ctx.other = g, Qn, Qp
#         return x_round

#     @staticmethod
#     def backward(ctx, grad_output):
#         x_scale, s_grad = ctx.saved_tensors
#         g, Qn, Qp = ctx.other
#         ind_under = (x_scale < Qn).float()
#         ind_over  = (x_scale > Qp).float()
#         ind_mid = 1.0 - ind_under - ind_over

#         # gradient for weight
#         grad_weight = ind_mid * ((grad_output * s_grad) if s_grad is not None else grad_output)

#         # gradient for scale
#         grad = (ind_under * Qn + ind_over * Qp + ind_mid * (-x_scale + torch.round(x_scale)))
#         step_size = grad_output * g
#         grad_scale = grad * step_size

#         if s_grad is None:
#             return grad_weight, None, None, None, None, None, None

#         return grad_weight, grad_scale, None, None, None, None, None

# class DyadicLSQAct(nn.Module):
#     def __init__(self, rescale_bits, mult_bit, scale_init=None):
#         super(DyadicLSQAct, self).__init__()
#         n = signed_max_bits(rescale_bits)
#         self.Qn = -n
#         self.Qp = n
#         self.mult_bit = mult_bit
#         scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
#         self.scale = nn.Parameter(scale_init)
#         self.register_buffer('mult', torch.ones(1, requires_grad=False))
#         self.register_buffer('shift', torch.ones(1, requires_grad=False))

#     def get_scales(self):
#         return dyadic_to_scale(self.mult, self.shift)

#     def forward(self, x, a_s=None):
#         if self.scale.item() == float("inf"):
#             assert a_s != None, 'a_s = {}'.format(a_s)
#             with torch.no_grad():
#                 min_val, max_val = torch.aminmax(x)
#                 init_scale = (max_val - min_val)/(self.Qp-self.Qn)
#                 self.scale.copy_(init_scale.unsqueeze(0))

#         if a_s != None : # On Training
#             x_int32 = torch.div(x, a_s)
#             rescale = (a_s.type(torch.double) / self.scale.type(torch.double)).type(torch.float)
#             self.mult, self.shift = scale_to_dyadic(rescale, self.mult_bit)

#         else : # On Validation
#             x_int32 = x

#         g = 1.0/math.sqrt(x_int32.numel() * self.Qp)
#         return DyadicLSQFun.apply(x_int32, None, self.mult, self.shift, g, self.Qn, self.Qp)

# class DyadicLSQResAct(nn.Module):
#     def __init__(self, rescale_bits, mult_bit, scale_init=None):
#         super(DyadicLSQResAct, self).__init__()
#         n = signed_max_bits(rescale_bits)
#         self.Qn = -n
#         self.Qp = n
#         self.mult_bit = mult_bit
#         scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
#         self.scale = nn.Parameter(scale_init)
#         self.register_buffer('res_mult', torch.ones(1, requires_grad=False))
#         self.register_buffer('res_shift', torch.ones(1, requires_grad=False))
#         self.register_buffer('mult', torch.ones(1, requires_grad=False))
#         self.register_buffer('shift', torch.ones(1, requires_grad=False))

#     def get_scales(self):
#         return [dyadic_to_scale(self.res_mult, self.res_shift), dyadic_to_scale(self.mult, self.shift)]

#     def forward(self, x, a_s=None, res_x=None, res_a_s=None):
#         if self.scale.item() == float("inf"):
#             assert a_s != None, 'a_s = {}'.format(a_s)
#             with torch.no_grad():
#                 min_val, max_val = torch.aminmax(x)
#                 init_scale = (max_val - min_val)/(self.Qp-self.Qn)
#                 self.scale.copy_(init_scale.unsqueeze(0))

#         if a_s != None and res_a_s != None: # On Training
#             x_int32 = torch.div(x, a_s)
#             res_x_int8 = torch.div(res_x, res_a_s)

#             rescale0 = res_a_s / a_s
#             rescale  = a_s / self.scale
#             self.res_mult, self.res_shift = scale_to_dyadic(rescale0, 8) # RecAccel aware: res_mult value fits into a matrix
#             self.mult, self.shift = scale_to_dyadic(rescale, self.mult_bit)

#         else : # On Validation
#             assert a_s == None and res_a_s == None, 'a_s = {}, res_a_s = {}'.format(a_s, res_a_s)
#             x_int32 = x
#             res_x_int8 = res_x

#         g = 1.0/math.sqrt(x_int32.numel() * self.Qp)
#         res_x_int32 = DyadicLSQFun.apply(res_x_int8, rescale0, self.res_mult, self.res_shift, g, self.Qn, self.Qp)
#         mix_int32 = x_int32 + res_x_int32
#         out_int8 = DyadicLSQFun.apply(mix_int32, rescale, self.mult, self.shift, g, self.Qn, self.Qp)
        
#         return out_int8


# @autograd.no_grad()
# def add_lsqmodule(net, constr_weight):
#     for name, module in net.named_modules():
#         if isinstance(module, Conv2d) or isinstance(module, Linear):
#             scale_init = torch.full((1,), module.weight.abs().mean().item())
#             module.wquantizer = LsqWeight(constraint=constr_weight, scale_init=scale_init.clone())