import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math

from ..utils import signed_max_bits, get_scale_approx

class LSQFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, g, Qn, Qp):
        # assert scale > 0, 'scale = {}'.format(scale)
        x_scale = torch.div(x, scale)
        x_clamp = torch.clamp(x_scale, Qn, Qp)
        x_round = torch.round(x_clamp)
        # w_q = x_round * scale

        ctx.save_for_backward(x_scale, scale)
        ctx.other = g, Qn, Qp
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        x_scale, scale = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        ind_under = (x_scale < Qn).float()
        ind_over  = (x_scale > Qp).float()
        ind_mid = 1.0 - ind_under - ind_over

        # gradient for weight
        grad_weight = ind_mid * (grad_output / scale)

        # gradient for scale
        grad = (ind_under * Qn + ind_over * Qp + ind_mid * (-x_scale + torch.round(x_scale)))
        step_size = grad_output * g
        grad_scale = grad * step_size

        return grad_weight, grad_scale, None, None, None

class LSQWeight(nn.Module):
    def __init__(self, num_bits, scale_init=None):
        super(LSQWeight, self).__init__()
        n = signed_max_bits(num_bits)
        self.num_bits = num_bits
        self.Qn = -n
        self.Qp = n
        scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
        self.scale = nn.Parameter(scale_init)

    # def extra_repr(self):
    #     return 'constraint=%s' % self.constraint

    def forward(self, x):
        if self.scale.item() == float("inf"):
            with torch.no_grad():
                min_val, max_val = torch.aminmax(x)
                init_scale = (max_val - min_val)/(self.Qp-self.Qn)
                # std, mean = torch.std_mean(x)
                # init_scale = max(abs(mean - 3*std), abs(mean + 3*std)) / (2 ** (self.num_bits-1))
                # print("Weight: ", init_scale)
                self.scale.copy_(init_scale.unsqueeze(0))

        g = 1.0/math.sqrt(x.numel() * self.Qp)
        return LSQFun.apply(x, self.scale, g, self.Qn, self.Qp)

class LSQAct(nn.Module):
    def __init__(self, rescale_bits, scale_init=None):
        super(LSQAct, self).__init__()
        n = signed_max_bits(rescale_bits)
        self.Qn = -n
        self.Qp = n
        scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
        self.scale = nn.Parameter(scale_init)
    
    def forward(self, x, scale=None):
        if self.scale.item() == float("inf"):
            with torch.no_grad():
                min_val, max_val = torch.aminmax(x)
                init_scale = (max_val - min_val)/(self.Qp-self.Qn)
                # print("Act: ", init_scale)
                self.scale.copy_(init_scale.unsqueeze(0))
        
        g = 1.0/math.sqrt(x.numel() * self.Qp)
        return LSQFun.apply(x, self.scale, g, self.Qn, self.Qp)

class LSQResAct(nn.Module):
    def __init__(self, rescale_bits, scale_init=None):
        super(LSQResAct, self).__init__()
        n = signed_max_bits(rescale_bits)
        self.Qn = -n
        self.Qp = n
        scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
        self.scale = nn.Parameter(scale_init)

    # def get_scale(self):
    #     return (self.mult, self.shift)

    def forward(self, x, a_s=None, res_x=None, res_a_s=None):
        mix = x + res_x
        if self.scale.item() == float("inf"):
            with torch.no_grad():
                min_val, max_val = torch.aminmax(mix)
                init_scale = (max_val - min_val)/(self.Qp-self.Qn)
                self.scale.copy_(init_scale.unsqueeze(0))

        g = 1.0/math.sqrt(x.numel() * self.Qp)
        out_int8 = LSQFun.apply(mix, self.scale, g, self.Qn, self.Qp)
        
        return out_int8

class DyadicLSQFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s_grad, mult, shift, g, Qn, Qp):
        scale = (mult.type(torch.double) / (2.0 ** shift).type(torch.double)).type(torch.float)
        x_scale = x * scale
        x_clamp = torch.clamp(x_scale, Qn, Qp)
        x_round = torch.round(x_clamp)

        ctx.save_for_backward(x_scale, s_grad)
        ctx.other = g, Qn, Qp
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        x_scale, s_grad = ctx.saved_tensors
        g, Qn, Qp = ctx.other
        ind_under = (x_scale < Qn).float()
        ind_over  = (x_scale > Qp).float()
        ind_mid = 1.0 - ind_under - ind_over

        # gradient for weight
        grad_weight = ind_mid * ((grad_output * s_grad) if s_grad is not None else grad_output)

        # gradient for scale
        grad = (ind_under * Qn + ind_over * Qp + ind_mid * (-x_scale + torch.round(x_scale)))
        step_size = grad_output * g
        grad_scale = grad * step_size

        if s_grad is None:
            return grad_weight, None, None, None, None, None, None

        return grad_weight, grad_scale, None, None, None, None, None

class DyadicLSQAct(nn.Module):
    def __init__(self, rescale_bits, mult_bit, scale_init=None):
        super(DyadicLSQAct, self).__init__()
        n = signed_max_bits(rescale_bits)
        self.Qn = -n
        self.Qp = n
        self.mult_bit = mult_bit
        scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
        self.scale = nn.Parameter(scale_init)
        self.register_buffer('mult', torch.ones(1, requires_grad=False))
        self.register_buffer('shift', torch.ones(1, requires_grad=False))

    def get_scale(self):
        return (self.mult, self.shift)

    def forward(self, x, a_s=None):
        if self.scale.item() == float("inf"):
            assert a_s != None, 'a_s = {}'.format(a_s)
            with torch.no_grad():
                min_val, max_val = torch.aminmax(x)
                init_scale = (max_val - min_val)/(self.Qp-self.Qn)
                self.scale.copy_(init_scale.unsqueeze(0))

        if a_s != None : # On Training
            x_int32 = torch.div(x, a_s)
            rescale = (a_s.type(torch.double) / self.scale.type(torch.double)).type(torch.float)
            self.mult, self.shift = get_scale_approx(rescale, self.mult_bit)

        else : # On Validation
            x_int32 = x

        g = 1.0/math.sqrt(x_int32.numel() * self.Qp)
        return DyadicLSQFun.apply(x_int32, None, self.mult, self.shift, g, self.Qn, self.Qp)

class DyadicLSQResAct(nn.Module):
    def __init__(self, rescale_bits, mult_bit, scale_init=None):
        super(DyadicLSQResAct, self).__init__()
        n = signed_max_bits(rescale_bits)
        self.Qn = -n
        self.Qp = n
        self.mult_bit = mult_bit
        scale_init = scale_init if scale_init is not None else torch.tensor([float("inf")])
        self.scale = nn.Parameter(scale_init)
        self.register_buffer('mult', torch.ones(1, requires_grad=False))
        self.register_buffer('shift', torch.ones(1, requires_grad=False))

    def get_scale(self):
        return (self.mult, self.shift)

    def forward(self, x, a_s=None, res_x=None, res_a_s=None):
        if self.scale.item() == float("inf"):
            assert a_s != None, 'a_s = {}'.format(a_s)
            with torch.no_grad():
                min_val, max_val = torch.aminmax(x)
                init_scale = (max_val - min_val)/(self.Qp-self.Qn)
                self.scale.copy_(init_scale.unsqueeze(0))

        if a_s != None and res_a_s != None: # On Training
            x_int32 = torch.div(x, a_s)
            res_x_int8 = torch.div(res_x, res_a_s)

            rescale0 = (res_a_s.type(torch.double) / a_s.type(torch.double)).type(torch.float)
            rescale  = (a_s.type(torch.double) / self.scale.type(torch.double)).type(torch.float)
            self.res_mult, self.res_shift = get_scale_approx(rescale0, 8) # RecAccel aware: res_mult value fits into a matrix
            self.mult, self.shift = get_scale_approx(rescale, self.mult_bit)

        else : # On Validation
            assert a_s == None and res_a_s == None, 'a_s = {}, res_a_s = {}'.format(a_s, res_a_s)
            x_int32 = x
            res_x_int8 = res_x

        g = 1.0/math.sqrt(x_int32.numel() * self.Qp)
        res_x_int32 = DyadicLSQFun.apply(res_x_int8, rescale0, self.res_mult, self.res_shift, g, self.Qn, self.Qp)
        mix_int32 = x_int32 + res_x_int32
        out_int8 = DyadicLSQFun.apply(mix_int32, rescale, self.mult, self.shift, g, self.Qn, self.Qp)
        
        return out_int8
# @autograd.no_grad()
# def add_lsqmodule(net, constr_weight):
#     for name, module in net.named_modules():
#         if isinstance(module, Conv2d) or isinstance(module, Linear):
#             scale_init = torch.full((1,), module.weight.abs().mean().item())
#             module.wquantizer = LsqWeight(constraint=constr_weight, scale_init=scale_init.clone())