import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math

from ..utils import signed_max_bits

class LSQFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, g, Qn, Qp):
        assert scale > 0, 'scale = {}'.format(scale)
        x_scale = torch.div(x, scale)
        x_clamp = torch.clamp(x_scale, Qn, Qp)
        x_round = torch.round(x_clamp)
        # w_q = x_round * scale

        ctx.save_for_backward(x_scale)
        ctx.other = g, Qn, Qp
        return x_round

    @staticmethod
    def backward(ctx, grad_output):
        x_scale = ctx.saved_tensors[0]
        g, Qn, Qp = ctx.other
        ind_under = (x_scale < Qn).float()
        ind_over  = (x_scale > Qp).float()
        ind_mid = 1.0 - ind_under - ind_over

        # gradient for weight
        grad_weight = ind_mid * grad_output

        # gradient for scale
        grad = (ind_under * Qn + ind_over * Qp + ind_mid * (-x_scale + x_scale.round()))
        step_size = grad_output * g
        grad_scale = grad * step_size

        return grad_weight, grad_scale, None, None, None

class LSQWeight(nn.Module):
    def __init__(self, num_bits, w_num, scale_init=None):
        super(LSQWeight, self).__init__()
        n = signed_max_bits(num_bits)
        self.num_bits = num_bits
        self.Qn = -n
        self.Qp = n
        self.g = 1.0/math.sqrt(w_num * self.Qp)
        scale_init = scale_init if scale_init is not None else torch.ones(1)
        self.scale = nn.Parameter(scale_init)

    # def extra_repr(self):
    #     return 'constraint=%s' % self.constraint
    def initialize_scale(self, weight):
        with torch.no_grad():
            init_scale = (weight.max() - weight.min())/(self.Qp-self.Qn)
            # std, mean = torch.std_mean(weight)
            # init_scale = max(abs(mean - 3*std), abs(mean + 3*std)) / (2 ** (self.num_bits-1))
            # init_scale = weight.abs().mean()*2/math.sqrt(self.Qp)
            self.scale.copy_(init_scale.unsqueeze(0))

    def forward(self, x):
        return LSQFun.apply(x, self.scale, self.g, self.Qn, self.Qp)

class LSQActivation(nn.Module):
    def __init__(self, rescale_bits, a_num, scale_init=None):
        super(LSQActivation, self).__init__()
        n = signed_max_bits(rescale_bits)
        self.Qn = -n
        self.Qp = n
        self.g = 1.0/math.sqrt(a_num * self.Qp)
        scale_init = scale_init if scale_init is not None else torch.ones(1)
        self.scale = nn.Parameter(scale_init)

    def initialize_scale(self, act):
        with torch.no_grad():
            init_scale = (act.max() - act.min())/(self.Qp-self.Qn)
            self.scale.copy_(init_scale.unsqueeze(0))

    def forward(self, x):
        return LSQFun.apply(x, self.s, self.g, self.Qn, self.Qp)

# @autograd.no_grad()
# def add_lsqmodule(net, constr_weight):
#     for name, module in net.named_modules():
#         if isinstance(module, Conv2d) or isinstance(module, Linear):
#             scale_init = torch.full((1,), module.weight.abs().mean().item())
#             module.wquantizer = LsqWeight(constraint=constr_weight, scale_init=scale_init.clone())