import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math

from ..utils import signed_minmax, scale_to_dyadic, dyadic_to_scale

def grad_scale(x: torch.Tensor, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def dyadic_scale(scale: torch.Tensor, mult_bit):
    m, e = scale_to_dyadic(1 / scale, mult_bit)
    d_scale = 1 / dyadic_to_scale(m, e)
    return d_scale.detach() - scale.detach() + scale

def round_pass(x: torch.Tensor):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class LinearLSQ(Module):
    def __init__(self, linear, weight_bit=8, bias_bit=32, training=True):
        super(LinearLSQ, self).__init__()
        self.has_bias = (linear.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.scale = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

        self.training = training
        # self.bquantizer = STEQuantizer(bias_bit, linear.bias.numel()) if self.has_bias else None
        self.set_param(linear)

    def __repr__(self):
        s = super(LinearLSQ, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={})".format(
            self.weight_bit, self.bias_bit, self.has_bias)
        return s

    def set_param(self, linear):
        # self.register_buffer('w_s', torch.zeros(1)) #not needed, just for analyzing purpose
        self.register_buffer('w_int', torch.zeros_like(linear.weight, requires_grad=False))
        if self.has_bias:
            self.register_buffer('b_int', torch.zeros_like(linear.bias, requires_grad=False))
        else:
            self.b_int = None
        self.linear = linear

    def set_training(self, set=True):
        self.training = set
    
    def forward(self, x, a_s):
        if self.training:
            x_q = x / a_s
            Qn, Qp = signed_minmax(self.weight_bit)
            bQn, bQp = signed_minmax(self.bias_bit)
            if self.training and self.init_state == 0:
                init_scale = (self.linear.weight.max() - self.linear.weight.min())/(Qp-Qn)
                # init_scale = 2 * self.linear.weight.abs().mean() / math.sqrt(Qp)
                self.scale.data.copy_(init_scale)
                self.init_state.fill_(1)
 
            g = 1.0 / math.sqrt(self.linear.weight.numel() * Qp)

            w_s = grad_scale(self.scale, g)
            self.w_int = round_pass((self.linear.weight / w_s).clamp(Qn, Qp))

            b_s = w_s * a_s
            self.b_int = round_pass((self.linear.bias / b_s).clamp(bQn, bQp)) if self.has_bias else None
            return F.linear(x_q, self.w_int, self.b_int) * b_s, b_s

        else:
            return F.linear(x, self.w_int, self.b_int), None

class ConvLSQ(Module):
    def __init__(self, conv, weight_bit=8, bias_bit=32, training=True):
        super(ConvLSQ, self).__init__()
        self.has_bias = (conv.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.scale = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

        self.training = training
        # self.bquantizer = STEQuantizer(bias_bit, linear.bias.numel()) if self.has_bias else None
        self.set_param(conv)

    def __repr__(self):
        s = super(ConvLSQ, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={})".format(
            self.weight_bit, self.bias_bit, self.has_bias)
        return s

    def set_param(self, conv):
        # self.register_buffer('w_s', torch.zeros(1)) #not needed, just for analyzing purpose
        self.register_buffer('w_int', torch.zeros_like(conv.weight, requires_grad=False))
        if self.has_bias:
            self.register_buffer('b_int', torch.zeros_like(conv.bias, requires_grad=False))
        else:
            self.b_int = None
        self.conv = conv

    def set_training(self, set=True):
        self.training = set
    
    def forward(self, x, a_s):
        if self.training:
            x_q = x / a_s
            Qn, Qp = signed_minmax(self.weight_bit)
            bQn, bQp = signed_minmax(self.bias_bit)
            if self.training and self.init_state == 0:
                init_scale = (self.conv.weight.max() - self.conv.weight.min())/(Qp-Qn)
                # init_scale = 2 * self.conv.weight.abs().mean() / math.sqrt(Qp)
                self.scale.data.copy_(init_scale)
                self.init_state.fill_(1)
 
            g = 1.0 / math.sqrt(self.conv.weight.numel() * Qp)

            w_s = grad_scale(self.scale, g)
            self.w_int = round_pass((self.conv.weight / w_s).clamp(Qn, Qp))

            b_s = w_s * a_s
            self.b_int = round_pass((self.conv.bias / b_s).clamp(bQn, bQp)) if self.has_bias else None
            return F.conv2d(x_q, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups) * b_s, b_s

        else:
            return F.conv2d(x, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups), None

class ActLSQ(Module):
    def __init__(self, to_bit=8, mult_bit=16, training=True):
        super(ActLSQ, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.scale = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('s', torch.ones(1, requires_grad=False))

    def __repr__(self):
        s = super(ActLSQ, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, from_fp32={self.from_fp32}, training={self.training}, ReLU_clip={self.ReLU_clip})"
        return s

    def get_scales(self):
        return [self.s]

    def set_training(self, set=True):
        self.training = set

    def forward(self, x, a_s=None):
        Qn, Qp = signed_minmax(self.to_bit)
        if self.training:
            if a_s == None:
                x_q = x
            else:
                x_q = x / a_s

            if self.training and self.init_state == 0:
                # init_scale = 2 * x.abs().mean() / math.sqrt(Qp)
                init_scale = (x.max() - x.min())/(Qp-Qn)
                self.scale.data.copy_(init_scale)
                self.init_state.fill_(1)
            
            g = 1.0 / math.sqrt(x_q.numel() * Qp)
            scale = grad_scale(self.scale, g)

            if a_s == None:
                self.s = dyadic_scale(scale, self.mult_bit) #scale#
                x_round = round_pass((x_q / self.s).clamp(Qn, Qp))
                return x_round * scale, scale
            else:
                self.s = dyadic_scale(scale / a_s, self.mult_bit) #scale / a_s#
                x_round = round_pass((x_q / self.s).clamp(Qn, Qp))
                return x_round * self.s * a_s, self.s * a_s

        else:
            x_round = round_pass((x / self.s).clamp(Qn, Qp))
            return x_round, None

class ResActLSQ(Module):
    def __init__(self, to_bit=8, mult_bit=16, training=True, to_fp32=False):
        super(ResActLSQ, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.to_fp32 = to_fp32
        self.scale = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('s', torch.ones(1, requires_grad=False))
        self.register_buffer('align_s', torch.ones(1, requires_grad=False))

    def __repr__(self):
        s = super(ResActLSQ, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, from_fp32={self.from_fp32}, training={self.training}, ReLU_clip={self.ReLU_clip})"
        return s

    def get_scales(self):
        return [self.align_s, self.s]

    def set_training(self, set=True):
        self.training = set

    def forward(self, x, a_s=None, res_x=None, res_a_s=None):
        Qn, Qp = signed_minmax(self.to_bit)
        if self.training:
            res_x_q = res_x / res_a_s
            x_q = x / a_s
            self.align_s = dyadic_scale(a_s/res_a_s, self.mult_bit)
            res_x_align = round_pass((res_x_q / self.align_s).clamp(Qn, Qp))
            mix_x_q = x_q + res_x_align

            if self.training and self.init_state == 0:
                mix_x = mix_x_q * a_s
                # init_scale = 2 * mix_x.abs().mean() / math.sqrt(Qp)
                init_scale = (mix_x.max() - mix_x.min())/(Qp-Qn)
                self.scale.data.copy_(init_scale)
                self.init_state.fill_(1)

            g = 1.0 / math.sqrt(mix_x_q.numel() * Qp)
            scale = grad_scale(self.scale, g)
            self.s = dyadic_scale(scale / a_s, self.mult_bit)
            mix_x_round = round_pass((mix_x_q / self.s).clamp(Qn, Qp))

            return mix_x_round * scale, scale

        else:
            res_x_align = round_pass((res_x / self.align_s).clamp(Qn, Qp))
            mix_x = x + res_x_align
            mix_x_round = round_pass((mix_x / self.s).clamp(Qn, Qp))
            if self.to_fp32:
                return mix_x_round*self.scale, None

            else:
                return mix_x_round, None

def set_training(model, set=True):
    for n, m in model.named_modules():
        if type(m) in [LinearLSQ, ConvLSQ, ActLSQ, ResActLSQ]:
            m.set_training(set)