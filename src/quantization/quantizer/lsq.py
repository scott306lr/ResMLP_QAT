import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math

from ..utils import signed_minmax, scale_to_dyadic, dyadic_to_scale

def grad_scale(x: torch.Tensor, scale: torch.Tensor):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

# def replace_grad_scale(x: torch.Tensor, scale: torch.Tensor):
#     y = x
#     y_grad = x * scale
#     return y.detach() - scale.detach() + scale

def dyadic_scale(scale: torch.Tensor, mult_bit):
    m, e = scale_to_dyadic(1 / scale, mult_bit)
    d_scale = 1 / dyadic_to_scale(m, e)
    return d_scale.detach() - scale.detach() + scale, (m, e)

def round_pass(x: torch.Tensor):
    y = x.round()#(x + 0.5).round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

class LinearLSQ(Module):
    def __init__(self, linear, weight_bit=8, bias_bit=32, training=True):
        super(LinearLSQ, self).__init__()
        self.has_bias = (linear.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.set_param(linear)
        self.scale = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def __repr__(self):
        s = super(LinearLSQ, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={})".format(
            self.weight_bit, self.bias_bit, self.has_bias)
        return s

    def set_param(self, linear):
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
            Qn, Qp = signed_minmax(self.weight_bit)
            bQn, bQp = signed_minmax(self.bias_bit)
            g = 1.0 / math.sqrt(torch.count_nonzero(self.linear.weight) * Qp)
            # requant inputs
            x_q = x / a_s
            
            # initialize scale on first input
            if self.init_state == 0:
                y = self.linear.weight.abs()
                std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
                # init_scale = torch.max(torch.abs(mean - 3*std), torch.abs(mean + 3*std))/Qp
                init_scale = 2*mean / math.sqrt(Qp)
                self.scale.data.copy_(init_scale)
                self.init_state.fill_(1)

            # gives scale a lsq gradient
            w_s = grad_scale(self.scale, g)
            b_s = w_s * a_s

            # print("Linear: ", self.linear.weight, self.linear.bias)
            # quantize parameters, then calculate
            self.w_int = round_pass((self.linear.weight / w_s).clamp(Qn, Qp))
            self.b_int = round_pass((self.linear.bias / b_s).clamp(bQn, bQp)) if self.has_bias else None
            return F.linear(x_q, self.w_int, self.b_int) * b_s, b_s

        else:
            return F.linear(x, self.w_int, self.b_int), None

class LinearBNLSQ(Module):
    def __init__(self, bn, weight_bit=8, bias_bit=32, training=True, batch_init=20):
        super(LinearBNLSQ, self).__init__()
        self.has_bias = (bn.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.set_param(bn)
        self.scale = Parameter(torch.Tensor(1))
        self.batch_init = batch_init
        self.register_buffer('init_state', torch.zeros(1))

    def __repr__(self):
        s = super(LinearBNLSQ, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={})".format(
            self.weight_bit, self.bias_bit, self.has_bias)
        return s

    def set_param(self, bn):
        self.register_buffer('w_int', torch.zeros_like(bn.weight, requires_grad=False))
        if self.has_bias:
            self.register_buffer('b_int', torch.zeros_like(bn.bias, requires_grad=False))
        else:
            self.b_int = None
        self.bn = bn

    def set_training(self, set=True):
        self.training = set
    
    def forward(self, x, a_s):
        if self.training:
            Qn, Qp = signed_minmax(self.weight_bit)
            bQn, bQp = signed_minmax(self.bias_bit)
            g = 1.0 / math.sqrt(torch.count_nonzero(self.bn.weight) * Qp)
            # requant inputs
            x_q = x / a_s

            batch_mean = torch.mean(x, dim=(0, 1))
            batch_var = torch.var(x, dim=(0, 1))

            # update mean and variance in running stats
            self.bn.running_mean = self.bn.running_mean * (1 - self.bn.momentum) +  batch_mean * self.bn.momentum
            self.bn.running_var = self.bn.running_var * (1 - self.bn.momentum) + batch_mean * self.bn.momentum

            output_factor = self.bn.weight / torch.sqrt(batch_var + self.bn.eps)
            weight = torch.diag(output_factor)
            bias = - output_factor * batch_mean + self.bn.bias

            # initialize scale on first input
            if self.init_state == 0:
                y = weight.abs()
                std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
                init_scale = 2*mean / math.sqrt(Qp)
                # init_scale = torch.max(torch.abs(mean - 3*std), torch.abs(mean + 3*std))/Qp
                self.scale.data = init_scale
                self.init_state += 1

            elif self.init_state < self.batch_init:
                y = weight.abs()
                std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
                init_scale = 2*mean / math.sqrt(Qp)
                #init_scale = torch.max(torch.abs(mean - 3*std), torch.abs(mean + 3*std))/Qp
                self.scale.data = 0.9*self.scale.data + 0.1*init_scale
                self.init_state += 1

            # gives scale a lsq gradient
            w_s = grad_scale(self.scale, g)
            b_s = w_s * a_s

            # quantize parameters, then calculate
            self.w_int = round_pass((weight / w_s).clamp(Qn, Qp))
            self.b_int = round_pass((bias / b_s).clamp(bQn, bQp)) if self.has_bias else None
            # print("x: ", x.shape)
            # print("BN: ", self.w_int.shape, self.b_int.shape)
            return F.linear(x_q, self.w_int, self.b_int) * b_s, b_s

        else:
            return F.linear(x, self.w_int, self.b_int), None

class ConvLSQ(Module):
    def __init__(self, conv, weight_bit=8, bias_bit=32, training=True):
        super(ConvLSQ, self).__init__()
        self.has_bias = (conv.bias is not None)
        self.weight_bit = weight_bit
        self.bias_bit = bias_bit
        self.training = training
        self.set_param(conv)
        self.scale = Parameter(torch.Tensor(1))
        self.register_buffer('init_state', torch.zeros(1))

    def __repr__(self):
        s = super(ConvLSQ, self).__repr__()
        s = "(" + s + " weight_bit={}, bias_bit={}, has_bias={})".format(
            self.weight_bit, self.bias_bit, self.has_bias)
        return s

    def set_param(self, conv):
        self.register_buffer('w_int', torch.zeros_like(conv.weight, requires_grad=False))
        if self.has_bias:
            self.register_buffer('b_int', torch.zeros_like(conv.bias, requires_grad=False))
        else:
            self.b_int = None
        self.conv = conv

    def set_training(self, set=True):
        self.training = set

    def reset_init_state(self):
        self.init_state.fill_(0)
    
    def forward(self, x, a_s):
        if self.training:
            Qn, Qp = signed_minmax(self.weight_bit)
            bQn, bQp = signed_minmax(self.bias_bit)
            g = 1.0 / math.sqrt(torch.count_nonzero(self.conv.weight) * Qp)
            # requant inputs
            x_q = x / a_s

            # initialize scale on first input
            if self.init_state == 0:
                y = self.conv.weight.abs()
                std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
                init_scale = 2*mean / math.sqrt(Qp)
                
                # init_scale = torch.max(torch.abs(mean - 3*std), torch.abs(mean + 3*std))/Qp
                self.scale.data.copy_(init_scale)   
                self.init_state.fill_(1)

            # gives scale a lsq gradient
            w_s = grad_scale(self.scale, g)
            b_s = w_s * a_s

            # quantize parameters, then calculate
            self.w_int = round_pass((self.conv.weight / w_s).clamp(Qn, Qp))
            self.b_int = round_pass((self.conv.bias / b_s).clamp(bQn, bQp)) if self.has_bias else None
            return F.conv2d(x_q, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups) * b_s, b_s

        else:
            return F.conv2d(x, self.w_int, self.b_int, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups), None

class ActLSQ(Module):
    def __init__(self, to_bit=8, mult_bit=16, training=True, batch_init=20):
        super(ActLSQ, self).__init__()
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.scale = Parameter(torch.Tensor(1))
        self.batch_init = batch_init
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('s', torch.ones(1, requires_grad=False))
        self.register_buffer('mult', torch.ones(1, requires_grad=False, dtype=torch.int64))
        self.register_buffer('shift', torch.ones(1, requires_grad=False, dtype=torch.int64))

    def __repr__(self):
        s = super(ActLSQ, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, from_fp32={self.from_fp32}, training={self.training}, ReLU_clip={self.ReLU_clip})"
        return s

    def get_scales(self, name):
        return [
            (f"{name}_s", self.s),
            (f"{name}_inf_scale", self.mult / 2**self.shift)
        ]

    def set_training(self, set=True):
        self.training = set

    def forward(self, x, a_s=None):
        Qn, Qp = signed_minmax(self.to_bit)

        if self.training:
            g = 1.0 / math.sqrt(x.numel() * Qp)
            # requant inputs, first layer's input will always be fp
            x_q = x / a_s if (a_s != None) else x

            # initialize scale on first input
            if self.init_state == 0:
                y = x.detach().abs()
                # init_scale = (y.max()*2)/(Qp-Qn)
                init_scale = y.mean()*2 / math.sqrt(Qp)
                self.scale.data = init_scale
                self.init_state += 1

            elif self.init_state < self.batch_init:
                y = x.detach().abs()
                # init_scale = (y.max()*2)/(Qp-Qn)
                init_scale = y.mean()*2 / math.sqrt(Qp)
                self.scale.data = 0.9*self.scale.data + 0.1*init_scale
                self.init_state += 1

            # gives scale a lsq gradient
            scale = grad_scale(self.scale, g)

            # calculate approximate dyadic value
            # while preserving original scale gradient
            # then quantize sum
            if a_s == None:
                self.s, (self.mult, self.shift) = dyadic_scale(scale, self.mult_bit)
                x_round = round_pass((x_q / self.s).clamp(Qn, Qp))
                # x_round = round_pass((x_q / self.s + 0.5).clamp(Qn, Qp))
                return x_round * self.s, self.s
            else:
                self.s, (self.mult, self.shift) = dyadic_scale(scale / a_s, self.mult_bit)
                x_round = round_pass((x_q / self.s).clamp(Qn, Qp))
                # x_round = round_pass((x_q / self.s + 0.5).clamp(Qn, Qp))
                return x_round * self.s * a_s, self.s * a_s

        else:
            # quantize sum
            x_round = round_pass((x / self.s).clamp(Qn, Qp))
            # x_round = round_pass((x / self.s + 0.5).clamp(Qn, Qp))
            # x_round = (
            #     torch.bitwise_right_shift(
            #         x.type(torch.int64)*self.mult + 1, 
            #         self.shift
            #     )
            # ).type(torch.float)

            return x_round, None

class ResActLSQ(Module):
    def __init__(self, from_bit=32, to_bit=8, mult_bit=16, training=True, to_fp32=False, batch_init=20):
        super(ResActLSQ, self).__init__()
        self.from_bit = from_bit
        self.to_bit = to_bit
        self.mult_bit = mult_bit
        self.training = training
        self.to_fp32 = to_fp32
        self.scale = Parameter(torch.Tensor(1))
        self.batch_init = batch_init
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('align_int', torch.ones(1, requires_grad=False))
        #self.register_buffer('align_mult', torch.ones(1, requires_grad=False, dtype=torch.int64))
        #self.register_buffer('align_shift', torch.ones(1, requires_grad=False, dtype=torch.int64))
        self.register_buffer('s', torch.ones(1, requires_grad=False))
        self.register_buffer('mult', torch.ones(1, requires_grad=False, dtype=torch.int64))
        self.register_buffer('shift', torch.ones(1, requires_grad=False, dtype=torch.int64))

    def __repr__(self):
        s = super(ResActLSQ, self).__repr__()
        s = f"({s} to_bit={self.to_bit}, mult_bit={self.mult_bit}, from_fp32={self.from_fp32}, training={self.training}, ReLU_clip={self.ReLU_clip})"
        return s

    def get_scales(self, name):
        return [
            (f"{name}_align_inf_scale", self.align_int),
            #(f"{name}_align_int_scale", self.align_mult / 2**self.align_shift),
            #(f"{name}_align_int_scale", self.align_mult / 2**self.align_shift),
            (f"{name}_s", self.s),
            (f"{name}_inf_scale", self.mult / 2**self.shift),
        ]

    def set_training(self, set=True):
        self.training = set

    def forward(self, x, a_s=None, res_x=None, res_a_s=None):
        Qn, Qp = signed_minmax(self.to_bit)
        rQn, rQp = signed_minmax(self.from_bit)

        if self.training:
            g = 1.0 / math.sqrt(x.numel() * Qp)
            # requant inputs
            res_x_q = res_x / res_a_s
            x_q = x / a_s

            # align residual input and quantize
            # ! shift should be as same as rescale's
            #self.align_s, (self.align_mult, self.align_shift) = dyadic_scale(a_s/res_a_s, 8) 
            self.align_int = round_pass(res_a_s/a_s)
            res_x_align = round_pass((res_x_q * self.align_int).clamp(rQn, rQp))
            # res_x_align = round_pass((res_x_q / self.align_s).clamp(rQn, rQp))
            # res_x_align = round_pass((res_x_q / self.align_s + 0.5).clamp(rQn, rQp))
            
            # obtain sum
            mix_x_q = x_q + res_x_align
            
            # initialize scale on first input
            if self.init_state == 0:
                mix_x = mix_x_q * a_s
                y = mix_x.detach().abs()
                # init_scale = (y.max()*2)/(Qp-Qn)
                init_scale = y.mean()*2 / math.sqrt(Qp)
                self.scale.data = init_scale
                self.init_state += 1

            elif self.init_state < self.batch_init:
                mix_x = mix_x_q * a_s
                y = mix_x.detach().abs()
                # init_scale = (y.max()*2)/(Qp-Qn)
                init_scale = y.mean()*2 / math.sqrt(Qp)
                self.scale.data = 0.9*self.scale.data + 0.1*init_scale
                self.init_state += 1

            # gives scale a lsq gradient
            scale = grad_scale(self.scale, g)

            # calculate approximate dyadic value and quantize
            # while preserving original scale gradient
            self.s, (self.mult, self.shift) = dyadic_scale(scale / a_s, self.mult_bit)
            mix_x_round = round_pass((mix_x_q / self.s).clamp(Qn, Qp))

            return mix_x_round * scale, scale

        else:
            # align residual input
            res_x_align = round_pass((res_x * self.align_int).clamp(rQn, rQp))
            # res_x_align = round_pass((res_x / self.align_s).clamp(rQn, rQp))
            # res_x_align = round_pass((res_x / self.align_s + 0.5).clamp(rQn, rQp))
            # res_x_align = (
            #     torch.bitwise_right_shift(
            #         res_x.type(torch.int64)*self.align_mult + 1, 
            #         self.align_shift
            #     )
            # ).type(torch.float)

            # obtain sum
            mix_x = x + res_x_align

            # quantize sum
            mix_x_round = round_pass((mix_x / self.s).clamp(Qn, Qp))
            # mix_x_round = round_pass((mix_x / self.s+0.5).clamp(Qn, Qp))
            # mix_x_round = (
            #     torch.bitwise_right_shift(
            #         mix_x.type(torch.int64)*self.mult + 1, 
            #         self.shift
            #     )
            # ).type(torch.float)
            
            if self.to_fp32: # last layer, connecting back to fp calculation
                return mix_x_round*self.scale, None
            else:
                return mix_x_round, None

def set_training(model, set=True):
    for n, m in model.named_modules():
        if type(m) in [LinearLSQ, ConvLSQ, ActLSQ, ResActLSQ]:
            m.set_training(set)