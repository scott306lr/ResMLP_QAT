import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import math

from ..utils import signed_minmax, scale_to_dyadic, dyadic_to_scale

def grad_scale(x: torch.Tensor, scale: torch.Tensor):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad

def dyadic_scale(scale: torch.Tensor, mult_bit):
    m, e = scale_to_dyadic(1 / scale, mult_bit)
    d_scale = 1 / dyadic_to_scale(m, e)
    return d_scale.detach() - scale.detach() + scale, (m, e)

def round_pass(x: torch.Tensor):
    y = (x).round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

# TODO: replace scale finding with a LSQ_Observer Module
# def get_observer_method(mode, Qn, Qp):
#     if mode == "minmax":
#         def observer(x: torch.Tensor):
#             y = x.detach()
#             return torch.max(y) / Qp

#     elif mode == "lsq":
#         def observer(x: torch.Tensor):
#             y = x.detach()
#             mean = torch.mean(y[y.nonzero(as_tuple=True)])
#             return 2*mean / math.sqrt(Qp)

#     elif mode == "lsq+":
#         def observer(x: torch.Tensor):
#             y = x.detach()
#             std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
#             return torch.max(torch.abs(mean - 3*std), torch.abs(mean + 3*std))/Qp
#     else:
#         raise NotImplementedError
#     return observer

class _BaseLSQ(Module):
    def __init__(self, to_bit=8, training=True):
        super().__init__()
        self.Qn, self.Qp = signed_minmax(to_bit)
        self.to_bit = to_bit
        self.scale = Parameter(torch.Tensor(1))
        self.g = None
        self.training = training

    def extra_repr(self):
        return f"training={self.training} to_bit={self.to_bit}"

    def set_training(self, set=True):
        self.training = set

    def update_scale(self, x: torch.Tensor, mode='minmax', momentum=1):
        y = x.detach().abs()
        if mode == 'minmax':
            init_scale = torch.max(y) / self.Qp
        elif mode == 'lsq':
            std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
            init_scale = 2*mean / math.sqrt(self.Qp)
        elif mode == 'lsq+':
            std, mean = torch.std_mean(y[y.nonzero(as_tuple=True)])
            init_scale = torch.max(torch.abs(mean - 3*std), torch.abs(mean + 3*std))/self.Qp
        else:
            raise ValueError('Invalid mode')

        self.scale.data = (1-momentum)*self.scale.data + momentum*init_scale
        
    def forward(self, x, scale=None):
        raise NotImplementedError

class LinearLSQ(_BaseLSQ):
    def __init__(self, linear: nn.Linear, bias_bit=32, to_bit=8, training=True):
        # super(LinearLSQ, self).__init__(to_bit, training)
        _BaseLSQ.__init__(self, to_bit, training)
        self.inherit_layer(linear)
        self.bias_bit = bias_bit
        self.bQn, self.bQp = signed_minmax(self.bias_bit)
        self.register_buffer('init_state', torch.zeros(1))

    def __repr__(self):
        s = super(LinearLSQ, self).__repr__()
        s = f'{s}({self.in_features}, {self.out_features}, bias_bit={self.bias_bit})'
        return s

    def inherit_layer(self, linear: nn.Linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data)
        self.register_buffer('w_int', torch.zeros_like(linear.weight.data))
        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data)
            self.register_buffer('b_int', torch.zeros_like(linear.bias.data))
        else:
            self.register_parameter('bias', None)
            self.register_buffer('b_int', None)
        self.g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)

    def inference(self, x: torch.Tensor):
        return F.linear(x, self.w_int, self.b_int)
    
    def forward(self, x, a_s):
        if self.training:
            # requant inputs
            x_q = x / a_s
            
            # initialize scale on first input
            if self.init_state == 0:
                self.update_scale(self.weight, mode='lsq')
                self.init_state.fill_(1)

            # calculate scales
            w_s = grad_scale(self.scale, self.g) # gives scale a lsq gradient
            b_s = w_s * a_s

            # quantize weights and bias
            self.w_int = round_pass((self.weight / w_s).clamp(self.Qn, self.Qp))
            if self.bias is not None:
                self.b_int = round_pass((self.bias / b_s).clamp(self.bQn, self.bQp))
            
            return self.inference(x_q) * b_s, b_s
        else:
            return self.inference(x), None

class LinearBNLSQ(LinearLSQ):
    def __init__(self, bn: nn.BatchNorm1d, bias_bit=32, to_bit=8, training=True):
        LinearLSQ.__init__(self, bn, bias_bit, to_bit, training)
    
    def inherit_layer(self, bn: nn.BatchNorm1d):
        self.in_features = bn.num_features
        self.out_features = bn.num_features

        output_factor = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        weight = torch.diag(output_factor)
        bias = - output_factor * bn.running_mean + bn.bias
        self.weight = Parameter(weight)
        self.bias = Parameter(bias)
        self.register_buffer('w_int', torch.zeros_like(self.weight.data))
        self.register_buffer('b_int', torch.zeros_like(self.bias.data))
        self.g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)

class ConvLSQ(LinearLSQ):
    def __init__(self, conv: nn.Conv2d, bias_bit=32, to_bit=8, training=True):
        LinearLSQ.__init__(self, conv, bias_bit, to_bit, training)
    
    def __repr__(self):
        s = super(ConvLSQ, self).__repr__()
        s = f'{s}({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias_bit={self.bias_bit})'

    def inherit_layer(self, conv: nn.Conv2d):
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode

        self.weight = Parameter(conv.weight.data)
        self.register_buffer('w_int', torch.zeros_like(conv.weight.data, dtype=torch.int8))
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data)
            self.register_buffer('b_int', torch.zeros_like(conv.bias.data, dtype=torch.int8))
        else:
            self.register_parameter('bias', None)
            self.register_buffer('b_int', None)
        self.g = 1.0 / math.sqrt(self.weight.numel() * self.Qp)

    def inference(self, x: torch.Tensor):
        return F.conv2d(x, self.w_int, self.b_int, self.stride, self.padding, self.dilation, self.groups)

class ActLSQ(_BaseLSQ):
    def __init__(self, mult_bit=16, batch_init=20, return_fp=False, to_bit=8, training=True):
        _BaseLSQ.__init__(self, to_bit, training)
        self.mult_bit = mult_bit
        self.batch_init = batch_init
        self.return_fp = return_fp
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('s', torch.ones(1))
        self.register_buffer('mult', torch.ones(1, dtype=torch.int64))
        self.register_buffer('shift', torch.ones(1, dtype=torch.int64))

    def __repr__(self):
        s = super(ActLSQ, self).__repr__()
        s = f'{s}(to_bit={self.to_bit}, mult_bit={self.mult_bit})'
        return s

    def get_scales(self, name):
        return [
            (f"rescale/{name}_rescale", self.s),
            # (f"{name}_SA", 1/self.s),
            (f"{name}_inf_scale", self.mult / 2**self.shift)
        ]

    def set_training(self, set=True):
        self.training = set

    def forward(self, x, a_s=None):
        if a_s == None: a_s = 1.0
        if self.training:
            # requant inputs, first layer's input will always be fp
            x_q = x / a_s

            # initialize scale on first input
            if self.init_state == 0:
                # self.update_scale(x, mode='lsq', momentum=1)
                self.update_scale(x, mode='minmax', momentum=1)
                self.init_state += 1

            elif self.init_state < self.batch_init:
                # self.update_scale(x, mode='lsq', momentum=0.1)
                self.update_scale(x, mode='minmax', momentum=0.1)
                self.init_state += 1

            # gives scale a lsq gradient
            if self.g is None: self.g = 1.0 / math.sqrt(x.numel() * self.Qp)
            scale = grad_scale(self.scale, self.g)

            # calculate approximate dyadic value, then quantize sum
            self.s, (self.mult, self.shift) = dyadic_scale(scale / a_s, self.mult_bit)
            x_round = round_pass((x_q / self.s).clamp(self.Qn, self.Qp))
            return x_round * scale, scale

        else: # on inference
            x_round = torch.floor((x * self.mult / 2**self.shift)+0.5).clamp(self.Qn, self.Qp)
            if self.return_fp: # last layer, return output as fp32
                return x_round*self.scale, self.scale
            else:
                return x_round, None

class ResActLSQ(_BaseLSQ):
    def __init__(self, bias_bit=32, mult_bit=16, batch_init=20, return_fp=False, to_bit=8, training=True):
        _BaseLSQ.__init__(self, to_bit, training)
        self.bias_bit = bias_bit
        self.rQn, self.rQp = signed_minmax(self.bias_bit)
        self.mult_bit = mult_bit
        self.batch_init = batch_init
        self.return_fp = return_fp
        
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('align_int', torch.ones(1))
        self.register_buffer('s', torch.ones(1))
        self.register_buffer('mult', torch.ones(1, dtype=torch.int64))
        self.register_buffer('shift', torch.ones(1, dtype=torch.int64))
        self.register_buffer('S_cur', torch.ones(1))
        self.register_buffer('S_res', torch.ones(1))

    def __repr__(self):
        s = super(ResActLSQ, self).__repr__()
        s = f"({s} bias_bit={self.bias_bit}, mult_bit={self.mult_bit})"
        return s

    def get_scales(self, name):
        return [
            (f"align/{name}_S_cur", self.S_cur),
            (f"align/{name}_S_res", self.S_res),
            (f"rescale/{name}_rescale", self.s),
            (f"{name}_align_inf_scale", self.align_int),
        ]

    def set_training(self, set=True):
        self.training = set

    def forward(self, x, a_s=None, res_x=None, res_a_s=None):
        if self.training:
            # requant inputs
            res_x_q = res_x / res_a_s
            x_q = x / a_s

            # align residual input and quantize
            self.S_res = res_a_s
            self.S_cur = a_s
            self.align_int = round_pass((res_a_s/a_s).clamp(self.Qn, self.Qp))
            res_x_align = round_pass((res_x_q * self.align_int).clamp(self.rQn, self.rQp))
            
            # obtain sum
            mix_x_q = x_q + res_x_align
            
            # initialize scale on first input
            if self.init_state == 0:
                mix_x = res_x + x # mix_x_q * a_s
                # self.update_scale(mix_x, mode='lsq', momentum=1)
                self.update_scale(mix_x, mode='minmax', momentum=1)
                self.init_state += 1

            elif self.init_state < self.batch_init:
                mix_x = res_x + x # mix_x_q * a_s
                # self.update_scale(mix_x, mode='lsq', momentum=0.1)
                self.update_scale(mix_x, mode='minmax', momentum=0.1)
                self.init_state += 1

            # gives scale a lsq gradient
            if self.g is None: self.g = 1.0 / math.sqrt(x.numel() * self.Qp)
            scale = grad_scale(self.scale, self.g)

            # calculate approximate dyadic value and quantize
            self.s, (self.mult, self.shift) = dyadic_scale(scale / a_s, self.mult_bit)
            mix_x_round = round_pass((mix_x_q / self.s).clamp(self.Qn, self.Qp))

            return mix_x_round * scale, scale

        else:
            # align residual input
            res_x_align = (res_x * self.align_int).clamp(self.rQn, self.rQp)

            # obtain sum
            mix_x = x + res_x_align

            # quantize sum
            mix_x_round = torch.floor((mix_x / self.s)+0.5).clamp(self.Qn, self.Qp)
            if self.return_fp: # last layer, return output as fp32
                return mix_x_round*self.scale, None
            else:
                return mix_x_round, None

def set_training(model, set=True):
    cnt = 0
    for n, m in model.named_modules():
        if issubclass(type(m), _BaseLSQ):
            cnt += 1
            m.set_training(set)
    print(f"set {cnt} layers to {set}")