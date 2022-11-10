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

    else:
        raise NotImplementedError
    
class LSQObserver(Module):
    def __init__(self, mode, Qn, Qp, calibrate_count=1, momentum=0.1):
        super().__init__()
        self.mode = mode
        self.Qn = Qn
        self.Qp = Qp
        self.scale_func = get_scale_func(mode, Qn, Qp)
        self.momentum = momentum
        self.scale = Parameter(torch.tensor(0.))
        self.g = None
        self.register_buffer('calibrate_count', torch.tensor(calibrate_count))
        self.register_buffer('counter', torch.tensor(0))
    
    def __repr__(self):
        return f"LSQObserver(mode={self.mode}, Qn={self.Qn}, Qp={self.Qp}, calibrate_count={self.calibrate_count}, momentum={self.momentum})"
    
    def init_scale_counter(self):
        self.counter.data = torch.tensor(0)

    def get_scale(self):
        return self.scale

    def forward(self, x: torch.Tensor):
        if self.counter < self.calibrate_count:
            if self.counter == 0:
                prev = self.scale.data
                self.scale.data = self.scale_func(x)
                print(f"\nprev: {prev}, cur: {self.scale.data}")
            else:
                self.scale.data = (1-self.momentum)*self.scale.data + self.momentum*self.scale_func(x)
            self.counter += 1
        
        if self.g is None: 
            self.g = 1.0 / math.sqrt(x.numel() * self.Qp)

        scale = grad_scale(self.scale, self.g)
        return scale

class _QBase(Module):
    def __init__(self, to_bit=8, training=True):
        super().__init__()
        self.Qn, self.Qp = signed_minmax(to_bit)
        self.to_bit = to_bit
        self.observer = LSQObserver(mode='minmax', Qn=self.Qn, Qp=self.Qp)
        self.training = training

    def extra_repr(self):
        return f"training={self.training} to_bit={self.to_bit}"

    def set_training(self, set=True):
        self.training = set
        
    def forward(self, x, scale=None):
        raise NotImplementedError

class QLinear(_QBase):
    def __init__(self, linear: nn.Linear, bias_bit=32, to_bit=8, training=True):
        _QBase.__init__(self, to_bit, training)
        self.inherit_layer(linear)
        self.bias_bit = bias_bit
        self.bQn, self.bQp = signed_minmax(self.bias_bit)

    def __repr__(self):
        s = super(QLinear, self).__repr__()
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

    def inference(self, x: torch.Tensor):
        return F.linear(x, self.w_int, self.b_int)

    def forward(self, x, a_s):
        if self.training:
            # requant inputs
            x_q = x / a_s
            
            # initialize scale on first input
            w_s = self.observer(self.weight)
            b_s = w_s * a_s

            # quantize weights and bias
            self.w_int = round_pass((self.weight / w_s).clamp(self.Qn, self.Qp))
            if self.bias is not None:
                self.b_int = round_pass((self.bias / b_s).clamp(self.bQn, self.bQp))
            
            return self.inference(x_q) * b_s, b_s
        else:
            return self.inference(x), None
        
    # def forward(self, x, a_s):
    #     return F.linear(x, self.weight, self.bias), a_s

class QLinearInner(QLinear):
    def __init__(self, linear: nn.Linear, bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, linear, bias_bit, to_bit, training)

    def inference(self, x: torch.Tensor):
        return x @ self.w_int + self.b_int
    
    # def forward(self, x, a_s):
    #     return x @ self.weight + self.bias, a_s#torch.ones_like(a_s)

class QLinearOuter(QLinear):
    def __init__(self, linear: nn.Linear, bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, linear, bias_bit, to_bit, training)

    def inference(self, x: torch.Tensor):
        return self.w_int @ x
    
    # def forward(self, x, a_s):
    #     return self.weight @ x, a_s#torch.ones_like(a_s)

class QLinearBN(QLinear):
    def __init__(self, bn: nn.BatchNorm1d, bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, bn, bias_bit, to_bit, training)
    
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

class QConv(QLinear):
    def __init__(self, conv: nn.Conv2d, bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, conv, bias_bit, to_bit, training)
    
    def __repr__(self):
        s = super(QConv, self).__repr__()
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

    def inference(self, x: torch.Tensor):
        return F.conv2d(x, self.w_int, self.b_int, self.stride, self.padding, self.dilation, self.groups)

    # def forward(self, x, a_s):
    #     return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups), a_s

class QAct(_QBase):
    def __init__(self, mult_bit=16, return_fp=False, to_bit=8, training=True):
        _QBase.__init__(self, to_bit, training)
        self.mult_bit = mult_bit
        self.return_fp = return_fp
        self.observer = LSQObserver(mode='lsq', Qn=self.Qn, Qp=self.Qp, calibrate_count=20, momentum=0.1)
        self.register_buffer('mult', torch.tensor(0))
        self.register_buffer('shift', torch.tensor(0))

    def __repr__(self):
        s = super(QAct, self).__repr__()
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
            scale = self.observer(x)
 
            # calculate approximate dyadic value, then quantize sum
            self.s, (self.mult, self.shift) = dyadic_scale(scale / a_s, self.mult_bit)
            x_round = round_pass((x_q / self.s).clamp(self.Qn, self.Qp))
            return x_round * scale, scale

        else: # on inference
            x_round = torch.floor((x * self.mult / 2**self.shift)+0.5).clamp(self.Qn, self.Qp)
            if self.return_fp: # last layer, return output as fp32 
                scale = self.observer.get_scale()
                return x_round*scale, scale
            else:
                return x_round, None
    
    # def forward(self, x, a_s):
    #     return x, a_s

class QResAct(_QBase):
    def __init__(self, bias_bit=32, mult_bit=16, return_fp=False, to_bit=8, training=True):
        _QBase.__init__(self, to_bit, training)
        self.bias_bit = bias_bit
        self.rQn, self.rQp = signed_minmax(self.bias_bit)
        self.mult_bit = mult_bit
        self.return_fp = return_fp
        self.observer = LSQObserver(mode='lsq', Qn=self.Qn, Qp=self.Qp, calibrate_count=20, momentum=0.1)
        
        self.register_buffer('align_int', torch.tensor(0))
        self.register_buffer('mult', torch.tensor(0))
        self.register_buffer('shift', torch.tensor(0))
        self.register_buffer('S_cur', torch.tensor(0))
        self.register_buffer('S_res', torch.tensor(0))

    def __repr__(self):
        s = super(QResAct, self).__repr__()
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
            mix_x = res_x + x # mix_x_q * a_s
            scale = self.observer(mix_x)

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
                scale = self.observer.get_scale()
                return mix_x_round*scale, None
            else:
                return mix_x_round, None

    # def forward(self, x, a_s=None, res_x=None, res_a_s=None):
    #     return x+res_x, a_s
    

def set_training(model, set=True):
    cnt = 0
    for n, m in model.named_modules():
        if issubclass(type(m), _QBase):
            cnt += 1
            m.set_training(set)
    print(f"Set {cnt} layers to {set}.")

def init_scale_counter(model):
    cnt = 0
    for n, m in model.named_modules():
        if issubclass(type(m), LSQObserver):
            cnt += 1
            m.init_scale_counter()
    print(f"Reset {cnt} counters.")