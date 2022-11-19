from typing import List
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from .quant_func import *

## Observer Classes
class _Observer(Module):
    def __init__(self, Qn, Qp, name="Observer"):
        super().__init__()
        self.Qn = Qn
        self.Qp = Qp
        self.name = name
        self.scale = None
        self.register_buffer('counter', torch.tensor(0))
    
    def __repr__(self):
        return f"Observer(Qn={self.Qn}, Qp={self.Qp})"
    
    def init_scale_counter(self):
        self.counter.data = torch.tensor(0)

    def get_scale(self):
        return self.scale

    def forward(self, x: torch.Tensor):
        raise NotImplementedError

class MinmaxObserver(_Observer):
    def __init__(self, Qn, Qp, name="MinMaxObserver", momentum=0.1):
        _Observer.__init__(self, Qn, Qp, name)
        self.momentum = momentum
        self.register_buffer('min', torch.tensor(0.))
        self.register_buffer('max', torch.tensor(0.))
        self.register_buffer('scale', torch.tensor(0.))

    def __repr__(self):
        return f"MinmaxObserver(mode={self.mode}, Qn={self.Qn}, Qp={self.Qp}, calibrate_count={self.calibrate_count}, momentum={self.momentum})"

    def forward(self, x: torch.Tensor):
        y = x.detach().abs()
        min, max = y.min(), y.max()
        if self.counter == 0:
            self.min, self.max = min, max
            self.counter = 1
        else:
            self.max = (1-self.momentum)*self.max + self.momentum*max
            self.min = (1-self.momentum)*self.min  + self.momentum*min
        
        self.scale.data = (self.max - self.min) / (self.Qp - self.Qn)
        return self.scale

class LSQObserver(_Observer):
    def __init__(self, Qn, Qp, name="LSQObserver", mode="lsq", calibrate_count=1, momentum=0.1):
        _Observer.__init__(self, Qn, Qp, name)
        self.mode = mode
        self.scale_func = get_scale_func(mode, Qn, Qp)
        self.momentum = momentum
        self.register_buffer('calibrate_count', torch.tensor(calibrate_count))
        self.scale = Parameter(torch.tensor(0.))
        self.g = None

    def __repr__(self):
        return f"LSQObserver(mode={self.mode}, Qn={self.Qn}, Qp={self.Qp}, calibrate_count={self.calibrate_count}, momentum={self.momentum})"

    def forward(self, x: torch.Tensor):
        if self.counter < self.calibrate_count:
            if self.counter == 0:
                prev = self.scale.data
                self.scale.data = self.scale_func(x)
                std, mean = torch.std_mean(x[x.nonzero(as_tuple=True)])
                print(f"\n{self.name}:")
                print(f"\tinput val: std:{std}, mean:{mean}")
                print(f"\tscale val: {self.scale.data}")

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
        self.observer = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='lsq')
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

    # def test(self, x_q, b_s):
    #     return F.linear(x_q, self.w_int*b_s, self.bias), b_s

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
            # return self.test(x_q, b_s)
        else:
            return self.inference(x), None
        
    # def forward(self, x, a_s):
    #     return F.linear(x, self.weight, self.bias), a_s

#TODO: bias correction & retain BN on train
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
        self.register_buffer('w_int', torch.zeros_like(conv.weight.data))
        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data)
            self.register_buffer('b_int', torch.zeros_like(conv.bias.data))
        else:
            self.register_parameter('bias', None)
            self.register_buffer('b_int', None)

    def inference(self, x: torch.Tensor):
        return F.conv2d(x, self.w_int, self.b_int, self.stride, self.padding, self.dilation, self.groups)

    def test(self, x_q, b_s):
        return F.conv2d(x_q, self.w_int*b_s, self.bias, self.stride, self.padding, self.dilation, self.groups), b_s
    # def forward(self, x, a_s):
    #     return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups), a_s

class QAct(_QBase):
    def __init__(self, mult_bit=16, return_fp=False, to_bit=8, training=True):
        _QBase.__init__(self, to_bit, training)
        self.mult_bit = mult_bit
        self.return_fp = return_fp
        self.observer = LSQObserver(mode='lsq', Qn=self.Qn, Qp=self.Qp, calibrate_count=20, momentum=0.1, name="Act")
        self.register_buffer('s', torch.tensor(0))
        self.register_buffer('mult', torch.tensor(0))
        self.register_buffer('shift', torch.tensor(0))
        

    def __repr__(self):
        s = super(QAct, self).__repr__()
        s = f'{s}(to_bit={self.to_bit}, mult_bit={self.mult_bit})'
        return s

    def get_scales(self, name):
        return [
            # (f"rescale/{name}_rescale", self.s),
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
            x_round = round_pass((x_q / self.s)).clamp(self.Qn, self.Qp)
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
        self.observer = LSQObserver(mode='lsq', Qn=self.Qn, Qp=self.Qp, calibrate_count=20, momentum=0.1, name="ADD")
        
        self.register_buffer('align_int', torch.tensor(0))
        self.register_buffer('s', torch.tensor(0))
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
            # (f"rescale/{name}_rescale", self.s),
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
            self.align_int = round_pass((res_a_s/a_s))#.clamp(self.Qn, self.Qp)) #! This clamping limits align range, for experiments without limitation, pls remove.
            res_x_align = round_pass((res_x_q * self.align_int)).clamp(self.rQn, self.rQp)
            
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
    
# Quant Utils
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

#! Experimental, used for Hardware-Aware ResMLP
class QLinearInner(QLinear):
    def __init__(self, linear: nn.Linear, bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, linear, bias_bit, to_bit, training)
        self.observer = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='lsq')

    def inference(self, x: torch.Tensor):
        return x @ self.w_int + self.b_int
    
    def test(self, x_q, b_s):
        return x_q @ (self.w_int*b_s) + self.bias, b_s
    
    def forward(self, x, a_s):
        return x @ self.weight + self.bias, None#torch.ones_like(a_s)

class QLinearOuter(QLinear):
    def __init__(self, linear: nn.Linear, bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, linear, bias_bit, to_bit, training)
        self.observer = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='lsq')

    def inherit_layer(self, linear: nn.Linear):
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = Parameter(linear.weight.data.t())
        self.register_buffer('w_int', torch.zeros_like(linear.weight.data))
        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data)
            self.register_buffer('b_int', torch.zeros_like(linear.bias.data))
        else:
            self.register_parameter('bias', None)
            self.register_buffer('b_int', None)
            
    # def inference(self, x: torch.Tensor):
    #     return self.w_int @ x
    
    # def test(self, x_q, b_s):
    #     return (self.w_int*b_s) @ x_q, b_s
    
    def forward(self, x, a_s):
        return self.weight @ x, None#torch.ones_like(a_s)

class QCrossPatch(_QBase):
    def __init__(self, linears: List[nn.Linear], mult_bit=16, bias_bit=32, to_bit=8, training=True):
        _QBase.__init__(self, to_bit, training)
        self.inherit_layer(linears)
        self.bias_bit = bias_bit
        self.bQn, self.bQp = signed_minmax(self.bias_bit)
        self.observer = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='lsq', name="W1")
        self.observer2 = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='lsq', name="W2")

        self.mult_bit = mult_bit
        self.observer_act = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='lsq', calibrate_count=20, momentum=0.1, name="Wact")
        self.register_buffer('mult', torch.tensor(0))
        self.register_buffer('shift', torch.tensor(0))
    
    def inherit_layer(self, linears: List[nn.Linear]):
        norm, attn, gamma = linears[0], linears[1], linears[2]
        self.in_features = norm.in_features
        self.out_features = gamma.out_features

        self.norm_w = Parameter(norm.weight.data, requires_grad=False)
        self.norm_b = Parameter(norm.bias.data, requires_grad=False)
        self.attn_w = Parameter(attn.weight.data, requires_grad=False)
        self.attn_b = Parameter(attn.bias.data, requires_grad=False)
        self.gamma_w = Parameter(gamma.weight.data, requires_grad=False)

        W1 = self.norm_w * self.gamma_w
        B1 = self.norm_b.repeat(196,1) @ self.gamma_w + torch.inverse(self.attn_w) @ self.attn_b.repeat(384,1).T @ self.gamma_w
        W2 = self.attn_w

        self.register_buffer('w1_int', torch.zeros_like(W1.data))
        self.register_buffer('b1_int', torch.zeros_like(B1.data))
        self.register_buffer('w2_int', torch.zeros_like(W2.data))

    def inference(self, x: torch.Tensor):
        return x @ self.w1_int + self.b1_int

    def inference2(self, x: torch.Tensor):
        return self.w2_int @ x

    def forward(self, x, a_s):
        if self.training:
            # merge weights
            W1 = self.norm_w * self.gamma_w
            B1 = self.norm_b.repeat(196,1)@ self.gamma_w + torch.inverse(self.attn_w) @ self.attn_b.repeat(384,1).T @ self.gamma_w
            W2 = self.attn_w

            #! W1
            # requant inputs
            x_q = x / a_s

            # initialize scale on first input
            w1_s = self.observer(W1)
            b1_s = w1_s * a_s

            # quantize weights and bias #1
            self.w1_int = round_pass((W1 / w1_s).clamp(self.Qn, self.Qp))
            self.b1_int = round_pass((B1 / b1_s).clamp(self.bQn, self.bQp))
            x1 = self.inference(x_q)

            #! ACT
            scale = self.observer_act(x1)
            self.s, (self.mult, self.shift) = dyadic_scale(scale / a_s, self.mult_bit)
            x_round = round_pass((x_q / self.s)).clamp(self.Qn, self.Qp)

            #! W2
            # initialize scale on first input
            w2_s = self.observer2(W2)
            b2_s = w2_s * scale

            # quantize weights and bias #1
            self.w2_int = round_pass((W2 / w2_s).clamp(self.Qn, self.Qp))
            x2 = self.inference2(x_round)
            
            return x2 * b2_s, b2_s

        else:
            x1 = self.inference(x)
            x_round = torch.floor((x1 * self.mult / 2**self.shift)+0.5).clamp(self.Qn, self.Qp)
            x2 = self.inference2(x_round)
            return x2, None

    # def forward(self, x, a_s):
    #     x = F.linear(x, self.norm_w, self.norm_b).transpose(1, 2)
    #     x = F.linear(x, self.attn_w, self.attn_b).transpose(1, 2)
    #     x = F.linear(x, self.gamma_w, None)
    #     return x, a_s


class QCrossLayer1(_QBase):
    def __init__(self, linears: List[nn.Linear], bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, linears, bias_bit, to_bit, training)
        self.observer = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='minmax', name="CL1")

    def inherit_layer(self, linears: List[nn.Linear]):
        norm, fc1 = linears[0], linears[1]
        self.in_features = fc1.in_features
        self.out_features = fc1.out_features

        self.norm_w = Parameter(norm.weight.data)
        self.norm_b = Parameter(norm.bias.data)
        self.fc1_w = Parameter(fc1.weight.data)
        self.fc1_b = Parameter(fc1.bias.data)

        self.register_buffer('w_int', torch.zeros_like(fc1.weight.data))
        self.register_buffer('b_int', torch.zeros_like(fc1.bias.data))

    def inference(self, x: torch.Tensor):
        return F.linear(x, self.w_int, self.b_int)

    def forward(self, x, a_s):
        if self.training:
            # requant inputs
            x_q = x / a_s
            
            # merge weights
            weight = self.fc1_w @ self.norm_w
            bias = self.fc1_b + F.linear(self.fc1_w, self.norm_b)

            # initialize scale on first input
            w_s = self.observer(weight)
            b_s = w_s * a_s

            # quantize weights and bias
            self.w_int = round_pass((weight / w_s).clamp(self.Qn, self.Qp))
            self.b_int = round_pass((bias / b_s).clamp(self.bQn, self.bQp))
            
            return self.inference(x_q) * b_s, b_s
            # return self.test(x_q, b_s)
        else:
            return self.inference(x), None
        
    # def forward(self, x, a_s):
    #     weight = self.fc1_w @ self.norm_w
    #     bias = self.fc1_b + F.linear(self.fc1_w, self.norm_b)
    #     return F.linear(x, weight, bias), a_s

class QCrossLayer2(QLinear):
    def __init__(self, linears: List[nn.Linear], bias_bit=32, to_bit=8, training=True):
        QLinear.__init__(self, linears, bias_bit, to_bit, training)
        self.observer = LSQObserver(Qn=self.Qn, Qp=self.Qp, mode='minmax', name="CL1")

    def inherit_layer(self, linears: List[nn.Linear]):
        fc2, gamma_2 = linears[0], linears[1]
        self.in_features = fc2.in_features
        self.out_features = fc2.out_features

        self.fc2_w = Parameter(fc2.weight.data)
        self.fc2_b = Parameter(fc2.bias.data)
        self.gamma2_w = Parameter(gamma_2.weight.data)

        self.register_buffer('w_int', torch.zeros_like(fc2.weight.data))
        self.register_buffer('b_int', torch.zeros_like(fc2.bias.data))

    def inference(self, x: torch.Tensor):
        return F.linear(x, self.w_int, self.b_int)

    def forward(self, x, a_s):
        if self.training:
            # requant inputs
            x_q = x / a_s
            
            # merge weights
            weight = self.gamma2_w @ self.fc2_w 
            bias = F.linear(self.gamma2_w, self.fc2_b)

            # initialize scale on first input
            w_s = self.observer(weight)
            b_s = w_s * a_s

            # quantize weights and bias
            self.w_int = round_pass((weight / w_s).clamp(self.Qn, self.Qp))
            self.b_int = round_pass((bias / b_s).clamp(self.bQn, self.bQp))
            
            return self.inference(x_q) * b_s, b_s
            # return self.test(x_q, b_s)
        else:
            return self.inference(x), None
    
    # def forward(self, x, a_s):
    #     weight = self.gamma2_w @ self.fc2_w 
    #     bias = F.linear(self.gamma2_w, self.fc2_b)
    #     return F.linear(x, weight, bias), a_s
