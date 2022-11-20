from typing import List
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from .quant_func import *
from .observer import *

WEIGHT_SHARING = True
USE_BQ = True

clamp = F.hardtanh  # softclamp

def quantize(v, s, p, signed, per_channel=False):
    if signed:
        Qn = -2 ** (p - 1)
        Qp = 2 ** (p - 1) - 1
        if p == 1:
            Qp = 1
        nweight = v.nelement()
        if per_channel:
            nweight /= v.size(0)
        gradScaleFactor = 1 / math.sqrt(nweight * Qp)
    else:
        Qn = 0
        Qp = 2 ** p - 1
        nfeatures = v.nelement()
        gradScaleFactor = 1 / math.sqrt(nfeatures * Qp)

    s = grad_scale(s, gradScaleFactor)
    v = v / s.abs()
    v = clamp(v, Qn, Qp)
    if signed and p == 1:
        vbar = sign_pass(v)
    else:
        vbar = round_pass(v)
    vhat = vbar * s.abs()
    return vhat


class LSQQuantize(nn.Module):
    
    def __init__(self, signed=True, bitwidth=8, ws=WEIGHT_SHARING, init='lsq'):
        super(LSQQuantize, self).__init__()
        self.scale = nn.Parameter(torch.Tensor(1))
        self.signed = signed
        self.bits = bitwidth
        self.ws = ws  # whether to use weight sharing
        self.init = init
        self.initialized = False
        self.weight = None

    def _quantize(self, x):
        if not self.initialized:  # calibration, make sure inputs is pretrained weight
            if not self.ws:
                self.weight = nn.Parameter(torch.zeros_like(x, requires_grad=True))
                self.weight.data = x.detach() if self.ws else copy.deepcopy(x.detach())
            if self.signed:
                q_max = 2 ** (self.bits - 1) - 1
                if self.bits == 1:
                    q_max = 1
            else:
                q_max = 2 ** (self.bits) - 1
            if self.init == 'lsq':
                self.scale.data = 2.0 / math.sqrt(q_max) * x.abs().mean().detach().view(1)
            else:
                assert self.init == 'lsq+'
                mu = x.mean().detach().view(1)
                std = x.std().detach().view(1)
                s_init = torch.max(torch.abs(mu - 3*std), torch.abs(mu + 3*std))
                self.scale.data = s_init / (q_max + 1.)
            self.initialized = True

        # Allow non Weight sharing
        if not self.ws:
            x = self.weight

        return quantize(x, self.scale, self.bits, self.signed)

    def forward(self, x):
        dqx = self._quantize(x)
        return dqx

    def re_organize_weights(self, sorted_idx):
        pass


class FakeQuantize(nn.Module):
    CALIBRATE = False
    CALIBRATION_CRITERION = 'kl'
    """
    Fake Quantizer that allows learnable scale and offset
    To use BatchQuant, set USE_BQ=True
    """

    def __init__(self, signed=True, bitwidth=8, use_bq=USE_BQ, affine=True, grad_scale=False):
        super(FakeQuantize, self).__init__()
        # the scale is initialized as 0 because it is passed through a softplus
        self.scale = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.offset = nn.Parameter(torch.zeros(1, requires_grad=True))
        self.use_bq = use_bq
        self.affine = affine
        self.grad_scale = grad_scale
        self.signed = signed
        self.observer = BatchMinMaxObserver(
            signed=signed,
            bitwidth=bitwidth,
            qscheme=torch.per_tensor_affine)
        self.bitwidth = bitwidth
        self.calibrate_func = None

    def quantize(self, x):
        if self.signed:
            qmin, qmax = -2 ** (self.bitwidth - 1), 2 ** (self.bitwidth - 1) - 1
        else:
            qmin, qmax = 0, 2 ** self.bitwidth - 1

        if self.use_bq:
            observer_scale, zp = self.observer.calculate_qparams()
            observer_scale = observer_scale.to(self.scale.device)
            zp = zp.to(self.scale.device)
        else:
            observer_scale = 1.
            zp = 0.

        if self.CALIBRATE and self.affine and self.calibrate_func is not None:
            self.calibrate_func(x, observer_scale, zp, self.scale, self.offset, qmin, qmax)

        if self.affine:
            learned_scale = F.softplus(self.scale, beta=math.log(2.))
            learned_offset = self.offset
            if self.grad_scale:
                factor = 1. / math.sqrt(x.nelement() * qmax)
                learned_scale = grad_scale(learned_scale, factor)
                learned_offset = grad_scale(learned_offset, factor)
        else:
            learned_scale = 1.
            learned_offset = 0.

        # Scaled Quant with proper clipping, correct variant 2
        # Nicer gradient, better performance
        qx = x * learned_scale / observer_scale + zp + learned_offset
        qx = torch.clamp(qx, qmin, qmax)
        qx = FakeRoundOp.apply(qx)
        dqx = (qx - zp - learned_offset) * observer_scale / learned_scale

        return qx, dqx

    def forward(self, x):
        if self.training or isinstance(self.observer, BatchMinMaxObserver):
            if self.use_bq:
                x = self.observer(x)
        qx, dqx = self.quantize(x)
        return dqx



class DynamicWeightQuantizer(nn.Module):
    """
    Dynamic Quantizer
    """
    def __init__(self, signed, bits_list):
        super(DynamicWeightQuantizer, self).__init__()
        self.quantizers = nn.ModuleDict([
            ['4', LSQQuantize(bitwidth=2, signed=signed)],
            ['8', LSQQuantize(bitwidth=3, signed=signed)],
            ['32', nn.Identity()]
        ])
        self.bit_list = bits_list
        self.active_bit = max(self.bit_list)

    def set_bit(self, bit):
        assert bit in self.bit_list
        self.active_bit = bit

    def forward(self, x):
        return self.quantizers[str(self.active_bit)](x)
        


class DynamicActivationQuantizer(nn.Module):
    """
    Dynamic Quantizer
    """
    def __init__(self, signed, bits_list):
        super(DynamicActivationQuantizer, self).__init__()
        self.quantizers = nn.ModuleDict([
            ['4', FakeQuantize(bitwidth=4, signed=signed)],
            ['8', FakeQuantize(bitwidth=8, signed=signed)],
            ['32', nn.Identity()]
        ])
        self.signed = signed
        self.bits_list = bits_list
        self.active_bit = max(self.bits_list)

    def set_bit(self, bit):
        assert bit in self.bits_list
        self.active_bit = bit

    def forward(self, x):
        return self.quantizers[str(self.active_bit)](x)



class DynamicQLinear(nn.Module):
    def __init__(self, 
        linear: nn.Linear, bits_list,
        signed = True):

        self.inherit_layer(linear) # set the correspond feature of linear layer based on the given FP32 model
        self.active_out_features = self.max_out_features
        self.bits_list = bits_list
        self.weight_quantizer = DynamicWeightQuantizer(signed=True, bits_list=self.bits_list)
        self.signed = signed
        self.activation_quantizer = DynamicActivationQuantizer(signed=True, bits_list=bits_list)


    def inherit_layer(self, linear: nn.Linear):
        self.max_in_features = linear.in_features
        self.max_out_features = linear.out_features
        self.weight = Parameter(linear.weight.data)
        if linear.bias is not None:
            self.bias = Parameter(linear.bias.data)
        else:
            self.bias = None
        


    def forward(self, x, out_features=None):
        # first quantize the activation from previous layer
        x = self.activation_quantizer(x)
        if out_features is None:
            out_features = self.active_out_features
        in_features = x.size(1)
        quantized_weight = self.weight_quantizer(self.weight)
        active_quantized_weight = quantized_weight[:out_features, :in_features].contiguous()
        bias = self.bias[:out_features] if self.bias else None
        return F.linear(x, active_quantized_weight, bias)


class DynamicQConv(nn.Module):
    def __init__(self, conv: nn.Conv2d, bits_list, signed=True):
        
        self.bits_list = bits_list
        self.weight_quantizer = DynamicWeightQuantizer(signed=True, bits_list=self.bits_list)
        self.signed = signed
        self.activation_quantizer = DynamicActivationQuantizer(signed=True, bits_list=self.bits_list)


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

        if conv.bias is not None:
            self.bias = Parameter(conv.bias.data)
        else:
            self.bias = None

    def forward(self, x):
        x = self.activation_quantizer(x)
        quantized_weight = self.weight_quantizer(self.weight)
        bias = self.bias
        return F.conv2d(x, quantized_weight, bias, self.stride, self.padding, self.dilation, self.groups)