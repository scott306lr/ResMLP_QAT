import torch
import torch.nn as nn

from ..quantization.quantizer.lsq import _QBase, QLinear, QAct, QResAct, QLinearInner, QLinearOuter
# from ..models.resmlp_v3 import Inner, Outer
from ..models.resmlp_model_v4 import Inner, Outer


def get_linear_layers(model, specify_names=None, prefix=""):
    linear_layers = []
    for name, module in model.named_modules():
        if (specify_names is not None) and (name not in specify_names):
            continue
        if isinstance(module, nn.Linear) or isinstance(module, Inner) or isinstance(module, Outer) or isinstance(module, QLinear) or isinstance(module, QLinearInner) or isinstance(module, QLinearOuter):
            # print(f'{prefix}{name}')
            linear_layers.append((f'{prefix}{name}', module))
    return linear_layers


def get_quant_layers(model, specify_names=None, prefix=""):
    qlayers = []
    for name, module in model.named_modules():
        if (specify_names is not None) and (name not in specify_names):
            continue
        if issubclass(type(module), _QBase):
            qlayers.append((f'{prefix}{name}', module))
    return qlayers


def get_act_layers(model, specify_names=None, prefix=""):
    qlayers = []
    for name, module in model.named_modules():
        if (specify_names is not None) and (name not in specify_names):
            continue
        if isinstance(module, QAct) or isinstance(module, QResAct):
            qlayers.append((f'{prefix}{name}', module))
    return qlayers


class HookHandler():
    def __init__(self):
        self.handlers = []

    # func is a function with two inputs (org_val, output):
    # org_val: is a original value of the dict location about to update
    # output:  is the output of the hooked layer
    def create_hook(self, to_dict, func):
        def rec_act_name(name):
            def hook(model, input, output):
                if name in to_dict:
                    to_dict[name] = func(to_dict[name], output)
                else:
                    to_dict[name] = func(None, output)
            return hook
        return rec_act_name

    def apply_hook(self, hook_func, model_layers):
        for layers in model_layers:
            for n, m in layers:
                self.handlers.append(m.register_forward_hook(hook_func(n)))

    def create_apply_hook(self, func, to_dict, model_layers):
        self.apply_hook(self.create_hook(to_dict, func), model_layers)

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()
            self.handlers = []
