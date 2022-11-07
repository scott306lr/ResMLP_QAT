import torch
import torch.nn as nn

from ..quantization.quantizer.lsq import QLinear

def get_linear_layers(model, specify_names=None, prefix=""):
    linear_layers = []
    for name, module in model.named_modules():
        if (specify_names is not None) and (name not in specify_names): 
            continue
        if isinstance(module, nn.Linear) or isinstance(module, QLinear):
            linear_layers.append((f'{prefix}{name}', module))
    return linear_layers

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
                self.handlers.append( m.register_forward_hook(hook_func(n)) )
    
    def create_apply_hook(self, func, to_dict, model_layers):
        self.apply_hook(self.create_hook(to_dict, func), model_layers)
    
    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()
            self.handlers = []