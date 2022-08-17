import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath

# from resmlp import resmlp_24
class Q_PatchEmbed(nn.Module):
    def __init__(self, bias_bit=None, full_precision_flag=False):
        super(Q_PatchEmbed, self).__init__()
        self.bias_bit
        self.full_precision_flag=full_precision_flag

    def set_param(self, patch):
        self.proj = QuantConv2d(bias_bit=self.bias_bit, full_precision_flag=self.full_precision_flag)
        self.proj.set_param(patch.proj)
        self.norm = nn.Identity()
    
    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, w_s, a_s = self.proj(x, a_s)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, a_s

class Q_Mlp(nn.Module):
    def __init__(self, bias_bit=None, full_precision_flag=False):
        super(Q_Mlp, self).__init__()
        self.bias_bit = bias_bit
        self.full_precision_flag=full_precision_flag

    def set_param(self, mlp):
        self.fc1 = QuantLinear(bias_bit=self.bias_bit, full_precision_flag=self.full_precision_flag)
        self.fc1.set_param(mlp.fc1)
        self.fc2 = QuantLinear(bias_bit=self.bias_bit, full_precision_flag=self.full_precision_flag, ReLU_clip=True)
        self.fc2.set_param(mlp.fc2)
    
    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, w_s, a_s = self.fc1(x, a_s)
        x, w_s, a_s = self.fc2(x, a_s)
        return x, a_s

class Q_Layer(nn.Module):
    def __init__(self, full_precision_flag=False, res_fp=False):
        super(Q_Layer, self).__init__()
        self.full_precision_flag=full_precision_flag
        self.res_fp=res_fp

    def set_param(self, block):        
        self.norm1 = QuantLinear(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.norm1.set_param(block.norm1)
        self.act1 = QuantAct(activation_bit=8, full_precision_flag=self.full_precision_flag)

        self.attn = QuantLinear(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.attn.set_param(block.attn)

        self.gamma_1 = QuantLinear(full_precision_flag=self.full_precision_flag)
        self.gamma_1.set_param(block.gamma_1)

        self.add_1 = EltwiseAddQuantLinear()

        self.norm2 = QuantLinear(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.norm2.set_param(block.norm2)

        self.mlp = Q_Mlp(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.mlp.set_param(block.mlp)

        self.gamma_2 = QuantLinear(full_precision_flag=self.full_precision_flag)
        self.gamma_2.set_param(block.gamma_2)

        self.add_2 = EltwiseAddQuantLinear()

    # ! this implementation only works for per-tensor (transpose)
    def forward(self, x, a_s=None):
        identity, org_a_s = x, a_s

        # ----- Cross-patch sublayer ----- START
        x, w_s, a_s = self.norm1(x, a_s)
        x = x.transpose(1,2)
        x, w_s, a_s = self.attn(x, a_s)
        x = x.transpose(1,2)
        x, w_s, a_s = self.gamma_1(x, a_s)
        # ----- Cross-patch sublayer ----- END
        x, w_s, a_s = self.add_1(x, identity, w_s, a_s, org_a_s)
        identity, org_a_s = x, a_s

        # ---- Cross-channel sublayer ---- START
        x, w_s, a_s = self.norm2(x, a_s)
        x, w_s, a_s = self.mlp(x, a_s)
        x, w_s, a_s = self.gamma_2(x, a_s)
        # ---- Cross-channel sublayer ---- END
        x, w_s, a_s = self.add_2(x, identity, w_s, a_s, org_a_s)

        return x, a_s

class Q_ResMLP24(nn.Module):
    """
        Quantized ResMLP24 model.
    """
    def __init__(self, model, full_precision_flag=False, res_fp=False):
        super().__init__()
        self.full_precision_flag = full_precision_flag
        self.res_fp = res_fp
        
        self.quant_patch = Q_PatchEmbed(full_precision_flag=self.full_precision_flag)
        self.quant_patch.set_param(getattr(model, 'patch_embed'))

        blocks = getattr(model, 'blocks')
        for block_num in range(0, 24):
            mlp_layer = getattr(blocks, "{}".format(block_num))
            quant_mlp_layer = Q_Layer(full_precision_flag=self.full_precision_flag, res_fp=self.res_fp)
            quant_mlp_layer.set_param(mlp_layer)
            setattr(self, "layer{}".format(block_num), quant_mlp_layer)

        self.norm = getattr(model, 'norm')
        self.head = getattr(model, 'head')

    def forward(self, x):
        B = x.shape[0]
        x, act_scaling_factor = self.quant_input(x)
        x, act_scaling_factor = self.quant_patch(x, act_scaling_factor)

        for block_num in range(0, 24):
            layer = getattr(self, f"layer{block_num}")
            x, act_scaling_factor = layer(x, act_scaling_factor)

        #TODO fp32 to int8, not yet done
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B,1,-1)
        x = x[:, 0]
        x = self.head(x)
        # make sure x have only 2-dimensions [x] -> [[x]], [[[x]]] -> [[x]]
        x = x.view(x.size(0), -1)
        
        return x

def q_resmlp24(model, full_precision_flag=False, res_fp=False):
    net = Q_ResMLP24(model, full_precision_flag=full_precision_flag, res_fp=res_fp)
    return net