import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath

# from resmlp import resmlp_24
class Q_PatchEmbed(nn.Module):
    """
       Quantized PatchEmbed.
    """
    def __init__(self, full_precision_flag=False):
        super(Q_PatchEmbed, self).__init__()
        self.full_precision_flag=full_precision_flag

    def set_param(self, patch):
        proj = patch.proj
        self.proj = QuantConv2d(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.proj.set_param(proj)
        
        # norm = patch.norm
        # if norm != nn.Identity():
        #     self.norm = QuantLinear(full_precision_flag=self.full_precision_flag)
        #     self.norm.set_param(norm)
        self.quant_act_int32 = QuantAct(full_precision_flag=self.full_precision_flag)
        self.norm = nn.Identity()
    
    def forward(self, x, act_scaling_factor=None):
        # forward using the quantized modules
        x, weight_scaling_factor = self.proj(x, act_scaling_factor)
        x = x.flatten(2).transpose(1, 2)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)
        x = self.norm(x)

        return x, act_scaling_factor


class Q_Mlp(nn.Module):
    """
       Quantized MLP.
    """
    def __init__(self, bias_bit=None, full_precision_flag=False):
        super(Q_Mlp, self).__init__()
        self.bias_bit = bias_bit
        self.full_precision_flag=full_precision_flag

    def set_param(self, mlp):
        fc1 = mlp.fc1
        self.fc1 = QuantLinear(bias_bit=self.bias_bit, full_precision_flag=self.full_precision_flag)
        self.fc1.set_param(fc1)
        self.quant_act1 = QuantAct(full_precision_flag=self.full_precision_flag)

        fc2 = mlp.fc2
        self.fc2 = QuantLinear(bias_bit=self.bias_bit, full_precision_flag=self.full_precision_flag)
        self.fc2.set_param(fc2)
        self.quant_act2 = QuantAct(full_precision_flag=self.full_precision_flag)
    
    def forward(self, x, act_scaling_factor=None):
        # forward using the quantized modules
        x, weight_scaling_factor = self.fc1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)
        x, weight_scaling_factor = self.fc2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        return x, act_scaling_factor

class Q_Layer(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self, full_precision_flag=False, res_fp=False):
        super(Q_Layer, self).__init__()
        self.full_precision_flag=full_precision_flag
        self.res_fp=res_fp

    def set_param(self, block):
        self.quant_act = QuantAct(full_precision_flag=self.full_precision_flag)
        
        norm1 = block.norm1
        self.norm1 = QuantLinear(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.norm1.set_param(norm1)
        self.quant_act1 = QuantAct(full_precision_flag=self.full_precision_flag)

        attn = block.attn
        self.attn = QuantLinear(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.attn.set_param(attn)
        self.quant_act2 = QuantAct(full_precision_flag=self.full_precision_flag)

        gamma_1 = block.gamma_1
        self.gamma_1 = QuantLinear(full_precision_flag=self.full_precision_flag)
        self.gamma_1.set_param(gamma_1)
        self.quant_act_int32_1 = QuantAct(full_precision_flag=self.full_precision_flag)

        norm2 = block.norm2
        self.norm2 = QuantLinear(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.norm2.set_param(norm2)
        self.quant_act3 = QuantAct(full_precision_flag=self.full_precision_flag)

        mlp = block.mlp
        self.mlp = Q_Mlp(bias_bit=32, full_precision_flag=self.full_precision_flag)
        self.mlp.set_param(mlp)

        gamma_2 = block.gamma_2
        self.gamma_2 = QuantLinear(full_precision_flag=self.full_precision_flag)
        self.gamma_2.set_param(gamma_2)
        self.quant_act_int32_2 = QuantAct(full_precision_flag=self.full_precision_flag)

    def forward(self, x, act_scaling_factor=None):
        # Cross-patch sublayer
        # ! this implementation only works for per-tensor (transpose)
        identity = x
        if not self.full_precision_flag:
            identity_act_scaling_factor = act_scaling_factor.clone()
        else:
            identity_act_scaling_factor = 1
        
        x, weight_scaling_factor = self.norm1(x, act_scaling_factor)
        x = x.transpose(1,2)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.attn(x, act_scaling_factor)
        x = x.transpose(1,2)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)
        
        x, weight_scaling_factor = self.gamma_1(x, act_scaling_factor)
        
        x = x + identity
        if not self.res_fp:
            x, act_scaling_factor = self.quant_act_int32_1(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32_1(x)

        # Cross-channel sublayer
        identity = x
        if not self.full_precision_flag:
            identity_act_scaling_factor = act_scaling_factor.clone()
        else:
            identity_act_scaling_factor = 1

        x, weight_scaling_factor = self.norm2(x, act_scaling_factor)
        x, act_scaling_factor = self.quant_act3(x, act_scaling_factor, weight_scaling_factor)

        x, act_scaling_factor = self.mlp(x, act_scaling_factor)

        x, weight_scaling_factor = self.gamma_2(x, act_scaling_factor)
        
        x = x + identity
        if not self.res_fp:
            x, act_scaling_factor = self.quant_act_int32_2(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32_2(x)

        return x, act_scaling_factor

class Q_ResMLP24(nn.Module):
    """
        Quantized ResMLP24 model.
    """
    def __init__(self, model, full_precision_flag=False, res_fp=False):
        super().__init__()
        self.full_precision_flag = full_precision_flag
        self.res_fp = res_fp
        patch_embed = getattr(model, 'patch_embed')
        blocks = getattr(model, 'blocks')
        norm = getattr(model, 'norm')
        head = getattr(model, 'head')
        
        self.quant_input = QuantAct(full_precision_flag=self.full_precision_flag)

        # self.quant_patch_embed_conv = QuantConv2d()
        # self.quant_patch_embed_conv.set_param(patch_embed.proj)
        self.quant_patch = Q_PatchEmbed(full_precision_flag=self.full_precision_flag)
        self.quant_patch.set_param(patch_embed)

        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for block_num in range(0, 24):
            mlp_layer = getattr(blocks, "{}".format(block_num))
            quant_mlp_layer = Q_Layer(full_precision_flag=self.full_precision_flag, res_fp=self.res_fp)
            quant_mlp_layer.set_param(mlp_layer)
            setattr(self, "layer{}".format(block_num), quant_mlp_layer)

        # self.norm = QuantLinear
        # self.norm.set_param(norm)
        # self.quant_act_output = QuantAct()
        # self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)
        # self.head = QuantLinear()
        # self.head.set_param(head)

        # Currently, we use fp32 here
        self.norm = norm
        self.head = head
        


    def forward(self, x):
        B = x.shape[0]
        x, act_scaling_factor = self.quant_input(x)
        x, act_scaling_factor = self.quant_patch(x, act_scaling_factor)

        for block_num in range(0, 24):
                tmp_func = getattr(self, f"layer{block_num}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        #fp32
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


# model = resmlp_24()
# model = q_resmlp24(model)
# print(model)