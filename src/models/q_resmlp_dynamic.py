import torch
import torch.nn as nn
import copy

from ..quantization.quantizer.lsq import *
from ..quantization.quantizer.lsq_dynamic import *
# from ..quantization.quantizer.lsq import *
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath



class Dyanmic_Q_MLP(nn.Module):
    def __init__(self, mlp, bits_list):
        super(Dyanmic_Q_MLP, self).__init__()
        self.set_param(mlp)
        self.bits_list = bits_list
    def set_param(self, mlp):
        self.fc1 = DynamicQLinear(mlp.fc1, bits_list=self.bits_list)
        self.relu = torch.nn.ReLU()
        self.fc2 = DynamicQLinear(mlp.fc2, bits_list=self.bits_list)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# from resmlp import resmlp_24
class QPatchEmbed(nn.Module):
    def __init__(self, patch, bits_list):
        super(QPatchEmbed, self).__init__()
        self.bits_list = bits_list
        self.set_param(patch)

    def set_param(self, patch):
        self.proj = DynamicQConv(patch.proj)
        self.norm = nn.Identity()
    

    def forward(self, x):
        # forward using the quantized modules
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class Dynamic_QLayer_Block(nn.Module):
    def __init__(self, block, layer, bits_list):
        super(Dynamic_QLayer_Block, self).__init__()
        self.layer = layer 
        self.bits_list = bits_list
        self.set_param(block, layer)
    def set_param(self, block, layer):
        self.norm1 = DynamicQLinear(block.norm1, self.bits_list)
        self.attn = DynamicQLinear(block.attn, self.bits_list)
        self.gamma_1 = DynamicQLinear(block.gamma_1, self.bits_list)
        self.norm2 = DynamicQLinear(block.norm2, self.bits_list)
        self.mlp = DynamicQLinear(block.mlp, self.bits_list)
        self.gamma_2 = DynamicQLinear(block.gamma2, self.bits_list)

    def forward(self, x):

        # ----------cross-patch sublayer (start)----------
        org_x = x
        x = self.norm1(x)
        x = x.transpose(1,2)
        x = self.attn(x)
        x = x.transpose(1,2)
        x = self.gamma_1(x) + org_x
        # ----------cross-patch sublayer (end)----------

        org_x = x
        # ----------cross-channel sublayer (start)----------   
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.gamma_2(x) + org_x
        return x



class Dynamic_Q_ResMLP24(nn.Module):
    def __init__(self, model, bits_list):
        super().__init__()
        self.bits_list = bits_list
        self.patch_embed = QPatchEmbed(model.patch_embed, self.bits_list)
        self.blocks = nn.ModuleList([
            Dynamic_QLayer_Block(model.blocks[i], 
                                layer=i, bits_list=self.bits_list) 
            for i in range(24)
        ])
        self.norm = model.norm
        self.head = model.head


    def forward(self, x):
        Batch = x.shape[0]
        x = self.patch_embed(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
        
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B, 1, -1)
        x = x[:, 0]
        x = self.head(x)
        x = x.view(x.size(0), -1)
        return x


