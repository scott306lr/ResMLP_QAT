import torch
import torch.nn as nn
import copy

from ..quantization_utils.quant_modules import *
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath

# from resmlp import resmlp_24
class QPatchEmbed(nn.Module):
    def __init__(self, patch, bias_bit=None, to_bit=16):
        super(QPatchEmbed, self).__init__()
        self.bias_bit = bias_bit
        self.set_param(patch, to_bit)

    def set_param(self, patch, to_bit):
        self.proj = QConv2d(patch.proj, regular=True) # Same as normal conv, with an additional ouput (value is None)
        self.norm = nn.Identity()
        self.act  = QAct(from_fp32=True, to_bit=to_bit, regular=False) # First layer, accepts fp as input even on validation
    
    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.proj(x, a_s)
        x = x.flatten(2).transpose(1, 2)
        x, a_s = self.act(x, a_s)
        x = self.norm(x)

        return x, a_s

class Q_Mlp(nn.Module):
    def __init__(self, mlp, regular=False):
        super(Q_Mlp, self).__init__()
        self.set_param(mlp, regular=regular)

    def set_param(self, mlp, regular=False):
        self.fc1 = QLinear(mlp.fc1, regular=regular)
        self.act1 = QAct(ReLU_clip=True, regular=regular)
        self.fc2 = QLinear(mlp.fc2, regular=regular)
        self.act2 = QAct(regular=regular)
    
    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.fc1(x, a_s)
        x, a_s = self.act1(x, a_s)
        x, a_s = self.fc2(x, a_s)
        x, a_s = self.act2(x, a_s)
        return x, a_s

class QLayer_Block(nn.Module):
    def __init__(self, block, layer):
        super(QLayer_Block, self).__init__()
        self.set_param(block, layer)

    def set_param(self, block, layer):  
        # set to 24: all quant
        #! to make model fp only,      
        ALL_FP_LAYER = 24 
        if layer >= ALL_FP_LAYER:
            #to_bit = 8
            regular = True
        else:
            #to_bit = 16
            regular = False

        self.norm1 = QLinear(block.norm1, regular=regular)
        self.act1 = QAct(regular=regular)

        self.attn = QLinear(block.attn, regular=regular)
        self.act2 = QAct(regular=regular)

        self.gamma_1 = QLinear(block.gamma_1, regular=regular)
        self.add_1 = QResAct(regular=regular)

        self.norm2 = QLinear(block.norm2, regular=regular)
        self.act3 = QAct(regular=regular)

        self.mlp = Q_Mlp(block.mlp, regular=regular)

        self.gamma_2 = QLinear(block.gamma_2, regular=regular)

        if layer == ALL_FP_LAYER-1: 
            self.add_2 = QResAct(to_fp32=True, regular=regular) # dequant output back to fp
        else:
            self.add_2 = QResAct(regular=regular)

    # ! this implementation only works for per-tensor (transpose)
    def forward(self, x, a_s=None):
        org_x, org_a_s = x, a_s

        # ----- Cross-patch sublayer ----- START
        x, a_s = self.norm1(x, a_s)
        x, a_s = self.act1(x, a_s)

        x = x.transpose(1,2)
        x, a_s = self.attn(x, a_s)
        x, a_s = self.act2(x, a_s)
        x = x.transpose(1,2)

        x, a_s = self.gamma_1(x, a_s)
        x, a_s = self.add_1(x, a_s, org_x, org_a_s)
        # ----- Cross-patch sublayer ----- END
        org_x, org_a_s = x, a_s
        
        # ---- Cross-channel sublayer ---- START
        x, a_s = self.norm2(x, a_s)
        x, a_s = self.act3(x, a_s)

        x, a_s = self.mlp(x, a_s)

        x, a_s = self.gamma_2(x, a_s)
        x, a_s = self.add_2(x, a_s, org_x, org_a_s)
        # ---- Cross-channel sublayer ---- END
        return x, a_s

class Q_ResMLP24(nn.Module):
    """
        Quantized ResMLP24 model.
    """
    def __init__(self, model):
        super().__init__()
        # self.quant_input = QAct()
        self.quant_patch = QPatchEmbed(model.patch_embed, to_bit=8)
        self.blocks = nn.ModuleList([QLayer_Block(model.blocks[i], layer=i) for i in range(24)])
        self.norm = model.norm#QLinear(model.norm) #model.norm
        self.head = model.head#QLinear(getattr(model, 'head'))

    def forward(self, x):
        B = x.shape[0]

        a_s = None
        # x, a_s = self.quant_input(x, a_s)
        x, a_s = self.quant_patch(x, a_s)

        for i, blk in enumerate(self.blocks):
            x, a_s = blk(x, a_s)
   
        #! all fp32 below
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B, 1, -1)
        x = x[:, 0]
        x = self.head(x)
        x = x.view(x.size(0), -1)
        
        return x

def q_resmlp(model):
    net = Q_ResMLP24(model)
    return net