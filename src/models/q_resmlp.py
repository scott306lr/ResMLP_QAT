import torch
import torch.nn as nn
import copy

from ..quantization.quantizer.lsq import *
# from ..quantization.quantizer.lsq import *
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath

# from resmlp import resmlp_24
class QPatchEmbed(nn.Module):
    def __init__(self, patch, bias_bit=None, to_bit=8):
        super(QPatchEmbed, self).__init__()
        self.bias_bit = bias_bit
        self.set_param(patch, to_bit)

    def set_param(self, patch, to_bit):
        self.proj = ConvLSQ(patch.proj)
        self.norm = nn.Identity()
        self.act  = ActLSQ(to_bit=to_bit)
    
    def get_scales(self):
        scales = []
        # scales += self.proj.get_scales("PatchEmbed_Conv")
        scales += self.act.get_scales("PatchEmbed_Act")
        return scales

    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.proj(x, a_s)
        x = x.flatten(2).transpose(1, 2)
        x, a_s = self.act(x, a_s)
        x = self.norm(x)

        return x, a_s

class Q_Mlp(nn.Module):
    def __init__(self, mlp):
        super(Q_Mlp, self).__init__()
        self.set_param(mlp)

    def set_param(self, mlp):
        self.fc1 = LinearLSQ(mlp.fc1)
        self.relu = torch.nn.ReLU()
        self.act1 = ActLSQ()
        self.fc2 = LinearLSQ(mlp.fc2)
        self.act2 = ActLSQ()
    
    def get_scales(self):
        scales = []
        # scales += self.fc1.get_scales(f"L{self.layer}_L5")
        # scales += self.fc2.get_scales(f"L{self.layer}_L6")
        # scales += self.act1.get_scales(f"L{self.layer}_Act5")
        # scales += self.act2.get_scales(f"L{self.layer}_Act6")
        return scales
    
    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.fc1(x, a_s)
        x = self.relu(x)
        x, a_s = self.act1(x, a_s)
        x, a_s = self.fc2(x, a_s)
        x, a_s = self.act2(x, a_s)
        return x, a_s

class QLayer_Block(nn.Module):
    def __init__(self, block, layer, res_to_bit):
        super(QLayer_Block, self).__init__()
        self.layer = layer
        self.res_to_bit = res_to_bit
        self.set_param(block, layer)

    def set_param(self, block, layer):  
        self.norm1 = LinearLSQ(block.norm1)
        self.act1 = ActLSQ()

        self.attn = LinearLSQ(block.attn)
        self.act2 = ActLSQ()

        self.gamma_1 = LinearLSQ(block.gamma_1)
        self.add_1 = ResActLSQ(to_bit=self.res_to_bit)

        self.norm2 = LinearLSQ(block.norm2)
        self.act3 = ActLSQ()

        self.mlp = Q_Mlp(block.mlp)

        self.gamma_2 = LinearLSQ(block.gamma_2)

        if layer == 24-1:
            self.add_2 = ResActLSQ(to_bit=self.res_to_bit, return_fp=True) # dequant output back to fp
        else:
            self.add_2 = ResActLSQ(to_bit=self.res_to_bit, return_fp=False)

    def get_scales(self):
        scales = []
        # scales += self.norm1.get_scales(f"L{self.layer}_L1")
        # scales += self.attn.get_scales(f"L{self.layer}_L2")
        # scales += self.gamma_1.get_scales(f"L{self.layer}_L3")
        # scales += self.norm2.get_scales(f"L{self.layer}_L4")
        # scales += self.gamma_2.get_scales(f"L{self.layer}_L7")

        # scales += self.act1.get_scales(f"L{self.layer}_Act1")
        # scales += self.act2.get_scales(f"L{self.layer}_Act2")
        # scales += self.add_1.get_scales(f"L{self.layer}_Act3")
        # scales += self.act3.get_scales(f"L{self.layer}_Act4")
        # scales += self.add_2.get_scales(f"L{self.layer}_Act7")

        scales += self.add_1.get_scales(f"L{self.layer}_Add1")
        scales += self.add_2.get_scales(f"L{self.layer}_Add2")
        return scales
   
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

RES_RESCALE_BIT = 8
class Q_ResMLP24(nn.Module):
    """
        Quantized ResMLP24 model.
    """
    def __init__(self, model):
        super().__init__()
        self.quant_input = ActLSQ(to_bit=8)
        self.quant_patch = QPatchEmbed(model.patch_embed, to_bit=RES_RESCALE_BIT)
        self.blocks = nn.ModuleList([QLayer_Block(model.blocks[i], layer=i, res_to_bit=RES_RESCALE_BIT) for i in range(24)])
        self.norm = model.norm#QLinear(model.norm) #model.norm
        self.head = model.head#QLinear(getattr(model, 'head'))

    def get_scales(self):
        scales = []
        # scales += self.quant_input.get_scales(f"Input_Act")
        scales += self.quant_patch.get_scales()

        for i, blk in enumerate(self.blocks):
            # if i >= ALL_FP_LAYER-1 and i <= ALL_FP_LAYER+1 : 
            scales += blk.get_scales()

        return scales

    def forward(self, x):
        B = x.shape[0]

        a_s = None
        x, a_s = self.quant_input(x, a_s)
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