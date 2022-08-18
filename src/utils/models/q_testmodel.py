import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath

# from resmlp import resmlp_24
class QPatchEmbed(nn.Module):
    def __init__(self, patch, bias_bit=None, full_precision_flag=False):
        super(QPatchEmbed, self).__init__()
        self.bias_bit = bias_bit
        self.full_precision_flag=full_precision_flag
        self.set_param(patch)

    def set_param(self, patch):
        self.proj = QConv2d(patch.proj)
        self.norm = nn.Identity()
    
    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.proj(x, a_s)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, a_s

class Q_Mlp(nn.Module):
    def __init__(self, mlp, full_precision_flag=False):
        super(Q_Mlp, self).__init__()
        self.full_precision_flag=full_precision_flag
        self.set_param(mlp)

    def set_param(self, mlp):
        self.fc1 = QLinear(mlp.fc1)
        self.act1 = QAct(ReLU_clip=True)
        self.fc2 = QLinear(mlp.fc2)
        self.act2 = QAct()
    
    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.fc1(x, a_s)
        x, a_s = self.act1(x, a_s)
        x, a_s = self.fc2(x, a_s)
        x, a_s = self.act2(x, a_s)
        return x, a_s

class QLayer_Block(nn.Module):
    def __init__(self, block, full_precision_flag=False):
        super(QLayer_Block, self).__init__()
        self.full_precision_flag=full_precision_flag
        self.set_param(block)

    def set_param(self, block):        
        self.norm1 = QLinear(block.norm1)
        self.act1 = QAct()

        self.attn = QLinear(block.attn)
        self.act2 = QAct()

        self.gamma_1 = QLinear(block.gamma_1)
        self.add_1 = QResAct()

        self.norm2 = QLinear(block.norm2)
        self.act3 = QAct()

        self.mlp = Q_Mlp(block.mlp)
        self.act4 = QAct()

        self.gamma_2 = QLinear(block.gamma_2)
        self.add_2 = QResAct()

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
        x, a_s = self.act4(x, a_s)

        x, a_s = self.gamma_2(x, a_s)
        x, a_s = self.add_2(x, a_s, org_x, org_a_s)
        # ---- Cross-channel sublayer ---- END
        return x, a_s

class Q_ResMLP24(nn.Module):
    """
        Quantized ResMLP24 model.
    """
    def __init__(self, model, full_precision_flag=False):
        super().__init__()
        self.full_precision_flag = full_precision_flag
        
        self.quant_input = QAct()
        self.quant_patch = QPatchEmbed(getattr(model, 'patch_embed'))
        self.blocks = nn.ModuleList([QLayer_Block(getattr(getattr(model, 'blocks'), "{}".format(i))) for i in range(24)])
        # self.act = QAct() # for average pooling
        self.norm = getattr(model, 'norm')#QLinear(getattr(model, 'norm'))
        self.head = getattr(model, 'head')#QLinear(getattr(model, 'head'))

    def forward(self, x):
        B = x.shape[0]
        x, a_s = self.quant_input(x)
        x, a_s = self.quant_patch(x, a_s)

        for i, blk in enumerate(self.blocks):
            x, a_s = blk(x, a_s)

        #! all fp32 below
        x = self.norm(x)
        x = x.mean(dim=1).reshape(B,1,-1)
        x = x[:, 0]
        x = self.head(x)
        # make sure x have only 2-dimensions [x] -> [[x]], [[[x]]] -> [[x]]
        x = x.view(x.size(0), -1)
        
        return x

def q_test(model, full_precision_flag=False):
    net = Q_ResMLP24(model, full_precision_flag=full_precision_flag)
    return net