import torch
import torch.nn as nn
import copy

from quantizer import *
from timm.models.vision_transformer import Mlp, PatchEmbed, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath


class QPatchEmbed(nn.Module):
    def __init__(self, patch, bias_bit=None, to_bit=8):
        super(QPatchEmbed, self).__init__()
        self.bias_bit = bias_bit
        self.set_param(patch, to_bit)

    def set_param(self, patch, to_bit):
        self.proj = QConv(patch.proj)
        self.norm = nn.Identity()
        self.act = QAct(to_bit=to_bit, obs_mode="patch_mod")

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
        self.fc1 = QLinear(mlp.fc1)
        self.relu = torch.nn.ReLU()
        self.act = QAct()
        self.fc2 = QLinear(mlp.fc2)

    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.fc1(x, a_s)
        x = self.relu(x)
        x, a_s = self.act(x, a_s)
        x, a_s = self.fc2(x, a_s)
        return x, a_s


QUANT_LAYERS = 24


class QLayer_Block(nn.Module):
    def __init__(self, block, layer, res_to_bit):
        super(QLayer_Block, self).__init__()
        self.layer = layer
        self.res_to_bit = res_to_bit
        self.set_param(block, layer)

    def set_param(self, block, layer):
        if layer == 0:
            self.crossPatch = QCrossPatch([block.norm1, block.attn, block.gamma_1], obs_mode=[
                                          'lab_weight', 'act0_mod', 'out0_mod'])
        else:
            self.crossPatch = QCrossPatch([block.norm1, block.attn, block.gamma_1], obs_mode=[
                                          'lab_weight', 'lab_inner_act', 'lab_outer'])
        self.add_1 = QResAct(to_bit=self.res_to_bit, obs_mode='lab_res1')

        self.linear1 = QCrossLayer1(
            [block.norm2, block.mlp.fc1], obs_mode='lab_cross1')
        self.relu = torch.nn.ReLU()
        self.act3 = QAct()
        self.linear2 = QCrossLayer2(
            [block.mlp.fc2, block.gamma_2], obs_mode='lab_cross2')

        if layer == QUANT_LAYERS-1:
            # dequant output back to fp
            self.add_2 = QResAct(to_bit=self.res_to_bit,
                                 return_fp=True, obs_mode='lab_res2')
        else:
            self.add_2 = QResAct(to_bit=self.res_to_bit,
                                 return_fp=False, obs_mode='lab_res2')

    # ! this implementation only works for per-tensor (transpose)
    def forward(self, x, a_s=None):
        org_x, org_a_s = x, a_s

        # ----- Cross-patch sublayer ----- START
        x, a_s = self.crossPatch(x, a_s)

        x, a_s = self.add_1(x, a_s, org_x, org_a_s)
        # ----- Cross-patch sublayer ----- END
        org_x, org_a_s = x, a_s

        # ---- Cross-channel sublayer ---- START
        x, a_s = self.linear1(x, a_s)
        x = self.relu(x)
        x, a_s = self.act3(x, a_s)

        x, a_s = self.linear2(x, a_s)
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
        self.quant_input = QAct(to_bit=8)
        self.quant_patch = QPatchEmbed(
            model.patch_embed, to_bit=RES_RESCALE_BIT)
        self.blocks = nn.ModuleList([QLayer_Block(
            model.blocks[i], layer=i, res_to_bit=RES_RESCALE_BIT) for i in range(24)])
        self.norm = model.norm
        self.head = model.head

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


def q_resmlp_v3(model):
    net = Q_ResMLP24(model)
    return net
