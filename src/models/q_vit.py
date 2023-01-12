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
        self.proj = QConv(patch.proj)
        self.norm = nn.Identity()
        #self.act  = QAct(to_bit=to_bit)
    
    def get_scales(self):
        scales = []
        # scales += self.proj.get_scales("PatchEmbed_Conv")
        scales += self.act.get_scales("PatchEmbed_Act")
        return scales

    def forward(self, x, a_s=None):
        # forward using the quantized modules
        x, a_s = self.proj(x, a_s)
        x = x.flatten(2).transpose(1, 2)
        #x, a_s = self.act(x, a_s)
        #x = self.norm(x)

        return x, a_s

#class QMul()


class Q_Attention(nn.Module):
    def __init__(self, attention):
        super(Q_Attention, self).__init__()
        self.set_param(attention)
    def set_param(self, attention):
        self.qkv = QLinear(attention.qkv)
        self.proj = QLinear(attention.proj)
        self.act1 = QAct(return_fp=True)
        self.act2 = QAct()
        self.act3 = QAct()
        self.act_attn = QAct()
        self.num_heads = attention.num_heads
        self.scale = attention.scale
        self.attn_drop = attention.attn_drop
    def forward(self , x, a_s=None):
        B, N, C = x.shape
        x, a_s = self.qkv(x, a_s)
        # fp32 start
        x, a_s = self.act1(x, a_s)
        
        qkv = x.reshape(B, N, 3, self.num_heads,
                        C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        attn = (q @ k.transpose(-2, -1)) * self.scale # here the scale is for atttention itself not for quantization
        #attn, attn_a_s = self.act_attn(attn, a_s * a_s)
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # fp32 end
        x, a_s = self.act2(x)
        x, a_s = self.proj(x, a_s)
        x, a_s = self.act3(x, a_s)
        
        return x, a_s
        
class Q_Mlp(nn.Module):
    def __init__(self, mlp):
        super(Q_Mlp, self).__init__()
        self.set_param(mlp)

    def set_param(self, mlp):
        self.fc1 = QLinear(mlp.fc1)
        self.act1 = QAct(return_fp=True)
        self.gelu = torch.nn.GELU()
        self.act2 = QAct()
        self.fc2 = QLinear(mlp.fc2)
    
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
        x, a_s = self.act1(x, a_s)
        # row back to FP32
        x = self.gelu(x)
        x, a_s = self.act2(x)
        x, a_s = self.fc2(x, a_s)
        return x, a_s
    
QUANT_LAYERS = 12

class QViT_Block(nn.Module):
    def __init__(self, block, layer, res_to_bit):
        super(QViT_Block, self).__init__()
        self.layer = layer
        self.res_to_bit = res_to_bit
        self.set_param(block, layer)

    def set_param(self, block, layer):  
        self.norm1 = QLayerNorm(block.norm1)
        self.act1 = QAct()

        self.attn = Q_Attention(block.attn)
        self.drop_path = block.drop_path
        self.act2 = QAct()

        self.norm2 = QLayerNorm(block.norm2)
        self.act3 = QAct()
        self.mlp = Q_Mlp(block.mlp)
        self.act4 = QAct()

        
        #self.act_res1 = QAct()
        #self.act_res2 = QAct()
        # for residual add
        self.residual_attn = QResAct(to_bit=self.res_to_bit)
        if layer == QUANT_LAYERS-1:
            self.residual_mlp = QResAct(to_bit=self.res_to_bit, return_fp=True)
        else:
            self.residual_mlp = QResAct(to_bit=self.res_to_bit, return_fp=False)

    def get_scales(self):
        scales = []
        return scales
   
    # ! this implementation only works for per-tensor (transpose)
    def forward(self, x, a_s=None):
                    
        org_x, org_a_s = x, a_s

        # ------ Attention Part ------- START
        # 

        x = self.norm1(x, a_s) # currently the layerNorm return FP32
        x, a_s = self.act1(x) # quantize the output of layernorm
        x, a_s = self.attn(x, a_s)
        x, a_s = self.act2(x, a_s)
    
        x, a_s = self.residual_attn(x, a_s, org_x, org_a_s)
        
    
            # self.norm1.train()
            # self.act1.train()
            # self.attn.train()
            # self.act2.train()
            # self.residual_attn.train()
        '''
        if (not self.training):
            testing_x, testing_a = org_x, org_a_s
            self.norm1.train()
            self.act1.train()
            self.attn.train()
            self.act2.train()
            self.residual_attn.train()
            testing_x = testing_x * testing_a
            
            testing_x = self.norm1(testing_x, testing_a)
            testing_x, a_test = self.act1(testing_x)
            testing_x, a_test = self.attn(testing_x, a_test)
            testing_x, a_test = self.act2(testing_x, a_test)
            testing_x, a_test = self.residual_attn(testing_x, a_test, org_x, org_a_s)
            print("cosine similarity of Attention module:", F.cosine_similarity( testing_x,  x).sum(dim = -1))
        '''
        

        
        # ------ Attention Part ------- END
        
        # ----- MLP part ----- START
        org_x, org_a_s = x, a_s
        x = self.norm2(x, a_s)
        x, a_s = self.act3(x)
        
    
        
            
        x, a_s = self.mlp(x, a_s)    
        x, a_s = self.act4(x, a_s)
        
        #if not self.training:
        #    x = x * a_s
        #    org_x = org_x * org_a_s
            # self.norm1.train()
            # self.act1.train()
            # self.attn.train()
            # self.act2.train()
            # self.residual_attn.train()
            #self.norm2.train()
            #self.act3.train()
            #self.mlp.train()
            #self.act4.train()
        #    self.residual_mlp.train()
        
        
        #x = x + org_x
        #x, a_s = self.act_res1(x, a_s)
        x, a_s = self.residual_mlp(x, a_s, org_x, org_a_s)
        
        #print(x)
        
        #if not self.training:
        #    x = x / a_s
        '''
        if (not self.training):
            testing_x, testing_a = org_x, org_a_s
            self.norm2.train()
            self.act3.train()
            self.mlp.train()
            self.act4.train()
            self.residual_mlp.train()
            testing_x = testing_x * testing_a
            
            testing_x = self.norm2(testing_x, testing_a)
            testing_x, a_test = self.act3(testing_x)
            testing_x, a_test = self.mlp(testing_x, a_test)
            testing_x, a_test = self.act4(testing_x, a_test)
            testing_x, a_test = self.residual_mlp(testing_x, a_test, org_x, org_a_s)
            print("cosine similarity of MLP module:", F.cosine_similarity( testing_x,  x).sum(dim = -1))
            exit(1)
        '''
        # ----- MLP part ----- END
        return x, a_s

RES_RESCALE_BIT = 8
class Q_ViT(nn.Module):
    """
        Quantized ResMLP24 model.
    """
    def __init__(self, model):
        super().__init__()
        self.quant_input = QAct(to_bit=8)
        self.quant_patch = QPatchEmbed(model.patch_embed, to_bit=RES_RESCALE_BIT)
        self.QConcatAct = QConcatAct()
        self.QAddposEmbedAct = QResAct()
        self.blocks = nn.ModuleList([QViT_Block(model.blocks[i], layer=i, res_to_bit=RES_RESCALE_BIT) for i in range(12)])
        self.norm = model.norm#QLinear(model.norm) #model.norm
        self.head = model.head#QLinear(getattr(model, 'head'))
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.quant_pos_embed = QAct()
        
    def get_scales(self):
        scales = []
        # scales += self.quant_input.get_scales(f"Input_Act")
        #scales += self.quant_patch.get_scales()

        #for i, blk in enumerate(self.blocks):
            # if i >= ALL_FP_LAYER-1 and i <= ALL_FP_LAYER+1 : 
        #    scales += blk.get_scales()

        return scales

    def forward_features(self, x):
        B = x.shape[0]

        a_s = None
        x, a_s = self.quant_input(x, a_s)
        x, a_s = self.quant_patch(x, a_s)

        
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x, a_s = self.QConcatAct(x, cls_tokens, a_s)
        pos_emb, a_pos_emb = self.quant_pos_embed(self.pos_embed)
        x, a_s = self.QAddposEmbedAct(x, a_s, pos_emb, a_pos_emb)
        
        org_x, org_a = x, a_s
        
        #print(a_s)
       # if not self.training:
        #    x = x * a_s
        #x = (x / a_s).round().clamp(self.QAddposEmbedAct.Qn, self.QAddposEmbedAct.Qp)
        for i, blk in enumerate(self.blocks):
            
            #self.blocks[i].train()
            x, a_s = blk(x, a_s)
        '''
        testing_x = x
        
        
        if not self.training:
            testing_out, _ = self.blocks[0](org_x, org_a)
            self.blocks[0].train()
            
            x = org_x * org_a
            training_out, _ = self.blocks[0](x, org_a)
            #x, a_s = blk(x, a_s)
       
            print(F.cosine_similarity(training_out, testing_out).sum(dim = -1))
            exit(1)
        '''
        #! all fp32 below
        x = self.norm(x)
        x = x[:, 0]
        
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    


def q_vit(model):
    net = Q_ViT(model)
    return net