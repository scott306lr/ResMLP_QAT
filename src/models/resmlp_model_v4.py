# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath


__all__ = [
    'resmlp_24_v4'
]

# class Affine(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.alpha = nn.Parameter(torch.ones(dim))
#         self.beta = nn.Parameter(torch.zeros(dim))

#     def forward(self, x):
#         return self.alpha * x + self.beta 

def Affine(dim):
    return nn.Linear(dim, dim)

class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(384, 384))
        self.bias = nn.Parameter(torch.zeros(196,384))

    def forward(self, x):
        return x @ self.weight  + self.bias
    
class Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(196,196))

    def forward(self, x):
        return self.weight @ x
    
class layers_scale_mlp_blocks(nn.Module):

    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.ReLU, init_values=1e-4,num_patches = 196):
        super().__init__()
        self.inner = Inner()
        self.outer = Outer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.Linear(dim, dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(4.0 * dim), act_layer=act_layer, drop=drop)
        self.gamma_2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        residual = x
        x = torch.add(residual, self.drop_path(self.outer(self.inner(x))))
        
        residual = x
        x = torch.add(residual, self.drop_path(self.gamma_2(self.mlp(self.norm2(x)))))
        return x 



class resmlp_models(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,drop_rate=0.,
                 Patch_layer=PatchEmbed,act_layer=nn.ReLU,
                drop_path_rate=0.0,init_scale=1e-4):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  

        self.patch_embed = Patch_layer(
                img_size=img_size, patch_size=patch_size, in_chans=int(in_chans), embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        dpr = [drop_path_rate for i in range(depth)]

        self.blocks = nn.ModuleList([
            layers_scale_mlp_blocks(
                dim=embed_dim,drop=drop_rate,drop_path=dpr[i],
                act_layer=act_layer,init_values=init_scale,
                num_patches=num_patches)
            for i in range(depth)])


        self.norm = Affine(embed_dim)



        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        for i , blk in enumerate(self.blocks):
            x  = blk(x)

        x = self.norm(x)
        x = x.mean(dim=1).reshape(B,1,-1)

        return x[:, 0]

    def forward(self, x):
        x  = self.forward_features(x)
        x = self.head(x)
        return x 
  
@register_model
def resmlp_24_v4(pretrained=False,dist=False,dino=False,pretrained_cfg=False, **kwargs):
    model = resmlp_models(
        patch_size=16, embed_dim=384, depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-5,**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load("v3/decayed.pth", map_location='cpu')["model"]
        
        # modified_ckpt={}
        # for k, v in checkpoint.items():
        #     if "inner.weight" in k:
        #         modified_ckpt[k] = torch.diag(v)
        #     else:
        #         modified_ckpt[k] = v
        
        # checkpoint = modified_ckpt
        model.load_state_dict(checkpoint)
    return model
