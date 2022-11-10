from bit_config import *
from utils import *
import torch
from torchsummary import summary
import seaborn as sns

model = resmlp_24(pretrained=True)
qmodel = q_resmlp24(model, full_precision_flag=True)

mdict = model.state_dict()
qmdict = qmodel.state_dict()

# for i, name in enumerate(mdict):
#   if i < 20:
#     print(f"{name}:")

# for i, name in enumerate(qmdict):
#   if i < 80:
#     if ('weight' in name or 'bias' in name) and 'integer' not in name:
#       print(f"{name}:")
  
# print( torch.equal(qmdict['quant_patch.proj.weight'], mdict['patch_embed.proj.weight']))
# print( torch.equal(qmdict['quant_patch.proj.bias'], mdict['patch_embed.proj.bias']))
# print( torch.equal(qmdict['layer0.norm1.weight'], mdict['blocks.0.norm1.weight']))
# print( torch.equal(qmdict['layer0.norm1.bias'], mdict['blocks.0.norm1.bias']))
# print( torch.equal(qmdict['layer0.attn.weight'], mdict['blocks.0.attn.weight']))
# print( torch.equal(qmdict['layer0.attn.bias'], mdict['blocks.0.attn.bias']))
# print( torch.equal(qmdict['layer0.norm2.weight'], mdict['blocks.0.norm2.weight']))
# print( torch.equal(qmdict['layer0.norm2.bias'], mdict['blocks.0.norm2.bias']))
# print( torch.equal(qmdict['layer0.mlp.fc1.weight'], mdict['blocks.0.mlp.fc1.weight']))
# print( torch.equal(qmdict['layer0.mlp.fc1.bias'], mdict['blocks.0.mlp.fc1.bias']))

# summary(model, (3, 224, 224), device='cpu')

ax = sns.heatmap(qmdict['layer0.norm1.weight'])
ax.plot()