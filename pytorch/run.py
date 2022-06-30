import numpy as np
import torch
import torch.nn as nn

from functools import partial

from timm.models.vision_transformer import Mlp, PatchEmbed , _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_,  DropPath

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta    
    
class layers_scale_mlp_blocks(nn.Module):

    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.ReLU,init_values=1e-4,num_patches = 196):
        super().__init__()
        self.norm1 = nn.Linear(384, 384)
#         self.norm1 = Affine(384)
        self.attn = nn.Linear(num_patches, num_patches)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.Linear(384, 384)
#         self.norm2 = Affine(384)
        self.mlp = Mlp(in_features=dim, hidden_features=int(4.0 * dim), act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Linear(384, 384, bias=False)
        self.gamma_2 = nn.Linear(384, 384, bias=False)
#         self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
#         self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
    def forward(self, x):
        
        
        # org quant
        residual = x
        x = self.norm1(x)
        x = x.transpose(1,2)
        x = self.attn(x)
        x = x.transpose(1,2)
        x = self.gamma_1(x)
                
        x = self.dequant(residual) + self.dequant(x)
        x = self.quant(x)
#         x = residual + self.gamma_1(x)

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.gamma_2(x)
        x = self.dequant(residual) + self.dequant(x)
        x = self.quant(x)
#         x = residual + self.gamma_2 * x
        
        '''
        # tmp no use
        residual = x
        x = self.norm1(x)
        x = x.transpose(1,2)
        x = self.attn(x)
        x = x.transpose(1,2)
        x = self.gamma_1(x)
                
#         x = residual + self.gamma_1(x)

        residual = x
        x = self.quant(x)
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.gamma_2(x)
        x = residual + self.dequant(x)
#         x = residual + self.gamma_2 * x

        '''
        return x 


class resmlp_models(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,drop_rate=0.,
                 Patch_layer=PatchEmbed,act_layer=nn.ReLU,
                drop_path_rate=0.0,init_scale=1e-4):
        super().__init__()

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

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


        self.norm = nn.Linear(384, 384)
#         self.norm = Affine(384)


        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
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
#         x = self.quant(x)
        x = self.forward_features(x)
        x = self.head(x)
#         x = self.dequant(x)
        return x 


org_model = resmlp_models(
        patch_size=16, embed_dim=384, depth=24,
        Patch_layer=PatchEmbed,
        init_scale=1e-6)

a,b=org_model.load_state_dict(torch.load("ResMLP_S24_ReLU_99dense.pth"), strict=False)
org_model.eval()


def validate(val_loader, model, criterion, print_freq):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg, top5.avg


import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def data_loader(root, batch_size=256, workers=1, pin_memory=True):
    traindir = os.path.join(root, 'train')
    valdir = os.path.join(root, 'val')
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.RandomResizedCrop(224, interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(int((256/224)*224),interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        sampler=None
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory
    )
    return train_loader, val_loader

batch_size = 32
val_dir = '/mnt/disk1/imagenet/'
workers = 8
train_loader, val_loader = data_loader(val_dir, batch_size, 8, False)

from torch.quantization.qconfig import QConfig
from torch.quantization.fake_quantize import FakeQuantize, default_weight_fake_quant
from torch.quantization.observer import MovingAverageMinMaxObserver



model_fp32.qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255,
                                                                dtype=torch.quint8,
                                                                qscheme=torch.per_tensor_symmetric,
                                                                reduce_range=False),
                              weight=default_weight_fake_quant)
# model_fp32_fused = torch.quantization.fuse_modules(model_fp32,[['mlp','relu']])
model_prepared = torch.quantization.prepare_qat(model_fp32)


# train loop
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model_prepared.parameters(), lr = 0.0001)
model_prepared.cpu()


train_one_epoch(model_prepared, criterion, optimizer, train_loader, device='cpu', ntrain_batches=600)
# model_prepared, val_acc_history = train_model(model_prepared, train_loader, val_loader, criterion, optimizer, num_epochs=10, is_inception=False)

model_prepared.eval()
model_prepared.cpu()
model_int8 = torch.quantization.convert(model_prepared)


# run the model, relevant calculations will happen in int8
input_fp32 = torch.randn(1, 3, 224, 224)
res = model_int8(input_fp32)
