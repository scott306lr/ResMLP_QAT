import math
import sys, os
from xmlrpc.client import Boolean
from tqdm import tqdm

import torch
from typing import Iterable, Optional
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from timm.utils import AverageMeter, accuracy, ModelEma
from timm.data import Mixup

import torch.optim as optim
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

import wandb


def qat_train_model(model, train_loader, test_loader, learning_rate, epochs, num_classes, device, with_mixup, with_ema, log_interval=10, save_interval=-1, save_dir='qat_weights', wandb=False):
    model.to(device)
    train_criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=0.9,
                            weight_decay=1e-4)
    # will never reach milestone since we use only few epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=[100, 150],
                                                        gamma=0.1,
                                                        last_epoch=-1)     
    # additional data augmentation (mixup)
    mixup_fn = None
    if with_mixup:
        train_criterion = SoftTargetCrossEntropy()
        mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
            prob=1.0, switch_prob=0.5, mode='batch',
            label_smoothing=0.1, num_classes=num_classes)
    
    # Exponential Moving Average (ema)
    model_ema = None
    if with_ema:
        model_ema = ModelEma(
            model,
            decay=0.99996,
            device='cpu',
            resume='')

    for epoch in tqdm(range(epochs)):
        model.train()
        train_one_epoch(model=model, criterion=train_criterion,
                  train_loader=train_loader, test_loader=test_loader, optimizer=optimizer,
                  device=device, epoch=epoch, max_norm=None, log_interval=log_interval, save_interval=save_interval, save_dir=save_dir,
                  model_ema=model_ema, mixup_fn=mixup_fn, use_wandb=wandb)
        scheduler.step()

    return model

def train_one_epoch(model: torch.nn.Module, criterion: CrossEntropyLoss,
                    train_loader: Iterable, test_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,  log_interval: int = 10, save_interval: int = -1, save_dir: str='qat_weights',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, use_wandb: bool = False):
    eval_criterion = CrossEntropyLoss()                
    model.train()
    model.to(device)
    
    pbar = tqdm(train_loader, leave=False)
    for i, (inputs, labels) in enumerate(pbar):
        model.to(device)
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        if mixup_fn is not None:
            inputs, labels = mixup_fn(inputs, labels)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss_val = loss.item()

        loss.backward()
        optimizer.step()

        if model_ema is not None:
            model_ema.update(model)
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss_val))
            sys.exit(1)

        # calculate loss and accuracy
        if i % log_interval == 0:
            if mixup_fn:
                prec1, prec5 = accuracy(outputs.data, torch.argmax(labels.data, dim=1), topk=(1, 5))
            else:
                prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            output_str = "i: {:d} Eval Loss: {:.3f} Top1: {:.3f} Top5: {:.3f}".format(i, loss_val, prec1, prec5)

            # log
            if use_wandb:
                wandb.log({"train_loss": loss_val, "train_acc": prec1})
        else:
            output_str = "i: {:02d} Eval Loss: {:.3f}".format(i, loss_val)
        pbar.set_description(output_str)

        # Evaluation
        if i > 0 and i % save_interval == 0: 
            model.cpu()
            qmodel = torch.quantization.convert(model, inplace=False)
            qmodel.eval()

            eval_loss, top1_acc, top5_acc = evaluate_model(model=qmodel,
                                                    test_loader=test_loader,
                                                    device="cpu",
                                                    criterion=eval_criterion)
            print("Epoch: {:d} Iter: {:d} Eval Loss: {:.3f} Top1: {:.3f} Top5: {:.3f}".format(
                epoch, i, eval_loss, top1_acc, top5_acc))
            
            # log
            if use_wandb:
                wandb.log({"test_loss": eval_loss, "test_acc": top1_acc})

            # save
            fname = 'acc{:.3f}_loss{:.3f}_e{:d}_i{:d}.pth'.format(
                top1_acc, eval_loss, epoch, i)
            save_model(qmodel, save_dir, fname)

    return


def evaluate_model(model, test_loader, device, criterion=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    model.to(device)
    
    for inputs, labels in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        #_, preds = torch.max(outputs, 1)
        if criterion is not None:
            loss = criterion(outputs, labels)
        else:
            loss = 0
        
        
        prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
    
    return losses.avg, top1.avg, top5.avg

def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)
    return model

def save_model(model, model_dir, model_filename):

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.save(model.state_dict(), model_filepath)


def load_model(model, model_filepath, device):

    model.load_state_dict(torch.load(model_filepath, map_location=device))

    return model
