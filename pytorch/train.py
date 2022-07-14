import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.quantization.qconfig import QConfig
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver

from argparse import ArgumentParser
import numpy as np

from run_model import qat_train_model
from run_model import load_model, save_torchscript_model

import resmlp
from timm.models import create_model

import wandb

class QuantizedResMLP(nn.Module):
    def __init__(self, module):
        super(QuantizedResMLP, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.module = module

    def forward(self, x):
        x = self.quant(x)
        x = self.module(x)
        x = self.dequant(x)
        return x

def main():
  parser = ArgumentParser(description="Quantize Aware Training for ResMLP, also supports tfds datasets.")
  parser.add_argument('--dict_path',  default='fp32_weights/ResMLP_S24_ReLU_fp32_80.602.pth',  help='Location of fp32 model weight.')
  parser.add_argument('--data_name',  default='imagenet2012',                 help='Name of the dataset.')
  parser.add_argument('--data_dir',   default='/mnt/disk1/imagenet/',         help='Directory of the dataset.')
  parser.add_argument('--tfds',       default=False,  action='store_true',    help='Enable if dataset is from tfds.')
  parser.add_argument('--batch_size', default=64,     type=int,               help='Dataset batch size.')
  parser.add_argument('--input_size', default=224,    type=int,               help='Model input size.')
  parser.add_argument('--epochs',     default=5,      type=int,               help='Epochs, will generate a .pth file on each epoch.')
  parser.add_argument('--lr',         default=1e-6,   type=float,             help='Learning rate.')
  parser.add_argument('--mixup',      default=False,  action='store_true',    help='Enable mixup on training.')
  parser.add_argument('--workers',    default=8,      type=int,               help='Workers, for parallel computing.')
  parser.add_argument('--save_dir',   default='qat_weights',                  help='Directory to save after each epoch.')
  parser.add_argument('--per_save',   default=10,     type=int,               help='Amount of data to train before jumping to next epoch.')
  parser.add_argument('--wandb',      default=False,  action='store_true',    help='Run with wandb.')
  args = parser.parse_args()
  
  DICT_PATH  = args.dict_path 
  DATA_NAME  = args.data_name
  DATA_DIR   = args.data_dir
  TFDS       = args.tfds

  BATCH_SIZE = args.batch_size
  INPUT_SIZE = args.input_size
  EPOCHS     = args.epochs
  LR         = args.lr
  WITH_MIXUP = args.mixup

  WORKERS    = args.workers
  SAVE_DIR   = args.save_dir
  PER_SAVE   = args.per_save

  WANDB      = args.wandb
    
  print(f"WANDB: {WANDB}")
  if WANDB:
    # wandb login
    # wandb.login(key=)
    wandb.init(project="resmlp_qat")
    wandb.config = {
      "learning_rate": LR,
      "epochs": EPOCHS,
      "batch_size": BATCH_SIZE
    }

  # status
  print(f"DICT_PATH: {DICT_PATH}")
  print(f"EPOCHS: {EPOCHS}")
  print(f"BATCH_SIZE: {BATCH_SIZE}")
  print(f"LR: {LR}")
  print(f"PER_SAVE: {PER_SAVE}")
  print(f"WORKERS: {WORKERS}")

  # device = CUDA
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device:", device)

  # set seed
  seed = 336 # args.seed + utils.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)

  # check for best cudnn ops before training starts.
  cudnn.benchmark = True

  # build train/val dataset
  # create sampler (if dataset from tfds, can't apply sampler)
  # build up dataloader
  if not TFDS:
    from datasets import data_loader
    data_loader_train, data_loader_val, NUM_CLASSES = data_loader(
        name=DATA_NAME,
        root=DATA_DIR,
        input_size=INPUT_SIZE, 
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
    )
  else:
    from datasets import tfds_data_loader
    data_loader_train, data_loader_val, NUM_CLASSES = tfds_data_loader(
        name=DATA_NAME,
        root=DATA_DIR,
        input_size=INPUT_SIZE, 
        batch_size=BATCH_SIZE,
        num_workers=WORKERS,
    )

  # create model
  float_model = create_model('resmlp_24', num_classes=NUM_CLASSES).to(device)
  float_model = load_model(float_model, DICT_PATH, device)

  # fuse
  for basic_block_name, basic_block in float_model.blocks.named_children():
    for sub_block_name, sub_block in basic_block.named_children():
      if sub_block_name == "mlp":
        torch.quantization.fuse_modules(
          sub_block, [['fc1', 'act']],
          inplace=True)

  # apply quant/dequant stabs
  #float_model1 = torch.quantization.add_quant_dequant(float_model)
  float_model = QuantizedResMLP(module=float_model)

  # quantization configurations
  float_model.qconfig = QConfig(
    activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, 
                                  #quant_min=-128, quant_max=127, dtype=torch.qint8,
                                  quant_min=0, quant_max=255, dtype=torch.quint8, 
                                  qscheme=torch.per_tensor_symmetric, reduce_range=False),
    weight=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, 
                                  quant_min=-128, quant_max=127, dtype=torch.qint8, 
                                  #quant_min=0, quant_max=255, dtype=torch.quint8,
                                  qscheme=torch.per_tensor_symmetric, reduce_range=False)
  )
  print(float_model.qconfig)

  # train & save fp32 model on each epoch
  print("Training Model with QAT...")
  quantized_model = torch.quantization.prepare_qat(float_model, inplace=False)
  quantized_model.train()
  quantized_model = qat_train_model(quantized_model, data_loader_train, data_loader_val, LR, EPOCHS, NUM_CLASSES, device, with_mixup=WITH_MIXUP, save_interval=PER_SAVE, save_dir=SAVE_DIR, wandb=WANDB)

  # convert weight to int8, replace model to quantized ver.
  # quantized_model.cpu()
  # torch.quantization.convert(quantized_model, inplace=True)
  # quantized_model.eval()

  # # save int8 model
  # save_torchscript_model(model=quantized_model, 
  #                         model_dir='qat_weights', 
  #                         model_filename='qat_Test1.pth')

if __name__ == "__main__":
    main()


