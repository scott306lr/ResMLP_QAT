import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
import numpy as np

from run_model import qat_train_model
from run_model import load_model, save_torchscript_model

import wandb
import resmlp
from timm.models import create_model

import torch.quantization
import torch.quantization._numeric_suite as ns

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
  parser.add_argument('--tfds',       default=False,  type=bool,              help='Enable if dataset is from tfds.')
  parser.add_argument('--batch_size', default=64,     type=int,               help='Dataset batch size.')
  parser.add_argument('--input_size', default=224,    type=int,               help='Model input size.')
  parser.add_argument('--epochs',     default=5,      type=int,               help='Epochs, will generate a .pth file on each epoch.')
  parser.add_argument('--lr',         default=1e-4,   type=float,             help='Learning rate.')
  parser.add_argument('--mixup',      default=True,   type=bool,              help='Enable mixup on training.')
  parser.add_argument('--workers',    default=0,      type=int,               help='Workers, for parallel computing.')
  parser.add_argument('--save_dir',   default='qat_weights',                 help='Directory to save after each epoch.')
  parser.add_argument('--per_save',   default=10,     type=int,               help='Amount of data to train before jumping to next epoch.')
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

  # status
  print(f"DICT_PATH: {DICT_PATH}")
  print(f"BATCH_SIZE: {BATCH_SIZE}")
  print(f"LR: {LR}")
  print(f"EPOCHS: {EPOCHS}")

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
  # if not TFDS:
  #   from datasets import data_loader
  #   data_loader_train, data_loader_val, NUM_CLASSES = data_loader(
  #       name=DATA_NAME,
  #       root=DATA_DIR,
  #       input_size=INPUT_SIZE, 
  #       batch_size=BATCH_SIZE,
  #       num_workers=WORKERS,
  #   )
  # else:
  #   from datasets import tfds_data_loader
  #   data_loader_train, data_loader_val, NUM_CLASSES = tfds_data_loader(
  #       name=DATA_NAME,
  #       root=DATA_DIR,
  #       input_size=INPUT_SIZE, 
  #       batch_size=BATCH_SIZE,
  #       num_workers=WORKERS,
  #   )

  NUM_CLASSES = 224

  # create model
  float_model = create_model('resmlp_24', num_classes=NUM_CLASSES).to(device)

  # fuse
  for basic_block_name, basic_block in float_model.blocks.named_children():
    for sub_block_name, sub_block in basic_block.named_children():
      if sub_block_name == "mlp":
        torch.quantization.fuse_modules(
          sub_block, [['fc1', 'act']],
          inplace=True)

  # apply quant/dequant stabs
  float_model = QuantizedResMLP(module=float_model)

  # quantization configurations
  float_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
  print(float_model.qconfig)

  # train & save fp32 model on each epoch
  print("Training Model with QAT...")
  quantized_model = torch.quantization.prepare_qat(float_model, inplace=False)
  quantized_model.train()
  #quantized_model = qat_train_model(quantized_model, data_loader_train, data_loader_val, LR, EPOCHS, NUM_CLASSES, device, with_mixup=WITH_MIXUP, save_interval=PER_SAVE, save_dir=SAVE_DIR)

  # convert weight to int8, replace model to quantized ver.
  quantized_model.cpu()
  torch.quantization.convert(quantized_model, inplace=True)
  quantized_model.eval()

  quantized_model = load_model(quantized_model, DICT_PATH, "cpu")
  
  dd = quantized_model.module.state_dict()
  for a in dd:
    q = dd[a][0]
    print(f"{a}:")
    #print(q[0])
    print(q.dequantize())
    break


if __name__ == "__main__":
    main()


