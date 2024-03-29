import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.quantization.qconfig import QConfig
from torch.quantization.fake_quantize import FakeQuantize
from torch.quantization.observer import MovingAverageMinMaxObserver

from argparse import ArgumentParser
import numpy as np

from run_model import evaluate_model
from run_model import save_model, load_model, load_torchscript_model, save_torchscript_model

import resmlp
from timm.models import create_model

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
  parser.add_argument('--dict_path',  default='qat_weights/mixup_ema_MAMMobs/acc75.020_loss1.434_e0_i600.pth',    help='Location of int8 model weight.')
  parser.add_argument('--data_name',  default='imagenet2012',                 help='Name of the dataset.')
  parser.add_argument('--data_dir',   default='/mnt/disk1/imagenet/',         help='Directory of the dataset.')
  parser.add_argument('--tfds',       default=False,  action='store_true',    help='Enable if dataset is from tfds.')
  parser.add_argument('--batch_size', default=32,     type=int,               help='Dataset batch size.')
  parser.add_argument('--input_size', default=224,    type=int,               help='Model input size.')
  parser.add_argument('--epochs',     default=5,      type=int,               help='Epochs, will generate a .pth file on each epoch.')
  parser.add_argument('--workers',    default=0,      type=int,               help='Workers, for parallel computing.')
  args = parser.parse_args()
  
  DICT_PATH  = args.dict_path 
  DATA_NAME  = args.data_name
  DATA_DIR   = args.data_dir
  TFDS       = args.tfds

  BATCH_SIZE = args.batch_size
  INPUT_SIZE = args.input_size
  EPOCHS     = args.epochs

  WORKERS    = args.workers
  
  print(f"DICT_PATH: {DICT_PATH}")
  print(f"BATCH_SIZE: {BATCH_SIZE}")
  print(f"EPOCHS: {EPOCHS}")

  # device = CUDA
  #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  device = "cpu"
  print("Device:", device)

  # set seed
  seed = 336 # args.seed + utils.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)

  # check for best cudnn ops before training starts.
  cudnn.benchmark = True

  # build train/val dataset
  # create sampler (if dataset from tfds, can't apply sampler) (distributed ver. to be done)
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

  # train & save fp32 model on each epoch
  quantized_model = torch.quantization.prepare_qat(float_model, inplace=False)

  # convert weight to int8, replace model to quantized ver.
  quantized_model.cpu()
  torch.quantization.convert(quantized_model, inplace=True)
  quantized_model.eval()

  # load and evaluate
  print("Loading Model...")
  quantized_model = load_model(quantized_model, model_filepath=DICT_PATH, device=device)
  quantized_model.eval()
  
  print("Start evaluating...")
  criterion = nn.CrossEntropyLoss()
  eval_loss, top1_acc, top5_acc = evaluate_model(model=quantized_model,
                                                  test_loader=data_loader_val,
                                                  device=device,
                                                  criterion=criterion)
  print("Epoch: {:d} Eval Loss: {:.3f} Top1: {:.3f} Top5: {:.3f}".format(
      -1, eval_loss, top1_acc, top5_acc))

  save_model(model=quantized_model, 
                model_dir='qat_weights', 
                model_filename='qat_Test1.pth')

if __name__ == "__main__":
    main()


