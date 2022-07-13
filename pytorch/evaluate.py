import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
import numpy as np

from run_model import evaluate_model
from run_model import save_model, load_model, load_torchscript_model, save_torchscript_model

import resmlp
from timm.models import create_model

class QuantizedResMLP(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResMLP, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def main():
  parser = ArgumentParser(description="Quantize Aware Training for ResMLP, also supports tfds datasets.")
  parser.add_argument('--dict_path',  default='qat_weights/qat_Test1.pth',    help='Location of int8 model weight.')
  parser.add_argument('--data_name',  default='imagenet2012',                 help='Name of the dataset.')
  parser.add_argument('--data_dir',   default='/mnt/disk1/imagenet/',         help='Directory of the dataset.')
  parser.add_argument('--tfds',       default=False,  type=bool,              help='Enable if dataset is from tfds.')
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

  # # create model
  # model = create_model('resmlp_24', num_classes=NUM_CLASSES).to(device)
  # model = load_model(model, DICT_PATH, device)

  # # fuse
  # fused_model = model#copy.deepcopy(model)
  # for basic_block_name, basic_block in fused_model.blocks.named_children():
  #   for sub_block_name, sub_block in basic_block.named_children():
  #     if sub_block_name == "mlp":
  #       torch.quantization.fuse_modules(
  #         sub_block, [['fc1', 'act']],
  #         inplace=True)

  # # apply quant/dequant stabs
  # quantized_model = QuantizedResMLP(model_fp32=fused_model)

  # load and evaluate
  quantized_model = load_torchscript_model(model_filepath=DICT_PATH, device=device)
  quantized_model.eval()
  
  criterion = nn.CrossEntropyLoss()
  eval_loss, top1_acc, top5_acc = evaluate_model(model=quantized_model,
                                                  test_loader=data_loader_val,
                                                  device=device,
                                                  criterion=criterion)
  print("Epoch: {:d} Eval Loss: {:.3f} Top1: {:.3f} Top5: {:.3f}".format(
      -1, eval_loss, top1_acc, top5_acc))

  save_torchscript_model(model=quantized_model, 
                          model_dir='qat_weights', 
                          model_filename='qat_Test1.pth')

if __name__ == "__main__":
    main()


