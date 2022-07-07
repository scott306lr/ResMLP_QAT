import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import copy

from run_model import evaluate_model, train_one_epoch, qat_train_model
from run_model import save_torchscript_model, load_torchscript_model
from datasets import data_loader #tfds_data_loader, 
import resmlp

from timm.models import create_model
from timm.data import Mixup
import torch.optim as optim
from timm.loss import SoftTargetCrossEntropy
from timm.utils import NativeScaler, get_state_dict, ModelEma

# Parameters
INPUT_SIZE = 224
DICT_PATH = 'ResMLP_S24_ReLU_99dense.pth' 

DATA_NAME = 'imagenet2012'
DATA_DIR = '/mnt/disk1/imagenet/'

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4#0.003

WORKERS = 0 #8

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
  print(f"BATCH_SIZE: {BATCH_SIZE}")
  print(f"LR: {LR}")
  print(f"EPOCHS: {EPOCHS}")

  # device = CUDA
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("Device: ", device)

  # set seed
  seed = 336 # args.seed + utils.get_rank()
  torch.manual_seed(seed)
  np.random.seed(seed)

  # check for best cudnn ops before training starts.
  cudnn.benchmark = True

  # build train/val dataset
  # create sampler (if dataset from tfds, can't apply sampler) (distributed ver. to be done)
  # build up dataloader
  data_loader_train, data_loader_val, NUM_CLASSES = data_loader(
      name=DATA_NAME,
      root=DATA_DIR,
      input_size=INPUT_SIZE, 
      batch_size=BATCH_SIZE,
      num_workers=WORKERS,
  )

  # additional data augmentation (mixup)
  mixup_fn = Mixup(
      mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
      prob=1.0, switch_prob=0.5, mode='batch',
      label_smoothing=0.1, num_classes=NUM_CLASSES)

  # create model
  model = create_model('resmlp_24', num_classes=NUM_CLASSES).cuda()
  model.load_state_dict(torch.load(DICT_PATH), strict=False)

  # fuse
  fused_model = model#copy.deepcopy(model)
  for basic_block_name, basic_block in fused_model.blocks.named_children():
    for sub_block_name, sub_block in basic_block.named_children():
      if sub_block_name == "mlp":
        torch.quantization.fuse_modules(
          sub_block, [['fc1', 'act']],
          inplace=True)

  # apply quant/dequant stabs
  quantized_model = QuantizedResMLP(model_fp32=fused_model)
  quantized_model.train()

  # quantization configurations
  quantized_model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
  print(quantized_model.qconfig)

  # train
  print("Training QAT Model...")
  quantized_model.train()
  torch.quantization.prepare_qat(quantized_model, inplace=True)
  model = qat_train_model(model, data_loader_train, data_loader_val, LR, EPOCHS, NUM_CLASSES, device, with_mixup=True, early_stop=2000)

  # convert weight to int8, replace model to quantized ver.
  quantized_model.cpu()
  torch.quantization.convert(quantized_model, inplace=True)
  quantized_model.eval()

  input_fp32 = torch.randn((1, 3, 224, 224), dtype=torch.float32, device="cpu")
  quantized_model(input_fp32)

  # save
  SAVE_PATH = 'modeltest.pth'
  save_torchscript_model(model=quantized_model, 
                          model_dir='qat_weights', 
                          model_filename='qat_Test0.pth')

  # load
  quantized_model = load_torchscript_model(model_filepath='qat_weights/qat_Test0.pth', device="cpu")
  quantized_model.eval()

  criterion = nn.CrossEntropyLoss().cuda()
  eval_loss, top1_acc, top5_acc = evaluate_model(model=quantized_model,
                                                  test_loader=data_loader_val,
                                                  device="cpu",
                                                  criterion=criterion)
  print("Epoch: {:02d} Eval Loss: {:.3f} Top1: {:.3f} Top5: {:.3f}".format(
      -1, eval_loss, top1_acc, top5_acc))

if __name__ == "__main__":
    main()


