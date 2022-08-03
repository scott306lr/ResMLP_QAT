from torchvision import models
from aimet_torch.cross_layer_equalization import equalize_model
from utils import *

def cross_layer_equalization_auto(model):
    model = model.eval()
    # Performs BatchNorm fold, Cross layer scaling and High bias folding
    equalize_model(model, input_shape)


model = resmlp_24(pretrained=True)
input_shape = (1, 3, 224, 224)

cross_layer_equalization_auto(model, input_shape)