import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.models import *
from src.model_analysis.visualize import layer_dist, act_dist, simulate_input, scale_plot, add_value_labels
from src.post_quant.utils import get_linear_layers, get_quant_layers

LOAD_DIR = 'share_resmlp_qat/model_best.pth.tar'

model = resmlp_24(pretrained=True)

qmodel = q_resmlp_v3(model)
qmodel.load_state_dict(torch.load(LOAD_DIR, map_location='cpu')['state_dict'])

from src.post_quant.utils import HookHandler
from src.data_utils import getTrainData, calibrate

def detach_qact_func(org_val, output):
    # if org_val is None:
    #     arr = []
    # else:
    #     arr = org_val

    # arr.append(output[0].cpu().detach().numpy())
    # return arr
    return output[0].cpu().detach().numpy()

def get_quantized_activation(qmodel):
    all_layers = []
    for i in range(24):
        all_layers.append(get_quant_layers(qmodel.blocks[i], prefix=f"{i}_"))

    activations = {}
    hook_handler = HookHandler()
    hook_handler.create_apply_hook(detach_qact_func, activations, all_layers)

    print("Loading a small piece of training data...")
    data_loader = getTrainData(dataset='imagenet', path="/mnt/disk1/imagenet/", batch_size=16, data_percentage=0.0001)
    print("Calibrating...")
    inputs = calibrate(data_loader, qmodel, eval=False, only_once=True)
    hook_handler.remove_hook()

    inputs = inputs[0] #np.array([val.detach().numpy() for val in inputs])

    all_layer_data = []
    for layer in all_layers:
        for n, m in layer:
            all_layer_data.append((n, np.array(activations[n])))

    return inputs, all_layer_data

inputs, outputs = get_quantized_activation(qmodel.eval())

import pickle
with open("share_resmlp_qat/inputs.pkl", "wb") as f:
    pickle.dump(inputs, f)

with open("share_resmlp_qat/outputs.pkl", "wb") as f:
    pickle.dump(outputs, f)