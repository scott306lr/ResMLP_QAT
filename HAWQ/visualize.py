import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt


def my_boxplot(data, labels, name, ax):
  if ax is None:
    fig, ax = plt.subplots(1,1, figsize=(20, 5))
    
  ax.boxplot(data, labels=labels)
  ax.set_title(name, size=30)
  ax.tick_params(labelrotation=30)
  for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

def simulate_input():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    train_resolution = 224  
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(train_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

    # simulate input
    x = np.array(np.rint(np.random.rand(500, 375, 3) * 255), dtype=np.uint8)
    x = transform(x).unsqueeze(0)
    return x

class HookHandler():
  def __init__(self):
    self.handlers = []
  
  def apply_hook(self, hook_func, model_layers):
    for layer in model_layers:
      for n, m in layer:
        self.handlers.append( m.register_forward_hook(hook_func(n)) )
  
  def remove_hook(self):
    for handle in self.handlers:
      handle.remove()
    self.handlers = []

def create_act_hook(to_dict):
  def rec_act_name(name):
    def hook(model, input, output):
        to_dict[name] = output.detach().numpy().flatten()
    return hook
  return rec_act_name

def get_linear_layers(model, specify_names=None, prefix=""):
    linear_layers = []
    for name, module in model.named_modules():
        if (specify_names is not None) and (name not in specify_names):
            continue 
        
        if isinstance(module, nn.Linear):
            linear_layers.append((f'{prefix}{name}', module))
    return linear_layers

def layer_dist(model, start, end, show_layers=None, type="weight", name="Distribution of Each Layer", ax=None):
  data = []
  labels = []
  for i in range(start, end+1):
    todo_layer = model.blocks[i]
    tlist = get_linear_layers(todo_layer, prefix=f"{i}-", specify_names=show_layers)

    for n, m in tlist:
      if type == "weight":
        layers = m.weight
      elif type == "bias":
        layers = m.bias
      else: raise NameError(f'Type:{type} not found')
      
      if layers is not None:
        val = layers.detach().numpy().flatten()
        data.append(val)
        labels.append(n)

  my_boxplot(data, labels, name, ax)

def act_dist(model, start, end, show_layers=None, name="Activation Distribution from Gamma_1/Gamma_2 for Each Layer", ax=None):
  model_layers = []
  for i in range(start, end+1):
      todo_layer = model.blocks[i]
      model_layers.append(get_linear_layers(todo_layer, specify_names=show_layers, prefix=f"{i}-"))

  activations = {}
  hook_handler = HookHandler()
  hook_handler.apply_hook(create_act_hook(to_dict=activations), model_layers)
  model(simulate_input())
  hook_handler.remove_hook()
  
  data = []
  labels = []
  for layer in model_layers:
    for n, m in layer:
      data.append(activations[n])
      labels.append(n)
  
  my_boxplot(data, labels, name, ax)