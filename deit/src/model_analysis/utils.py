import torch
import numpy as np
from torchvision import transforms
import os, time

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

def same_output(model, org_model, eps=1e-8):
    x = simulate_input()
    return torch.allclose(model(x), org_model(x), atol=eps)