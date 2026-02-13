# utils.py
import torch
from torchvision.utils import save_image
import os

def save_output(batch, output, folder, epoch):
    os.makedirs(folder, exist_ok=True)
    for i in range(batch.size(0)):
        save_image(output[i], os.path.join(folder, f'epoch{epoch}_img{i}.png'))
