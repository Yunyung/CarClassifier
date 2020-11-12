import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

import timm
from pprint import pprint

from dataset import TrainDataset
from configs import *


# Data augmentation and normalization for training
# Just normalization for validation
transform_options = [
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5)
    #transforms.RandomRotation(degrees=[-30, 30])
    #transforms.GaussianBlur(kernel_size=5)
    transforms.RandomAffine(0, shear=20)
]

data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(IMG_SIZE)
        # ,
        #transforms.RandomHorizontalFlip()
        # ,
        transforms.RandomApply([
             transforms.RandomChoice(transform_options)
        ], p=1.0)
       
    ]),
    'val': transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


img_path = "data/training_data/"  + "000001.jpg"
img = Image.open(img_path).convert("RGB")
img = data_transforms["train"](img)
img.show()
img.save("RandomAffine.png")