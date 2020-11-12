import torch
import numpy as np
from torchvision import transforms, utils
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

from configs import *

class TrainDataset(Dataset):
    def __init__(self, data_dir, csv_file_path, transform=None):
        data = pd.read_csv(csv_file_path, dtype=str)
        self.file_ids = data['id']
        self.labels = data['label_int']
        self.data_dir = data_dir
        self.transform = transform
        self.imgs = []

        message = f'loading imgs(in {data_dir}) to dataset...'
        for f_id in tqdm(self.file_ids, mininterval=3, desc=message, ncols= 100):
            img_path = self.data_dir + "/" + f_id + ".jpg"
            img = Image.open(img_path).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))

            self.imgs.append(img)
        print("Sucessfully loaded imgs")

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = int(self.labels[idx])
        
        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    def __len__(self):
        return len(self.imgs)

    def getWeightedRandomSampler(self):
        labels = np.array(self.labels, dtype="int64")
        class_sample_count = np.zeros(NUM_CLASSES)
        for i in range(NUM_CLASSES):
            class_sample_count[i]  = sum(labels == i)
            #print(f'target: {i} has {class_sample_count[i]} element')

        weight = 1. / class_sample_count
        # print(weight)

        samples_weight = torch.tensor([weight[t] for t in labels])
        # print(samples_weight)
        # print(len(samples_weight))

        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

        return sampler

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_ids = os.listdir(data_dir)

    def __getitem__(self, idx):
        img_path = self.data_dir + "/" + self.file_ids[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, self.file_ids[idx]

    def __len__(self):
        return len(self.file_ids)

    def get_file_ids(self):
        return self.file_ids

if __name__ == "__main__":
    pass