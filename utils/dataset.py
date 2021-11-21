# -*- coding: utf-8 -*-
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
import skimage.io as io
import skimage.transform as trans
import skimage
from utils.config import DefaultConfig
import numpy as np
import random
import re
from utils.utils import *

preprocess = transforms.Compose(
                            [
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                             ]
                            )

class LivDetDataset(Dataset):
    def __init__(self,txtpath,search,transform=preprocess):
        with open(txtpath, mode='r') as ftxt:
            pathl = ftxt.readlines()
        imgs = []
        for row in pathl:
            row = row.replace('\n','')
            cond = True
            for s in search:
                if s not in row:
                    cond = False
            if cond:
                if 'Live' in row:
                    imgs.append([row,0])
                if 'Fake' in row:
                    imgs.append([row,1])
        self.imgs = imgs
        self.transform = transform
    def __getitem__(self, index):
        fp, label = self.imgs[index]
        img = io.imread(fp, as_gray=True)
        if self.transform is not None:
            if len(img.shape)==2:
                img = img.reshape(img.shape[0],img.shape[1],1)
            if img.shape[-1]==1:
                img = np.tile(img,(1,1,3))
            if img.shape[-1]==4:
                img = img[:,:,:3]
            img = self.transform(img)
        return img,label
    def __len__(self):
        return len(self.imgs)

class Patch_LivDetDataset(Dataset):
    def __init__(self, dataset, random_number=3):
        self.dataset = dataset
        self.random_number = random_number
    def __getitem__(self, index):
        imgs, l  = self.dataset[index]
        l = np.array([l])
        imgs = imgs.unsqueeze(0)
        imgs,l = patch_extraction(imgs, l, random_cond=True, random_number=self.random_number)
        imgs = data_augment(imgs)
        return imgs, l
    def __len__(self):
        return self.dataset.__len__()


