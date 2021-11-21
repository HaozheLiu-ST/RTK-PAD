# -*- coding: utf-8 -*-
from __future__ import print_function
import math
import os
import random
import copy
import scipy
import numpy as np
from skimage.transform import resize, rotate
import torch
from torchvision import transforms,utils
from numpy import fliplr, flipud

def data_augment(x,prob=0.5):
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    for i in range(img_batches):
        im = imgs[i,:,:,:].transpose(0, 2)
        im = im.numpy()
        if random.random() >= prob:
            im = fliplr(im)
        if random.random() >= prob:
            im = flipud(im)
        if random.random() >= prob:
            temp = random.random()
            for j in range(8):
                if temp<= (j+1)/8.0:
                    im = rotate(im, j*45)
        im = im.copy()
        im = torch.from_numpy(im)
        im = im.transpose(2, 0).unsqueeze(0)
        imgs[i,:,:,:] = im
    return imgs
def cut_out(x, nholes, length, prob=0.5):
    if random.random() >= prob:
        return x
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    for i in range(img_batches):
        mask = np.ones((img_rows,img_cols))
        for n in range(nholes):
            c_x = np.random.randint(img_cols)
            c_y = np.random.randint(img_rows)

            y1 = np.clip(c_y - length // 2, 0, img_rows)
            y2 = np.clip(c_y + length // 2, 0, img_rows)
            x1 = np.clip(c_x - length // 2, 0, img_cols)
            x2 = np.clip(c_x + length // 2, 0, img_cols)
            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(imgs[i,:,:,:])
        imgs[i,:,:,:] = mask * imgs[i,:,:,:]
    return imgs

def ROI_extraction(x):
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    length = 96
    im_list = []
    for i in range(img_batches):
        im = imgs[i].numpy()
        patch_list = []
        std_list = []
        for h in range(0,int(img_rows-length),int(length/4)):
            for w in range(0,int(img_cols-length),int(length/4)):
                im_patch = im[:,h:h+length,w:w+length]
                m = np.mean(im_patch)
                std = np.std(im_patch)
                patch_list.append(im_patch)
                std_list.append(std)

        var_list = sorted(zip(std_list,range(len(std_list))),reverse=True)
        patch_order = [var[1] for var in var_list ]
        ordered_patch_list = []
        for i in patch_order:
            ordered_patch_list.append(patch_list[i])
        im_list.append(ordered_patch_list)
    return im_list
def patch_extraction(x, l_list, random_cond, random_number):
    img_batches, img_deps, img_rows, img_cols = x.shape
    img_list = ROI_extraction(x)
    switch_patch = []
    patchl = []
    if random_cond:
        for im,l in list(zip(img_list,l_list)):
            for i in range(random_number):
                num =np.random.randint(0,70)
                num = np.clip(num,0,len(im)-1)
                switch_patch.append(resizetensor(im[num]))
                patchl.append(l)
    else:
        for im,l in list(zip(img_list,l_list)):
            for i in range(70):#TOP 70
                switch_patch.append(resizetensor(im[i]))
                patchl.append(l)
    switch_patch = np.array(switch_patch)
    switch_patch = torch.from_numpy(switch_patch)
    patchl = np.array(patchl)
    patchl = torch.from_numpy(patchl)
    return switch_patch, patchl

def resizetensor(x, size=(224,224,3)):
    x = x.transpose(1,2,0)
    x = resize(x, size)
    x = x.transpose(2,0,1)
    return x

def local_pixel_shuffling(x, num_block=5, prob=0.5,length=32):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_batches, img_deps, img_rows, img_cols = x.shape
    for i in range(img_batches):
        for _ in range(num_block):
            block_noise_size_x = length
            block_noise_size_y = length
            noise_x = random.randint(0, img_rows-block_noise_size_x)
            noise_y = random.randint(0, img_cols-block_noise_size_y)
            window = orig_image[i, 0,noise_x:noise_x+block_noise_size_x,
                                   noise_y:noise_y+block_noise_size_y,
                               ]
            window = window.flatten()
            np.random.shuffle(window)
            window = window.reshape((block_noise_size_x,
                                     block_noise_size_y))
            for d in range(img_deps):
                image_temp[i, d,noise_x:noise_x+block_noise_size_x,
                          noise_y:noise_y+block_noise_size_y] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def patch_extraction_grad_cam(x, mask):
    img_batches, img_deps, img_rows, img_cols = x.shape
    imgs = copy.deepcopy(x)
    length = 96
    im_list = []
    for i in range(img_batches):
        im = imgs[i].numpy()
        m = mask[i]
        patch_list = []
        m_list = []
        for h in range(0,int(img_rows-length),int(length/4)):
            for w in range(0,int(img_cols-length),int(length/4)):
                im_patch = im[:,h:h+length,w:w+length]
                m_patch = m[h:h+length,w:w+length]
                patch_list.append(im_patch)
                m_patch_m = np.mean(m_patch)
                m_list.append(m_patch_m)

        var_list = sorted(zip(m_list,range(len(m_list))),reverse=True)
        patch_order = [var[1] for var in var_list ]
        ordered_patch_list = []
        for i in patch_order:
            ordered_patch_list.append(patch_list[i])
        im_list.append(ordered_patch_list)
    return im_list


