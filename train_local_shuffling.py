# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import argparse
from utils.dataset import LivDetDataset
from models.network import mobilenetv3_large, PerceptualLoss, Unet
from utils.config import DefaultConfig
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms,utils
from torchvision.utils import save_image
from utils.utils import *
import os

config = DefaultConfig()

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--sensor", type=str, default='D')
args = parser.parse_args()

switch = {
    'O': 'Orcathus',
    'G': 'GreenBit',
    "D":'DigitalPersona'
}
t = switch[args.sensor]

def mobilev3net_init():
    train_data = LivDetDataset(config.data_path, [t,'train'])
    test_data = LivDetDataset(config.data_path, [t,'test'])

    net = mobilenetv3_large()
    model_dict = net.state_dict()
    pretrained_dict = torch.load(config.pretrain_image_pth_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.cuda()


    train_loader = DataLoader(train_data, batch_size=config.batch_size_for_train,shuffle=config.shuffle_train)

    test_loader = DataLoader(test_data, batch_size=config.batch_size_for_test,shuffle=config.shuffle_test)

    return net, train_loader, test_loader

def unet_init():
    train_data = LivDetDataset(config.data_path, [t,'train'])
    test_data = LivDetDataset(config.data_path, [t,'test'])

    net = mobilenetv3_large()

    model_dict = net.state_dict()
    pretrained_dict = torch.load(config.pretrain_image_pth_path)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    Perceptualnet = mobilenetv3_large()
    Perceptualnet.load_state_dict(model_dict)

    unet = Unet(net)

    perceptual_loss = PerceptualLoss(unet,Perceptualnet)

    perceptual_loss.cuda()

    train_loader = DataLoader(train_data, batch_size=config.batch_size_for_train,shuffle=config.shuffle_train)

    test_loader = DataLoader(test_data, batch_size=config.batch_size_for_test,shuffle=config.shuffle_test)

    return net, unet, perceptual_loss, train_loader, test_loader
def one_hot(x,classnum):
    out = torch.zeros(x.shape[0],classnum)
    index = torch.LongTensor(x).view(-1,1)
    return out.scatter_(dim=1,index=index,value=1)

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net, unet, perceptual_loss, train_loader, test_loader = unet_init()

    train_dataiter = iter(train_loader)
    test_dataiter = iter(test_loader)


    if config.opt == 'Adam':
        optimizer_unet = torch.optim.Adam(perceptual_loss.get_net_parameters(), lr=config.learning_rate)

    if config.opt == 'SGD':
        optimizer_unet = torch.optim.SGD(perceptual_loss.get_net_parameters(), lr=config.learning_rate, momentum=0.9)


    acc_list = []

    for i in range(5):
        acc_list.append(config.save_loss_thres)

    for i in range(config.Epoch_num):
        net.train()
        for o in range(config.opt_num):
            try:
                imgs, l = train_dataiter.next()
            except StopIteration:
                train_dataiter = iter(train_loader)
                imgs, l = train_dataiter.next()

            imgs,l = patch_extraction(imgs, l, random_cond=True, random_number=config.patch_random_number)

            imgs = data_augment(imgs)

            imgs = imgs.type(torch.FloatTensor).to(device)
            l = l.to(device).view(-1)

            x = local_pixel_shuffling(imgs)
            crite_unet = perceptual_loss(x,imgs)
            crite_unet.backward()
        optimizer_unet.step()
        optimizer_unet.zero_grad()
        if (i+1)%10 == 0:
            total = torch.zeros(1).squeeze().cuda()
            loss_percept = torch.zeros(1).squeeze().cuda()
            net.eval()
            with torch.no_grad():
                for j in range(config.test_batch_num):
                    try:
                        img, l = test_dataiter.next()
                    except StopIteration:
                        test_dataiter = iter(test_loader)
                        img, l = test_dataiter.next()
                    img,l = patch_extraction(img, l, random_cond=True, random_number=config.patch_random_number)
                    img = img.type(torch.FloatTensor).to(device)
                    l = l.to(device).view(-1)

                    x = local_pixel_shuffling(img,prob=1.)
                    loss_percept += perceptual_loss(x,img)
                    tx = x
                    label = img
                    re = unet(x)
                    total += len(l)
                loss_percept = (loss_percept/total).cpu().detach().data.numpy()
                print("\rBatch[{}/{}]  Perceptual_loss:{:4f}".format(i+1,config.Epoch_num, loss_percept), end='')
                if loss_percept <= config.save_loss_thres and loss_percept<=max(acc_list):
                    acc_list[acc_list.index(max(acc_list))] = loss_percept
                    acc_list = sorted(acc_list)
                    torch.save(net.state_dict(),os.path.join(config.save_shuffle_path,args.sensor+'_TOP_'+str(acc_list.index(loss_percept)+1)+'Net.pth'))
                    print("Save model! Path:{} Loss: {:4f}".format(os.path.join(config.save_shuffle_path,args.sensor+'Net.pth'), loss_percept))

                    torch.save(unet.state_dict(),os.path.join(config.save_shuffle_path,args.sensor+'_TOP_'+str(acc_list.index(loss_percept)+1)+'UNet.pth'))
                    print("Save model! Path:{} Loss: {:4f}".format(os.path.join(config.save_shuffle_path,args.sensor+'UNet.pth'), loss_percept))
                    save_image(tx,os.path.join(config.save_sample_path,args.sensor+'sampleInput-{}.png'.format(str(acc_list.index(loss_percept)+1))))
                    save_image(re,os.path.join(config.save_sample_path,args.sensor+'Reconstruct-{}.png'.format(str(acc_list.index(loss_percept)+1))))
                    save_image(label,os.path.join(config.save_sample_path,args.sensor+'Label-{}.png'.format(str(acc_list.index(loss_percept)+1))))
