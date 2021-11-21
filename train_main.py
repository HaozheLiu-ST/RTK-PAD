# -*- coding: utf-8 -*-
import sys

import torch
import torch.nn as nn
import numpy as np
import argparse
from utils.dataset import LivDetDataset, Patch_LivDetDataset
from utils.utils import local_pixel_shuffling, data_augment, cut_out
import os
from models.network import mobilenetv3_large
from utils.config import DefaultConfig
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import *
from utils.eval import ACE_TDR_Cal
import time

config = DefaultConfig()

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--train_sensor", type=str, default='D')
parser.add_argument("--mode", type=str, default='Whole')  # [Patch, Whole]
parser.add_argument("--savedir", type=str, default='./model/test/')
args = parser.parse_args()
"""
print information of config
"""

print("Training Protocol")
print("Epoch Total number:{}".format(config.Epoch_num))
print("Train Batch Size is {:^.2f} x {:^.2f}".format(config.batch_size_for_train, config.opt_num))
print("Test Batch Size is {:^.2f} x {:^.2f}".format(config.test_batch_num, config.batch_size_for_test))
print("Shuffle Data for Training is {}".format(config.shuffle_train))
print("Shuffle Data for Test is {}".format(config.shuffle_test))
print("Test Split Times: {:^.2f}".format(config.test_split_time))

print("Learning Parameters")
if args.mode == 'Whole':
    print("Cut Out is {}".format(config.cut_out))
    print("Cut Out Holes is {:^.2f}".format(config.n_holes))
    print("Cut Out Holes Length is {:^.2f}".format(config.length))
if args.mode == 'Patch':
    print("Fine Tuning From In-Paiting Task is {}".format(config.local_shuffle_pretrain))
    print("patch Random Number is {}".format(config.patch_random_number))
print("Optimizer is {}".format(config.opt))
print("Learning Rate is {:^.2f}".format(config.learning_rate))
print("Save Acc Threshold is {:^.2f}".format(config.save_acc_thres))

print("Path Information")
if config.shuffle_train and args.mode == 'Patch':
    print("Pretrain In-painting Task is {}".format(config.patch_Local_model_path))
else:
    print("Pretrain Model Path is {}".format(config.pretrain_image_pth_path))

t_train = config.switch[args.train_sensor]



def Global_Training_init():
    train_data = LivDetDataset(config.data_path, [t_train, 'train'])

    val_data = LivDetDataset(config.data_path, [t_train, 'test'])

    test_data = LivDetDataset(config.data_path, [t_train, 'test'])

    net = mobilenetv3_large()
    model_dict = net.state_dict()
    pretrained_dict = torch.load(config.pretrain_image_pth_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    net.cuda()

    train_batch_size = config.batch_size_for_train

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=config.shuffle_train, pin_memory=True,
                              num_workers=16)

    val_loader = DataLoader(val_data, batch_size=1, shuffle=config.shuffle_test, pin_memory=True, num_workers=4)

    test_loader = DataLoader(test_data, batch_size=1, shuffle=config.shuffle_test, pin_memory=True, num_workers=4)

    return net, train_loader, val_loader, test_loader


def Patch_Training_init():
    train_data = LivDetDataset(config.data_path, [t_train, 'train'])

    val_data = LivDetDataset(config.data_path, [t_train, 'test'])

    test_data = LivDetDataset(config.data_path, [t_train, 'test'])

    patch_train_data = Patch_LivDetDataset(train_data, random_number=3)

    patch_val_data = Patch_LivDetDataset(val_data, random_number=3)

    patch_test_data = Patch_LivDetDataset(test_data, random_number=3)

    patch_train_loader = DataLoader(patch_train_data, batch_size=config.batch_size_for_train,
                                    shuffle=config.shuffle_train, pin_memory=True, num_workers=8)
    patch_val_loader = DataLoader(patch_val_data, batch_size=1, shuffle=config.shuffle_test, pin_memory=True,
                                  num_workers=1)
    patch_test_loader = DataLoader(patch_test_data, batch_size=1, shuffle=config.shuffle_test, pin_memory=True,
                                   num_workers=1)

    net = mobilenetv3_large()
    if config.local_shuffle_pretrain:
        net_name = args.train_sensor + '_TOP_1Net.pth'
        net.load_state_dict(torch.load(config.patch_Local_model_path + net_name))
    else:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(config.pretrain_image_pth_path)
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    return net, patch_train_loader, patch_val_loader, patch_test_loader


def train_whole():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, train_loader, val_loader, test_loader = Global_Training_init()

    net.cuda()
    net_loss = nn.CrossEntropyLoss().cuda()

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    if config.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    acc_list = []

    for i in range(5):
        acc_list.append(config.save_acc_thres)

    t = tqdm(train_loader)

    loss_tr = 0
    loss_te = 0
    loss_val = 0
    acc_te = 0
    acc_val = 0
    tdr_val = 0
    tdr_te = 0
    for e in range(config.Epoch_num):
        t = tqdm(train_loader)
        t.set_description("Whole Epoch [{}/{}]".format(e + 1, config.Epoch_num))
        opt_counter = 0
        test_counter = 0
        test_dataiter = iter(test_loader)
        val_dataiter = iter(val_loader)
        for b, (imgs, ls) in enumerate(t):
            net.train()
            imgs = data_augment(imgs)
            if config.cut_out:
                imgs = cut_out(imgs, config.n_holes, config.length)
            imgs = imgs.type(torch.FloatTensor).to(device)
            l = ls.to(device).view(-1)
            out = net(imgs)
            crite = net_loss(out, l)
            loss = crite.cpu().detach().data.numpy()
            loss_tr = 0.6 * loss_tr + 0.4 * loss
            crite.backward()
            opt_counter += 1

            if opt_counter == config.opt_num:  # optimize
                optimizer.step()
                optimizer.zero_grad()
                opt_counter = 0
                test_counter += 1

            if test_counter == config.test_split_time:
                net.eval()
                with torch.no_grad():
                    for ind_ in range(2):
                        correct = torch.zeros(1).squeeze().cuda()
                        total = torch.zeros(1).squeeze().cuda()
                        loss = torch.zeros(1).squeeze().cuda()
                        result = []
                        for j in range(config.test_batch_num):
                            for b_i in range(config.batch_size_for_test):
                                if ind_ == 0:
                                    try:
                                        img, l = test_dataiter.next()
                                    except StopIteration:
                                        test_dataiter = iter(test_loader)
                                        img, l = test_dataiter.next()
                                else:  # ind=1
                                    try:
                                        img, l = val_dataiter.next()
                                    except StopIteration:
                                        val_dataiter = iter(val_loader)
                                        img, l = val_dataiter.next()
                                img = img.type(torch.FloatTensor).to(device)
                                l = l.to(device).view(-1)

                                out = net(img)
                                out_f = F.softmax(out, dim=1)
                                loss += net_loss(out, l)
                                pred = torch.argmax(out_f, 1)
                                correct += (pred == l).sum().float()
                                total += len(l)
                                result.append(
                                    [l.cpu().detach().data.numpy()[0], out_f.cpu().detach().data.numpy()[0, 1]])
                        if ind_ == 0:
                            acc_te = (correct / total).cpu().detach().data.numpy()
                            loss_te = (loss / total).cpu().detach().data.numpy()
                            ace, tdr_te = ACE_TDR_Cal(result)
                        else:
                            acc_val = (correct / total).cpu().detach().data.numpy()
                            loss_val = (loss / total).cpu().detach().data.numpy()
                            ace, tdr_val = ACE_TDR_Cal(result)

                test_counter = 0
                crit_save = acc_te
                if crit_save >= min(acc_list):
                    acc_list[acc_list.index(min(acc_list))] = crit_save
                    acc_list = sorted(acc_list, reverse=True)
                    net_path = os.path.join(args.savedir,
                                            'Global_train_' + args.train_sensor + '_test_' + args.train_sensor + '_TOP_' + str(
                                                acc_list.index(crit_save) + 1) + 'Net.pth')
                    torch.save(net.state_dict(), net_path)
                    print("Save model! Path:{} Criterion: {:4f}".format(net_path, crit_save))

            if test_counter == 0:
                t.set_postfix_str(
                    'Val_Acc:{:^2f}, Test_Acc:{:^2f}, TDR_val:{:^2f}, TDR_test:{:^2f}'.format(acc_val, acc_te, tdr_val,
                                                                                              tdr_te))
            else:
                t.set_postfix_str(
                    'TrLoss : {:^2f}, Val_Loss:{:^2f}, Test_Loss:{:^2f}'.format(loss_tr, loss_val, loss_te))
            t.update()


def train_patch():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net, train_loader, val_loader, test_loader = Patch_Training_init()

    net.cuda()
    net_loss = nn.CrossEntropyLoss().cuda()

    if config.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    if config.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    acc_list = []

    for i in range(5):
        acc_list.append(config.save_acc_thres)

    loss_tr = 0
    loss_te = 0
    loss_val = 0
    acc_te = 0
    acc_val = 0
    tdr_val = 0
    tdr_te = 0
    for e in range(config.Epoch_num):
        t = tqdm(train_loader)
        t.set_description("Patch Epoch [{}/{}]".format(e + 1, config.Epoch_num))
        opt_counter = 0
        test_counter = 0
        test_dataiter = iter(test_loader)
        val_dataiter = iter(val_loader)
        for b, (imgs, ls) in enumerate(t):
            imgs = imgs.view(-1, 3, 224, 224)
            ls = ls.view(-1)

            net.train()
            imgs = imgs.type(torch.FloatTensor).to(device)
            l = ls.to(device).view(-1)
            out = net(imgs)
            crite = net_loss(out, l)
            loss = crite.cpu().detach().data.numpy()
            loss_tr = 0.6 * loss_tr + 0.4 * loss
            crite.backward()
            opt_counter += 1

            if opt_counter == config.opt_num:  # optimize
                optimizer.step()
                optimizer.zero_grad()
                opt_counter = 0
                test_counter += 1

            if test_counter == config.test_split_time:
                net.eval()
                with torch.no_grad():
                    for ind_ in range(2):
                        correct = torch.zeros(1).squeeze().cuda()
                        total = torch.zeros(1).squeeze().cuda()
                        loss = torch.zeros(1).squeeze().cuda()
                        result = []
                        for j in range(config.test_batch_num):
                            for b_i in range(config.batch_size_for_test):
                                if ind_ == 0:
                                    try:
                                        img, l = test_dataiter.next()
                                    except StopIteration:
                                        test_dataiter = iter(test_loader)
                                        img, l = test_dataiter.next()
                                else:  # ind=1
                                    try:
                                        img, l = val_dataiter.next()
                                    except StopIteration:
                                        val_dataiter = iter(val_loader)
                                        img, l = val_dataiter.next()
                                img = img.view(-1, 3, 224, 224)
                                img = img.type(torch.FloatTensor).to(device)
                                l = l.to(device).view(-1)

                                out = net(img)

                                out_f = F.softmax(out, dim=1)
                                loss += net_loss(out, l)

                                pred = torch.argmax(out, 1)
                                correct += (pred == l).sum().float()
                                total += len(l)

                                result.append([np.mean(l.cpu().detach().data.numpy()[:]),
                                               np.mean(out_f.cpu().detach().data.numpy()[:, 1])])
                        if ind_ == 0:
                            acc_te = (correct / total).cpu().detach().data.numpy()
                            loss_te = (loss / total).cpu().detach().data.numpy()
                            ace, tdr_te = ACE_TDR_Cal(result)
                        else:
                            acc_val = (correct / total).cpu().detach().data.numpy()
                            loss_val = (loss / total).cpu().detach().data.numpy()
                            ace, tdr_val = ACE_TDR_Cal(result)

                test_counter = 0
                crit_save = acc_te
                if crit_save >= min(acc_list):
                    acc_list[acc_list.index(min(acc_list))] = crit_save
                    acc_list = sorted(acc_list, reverse=True)
                    net_path = os.path.join(args.savedir,
                                            'Patch_train_' + args.train_sensor + '_test_' + args.train_sensor + '_TOP_' + str(
                                                acc_list.index(crit_save) + 1) + 'Net.pth')
                    torch.save(net.state_dict(), net_path)
                    print("Save model! Path:{} Criterion: {:4f}".format(net_path, crit_save))

            if test_counter == 0:
                t.set_postfix_str(
                    'Val_Acc:{:^2f}, Test_Acc:{:^2f}, TDR_val:{:^2f}, TDR_test:{:^2f}'.format(acc_val, acc_te, tdr_val,
                                                                                              tdr_te))
            else:
                t.set_postfix_str(
                    'TrLoss : {:^2f}, Val_Loss:{:^2f}, Test_Loss:{:^2f}'.format(loss_tr, loss_val, loss_te))
            t.update()
            time.sleep(1.5)


if __name__ == '__main__':
    if args.mode == 'Patch':
        train_patch()
    if args.mode == 'Whole':
        train_whole()