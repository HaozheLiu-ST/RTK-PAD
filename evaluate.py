# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from utils.dataset import LivDetDataset
from models.network import mobilenetv3_large
from utils.config import DefaultConfig
from tqdm import tqdm
from utils.grad_cam import GradCam
from utils.utils import patch_extraction, resize, patch_extraction_grad_cam, resizetensor

config = DefaultConfig()

def ACE_TDR_Cal(arr_result, rate=0.01):
    ace_list = []
    tdr_list = [0, ]
    arr_result = np.array(arr_result)
    total = len(arr_result)
    with tqdm(total=total, ncols=100, desc='Find Threshold') as t:
        for thres in arr_result[:, 1]:
            TP = TN = FP = FN = 0
            for l, sc in arr_result:
                if sc > thres and l == 1:
                    TP = TP + 1.
                elif sc <= thres and l == 0:
                    TN = TN + 1.
                elif sc < thres and l == 1:
                    FN = FN + 1.
                else:
                    FP = FP + 1.
            Ferrlive = FP / (FP + TN + 1e-7)
            Ferrfake = FN / (FN + TP + 1e-7)
            FDR = FP / (FP + TN + 1e-7)
            TDR = TP / (TP + FN + 1e-7)
            if FDR < rate:
                tdr_list.append(TDR)
            ace_list.append((Ferrlive + Ferrfake) / 2.)
            t.set_postfix_str('ACE : {:^7.3f} TDR@FDR=1% : {:^7.3f}'.format(min(ace_list), max(tdr_list)))
            t.update()
    return min(ace_list), max(tdr_list)


def test_fusion(net_list, search, name, sensor_te, patch_num=2, strategy='baseline'):
    test_data = LivDetDataset(config.data_path, search)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_dataiter = iter(test_loader)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_whole, net_patch = net_list
    net_whole.cuda()
    net_whole.eval()
    net_patch.cuda()
    net_patch.eval()
    correct = torch.zeros(1).squeeze().cuda()
    result = []
    if strategy == 'GradCam':
        grad_cam_2 = GradCam(model=net_whole, use_cuda=True, feature_id=[2])
        grad_cam_4 = GradCam(model=net_whole, use_cuda=True, feature_id=[4])
        grad_cam_7 = GradCam(model=net_whole, use_cuda=True, feature_id=[7])
    print('Init Successful')
    with tqdm(total=len(test_data.imgs), ncols=110, desc=name + ' Test:' + sensor_te) as t:
        inum = 0
        for img, l in test_dataiter:
            res = []
            l_ = l.to(device).view(-1)
            img_ = img.type(torch.FloatTensor).to(device)

            out = net_whole(img_)
            out = F.softmax(out, dim=1)
            pred = torch.argmax(out, 1)
            correct += (pred == l_).sum().float()

            live_s, spoof_s = out.cpu().detach().data.numpy()[0]
            res.append(spoof_s)

            if strategy == 'baseline':
                imgs, l = patch_extraction(img, l, random_cond=True, random_number=patch_num)
            elif strategy == 'GradCam':
                imgs = []
                l_list = []
                for target_index in range(2):
                    mask_1 = grad_cam_2(img_, target_index)
                    mask_1 = resize(mask_1, (img.shape[2], img.shape[3]))

                    mask_2 = grad_cam_4(img_, target_index)
                    mask_2 = resize(mask_2, (img.shape[2], img.shape[3]))

                    mask_3 = grad_cam_7(img_, target_index)
                    mask_3 = resize(mask_3, (img.shape[2], img.shape[3]))

                    mask = (mask_1 + mask_2 + mask_3) / 3.

                    patch = patch_extraction_grad_cam(img, [mask])[0][:int(patch_num / 2)]
                    for p in patch:
                        p = resizetensor(p)
                        imgs.append(p)
                        l_list.append(l[0])
                imgs = np.array(imgs)
                imgs = torch.from_numpy(imgs)
                l = np.array(l_list)
                l = torch.from_numpy(l)
            imgs = imgs.type(torch.FloatTensor).to(device)
            l = l.to(device).view(-1)
            out = net_patch(imgs)
            out = F.softmax(out, dim=1)
            pred = torch.argmax(out, 1)
            correct += (pred == l).sum().float()
            res.append(torch.mean(out, dim=0).cpu().detach().data.numpy()[1])
            inum += (patch_num + 1)
            acc = (correct / inum).cpu().detach().data.numpy()
            result.append([l_.cpu().detach().data.numpy()[0], res[0] + res[1]])
            t.update()

    with open(name + 'Test_' + sensor_te + '.txt', 'w+') as fileop:
        for r in result:
            r = str(r).replace('[', '').replace(']', '')
            fileop.write(r)
            fileop.write('\n')

    ace, tdr = ACE_TDR_Cal(result, rate=0.01)
    print('ACE : {:^4f}    TDR@FDR=1% : {:^4f}'.format(ace, tdr))
    return ace, tdr



if __name__ == '__main__':
    Switch = {
        'O': 'Orcathus',
        'G': 'GreenBit',
        'D': 'DigitalPersona'
    }

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--test_sensor", type=str, default='O')
    parser.add_argument("--global_model_path", type=str, default=None)
    parser.add_argument("--patch_model_path", type=str, default=None)
    parser.add_argument("--patch_num", type=int, default=2)  # patch_num
    args = parser.parse_args()

    net_patch = mobilenetv3_large()
    net_global = mobilenetv3_large()
    net_patch.load_state_dict(torch.load(args.patch_model_path))
    net_global.load_state_dict(torch.load(args.global_model_path))

    test_fusion(
        [net_global, net_patch],
        search=[Switch[args.test_sensor], 'test'],
        name='Fusion_RTK_Log',
        sensor_te=Switch[args.test_sensor],
        strategy='GradCam',
        patch_num=args.patch_num,
    )