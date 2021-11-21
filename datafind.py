import os
import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--data_path", type=str, default=None)
args = parser.parse_args()
path_list = []

def walFile(file,text):
    for root, dirs, files in os.walk(file):
        for f in files:
            f = os.path.join(root,f)
            cond = True
            for t in text:
                if t not in f:
                    cond = False
            if cond:
                path_list.append(f)

def imgFind():
    walFile(args.data_path,['png'])
    walFile(args.data_path,['bmp'])
    g = tg = fg = td = fd = to = fo = d =o = 0
    for f in  path_list:
        if 'GreenBit' in f:
            g = g + 1
            if 'train' in f:
                tg = tg + 1
            if 'test' in  f:
                fg = fg + 1
        if 'DigitalPersona' in f:
            d = d + 1
            if 'train' in f:
                td = td + 1
            if 'test' in  f:
                fd = fd + 1
        if 'Orcathus' in f:
            o = o + 1
            if 'train' in f:
                to = to + 1
            if 'test' in  f:
                fo = fo + 1
    print('GreenBit: {}'.format(g))
    print('GreenBit train: {}'.format(tg))
    print('GreenBit test: {}'.format(fg))

    print('DigitalPersona: {}'.format(g))
    print('DigitalPersona train: {}'.format(tg))
    print('DigitalPersona test: {}'.format(fg))

    print('Orcathus: {}'.format(g))
    print('Orcathus train: {}'.format(tg))
    print('Orcathus test: {}'.format(fg))

if __name__ == '__main__':
    imgFind()
    with open('data_path.txt','w+') as fileop:
        for f in path_list:
            fileop.write(f)
            fileop.write('\n')
