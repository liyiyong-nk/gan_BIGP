#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 wuenze <wuenze@HKML-5>
#
# Distributed under terms of the MIT license.

"""

"""
import argparse, itertools, os, time
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from PIL import Image
import torch
import torch.nn as nn
import cv2
from models.models import Generator, Discriminator, SubMobileResnetGenerator
from utils.utils import *
from utils.perceptual import *
from utils.fid_score import calculate_fid_given_paths
from datasets.datasets import ImageDataset, PairedImageDataset


g_path = '/home/wuenze/code/gan-compression/pretrained/cycle_gan/horse2zebra/supernet/latest_net_G.pth'
g_path_new = './yiyong_pth/gc/supernet_netG.pth'
dim_lst = [32, 64, 128] + [128, 128, 128, 128, 128, 128, 128 ,128 ,128] + [128, 64, 32]
netG = SubMobileResnetGenerator(3, 3, dim_lst=dim_lst).cuda()


read_dict = torch.load(g_path)
pretrain_dict = {}
model_dict = netG.state_dict()
for key in read_dict:
    print (key)
    if 'conv_block.6' in key and key not in model_dict:
        new_key = key.replace('conv_block.6', 'conv_block.5')
    else:
        new_key = key
    pretrain_dict[new_key] = read_dict[key]
model_dict.update(pretrain_dict)
netG.load_state_dict(model_dict)
torch.save(netG.state_dict(), g_path_new)
idx = 0
for m in netG.modules():
    if isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
        idx += 1
        gamma = m.weight.data.cpu().numpy()
        beta = m.bias.data.cpu().numpy()
        print (gamma)
        print (beta)
        # channel_num = np.sum(gamma!=0)
        # if idx == 9 or idx == 5:
        #     channel_num -= 1
        # dim_lst.append(channel_num)
print(dim_lst)