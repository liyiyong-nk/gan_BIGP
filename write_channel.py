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
import glob

netG = Generator(3, 3)

if __name__ == "__main__":
    # FID = calculate_fid_given_paths(img_paths_for_FID)
    # print (FID)
    # path_pths = '/DATA/disk1/wuenze/code/GAN-Slimming/results/horse2zebra/A2B/'
    path_pths = '/DATA/disk1/wuenze/code/GAN-Slimming/yiyong_pth/exper/GS32_rho0.011_beta20_vgg_e200-b8_sgd_mom0.5_lrgamma0.1_adam_lrw1e-05_wd0.001/pth/'
    # path_pths1 = '/DATA/disk1/wuenze/code/GAN-Slimming/yiyong_pth/exper/model_ablation_study/*/pth'
    # path_pths1 = '/DATA/disk1/wuenze/code/GAN-Slimming/yiyong_pth/exper/model_ablation_study/sorted/'
    pths = glob.glob(os.path.join(path_pths, '*.pth'))
    # pths = pths + glob.glob(os.path.join(path_pths1, '*.pth'))
    for pth in pths:
        netG.load_state_dict(torch.load(pth))
        channel_num = none_zero_channel_num(netG)
        print (channel_num)
        di, name = pth.rsplit('/', 1)
        name = name.split('.')[0]
        txt = os.path.join(di, name+'.txt')
        print(txt)
        with open(txt, 'w') as fout:
            fout.write(str(channel_num))


    # need_paths = os.listdir(need_dir)
    # for pa in need_paths:
    #     pa = os.path.join(need_dir, pa)
    #     save_path_txt = os.path.join(pa, 'fid.txt')
    #     print (pa)
    #     print (save_path_txt)
    #     img_paths_for_FID = [pa, args.image_gt_dir]
    #     FID = calculate_fid_given_paths(img_paths_for_FID)
    #     print (FID)
    #     with open(save_path_txt, 'w') as fout:
    #         fout.write(str(FID))
