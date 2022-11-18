#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 wuenze <wuenze@HKML-5>
#
# Distributed under terms of the MIT license.

"""

"""

import argparse
import numpy as np
import torch
from models.models import Generator
from utils.utils import *
import os
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='7')
parser.add_argument('--image_dir', default='./datasets/horse2zebra/test/A/')
parser.add_argument('--result_dir', default='./results/horse2zebra/test/')
parser.add_argument('--pth', default='./results/horse2zebra/A2B/test/pth/netG_A2B_epoch_199.pth')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--quant', action='store_true', help='enable quantization (for both activation and weight)')
parser.add_argument('--subnet_model_path', default='')
# parser.add_argument('--subnet_model_path', default='./subnet_structures/horse2zebra/B2A/GS32/pth/epoch10_netG.npy')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def get_image_file(folder):
    imagelist =[]
    for parent, dirnames, filenames in os.walk(folder):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.jpeg','.png', '.webp')):
                imagelist.append(os.path.join(parent, filename))
    return imagelist

def adjust_dynamic_range (data, drange_in, drange_out):
    if drange_in != drange_out:
        scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
            np.float32(drange_in[1]) - np.float32(drange_in[0]))
        bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
        data = data * scale + bias
    return data


def to_range(images, min_value=0.0, max_value=1.0, dtype=None):
    assert \
        np.min(images) >= -1.0 - 1e-5 and np.max(images) <= 1.0 + 1e-5 \
        and (images.dtype == np.float32 or images.dtype == np.float64), \
        'The input images should be float64(32) and in the range of [-1.0, 1.0]!'
    if dtype is None:
        dtype = images.dtype
    return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)




def inference(image_path, model):
    # img = Image.open(image_path)
    # img = img.convert("RGB")
    img_np = cv2.imread(image_path)
    img_np = adjust_dynamic_range(img_np.astype(np.float32), [0, 255], [-1., 1.])
    img_np = img_np[np.newaxis, :, :, ::-1].transpose(0, 3, 1, 2)
    img = torch.from_numpy(img_np.copy()).cuda()
    res = model(img)

    res_numpy = res.cpu().float().detach().numpy()
    res_numpy = res_numpy.transpose(0, 2, 3, 1)
    res_numpy = np.clip(res_numpy, -1, 1)
    out = to_range(res_numpy, 0, 255, np.uint8)[0,:,:,::-1]
    return out

def run_img_dir():
    if args.subnet_model_path != '':
        # dim_lst_path = os.path.join(args.subnet_model_path, 'epoch%d_netG.npy' % 199)
        dim_lst = np.load(args.subnet_model_path)
        print (dim_lst)
        netG = Generator(args.input_nc, args.output_nc, dim_lst=np.load(args.subnet_model_path), quant=args.quant).cuda()
    else:
        netG = Generator(args.input_nc, args.output_nc, quant=args.quant).cuda()
    # g_path = os.path.join(foreign_dir, dense_model_folder, args.dataset, 'pth', 'netG_%s_epoch_%d.pth' % (args.task, 199) )
    # g_path = args.pth
    g_path = './results/horse2zebra/A2B/alpha1e-05_contral_rate0.001_tspv1/pth/epoch120_netG.pth'
    # g_path = './subnet_structures/horse2zebra/B2A/GS32/pth/epoch40_netG.pth'
    # g_path = '/data1/liyiyong/gan-slimming/pretrained_dense_model/horse2zebra/pth/epoch199_netG.pth'
    
    netG.load_state_dict(torch.load(g_path))
    print('load G from %s' % g_path)
    print(netG)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    img_paths = get_image_file(args.image_dir)
    for img_path in img_paths:
        img_new = inference(img_path, netG)
        img_name = img_path.split('/')[-1]
        img_new_path = os.path.join(args.result_dir, img_name)
        cv2.imwrite(img_new_path, img_new)

if __name__ == '__main__':
    run_img_dir()
