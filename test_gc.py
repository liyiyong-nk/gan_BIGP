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
from models.models import Generator, Discriminator, Conv2dQuant, ConvTrans2dQuant, SubMobileResnetGenerator
from utils.utils import *
import os
import cv2
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='7')
parser.add_argument('--image_dir', default='./datasets/horse2zebra/test/A/testA/')
parser.add_argument('--result_dir', default='./results/horse2zebra/test/')
parser.add_argument('--pth', default='./results/horse2zebra/A2B/test/pth/netG_A2B_epoch_199.pth')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--quant', action='store_true', help='enable quantization (for both activation and weight)')
parser.add_argument('--subnet_model_path', default='')

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
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = img_np[np.newaxis, :, :, :]
    img_np = img_np.transpose(0, 3, 1, 2)
    img = torch.from_numpy(img_np.copy()).cuda()
    with torch.no_grad():
        res = model(img)

    res_numpy = res.cpu().float().detach().numpy()
    res_numpy = res_numpy.transpose(0, 2, 3, 1)
    res_numpy = np.clip(res_numpy, -1, 1)
    out = to_range(res_numpy, 0, 255, np.uint8)[0,:,:,:]
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    return out

def cal_sparse_loss(model):
    idx = 0
    loss = 0.0
    now_pp = 0
    now_con = 0
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            # first = m.parameters()
            p = m.weight
            idx += 1
            if p is None:
                print ('None')
                now_pp = 0.0
            else:
                now_pp = p.abs()
                
                
        
        if isinstance(m, nn.Conv2d):
            # second = m.parameters()
            # for p in second:
            p = m.weight
            now_con = p * p
            now_con = now_con[:, :, 0,0].sum(axis=0)
            loss = loss + (now_pp * now_con).sum()
            print (loss)
            now_pp = 0.0
    return loss


def run_img_dir():
    if args.subnet_model_path == 'gc':
        # dim_lst_path = os.path.join(args.subnet_model_path, 'epoch%d_netG.npy' % 199)
        print ('sub mobile resnet ,Gan compression')
        dim_lst = [32, 64, 128] + [128, 128, 128, 128, 128, 128, 128 ,128 ,128] + [128, 64, 32]
        netG = SubMobileResnetGenerator(3, 3, dim_lst=dim_lst).cuda()
    else:
        netG = Generator(args.input_nc, args.output_nc, quant=args.quant).cuda()
    # g_path = os.path.join(foreign_dir, dense_model_folder, args.dataset, 'pth', 'netG_%s_epoch_%d.pth' % (args.task, 199) )
    g_path = args.pth
    netG.load_state_dict(torch.load(g_path))
    print('load G from %s' % g_path)

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    else:
        oldimgs = glob.glob(os.path.join(args.result_dir, '*'))
        for oldimg in oldimgs:
            os.remove(oldimg)

    img_paths = get_image_file(args.image_dir)
    for img_path in img_paths:
        img_new = inference(img_path, netG)
        img_name = img_path.split('/')[-1]
        img_new_path = os.path.join(args.result_dir, img_name)
        cv2.imwrite(img_new_path, img_new)


if __name__ == '__main__':
    run_img_dir()
    # loss = run_sparse_loss()
    # print (loss)
