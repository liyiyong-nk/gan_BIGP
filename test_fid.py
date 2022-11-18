import os
import torch
import argparse
import numpy as np
import cv2
from utils.utils import *
from utils.perceptual import *
from utils.fid_score import calculate_fid_given_paths
parser = argparse.ArgumentParser()
parser.add_argument('--image_gt_dir', default='/home/wuenze/data/gan/horse2zebra/B/')
parser.add_argument('--image_fake_dir', default='/home/wuenze/data/horse2zebra/test/B/testB/')
parser.add_argument('--gpu', default='7')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
img_paths_for_FID = [args.image_fake_dir, args.image_gt_dir]
FID = calculate_fid_given_paths(img_paths_for_FID)
print (FID)
