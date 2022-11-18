import argparse, itertools, os
import numpy as np 
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from skimage import img_as_ubyte

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn

from models.models import Generator, Discriminator, Conv2dQuant, ConvTrans2dQuant, SubMobileResnetGenerator
from utils.utils import *
from scipy.spatial import distance

use_cuda = True
input_nc, output_nc = 3, 3
mask_index = [0, 4, 16, 20, 32,36, 48, 52, 156, 160]
mask_index = [1, 2, 5, 6, 9, 10,13, 14, ] # convtranspose2d not involved
model_length_1 = [32, 64, 128,128,128,128,128,128, 64, 32]
model_length = {}
for idx, i in enumerate(mask_index):
    model_length[i] = model_length_1[idx] 
# g_path = '/home/wuenze/code/gan-compression/pretrained/cycle_gan/horse2zebra/supernet/latest_net_G.pth'
g_path = './yiyong_pth/gc/supernet_netG.pth'
g_path_new = './yiyong_pth/gc/compresssed_center_netG.pth'
dim_lst = [32, 64, 128] + [128, 128, 128, 128, 128, 128, 128 ,128 ,128] + [128, 64, 32]
dense_model = SubMobileResnetGenerator(3, 3, dim_lst=dim_lst).cuda()

read_dict = torch.load(g_path)
pretrain_dict = {}
model_dict = dense_model.state_dict()
# for key in read_dict:
#     print (key)
#     if 'conv_block.6' in key and key not in model_dict:
#         new_key = key.replace('conv_block.6', 'conv_block.5')
#     else:
#         new_key = key
#     pretrain_dict[new_key] = read_dict[key]
# model_dict.update(pretrain_dict)
dense_model.load_state_dict(torch.load(g_path))
# dense_model = Generator(input_nc, output_nc, quant=quant)
# dense_model.load_state_dict(torch.load(g_path))
# dense_model = nn.DataParallel(dense_model)
# measure_model(dense_model, 256, 256) # 54168.000000M
idx_lst_oth = [
           1,4,6,10,14,5,
           9, 
           13,
           40, 41]
idx_lst = [1, 2, 6, 10, 14]
# idx_lst = [1, 2,  
#             6, 
#             10,
#             14,
#            40, 41]

# idx_lst = [1, 2, 
#             6,
#             10,
#             14,
#            ]
# idx_lst_oth = [5, 9, 13, 40, 41]
def get_filter_similar(weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
    if len(weight_torch.size()) == 4:
        filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
        similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
        weight_vec = weight_torch.view(weight_torch.size()[0], -1)

        if dist_type == "l2" or "cos":
            norm = torch.norm(weight_vec, 2, 1)
            norm_np = norm.cpu().numpy()
        elif dist_type == "l1":
            norm = torch.norm(weight_vec, 1, 1)
            norm_np = norm.cpu().numpy()
        filter_small_index = []
        filter_large_index = []
        filter_large_index = norm_np.argsort()[filter_pruned_num:]
        filter_small_index = norm_np.argsort()[:filter_pruned_num]

        # # distance using pytorch function
        # similar_matrix = torch.zeros((len(filter_large_index), len(filter_large_index)))
        # for x1, x2 in enumerate(filter_large_index):
        #     for y1, y2 in enumerate(filter_large_index):
        #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
        #         pdist = torch.nn.PairwiseDistance(p=2)
        #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
        # # more similar with other filter indicates large in the sum of row
        # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

        # distance using numpy function
        indices = torch.LongTensor(filter_large_index).cuda()
        weight_vec_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
        # for euclidean distance
        if dist_type == "l2" or "l1":
            similar_matrix = distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'euclidean')
        elif dist_type == "cos":  # for cos similarity
            similar_matrix = 1 - distance.cdist(weight_vec_after_norm, weight_vec_after_norm, 'cosine')
        similar_sum = np.sum(np.abs(similar_matrix), axis=0)

        # for distance similar: get the filter index with largest similarity == small distance
        similar_large_index = similar_sum.argsort()[similar_pruned_num:]
        similar_small_index = similar_sum.argsort()[:  similar_pruned_num]
        similar_index_for_filter = [filter_large_index[i] for i in similar_small_index]

        print('filter_large_index', filter_large_index)
        print('filter_small_index', filter_small_index)
        print('similar_sum', similar_sum)
        print('similar_large_index', similar_large_index)
        print('similar_small_index', similar_small_index)
        print('similar_index_for_filter', similar_index_for_filter)
        # kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
        # for x in range(0, len(similar_index_for_filter)):
        #     codebook[
        #     similar_index_for_filter[x] * kernel_length: (similar_index_for_filter[x] + 1) * kernel_length] = 0
        print("similar index done")
    else:
        pass
    return similar_index_for_filter
def convert2tensor(x):
    x = torch.FloatTensor(x)
    return x
similar_matrix = {}
cnt = 0
flag_change = False
for index, m in enumerate(dense_model.modules()):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        cnt += 1
        # print (cnt, m)
        item = m.weight
        # print (item.size())
        if cnt in mask_index:
            similar_matrix[cnt] = get_filter_similar(item.data, 1,
                                                                    0.5,
                                                                    model_length[cnt])

            flag_change = True
    if isinstance(m, nn.InstanceNorm2d) and flag_change:
        print (cnt)
        print (similar_matrix[cnt])
        m.weight.data[similar_matrix[cnt]] = 0
        # print (m.weight.data)
        flag_change = False
print("mask Ready")
print (similar_matrix)

torch.save(dense_model.state_dict(), g_path_new)