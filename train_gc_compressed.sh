#! /bin/sh
#
# train_gc.sh
# Copyright (C) 2021 wuenze <wuenze@HKML-5>
#
# Distributed under terms of the MIT license.
#


python gs_gc.py --gpu 1 --alpha 1e-4 --epoch 1200 --lrgamma 0.001 --name gc_compressed \
  --dataset horse2zebra_gc \
  --pretrain_g_path /home/wuenze/code/GAN-Slimming/pretrained_dense_model/horse2zebra_gc/compressed//latest_net_G.pth \
  --pretrain_d_path /home/wuenze/code/GAN-Slimming/pretrained_dense_model/horse2zebra_gc/compressed//latest_net_D.pth 
  
