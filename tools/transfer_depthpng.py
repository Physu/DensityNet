# _*_ coding: utf-8 _*_
import scipy.io as sio
import os
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片
from shutil import copytree, ignore_patterns, copy

map_data = sio.loadmat(r'./data/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat')
sunrgbd_trainval_depth = r'./data/sunrgbd/sunrgbd_trainval/depth_png'
sunrgbd_trainval_depth_bfx = r'./data/sunrgbd/sunrgbd_trainval/depth_bfx_png'
if not os.path.exists(sunrgbd_trainval_depth):
    os.mkdir(sunrgbd_trainval_depth)
if not os.path.exists(sunrgbd_trainval_depth_bfx):
    os.mkdir(sunrgbd_trainval_depth_bfx)

for idx in range(10335):
    base_path = map_data['SUNRGBDMeta'][0][idx][0][0]
    png_name = map_data['SUNRGBDMeta'][0][idx][6][0]
    png_path = os.path.join('./data/sunrgbd/OFFICIAL_SUNRGBD', base_path, 'depth', png_name)
    png_bfx_path = os.path.join('./data/sunrgbd/OFFICIAL_SUNRGBD', base_path, 'depth_bfx', png_name)
    dst_depth = os.path.join(sunrgbd_trainval_depth, f'{idx+1:06d}.png')
    dst_depth_bfx = os.path.join(sunrgbd_trainval_depth_bfx, f'{idx+1:06d}.png')
    copy(png_path, dst_depth)
    copy(png_bfx_path, dst_depth_bfx)
    print(f'{idx+1:06d}.png')