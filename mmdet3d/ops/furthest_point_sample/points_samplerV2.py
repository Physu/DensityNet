import numpy as np
import torch
from mmcv.runner import force_fp32
from torch import nn as nn
from typing import List

from .furthest_point_sample import (furthest_point_sample,
                                    furthest_point_sample_with_dist)
from .utils import calc_square_dist
from mmcv.cnn import ConvModule
import torch.nn.functional as F


# from tensorboardX import SummaryWriter
# from mmdet3d.models.roi_heads.mask_heads.pointsample_head import PointSampleHead
#
# path_eps = 'newback/0426_trainval_densitynet_point_wise_mask/'
# writer = SummaryWriter(path_eps)


def get_sampler_type(sampler_type):
    """Get the type and mode of points sampler.

    Args:
        sampler_type (str): The type of points sampler.
            The valid value are "D-FPS", "F-FPS", or "FS".

    Returns:
        class: Points sampler type.
    """
    if sampler_type == 'D-FPS':
        sampler = DFPS_Sampler
    elif sampler_type == 'F-FPS':
        sampler = FFPS_Sampler
    elif sampler_type == 'FS':
        sampler = FS_Sampler
    elif sampler_type == 'MS3':
        sampler = Mask_Sampler3
    elif sampler_type == 'MS4':
        sampler = Mask_Sampler4
    else:
        raise ValueError('Only "sampler_type" of "D-FPS", "F-FPS", "FS", or "MS"'
                         f' are supported, got {sampler_type}')

    return sampler


class Points_SamplerV2(nn.Module):
    """Points sampling.

    Args:
        num_point (list[int]): Number of sample points.
        fps_mod_list (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
            M-FPS: using CNNs to make a mask for points sampling
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
    """

    def __init__(self,
                 num_point: List[int],
                 fps_mod_list: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1]):
        super(Points_SamplerV2, self).__init__()
        # FPS would be applied to different fps_mod in the list,
        # so the length of the num_point should be equal to
        # fps_mod_list and fps_sample_range_list.
        assert len(num_point) == len(fps_mod_list) == len(
            fps_sample_range_list)
        self.num_point = num_point
        self.fps_sample_range_list = fps_sample_range_list
        self.samplers = nn.ModuleList()
        # self.sampler_fs = nn.ModuleList()
        # self.sampler_fs.append(FS_Sampler())
        self.iter = 0
        self.loss_sum = 0

        for fps_mod in fps_mod_list:
            self.samplers.append(get_sampler_type(fps_mod)())
            self.flag = fps_mod
            # if self.flag == 'MS1' or self.flag == 'MS2' or self.flag == 'MS3' or self.flag == 'MS4':
            #     self.optimizer = torch.optim.SGD(self.samplers.parameters(), lr=0.01)  # 设置优化器参数,lr=0.002指的是学习率的大小
            #     self.scheduler_step = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, 0.1)
            #     self.loss_func = torch.nn.CrossEntropyLoss()

        self.fp16_enabled = False
        self.num_true_mask = 0
        self.num_true_fps = 0
        self.num_1000 = 0

    @force_fp32()
    def forward(self, points_xyz, features, point_wise_mask=None):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) Descriptors of the features.
            point_wise_mask:(B,N)

        Return：
            Tensor: (B, npoint, sample_num) Indices of sampled points.
            point_wise_mask:
        """
        indices = []
        preds = []
        last_fps_end_index = 0
        B, N, _ = points_xyz.size()

        for fps_sample_range, sampler, npoint in zip(
                self.fps_sample_range_list, self.samplers, self.num_point):
            assert fps_sample_range < points_xyz.shape[1]
            # fps
            if fps_sample_range == -1:
                sample_points_xyz = points_xyz[:, last_fps_end_index:]
                sample_features = features[:, :, last_fps_end_index:]
            else:
                sample_points_xyz = points_xyz[:, last_fps_end_index:fps_sample_range]
                sample_features = features[:, :, last_fps_end_index:fps_sample_range]

            fps_idx = sampler(sample_points_xyz.contiguous(), sample_features,
                              npoint)

            if self.flag == 'MS1' or self.flag == 'MS2' or self.flag == 'MS3' or self.flag == 'MS4':
                # self.iter = self.iter + B
                preds = fps_idx[1]  # 注意这个取值顺序，先preds，再fps_idx
                fps_idx = fps_idx[0]
            else:
                preds = None

                # 统计重复的数值
                # num_of = self.gather_point_wise_mask(point_wise_mask, fps_idx)
                # i = 0
                # for point_true, point_true_after in zip(point_wise_mask, num_of):
                #     print(f"{i} index have point_true:{len(torch.nonzero(point_true))}"
                #           f"point_true_after:{len(torch.nonzero(point_true_after))}")
                #     i = i+1

                '''
                sampler_fs = FS_Sampler()
                fps_idx_fs = sampler_fs(sample_points_xyz.contiguous(), sample_features, 512)
                
                if point_wise_mask is not None:
                    for sub_idx, sub_fs_idx, mask in zip(fps_idx, fps_idx_fs, point_wise_mask):
                        # dex = torch.nonzero(sub_idx)
                        mask_num_true = len(torch.nonzero(mask))

                        val = torch.index_select(mask, dim=0, index=sub_idx.long())
                        val_fs = torch.index_select(mask, dim=0, index=sub_fs_idx.long())
                        num = torch.nonzero(val)
                        num_fs = torch.nonzero(val_fs)
                        print(f"MS4 total 4096, mask_num_true:{mask_num_true} MS4 num right:{len(num)}"
                              f" FS num right:{len(num_fs)}")
                '''
            # if self.flag == 'FS':
                # for sub_idx, mask in zip(fps_idx, point_wise_mask):
                #     # dex = torch.nonzero(sub_idx)
                #     mask_num_true = len(torch.nonzero(mask))
                #
                #     val = torch.index_select(mask, dim=0, index=sub_idx.long())
                #     num = torch.nonzero(val)
                #     print(f"FS total 4096, mask_num_true:{mask_num_true} num right:{len(num)}")
                #     self.num_true_mask = self.num_true_mask + mask_num_true
                #     self.num_true_fps = self.num_true_fps + len(num)
                #     self.num_1000 = self.num_1000 + 1
                #     if self.num_1000 > 1000:
                #         print(f"*************************************************")
                #         print(f"self.point_mask:{self.num_true_mask}")
                #         print(f"self.point_fps:{self.num_true_fps}")
                #         print(f"self.num_1000:{self.num_1000}")


                # loss = 0
                # pointwise
                # for pred, mask in zip(preds, point_wise_mask):
                #     # pred = torch.squeeze(pred, dim=0)
                #     mask = mask.long()
                #     loss = loss + self.loss_func(pred, mask.detach())

                # self.optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                # self.optimizer.step()

                # self.loss_sum = self.loss_sum + loss.detach()
                # fps_idx = torch.squeeze(idx[:, [1], :], dim=1).to(torch.int32)

            indices.append(fps_idx + last_fps_end_index)
            last_fps_end_index += fps_sample_range
        indices = torch.cat(indices, dim=1)

        return indices, preds

    def gather_point_wise_mask(self, point_wise_mask, cur_indices):
        '''
        # update the point_wise_mask，19384 ->4096 ->1024
        :param point_wise_mask: B,C,N1
        :param cur_indices: B,N2
        :return:
        '''

        indices = []
        B, N = cur_indices.size()
        for mask, cur_indice in zip(point_wise_mask, cur_indices):

            mask_ = cur_indice.new_tensor(False).repeat(N).bool()
            for index, indice in enumerate(cur_indice):
                mask_[index] = mask[cur_indice[index]]
            indices.append(mask_)

        return indices



class DFPS_Sampler(nn.Module):
    """DFPS_Sampling.

    Using Euclidean distances of points for FPS.
    """

    def __init__(self):
        super(DFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with D-FPS."""
        fps_idx = furthest_point_sample(points.contiguous(), npoint)
        return fps_idx


class FFPS_Sampler(nn.Module):
    """FFPS_Sampler.

    Using feature distances for FPS.
    """

    def __init__(self):
        super(FFPS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with F-FPS."""
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(
            features_for_fps, features_for_fps, norm=False)
        fps_idx = furthest_point_sample_with_dist(features_dist, npoint)
        return fps_idx


class FS_Sampler(nn.Module):
    """FS_Sampling.

    Using F-FPS and D-FPS simultaneously.
    """

    def __init__(self):
        super(FS_Sampler, self).__init__()

    def forward(self, points, features, npoint):
        """Sampling points with FS_Sampling."""
        features_for_fps = torch.cat([points, features.transpose(1, 2)], dim=2)
        features_dist = calc_square_dist(
            features_for_fps, features_for_fps, norm=False)
        fps_idx_ffps = furthest_point_sample_with_dist(features_dist, npoint)
        fps_idx_dfps = furthest_point_sample(points, npoint)
        fps_idx = torch.cat([fps_idx_ffps, fps_idx_dfps], dim=1)
        return fps_idx


class Mask_Sampler3(nn.Module):
    """Mask_Sampling.

    Using CNNs features of points for Point Sampling.
    """

    def __init__(self):
        super(Mask_Sampler3, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU(inplace=True)
        #########################################
        self.encode1 = Encoder(128, 32)
        # self.encode2 = Encoder(128, 256)
        # self.encode3 = Encoder(256, 256)  # 8*8

        # self.decode3 = Decoder(256, 256 + 128, 128)  # 16*16
        # self.decode2 = Decoder(128, 128 + 64, 64)  # 32*32
        # self.decode1 = Decoder(64, 32 + 128, 32)  # 64
        self.decode0 = Decoder(32, 256, 128)

        self.conv_last = nn.Conv2d(128, 2, 1)  # 中间 2 控制最后的输出的尺寸是(b,1,4096)，或者(b,2,4096)
        self.counter = 0

    def forward(self, points, features, npoint):
        '''
        input # B,128,1024 ->B,128,32,32
        e1 # 256,16,16
        e2 # 512,8,8
        f # 1024,4,4
        d3 # 512,8,8
        d2 # 256,16,16
        d1 # 128,32,32
        out # 2,32,32

        '''
        B, N, _ = points.size()

        xyz_spatial_feature_0 = self.conv1(points.transpose(1, 2))  # (B,C,N) 即(2,4096,3)->(2,3,4096)
        xyz_spatial_feature_0 = self.bn1(xyz_spatial_feature_0)
        xyz_spatial_feature_0 = self.act1(xyz_spatial_feature_0)

        feature_C0 = torch.cat((xyz_spatial_feature_0, features), dim=1)  # B,128, 4096

        feature_conv2 = feature_C0.view(B, 128, 64, 64)  # (B,128,64,64) # 残差网络初始输入
        e1 = self.encode1(feature_conv2)  # (B,128,64,64)-> ([B, 32, 32, 32])
        # e2 = self.encode2(e1)  # torch.Size([B, 256, 16, 16])
        # e3 = self.encode3(e2)  # # torch.Size([B, 256, 8, 8])
        #
        # d3 = self.decode3(e3, e2)  # (B,128,16,16)
        # d2 = self.decode2(d3, e1)  # (B, 64,32,32)
        # d1 = self.decode1(d2, feature_conv2)  # (B,32,64,64)
        d1 = self.decode0(e1, feature_conv2)  # (B,32,64,64)

        preds_1 = self.conv_last(d1)  # (B,2,64,64)
        preds_1 = torch.flatten(preds_1, start_dim=2, end_dim=3)  # (B,2,4096)
        # preds = preds_1.squeeze(dim=1)  # (B,4096)
        preds_1 = preds_1.transpose(1, 2)
        preds_2 = F.avg_pool2d(preds_1, kernel_size=[1, 2])
        # preds = torch.sigmoid(preds_1)  # (b,2,4096) # 这个 2 表示对应的类别

        # preds_squeeze = preds[:, :, [1]]
        _, indices = torch.topk(preds_2.squeeze(2), npoint, dim=1)  # (b,1,4096)->选出1024 point
        indices = indices.int()  # 将torch.int64->torch.int32
        # self.counter = self.counter + 1
        # if self.counter % 10 == 0:
        #     print(f"indices:{indices[:,:500]}")
        return [indices, preds_2]


class Mask_Sampler4(nn.Module):
    """Mask_Sampling.

    Using CNNs features of points for Point Sampling.
    """

    def __init__(self):
        super(Mask_Sampler4, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, 1, stride=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU(inplace=True)
        #########################################
        self.encode1 = Encoder(128, 128)
        # self.encode2 = Encoder(128, 256)
        # self.encode3 = Encoder(256, 256)  # 8*8
        #
        # self.decode3 = Decoder(256, 256 + 128, 128)  # 16*16
        # self.decode2 = Decoder(128, 128 + 64, 64)  # 32*32
        self.decode1 = Decoder(64, 32 + 128, 32)  # 64
        # self.decode0 = Decoder(64, 64 + 32, 32)
        self.conv_last = nn.Conv2d(32, 2, 1)  # 中间 2 控制最后的输出的尺寸是(b,1,4096)，或者(b,2,4096)
        self.counter = 0

    def forward(self, points, features, npoint):
        """Sampling points with M-FPS."""

        '''
        input # B,128,1024 ->B,128,32,32
        e1 # 256,16,16
        e2 # 512,8,8
        f # 1024,4,4
        d3 # 512,8,8
        d2 # 256,16,16
        d1 # 128,32,32
        out # 2,32,32

        '''
        B, N, _ = points.size()

        xyz_spatial_feature_0 = self.conv1(points.transpose(1, 2))  # (B,C,N) 即(2,4096,3)->(2,3,4096)
        xyz_spatial_feature_0 = self.bn1(xyz_spatial_feature_0)
        xyz_spatial_feature_0 = self.act1(xyz_spatial_feature_0)

        feature_C0 = torch.cat((xyz_spatial_feature_0, features), dim=1)  # B,128, 4096

        feature_conv2 = feature_C0.view(B, 128, 64, 64)  # (B,128,64,64) # 残差网络初始输入
        e1 = self.encode1(feature_conv2)  # (B,128,64,64)-> ([B, 128, 32, 32])
        e2 = self.encode2(e1)  # torch.Size([B, 256, 16, 16])
        e3 = self.encode3(e2)  # # torch.Size([B, 256, 8, 8])

        d3 = self.decode3(e3, e2)  # (B,128,16,16)
        d2 = self.decode2(d3, e1)  # (B, 64,32,32)
        d1 = self.decode1(d2, feature_conv2)  # (B,32,64,64)

        preds_1 = self.conv_last(d1)  # (B,2,64,64)
        preds_1 = torch.flatten(preds_1, start_dim=2, end_dim=3)  # (B,2,4096)
        # preds = preds_1.squeeze(dim=1)  # (B,4096)
        preds_1 = preds_1.transpose(1, 2)
        preds_2 = F.avg_pool2d(preds_1, kernel_size=[1, 2])
        # preds = torch.sigmoid(preds_1)  # (b,2,4096) # 这个 2 表示对应的类别

        # preds_squeeze = preds[:, :, [1]]
        _, indices = torch.topk(preds_2.squeeze(2), npoint, dim=1)  # (b,1,4096)->选出1024 point
        indices = indices.int()  # 将torch.int64->torch.int32
        # self.counter = self.counter + 1
        # if self.counter % 10 == 0:
        #     print(f"indices:{indices[:,:500]}")
        return [indices, preds_2]


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # kernel size, stride
        )

    def forward(self, x1):
        x1 = self.conv_relu(x1)
        return x1


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1



