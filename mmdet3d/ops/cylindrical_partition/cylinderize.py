# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import numpy as np

from .voxel_layer import dynamic_voxelize, hard_voxelize


# transformation between Cartesian coordinates and polar coordinates
def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class _Voxelization(Function):

    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)
            dynamic_voxelize(points, coors, voxel_size, coors_range, 3)
            return coors
        else:
            voxels = points.new_zeros(
                size=(max_voxels, max_points, points.size(1)))  # [16000, 32, 4]
            coors = points.new_zeros(size=(max_voxels, 3), dtype=torch.int)
            num_points_per_voxel = points.new_zeros(
                size=(max_voxels, ), dtype=torch.int)
            voxel_num = hard_voxelize(points, voxels, coors,
                                      num_points_per_voxel, voxel_size,
                                      coors_range, max_points, max_voxels, 3)




            # select the valid voxels
            voxels_out = voxels[:voxel_num]
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]  # voxel中点的数量 不存在大小排序
            return voxels_out, coors_out, num_points_per_voxel_out




voxelization = _Voxelization.apply


class cylinderization(nn.Module):

    def __init__(self,
                 cylinder_size,
                 point_cloud_range,
                 max_num_points,
                 max_cylinders=20000):
        super(cylinderization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.voxel_size = cylinder_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_cylinders, tuple):
            self.max_voxels = max_cylinders
        else:
            self.max_voxels = _pair(max_cylinders)

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(cylinder_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()  # round 四舍五入
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w] 这一步转换对后面的 mmdet3d/models/voxel_encoders/pillar_encoder.py L125-130计算有影响
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_cylidners = self.max_voxels[0]
        else:
            max_cylinders = self.max_voxels[1]

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_cylinders)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'cylinder_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_cylinders=' + str(self.max_voxels)
        tmpstr += ')'
        return tmpstr
