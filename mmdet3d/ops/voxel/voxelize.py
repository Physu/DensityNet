# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
import numpy as np

from .voxel_layer import dynamic_voxelize, hard_voxelize


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
            voxels_out = voxels[:voxel_num]  # 获取有效voxels
            coors_out = coors[:voxel_num]
            num_points_per_voxel_out = num_points_per_voxel[:voxel_num]  # voxel中点的数量 不存在大小排序
            return voxels_out, coors_out, num_points_per_voxel_out


voxelization = _Voxelization.apply


class Voxelization(nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000):
        super(Voxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size  # 得到 grid_size ([432., 496., 1.])
        grid_size = torch.round(grid_size).long()  # round 四舍五入  long int ([432, 496, 1])
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w] 这一步转换对后面的 mmdet3d/models/voxel_encoders/pillar_encoder.py L125-130计算有影响
        self.pcd_shape = [*input_feat_shape, 1][::-1]
        self.fixed_volume_space = False
        self.max_volume_space = [50, np.pi, 2]
        self.min_volume_space = [0, -np.pi, -4]
        self.grid_size=[480, 360, 32]

    def cylinderize(self, points):
        xyz_pol = self.cart2polar(points)
        xyz_pol = xyz_pol.cpu().numpy()

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = np.array(self.grid_size)
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        # voxel_position = self.polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        # label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        # label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        # processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        # data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, points[:, :2]), axis=1)

        # if len(data) == 2:
        #     return_fea = return_xyz
        # elif len(data) == 3:
        #     return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
        #
        # if self.return_test:
        #     data_tuple += (grid_ind, labels, return_fea, index)
        # else:
        #     data_tuple += (grid_ind, labels, return_fea)
        # return data_tuple
        return return_xyz

    # transformation between Cartesian coordinates and polar coordinates
    def cart2polar(self, input_xyz):
        rho = torch.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
        phi = torch.atan2(input_xyz[:, 1], input_xyz[:, 0])
        return torch.stack((rho, phi, input_xyz[:, 2]), axis=1)

    def polar2cat(self, input_xyz_polar):
        # print(input_xyz_polar.shape)
        x = input_xyz_polar[0] * torch.cos(input_xyz_polar[1])
        y = input_xyz_polar[0] * torch.sin(input_xyz_polar[1])
        return torch.stack((x, y, input_xyz_polar[2]), axis=0)

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        if self.training:
            max_voxels = self.max_voxels[0]
        else:
            max_voxels = self.max_voxels[1]
        cylinder = self.cylinderize(input)

        return voxelization(input, self.voxel_size, self.point_cloud_range,
                            self.max_num_points, max_voxels)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ')'
        return tmpstr
