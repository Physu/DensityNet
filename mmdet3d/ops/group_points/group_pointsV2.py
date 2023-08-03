import torch
from torch import nn as nn
from torch.autograd import Function
from typing import Tuple

from ..ball_query import ball_query
from . import group_points_ext
from .group_points import GroupAll, GroupingOperation


class QueryAndGroupV2(nn.Module):
    """Query and Group. 修改的地方主要在于 L109 多了一个self.return_triple

    Groups with a ball query of radius

    Args:
        max_radius (float): The maximum radius of the balls.
        sample_num (int): Maximum number of features to gather in the ball.
        min_radius (float): The minimum radius of the balls.
        use_xyz (bool): Whether to use xyz.
            Default: True.
        return_grouped_xyz (bool): Whether to return grouped xyz.
            Default: False.
        normalize_xyz (bool): Whether to normalize xyz.
            Default: False.
        uniform_sample (bool): Whether to sample uniformly.
            Default: False
        return_unique_cnt (bool): Whether to return the count of
            unique samples.
            Default: False.
    """

    def __init__(self,
                 max_radius,
                 sample_num,
                 min_radius=0,
                 use_xyz=True,
                 return_grouped_xyz=False,
                 normalize_xyz=False,
                 uniform_sample=False,
                 return_unique_cnt=False,
                 return_triple=False):
        super(QueryAndGroupV2, self).__init__()
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sample_num = sample_num
        self.use_xyz = use_xyz
        self.return_grouped_xyz = return_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.uniform_sample = uniform_sample
        self.return_unique_cnt = return_unique_cnt
        if self.return_unique_cnt:
            assert self.uniform_sample
        self.return_triple = return_triple

    def forward(self, points_xyz, center_xyz, features=None):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            center_xyz (Tensor): (B, npoint, 3) Centriods. 采样的中心定位点
            features (Tensor): (B, C, N) Descriptors of the features.

        Return：
            Tensor: (B, 3 + C, npoint, sample_num) Grouped feature.
        """
        idx = ball_query(self.min_radius, self.max_radius, self.sample_num,
                         points_xyz, center_xyz)

        if self.uniform_sample:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(
                        0,
                        num_unique, (self.sample_num - num_unique, ),
                        dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = points_xyz.transpose(1, 2).contiguous()
        # (B, 3, npoint, sample_num)
        grouped_xyz = grouping_operation(xyz_trans, idx)
        grouped_xyz -= center_xyz.transpose(1, 2).unsqueeze(-1)  # 返回的坐标都是减去中心点坐标之后的相对坐标了
        if self.normalize_xyz:
            grouped_xyz /= self.max_radius

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                # (B, C + 3, npoint, sample_num) 将xyz 和 feature 拼接到一起
                new_features = torch.cat([grouped_xyz, grouped_features],
                                         dim=1)
            else:
                new_features = grouped_features
        else:
            assert (self.use_xyz
                    ), 'Cannot have not features and not use xyz as a feature!'
            new_features = grouped_xyz

        ret = [new_features]
        if self.return_grouped_xyz:
            ret.append(grouped_xyz)
        if self.return_unique_cnt:
            ret.append(unique_cnt)
        if self.return_triple:
            ret.append(torch.cat([grouped_xyz, grouped_features], dim=1))
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


grouping_operation = GroupingOperation.apply