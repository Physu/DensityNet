import torch
from mmcv.cnn import ConvModule
from torch import nn as nn
from torch.nn import functional as F
from typing import List

from mmdet3d.ops import GroupAll, Points_Sampler, QueryAndGroup, gather_points, QueryAndGroupV2
from .builder import SA_MODULES
from mmdet3d.ops.furthest_point_sample.points_samplerV2 import Points_SamplerV2
from mmdet3d.ops.pointnet_modules.point_sa_module import PointSAModuleMSG


@SA_MODULES.register_module()
class PointSAModuleMSGMask(nn.Module):
    """Point set abstraction module with multi-scale grouping used in
    Pointnets.
    :todo 原始的设计的SA module, 每个SA包含三个不同尺度的radius，都采用SENet形式来设计。这个方案模型较大，效果提升不大

    Args:
        num_point (int): Number of points.
        radii (list[float]): List of radius in each ball query.
        sample_nums (list[int]): Number of samples in each ball query.
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
            F-FPS: using feature distances for FPS.
            D-FPS: using Euclidean distances of points for FPS.
            FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Default: False.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 mlp_channels: List[List[int]],  # 这个 list[0][0] 会+3，增加xyz维度，但后面的不会添加
                 xyz_se_channels: List[List[int]],
                 feat_se_channels: List[List[int]],
                 se_channels: List[List[int]],
                 se_shortcut_channels: List[List[int]],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False,
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod='max',
                 normalize_xyz: bool = False,
                 bias='auto',
                 use_origin_sa=True,
                 return_grouped_xyz=False,
                 return_triple=False,
                 ):
        super().__init__()

        assert len(radii) == len(sample_nums) == len(mlp_channels)
        assert pool_mod in ['max', 'avg']
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(mlp_channels, tuple):
            mlp_channels = list(map(list, mlp_channels))

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        else:
            raise NotImplementedError('Error type of num_point!')

        self.pool_mod = pool_mod
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.mlps_xyz_se = nn.ModuleList()
        self.mlps_feat_se = nn.ModuleList()
        self.change_se = nn.ModuleList()
        self.shortcut_se = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list
        # First finishing the class initialization!!!
        self.points_sampler = Points_SamplerV2(self.num_point, self.fps_mod_list,
                                               self.fps_sample_range_list)
        # self.points_sampler_debug = Points_SamplerV2(self.num_point, ['MS'], self.fps_sample_range_list)
        self.use_origin_sa = use_origin_sa

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    # whether to expand min radius, construct a hierarchical relationship
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroupV2(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    return_grouped_xyz=return_grouped_xyz,
                    return_triple=return_triple)  # 返回 feat，xyz，xyz+feat 三个值
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

            mlp_spec = mlp_channels[i]
            xyz_spec = xyz_se_channels[i]
            feat_spec = feat_se_channels[i]
            se_spec = se_channels[i]
            se_shortcut_spec = se_shortcut_channels[i]

            if use_xyz or return_triple:
                mlp_spec[0] += 3  # added 3 coordinate location dimension

            mlp = nn.Sequential()
            for i in range(len(mlp_spec) - 1):
                mlp.add_module(
                    f'layer{i}',
                    ConvModule(
                        mlp_spec[i],
                        mlp_spec[i + 1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
            self.mlps.append(mlp)

            if not self.use_origin_sa:
                xyz_se = nn.Sequential()  # xyz with squeeze and excitation
                for i in range(len(xyz_spec) - 1):
                    xyz_se.add_module(
                        f'layer{i}',
                        ConvModule(
                            xyz_spec[i],
                            xyz_spec[i + 1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=norm_cfg,
                            bias=bias))
                self.mlps_xyz_se.append(xyz_se)

                feat_se = nn.Sequential()  # xyz with squeeze and excitation
                for i in range(len(feat_spec) - 1):
                    feat_se.add_module(
                        f'layer{i}',
                        ConvModule(
                            feat_spec[i],
                            feat_spec[i + 1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=norm_cfg,
                            bias=bias))
                self.mlps_feat_se.append(feat_se)

                se_shortcut = nn.Sequential()
                se_shortcut.add_module(
                    f'layer{i}',
                    ConvModule(
                        se_shortcut_spec[0],
                        se_shortcut_spec[1],
                        kernel_size=(1, 1),
                        stride=(1, 1),
                        conv_cfg=dict(type='Conv2d'),
                        norm_cfg=norm_cfg,
                        bias=bias))
                self.shortcut_se.append(se_shortcut)

                # used to change se reulst to a special dimension
                change_mlp = nn.Sequential()
                change_mlp.add_module(
                        f'layer{i}',
                        ConvModule(
                            se_spec[0],
                            se_spec[1],
                            kernel_size=(1, 1),
                            stride=(1, 1),
                            conv_cfg=dict(type='Conv2d'),
                            norm_cfg=norm_cfg,
                            bias=bias))
                self.change_se.append(change_mlp)



    def forward(
        self,
        points_xyz: torch.Tensor,
        features: torch.Tensor = None,
        point_wise_mask: torch.Tensor = None,
        indices: torch.Tensor = None,
        target_xyz: torch.Tensor = None,
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """forward.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) features of each point.
                Default: None.
            indices (Tensor): (B, num_point) Index of the features.
                Default: None.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tensor: (B, M, 3) where M is the number of points.
                New features xyz.
            Tensor: (B, M, sum_k(mlps[k][-1])) where M is the number
                of points. New feature descriptors.
            Tensor: (B, M) where M is the number of points.
                Index of the features.
        """
        new_features_list = []
        preds = []
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            if point_wise_mask is not None:
                indices, preds = self.points_sampler(points_xyz, features, point_wise_mask)
            else:
                indices, preds = self.points_sampler(points_xyz, features)
            # 根据索引，获取所有的xyz坐标信息
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None

        if self.use_origin_sa:
            # 完成三次group的操作，就是三个radius做 ball_query 操作
            for i in range(len(self.groupers)):
                #  Return:  Tensor: (B, C + 3, 1, N) Grouped feature.
                # the new_xyz is the centroids
                new_features = self.groupers[i](points_xyz, new_xyz, features)

                # (B, mlp[-1], num_point, nsample)
                new_features = self.mlps[i](new_features)
                # In this QueryAndGroup, choose the max as output
                if self.pool_mod == 'max':
                    # (B, mlp[-1], num_point, 1)
                    new_features = F.max_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)])
                elif self.pool_mod == 'avg':
                    # (B, mlp[-1], num_point, 1)
                    new_features = F.avg_pool2d(
                        new_features, kernel_size=[1, new_features.size(3)])
                else:
                    raise NotImplementedError

                new_features = new_features.squeeze(-1)  # (B, mlp[-1], num_point)
                new_features_list.append(new_features)
        else:
            # 完成三次采样的操作，就是三个radius做 ball_query 操作
            for i in range(len(self.groupers)):
                #  Return:  Tensor: (B, C + 3, 1, N) Grouped feature.
                # :todo the new_xyz is the centroids
                new_features = self.groupers[i](points_xyz, new_xyz, features)
                feat_grouped = new_features[0]
                xyz_grouped = new_features[1]
                xyz_feat = new_features[2]  # 返回的feat + xyz：4，67，131 相当于分解过后，和最后混在一起的
                # (B, mlp[-1], num_point, nsample)
                new_features = self.mlps[i](xyz_feat)  # 经过这一步处理，channel 都变为4
                xyz_se_fetures = self.mlps_xyz_se[i](xyz_grouped)
                feat_se_features = self.mlps_feat_se[i](feat_grouped)

                xyz_se_fetures_sigmoid = torch.sigmoid(xyz_se_fetures)
                feat_se_features_sigmoid = torch.sigmoid(feat_se_features)

                excitation_by_add = xyz_se_fetures_sigmoid + feat_se_features_sigmoid  # attention
                new_features_1 = new_features * excitation_by_add

                # excitation_by_multi = xyz_se_fetures_sigmoid * feat_se_features_sigmoid  # two type excitation: add and multiply
                # new_features_1 = new_features * excitation_by_multi

                xyz_feat = self.shortcut_se[i](xyz_feat)  # change to special Dims
                new_features_2 = xyz_feat + new_features_1

                new_features_3 = self.change_se[i](new_features_2.transpose(1, 3))

                new_features_3 = new_features_3.transpose(1, 3)

                new_features = new_features_3.squeeze(-1)  # (B, mlp[-1], num_point)
                new_features_list.append(new_features)

        new_features_cat = torch.cat(new_features_list, dim=1)

        return new_xyz, new_features_cat, indices, preds


@SA_MODULES.register_module()
class PointSAModuleMask(PointSAModuleMSGMask):
    """Point set abstraction module used in Pointnets.

    Args:
        mlp_channels (list[int]): Specify of the pointnet before
            the global pooling for each scale.
        num_point (int): Number of points.
            Default: None.
        radius (float): Radius to group with.
            Default: None.
        num_sample (int): Number of samples in each ball query.
            Default: None.
        norm_cfg (dict): Type of normalization method.
            Default: dict(type='BN2d').
        use_xyz (bool): Whether to use xyz.
            Default: True.
        pool_mod (str): Type of pooling method.
            Default: 'max_pool'.
        fps_mod (list[str]: Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS'], Default: ['D-FPS'].
        fps_sample_range_list (list[int]): Range of points to apply FPS.
            Default: [-1].
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Default: False.
    """

    def __init__(self,
                 mlp_channels: List[int],
                 num_point: int = None,
                 radius: float = None,
                 num_sample: int = None,
                 norm_cfg: dict = dict(type='BN2d'),
                 use_xyz: bool = True,
                 pool_mod: str = 'max',
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 normalize_xyz: bool = False):
        super().__init__(
            mlp_channels=[mlp_channels],
            num_point=num_point,
            radii=[radius],
            sample_nums=[num_sample],
            norm_cfg=norm_cfg,
            use_xyz=use_xyz,
            pool_mod=pool_mod,
            fps_mod=fps_mod,
            fps_sample_range_list=fps_sample_range_list,
            normalize_xyz=normalize_xyz)
