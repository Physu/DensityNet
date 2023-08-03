import numpy as np
from mmcv import is_tuple_of

from mmdet3d.core.bbox import box_np_ops
from mmdet.datasets.builder import PIPELINES

import random
import sklearn.cluster as sc
import math
from .transforms_3d import IndoorPointSample
import cv2
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import BboxOverlaps3D
import torch
from mmdet3d.core.bbox import (BaseInstance3DBoxes, Box3DMode,
                               CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, bbox3d2roi,
                               bbox3d_mapping_back)
from shapely.geometry import Polygon


@PIPELINES.register_module()
class PointWiseMask(object):
    """Make a mask which indicate whether the points inside the ground-truth bounding box.

    Args:
        None.
    """

    def __init__(self):
        pass

    def __call__(self, input_dict):
        """Call function to filter points by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' keys are updated \
                in the result dict.
        """
        points = input_dict['points']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points_numpy = points.tensor.numpy()
        gt_bboxes_3d_np_backup = gt_bboxes_3d.tensor.clone().numpy()
        foreground_masks = box_np_ops.points_in_rbbox(points_numpy, gt_bboxes_3d_np_backup)

        foreground_masks = foreground_masks.max(1)
        # idx = np.where(foreground_masks == True)
        input_dict['point_wise_mask'] = foreground_masks
        # with open('./points_to_show/point_wise.npy', 'wb') as f:
        #     np.save(f, points_numpy[:, 0:3][foreground_masks])
        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(bbox_enlarge_range={})'.format(
            self.bbox_enlarge_range.tolist())
        return repr_str


@PIPELINES.register_module()
class BackgroundPointsFilterV2(object):
    """Filter background points near the bounding box.

    Args:
        bbox_enlarge_range (tuple[float], float): Bbox enlarge range.
    """

    def __init__(self, bbox_enlarge_range):
        assert (is_tuple_of(bbox_enlarge_range, float)
                and len(bbox_enlarge_range) == 3) \
               or isinstance(bbox_enlarge_range, float), \
            f'Invalid arguments bbox_enlarge_range {bbox_enlarge_range}'

        if isinstance(bbox_enlarge_range, float):
            bbox_enlarge_range = [bbox_enlarge_range] * 3
        self.bbox_enlarge_range = np.array(
            bbox_enlarge_range, dtype=np.float32)[np.newaxis, :]

    def __call__(self, input_dict):
        """Call function to filter points by the range and store original points for DBSCAN processing

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'points' and 'points_no_BKGFilter' keys are updated \
                in the result dict.
        """
        points = input_dict['points']  # 这个是 liDar 类型，注意了
        # points_ = np.array(input_dict['points'].tensor)
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        points_bin_filename = str(input_dict['sample_idx'])
        # gt_bboxes_3d_np_backup = gt_bboxes_3d.tensor.numpy()
        # .detach() 脱离了计算图，但还是共享内存
        # .clone() 仍在计算图内，但不共享内存
        input_dict["gt_bboxes_3d_np_backup"] = gt_bboxes_3d.tensor.clone().detach()  # to store original points, for DBSCAN processing
        input_dict["gt_bboxes_3d_np_backup_corners"] = gt_bboxes_3d.corners.clone().detach()
        # with open('points_to_show/backgroundpointfilter/origin_corners'+points_bin_filename+'.npy', 'ab+') as f:
        #     np.save(f, input_dict["gt_bboxes_3d_np_backup_corners"])

        gt_bboxes_3d_np = gt_bboxes_3d.tensor.clone().numpy()
        gt_bboxes_3d_np_no_lift = gt_bboxes_3d.tensor.clone().numpy()  # tensor 转为numpy
        enlarged_gt_bboxes_3d_test = gt_bboxes_3d_np.copy()
        # 将box的中心点提高了,不光rboxes跟着改变，而且corners也跟着变化
        gt_bboxes_3d_np[:, :3] = gt_bboxes_3d.gravity_center.clone().numpy()


        # enlarged_gt_bboxes_3d_backup = gt_bboxes_3d_np_backup.copy()
        # 这个是提升z轴高度之后的
        enlarged_gt_bboxes_3d = gt_bboxes_3d_np.copy()
        # xyz 方向上增大范围
        enlarged_gt_bboxes_3d[:, 3:6] += self.bbox_enlarge_range
        enlarged_gt_bboxes_3d_test[:, 3:6] += self.bbox_enlarge_range
        # backup 增大范围
        # enlarged_gt_bboxes_3d_backup[:, 3:6] += self.bbox_enlarge_range
        # with open('points_to_show/backgroundpointfilter/enlarge_corners'+points_bin_filename+'.npy', 'ab+') as f:
        #     np.save(f, gt_bboxes_3d_np.corners.clone().numpy())

        points_numpy = points.tensor.clone().numpy()

        # foreground_masks_backup = box_np_ops.points_in_rbbox(points_numpy,
        #                                               gt_bboxes_3d_np_backup)
        # enlarge_foreground_masks_backup = box_np_ops.points_in_rbbox(
        #     points_numpy, enlarged_gt_bboxes_3d_backup)
        # foreground_masks_backup = foreground_masks_backup.max(1)
        # enlarge_foreground_masks_backup = enlarge_foreground_masks_backup.max(1)
        # valid_masks_backup = ~np.logical_and(~foreground_masks_backup,
        #                               enlarge_foreground_masks_backup)
        #
        foreground_masks = box_np_ops.points_in_rbbox(points_numpy,
                                                      gt_bboxes_3d_np, origin=(0.5, 0.5, 0.5))

        foreground_masks_test = box_np_ops.points_in_rbbox(points_numpy,
                                                      gt_bboxes_3d_np_no_lift)

        enlarge_foreground_masks = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d, origin=(0.5, 0.5, 0.5))

        enlarge_foreground_masks_test = box_np_ops.points_in_rbbox(
            points_numpy, enlarged_gt_bboxes_3d_test)
        foreground_masks = foreground_masks.max(1)
        foreground_masks_test = foreground_masks_test.max(1)
        # with open('points_to_show/backgroundpointfilter/foreground_points'+points_bin_filename+'.npy', 'ab+') as f:
        #     np.save(f, points_numpy[foreground_masks])
        # with open('points_to_show/backgroundpointfilter/foreground_points_test' + points_bin_filename + '.npy',
        #           'ab+') as f:
        #     np.save(f, points_numpy[foreground_masks_test])
        enlarge_foreground_masks = enlarge_foreground_masks.max(1)
        enlarge_foreground_masks_test = enlarge_foreground_masks_test.max(1)
        # with open('points_to_show/backgroundpointfilter/enlarge_foreground_points'+points_bin_filename+'.npy', 'ab+') as f:
        #     np.save(f, points_numpy[enlarge_foreground_masks])
        # with open('points_to_show/backgroundpointfilter/enlarge_foreground_points_test' + points_bin_filename + '.npy',
        #           'ab+') as f:
        #     np.save(f, points_numpy[enlarge_foreground_masks_test])
        valid_masks = ~np.logical_and(~foreground_masks,
                                      enlarge_foreground_masks)
        valid_masks_test = ~np.logical_and(~foreground_masks_test,
                                      enlarge_foreground_masks_test)

        # with open('points_to_show/backgroundpointfilter/after_enlarge_foreground_points'+points_bin_filename+'.npy', 'ab+') as f:
        #     np.save(f, points_numpy[valid_masks])
        # with open(
        #         'points_to_show/backgroundpointfilter/after_enlarge_foreground_points_test' + points_bin_filename + '.npy',
        #         'ab+') as f:
        #     np.save(f, points_numpy[valid_masks_test])

        input_dict['points'] = points[valid_masks]

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[valid_masks]

        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[valid_masks]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += '(bbox_enlarge_range={})'.format(
            self.bbox_enlarge_range.tolist())
        return repr_str


# added by physu
@PIPELINES.register_module()
class DBSCANSampleShrinkV5(object):
    """Sample Proposal objects to the data.

      Args:
          db_sampler (dict): Config dict of the database sampler.
          sample_2d (bool): Whether to also paste 2D image patch to the images
              This should be true when applying multi-modality cut-and-paste.
              Defaults to False.
      """

    def __init__(self, eps=1, scales=[1, 3, 5], num_points=16384, shrink=[0.05, 0.05, 0.05], alternate=True, downsample=(0.0, 0.3, 0.5)):
        # define the initial eps value
        self.eps = eps
        self.scales = scales
        self.num_points = num_points
        self.shrink = shrink
        self.all_gt_rboxes = []
        self.all_final_rboxes = []
        self.alternate = alternate
        self.downsample = downsample

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (np.ndarray): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points, boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    @staticmethod
    def points_random_sampling(points,
                               num_samples,
                               replace=None,
                               return_choices=False):
        """Points random sampling.

        Sample points to a certain number.

        Args:
            points (np.ndarray): 3D Points.
            num_samples (int): Number of samples to be sampled.
            replace (bool): Whether the sample is with or without replacement.
            Defaults to None.
            return_choices (bool): Whether return choice. Defaults to False.

        Returns:
            tuple[np.ndarray] | np.ndarray:

                - points (np.ndarray): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if replace is None:
            replace = (points.shape[0] < num_samples)
            # 从数组中随机抽取元素，这时候就需要用到np.random.choice()
        choices = np.random.choice(
            points.shape[0], num_samples, replace=replace)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation, \
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated \
                in the result dict.
        """
        points_ = input_dict['points'][:, 0:3]
        gt_bboxes_3d_np_backup = input_dict['gt_bboxes_3d'].tensor.clone().numpy()
        gt_bboxes_3d_np_backup_corners = input_dict['gt_bboxes_3d'].corners.clone().numpy()
        # points_bin_filename = input_dict['sample_idx']  # int idx
        # gt_bboxes_3d = input_dict['gt_bboxes_3d']
        # gt_bboxes_3d_np = gt_bboxes_3d.tensor.numpy()  # tensor 转为numpy

        # 没有gravity_center处理
        points = np.array(points_.tensor)


        # 用来获取lidarpoints 类的 attribute tensor
        final_cubes_ = []
        final_tensor = []
        rboxes2proceeding = []
        boxes2proceeding = []
        if self.alternate:
            flag = random.randint(0, 1)
        else:
            flag = 0
        if flag:
            for scale_index, scale in enumerate(self.scales):
                eps_ = self.eps * scale
                y2_pred = sc.DBSCAN(eps_, min_samples=3).fit_predict(points)
                num_difference = len(set(y2_pred))  # 聚类了多少类，-1 表示未归属任何一类
                cluster = []

                for i in np.arange(num_difference - 1):
                    sub_cluster = []
                    # get the prediction class index
                    index = np.where(y2_pred == i)
                    # save the same class's point cloud
                    for value in index:
                        for sub_value in value:
                            # get the points belong to one cluster
                            sub_cluster.append(points[int(sub_value)])
                    cluster.append(sub_cluster)  # [[1,2,3,4],[5,6,7,8]]
                    # cluster.extend(sub_cluster)  # [[1,2,3,4,5,6,7,8]]
                subcubes, rcubes = self.gennerate_boxes_dbscan_v2(cluster, self.shrink)
                final_cubes_.extend(subcubes)
                final_tensor.extend(rcubes)
            final_cubes = np.array(final_cubes_)
            # with open('points_to_show/final_cubes.npy', 'ab+') as f:
            #     np.save(f, final_cubes)

            for tensor, corners in zip(gt_bboxes_3d_np_backup, gt_bboxes_3d_np_backup_corners):
                for tensor_filter, cubes_filter in zip(final_tensor, final_cubes):
                    overlap_four = self.by_diameter_to_filter(corners, tensor, cubes_filter, tensor_filter)  # 返回True or False
                    if overlap_four:
                        rboxes2proceeding.append(tensor_filter)
                        # boxes2proceeding.append(cubes_filter)

            # with open('points_to_show/rboxes2proceeding.npy', 'ab+') as f:
            #     np.save(f, rboxes2proceeding)
            # with open('points_to_show/points/points_' + str(points_bin_filename) + '.npy', 'ab+') as f:
            #     np.save(f, points)

            # with open('points_to_show/db_cubes/boxes2proceeding_'+str(points_bin_filename)+'.npy', 'ab+') as f:
            #     np.save(f, boxes2proceeding)
            # with open('points_to_show/gt_corners/gt_bboxes_3d_np_backup_corners_'+str(points_bin_filename)+'.npy', 'ab+') as f:
            #     np.save(f, gt_bboxes_3d_np_backup_corners)

            if len(rboxes2proceeding) > 0:
                rgt_masks = box_np_ops.points_in_rbbox(points, np.array(gt_bboxes_3d_np_backup))
                # 这个处理的方法可以将不同尺度下的db_cluster_box 都纳入考虑
                rboxes2proceeding_masks = box_np_ops.points_in_rbbox(points, np.array(rboxes2proceeding))

                foreground_masks = rgt_masks.max(1)
                dbscan_masks = rboxes2proceeding_masks.max(1)
                # 下面代码可视化分析用

                # with open('points_to_show/points_gt/points_rgt_'+str(points_bin_filename)+'.npy', 'ab+') as f:
                #     np.save(f, points[foreground_masks])
                # with open('points_to_show/points_dbscan/points_dbscan_masks_'+str(points_bin_filename)+'.npy', 'ab+') as f:
                #     np.save(f, points[dbscan_masks])
                # with open('points_to_show/pointBK.npy', 'ab+') as f:
                #     np.save(f, points)

                # 下面是降采样的过程

                dbscan_masks_index = np.where(dbscan_masks == True)  # 选出为True的索引
                index_downsample = random.randint(0, 2)
                size = int(len(dbscan_masks_index[0]) * self.downsample[index_downsample])  # 最后保留10%的点
                dbscan_masks_random_choice = np.random.choice(dbscan_masks_index[0], size=size, replace=False)
                for index in dbscan_masks_random_choice:
                    dbscan_masks[index] = False

                ##################
                valid_masks = np.logical_or(np.logical_and(foreground_masks, ~dbscan_masks), ~np.logical_or(foreground_masks, dbscan_masks))
                # valid_masks2 = np.logical_or(np.logical_and(foreground_masks, ~dbscan_masks), ~foreground_masks)
                # valid_masks_num = np.nonzero(valid_masks)
                # valid_masks2_num = np.nonzero(valid_masks2)


                input_dict['points'] = input_dict['points'][valid_masks]
                # dbscan_masks_num2 = np.nonzero(dbscan_masks2)
                # dbscan_masks_num = np.nonzero(dbscan_masks)
                # dbscan_masks2 = dbscan_masks.copy()
                # size2 = int(len(dbscan_masks_index[0]) * 0.9)  # 用于可视化，最后保留10%的点
                # dbscan_masks_random_choice2 = np.random.choice(dbscan_masks_index[0], size=size2, replace=False)
                # for index in dbscan_masks_random_choice2:
                #     dbscan_masks2[index] = False
                #     foreground_masks[index] = False
                # with open('points_to_show/points_downsample/points_dbscan_masks_downsample_'+str(points_bin_filename)+'.npy', 'ab+') as f:
                #     np.save(f, points[foreground_masks])
                # with open('points_to_show/points_final/points_return_'+str(points_bin_filename)+'.npy', 'ab+') as f:
                #     np.save(f, np.array(points[valid_masks]))  # 获取gt corner信息

            indoorpointsample = IndoorPointSample(num_points=16384)
            input_dict = indoorpointsample(input_dict)

        else:
            # 如果flag为1，则直接调用原来的方法，达到交替使用的目的
            indoorpointsample = IndoorPointSample(num_points=16384)
            input_dict = indoorpointsample(input_dict)

        return input_dict


    @staticmethod
    def gennerate_boxes_dbscan_v2(cluster, shrink):
        '''
        :param cluster: all the same cluster points
        :return: a bounding box's six planes
        '''
        # point_reserve = []
        cubes = []
        rcubes = []

        # index_reserve = np.array([])
        # initial six plans to construct a cube
        for index, subcluster in enumerate(cluster):
            cmax = np.max(subcluster, axis=0)
            cmin = np.min(subcluster, axis=0)

            x_max = cmax[0]
            y_max = cmax[1]
            z_max = cmax[2]
            x_min = cmin[0]
            y_min = cmin[1]
            z_min = cmin[2]
            width = x_max - x_min
            height = y_max - y_min
            length = z_max - z_min
            center_x = (x_max + x_min) / 2
            center_y = (y_max + y_min) / 2
            center_z = (z_max + z_min) / 2

            rbox = [center_x, center_y, center_z - length/2, width, height, length, 0.0]
            if width > 5 or height > 5 or length < 0.16:
                continue
            # distance = math.sqrt(pow(center_x, 2) + pow(center_y, 2) + pow(center_z, 2))
            # 对应lidar坐标
            sub_cube = [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_max],
                [x_min, y_max, z_min],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_max, y_max, z_min]
            ]

            cubes.append(sub_cube)
            rcubes.append(rbox)

        return cubes, rcubes


    def pairs_compare(self, gt_rboxes, rboxes):
        index_ = []
        for index_gt, gt_rbox in enumerate(gt_rboxes):
            for index_db, rbox in enumerate(rboxes):
                # width height length filter
                gt_volume = gt_rbox[3] * gt_rbox[4] * gt_rbox[5]
                db_volume = rbox[3] * rbox[4] * rbox[5]
                if rbox[3] > gt_rbox[3] * 1.2 or rbox[4] > gt_rbox[4] * 1.2 \
                        or rbox[5] > gt_rbox[5] * 1.2 or rbox[5] < 0.2:
                    continue
                elif math.sqrt(
                        pow(rbox[0] - gt_rbox[0], 2) + pow(rbox[1] - gt_rbox[1], 2) + pow(rbox[2] - gt_rbox[2], 2)) \
                        > (gt_rbox[3] + gt_rbox[4])/2:
                    continue
                elif db_volume > gt_volume * 1.5:
                    continue
                else:
                    index_.append([index_gt, index_db])
        return index_


    def pairs_compare_save(self, gt_rboxes, rboxes):
        index_ = []
        index_save = []
        for index_gt, gt_rbox in enumerate(gt_rboxes):
            for index_db, rbox in enumerate(rboxes):
                # width height length filter
                gt_volume = gt_rbox[3] * gt_rbox[4] * gt_rbox[5]
                db_volume = rbox[3] * rbox[4] * rbox[5]
                if rbox[3] > gt_rbox[3] * 1.2 or rbox[4] > gt_rbox[4] * 1.2 \
                        or rbox[5] > gt_rbox[5] * 1.2 or rbox[5] < 0.16:
                    continue
                elif math.sqrt(
                        pow(rbox[0] - gt_rbox[0], 2) + pow(rbox[1] - gt_rbox[1], 2) + pow(rbox[2] - gt_rbox[2], 2)) \
                        > (gt_rbox[3] + gt_rbox[4])/2:
                    continue
                elif db_volume > gt_volume * 1.5:
                    continue
                else:
                    index_.append([index_gt, index_db])
                    index_save.append(index_db)
        return index_, index_save


    def pairs_compare_savev2(self, gt_rboxes, rboxes, boxes):
        index_ = []
        index_save = []
        for index_gt, gt_rbox in enumerate(gt_rboxes):
            for index_db, rbox in enumerate(rboxes):
                # width height length filter
                gt_volume = gt_rbox[3] * gt_rbox[4] * gt_rbox[5]
                db_volume = rbox[3] * rbox[4] * rbox[5]
                if rbox[3] > gt_rbox[3] * 1.2 or rbox[4] > gt_rbox[4] * 1.2 \
                        or rbox[5] > gt_rbox[5] * 1.2 or rbox[5] < 0.16:
                    continue
                elif math.sqrt(
                        pow(rbox[0] - gt_rbox[0], 2) + pow(rbox[1] - gt_rbox[1], 2) + pow(rbox[2] - gt_rbox[2], 2)) \
                        > (gt_rbox[3] + gt_rbox[4])/2:
                    continue
                elif db_volume > gt_volume * 1.5:
                    continue
                else:
                    index_.append([index_gt, index_db])
                    index_save.append(index_db)
        return index_, index_save


    def pairs_compare_according_gt_index(self, gt_rboxes, rboxes):
        '''

        :param gt_rboxes:
        :param rboxes:
        :return: 按照gt index 返回对应的候选 candidate，即第0个，对应几个dbcubes;第1个，对应几个dbcubes
        '''
        according_index = []
        # masks = ~np.ones((len(gt_rboxes), len(rboxes)), dtype=np.bool)
        for index_gt, gt_rbox in enumerate(gt_rboxes):
            index_ = []
            for index_db, rbox in enumerate(rboxes):
                # width height length filter
                if rbox[3] > gt_rbox[3] * 2 or rbox[4] > gt_rbox[4] * 2 \
                        or rbox[5] > gt_rbox[5] * 2 or rbox[5] < 0.2:
                    continue  # 尺寸限制
                elif math.sqrt(
                        pow(rbox[0] - gt_rbox[0], 2) + pow(rbox[1] - gt_rbox[1], 2) + pow(rbox[2] - gt_rbox[2], 2)) \
                        > (gt_rbox[3] + gt_rbox[4])/2:
                    continue  # 二者中心距离超过 长宽之和/2

                gt_volume = gt_rbox[3] * gt_rbox[4] * gt_rbox[5]
                db_volume = rbox[3] * rbox[4] * rbox[5]
                if db_volume > gt_volume * 8:
                    continue  # 体积限制
                else:
                    index_.append([index_gt, index_db])
            according_index.append(index_)
        return according_index


    def iou_four_point(self, boxes1, tensor, boxes2, tensor_filter):
        # 注意再lidar 坐标系下：顺序是先前面 顺时针，然后后面 顺时针，这个顺序
        # 所以在这里应该是0-4-7-3这样一个顺序， 注意这里必须是mask后的顺序，
        # corner1[1]对应的位置其实是原来八个点中的第4个
        # boxes1 代表gt， boxes2 代表聚类出来 corner
        '''

        Args:
            boxes1: (8,3) gt 8 corners
            tensor: (7,) gt xyz width length height
            boxes2: dbscan 8 corners
            tensor_filter: (7,) dbscan xyz width length height

        Returns:
            iou whether there is a intersection area between boxes1 and boxes2

        '''
        mask = [True, False, False, True, True, False, False, True]
        # if boxes2[2][2] > boxes1[2][2]:
        #     # 将聚类产生的box高度大于 gt 高度最高点的过滤掉
        #     return 0.0

        corner1 = boxes1[mask]
        corner2 = boxes2[mask]
        cor1 = Polygon([(corner1[0][0], corner1[0][1]), (corner1[1][0], corner1[1][1]),
                        (corner1[3][0], corner1[3][1]), (corner1[2][0], corner1[2][1])]).buffer(0.01)

        cor2 = Polygon([(corner2[0][0], corner2[0][1]), (corner2[1][0], corner2[1][1]),
                        (corner2[3][0], corner2[3][1]), (corner2[2][0], corner2[2][1])]).buffer(0.01)

        union_area = cor1.union(cor2).area
        # 之所以还要开方运算，是为了将向量变为标量，进行下一步的diameter 计算
        diameter1 = np.sqrt((boxes1[0][0] - boxes1[7][0]) ** 2 + (boxes1[0][1] - boxes1[7][1]) ** 2)
        diameter2 = np.sqrt((boxes2[0][0] - boxes2[7][0]) ** 2 + (boxes2[0][1] - boxes2[7][1]) ** 2)
        diameterof2boxes = (diameter1/2 + diameter2/2) ** 2
        diameter = ((boxes1[0][0] + boxes1[7][0]) / 2 - (boxes2[0][0] + boxes2[7][0]) / 2) ** 2 + \
                   ((boxes1[0][1] + boxes1[7][1]) / 2 - (boxes2[0][1] + boxes2[7][1]) / 2) ** 2
        distance = diameter < diameterof2boxes

        diameter_gt = np.sqrt(tensor[3] ** 2 + tensor[4] ** 2)
        diameter_db = np.sqrt(tensor_filter[3] ** 2 + tensor_filter[4] ** 2)
        diameter_2boxes = (diameter_gt/2 + diameter_db/2)**2
        diameter_ = (tensor[0] - tensor_filter[0])**2 + (tensor[1] - tensor_filter[1]) ** 2

        if union_area == 0.0:
            iou = 0.0
            iou_flag = False
        else:
            iou = cor1.intersection(cor2).area / union_area
            if iou == 0.0:
                iou_flag = False
            else:
                iou_flag = True
        if iou_flag != distance:
            print(f"iou:{iou}, distance:{distance}, iou_flag:{iou_flag}")
        return iou


    def by_diameter_to_filter(self, boxes1, tensor, boxes2, tensor_filter):
        # 之所以还要开方运算，是为了将向量变为标量，进行下一步的diameter 计算

        diameter_gt = np.sqrt(tensor[3] ** 2 + tensor[4] ** 2)
        diameter_db = np.sqrt(tensor_filter[3] ** 2 + tensor_filter[4] ** 2)
        diameter_2boxes = (diameter_gt / 2 + diameter_db / 2) ** 2
        diameter_ = (tensor[0] - tensor_filter[0]) ** 2 + (tensor[1] - tensor_filter[1]) ** 2

        if diameter_ < diameter_2boxes:
            return True
        else:
            return False