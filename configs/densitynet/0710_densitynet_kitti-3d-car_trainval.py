model = dict(
    type='DensityMaskNet',
    backbone=dict(
        type='PointNet2SAMSGMask',
        in_channels=4,
        num_points=(4096, 1024, (256, 256)),  # num_points=(4096, 512, (256, 256)),
        radii=((0.2, 0.4, 0.8), (0.4, 0.8, 1.6), (1.6, 3.2, 4.8)),
        num_samples=((32, 32, 64), (32, 32, 64), (32, 32, 32)),
        sa_channels=(((16, 16, 32), (16, 16, 32), (32, 32, 64)),
                     ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                     ((128, 128, 256), (128, 192, 256), (128, 256, 256))),
        aggregation_channels=(64, 128, 256),
        xyz_se_channels=(((3, 16, 32), (3, 16, 32), (3, 32, 64)),
                         ((3, 32, 128), (3, 64, 128), (3, 96, 128)),
                         ((3, 64, 256), (3, 128, 256), (3, 192, 256))),
        feat_se_channels=(((1, 16, 32), (1, 16, 32), (1, 32, 64)),
                          ((64, 64, 128), (64, 64, 128), (64, 96, 128)),
                          ((128, 128, 256), (128, 128, 256), (128, 192, 256))),
        se_channels=(((32, 1), (32, 1), (64, 1)),
                     ((32, 1), (32, 1), (64, 1)),
                     ((32, 1), (32, 1), (32, 1))),  # 和 num_samples 设置有关
        se_shortcut_channels=(((4, 32), (4, 32), (4, 64)),
                              ((67, 128), (67, 128), (67, 128)),
                              ((131, 256), (131, 256), (131, 256))),
        fps_mods=('D-FPS', 'MS4', ('F-FPS', 'D-FPS')),
        fps_sample_range_lists=(-1, -1, (512, -1)),
        norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.1),
        sa_cfg=dict(
            type='PointSAModuleMSGMaskV1',
            pool_mod='max',
            use_xyz=False,
            # True: return xyz+feat False:return feat, there is no other option, so set "False" first, but "return_triple" we get same as set "True"
            normalize_xyz=False,
            use_origin_sa=True,
            # use original version or auther designed  # 如果使用原版sa，use_origin_sa 置为 True, 和return_triple 和 return_group_xyz 需要置为False
            return_grouped_xyz=True,  # whether return xyz and feat separately in QueryAndGroupV2
            return_triple=True  # whether return xyz + feat
         )),
    bbox_head=dict(
        type='DensityMaskNet3DHead',
        in_channels=256,
        vote_module_cfg=dict(
            in_channels=256,
            num_points=256,
            gt_per_seed=1,
            conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
            with_res_feat=False,
            vote_xyz_range=(3.0, 3.0, 2.0)),
        vote_aggregation_cfg=dict(
            type='PointSAModuleMSGMaskV1',
            num_point=256,
            radii=(4.8, 6.4),
            sample_nums=(16, 32),  # sa 的输入（B，）
            mlp_channels=((256, 256, 256, 512), (256, 256, 512, 1024)),
            xyz_se_channels=((3, 64, 512), (3, 128, 1024)),
            feat_se_channels=((256, 256, 512), (256, 512, 1024)),
            se_channels=((16, 1), (32, 1)),
            se_shortcut_channels=((259, 512), (259, 1024)),
            norm_cfg=dict(type='BN2d', eps=0.001, momentum=0.1),
            normalize_xyz=False,
            bias=True,
            sa_index=2,  # 为了调用 se module的 pointnet
            use_xyz=False,  # 为了实现 QueryAndGroupV2中 xyz 和 feature 分开返回
            return_grouped_xyz=True,  # whether return xyz and feat separately in QueryAndGroupV2
            return_triple=True  # whether return xyz + feat
        ),  # 使用和最后一层SA同样的设置
        pred_layer_cfg=dict(
            in_channels=1536,
            shared_conv_channels=(512, 128),
            cls_conv_channels=(128, ),
            reg_conv_channels=(128, ),
            conv_cfg=dict(type='Conv1d'),
            norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
            bias=True),
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.1),
        objectness_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='sum',
            loss_weight=1.0),
        center_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        dir_class_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0),
        dir_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        size_res_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        corner_loss=dict(
            type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        vote_loss=dict(type='SmoothL1Loss', reduction='sum', loss_weight=1.0),
        mask_loss=dict(
            type='CrossEntropyLoss', reduction='sum', loss_weight=1.0, use_sigmoid=True),
        num_classes=1,
        bbox_coder=dict(
            type='AnchorFreeBBoxCoder', num_dir_bins=12, with_rot=True)),
    train_cfg=dict(
        sample_mod='spec', pos_distance_thr=10.0, expand_dims_length=0.05),
    test_cfg=dict(
        nms_cfg=dict(type='nms', iou_thr=0.1),
        sample_mod='spec',
        score_thr=0.0,
        per_class_proposal=True,
        max_output_num=100))
dataset_type = 'KittiDataset'
data_root = 'data/kitti/'
class_names = ['Car']
point_cloud_range = [0, -40, -5, 70, 40, 3]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    data_root='data/kitti/',
    info_path='data/kitti/kitti_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
    classes=['Car'],
    sample_groups=dict(Car=15))
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        file_client_args=dict(backend='disk')),
    dict(type='PointsRangeFilter', point_cloud_range=[0, -40, -5, 70, 40, 3]),
    dict(type='ObjectRangeFilter', point_cloud_range=[0, -40, -5, 70, 40, 3]),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root='data/kitti/',
            info_path='data/kitti/kitti_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1], filter_by_min_points=dict(Car=5)),
            classes=['Car'],
            sample_groups=dict(Car=15))),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(
        type='ObjectNoise',
        num_try=100,
        translation_std=[1.0, 1.0, 0],
        global_rot_range=[0.0, 0.0],
        rot_range=[-1.0471975511965976, 1.0471975511965976]),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.9, 1.1]),
    dict(type='BackgroundPointsFilter', bbox_enlarge_range=(0.5, 2.0, 0.5)),
    dict(type='IndoorPointSample', num_points=16384),
    dict(type='DefaultFormatBundle3D', class_names=['Car']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[0, -40, -5, 70, 40, 3]),
            dict(type='IndoorPointSample', num_points=16384),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Car'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(type='DefaultFormatBundle3D', class_names=['Car'], with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type='KittiDataset',
            data_root='data/kitti/',
            ann_file='data/kitti/kitti_infos_trainval.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=4,
                    use_dim=4,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[0, -40, -5, 70, 40, 3]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[0, -40, -5, 70, 40, 3]),
                dict(
                    type='ObjectSample',
                    db_sampler=dict(
                        data_root='data/kitti/',
                        info_path='data/kitti/kitti_dbinfos_train.pkl',
                        rate=1.0,
                        prepare=dict(
                            filter_by_difficulty=[-1],
                            filter_by_min_points=dict(Car=5)),
                        classes=['Car'],
                        sample_groups=dict(Car=15))),
                dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
                dict(
                    type='ObjectNoise',
                    num_try=100,
                    translation_std=[1.0, 1.0, 0],
                    global_rot_range=[0.0, 0.0],
                    rot_range=[-1.0471975511965976, 1.0471975511965976]),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.9, 1.1]),
                dict(
                    type='BackgroundPointsFilterV2',
                    bbox_enlarge_range=(0.5, 2.0, 0.5)),
                dict(
                    type='DBSCANSampleShrinkV5',
                    eps=1,
                    scales=[0.12, 0.32, 0.8],
                    num_points=16384,
                    shrink=[0, 0, 0],
                    downsample=[0.1, 0.3, 0.5]),
                # dict(type='IndoorPointSample', num_points=16384),
                dict(type='PointWiseMask'),
                dict(type='DefaultFormatBundle3DV2', class_names=['Car']),
                dict(
                    type='Collect3DV2',
                    keys=[
                        'points', 'gt_bboxes_3d', 'gt_labels_3d',
                        'point_wise_mask'
                    ])
            ],
            modality=dict(use_lidar=True, use_camera=False),
            classes=['Car'],
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -40, -5, 70, 40, 3]),
                    dict(type='IndoorPointSample', num_points=16384),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Car'],
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='KittiDataset',
        data_root='data/kitti/',
        ann_file='data/kitti/kitti_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[0, -40, -5, 70, 40, 3]),
                    dict(type='IndoorPointSample', num_points=16384),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        modality=dict(use_lidar=True, use_camera=False),
        classes=['Car'],
        test_mode=True,
        box_type_3d='LiDAR'))
evaluation = dict(
    interval=100,
    pipeline=[
        dict(
            type='LoadPointsFromFile',
            coord_type='LIDAR',
            load_dim=4,
            use_dim=4,
            file_client_args=dict(backend='disk')),
        dict(
            type='DefaultFormatBundle3D',
            class_names=['Car'],
            with_label=False),
        dict(type='Collect3D', keys=['points'])
    ])
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=30,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'newback/3dssd_8x1_kitti-3d-car_trainval'
load_from = None
resume_from = None
workflow = [('train', 1)]
lr = 0.001
optimizer = dict(type='AdamW', lr=0.001, weight_decay=0)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[45, 60])
runner = dict(type='EpochBasedRunner', max_epochs=80)
gpu_ids = range(0, 1)
