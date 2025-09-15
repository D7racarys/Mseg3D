import itertools
import logging
import os.path as osp
import warnings

import numpy as np
from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)

# ==== Model config =====
model = dict(
    type="SegHubSegNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=4,
    ),
    backbone=dict(
        type="UNetSCN3D",
        num_input_features=4,
        norm_cfg=norm_cfg,
    ),
    img_backbone=dict(
        type="HRNet",
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block="BOTTLENECK",
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block="BASIC",
                num_blocks=(4, 4),
                num_channels=(18, 36),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block="BASIC",
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block="BASIC",
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144),
            ),
        ),
    ),
    img_head=dict(
        type="FCNImgHead",
        in_channels=270,
        channels=128,
        num_classes=16,
        dropout_ratio=0.1,
        ignore_label=0,
        loss_weight=0.4,
        loss_seg=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    point_head=dict(
        type="HubSegHead",
        class_agnostic=False,
        num_class=16,
        model_cfg=dict(
            IGNORED_LABEL=0,
            HUB2FUSE_CFG=dict(
                IMG_DIMS=[64, 128, 256, 512],
                LIDAR_DIMS=[64, 128, 256, 512],
                FUSED_DIMS=[64, 128, 256, 512],
                NUM_HEADS=[1, 2, 5, 8],
                SQH_CFG=dict(
                    NUM_BLOCKS=2,
                    MLP_RATIO=4.0,
                    QKV_BIAS=True,
                    QK_SCALE=None,
                    DROP_RATE=0.1,
                    ATTN_DROP_RATE=0.1,
                    SR_RATIO=1,
                ),
            ),
            SF_PHASE_CFG=dict(
                d_model=512,
                nhead=8,
                num_encoder_layers=6,
                dim_feedforward=1024,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=False,
            ),
            PGCN_CFG=dict(
                num_neighbor=16,
                feat_channels=512,
                num_layers=2,
                dropout=0.1,
                activation="relu",
                normalize_before=False,
            ),
        ),
    ),
    train_cfg=None,
    test_cfg=dict(mode="whole")
)

# ==== Dataset config =====
dataset_type = "SemanticNuscDataset"
data_root = "data/nuScenes"

train_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=4),
    dict(type="LoadAnnotations3D", with_bbox_3d=False, with_label_3d=False, with_seg_3d=True),
    dict(type="LoadMultiViewImageFromFiles"),
    dict(type="PointSegClassMapping"),
    dict(type="DefaultFormatBundle3D", class_names=None),
    dict(type="Collect3D", keys=["points", "points_label", "img"], meta_keys=["file_name", "scene_token", "sample_token", "lidar2img"]),
]

test_pipeline = [
    dict(type="LoadPointsFromFile", coord_type="LIDAR", load_dim=5, use_dim=4),
    dict(type="LoadMultiViewImageFromFiles"),
    dict(type="DefaultFormatBundle3D", class_names=None, with_label=False),
    dict(type="Collect3D", keys=["points", "img"], meta_keys=["file_name", "scene_token", "sample_token", "lidar2img"]),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/nuscenes_infos_train.pkl",
        pipeline=train_pipeline,
        classes=None,
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=False,
        filter_empty_gt=True,
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        classes=None,
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=True,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + "/nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        classes=None,
        modality=dict(use_lidar=True, use_camera=True),
        test_mode=True,
    ),
)

# ==== Training config =====
optimizer = dict(
    type="AdamW",
    lr=0.01,
    weight_decay=0.01,
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy="CosineAnnealing",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
momentum_config = None

# ==== Runtime config =====
total_epochs = 12
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
    ],
)
evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
)
checkpoint_config = dict(interval=1)