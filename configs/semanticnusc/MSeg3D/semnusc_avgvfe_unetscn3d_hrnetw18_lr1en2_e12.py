import itertools
import logging
from typing import Sequence

from addict.addict import Dict
# from hrnet_cfg import hrnet_w48
from hrnet_cfg import hrnet_w18
from fcn_cfg import fcn_head

# 使用的是HRNet-w48作为backbone图像和UNetSCN3D作为backbone点云结合的方法
num_class=17
ignore_class=0



use_img = True
# NOTE: keep the order
cam_chan = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
cam_names = ['1', '2', '3', '4', '5', '6'] 

# 图像均值和方差，用于标准化
nusc_mean = [0.40789654, 0.44719302, 0.47026115] # BGR
nusc_std = [0.28863828, 0.27408164, 0.27809835] # BGR
cam_attributes = {
    '1': dict(mean=nusc_mean, std=nusc_std),
    '2': dict(mean=nusc_mean, std=nusc_std),
    '3': dict(mean=nusc_mean, std=nusc_std),
    '4': dict(mean=nusc_mean, std=nusc_std),
    '5': dict(mean=nusc_mean, std=nusc_std),
    '6': dict(mean=nusc_mean, std=nusc_std),
}



hrnet_w18_cfg = dict(
    pretrained='./work_dirs/pretrained_models/hrnetv2_w18-00eb2006.pth',
    frozen_stages=3, # memory saving by the frozen 3/4 stages
    norm_eval=False, 
)
hrnet_w18.update(hrnet_w18_cfg)



fcn_head_cfg = dict(
    type="FCNMSeg3DHead",           # 分割头类型，FCN-based 3D Seg Head
    num_classes=num_class,         # 输出类别数（不含 ignore 类）
    ignore_index=ignore_class,     # 忽略的类别索引，通常是背景或无标签
    in_index=(0, 1, 2, 3),         # 从 HRNet 哪几个 stage 获取特征（4个分支）
    in_channels=[18, 36, 72, 144], # 每个分支的通道数，对应 HRNet-w18 输出
    num_convs=2,                   # 每个分支的卷积层数
    channels=48,                   # 最后融合时通道数（压缩用）
    loss_weight=0.5,               # 损失函数的权重（用于和主头融合损失）
)
fcn_head.update(fcn_head_cfg)





point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size=[0.1, 0.1, 0.2]


# model settings
model = dict(
    # type="SegNet",
    type="SegMSeg3DNet",

    pretrained=None,


    # img branch
    img_backbone = hrnet_w18 if use_img else None,
    # 负责将 HRNet 提取的多尺度特征进一步融合和分类，输出图像分割结果或特征。
    img_head = fcn_head if use_img else None,    
    

    # point cloud branch
    reader=dict(
        type="ImprovedMeanVoxelFeatureExtractor",
        num_input_features=5, 
    ),
    backbone=dict(
        # type="UNetV6", 
        type="UNetSCN3D", 
        num_input_features=5+8,
        ds_factor=8, # 下采样因子
        us_factor=8, # 上采样因子
        point_cloud_range=point_cloud_range, 
        voxel_size=voxel_size, 
        model_cfg=dict(
            SCALING_RATIO=2, # 通道缩放比
        ),
    ),

    # 接收点云主干（如 UNetSCN3D）和图像分支融合后的特征
    # 对每个体素或点进行类别预测，实现点云语义分割
    point_head=dict(
        type="PointSegMSeg3DHead",

        class_agnostic=False, # 是否使用类别无关的分割头
        num_class=num_class,
        model_cfg=dict(
            VOXEL_IN_DIM=32,  # 输入点云体素特征维度
            VOXEL_CLS_FC=[64], # 点云体素分类全连接层配置
            VOXEL_ALIGN_DIM=64, # 点云体素对齐维度
            IMAGE_IN_DIM=48,    # 输入图像特征维度
            IMAGE_ALIGN_DIM=64, # 图像特征对齐维度

            GEO_FUSED_DIM=64, # 几何融合特征维度
            OUT_CLS_FC=[64, 64], # 输出分类全连接层配置
            IGNORED_LABEL=ignore_class, # 忽略的标签类别
            DP_RATIO=0.25, # dropout 比例

            MIMIC_FC=[64, 64], # 模仿学习全连接层配置

            # 含有Transformer的相关参数，可调整
            SFPhase_CFG=dict(
                embeddings_proj_kernel_size=1,  # 投影卷积核大小
                d_model=96, 
                n_head=4, # 注意力头数
                n_layer=6, # decreasing this can save memory
                n_ffn=192,
                drop_ratio=0,
                activation="relu",
                pre_norm=False,
            ),
        )
    )
)

train_cfg = dict()
test_cfg = dict()


# dataset settings
dataset_type = "SemanticNuscDataset"
data_root =  "/data/luochao/Paper/UniPAD/data/nuscenes"
nsweeps = 1 # 将多少帧点云（sweeps）进行叠加融合来作为当前样本的输入


train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    npoints=100000,
    global_rot_noise=[-0.78539816, 0.78539816],     
    global_scale_noise=[0.95, 1.05],    
    global_translate_std=0.5,
)
val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)
test_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)


train_image_preprocessor = dict(
    shuffle_points=train_preprocessor["shuffle_points"],
    random_horizon_flip=True,

    # 随机变化图像的亮度、对比度、饱和度和色调，参数表示上下浮动的百分比范围
    random_color_jitter_cfg=dict(
        brightness=0.3, 
        contrast=0.3, 
        saturation=0.3, 
        hue=0.1
    ),

    random_jpeg_compression_cfg=dict(
        quality_noise=[30, 70],
        probability=0.5,
    ),

    random_rescale_cfg=dict(
        ratio_range=(1.0, 1.5), 
    ),
    
    random_crop_cfg=dict( 
        crop_size=(640, 960), # NOTE: (H, W)
        cat_max_ratio=0.75,
        kept_min_ratio=0.60, 
        ignore_index=ignore_class, 
        unvalid_cam_id=0, 
        try_times=3,
    ),

)
val_image_preprocessor = dict(
    shuffle_points=val_preprocessor["shuffle_points"],
    random_horizon_flip=False,
)
test_image_preprocessor = dict(
    shuffle_points=test_preprocessor["shuffle_points"],
    random_horizon_flip=False,
)


voxel_generator = dict(
    range=point_cloud_range,
    voxel_size=voxel_size,
    max_points_in_voxel=5,
    max_voxel_num=[300000, 300000],
)


# 做一些前期准备，不涉及到网络结构
# 包括加载数据、标签、预处理、数据增强和点云的体素化和体素标签分配
# 最后整理为训练所需的统一输入格式
train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, use_img=use_img),
    dict(type="LoadImageFromFile", use_img=use_img),
    dict(type="LoadPointCloudAnnotations", with_bbox=False),
    dict(type="LoadImageAnnotations", points_cp_radius=2),
    # 图像和点云的预处理和增强
    dict(type="SegPreprocess", cfg=train_preprocessor, use_img=use_img),
    dict(type="SegImagePreprocess", cfg=train_image_preprocessor),
    dict(type="SegVoxelization", cfg=voxel_generator),
    # 为体素指派语义标签
    dict(type="SegAssignLabel", cfg=dict(voxel_label_enc="compact_value")),
    dict(type="Reformat"),
]
val_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, use_img=use_img),
    dict(type="LoadImageFromFile", use_img=use_img),
    dict(type="SegPreprocess", cfg=val_preprocessor, use_img=use_img),
    dict(type="SegImagePreprocess", cfg=val_image_preprocessor),
    dict(type="SegVoxelization", cfg=voxel_generator),
    dict(type="Reformat"),
]
test_pipeline = []


train_anno = "/data/luochao/Paper/UniPAD/data/nuscenes/infos_train_10sweeps_segdet_withvelo_filter_True.pkl"
val_anno = "/data/luochao/Paper/UniPAD/data/nuscenes/infos_val_10sweeps_segdet_withvelo_filter_True.pkl"
test_anno = "/data/luochao/Paper/UniPAD/data/nuscenes/infos_test_10sweeps_segdet_withvelo_filter_True.pkl"


data = dict(
    samples_per_gpu=3, 
    workers_per_gpu=8, 
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        cam_names=cam_names,
        cam_chan=cam_chan,
        cam_attributes=cam_attributes,
        img_resized_shape=(960, 640), # (width, height) in opencv format
        nsweeps=nsweeps,
        load_interval=1, 
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        cam_names=cam_names,
        cam_chan=cam_chan,
        cam_attributes=cam_attributes,
        img_resized_shape=(960, 640), 
        nsweeps=nsweeps,
        load_interval=1,
        pipeline=val_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        test_mode=True,
        ann_file=test_anno,
        cam_names=cam_names,
        cam_chan=cam_chan,
        cam_attributes=cam_attributes,
        img_resized_shape=(960, 640), 
        nsweeps=nsweeps,
        pipeline=test_pipeline,
    ),
)



# optimizer
# 这个设置表示梯度裁剪（Gradient Clipping），用于防止梯度爆炸，提高训练稳定性
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.01, moms=[0.95, 0.85], 
    div_factor=10.0,  # 初始学习率是 lr_max / div_factor
    pct_start=0.4, # 前 40% 的训练过程用于升高学习率，后 60% 用于降低学习率
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)

# yapf:enable
# runtime settings
total_epochs = 12

device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None 
workflow = [('train', 1)]

sync_bn_type = "torch"