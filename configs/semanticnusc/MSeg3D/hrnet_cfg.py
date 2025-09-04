import itertools
import logging
from typing import Sequence

from addict.addict import Dict


# model settings
# norm_cfg = dict(type='SyncBN', requires_grad=True) # mmcv style
norm_cfg = dict(type='BN', requires_grad=True) # det3d style

# hrnet_w18和hrnet_w48的区别
# hrnet_w48 (宽模型)：
# 优点： 通常具有更强的特征表示能力，可能在更复杂的任务或需要更高精度的任务上表现更好。
# 缺点： 参数量更大，计算成本更高，内存占用更多，训练时间更长。需要更多的计算资源（GPU 内存和计算能力）。
# hrnet_w18则相反

hrnet_w18=dict(
    type='HRNet',
    pretrained='./work_dirs/pretrained_models/hrnetv2_w18-00eb2006.pth',
    norm_cfg=norm_cfg,
    norm_eval=False,
    extra=dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            num_channels=(18, 36)),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            num_channels=(18, 36, 72)),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            num_channels=(18, 36, 72, 144)),
        )
    )


hrnet_w48=dict(
    type='HRNet',
    # pretrained='open-mmlab://msra/hrnetv2_w48', # download from internet
    pretrained='./work_dirs/pretrained_models/hrnetv2_w48-d2186c55.pth', # file system

    norm_cfg=norm_cfg,
    norm_eval=False,
    extra=dict(
        stage1=dict(
            num_modules=1,
            num_branches=1,
            block='BOTTLENECK',
            num_blocks=(4, ),
            num_channels=(64, )
            ),
        stage2=dict(
            num_modules=1,
            num_branches=2,
            block='BASIC',
            num_blocks=(4, 4),
            # num_channels=(18, 36),    # w18
            num_channels=(48, 96),      # w48
            ),
        stage3=dict(
            num_modules=4,
            num_branches=3,
            block='BASIC',
            num_blocks=(4, 4, 4),
            # num_channels=(18, 36, 72),    # w18
            num_channels=(48, 96, 192),     # w48
            ),
        stage4=dict(
            num_modules=3,
            num_branches=4,
            block='BASIC',
            num_blocks=(4, 4, 4, 4),
            # num_channels=(18, 36, 72, 144)    # w18
            num_channels=(48, 96, 192, 384)     # w48
            ),
        )
    )