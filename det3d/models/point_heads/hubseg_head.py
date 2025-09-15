import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from ..registry import POINT_HEADS
from det3d.core.utils.loss_utils import lovasz_softmax
from ..fusion.ffm import FeatureRectifyModule as FRM
from ..fusion.ffm import FeatureFusionModule as FFM
from ..fusion.mspa import MSPABlock
from ..fusion.sqh import SelfQueryHubModule as SQH


@POINT_HEADS.register_module
class HubSegHead(nn.Module):
    """
    HubSeg: 融合CMNeXt的Hub2fuse机制的点云分割头部
    """
    def __init__(self, class_agnostic, num_class, model_cfg, **kwargs):
        super().__init__()
        
        if class_agnostic:
            self.num_class = 1
        else:
            self.num_class = num_class
        
        self.model_cfg = model_cfg
        norm_layer = partial(nn.BatchNorm2d, eps=1e-6)
        act_layer = nn.ReLU
        
        # 获取配置参数
        self.hub2fuse_cfg = model_cfg.get("HUB2FUSE_CFG", {})
        self.sf_phase_cfg = model_cfg.get("SF_PHASE_CFG", {})
        self.pgcn_cfg = model_cfg.get("PGCN_CFG", {})
        
        # 特征维度
        self.img_dims = self.hub2fuse_cfg.get("IMG_DIMS", [64, 128, 256, 512])
        self.lidar_dims = self.hub2fuse_cfg.get("LIDAR_DIMS", [64, 128, 256, 512])
        self.fused_dims = self.hub2fuse_cfg.get("FUSED_DIMS", [64, 128, 256, 512])
        
        # 注意力头数
        self.num_heads = self.hub2fuse_cfg.get("NUM_HEADS", [1, 2, 5, 8])
        
        # 创建特征校正模块（FRM）
        self.frms = nn.ModuleList([
            FRM(dim=self.img_dims[i], reduction=1) 
            for i in range(len(self.img_dims))
        ])
        
        # 创建特征融合模块（FFM）
        self.ffms = nn.ModuleList([
            FFM(dim=self.img_dims[i], reduction=1, num_heads=self.num_heads[i], norm_layer=nn.BatchNorm2d)
            for i in range(len(self.img_dims))
        ])
        
        # 创建多尺度并行注意力块（MSPA）用于LiDAR特征增强
        self.mspa_blocks = nn.ModuleList([
            MSPABlock(dim=self.lidar_dims[i], mlp_ratio=4, num_heads=self.num_heads[i], norm_cfg=dict(type='BN', requires_grad=True))
            for i in range(len(self.lidar_dims))
        ])
        
        # 获取Self Query Hub配置
        self.sqh_cfg = self.hub2fuse_cfg.get("SQH_CFG", {})
        num_blocks = self.sqh_cfg.get("NUM_BLOCKS", 2)
        mlp_ratio = self.sqh_cfg.get("MLP_RATIO", 4.0)
        qkv_bias = self.sqh_cfg.get("QKV_BIAS", True)
        qk_scale = self.sqh_cfg.get("QK_SCALE", None)
        drop_rate = self.sqh_cfg.get("DROP_RATE", 0.1)
        attn_drop_rate = self.sqh_cfg.get("ATTN_DROP_RATE", 0.1)
        sr_ratio = self.sqh_cfg.get("SR_RATIO", 1)
        
        # 创建Self Query Hub模块用于特征增强
        self.sqh_modules = nn.ModuleList([
            SQH(
                in_dim=self.lidar_dims[i], 
                out_dim=self.lidar_dims[i], 
                num_heads=self.num_heads[i], 
                num_blocks=num_blocks,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                sr_ratio=sr_ratio
            )
            for i in range(len(self.lidar_dims))
        ])
        
        # 融合后的特征处理
        self.fused_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.fused_dims[i], self.fused_dims[i], kernel_size=3, padding=1, bias=False),
                norm_layer(self.fused_dims[i]),
                act_layer(inplace=True)
            ) for i in range(len(self.fused_dims))
        ])
        
        # 最终分类头
        final_dim = sum(self.fused_dims)
        self.classifier = nn.Sequential(
            nn.Conv2d(final_dim, 256, kernel_size=3, padding=1, bias=False),
            norm_layer(256),
            act_layer(inplace=True),
            nn.Conv2d(256, self.num_class, kernel_size=1)
        )
        
        # 损失函数
        self.ignored_label = model_cfg.get("IGNORED_LABEL", 0)
        self.cross_entropy_func = nn.CrossEntropyLoss(ignore_index=self.ignored_label)
        self.lovasz_softmax_func = lovasz_softmax
        
        self.forward_ret_dict = {}
        self.tasks = ["out"]
    
    def _reshape_features(self, features, batch_size):
        """
        重塑特征以适应Hub2fuse处理
        """
        # 假设features的形状为[B*num_cams, C, H, W]
        _, C, H, W = features.shape
        num_cams = features.shape[0] // batch_size
        
        # 重塑为[B, num_cams, C, H, W]
        features = features.view(batch_size, num_cams, C, H, W)
        
        # 取平均或最大值以获得单一视图表示
        features = features.mean(dim=1)  # [B, C, H, W]
        
        return features
    
    def forward(self, batch_dict, return_loss=True):
        """
        前向传播函数
        """
        # 获取图像和点云特征
        image_features = batch_dict["image_features"]  # [B, num_cams, C, H, W]
        batch_size = batch_dict["batch_size"]
        
        # 获取LiDAR特征（假设有多个尺度）
        lidar_features = []
        for i in range(len(self.lidar_dims)):
            key = f"lidar_features_stage{i+1}"
            if key in batch_dict:
                lidar_features.append(batch_dict[key])
            else:
                # 如果没有多尺度特征，使用最后一个特征
                if i == 0:
                    lidar_features.append(batch_dict["voxel_features"])
                else:
                    # 进行下采样以创建多尺度特征
                    prev_feat = lidar_features[-1]
                    down_feat = F.avg_pool2d(prev_feat, kernel_size=2, stride=2)
                    lidar_features.append(down_feat)
        
        # 获取图像特征（假设有多个尺度）
        img_features = []
        for i in range(len(self.img_dims)):
            key = f"img_features_stage{i+1}"
            if key in batch_dict:
                img_features.append(batch_dict[key])
            else:
                # 如果没有多尺度特征，使用最后一个特征并进行处理
                if i == 0:
                    # 重塑图像特征
                    img_feat = self._reshape_features(image_features.view(-1, *image_features.shape[2:]), batch_size)
                    img_features.append(img_feat)
                else:
                    # 进行下采样以创建多尺度特征
                    prev_feat = img_features[-1]
                    down_feat = F.avg_pool2d(prev_feat, kernel_size=2, stride=2)
                    img_features.append(down_feat)
        
        # 应用MSPA块增强LiDAR特征
        enhanced_lidar_features = []
        for i, feat in enumerate(lidar_features):
            # 首先通过Self Query Hub模块增强特征
            sqh_feat = self.sqh_modules[i](feat)
            # 然后通过MSPA块进一步增强特征
            enhanced_feat = self.mspa_blocks[i](sqh_feat)
            enhanced_lidar_features.append(enhanced_feat)
        
        # 应用Hub2fuse进行特征融合
        fused_features = []
        for i in range(len(self.img_dims)):
            # 特征校正
            rectified_img, rectified_lidar = self.frms[i](img_features[i], enhanced_lidar_features[i])
            
            # 特征融合
            fused_feat = self.ffms[i](rectified_img, rectified_lidar)
            
            # 额外处理
            fused_feat = self.fused_conv[i](fused_feat)
            fused_features.append(fused_feat)
        
        # 上采样所有特征到相同分辨率
        target_size = fused_features[0].shape[2:]
        for i in range(1, len(fused_features)):
            fused_features[i] = F.interpolate(fused_features[i], size=target_size, mode='bilinear', align_corners=False)
        
        # 拼接多尺度特征
        multi_scale_features = torch.cat(fused_features, dim=1)
        
        # 最终分类
        logits = self.classifier(multi_scale_features)
        
        # 保存结果用于损失计算
        if return_loss:
            self.forward_ret_dict.update({
                "out_logits": logits,
                "point_sem_labels": batch_dict["point_sem_labels"],
            })
        
        # 更新batch_dict
        batch_dict.update({
            "seg_logits": logits,
        })
        
        return batch_dict
    
    def get_loss(self, point_loss_dict=None):
        """
        计算损失
        """
        point_loss = 0
        if point_loss_dict is None:
            point_loss_dict = {}
        
        # 交叉熵损失
        out_ce_loss = self.cross_entropy_func(
            self.forward_ret_dict["out_logits"],
            self.forward_ret_dict["point_sem_labels"].long(),
        )
        
        # Lovasz损失
        out_lvsz_loss = self.lovasz_softmax_func(
            F.softmax(self.forward_ret_dict["out_logits"], dim=1),
            self.forward_ret_dict["point_sem_labels"].long(),
            ignore=self.ignored_label,
        )
        
        # 总损失
        out_loss = out_ce_loss + out_lvsz_loss
        point_loss += out_loss
        
        # 记录损失
        point_loss_dict["out_ce_loss"] = out_ce_loss.detach()
        point_loss_dict["out_lvsz_loss"] = out_lvsz_loss.detach()
        
        return point_loss, point_loss_dict
    
    def predict(self, example, test_cfg):
        """
        预测函数
        """
        batch_dict = self.forward(example, return_loss=False)
        seg_logits = batch_dict["seg_logits"]
        
        # 获取预测类别
        seg_pred = seg_logits.argmax(dim=1)
        
        # 构建结果字典
        results = {
            "seg_pred": seg_pred,
            "seg_logits": seg_logits,
        }
        
        return results