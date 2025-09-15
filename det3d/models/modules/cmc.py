import torch
import torch.nn as nn
import torch.nn.functional as F

class CMC(nn.Module):
    def __init__(self, img_H=256, img_W=256, feat_dim=512, lidar_dim=64, hub_dim=512):
        super().__init__()
        self.img_H = img_H
        self.img_W = img_W
        self.feat_dim = feat_dim
        
        # FOV 外点的映射器（MLP）
        self.lidar_mapper = nn.Sequential(
            nn.Linear(lidar_dim, hub_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hub_dim, feat_dim)
        )

    def forward(self, img_feat, pts_3d, lidar_feat, cam_intrinsic, cam_extrinsic):
        """
        img_feat: [B, C=512, H, W]  图像特征图
        pts_3d: [B, N, 3]  LiDAR点 (x,y,z)
        lidar_feat: [B, N, lidar_dim]  LiDAR点的特征
        cam_intrinsic: [B, 3, 3]
        cam_extrinsic: [B, 4, 4]
        """
        B, C, H, W = img_feat.shape
        N = pts_3d.shape[1]

        # ---- Step 1: 点投影到图像平面 ----
        ones = torch.ones((B, N, 1), device=pts_3d.device)
        pts_h = torch.cat([pts_3d, ones], dim=-1)  # [B, N, 4]
        cam_pts = torch.bmm(pts_h, cam_extrinsic.transpose(1, 2))  # [B, N, 4]
        cam_pts = cam_pts[..., :3]

        # 深度 > 0 的点才在前方
        in_front = cam_pts[..., 2] > 0

        # 像素坐标
        proj = torch.bmm(cam_pts, cam_intrinsic.transpose(1, 2))  # [B, N, 3]
        u = proj[..., 0] / proj[..., 2]
        v = proj[..., 1] / proj[..., 2]

        # 归一化到 [-1, 1] 区间（用于 grid_sample）
        u_norm = (u / (W - 1)) * 2 - 1
        v_norm = (v / (H - 1)) * 2 - 1
        grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(1)  # [B,1,N,2]

        # ---- Step 2: 双线性插值获取 FOV 内点特征 ----
        img_feat_exp = img_feat.unsqueeze(2)  # [B,C,1,H,W]
        sampled = F.grid_sample(img_feat_exp.view(B, C, H, W), grid, align_corners=True)
        sampled = sampled.squeeze(2).permute(0, 2, 1)  # [B, N, C]

        # ---- Step 3: 区分 FOV 内外 ----
        inside_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H) & in_front  # [B, N]

        # FOV 外点通过 MLP 映射
        mapped = self.lidar_mapper(lidar_feat)  # [B, N, C]

        # 最终融合特征
        final_feat = torch.where(inside_mask.unsqueeze(-1), sampled, mapped)

        return final_feat  # [B, N, 512]
