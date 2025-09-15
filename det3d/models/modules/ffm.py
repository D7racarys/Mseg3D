import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class ChannelWeights(nn.Module):
    """计算通道权重的模块"""
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # 平均池化
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_out).view(b, c, 1, 1)
        # 最大池化
        max_out = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_out).view(b, c, 1, 1)
        # 合并并应用sigmoid
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialWeights(nn.Module):
    """计算空间权重的模块"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化和最大池化（沿通道维度）
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接并应用卷积
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out


class FeatureRectifyModule(nn.Module):
    """特征校正模块（FRM）"""
    def __init__(self, dim, reduction=1):
        super().__init__()
        self.channel_weights_rgb = ChannelWeights(dim, reduction)
        self.channel_weights_lidar = ChannelWeights(dim, reduction)
        self.spatial_weights_rgb = SpatialWeights()
        self.spatial_weights_lidar = SpatialWeights()

    def forward(self, rgb_feat, lidar_feat):
        # 计算通道和空间权重
        rgb_channel_weights = self.channel_weights_rgb(rgb_feat)
        lidar_channel_weights = self.channel_weights_lidar(lidar_feat)
        rgb_spatial_weights = self.spatial_weights_rgb(rgb_feat)
        lidar_spatial_weights = self.spatial_weights_lidar(lidar_feat)
        
        # 应用权重进行特征校正
        rgb_feat_rectified = rgb_feat * rgb_channel_weights * lidar_spatial_weights
        lidar_feat_rectified = lidar_feat * lidar_channel_weights * rgb_spatial_weights
        
        return rgb_feat_rectified, lidar_feat_rectified


class CrossAttention(nn.Module):
    """跨模态注意力机制"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(y).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossPath(nn.Module):
    """跨模态特征交互路径"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_rgb = norm_layer(dim)
        self.norm1_lidar = norm_layer(dim)
        self.attn_rgb2lidar = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.attn_lidar2rgb = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        
    def forward(self, rgb_feat, lidar_feat):
        # 将特征从[B, C, H, W]转换为[B, HW, C]
        B, C, H, W = rgb_feat.shape
        rgb_feat_flat = rgb_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        lidar_feat_flat = lidar_feat.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # 跨模态注意力
        rgb2lidar = self.attn_rgb2lidar(self.norm1_lidar(lidar_feat_flat), self.norm1_rgb(rgb_feat_flat))
        lidar2rgb = self.attn_lidar2rgb(self.norm1_rgb(rgb_feat_flat), self.norm1_lidar(lidar_feat_flat))
        
        # 残差连接
        rgb_feat_flat = rgb_feat_flat + self.drop_path(lidar2rgb)
        lidar_feat_flat = lidar_feat_flat + self.drop_path(rgb2lidar)
        
        # 将特征从[B, HW, C]转换回[B, C, H, W]
        rgb_feat_out = rgb_feat_flat.transpose(1, 2).reshape(B, C, H, W)
        lidar_feat_out = lidar_feat_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return rgb_feat_out, lidar_feat_out


class ChannelEmbed(nn.Module):
    """通道嵌入模块"""
    def __init__(self, dim, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(dim*2, dim, kernel_size=3, padding=1, bias=False),
            norm_layer(dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, rgb_feat, lidar_feat):
        # 拼接特征并进行投影
        feat = torch.cat([rgb_feat, lidar_feat], dim=1)
        return self.proj(feat)


class FeatureFusionModule(nn.Module):
    """特征融合模块（FFM）"""
    def __init__(self, dim, reduction=1, num_heads=8, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross_path = CrossPath(dim, num_heads=num_heads)
        self.channel_embed = ChannelEmbed(dim, norm_layer=norm_layer)
        
    def forward(self, rgb_feat, lidar_feat):
        # 跨模态特征交互
        rgb_feat_cross, lidar_feat_cross = self.cross_path(rgb_feat, lidar_feat)
        
        # 通道嵌入融合
        fused_feat = self.channel_embed(rgb_feat_cross, lidar_feat_cross)
        
        return fused_feat