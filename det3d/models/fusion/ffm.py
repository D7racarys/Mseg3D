import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class ChannelWeights(nn.Module):
    """Channel weights module for feature rectification.
    
    This module calculates channel-wise weights for feature rectification.
    
    Args:
        in_channels (int): Input channels.
        ratio (int): Reduction ratio for channel attention.
    """
    def __init__(self, in_channels, ratio=16):
        super(ChannelWeights, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y


class SpatialWeights(nn.Module):
    """Spatial weights module for feature rectification.
    
    This module calculates spatial-wise weights for feature rectification.
    
    Args:
        kernel_size (int): Kernel size for spatial attention.
    """
    def __init__(self, kernel_size=7):
        super(SpatialWeights, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class FeatureRectifyModule(nn.Module):
    """Feature Rectify Module (FRM).
    
    This module rectifies features from one modality using information from another modality.
    
    Args:
        in_channels (int): Input channels.
        ratio (int): Reduction ratio for channel attention.
        kernel_size (int): Kernel size for spatial attention.
    """
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super(FeatureRectifyModule, self).__init__()
        self.channel_weights = ChannelWeights(in_channels, ratio)
        self.spatial_weights = SpatialWeights(kernel_size)

    def forward(self, x, y):
        """
        Args:
            x: Feature to be rectified
            y: Reference feature for rectification
        """
        channel_weights = self.channel_weights(y)
        spatial_weights = self.spatial_weights(y)
        
        # Apply channel and spatial attention
        out = x * channel_weights * spatial_weights
        return out


class CrossAttention(nn.Module):
    """Cross-modal attention module.
    
    This module implements cross-modal attention between two modalities.
    
    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use bias in qkv projection.
        qk_scale (float): Scale factor for qk attention.
        attn_drop (float): Dropout rate for attention.
        proj_drop (float): Dropout rate for projection.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

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
    """Cross-modal interaction path.
    
    This module implements the cross-modal interaction path between two modalities.
    
    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        qkv_bias (bool): Whether to use bias in qkv projection.
        qk_scale (float): Scale factor for qk attention.
        drop (float): Dropout rate.
        attn_drop (float): Dropout rate for attention.
        drop_path (float): Drop path rate.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(CrossPath, self).__init__()
        self.norm1_x = norm_layer(dim)
        self.norm1_y = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, y):
        x_out = x + self.cross_attn(self.norm1_x(x), self.norm1_y(y))
        x_out = x_out + self.mlp(self.norm2(x_out))
        return x_out


class ChannelEmbed(nn.Module):
    """Channel embedding module.
    
    This module embeds features to a specified dimension.
    
    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        kernel_size (int): Kernel size for convolution.
        stride (int): Stride for convolution.
        padding (int): Padding for convolution.
        norm_layer (nn.Module): Normalization layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module (FFM).
    
    This module fuses features from two modalities using cross-modal attention.
    
    Args:
        img_dim (int): Image feature dimension.
        lidar_dim (int): LiDAR feature dimension.
        fused_dim (int): Fused feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        qkv_bias (bool): Whether to use bias in qkv projection.
        qk_scale (float): Scale factor for qk attention.
        drop (float): Dropout rate.
        attn_drop (float): Dropout rate for attention.
        drop_path (float): Drop path rate.
        norm_layer (nn.Module): Normalization layer.
        use_frm (bool): Whether to use Feature Rectify Module.
    """
    def __init__(self, img_dim, lidar_dim, fused_dim, num_heads=8, mlp_ratio=4., qkv_bias=False, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, use_frm=True):
        super(FeatureFusionModule, self).__init__()
        
        # Feature embedding
        self.img_embed = ChannelEmbed(img_dim, fused_dim)
        self.lidar_embed = ChannelEmbed(lidar_dim, fused_dim)
        
        # Feature rectification
        self.use_frm = use_frm
        if use_frm:
            self.img_frm = FeatureRectifyModule(fused_dim)
            self.lidar_frm = FeatureRectifyModule(fused_dim)
        
        # Cross-modal interaction
        self.img_to_lidar = CrossPath(
            fused_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer=norm_layer)
        self.lidar_to_img = CrossPath(
            fused_dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, norm_layer=norm_layer)
        
        # Output projection
        self.img_proj = nn.Conv2d(fused_dim, fused_dim, kernel_size=1)
        self.lidar_proj = nn.Conv2d(fused_dim, fused_dim, kernel_size=1)
        self.fusion_proj = nn.Conv2d(fused_dim * 2, fused_dim, kernel_size=1)

    def forward(self, img_feat, lidar_feat):
        # Feature embedding
        img_feat = self.img_embed(img_feat)
        lidar_feat = self.lidar_embed(lidar_feat)
        
        # Feature rectification
        if self.use_frm:
            img_feat_rect = self.img_frm(img_feat, lidar_feat)
            lidar_feat_rect = self.lidar_frm(lidar_feat, img_feat)
        else:
            img_feat_rect = img_feat
            lidar_feat_rect = lidar_feat
        
        # Reshape for cross attention
        B, C, H, W = img_feat_rect.shape
        img_feat_flat = img_feat_rect.flatten(2).transpose(1, 2)  # B, HW, C
        lidar_feat_flat = lidar_feat_rect.flatten(2).transpose(1, 2)  # B, HW, C
        
        # Cross-modal interaction
        img_feat_enhanced = self.lidar_to_img(img_feat_flat, lidar_feat_flat)
        lidar_feat_enhanced = self.img_to_lidar(lidar_feat_flat, img_feat_flat)
        
        # Reshape back
        img_feat_enhanced = img_feat_enhanced.transpose(1, 2).reshape(B, C, H, W)
        lidar_feat_enhanced = lidar_feat_enhanced.transpose(1, 2).reshape(B, C, H, W)
        
        # Output projection
        img_out = self.img_proj(img_feat_enhanced)
        lidar_out = self.lidar_proj(lidar_feat_enhanced)
        
        # Feature fusion
        fused_feat = torch.cat([img_out, lidar_out], dim=1)
        fused_feat = self.fusion_proj(fused_feat)
        
        return fused_feat, img_out, lidar_out