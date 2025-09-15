import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class SelfQueryHub(nn.Module):
    """
    Self Query Hub module for feature enhancement in Hub2fuse mechanism.
    This module performs self-attention on features and enhances them through
    multi-head self-attention mechanism.
    
    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Whether to use bias in qkv projection.
        qk_scale (float): Scale factor for qk attention.
        attn_drop (float): Dropout rate for attention.
        proj_drop (float): Dropout rate for projection.
        sr_ratio (int): Spatial reduction ratio.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(SelfQueryHub, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Self-attention projections
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        
        # Spatial reduction for efficiency
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q
        q = self.q(x)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)  # B, num_heads, HW, C//num_heads
        
        # Generate K, V with optional spatial reduction
        if self.sr_ratio > 1:
            x_kv = self.sr(x)
            _, _, H_kv, W_kv = x_kv.shape
            k = self.k(x_kv).reshape(B, self.num_heads, C // self.num_heads, H_kv * W_kv).permute(0, 1, 3, 2)
            v = self.v(x_kv).reshape(B, self.num_heads, C // self.num_heads, H_kv * W_kv).permute(0, 1, 2, 3)
        else:
            k = self.k(x).reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)
            v = self.v(x).reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 2, 3)
        
        # Self-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SelfQueryHubBlock(nn.Module):
    """
    Self Query Hub Block that combines self-attention with MLP for feature enhancement.
    
    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP expansion ratio.
        qkv_bias (bool): Whether to use bias in qkv projection.
        qk_scale (float): Scale factor for qk attention.
        drop (float): Dropout rate.
        attn_drop (float): Dropout rate for attention.
        drop_path (float): Drop path rate.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        sr_ratio (int): Spatial reduction ratio.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, sr_ratio=1):
        super(SelfQueryHubBlock, self).__init__()
        
        # First normalization and self-attention
        self.norm1 = norm_layer(dim)
        self.attn = SelfQueryHub(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        
        # Second normalization and MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, kernel_size=1),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, kernel_size=1),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Self-attention branch
        x_attn = self.norm1(x)
        x_attn = self.attn(x_attn)
        x = x + x_attn
        
        # MLP branch
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x = x + x_mlp
        
        return x


class SelfQueryHubModule(nn.Module):
    """
    Self Query Hub Module for feature enhancement in Hub2fuse mechanism.
    This module enhances features through self-attention and provides
    multi-scale feature outputs.
    
    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension.
        num_heads (int): Number of attention heads.
        num_blocks (int): Number of SelfQueryHubBlock blocks.
        mlp_ratio (float): MLP expansion ratio.
        qkv_bias (bool): Whether to use bias in qkv projection.
        qk_scale (float): Scale factor for qk attention.
        drop (float): Dropout rate.
        attn_drop (float): Dropout rate for attention.
        drop_path (float): Drop path rate.
        norm_layer (nn.Module): Normalization layer.
        sr_ratio (int): Spatial reduction ratio.
    """
    def __init__(self, in_dim, out_dim, num_heads=8, num_blocks=2, mlp_ratio=4., qkv_bias=True, 
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0., norm_layer=nn.BatchNorm2d, sr_ratio=1):
        super(SelfQueryHubModule, self).__init__()
        
        # Input projection if dimensions don't match
        self.in_proj = None
        if in_dim != out_dim:
            self.in_proj = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                norm_layer(out_dim),
                nn.ReLU(inplace=True)
            )
        
        # Self Query Hub blocks
        self.blocks = nn.ModuleList([
            SelfQueryHubBlock(
                dim=out_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path, norm_layer=norm_layer, sr_ratio=sr_ratio
            ) for _ in range(num_blocks)
        ])
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=False),
            norm_layer(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input projection if needed
        if self.in_proj is not None:
            x = self.in_proj(x)
        
        # Apply Self Query Hub blocks
        for block in self.blocks:
            x = block(x)
        
        # Output projection
        x = self.out_proj(x)
        
        return x