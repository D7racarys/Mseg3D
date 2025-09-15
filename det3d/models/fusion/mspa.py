import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule


class DWConv(nn.Module):
    """Depth-wise convolution module.
    
    Args:
        dim (int): Input channels.
        kernel_size (int): Kernel size for depth-wise convolution.
        stride (int): Stride for depth-wise convolution.
        padding (int): Padding for depth-wise convolution.
    """
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class Mlp(nn.Module):
    """MLP module with depth-wise convolution.
    
    Args:
        in_features (int): Input feature dimension.
        hidden_features (int): Hidden feature dimension.
        out_features (int): Output feature dimension.
        act_layer (nn.Module): Activation layer.
        drop (float): Dropout rate.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MSPoolAttention(nn.Module):
    """Multi-scale pooling attention module.
    
    Args:
        dim (int): Input feature dimension.
        num_heads (int): Number of attention heads.
        sr_ratio (int): Spatial reduction ratio.
        qkv_bias (bool): Whether to use bias in qkv projection.
        qk_scale (float): Scale factor for qk attention.
        attn_drop (float): Dropout rate for attention.
        proj_drop (float): Dropout rate for projection.
    """
    def __init__(self, dim, num_heads=8, sr_ratio=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(MSPoolAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=qkv_bias)
        
        # Multi-scale pooling
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Generate Q
        q = self.q(x)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)  # B, num_heads, HW, C//num_heads
        
        # Multi-scale pooling for K, V
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x)
        x_pool3 = self.pool3(x)
        
        # Generate K, V from multi-scale features
        kv1 = self.kv(x)
        kv2 = self.kv(x_pool1)
        kv3 = self.kv(x_pool2)
        kv4 = self.kv(x_pool3)
        
        # Combine multi-scale K, V
        kv = (kv1 + kv2 + kv3 + kv4) / 4.0
        kv = kv.reshape(B, 2, self.num_heads, C // self.num_heads, H * W).permute(1, 0, 2, 4, 3)
        k, v = kv[0], kv[1]  # B, num_heads, HW, C//num_heads
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MSPABlock(nn.Module):
    """Multi-scale parallel attention block.
    
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
        super(MSPABlock, self).__init__()
        
        # First normalization and attention
        self.norm1 = norm_layer(dim)
        self.attn = MSPoolAttention(
            dim, num_heads=num_heads, sr_ratio=sr_ratio, qkv_bias=qkv_bias, 
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # Second normalization and MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # Attention branch
        x_attn = self.norm1(x)
        x_attn = self.attn(x_attn)
        x = x + x_attn
        
        # MLP branch
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)
        x = x + x_mlp
        
        return x