import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class DWConv(nn.Module):
    """深度可分离卷积"""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    """MLP模块，包含两个全连接层和一个深度可分离卷积"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
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
    """多尺度池化注意力模块"""
    def __init__(self, dim, pool_ratios=[1, 2, 3, 6], num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        
        self.pool_ratios = pool_ratios
        self.pools = nn.ModuleList()
        for ratio in pool_ratios:
            self.pools.append(nn.ModuleList([
                nn.AdaptiveAvgPool2d(ratio),
                nn.Conv2d(dim, dim, 1, bias=qkv_bias),
            ]))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 生成查询
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1)
        
        # 多尺度池化生成键和值
        k_list = []
        v_list = []
        
        for pool, conv in self.pools:
            pool_x = pool(x)
            pool_x = conv(pool_x)
            
            k = self.k(pool_x).reshape(B, self.num_heads, C // self.num_heads, -1)
            v = self.v(pool_x).reshape(B, self.num_heads, C // self.num_heads, -1)
            
            k_list.append(k)
            v_list.append(v)
        
        # 拼接多尺度特征
        k = torch.cat(k_list, dim=-1)
        v = torch.cat(v_list, dim=-1)
        
        # 计算注意力
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重
        x = (attn @ v.transpose(-2, -1)).reshape(B, C, H, W)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MSPABlock(nn.Module):
    """多尺度并行注意力块"""
    def __init__(self, dim, mlp_ratio=4., pool_ratios=[1, 2, 3, 6], num_heads=8, qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_cfg=dict(type='BN', requires_grad=True)):
        super().__init__()
        
        self.norm_cfg = norm_cfg
        norm_layer = self._get_norm_layer()
        
        self.norm1 = norm_layer(dim)
        self.attn = MSPoolAttention(dim, pool_ratios, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def _get_norm_layer(self):
        """根据配置获取归一化层"""
        if self.norm_cfg['type'] == 'BN':
            return partial(nn.BatchNorm2d, eps=1e-6)
        elif self.norm_cfg['type'] == 'LN':
            return partial(nn.LayerNorm, eps=1e-6)
        else:
            return partial(nn.BatchNorm2d, eps=1e-6)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x