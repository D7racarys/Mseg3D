# 作用 ：实现PGCN（Point Graph Convolutional Network）的核心功能

# - 包含k-NN图构建算法，用于建立点云之间的邻接关系
# - 实现图卷积层，通过边特征进行特征传播和聚合
# - 支持训练时的随机采样和推理时的完整图处理
# - 提供可配置的邻居数量(k)和采样率参数

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from det3d.models.registry import HEADS


def knn(x, k):
    """Compute k-nearest neighbors for point cloud with simplified approach.
    
    Args:
        x (torch.Tensor): Input point coordinates [B, N, 3]
        k (int): Number of neighbors
        
    Returns:
        tuple: (indices, distances) for k-NN
    """
    device = x.device
    B, N, C = x.shape
    
    # Handle edge cases
    if N <= 0:
        idx = torch.zeros((B, 0, k), dtype=torch.long, device=device)
        dist = torch.zeros((B, 0, k), dtype=torch.float, device=device)
        return (idx, dist)
    
    # Ensure k doesn't exceed the number of points
    k = min(k, N)
    if k <= 0:
        idx = torch.zeros((B, N, 0), dtype=torch.long, device=device)
        dist = torch.zeros((B, N, 0), dtype=torch.float, device=device)
        return (idx, dist)
    
    # Move to CPU for computation to avoid CUDA issues
    x_cpu = x.cpu()
    
    idx_list = []
    dist_list = []
    
    for b in range(B):
        batch_coords = x_cpu[b]  # [N, 3]
        batch_idx = []
        batch_dist = []
        
        for i in range(N):
            # Calculate distances on CPU
            point = batch_coords[i:i+1]  # [1, 3]
            distances = torch.sum((batch_coords - point) ** 2, dim=1)  # [N]
            
            # Get k nearest neighbors
            _, indices = torch.topk(distances, k, largest=False)
            
            batch_idx.append(indices.tolist())
            batch_dist.append(distances[indices].tolist())
        
        idx_list.append(batch_idx)
        dist_list.append(batch_dist)
    
    # Convert back to GPU tensors
    idx = torch.tensor(idx_list, dtype=torch.long).to(device)  # [B, N, k]
    dist = torch.tensor(dist_list, dtype=torch.float).to(device)  # [B, N, k]
    
    knn_out = (idx, dist)
    return knn_out


def knn_feature(x, k=20):
    """Extract k-NN features for graph convolution.
    
    Args:
        x (torch.Tensor): Input features [B, C, N]
        k (int): Number of neighbors
        
    Returns:
        torch.Tensor: K-NN features [B, 4*C, N, k]
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = knn(x.transpose(1, 2), k=k)[0]  # (batch_size, num_points, k)
    dist = knn(x.transpose(1, 2), k=k)[1]
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    dist = dist.view(batch_size, num_points, k, 1).repeat(1, 1, 1, num_dims)

    # Concatenate: [x_i, x_j, x_j-x_i, dist]
    feature = torch.cat((x, feature, feature - x, dist), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature  # (batch_size, 4*num_dims, num_points, k)


class KnnFeature(nn.Module):
    """K-NN feature extraction module."""
    
    def __init__(self, d_in, d_out, k):
        super(KnnFeature, self).__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(d_in * 4, d_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_out),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, features):
        features = knn_feature(features.squeeze(-1), k=self.k)
        return self.conv(features)


class PointFeatureFusion(nn.Module):
    """Point feature fusion module with geometric information."""
    
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(PointFeatureFusion, self).__init__()
        self.num_neighbors = num_neighbors
        self.knnfeature = KnnFeature(d_in, d_out, num_neighbors)
        self.mlp = nn.Sequential(
            nn.Conv2d(10, d_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_out),
            nn.ReLU()
        )
        self.device = device

    def forward(self, coords, features, knn_output):
        features = self.knnfeature(features)
        
        # Finding neighboring points
        idx, dist = knn_output
        idx, dist = idx.to(self.device), dist.to(self.device)
        B, N, K = idx.size()
        
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2, -1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords, 2, extended_idx)  # shape (B, 3, N, K)

        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)

        return torch.cat((
            self.mlp(concat),
            features
        ), dim=-3)


class CrossPooling(nn.Module):
    """Cross pooling with attention mechanism."""
    
    def __init__(self, in_channels, out_channels):
        super(CrossPooling, self).__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = nn.Sequential(
            nn.Conv2d(3 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

    def forward(self, x):
        scores = self.score_fn(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        att_features = torch.sum(scores * x, dim=-1, keepdim=True)  # shape (B, d_in, N, 1)
        max_features = x.max(dim=-1, keepdim=True)[0]
        mean_features = x.mean(dim=-1, keepdim=True)
        features = torch.cat((att_features, max_features, mean_features), dim=1)
        return self.mlp(features)


class FeatureExtractionModule(nn.Module):
    """Feature extraction module combining point fusion and pooling."""
    
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(FeatureExtractionModule, self).__init__()
        self.num_neighbors = num_neighbors
        self.lse1 = PointFeatureFusion(d_in, d_out // 2, num_neighbors, device)
        self.pool2 = CrossPooling(d_out, d_out)

    def forward(self, coords, features):
        knn_output = knn(coords.contiguous(), self.num_neighbors)
        x = self.lse1(coords, features, knn_output)
        x = self.pool2(x)
        return x


class GetKnnGraph(nn.Module):
    """K-NN graph construction module."""
    
    def __init__(self, k, random_rate=1.0, isTrain=True):
        super(GetKnnGraph, self).__init__()
        self.k = k
        self.random_rate = random_rate
        self.isTrain = isTrain

    def forward(self, points):
        """Construct k-NN graph from point coordinates.
        
        Args:
            points (torch.Tensor): Point coordinates [B, N, 3]
            
        Returns:
            torch.Tensor: Edge indices for graph convolution
        """
        batch_size, num_points, _ = points.shape
        device = points.device
        
        # Compute k-NN
        idx, _ = knn(points, self.k)
        
        # Random sampling during training
        if self.isTrain and self.random_rate < 1.0:
            num_keep = int(self.k * self.random_rate)
            rand_idx = torch.randperm(self.k, device=device)[:num_keep]
            idx = idx[:, :, rand_idx]
        
        # Convert to edge index format
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1)
        batch_idx = batch_idx.expand(-1, num_points, idx.size(-1))
        
        source = torch.arange(num_points, device=device).view(1, -1, 1)
        source = source.expand(batch_size, -1, idx.size(-1))
        
        edge_index = torch.stack([
            (batch_idx * num_points + source).view(-1),
            (batch_idx * num_points + idx).view(-1)
        ], dim=0)
        
        return edge_index


class GraphConv(nn.Module):
    """Graph convolution layer."""
    
    def __init__(self, in_channels, out_channels):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU()
        )

    def forward(self, x, edge_index):
        """Forward pass of graph convolution.
        
        Args:
            x (torch.Tensor): Node features [B, C, N, 1]
            edge_index (torch.Tensor): Edge indices [2, E]
            
        Returns:
            torch.Tensor: Updated node features [B, C, N, 1]
        """
        B, C, N, _ = x.shape
        x = x.squeeze(-1)  # [B, C, N]
        
        # Flatten for edge-based operations
        x_flat = x.view(B * N, C)  # [B*N, C]
        
        # Gather source and target features
        source_idx, target_idx = edge_index
        x_source = x_flat[source_idx]  # [E, C]
        x_target = x_flat[target_idx]  # [E, C]
        
        # Concatenate source and target features
        edge_features = torch.cat([x_source, x_target - x_source], dim=1)  # [E, 2*C]
        
        # Apply convolution
        edge_features = edge_features.transpose(0, 1).unsqueeze(0)  # [1, 2*C, E]
        edge_features = self.conv(edge_features).squeeze(0).transpose(0, 1)  # [E, out_C]
        
        # Aggregate features back to nodes
        out = torch.zeros(B * N, self.out_channels, device=x.device)
        out.index_add_(0, source_idx, edge_features)
        
        # Reshape back
        out = out.view(B, self.out_channels, N).unsqueeze(-1)  # [B, out_C, N, 1]
        
        return out


@HEADS.register_module
class PGCN(BaseModule):
    """Point Graph Convolutional Network module for MSeg3D integration.
    
    Args:
        d_in (int): Input feature dimension
        d_out (int): Output feature dimension
        k (int): Number of neighbors for k-NN graph
        random_rate (float): Random sampling rate for graph construction
        device (str): Device for computation
        num_layers (int): Number of graph convolution layers
        init_cfg (dict, optional): Initialization config
    """
    
    def __init__(self, 
                 d_in=32, 
                 d_out=64, 
                 k=20, 
                 random_rate=0.8, 
                 device='cuda',
                 num_layers=1,
                 init_cfg=None):
        super(PGCN, self).__init__(init_cfg=init_cfg)
        
        self.k = k
        self.random_rate = random_rate
        self.device = device
        self.num_layers = num_layers
        
        # Feature extraction module
        self.conv1 = FeatureExtractionModule(
            d_in, d_out, int(self.k * self.random_rate), self.device
        )
        
        # Graph convolution modules
        self.graph_convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.graph_convs.append(GraphConv(d_out, d_out))
        
        # Graph construction modules
        self.knn = GetKnnGraph(k=self.k, random_rate=self.random_rate, isTrain=True)
        self.knn_test = GetKnnGraph(k=self.k, random_rate=self.random_rate, isTrain=False)

    def forward(self, points, features, mode='train'):
        """Forward pass of PGCN module.
        
        Args:
            points (torch.Tensor): Point coordinates [B, N, 3]
            features (torch.Tensor): Point features [B, C, N, 1]
            mode (str): 'train' or 'test' mode
            
        Returns:
            torch.Tensor: Enhanced point features [B, d_out, N, 1]
        """
        # Construct k-NN graph
        if mode == 'train':
            edge_index = self.knn(points.transpose(1, 2))
        else:
            edge_index = self.knn_test(points.transpose(1, 2))
        
        # Feature extraction with geometric information
        x = self.conv1(points, features)
        
        # Multi-layer graph convolution with residual connections
        for i, graph_conv in enumerate(self.graph_convs):
            if i == 0:
                # First layer
                x = graph_conv(x, edge_index)
            else:
                # Subsequent layers with residual connection
                residual = x
                x = graph_conv(x, edge_index)
                x = x + residual  # Residual connection
        
        return x

    def init_weights(self):
        """Initialize weights of the PGCN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)