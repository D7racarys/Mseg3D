# 作用 ：将PGCN模块无缝集成到MSeg3D架构中

# - 继承自原始MSeg3D分割头，保持所有多模态融合功能
# - 在前向传播中集成PGCN图特征学习
# - 实现体素特征与图特征的融合机制
# - 支持批处理和分布式训练

from inspect import stack
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from ..registry import POINT_HEADS
from det3d.core.utils.loss_utils import lovasz_softmax
from .point_utils import three_interpolate_wrap
from .context_module import LiDARSemanticFeatureAggregationModule, SemanticFeatureFusionModule
from .pgcn_module import PGCN


@POINT_HEADS.register_module
class PointSegMSeg3DPGCNHead(nn.Module):
    """MSeg3D segmentation head enhanced with PGCN module.
    
    This head integrates Point Graph Convolutional Network (PGCN) into the MSeg3D
    architecture to improve point cloud feature learning through graph-based operations.
    
    Args:
        class_agnostic (bool): Whether to use class agnostic prediction
        num_class (int): Number of segmentation classes
        model_cfg (dict): Model configuration containing PGCN and other parameters
    """
    
    def __init__(self, class_agnostic, num_class, model_cfg, **kwargs):
        super().__init__()
        
        if class_agnostic:
            self.num_class = 1
        else:
            self.num_class = num_class
        
        norm_layer = partial(nn.BatchNorm1d, eps=1e-6)
        act_layer = nn.ReLU

        # PGCN configuration
        pgcn_cfg = model_cfg.get("PGCN_CFG", {})
        self.use_pgcn = pgcn_cfg.get("USE_PGCN", True)
        self.pgcn_fusion_method = pgcn_cfg.get("FUSION_METHOD", "concat")  # 'concat', 'add', 'attention'
        
        voxel_in_channels = model_cfg["VOXEL_IN_DIM"]
        self.dp_ratio = model_cfg["DP_RATIO"]
        
        # PGCN module initialization
        if self.use_pgcn:
            pgcn_d_in = pgcn_cfg.get("D_IN", voxel_in_channels)
            pgcn_d_out = pgcn_cfg.get("D_OUT", 64)
            pgcn_k = pgcn_cfg.get("K", 20)
            pgcn_random_rate = pgcn_cfg.get("RANDOM_RATE", 0.8)
            pgcn_num_layers = pgcn_cfg.get("NUM_LAYERS", 1)  # Support for multiple layers
            
            self.pgcn = PGCN(
                d_in=pgcn_d_in,
                d_out=pgcn_d_out,
                k=pgcn_k,
                random_rate=pgcn_random_rate,
                device='cuda',
                num_layers=pgcn_num_layers
            )
            
            # Feature fusion layers for PGCN integration
            if self.pgcn_fusion_method == "concat":
                self.pgcn_fusion_dim = voxel_in_channels + pgcn_d_out
                self.pgcn_fusion_layer = nn.Sequential(
                    nn.Linear(self.pgcn_fusion_dim, voxel_in_channels),
                    norm_layer(voxel_in_channels),
                    act_layer()
                )
            elif self.pgcn_fusion_method == "attention":
                self.pgcn_fusion_dim = voxel_in_channels
                self.pgcn_attention = nn.MultiheadAttention(
                    embed_dim=voxel_in_channels,
                    num_heads=4,
                    dropout=0.1,
                    batch_first=True
                )
                self.pgcn_proj = nn.Linear(pgcn_d_out, voxel_in_channels)
            else:  # add
                self.pgcn_fusion_dim = voxel_in_channels
                self.pgcn_proj = nn.Linear(pgcn_d_out, voxel_in_channels)
        else:
            self.pgcn_fusion_dim = voxel_in_channels

        # Auxiliary segmentation head on voxel features
        self.voxel_cls_layers = self.make_convcls_head(
            fc_cfg=model_cfg["VOXEL_CLS_FC"],
            input_channels=self.pgcn_fusion_dim,  # Updated to use fused dimension
            output_channels=self.num_class,
            dp_ratio=self.dp_ratio,
        )

        # GF-Phase: geometry-based feature fusion phase
        voxel_align_channels = model_cfg["VOXEL_ALIGN_DIM"]
        self.gffm_lidar = nn.Sequential(
            nn.Linear(self.pgcn_fusion_dim, voxel_align_channels),  # Updated input dimension
            norm_layer(voxel_align_channels),
            act_layer(),
        ) 

        image_in_channels = model_cfg["IMAGE_IN_DIM"]
        image_align_channels = model_cfg["IMAGE_ALIGN_DIM"]
        self.gffm_camera = nn.Sequential(
            nn.Linear(image_in_channels, image_align_channels),
            norm_layer(image_align_channels),
            act_layer(),
        ) 

        fused_channels = model_cfg["GEO_FUSED_DIM"]
        self.gffm_lc = nn.Sequential(
            nn.Linear(voxel_align_channels + image_align_channels, fused_channels),
            nn.BatchNorm1d(fused_channels),
            act_layer(),
        ) 

        # Cross-modal feature completion
        self.lidar_camera_mimic_layer = self.make_convcls_head(
            fc_cfg=model_cfg["MIMIC_FC"],
            input_channels=voxel_align_channels,
            output_channels=image_align_channels,
            dp_ratio=0,
        )

        # SF-Phase: semantic-based feature fusion phase
        SFPhase_CFG = model_cfg["SFPhase_CFG"]
        self.lidar_sfam = LiDARSemanticFeatureAggregationModule()
        self.sffm = SemanticFeatureFusionModule(
            d_input_point=fused_channels, 
            d_input_embeddings1=image_in_channels, 
            d_input_embeddings2=self.pgcn_fusion_dim,  # Updated to use fused dimension
            embeddings_proj_kernel_size=SFPhase_CFG["embeddings_proj_kernel_size"], 
            d_model=SFPhase_CFG["d_model"], 
            nhead=SFPhase_CFG["n_head"], 
            num_decoder_layers=SFPhase_CFG["n_layer"], 
            dim_feedforward=SFPhase_CFG["n_ffn"],
            dropout=SFPhase_CFG["drop_ratio"],
            activation=SFPhase_CFG["activation"], 
            normalize_before=SFPhase_CFG["pre_norm"],
        )

        # Final output head for point-wise segmentation
        sem_fused_channels = self.sffm.d_model
        self.out_cls_layers = nn.Linear(sem_fused_channels, num_class)

        # Build loss
        self.forward_ret_dict = {}
        self.ignored_label = model_cfg["IGNORED_LABEL"]
        self.cross_entropy_func = nn.CrossEntropyLoss(ignore_index=self.ignored_label)
        self.lovasz_softmax_func = lovasz_softmax
        self.mimic_loss_func = nn.MSELoss() 
        
        # PGCN loss weight
        self.pgcn_loss_weight = pgcn_cfg.get("LOSS_WEIGHT", 0.1)
        self.tasks = ["out"]

    def make_convcls_head(self, fc_cfg, input_channels, output_channels, dp_ratio=0):
        """Create convolutional classification head."""
        fc_layers = []
        c_in = input_channels
        if dp_ratio > 0:
            fc_layers.append(nn.Dropout(dp_ratio))
            
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]

        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def fuse_pgcn_features(self, voxel_features, pgcn_features):
        """Fuse PGCN features with original voxel features.
        
        Args:
            voxel_features (torch.Tensor): Original voxel features [N, C_voxel]
            pgcn_features (torch.Tensor): PGCN enhanced features [N, C_pgcn, 1]
            
        Returns:
            torch.Tensor: Fused features [N, C_fused]
        """
        # Squeeze PGCN features to match voxel features shape
        pgcn_features = pgcn_features.squeeze(-1)  # [N, C_pgcn]
        
        if self.pgcn_fusion_method == "concat":
            # Concatenate features
            fused_features = torch.cat([voxel_features, pgcn_features], dim=1)
            fused_features = self.pgcn_fusion_layer(fused_features)
        elif self.pgcn_fusion_method == "attention":
            # Project PGCN features to match voxel feature dimension
            pgcn_proj = self.pgcn_proj(pgcn_features)
            
            # Apply attention mechanism
            voxel_features_unsqueezed = voxel_features.unsqueeze(0)  # [1, N, C]
            pgcn_proj_unsqueezed = pgcn_proj.unsqueeze(0)  # [1, N, C]
            
            attended_features, _ = self.pgcn_attention(
                query=voxel_features_unsqueezed,
                key=pgcn_proj_unsqueezed,
                value=pgcn_proj_unsqueezed
            )
            fused_features = attended_features.squeeze(0) + voxel_features
        else:  # add
            # Project and add features
            pgcn_proj = self.pgcn_proj(pgcn_features)
            fused_features = voxel_features + pgcn_proj
            
        return fused_features

    def get_loss(self, point_loss_dict=None):
        """Compute batch-wise loss including PGCN regularization."""
        point_loss = 0
        if point_loss_dict is None:
            point_loss_dict = {}

        # Voxel head loss
        voxel_ce_loss = self.cross_entropy_func( 
            self.forward_ret_dict["voxel_logits"], 
            self.forward_ret_dict["voxel_sem_labels"].long(), 
        )
        voxel_lvsz_loss = self.lovasz_softmax_func(
            F.softmax(self.forward_ret_dict["voxel_logits"], dim=-1),
            self.forward_ret_dict["voxel_sem_labels"].long(),
            ignore=self.ignored_label, 
        )
        voxel_loss = voxel_ce_loss + voxel_lvsz_loss
        point_loss += voxel_loss
        point_loss_dict["voxel_ce_loss"] = voxel_ce_loss.detach()
        point_loss_dict["voxel_lovasz_loss"] = voxel_lvsz_loss.detach()

        # Point loss
        out_ce_loss = self.cross_entropy_func(
            self.forward_ret_dict["out_logits"],
            self.forward_ret_dict["point_sem_labels"].long(),
        )
        out_lvsz_loss = self.lovasz_softmax_func(
            F.softmax(self.forward_ret_dict["out_logits"], dim=-1),
            self.forward_ret_dict["point_sem_labels"].long(),
            ignore=self.ignored_label, 
        )
        out_loss = out_ce_loss + out_lvsz_loss
        point_loss += out_loss
        point_loss_dict["out_ce_loss"] = out_ce_loss.detach()
        point_loss_dict["out_lovasz_loss"] = out_lvsz_loss.detach()

        # Mimic loss for feature completion
        out_mimic_loss = self.mimic_loss_func(
            self.forward_ret_dict["point_features_pcamera"],
            self.forward_ret_dict["point_features_camera"],
        )
        point_loss += out_mimic_loss
        point_loss_dict["out_mimic_loss"] = out_mimic_loss.detach()

        # PGCN regularization loss (optional)
        if self.use_pgcn and "pgcn_features" in self.forward_ret_dict:
            # Add graph regularization loss to encourage smooth features on connected nodes
            pgcn_reg_loss = self.compute_pgcn_regularization_loss()
            point_loss += self.pgcn_loss_weight * pgcn_reg_loss
            point_loss_dict["pgcn_reg_loss"] = pgcn_reg_loss.detach()

        return point_loss, point_loss_dict

    def compute_pgcn_regularization_loss(self):
        """Compute PGCN regularization loss for graph smoothness."""
        # Simple L2 regularization on PGCN features
        pgcn_features = self.forward_ret_dict["pgcn_features"]
        reg_loss = torch.mean(pgcn_features ** 2)
        return reg_loss

    def get_points_image_feature(self, input_img_feature, points_cuv, batch_idx):
        """Extract image features for given point coordinates."""
        # (batch, num_cam, num_chs, h, w) -> (batch, num_chs, num_cam, h, w)
        img_feature = input_img_feature.transpose(2, 1)

        batch_size, num_chs, num_cams, h, w = img_feature.shape
        point_feature_camera_list = []
        for i in range(batch_size):
            # (1, num_chs, num_cam, h, w)
            cur_img_feat = img_feature[i].unsqueeze(0)

            cur_batch_mask = (batch_idx == i)
            cur_points_cuv = points_cuv[cur_batch_mask]
            # (ni, 4) -> (1, 1, 1, ni, 4)
            cur_points_cuv = cur_points_cuv.reshape(1, 1, 1, cur_points_cuv.shape[0], cur_points_cuv.shape[-1])
            
            # Grid sample for feature extraction
            cur_points_feature_camera = F.grid_sample(
                cur_img_feat, 
                cur_points_cuv[..., (3, 2, 1)], 
                mode='bilinear', 
                padding_mode='zeros', 
                align_corners=True
            )

            # Reshape to [ni, num_chs]
            cur_points_feature_camera = cur_points_feature_camera.flatten(0, 1).flatten(1, 3).transpose(1, 0)
            point_feature_camera_list.append(cur_points_feature_camera)
        
        # Concatenate all batch features
        point_features_camera = torch.cat(point_feature_camera_list, dim=0)
        assert point_features_camera.shape[0] == points_cuv.shape[0]

        return point_features_camera

    def forward(self, batch_dict, return_loss=True, **kwargs):
        """Forward pass with PGCN integration.
        
        Args:
            batch_dict: Input batch containing point clouds, voxel features, and images
            return_loss: Whether to compute loss
            
        Returns:
            batch_dict: Updated batch dictionary with segmentation logits
        """
        batch_size = batch_dict["batch_size"]

        # Voxel features from the spconv backbone
        voxel_features = batch_dict["conv_point_features"]
        voxel_coords = batch_dict["conv_point_coords"]
        point_coords = batch_dict["points"]
        
        # Apply PGCN if enabled
        if self.use_pgcn:
            # Prepare point coordinates for PGCN (remove batch index)
            voxel_xyz = voxel_coords[:, 1:4]  # [N, 3]
            
            # Reshape voxel features for PGCN input [B, C, N, 1]
            # Group by batch for PGCN processing
            pgcn_features_list = []
            for b in range(batch_size):
                batch_mask = (voxel_coords[:, 0] == b)
                if batch_mask.sum() > 0:
                    batch_voxel_xyz = voxel_xyz[batch_mask].unsqueeze(0)  # [1, N_b, 3]
                    batch_voxel_features = voxel_features[batch_mask].unsqueeze(0).transpose(1, 2).unsqueeze(-1)  # [1, C, N_b, 1]
                    
                    # Apply PGCN
                    mode = 'train' if self.training else 'test'
                    batch_pgcn_features = self.pgcn(batch_voxel_xyz, batch_voxel_features, mode=mode)
                    pgcn_features_list.append(batch_pgcn_features.squeeze(0).transpose(0, 1))  # [N_b, C]
                else:
                    # Handle empty batch
                    pgcn_features_list.append(torch.zeros(0, self.pgcn.conv2.out_channels, device=voxel_features.device))
            
            # Concatenate PGCN features
            pgcn_features = torch.cat(pgcn_features_list, dim=0)  # [N_total, C_pgcn]
            
            # Store PGCN features for loss computation
            self.forward_ret_dict["pgcn_features"] = pgcn_features
            
            # Fuse PGCN features with original voxel features
            voxel_features_fused = self.fuse_pgcn_features(voxel_features, pgcn_features.unsqueeze(-1))
        else:
            voxel_features_fused = voxel_features

        # Voxel classification with fused features
        voxel_logits = self.voxel_cls_layers(voxel_features_fused)
        self.forward_ret_dict.update({
            "voxel_logits": voxel_logits,
        })
        if return_loss:
            self.forward_ret_dict.update({
                "voxel_sem_labels": batch_dict["voxel_sem_labels"],
                "point_sem_labels": batch_dict["point_sem_labels"],
                "batch_size": batch_size,
            })

        # Voxel features -> point lidar features with fused features
        point_features_lidar_0 = three_interpolate_wrap(
            new_coords=point_coords, 
            coords=voxel_coords, 
            features=voxel_features_fused,  # Use fused features
            batch_size=batch_size
        )        
        point_features_lidar = self.gffm_lidar(point_features_lidar_0)

        # Image feature maps -> point camera features
        image_features = batch_dict["image_features"]
        points_cuv = batch_dict["points_cuv"]
        valid_mask = (points_cuv[:, 0] == 1)
        
        point_features_camera_0 = self.get_points_image_feature(
            input_img_feature=image_features, 
            points_cuv=points_cuv[valid_mask],
            batch_idx=point_coords[:, 0][valid_mask],
        )
        point_features_camera = self.gffm_camera(point_features_camera_0)

        # Cross-modal feature completion
        point_features_pcamera = self.lidar_camera_mimic_layer(point_features_lidar[valid_mask])
        assert point_features_camera.shape[0] == point_features_pcamera.shape[0]
        
        if return_loss:
            self.forward_ret_dict.update({
                "point_features_pcamera": point_features_pcamera,
                "point_features_camera": point_features_camera.detach(),
            })
        
        # Feature completion
        point_features_camera_pad0 = torch.zeros(
            (valid_mask.shape[0], point_features_camera.shape[1]), 
            dtype=point_features_camera.dtype, 
            device=point_features_camera.device
        )
        point_features_camera_pad0[valid_mask] = point_features_camera
        
        point_features_pcamera_pad0 = torch.zeros(
            (valid_mask.shape[0], point_features_pcamera.shape[1]), 
            dtype=point_features_pcamera.dtype, 
            device=point_features_pcamera.device
        )
        point_features_pcamera_pad0[valid_mask] = point_features_pcamera

        point_features_ccamera = torch.where(
            valid_mask.unsqueeze(-1).expand_as(point_features_camera_pad0), 
            point_features_camera_pad0, 
            point_features_pcamera_pad0, 
        )

        # GFFM in GF-Phase
        point_features_lc = torch.cat([point_features_lidar, point_features_ccamera], dim=1)
        point_features_geo_fused = self.gffm_lc(point_features_lc)

        # SF-Phase with enhanced features
        camera_semantic_embeddings = batch_dict["camera_semantic_embeddings"]
        lidar_semantic_embeddings = self.lidar_sfam(
            feats=voxel_features_fused,  # Use fused features
            probs=voxel_logits, 
            batch_idx=voxel_coords[:, 0], 
            batch_size=batch_size,
        )
        
        point_features_sem_fused = self.sffm(
            input_point_features=point_features_geo_fused, 
            input_sem_embeddings1=camera_semantic_embeddings, 
            input_sem_embeddings2=lidar_semantic_embeddings, 
            batch_idx=point_coords[:, 0], 
            batch_size=batch_size,
        )

        # Final classification
        out_logits = self.out_cls_layers(point_features_sem_fused)

        batch_dict["out_logits"] = out_logits
        self.forward_ret_dict["out_logits"] = out_logits

        return batch_dict

    @torch.no_grad()
    def predict(self, example, test_cfg=None, **kwargs):
        """Prediction method for inference."""
        # Use the same prediction logic as the original MSeg3D head
        batch_size = len(example["num_voxels"])

        tta_flag = test_cfg.get('tta_flag', False)
        stack_points = example["points"][:, 0:4]

        ret_list = []
        if tta_flag:
            merge_type = test_cfg.get('merge_type', "ArithmeticMean")
            num_tta_tranforms = test_cfg.get('num_tta_tranforms', 4)
            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]
                meta_list = meta_list[:num_tta_tranforms*int(batch_size):num_tta_tranforms]
                
                stack_pred_logits = self.forward_ret_dict["out_logits"]
                stack_pred_logits = torch.softmax(stack_pred_logits, dim=-1)

                # Split and merge TTA predictions
                single_pc_list = []
                single_logits_list = []
                for i in range(batch_size):
                    bs_mask = stack_points[:, 0] == i
                    single_pc = stack_points[bs_mask]
                    single_logits = stack_pred_logits[bs_mask]
                    single_pc_list.append(single_pc)
                    single_logits_list.append(single_logits)

                merged_pc_list = []
                merged_pred_sem_labels_list = []
                merged_num_point_list = []
                for i in range(0, batch_size, num_tta_tranforms):
                    merged_pc_list.append(single_pc_list[i])
                    merged_num_point_list.append(single_pc_list[i].shape[0])
                    if merge_type == "ArithmeticMean":
                        merged_logits_list = single_logits_list[i: i+num_tta_tranforms]
                        merged_logits = torch.stack(merged_logits_list, dim=0)
                        merged_logits = torch.mean(merged_logits, dim=0)    
                    else: 
                        raise NotImplementedError
                    merged_pred_sem_labels = torch.argmax(merged_logits, dim=1)
                    merged_pred_sem_labels_list.append(merged_pred_sem_labels)

                left_ind = 0
                for i in range(int(batch_size/num_tta_tranforms)):
                    ret = {}
                    ret["metadata"] = meta_list[i]
                    ret["pred_point_sem_labels"] = merged_pred_sem_labels_list[i]
                    
                    if "point_sem_labels" in example: 
                        right_ind = sum(merged_num_point_list[:i+1])
                        ret["point_sem_labels"] = example["point_sem_labels"][left_ind:right_ind]
                        left_ind = right_ind

                    ret_list.append(ret)
        else: 
            if "metadata" not in example or len(example["metadata"]) == 0:
                meta_list = [None] * batch_size
            else:
                meta_list = example["metadata"]

                stack_pred_logits = self.forward_ret_dict["out_logits"]                
                stack_pred_sem_labels = torch.argmax(stack_pred_logits, dim=1)
                
                for i in range(batch_size):
                    ret = {}
                    ret["metadata"] = meta_list[i]

                    cur_bs_mask = (stack_points[:, 0] == i)
                    ret["pred_point_sem_labels"] = stack_pred_sem_labels[cur_bs_mask]

                    if "point_sem_labels" in example:
                        ret["point_sem_labels"] = example["point_sem_labels"][cur_bs_mask]

                    ret_list.append(ret)

        return ret_list