from .. import builder
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from det3d.torchie.trainer import load_checkpoint

# 所以，SegMSeg3DNet 的输出是：
# 训练时：一个包含总损失及各项子损失的字典。
# 推理时：三维点云上每个点的语义类别预测（以及可能的置信度）。

@DETECTORS.register_module
class SegMSeg3DNet(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        img_backbone,
        img_head,
        point_head,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        **kwargs,
    ):
        super(SegMSeg3DNet, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained=None
        )
        
        self.img_backbone = builder.build_img_backbone(img_backbone)
        self.img_head = builder.build_img_head(img_head)
        self.point_head = builder.build_point_head(point_head)
        


    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        try:
            load_checkpoint(self, pretrained, strict=False)
            print("init weight from {}".format(pretrained))
        except:
            print("no pretrained model at {}".format(pretrained))


    def extract_feat(self):
        assert False


    def forward(self, example, return_loss=True, **kwargs):
        """
        example是样本，也是数据pipeline中的res
        """
        # milestone
        # from ...datasets.utils.printres import print_dict_structure
        # print("In SegMSeg3DNet: example is")
        # print_dict_structure(example)

        voxels = example["voxels"]
        # 体素坐标
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]
        batch_size = len(num_voxels)
        # ensure that the points just including [bs_idx, x, y, z]
        points = example["points"][:, 0:4]





        # camera branch
        # images: [batch, num_cams, num_ch, h, w] like [1, 5, 3, 640, 960]
        images = example["images"]         
        num_cams, hi, wi = images.shape[1], images.shape[3], images.shape[4] 
        # images: (batch, num_cams=5, 3, h, w) -> (batch*num_cams=5, 3, h, w)
        images = images.view(-1, 3, hi, wi)
        
        # img_backbone_return: (batch*num_cams=5, c, ho, wo)
        img_backbone_return = self.img_backbone(images)
        img_data = dict(
            inputs=img_backbone_return,
            batch_size=batch_size,
        )
        # 参数的作用是控制模型在前向推理时是否返回损失（loss），用于训练阶段的反向传播和参数更新
        # 训练阶段开启 return_loss=True，模型会计算并返回损失值
        # 推理阶段设为 return_loss=False，模型只返回预测结果
        if return_loss:
            # (batch, num_cams=5, h, w) -> (batch*num_cams=5, h, w) -> (batch*num_cams=5, 1, h, w)
            images_sem_labels = example["images_sem_labels"]
            images_sem_labels = images_sem_labels.view(-1, hi, wi).unsqueeze(1)
            img_data["images_sem_labels"] = images_sem_labels
        img_data = self.img_head(batch_dict=img_data, return_loss=return_loss)
        # get image_features from the img_head
        image_features = img_data["image_features"]
        _, num_chs, ho, wo = image_features.shape
        # 将相机特征维度从四维恢复为五维
        # bathch_size * cam_num -> batch_size, cam_num
        image_features = image_features.view(batch_size, num_cams, num_chs, ho, wo)



        # lidar branch
        # construct a batch_dict like pv-rcnn
        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            voxel_coords=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
            points=points,
        )

        # VFE voxel feature encoding
        input_features = self.reader(data["features"], data["num_voxels"], data["voxel_coords"])
        data["voxel_features"] = input_features
        
        # backbone
        # 就是将点云体素特征输入到主干网络（UNetSCN3D）进行特征提取和空间编码
        data = self.backbone(data)

        # prepare labels for training
        if return_loss:
            data["voxel_sem_labels"] = example["voxel_sem_labels"]
            data["point_sem_labels"] = example["point_sem_labels"]



        # fusion and segmentation in point head
        data["points_cuv"] = example["points_cuv"]
        # example["points_cuv"] =
        #   [
        #       [0, 123.4, 321.5],   # 这个点落在前视相机图像 (u=123.4, v=321.5)
        #       [1, 512.7, 200.8],   # 这个点落在右视相机图像 (u=512.7, v=200.8)
        #   ]

        data["image_features"] = image_features
        data["camera_semantic_embeddings"] = img_data.get("camera_semantic_embeddings", None)
        data["metadata"] = example.get("metadata", None)

        # 前向传播得到输出结果
        data = self.point_head(batch_dict=data, return_loss=return_loss)

        # 训练时返回损失字典，包含总损失和各项子损失
        # 推理时只返回分割预测结果
        mask_near = example["mask_near"]
        if return_loss:
            seg_loss_dict = {}
            # 应该在这里传入mask
            point_loss, point_loss_dict = self.point_head.get_loss(mask_near=mask_near)

            # compute the img head loss
            img_loss, point_loss_dict = self.img_head.get_loss(point_loss_dict)


            # this item for Optimizer, formating as loss per task
            total_loss = point_loss + img_loss
            opt_loss = [total_loss]
            seg_loss_dict["loss"] = opt_loss

            # reformat for text logger
            for k, v in point_loss_dict.items():
                repeat_list = [v for i in range(len(opt_loss))]
                seg_loss_dict[k] = repeat_list

            return seg_loss_dict

        else:
            return self.point_head.predict(example=example, test_cfg=self.test_cfg)
