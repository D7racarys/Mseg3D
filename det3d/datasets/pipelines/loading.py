import os.path as osp
from turtle import shape
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 
import os
import cv2

from ..registry import PIPELINES



def select_points_in_frustum(points_2d, x1, y1, x2, y2):
    """
    for SemanticKITTI
    Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
    :param points_2d: point cloud projected into 2D
    :param points_3d: point cloud
    :param x1: left bound
    :param y1: upper bound
    :param x2: right bound
    :param y2: lower bound
    :return: points (2D and 3D) that are in the frustum
    """
    keep_ind = (points_2d[:, 0] > x1) * \
                (points_2d[:, 1] > y1) * \
                (points_2d[:, 0] < x2) * \
                (points_2d[:, 1] < y2)

    return keep_ind




def read_calib_semanticKITTI(calib_path):
    """
    for SemanticKITTI
    :param calib_path: Path to a calibration text file.
    :return: dict with calibration matrices.
    """
    calib_all = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if line == '\n':
                break
            key, value = line.split(':', 1)
            calib_all[key] = np.array([float(x) for x in value.split()])

    # reshape matrices
    calib_out = {}
    calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
    calib_out['Tr'] = np.identity(4)  # 4x4 matrix
    calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

    return calib_out



def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points



def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            dict_[k] = v[inds]

def read_file(path, tries=2, num_point_feature=4, painted=False):
    if painted:
        dir_path = os.path.join(*path.split('/')[:-2], 'painted_'+path.split('/')[-2])
        painted_path = os.path.join(dir_path, path.split('/')[-1]+'.npy')
        points =  np.load(painted_path)
        points = points[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]] # remove ring_index from features 
    else:
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :num_point_feature]

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep, painted=False):
    min_distance = 1.0
    points_sweep = read_file(str(sweep["lidar_path"]), painted=painted).T
    points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T


def read_single_semnusc_sweep(sweep, num_point_feature=5, painted=False, remove_close_flag=False):
    # NOTE: remove_close() will make the points.shape[0] and label.shape[0] mismatched.

    points_sweep = read_file(str(sweep["lidar_path"]), num_point_feature=num_point_feature, painted=painted).T
    
    if remove_close_flag:
        min_distance = 1.0
        points_sweep = remove_close(points_sweep, min_distance)

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T



def read_single_waymo(obj):
    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])

    points = np.concatenate([points_xyz, points_feature], axis=-1)
    
    return points 

def read_single_waymo_sweep(sweep):
    obj = get_obj(sweep['path'])

    points_xyz = obj["lidars"]["points_xyz"]
    points_feature = obj["lidars"]["points_feature"]

    # normalize intensity 
    points_feature[:, 0] = np.tanh(points_feature[:, 0])
    points_sweep = np.concatenate([points_xyz, points_feature], axis=-1).T # 5 x N

    nbr_points = points_sweep.shape[1]

    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot( 
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]

    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))
    
    return points_sweep.T, curr_times.T



def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 



@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)
        self.use_img = kwargs.get("use_img", False)

    def __call__(self, res, info):
        """
        加载点云数据的主函数
        
        主要功能：
        1. 根据不同的数据集类型加载点云数据
        2. 对于包含多个sweep的数据集，整合多帧点云数据
        3. 对于需要使用图像的情况，计算或获取点云在图像上的投影坐标
        
        参数说明：
        - res: 数据字典，用于存储加载的点云数据和其他信息
        - info: 包含数据路径和其他元信息的字典
        
        返回值：
        - 处理后的res和info字典
        
        注意：
        - 语义分割相关的数据集以"Semantic"为前缀
        - points_cp: 点云在图像上的投影坐标，形状为(npoints, 3)，格式为[cam_id, idx_of_width, idx_of_height]
        """

        # 设置数据集类型
        res["type"] = self.type

        # 1. 处理NuScenes数据集 (目标检测)
        if self.type == "NuScenesDataset":
            # 获取要加载的sweep数量
            nsweeps = res["lidar"]["nsweeps"]
            
            # 加载主点云文件
            lidar_path = Path(info["lidar_path"])
            points = read_file(str(lidar_path), painted=res["painted"])
            
            # 初始化点云和时间戳列表
            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]  # 主帧时间戳设为0
            
            # 检查sweep数量是否匹配
            assert (nsweeps - 1) == len(
                info["sweeps"]
            ), "nsweeps {} should equal to list length {}.".format(
                nsweeps, len(info["sweeps"])
            )
            
            # 随机选择并加载额外的sweep数据
            for i in np.random.choice(len(info["sweeps"]), nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = read_sweep(sweep, painted=res["painted"])
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)
            
            # 合并所有sweep的点云和时间戳
            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
            
            # 保存到结果字典中
            res["lidar"]["points"] = points
            res["lidar"]["times"] = times
            res["lidar"]["combined"] = np.hstack([points, times])  # 合并点云和时间戳信息
        
        # 2. 处理Waymo数据集 (目标检测)
        elif self.type == "WaymoDataset":
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            
            # 加载主点云文件
            obj = get_obj(path)
            points = read_single_waymo(obj)
            res["lidar"]["points"] = points
            
            # 如果需要加载多个sweep
            if nsweeps > 1: 
                # 初始化点云和时间戳列表
                sweep_points_list = [points]
                sweep_times_list = [np.zeros((points.shape[0], 1))]  # 主帧时间戳设为0
                
                # 检查sweep数量是否匹配
                assert (nsweeps - 1) == len(
                    info["sweeps"]
                ), "nsweeps {} should be equal to the list length {}.".format(
                    nsweeps, len(info["sweeps"])
                )
                
                # 按顺序加载额外的sweep数据
                for i in range(nsweeps - 1):
                    sweep = info["sweeps"][i]
                    points_sweep, times_sweep = read_single_waymo_sweep(sweep)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)
                
                # 合并所有sweep的点云和时间戳
                points = np.concatenate(sweep_points_list, axis=0)
                times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)
                
                # 保存到结果字典中
                res["lidar"]["points"] = points
                res["lidar"]["times"] = times
                res["lidar"]["combined"] = np.hstack([points, times])  # 合并点云和时间戳信息
        
        # 3. 处理SemanticKITTI数据集 (语义分割)
        elif self.type in ["SemanticKITTIDataset"]:
            path = info["path"]
            
            # 加载点云数据 (4维: x, y, z, intensity)
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            
            # 保存点云数据
            res["lidar"]["points"] = points
            res["lidar"]["times"] = None  # 语义分割数据集通常不使用时间戳
            res["lidar"]["combined"] = None
            
            # 如果需要使用图像信息，计算点云在图像上的投影
            if self.use_img:
                # 构建标定文件路径
                # 从velodyne路径转换到calib.txt路径
                calib_path = path[:-11].replace("velodyne", "calib.txt")
                
                # 读取标定数据
                calib = read_calib_semanticKITTI(calib_path)
                # 计算投影矩阵 (相机投影矩阵 P2 × 点云到相机的变换矩阵 Tr)
                proj_matrix = np.matmul(calib["P2"], calib["Tr"])  # shape: (3, 4)
                
                # 初始化点云投影坐标数组，初始值设为-100(无效值)
                pts_uv_all = np.ones([points.shape[0], 3]).astype(np.float32) * -100
                
                # 将点云转换为齐次坐标
                points_hcoords = np.concatenate([points[:, :3], np.ones([points.shape[0], 1], dtype=np.float32)], axis=1)
                # 投影到图像平面
                img_points = (proj_matrix @ points_hcoords.T).T
                # 归一化像素坐标 (除以z坐标)
                img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)
                
                # 设置图像尺寸 (SemanticKITTI的标准尺寸)
                im_width, im_height = 1224, 370
                # 计算在视锥体内的点云掩码
                frustum_mask = select_points_in_frustum(img_points, 0, 0, im_width, im_height)
                # 只保留车辆前方的点云 (x坐标大于0)
                mask = frustum_mask & (points[:, 0] > 0)
                
                # 填充有效的投影坐标
                # cam_id从1开始，遵循Waymo的风格
                pts_uv_all[mask, 0] = 1  # SemanticKITTI只有一个相机
                pts_uv_all[mask, 1:3] = img_points[mask, 0:2]  # 宽度和高度坐标
                
                # 保存投影坐标到结果字典
                # 格式: (npoints, 3), [cam_id, idx_of_width, idx_of_height]
                res["lidar"]["points_cp"] = pts_uv_all

        # 4. 处理SemanticWaymo数据集 (语义分割)
        elif self.type in ["SemanticWaymoDataset"]:
            path = info['path']
            nsweeps = res["lidar"]["nsweeps"]
            
            # 加载点云数据
            example_obj = get_obj(path)
            points = read_single_waymo(example_obj)
            
            # 保存点云数据
            res["lidar"]["points"] = points
            # 保存顶部激光雷达的点数量信息，用于后续选择ri_return1/ri_return2
            res["metadata"]["num_points_of_top_lidar"] = example_obj["lidars"]["num_points_of_top_lidar"]
            
            # 如果需要使用图像信息
            if self.use_img:
                # Waymo数据集直接提供点云的相机投影坐标
                points_cp = example_obj["lidars"]["points_cp"]  # (npoints, 3), [cam_id, idx_of_width, idx_of_height]
                res["lidar"]["points_cp"] = points_cp

        # 5. 处理SemanticNuscDataset数据集 (语义分割)
        elif self.type == "SemanticNuscDataset":
            # 构建激光雷达路径
            lidar_path = Path(info["lidar_path"])
            nsweeps = res["lidar"]["nsweeps"]
            
            # 加载点云数据 (5维)
            points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            res["lidar"]["points"] = points
             
            # 如果需要使用图像信息，计算点云在图像上的投影
            if self.use_img:
                # 获取相机通道信息
                cam_chan = res["cam"]["chan"]
                
                # 设置图像尺寸 (NuScenes的标准尺寸)
                im_shape = (900, 1600, 3)
                
                # 初始化点云投影坐标数组，初始值设为-100(无效值)
                # 格式: [cam_id, idx_of_width, idx_of_height]
                pts_uv_all = np.ones([points.shape[0], 3]).astype(np.float32) * -100
                
                # 对每个相机计算点云投影
                for cam_id, cam_sensor in enumerate(cam_chan):
                    # 获取相机的外部和内部参数
                    cam_from_global = info["cams_from_global"][cam_sensor]
                    cam_intrinsic = info["cam_intrinsics"][cam_sensor]
                    
                    # 点云从激光雷达坐标系转换到全局坐标系
                    ref_to_global = info["ref_to_global"]
                    pts_hom = np.concatenate([points[:, :3], np.ones([points.shape[0], 1])], axis=1)
                    pts_global = ref_to_global.dot(pts_hom.T)  # 4 * N
                    
                    # 从全局坐标系转换到相机坐标系
                    pts_cam = cam_from_global.dot(pts_global)[:3, :]  # 3 * N
                    
                    # 从相机坐标系投影到图像平面
                    pts_uv = view_points(pts_cam, np.array(cam_intrinsic), normalize=True).T  # N * 3
                    
                    # 过滤掉不在图像范围内或在相机后方的点云
                    # 留出1像素的边缘以避免边界效应
                    mask = (pts_cam[2, :] > 0) & (pts_uv[:, 0] > 1) & (pts_uv[:, 0] < im_shape[1] - 1) & (
                            pts_uv[:, 1] > 1) & (pts_uv[:, 1] < im_shape[0] - 1)
                    
                    # 填充有效的投影坐标
                    pts_uv_all[mask, :2] = pts_uv[mask, :2]  # 宽度和高度坐标
                    # cam_id从1开始，遵循Waymo的风格
                    pts_uv_all[mask, 2] = float(cam_id) + 1
                
                # 重新格式化坐标，使其与SemanticWaymo保持一致
                # 格式: [cam_id, idx_of_width, idx_of_height]
                points_cp = pts_uv_all[:, [2, 0, 1]]
                
                # 保存投影坐标到结果字典
                res["lidar"]["points_cp"] = points_cp
        
        # 如果数据集类型不支持，抛出异常
        else:
            raise NotImplementedError("不支持的数据集类型: {}".format(self.type))
        
        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    def __init__(self, with_bbox=True, **kwargs):
        self.with_bbox = with_bbox


    def __call__(self, res, info):
        """
        The semantic segmentation related datasets are denoted with prefix "Semantic".
        """

        # nusc det3d case preserved from CenterPoint
        if res["type"] in ["NuScenesDataset"] and "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0
            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        # waymo det3d case preserved from CenterPoint
        elif res["type"] == 'WaymoDataset' and "gt_boxes" in info:
            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
            }
        # kitti seg3d
        elif res["type"] == 'SemanticKITTIDataset':
            path = info["path"]
            learning_map = info["learning_map"]

            # get *.label path from *.bin path
            label_path = path.replace("velodyne", "labels").replace(".bin", ".label")
            # all_labels = np.fromfile(label_path, dtype=np.int32).reshape(-1)
            annotated_data = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            
            # semantic labels
            sem_labels = annotated_data & 0xFFFF
            # instance labels
            # inst_labels = annotated_data 
            inst_labels = annotated_data.astype(np.float32) 

            # label mapping 
            sem_labels = (np.vectorize(learning_map.__getitem__)(sem_labels)).astype(np.float32)

            res["lidar"]["annotations"] = {
                "point_sem_labels": sem_labels,
                "point_inst_labels": inst_labels,
            }
            # info["dim"]["sem_labels"] = 1
        
        # waymo seg3d
        elif res["type"] == 'SemanticWaymoDataset':
            # TYPE_UNDEFINED: 0
            assert info["seg_annotated"], "==> Seg annotated frames only!"
            anno_path = info['anno_path']
            obj = get_obj(anno_path)
            semantic_anno = obj["seg_labels"]["points_seglabel"] # (numpoints_toplidar, 2), [ins, sem]

            num_points_top_lidar = semantic_anno.shape[0]
            num_points_all_lidars = res["lidar"]["points"].shape[0]

            assert num_points_top_lidar == res["metadata"]["num_points_of_top_lidar"]["ri_return1"] + res["metadata"]["num_points_of_top_lidar"]["ri_return2"]
            semantic_anno_padded = np.zeros(shape=(num_points_all_lidars, semantic_anno.shape[-1]), dtype=semantic_anno.dtype)
            semantic_anno_padded[:num_points_top_lidar, :] = semantic_anno

            res["lidar"]["annotations"] = {
                "point_sem_labels": semantic_anno_padded[:, 1],
                "point_inst_labels": semantic_anno_padded[:, 0],
            }


        # nusc seg3d
        elif res["type"] == 'SemanticNuscDataset':
            learning_map = res['learning_map']
            data_root = '/data/luochao/Paper/UniPAD/data/nuscenes'
            # lidarseg_labels_filename: /data/luochao/Paper/UniPAD/data/nuscenes/lidarseg/v1.0-trainval/e6ca15bc5803457cba8d05f5e78f4e40_lidarseg.bin
            lidarseg_labels_filename = os.path.join(data_root, info['seganno_path'])

            point_sem_labels = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape(-1)
            point_sem_labels = np.vectorize(learning_map.__getitem__)(point_sem_labels).astype(np.float32)
            
            # NOTE: We have only parsed the semantic labels. If you want to use the instance labels, please check them carefully.
            point_inst_labels = np.zeros_like(point_sem_labels)
            
            res["lidar"]["annotations"] = {
                "point_sem_labels": point_sem_labels,
                "point_inst_labels":point_inst_labels
            }

        else:
            pass 

        return res, info




@PIPELINES.register_module
class LoadImageFromFile(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):

        dataset_type = res["type"]

        if dataset_type == "NuScenesDataset":
            raise NotImplementedError
        
        elif dataset_type in ["WaymoDataset"]:
            raise NotImplementedError
        
        elif dataset_type in ["SemanticKITTIDataset"]:
            path = info["path"]
            # image_2_path:  data/SemanticKITTI/dataset/sequences/01/image_2/000732.png
            image_2_path = path.replace('velodyne', 'image_2').replace('.bin', '.png')

            # reformat as waymo and nusc 
            cam_paths = {'1': image_2_path}
            cam_names = res["cam"]["names"]
            ori_images = [cv2.imread(cam_paths[cam_id]) for cam_id in cam_names]

            res["images"] = ori_images
            

        elif dataset_type in ["SemanticWaymoDataset"]:
            cam_names = res["cam"]["names"] # a list of ['1', '2', '3', '4', '5']
            cam_paths = info["cam_paths"]   # a dict. The key is set as the cam_id   
            ori_images = [cv2.imread(cam_paths[cam_id]) for cam_id in cam_names]

            res["images"] = ori_images

        elif dataset_type in ["SemanticNuscDataset"]:
            # cam_chan: a list of ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            # cam_names: a list of ['1', '2', '3', '4', '5', '6']
            cam_chan = res["cam"]["chan"]
            cam_names = res["cam"]["names"] 
            

            cam_paths = info["cam_paths"] 
            # img shape for all cameras: (900, 1600, 3)
            ori_images = [cv2.imread(cam_paths[cam_sensor]) for cam_sensor in cam_chan]

            res["images"] = ori_images


        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadImageAnnotations(object):
    """
    将点云的语义标签投影到图像上，生成稀疏的像素级标签
    
    主要功能：
    - 将点云中每个点的语义标签投影到对应的相机图像上
    - 在图像上生成与点云语义标签相对应的像素级标签图
    - 支持多种语义分割数据集（SemanticWaymoDataset、SemanticNuscDataset、SemanticKITTIDataset）
    
    注意：
    - 此处理流程应放置在LoadPointCloudFromFile/LoadImageFromFile/LoadPointCloudAnnotations之后
    - 仅适用于语义分割相关的数据集
    """
    
    def __init__(self, **kwargs):
        """
        初始化LoadImageAnnotations类
        
        参数说明：
        - points_cp_radius: 点云投影到图像上的半径大小，默认值为1
        """
        self.points_cp_radius = kwargs.get("points_cp_radius", 1)


    def __call__(self, res, info):
        """
        处理点云语义标签投影到图像的主函数
        
        主要功能：
        1. 从数据字典中提取相机名称、图像、点云投影坐标和点云语义标签
        2. 遍历每个相机，生成对应的像素级语义标签图
        3. 将生成的标签图添加到结果字典中
        
        参数说明：
        - res: 数据字典，包含加载的点云数据、图像和其他信息
        - info: 包含数据路径和其他元信息的字典
        
        返回值：
        - 处理后的res和info字典
        """

        dataset_type = res["type"]

        # 1. 处理NuScenesDataset (暂未实现)
        if dataset_type == "NuScenesDataset":
            raise NotImplementedError
        
        # 2. 处理WaymoDataset (暂未实现)
        elif dataset_type in ["WaymoDataset"]:
            raise NotImplementedError
            
        # 3. 处理语义分割数据集
        elif dataset_type in ["SemanticWaymoDataset", "SemanticNuscDataset", "SemanticKITTIDataset"]:
            # 提取相机名称（如['1', '2', '3', '4', '5']）
            cam_names = res["cam"]["names"]

            # 提取原始图像、点云投影坐标和点云语义标签
            ori_images = res["images"]
            ori_points_cp = res["lidar"]["points_cp"]  # [npoints, 3] 格式：[cam_id, idx_of_width, idx_of_height]
            ori_point_sem_labels = res["lidar"]["annotations"]["point_sem_labels"]  # [npoints,] 点云的语义标签
            
            # 初始化图像语义标签图列表
            ori_image_sem_maps = []
            
            # 遍历每个相机，生成对应的语义标签图
            for cam_id, ori_image in zip(cam_names, ori_images):
                H, W = ori_image.shape[0], ori_image.shape[1]  # 获取图像高度和宽度
                
                # 创建与图像尺寸相同的空语义标签图
                cur_ori_sem_map = np.zeros((H, W), dtype=ori_image.dtype)  # 通常为uint8类型
                
                # 筛选出当前相机视角下的点云投影坐标
                point_cam_id_mask = ori_points_cp[:, 0] == int(cam_id)  # 筛选cam_id匹配的点云
                cur_ori_points_cp = ori_points_cp[point_cam_id_mask]  # 获取当前相机的点云投影坐标
                
                # 提取宽度坐标、高度坐标和对应的语义标签
                cur_wid_coords = list(cur_ori_points_cp[:, 1])  # 宽度坐标
                cur_hei_coords = list(cur_ori_points_cp[:, 2])  # 高度坐标
                cur_sem_labels = list(ori_point_sem_labels[point_cam_id_mask])  # 对应的语义标签
                
                # 将每个点的语义标签绘制到图像上
                for i in range(len(cur_wid_coords)):
                    # 只处理有效的语义标签（大于0的值）
                    if cur_sem_labels[i] > 0:
                        # 在语义标签图上绘制圆形标记
                        cv2.circle(
                            cur_ori_sem_map, 
                            center=(int(cur_wid_coords[i]), int(cur_hei_coords[i])),  # 圆心坐标
                            radius=self.points_cp_radius,  # 圆的半径
                            color=int(cur_sem_labels[i]),  # 圆的颜色（语义标签值）
                            thickness=-1,  # 填充圆形
                        )

                # 将生成的语义标签图添加到列表
                ori_image_sem_maps.append(cur_ori_sem_map)

            # 创建图像标注字典
            img_gt_dict = {
                "image_sem_labels": ori_image_sem_maps,  # 存储所有相机的语义标签图
            } 

        else:
            # 不支持的数据集类型
            raise NotImplementedError(f"不支持的数据集类型: {dataset_type}")

        # 将图像标注添加到结果字典
        res["cam"]["annotations"] = img_gt_dict

        
        # from ..utils.printres import print_dict_structure
        # print("After LoadImageAnnotations: res is")
        # print_dict_structure(res)
        # print("info is")
        # print_dict_structure(info)
        return res, info







