import numpy as np
import torch
import random
from typing import Dict, Tuple
from ..registry import PIPELINES


@PIPELINES.register_module
class SegMaskPoints(object):
    def __init__(self, drop_ratio=0.3, mode="random", sector_params=None, **kwargs):
        self.drop_ratio = drop_ratio
        self.mode = mode
        self.sector_params = sector_params
    
    # ---------- 1) LiDAR 掉点（多种策略） ----------
    def __call__(self, res: Dict, info: Dict):
        """
        points: (N, 3) or (N, >=3) 原始点云坐标/属性
        labels: (N,) 每点语义标签（gt）
        drop_ratio: 要删除的点比例（0-1）
        mode: "random" | "range" | "azimuth_sector" | "clustered"
        返回:
        kept_points, kept_labels, non_overlap_mask (N boolean array: True=视场外/被删除)
        注意：
        - 我们不改变 labels 的索引顺序，仅返回 mask；若你的 pipeline 需要删除点再编码，则也可返回被保留点。
        """
        points = res["lidar"]["points_cp"]
        labels = res["lidar"]["annotations"]["point_sem_labels"]
        N = points.shape[0]
        non_overlap_mask = np.zeros(N, dtype=bool)
        if self.drop_ratio <= 0:
            return points, labels, non_overlap_mask

        if self.mode == "random":
            k = int(N * self.drop_ratio)
            idx = np.random.choice(N, size=k, replace=False)
            non_overlap_mask[idx] = True

        elif self.mode == "range":
            # 按距离（range）删除最远的一部分点（模拟远距无覆盖）
            ranges = np.linalg.norm(points[:, :3], axis=1)
            threshold = np.percentile(ranges, 100 * (1 - self.drop_ratio))
            non_overlap_mask[ranges >= threshold] = True

        elif self.mode == "azimuth_sector":
            # 删除一个方位扇区内的点（模拟某个相机视场缺失）
            # sector_params = {"center_deg": 0, "width_deg": 40}
            if sector_params is None:
                sector_params = {"center_deg": 0, "width_deg": 40}
            xy = points[:, :2]
            az = np.degrees(np.arctan2(xy[:,1], xy[:,0]))  # -180..180
            c = sector_params["center_deg"]
            w = sector_params["width_deg"] / 2.0
            low = c - w
            high = c + w
            # 考虑循环跨 -180/180 的情形：
            if low < -180 or high > 180:
                az_norm = (az + 360) % 360
                low_n = (low + 360) % 360
                high_n = (high + 360) % 360
                mask = (az_norm >= low_n) & (az_norm <= high_n)
            else:
                mask = (az >= low) & (az <= high)
            # 可能删除过多/过少，按比率裁剪：
            current_ratio = mask.sum() / N
            if current_ratio <= self.drop_ratio:
                non_overlap_mask[mask] = True
            else:
                idx_region = np.where(mask)[0]
                k = int(N * self.drop_ratio)
                choose = np.random.choice(idx_region, size=k, replace=False)
                non_overlap_mask[choose] = True

        elif self.mode == "clustered":
            # 删除若干聚类（简单实现：用 kmeans /随机seed点做半径删除）
            # 这里用随机 seed +半径 r 的方法
            num_clusters = max(1, int(self.drop_ratio * 5))  # heuristic
            N_remove = int(N * self.drop_ratio)
            removed = set()
            tries = 0
            while len(removed) < N_remove and tries < 20:
                tries += 1
                seed = np.random.randint(0, N)
                r = np.random.uniform(1.0, 5.0)  # meters, 可调
                d2 = np.sum((points[:, :3] - points[seed, :3])**2, axis=1)
                idx = np.where(d2 <= r*r)[0]
                for i in idx:
                    removed.add(int(i))
                if len(removed) >= N_remove:
                    break
            idxs = np.array(list(removed))[:N_remove]
            non_overlap_mask[idxs] = True

        else:
            raise ValueError("Unknown mode")

        # kept points:
        mask_near = ~non_overlap_mask
        res["mask_near"] = mask_near
        return res, info
