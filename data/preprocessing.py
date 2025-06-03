"""Функции для предобработки облаков точек и bounding box'ов."""

import numpy as np
import torch
from typing import List, Union


def normalize_pc(points: np.ndarray, pc_range: List[float]) -> np.ndarray:
    """Нормализует координаты XYZ в [0, 1]. Остальные фичи не трогает."""
    if points.shape[0] == 0:
        return points
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    coords = points[:, :3].copy()
    range_x = max(x_max - x_min, 1e-6)
    range_y = max(y_max - y_min, 1e-6)
    range_z = max(z_max - z_min, 1e-6)
    coords[:, 0] = (coords[:, 0] - x_min) / range_x
    coords[:, 1] = (coords[:, 1] - y_min) / range_y
    coords[:, 2] = (coords[:, 2] - z_min) / range_z
    coords = np.clip(coords, 0.0, 1.0)

    if points.shape[1] > 3:
        return np.hstack((coords, points[:, 3:]))
    else:
        return coords


def normalize_bbox(bbox: np.ndarray, pc_range: List[float]) -> np.ndarray:
    """Нормализует bbox [6,] (cx, cy, cz, w, h, l) в диапазон [0, 1]."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    bbox_out = bbox.copy()
    range_x = max(x_max - x_min, 1e-6)
    range_y = max(y_max - y_min, 1e-6)
    range_z = max(z_max - z_min, 1e-6)

    bbox_out[0] = (bbox[0] - x_min) / range_x
    bbox_out[1] = (bbox[1] - y_min) / range_y
    bbox_out[2] = (bbox[2] - z_min) / range_z
    bbox_out[3] = bbox[3] / range_x
    bbox_out[4] = bbox[4] / range_y
    bbox_out[5] = bbox[5] / range_z

    # Центр должен быть в [0, 1], размеры могут быть > 1, если объект больше pc_range
    bbox_out[:3] = np.clip(bbox_out[:3], 0.0, 1.0)
    return bbox_out


def denormalize_pc_coords(
        points_norm_xyz: Union[np.ndarray, torch.Tensor],
        pc_range: List[float]
) -> Union[np.ndarray, torch.Tensor]:
    """Денормализует координаты XYZ из [0, 1] в исходный диапазон."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    is_tensor = isinstance(points_norm_xyz, torch.Tensor)
    denorm = points_norm_xyz.clone() if is_tensor else points_norm_xyz.copy()
    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = z_max - z_min

    denorm[..., 0] = denorm[..., 0] * range_x + x_min
    denorm[..., 1] = denorm[..., 1] * range_y + y_min
    denorm[..., 2] = denorm[..., 2] * range_z + z_min
    return denorm


def denormalize_bbox(
        bbox_norm: Union[np.ndarray, torch.Tensor],
        pc_range: List[float]
) -> Union[np.ndarray, torch.Tensor]:
    """Денормализует bbox [..., 6] (cx, cy, cz, w, h, l) из [0, 1]."""
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    range_x = x_max - x_min
    range_y = y_max - y_min
    range_z = z_max - z_min
    is_tensor = isinstance(bbox_norm, torch.Tensor)
    denorm = bbox_norm.clone() if is_tensor else bbox_norm.copy()

    denorm[..., 0] = denorm[..., 0] * range_x + x_min
    denorm[..., 1] = denorm[..., 1] * range_y + y_min
    denorm[..., 2] = denorm[..., 2] * range_z + z_min
    denorm[..., 3] = denorm[..., 3] * range_x
    denorm[..., 4] = denorm[..., 4] * range_y
    denorm[..., 5] = denorm[..., 5] * range_z

    # Обеспечиваем минимальный размер
    min_size = 1e-4
    if is_tensor:
        denorm[..., 3:6] = torch.clamp(denorm[..., 3:6], min=min_size)
    else:
        denorm[..., 3:6] = np.maximum(denorm[..., 3:6], min_size)
    return denorm


def clip_points_to_range(points: np.ndarray, pc_range: List[float]) -> np.ndarray:
    """Обрезает точки, выходящие за пределы pc_range."""
    if points.shape[0] == 0:
        return points
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mask = (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]
