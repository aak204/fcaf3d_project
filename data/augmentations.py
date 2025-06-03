"""Функции аугментации для облаков точек и bounding box'ов."""

import numpy as np
import random
from typing import Tuple
from configs.fcaf3d_config import USE_NORMALS_AS_FEATURES


def rotate_points_along_z(points: np.ndarray, angle: float) -> np.ndarray:
    """Поворачивает точки [N, D] вокруг оси Z. Первые 3 столбца - XYZ."""
    if points.shape[0] == 0:
        return points
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    points_rotated_xyz = points[:, :3] @ rotation_matrix.T

    # Если есть нормали
    if points.shape[1] >= 6 and USE_NORMALS_AS_FEATURES:
        points_rotated_normals = points[:, 3:6] @ rotation_matrix.T
        other_features = points[:, 6:]
        return np.hstack((points_rotated_xyz, points_rotated_normals, other_features))
    elif points.shape[1] > 3:  # Если есть другие фичи, но не нормали
        return np.hstack((points_rotated_xyz, points[:, 3:]))
    else:  # Только XYZ
        return points_rotated_xyz


def rotate_gt_bboxes_along_z(bboxes: np.ndarray, angle: float) -> np.ndarray:
    """Поворачивает GT боксы [M, >=6] (cx, cy, cz, w, h, l, ...) вокруг оси Z."""
    if bboxes.shape[0] == 0:
        return bboxes
    # Поворачиваем только центры XY
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    centers_xy = bboxes[:, :2]
    rotated_centers_xy = centers_xy @ rotation_matrix.T

    bboxes_rotated = bboxes.copy()
    bboxes_rotated[:, :2] = rotated_centers_xy
    # Размеры и Z-центр для axis-aligned боксов не меняются при повороте вокруг Z
    return bboxes_rotated


def global_scale(
        points: np.ndarray,
        bboxes: np.ndarray,
        scale_range: Tuple[float, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Масштабирует точки (XYZ) и bbox (cx, cy, cz, w, h, l)."""
    if scale_range[0] == scale_range[1]:
        return points, bboxes
    scale = np.random.uniform(scale_range[0], scale_range[1])

    points_scaled = points.copy()
    points_scaled[:, :3] *= scale  # Масштабируем только XYZ

    bboxes_scaled = bboxes.copy()
    if bboxes.shape[0] > 0:
        bboxes_scaled[:, :6] *= scale  # Масштабируем центр и размеры

    return points_scaled, bboxes_scaled


def random_flip_along_axis(
        points: np.ndarray,
        bboxes: np.ndarray,
        axis: int = 0,  # 0 for X, 1 for Y
        flip_prob: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Случайно отражает точки и боксы вдоль указанной оси."""
    if random.random() < flip_prob:
        points_flipped = points.copy()
        points_flipped[:, axis] = -points_flipped[:, axis]  # Отражаем координату

        # Если есть нормали и отражаем по X (axis=0), отражаем normal_x (индекс 3)
        if axis == 0 and points.shape[1] >= 6 and USE_NORMALS_AS_FEATURES:
            points_flipped[:, 3] = -points_flipped[:, 3]
        # Если есть нормали и отражаем по Y (axis=1), отражаем normal_y (индекс 4)
        elif axis == 1 and points.shape[1] >= 6 and USE_NORMALS_AS_FEATURES:
            points_flipped[:, 4] = -points_flipped[:, 4]

        bboxes_flipped = bboxes.copy()
        if bboxes.shape[0] > 0:
            bboxes_flipped[:, axis] = -bboxes_flipped[:, axis]  # Отражаем центр

        return points_flipped, bboxes_flipped
    else:
        return points, bboxes


def point_cloud_noise(
        points: np.ndarray,
        noise_std: float,
        pc_range: np.ndarray  # Для масштабирования шума
) -> np.ndarray:
    """Добавляет Гауссов шум к НЕнормализованным координатам XYZ точек."""
    if points.shape[0] == 0 or noise_std <= 0:
        return points

    # Шум добавляем к НЕнормализованным координатам,
    # но std задан в масштабе нормализованных [0, 1]
    # Масштабируем std шума обратно к реальным координатам
    range_xyz = pc_range[3:] - pc_range[:3]
    noise_std_real = noise_std * range_xyz
    noise = np.random.normal(0, noise_std_real, size=(points.shape[0], 3))

    points_noisy = points.copy()
    points_noisy[:, :3] += noise
    return points_noisy


def random_point_dropout(points: np.ndarray, dropout_ratio: float) -> np.ndarray:
    """Случайно удаляет часть точек."""
    if points.shape[0] == 0 or dropout_ratio <= 0:
        return points
    n_points = points.shape[0]
    n_keep = int(n_points * (1.0 - dropout_ratio))
    if n_keep <= 0:
        return np.empty((0, points.shape[1]), dtype=points.dtype)

    # Используем быстрый способ выбора индексов
    keep_indices = np.random.choice(n_points, n_keep, replace=False)
    return points[keep_indices]
