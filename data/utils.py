"""Вспомогательные функции для работы с данными, включая сэмплирование."""

import numpy as np
from numba import jit, prange
import open3d as o3d

# --- Функции сэмплирования ---

def random_sampling(n_points: int, npoint: int) -> np.ndarray:
    """Выполняет случайное сэмплирование без повторений."""
    if n_points == 0:
        return np.empty(0, dtype=np.int32)
    if n_points <= npoint:
        return np.arange(n_points, dtype=np.int32)
    # Более быстрый вариант для больших массивов
    indices = np.random.permutation(n_points)[:npoint]
    return indices.astype(np.int32)


@jit(nopython=True, parallel=True)
def _square_distance_numba(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Вспомогательная Numba функция для FPS."""
    N = src.shape[0]
    M = dst.shape[0]
    dist = np.empty((N, M), dtype=src.dtype)
    for i in prange(N):
        for j in range(M):
            d = 0.0
            for k in range(src.shape[1]):
                diff = src[i, k] - dst[j, k]
                d += diff * diff
            dist[i, j] = d
    return dist


def farthest_point_sampling_numba(points_xyz: np.ndarray, npoint: int) -> np.ndarray:
    """
    Farthest Point Sampling через Open3D.

    Args:
        points_xyz (np.ndarray): Входное облако точек формы (N, 3).
        npoint (int): Число точек для выборки.

    Returns:
        np.ndarray: Индексы выбранных точек в оригинальном массиве.
    """
    # 1) Создаём PointCloud и заполняем координаты
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)

    # 2) Down-sampling: выбираем npoint самых «удалённых» точек
    down_pcd = pcd.farthest_point_down_sample(npoint)  # :contentReference[oaicite:0]{index=0}

    # 3) Получаем координаты выбранных точек
    sampled_pts = np.asarray(down_pcd.points)

    # 4) Чтобы вернуть индексы в исходном массиве, делаем быстрый поиск ближайших соседей
    tree = o3d.geometry.KDTreeFlann(pcd)
    indices = []
    for pt in sampled_pts:
        _, idx, _ = tree.search_knn_vector_3d(pt, 1)
        indices.append(idx[0])

    return np.array(indices, dtype=np.int32)


def downsample_point_cloud(
        points: np.ndarray,
        method: str,
        max_points: int,
        voxel_size: float = 0.0  # Пока не используется
) -> np.ndarray:
    """
    Уменьшает количество точек в облаке.

    Args:
        points (np.ndarray): Входное облако точек [N, D].
        method (str): Метод ('random', 'fps_numba').
        max_points (int): Целевое количество точек.
        voxel_size (float): Размер вокселя (для voxel downsampling, пока не реализовано).

    Returns:
        np.ndarray: Облако точек с уменьшенным количеством точек [max_points, D].
    """
    N, D = points.shape
    if N <= max_points or N == 0:
        return points

    indices = None
    if method == 'random':
        indices = random_sampling(N, max_points)
    elif method == 'fps_numba':
        # FPS работает только с XYZ
        indices = farthest_point_sampling_numba(points[:, :3].astype(np.float64), max_points)
    else:
        print(f"Warning: Unknown downsampling method '{method}'. Using 'fps_numba'.")
        indices = farthest_point_sampling_numba(points[:, :3].astype(np.float64), max_points)

    if indices is not None and len(indices) > 0:
        # Проверка валидности индексов (Numba иногда может вернуть мусор при ошибках)
        valid_indices = indices[(indices >= 0) & (indices < N)]
        if len(valid_indices) == max_points:
            return points[valid_indices]
        elif len(valid_indices) > 0:
            print(
                f"Warning: Sampler returned {len(valid_indices)} indices, expected {max_points}. Using available indices.")
            # Можно дополнить случайными, если нужно ровно max_points
            if len(valid_indices) < max_points:
                remaining_needed = max_points - len(valid_indices)
                # Выбираем дополнительные из тех, что еще не выбраны
                available_indices = np.setdiff1d(np.arange(N), valid_indices, assume_unique=True)
                if len(available_indices) >= remaining_needed:
                    extra_indices = np.random.choice(available_indices, remaining_needed, replace=False)
                else:  # Если не хватает уникальных, берем с повторениями из валидных
                    extra_indices = np.random.choice(valid_indices, remaining_needed, replace=True)
                final_indices = np.concatenate((valid_indices, extra_indices))
                return points[final_indices]
            else:  # Если вернулось больше (не должно быть для FPS/random)
                return points[valid_indices[:max_points]]

        else:  # Если sampler вернул пустые или невалидные индексы
            print(f"Warning: Sampler '{method}' returned invalid indices. Falling back to random sampling.")
            indices = random_sampling(N, max_points)
            return points[indices] if len(indices) > 0 else np.empty((0, D), dtype=points.dtype)
    else:  # Если метод не вернул индексы
        print(f"Warning: Sampler '{method}' failed. Falling back to random sampling.")
        indices = random_sampling(N, max_points)
        return points[indices] if len(indices) > 0 else np.empty((0, D), dtype=points.dtype)
