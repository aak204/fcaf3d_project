"""Вспомогательные функции и слои для PointNet++."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


# --- Базовые операции ---

def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет попарные квадраты евклидовых расстояний между двумя наборами точек.
    Input:
        src: [B, N, C] tensor
        dst: [B, M, C] tensor
    Output:
        dist: [B, N, M] tensor
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # B, N, M
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    # Иногда могут возникать небольшие отрицательные значения из-за точности float
    return torch.clamp(dist, min=0)


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Индексирует тензор `points` с помощью тензора индексов `idx`.
    Input:
        points: [B, N, C] or [B, N, D]
        idx: [B, S] or [B, S, K] tensor, где S - количество новых точек, K - количество соседей
    Return:
        new_points: [B, S, C] or [B, S, K, D] tensor
    """
    device = points.device
    B = points.shape[0]
    N = points.shape[1]
    D_feat = points.shape[-1]

    if N == 0:  # Обработка пустого облака
        shape_out = list(idx.shape) + [D_feat]
        return torch.empty(shape_out, device=device, dtype=points.dtype)

    # Убедимся, что индексы в допустимом диапазоне
    idx_clamped = torch.clamp(idx, 0, N - 1)

    if idx.dim() == 2:  # idx: [B, S]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).unsqueeze(1).expand(-1, idx.shape[1])
        return points[batch_indices, idx_clamped, :]  # [B, S, D]
    elif idx.dim() == 3:  # idx: [B, S, K]
        S, K = idx.shape[1], idx.shape[2]
        batch_indices = torch.arange(B, dtype=torch.long, device=device).view(B, 1, 1).expand(-1, S, K)
        return points[batch_indices, idx_clamped, :]  # [B, S, K, D]
    else:
        raise ValueError(f"Unsupported index dimension: {idx.dim()}")


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Использует итеративный Farthest Point Sampling для выбора подмножества точек.
    Input:
        xyz: [B, N, 3] tensor, координаты точек
        npoint: int, количество точек для сэмплирования
    Return:
        centroids: [B, npoint] tensor, индексы выбранных точек
    """
    device = xyz.device
    B, N, C = xyz.shape
    if N == 0:
        return torch.empty(B, 0, dtype=torch.long, device=device)

    npoint = min(npoint, N)  # Нельзя выбрать больше точек, чем есть
    if npoint <= 0:
        return torch.empty(B, 0, dtype=torch.long, device=device)

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e18  # Расстояние до ближайшего выбранного центроида

    # Выбираем случайную стартовую точку для каждого элемента батча
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest  # Добавляем самую дальнюю точку
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # Координаты новой точки
        dist = torch.sum((xyz - centroid) ** 2, -1)  # Квадрат расстояния до новой точки
        mask = dist < distance  # Обновляем минимальное расстояние
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # Находим самую удаленную от всех выбранных

    return centroids


def query_ball_point(
        radius: float,
        nsample: int,
        xyz: torch.Tensor,
        new_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Находит всех соседей в пределах заданного радиуса для каждой точки из new_xyz.
    Input:
        radius: float, радиус поиска
        nsample: int, максимальное количество соседей для выбора
        xyz: [B, N, 3], все точки
        new_xyz: [B, S, 3], точки, для которых ищутся соседи
    Return:
        group_idx: [B, S, nsample] tensor, индексы соседей для каждой точки в new_xyz
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape

    if N == 0:  # Если исходное облако пустое
        return torch.empty(B, S, 0, dtype=torch.long, device=device)

    # Нельзя выбрать больше соседей, чем есть точек
    nsample = min(nsample, N)
    if nsample <= 0:
        return torch.empty(B, S, 0, dtype=torch.long, device=device)

    group_idx = torch.arange(N, dtype=torch.long, device=device).view(1, 1, N).repeat(B, S, 1)
    sqrdists = square_distance(new_xyz, xyz)  # [B, S, N]
    # Помечаем точки вне радиуса индексом N (невалидный индекс)
    mask_outside = sqrdists > radius ** 2
    group_idx[mask_outside] = N
    # Сортируем по расстоянию (ближайшие сначала), берем первые nsample
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # Если для какой-то точки не нашлось nsample соседей в радиусе,
    # ее индексы будут N. Заменяем их на индекс самой точки (или первого соседа).
    # Это предотвращает ошибки при последующем index_points.
    first_valid_idx = group_idx[:, :, 0].unsqueeze(-1).repeat(1, 1, nsample)
    mask_invalid = group_idx >= N
    group_idx[mask_invalid] = first_valid_idx[mask_invalid]

    # Убедимся, что индексы валидны
    return torch.clamp(group_idx, 0, N - 1)


# --- Слои PointNet++ ---

class PointNetSetAbstraction(nn.Module):
    """Слой Set Abstraction из PointNet++."""

    def __init__(
            self,
            npoint: int,
            radius: float,
            nsample: int,
            in_channel: int,
            mlp: List[int],
            group_all: bool
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel  # Включает 3 для XYZ + фичи

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.last_channel_out = last_channel

    def _sample_and_group(
            self, npoint: int, radius: float, nsample: int, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Выполняет FPS и Ball Query."""
        B, N, C = xyz.shape
        S = npoint

        if N == 0:  # Обработка пустого входа
            empty_xyz = torch.empty(B, S, 3, device=xyz.device)
            feat_dim = points.shape[-1] if points is not None else 0
            empty_points_grouped = torch.empty(B, S, nsample, 3 + feat_dim, device=xyz.device)
            return empty_xyz, empty_points_grouped

        # Сэмплирование центроидов
        fps_idx = farthest_point_sample(xyz, npoint)  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]

        # Группировка соседей
        idx = query_ball_point(radius, nsample, xyz, new_xyz)  # [B, npoint, nsample]
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, nsample, 3]

        # Нормализация координат соседей относительно центроида
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)  # [B, npoint, nsample, 3]

        # Группировка фичей (если есть)
        if points is not None:
            grouped_points = index_points(points, idx)  # [B, npoint, nsample, D]
            # Конкатенация нормализованных координат и фичей
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)  # [B, npoint, nsample, 3+D]
        else:
            new_points = grouped_xyz_norm  # [B, npoint, nsample, 3]

        return new_xyz, new_points

    def _sample_and_group_all(
            self, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Группирует все точки."""
        device = xyz.device
        B, N, C = xyz.shape
        # Центроид - начало координат
        new_xyz = torch.zeros(B, 1, C, device=device)
        # Группируем все точки
        grouped_xyz = xyz.view(B, 1, N, C)  # [B, 1, N, 3]
        if points is not None:
            # Убедимся, что points имеет формат [B, N, D]
            if points.shape[1] != N:
                # Если формат [B, D, N], транспонируем
                if points.shape[2] == N:
                    points = points.permute(0, 2, 1)
                else:
                    raise ValueError(f"Несовместимые размеры points {points.shape} и xyz {xyz.shape}")
            new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)  # [B, 1, N, 3+D]
        else:
            new_points = grouped_xyz  # [B, 1, N, 3]
        return new_xyz, new_points

    def forward(
            self, xyz: torch.Tensor, points: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz (torch.Tensor): Координаты точек [B, N, 3].
            points (Optional[torch.Tensor]): Фичи точек [B, N, D].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - new_xyz: Координаты новых точек [B, npoint, 3].
                - new_points: Новые фичи [B, mlp[-1], npoint].
        """
        if xyz.shape[-1] != 3:
            raise ValueError(f"Ожидалось xyz с последней размерностью 3, получено {xyz.shape[-1]}")

        # Проверка консистентности размеров xyz и points
        if points is not None and points.shape[1] != xyz.shape[1]:
            # Если формат [B, D, N], транспонируем
            if points.shape[2] == xyz.shape[1]:
                points = points.permute(0, 2, 1)
            else:
                raise ValueError(f"Несовместимые размеры points {points.shape} и xyz {xyz.shape}")

        # Ожидаемое количество каналов на входе MLP = 3 (от grouped_xyz_norm) + D (фичи)
        feat_dim = points.shape[-1] if points is not None else 0
        expected_in_ch_mlp = 3 + feat_dim

        # Сэмплирование и группировка
        if self.group_all:
            new_xyz, new_points_grouped = self._sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points_grouped = self._sample_and_group(
                self.npoint, self.radius, self.nsample, xyz, points
            )
        # new_points_grouped: [B, S, K, 3+D]

        # Проверка размерности после группировки
        if new_points_grouped.shape[-1] != expected_in_ch_mlp:
            if new_points_grouped.shape[-1] > expected_in_ch_mlp:
                new_points_grouped = new_points_grouped[..., :expected_in_ch_mlp]
            else:  # Дополняем нулями, если не хватает (например, points был None)
                pad_size = expected_in_ch_mlp - new_points_grouped.shape[-1]
                padding = torch.zeros(
                    *new_points_grouped.shape[:-1], pad_size,
                    device=new_points_grouped.device, dtype=new_points_grouped.dtype
                )
                new_points_grouped = torch.cat([new_points_grouped, padding], dim=-1)

        # Транспонируем для Conv2d: [B, 3+D, nsample, npoint]
        new_points_grouped = new_points_grouped.permute(0, 3, 2, 1)

        # Проверка каналов перед MLP
        if new_points_grouped.shape[1] != self.mlp_convs[0].in_channels:
            raise RuntimeError(
                f"SA Layer {self.npoint}: Channel mismatch before MLP. "
                f"Expected {self.mlp_convs[0].in_channels}, got {new_points_grouped.shape[1]}"
            )

        # Применяем MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points_grouped = F.relu(bn(conv(new_points_grouped)))

        # Макс-пулинг по соседям: [B, mlp[-1], npoint]
        new_points_pooled = torch.max(new_points_grouped, 2)[0]

        return new_xyz, new_points_pooled


class PointNetFeaturePropagation(nn.Module):
    """Слой Feature Propagation из PointNet++."""

    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel  # Фичи с skip connection + интерполированные фичи
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.last_channel_out = last_channel

    def forward(
            self,
            xyz1: torch.Tensor,
            xyz2: torch.Tensor,
            points1: Optional[torch.Tensor],
            points2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            xyz1 (torch.Tensor): Координаты точек на детальном уровне [B, N, 3].
            xyz2 (torch.Tensor): Координаты точек на грубом уровне [B, S, 3].
            points1 (Optional[torch.Tensor]): Фичи с детального уровня SA [B, C1, N].
                                             Может быть None для самого первого FP слоя.
            points2 (torch.Tensor): Фичи с предыдущего (грубого) уровня FP/SA [B, C2, S].

        Returns:
            torch.Tensor: Новые фичи для детального уровня [B, mlp[-1], N].
        """
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        if points1 is not None:
            points1 = points1.contiguous()
        points2 = points2.contiguous()

        B, N, _ = xyz1.shape
        _, S, _ = xyz2.shape
        C2 = points2.shape[1]  # Количество фичей на грубом уровне

        # Интерполяция фичей с грубого уровня на детальный
        if S == 1:  # Особый случай для group_all
            interpolated_points = points2.repeat(1, 1, N)  # [B, C2, N]
        elif S == 0:  # Если грубый уровень пуст
            interpolated_points = torch.zeros(B, C2, N, device=xyz1.device, dtype=points2.dtype)
        elif N == 0:  # Если детальный уровень пуст
            interpolated_points = torch.empty(B, C2, 0, device=xyz1.device, dtype=points2.dtype)
        else:
            # Находим 3 ближайших соседа на грубом уровне для каждой точки детального уровня
            dists = square_distance(xyz1, xyz2)  # [B, N, S]
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            # Обратные расстояния для весов (добавляем epsilon для стабильности)
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm  # [B, N, 3] - веса интерполяции

            # Интерполируем фичи points2 [B, C2, S] -> [B, S, C2]
            # index_points(points2.permute(0, 2, 1), idx) -> [B, N, 3, C2]
            interpolated_points = torch.sum(
                index_points(points2.permute(0, 2, 1), idx) * weight.view(B, N, 3, 1),
                dim=2  # Суммируем по 3 соседям с весами
            )  # [B, N, C2]
            interpolated_points = interpolated_points.permute(0, 2, 1)  # [B, C2, N]

        # Конкатенация с фичами skip connection (points1)
        if points1 is not None:
            # Проверка согласованности N
            if points1.shape[2] != interpolated_points.shape[2]:
                if interpolated_points.shape[2] == 0 and points1.shape[2] > 0:
                    points1 = torch.empty(B, points1.shape[1], 0, device=points1.device, dtype=points1.dtype)
                elif points1.shape[2] == 0 and interpolated_points.shape[2] > 0:
                    points1 = torch.empty(B, 0, interpolated_points.shape[2], device=points1.device,
                                          dtype=points1.dtype)
                elif points1.shape[2] != 0 and interpolated_points.shape[2] != 0:
                    raise ValueError(
                        f"Shape mismatch FP concat: points1 {points1.shape}, interp {interpolated_points.shape}")

            if points1.shape[1] > 0:  # Если есть фичи на детальном уровне
                new_points = torch.cat([points1, interpolated_points], dim=1)  # [B, C1+C2, N]
            else:  # Если фичей не было (например, points1 был пустым тензором)
                new_points = interpolated_points
        else:  # Если points1 был None (самый первый FP слой)
            new_points = interpolated_points

        # Проверка каналов перед MLP
        if new_points.shape[1] != self.mlp_convs[0].in_channels:
            raise RuntimeError(
                f"FP Layer: Channel mismatch before MLP. "
                f"Expected {self.mlp_convs[0].in_channels}, got {new_points.shape[1]}"
            )

        # Применяем MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points
