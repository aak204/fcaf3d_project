"""Реализация IoU-based лоссов для регрессии bounding box'ов."""

import torch
from utils.metrics import compute_iou_3d
from configs import fcaf3d_config as cfg


def calculate_diou_3d_loss(
        preds_norm: torch.Tensor,
        targets_norm: torch.Tensor,
        eps: float = cfg.DIOU_LOSS_EPS
) -> torch.Tensor:
    """
    Вычисляет 3D DIoU Loss для axis-aligned боксов.
    Ожидает боксы в НОРМАЛИЗОВАННЫХ координатах [0, 1].

    Args:
        preds_norm (torch.Tensor): Предсказанные боксы [N, 6] (cx, cy, cz, w, h, l).
        targets_norm (torch.Tensor): Целевые боксы [N, 6] (cx, cy, cz, w, h, l).
        eps (float): Малое значение для стабильности.

    Returns:
        torch.Tensor: Значение DIoU loss для каждого бокса [N].
    """
    if preds_norm.numel() == 0 or targets_norm.numel() == 0:
        return torch.tensor(0.0, device=preds_norm.device, requires_grad=preds_norm.requires_grad)

    # 1. Вычисляем стандартный IoU (используя нормализованные боксы)
    # Убедимся, что размеры положительные
    preds_size = torch.relu(preds_norm[:, 3:6]) + eps
    targets_size = torch.relu(targets_norm[:, 3:6]) + eps
    preds_box = torch.cat([preds_norm[:, :3], preds_size], dim=1)
    targets_box = torch.cat([targets_norm[:, :3], targets_size], dim=1)

    iou = compute_iou_3d(preds_box, targets_box)  # [N]

    # 2. Вычисляем штраф за расстояние между центрами (в нормализованных координатах)
    center_dist_sq = torch.sum((preds_box[:, :3] - targets_box[:, :3]) ** 2, dim=-1)  # [N]

    # 3. Вычисляем квадрат диагонали минимального охватывающего бокса (в норм. коорд.)
    preds_min = preds_box[:, :3] - preds_box[:, 3:6] / 2
    preds_max = preds_box[:, :3] + preds_box[:, 3:6] / 2
    targets_min = targets_box[:, :3] - targets_box[:, 3:6] / 2
    targets_max = targets_box[:, :3] + targets_box[:, 3:6] / 2

    min_coords = torch.min(preds_min, targets_min)
    max_coords = torch.max(preds_max, targets_max)
    # Квадрат диагонали охватывающего бокса
    enclosing_diag_sq = torch.sum((max_coords - min_coords) ** 2, dim=-1)  # [N]

    # Штраф DIoU = квадрат расстояния / квадрат диагонали
    diou_penalty = center_dist_sq / (enclosing_diag_sq + eps)  # Добавляем eps для стабильности

    # DIoU Loss = 1 - IoU + Penalty
    diou_loss = 1.0 - iou + diou_penalty

    # Ограничиваем лосс >= 0
    return torch.clamp(diou_loss, min=0.0)
