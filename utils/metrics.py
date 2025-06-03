"""Функции для вычисления метрик (IoU, NMS)."""

import torch


def compute_iou_3d(pred_bbox: torch.Tensor, gt_bbox: torch.Tensor) -> torch.Tensor:
    """
    Вычисляет axis-aligned IoU для 3D боксов.
    Args:
        pred_bbox (torch.Tensor): Предсказанные боксы [..., N, 6] (cx, cy, cz, w, h, l).
        gt_bbox (torch.Tensor): Ground truth боксы [..., M, 6] (cx, cy, cz, w, h, l).
                                Всегда возвращает результат с размерностями, соответствующими N и M.
    Returns:
        torch.Tensor: Значения IoU [..., N, M].
    """
    # Проверка и обработка NaN/Inf
    pred_bbox = torch.nan_to_num(pred_bbox, nan=0.0, posinf=1e6, neginf=-1e6)
    gt_bbox = torch.nan_to_num(gt_bbox, nan=0.0, posinf=1e6, neginf=-1e6)

    # --- ИСПРАВЛЕННАЯ ЛОГИКА BROADCASTING ---
    # Цель: получить результат [..., N, M]
    # Добавляем необходимые измерения для broadcast
    # pred: [..., N, 6] -> [..., N, 1, 6]
    # gt:   [..., M, 6] -> [..., 1, M, 6]
    pred_bbox_expanded = pred_bbox.unsqueeze(-2)
    gt_bbox_expanded = gt_bbox.unsqueeze(-3)
    # --- КОНЕЦ ИСПРАВЛЕННОЙ ЛОГИКИ ---

    # Берем только геометрию [..., 6]
    pred_geom = pred_bbox_expanded[..., :6]
    gt_geom = gt_bbox_expanded[..., :6]

    # Обеспечиваем положительные размеры (используем relu для простоты)
    pred_size = torch.relu(pred_geom[..., 3:6])
    gt_size = torch.relu(gt_geom[..., 3:6])

    # Вычисляем min/max координаты углов (с учетом broadcast)
    pred_min = pred_geom[..., :3] - pred_size / 2
    pred_max = pred_geom[..., :3] + pred_size / 2
    gt_min = gt_geom[..., :3] - gt_size / 2
    gt_max = gt_geom[..., :3] + gt_size / 2

    # Пересечение
    intersect_min = torch.max(pred_min, gt_min)
    intersect_max = torch.min(pred_max, gt_max)
    intersect_size = torch.clamp(intersect_max - intersect_min, min=0)
    # Объем пересечения будет иметь размерность [..., N, M]
    intersect_volume = torch.prod(intersect_size, dim=-1)

    # Объемы исходных боксов (с учетом broadcast)
    pred_volume = torch.prod(pred_size, dim=-1) # [..., N, 1]
    gt_volume = torch.prod(gt_size, dim=-1)   # [..., 1, M]

    # Объединение
    union_volume = pred_volume + gt_volume - intersect_volume # [..., N, M]

    # IoU (избегаем деления на ноль)
    iou = torch.where(
        union_volume > 1e-6,
        intersect_volume / union_volume,
        torch.zeros_like(union_volume)
    )
    # Ограничиваем IoU диапазоном [0, 1]
    return torch.clamp(iou, min=0.0, max=1.0) # Возвращает [..., N, M]

# --- Функция nms_3d остается без изменений ---
def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    # ... (код nms_3d без изменений) ...
    if boxes.numel() == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    scores_sorted, indices = torch.sort(scores, descending=True)
    boxes_sorted = boxes[indices]
    keep = []
    suppressed = torch.zeros(boxes_sorted.shape[0], dtype=torch.bool, device=boxes.device)
    for i in range(boxes_sorted.shape[0]):
        if suppressed[i]:
            continue
        keep.append(indices[i].item())
        current_box = boxes_sorted[i].unsqueeze(0) # [1, 6]
        remaining_indices_mask = ~suppressed[(i + 1):]
        remaining_indices_local = torch.where(remaining_indices_mask)[0]
        if remaining_indices_local.numel() == 0:
            break
        remaining_indices_global = remaining_indices_local + (i + 1)
        remaining_boxes = boxes_sorted[remaining_indices_global] # [M, 6]
        # compute_iou_3d([1, 6], [M, 6]) -> [1, M] -> squeeze -> [M]
        iou = compute_iou_3d(current_box, remaining_boxes).squeeze(0)
        suppress_mask_remaining = iou > iou_threshold
        suppressed[remaining_indices_global[suppress_mask_remaining]] = True
    return torch.tensor(keep, dtype=torch.long, device=boxes.device) if keep else torch.tensor([], dtype=torch.long, device=boxes.device)