"""Реализация Focal Loss."""

import torch
import torch.nn.functional as F

from configs import fcaf3d_config as cfg  # Для alpha/gamma по умолчанию


def focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = cfg.FOCAL_LOSS_ALPHA,
        gamma: float = cfg.FOCAL_LOSS_GAMMA,
        reduction: str = 'sum'
) -> torch.Tensor:
    """
    Вычисляет Focal Loss.

    Args:
        inputs (torch.Tensor): Логиты предсказаний [B, C, N] или [N, C].
        targets (torch.Tensor): Целевые классы [B, N] или [N].
                                Ожидаются индексы классов (0 для фона).
        alpha (float): Вес для положительных классов.
        gamma (float): Фокусирующий параметр.
        reduction (str): Метод агрегации ('sum', 'mean', 'none').

    Returns:
        torch.Tensor: Значение Focal Loss.
    """
    if inputs.dim() > 2:
        # inputs: [B, C, N] -> [B*N, C]
        inputs_flat = inputs.permute(0, 2, 1).reshape(-1, inputs.shape[1])
        # targets: [B, N] -> [B*N]
        targets_flat = targets.reshape(-1)
    else:  # Уже плоские тензоры
        inputs_flat = inputs
        targets_flat = targets

    # Фильтруем невалидные таргеты (например, -1)
    valid_mask = targets_flat >= 0
    inputs_flat = inputs_flat[valid_mask]
    targets_flat = targets_flat[valid_mask]

    if inputs_flat.numel() == 0:
        # Возвращаем 0 с градиентом, если нет валидных таргетов
        return torch.tensor(0.0, device=inputs.device, requires_grad=inputs.requires_grad)

    # Стандартный Cross Entropy Loss
    ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')

    # Вероятности предсказанного класса
    p = torch.softmax(inputs_flat, dim=1)
    # Вероятности для истинного класса
    pt = p.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    # Стабильность: избегаем log(0)
    pt = torch.clamp(pt, 1e-7, 1.0 - 1e-7)

    # Модулирующий фактор (1 - pt)^gamma
    modulating_factor = (1.0 - pt) ** gamma

    # Веса классов alpha
    # alpha для FG классов (индекс > 0), (1-alpha) для BG класса (индекс 0)
    alpha_weight = torch.full_like(targets_flat, 1.0 - alpha, dtype=torch.float32)
    alpha_weight[targets_flat > 0] = alpha

    # Финальный Focal Loss
    focal_loss_val = alpha_weight * modulating_factor * ce_loss

    if reduction == 'sum':
        return focal_loss_val.sum()
    elif reduction == 'mean':
        # Важно: mean здесь считается по валидным элементам
        return focal_loss_val.mean()
    elif reduction == 'none':
        return focal_loss_val
    else:
        raise ValueError(f"Неподдерживаемый reduction: {reduction}")
