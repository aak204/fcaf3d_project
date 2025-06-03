"""Различные вспомогательные утилиты."""

import torch
from typing import Optional


def print_tensor_stats(
        tensor: Optional[torch.Tensor],
        name: str,
        detach: bool = True,
        indent: str = "  "
):
    """Выводит статистику по тензору для дебага."""
    print(f"{indent}{name}: ", end="")
    if tensor is None:
        print("None")
        return
    if not isinstance(tensor, torch.Tensor):
        print(f"Not a Tensor (type: {type(tensor)})")
        return
    if tensor.numel() == 0:
        print(f"Empty Tensor (shape: {tensor.shape})")
        return

    t = tensor.detach().float() if detach and tensor.requires_grad else tensor.float()
    shape = tuple(t.shape)
    device = tensor.device
    dtype = tensor.dtype
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    nan_count = torch.isnan(t).sum().item() if has_nan else 0
    inf_count = torch.isinf(t).sum().item() if has_inf else 0
    t_valid = t[~(torch.isnan(t) | torch.isinf(t))]

    if t_valid.numel() > 0:
        t_min = torch.min(t_valid).item()
        t_max = torch.max(t_valid).item()
        t_mean = torch.mean(t_valid).item()
        try:
            # median требует CPU для некоторых версий/типов
            t_median = torch.median(t_valid.cpu()).values.item()
        except Exception:
            t_median = float('nan')
        t_std = torch.std(t_valid).item()
        print(f"shape={shape}, dtype={dtype}, device={device}, "
              f"min={t_min:.4f}, max={t_max:.4f}, mean={t_mean:.4f}, "
              f"median={t_median:.4f}, std={t_std:.4f}", end="")
    else:
        print(f"shape={shape}, dtype={dtype}, device={device}, All NaN/Inf!", end="")

    if has_nan or has_inf:
        print(f", NaN={nan_count}, Inf={inf_count}", end="")
    if detach and hasattr(tensor, 'requires_grad') and tensor.requires_grad:
        print(f", grad={tensor.requires_grad}", end="")
    print()
