"""Основная функция потерь для FCAF3D."""
import traceback

import torch
import torch.nn.functional as F
from typing import Dict, Union, List
from configs import fcaf3d_config as cfg
from .focal_loss import focal_loss
from .iou_losses import calculate_diou_3d_loss
from .target_assigner import assign_targets_fcaf3d
from utils.misc import print_tensor_stats  # Для дебага


def compute_fcaf3d_loss(
        end_points: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        gt_bboxes_norm: torch.Tensor,
        num_objects: torch.Tensor,
        loss_weights: Dict = cfg.LOSS_WEIGHTS,
        num_fg_classes: int = cfg.NUM_FG_CLASSES,
        radius_scale: float = cfg.ASSIGNMENT_RADIUS_SCALE,
        min_radius: float = cfg.ASSIGNMENT_MIN_RADIUS,
        topk: int = cfg.ASSIGNMENT_TOPK,
        focal_alpha: float = cfg.FOCAL_LOSS_ALPHA,
        focal_gamma: float = cfg.FOCAL_LOSS_GAMMA,
        diou_eps: float = cfg.DIOU_LOSS_EPS,
        debug_loss: bool = cfg.DEBUG_LOSS_CALCULATION
) -> Dict[str, torch.Tensor]:
    """
    Вычисляет общую потерю для FCAF3D (с DIoU).

    Args:
        end_points (Dict): Выходы модели FCAF3D.
        gt_bboxes_norm (torch.Tensor): Нормализованные GT боксы [B, M_max, 7].
        num_objects (torch.Tensor): Количество GT объектов в батче [B,].
        loss_weights (Dict): Веса для разных компонентов потерь.
        ... (остальные параметры из конфига)

    Returns:
        Dict[str, torch.Tensor]: Словарь с компонентами потерь и общим лоссом.
    """
    # Извлекаем предсказания и координаты точек из выходов модели
    cls_preds_list = end_points.get('cls_preds')
    ctr_preds_list = end_points.get('ctr_preds')
    offset_preds_list = end_points.get('center_offset_preds')
    logsize_preds_list = end_points.get('size_log_preds')
    # Координаты точек для каждого уровня FP [xyz3, xyz2, xyz1, xyz0]
    points_coords_fp_list = end_points.get('fp_xyz')

    # Проверка наличия необходимых ключей
    required_keys = ['cls_preds', 'ctr_preds', 'center_offset_preds', 'size_log_preds', 'fp_xyz']
    if not all(k in end_points and end_points[k] is not None for k in required_keys):
        raise ValueError("Отсутствуют необходимые ключи в end_points для расчета потерь.")
    if len(points_coords_fp_list) != cfg.NUM_LEVELS:
        raise ValueError(f"Ожидалось {cfg.NUM_LEVELS} уровней FP в 'fp_xyz', получено {len(points_coords_fp_list)}")

    B = gt_bboxes_norm.shape[0]
    device = gt_bboxes_norm.device

    # 1. Назначение целей (Target Assignment)
    try:
        # points_coords_fp_list идет от грубого к детальному [xyz3, xyz2, xyz1, xyz0]
        cls_targets_list, ctr_targets_list, reg_targets_list, assign_masks_list = assign_targets_fcaf3d(
            points_coords_fp_list, gt_bboxes_norm, num_objects, num_fg_classes,
            radius_scale, min_radius, topk
        )
        # Списки таргетов также идут от грубого к детальному уровню FP
    except Exception as e:
        print(f"Ошибка во время назначения таргетов: {e}")
        traceback.print_exc()
        zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
        return {
            'total_loss': zero_loss, 'cls_loss': zero_loss.detach(),
            'ctr_loss': zero_loss.detach(), 'reg_loss': zero_loss.detach(),
            'num_assigned': torch.tensor(0, device=device)
        }

    # Инициализация суммарных потерь
    total_cls_loss = torch.tensor(0.0, device=device)
    total_ctr_loss = torch.tensor(0.0, device=device)
    total_reg_loss = torch.tensor(0.0, device=device)  # Для DIoU
    num_assigned_total = torch.tensor(0, dtype=torch.long, device=device)

    # 2. Расчет потерь для каждой предсказательной головы
    # Итерируемся по уровням, для которых есть головы (cfg.PREDICTION_HEAD_LEVELS)
    for head_idx, pred_level_idx in enumerate(cfg.PREDICTION_HEAD_LEVELS):
        # Получаем предсказания для текущей головы
        cls_preds = cls_preds_list[head_idx]  # [B, C, N_pred]
        ctr_preds = ctr_preds_list[head_idx]  # [B, 1, N_pred]
        offset_preds = offset_preds_list[head_idx]  # [B, 3, N_pred]
        logsize_preds = logsize_preds_list[head_idx]  # [B, 3, N_pred]

        # Получаем таргеты для соответствующего уровня FP
        # Индекс уровня FP (0=детальный, 1, 2, 3=грубый)
        fp_level_map_idx = cfg.NUM_LEVELS - 1 - pred_level_idx  # 3, 2, 1
        cls_targets = cls_targets_list[fp_level_map_idx]  # [B, N_target]
        ctr_targets = ctr_targets_list[fp_level_map_idx]  # [B, N_target]
        reg_targets = reg_targets_list[fp_level_map_idx]  # [B, N_target, 6]
        assign_masks = assign_masks_list[fp_level_map_idx]  # [B, N_target]
        points_coords = points_coords_fp_list[fp_level_map_idx]  # [B, N_target, 3]

        N_pred = cls_preds.shape[2]
        N_target = cls_targets.shape[1]
        if N_pred != N_target:
            print(f"КРИТИЧЕСКАЯ ОШИБКА: Несоответствие N в compute_loss! Голова {head_idx} (Уровень {pred_level_idx}) "
                  f"N_pred={N_pred} != N_target={N_target}. Пропуск уровня.")
            continue

        # --- Расчет компоненты Classification Loss (Focal Loss) ---
        # Считается по всем точкам уровня
        cls_loss_level = focal_loss(
            cls_preds, cls_targets, alpha=focal_alpha, gamma=focal_gamma, reduction='sum'
        )
        total_cls_loss += cls_loss_level

        # --- Расчет компонент Centerness и Regression Loss (только по назначенным точкам) ---
        assign_masks_flat = assign_masks.reshape(-1)  # [B*N_target]
        num_assigned_level = assign_masks_flat.sum().item()
        num_assigned_total += num_assigned_level

        if num_assigned_level > 0:
            # --- Centerness Loss (BCE) ---
            ctr_preds_flat = ctr_preds.permute(0, 2, 1).reshape(-1)[assign_masks_flat]  # [NumAssigned]
            ctr_targets_flat = ctr_targets.reshape(-1)[assign_masks_flat]  # [NumAssigned]
            ctr_loss_level = F.binary_cross_entropy_with_logits(
                ctr_preds_flat, ctr_targets_flat, reduction='sum'
            )
            total_ctr_loss += ctr_loss_level

            # --- Regression Loss (DIoU) ---
            # Извлекаем предсказания и таргеты для назначенных точек
            offset_preds_flat = offset_preds.permute(0, 2, 1).reshape(-1, 3)[assign_masks_flat]
            logsize_preds_flat = logsize_preds.permute(0, 2, 1).reshape(-1, 3)[assign_masks_flat]
            reg_targets_flat = reg_targets.reshape(-1, 6)[assign_masks_flat]
            offset_targets_flat = reg_targets_flat[:, :3]
            logsize_targets_flat = reg_targets_flat[:, 3:]

            # Координаты точек, которым назначены предсказания/таргеты
            points_coords_flat = points_coords.reshape(-1, 3)[assign_masks_flat]  # [NumAssigned, 3]

            # Декодируем предсказанные боксы (в нормализованных координатах)
            pred_centers_norm = points_coords_flat + offset_preds_flat
            pred_sizes_norm = torch.exp(logsize_preds_flat)
            pred_bboxes_norm_assigned = torch.cat([pred_centers_norm, pred_sizes_norm], dim=1)

            # Декодируем целевые боксы (в нормализованных координатах)
            target_centers_norm = points_coords_flat + offset_targets_flat
            target_sizes_norm = torch.exp(logsize_targets_flat)
            target_bboxes_norm_assigned = torch.cat([target_centers_norm, target_sizes_norm], dim=1)

            # Вычисляем DIoU loss
            reg_loss_level = calculate_diou_3d_loss(
                pred_bboxes_norm_assigned, target_bboxes_norm_assigned, eps=diou_eps
            ).sum()  # Суммируем по всем назначенным точкам уровня
            total_reg_loss += reg_loss_level

            # --- Debugging ---
            if debug_loss and B > 0 and assign_masks[0].any():
                print(
                    f"\n--- [DEBUG LOSS Level {pred_level_idx} Head {head_idx} Batch 0 ({assign_masks[0].sum().item()} assigned)] ---")
                print_tensor_stats(pred_bboxes_norm_assigned, "Assigned Pred BBoxes (Norm)")
                print_tensor_stats(target_bboxes_norm_assigned, "Assigned Target BBoxes (Norm)")
                print_tensor_stats(reg_loss_level, f"Reg Loss Level {pred_level_idx} (Sum)")
        # else: # Если нет назначенных точек на этом уровне
        # ctr_loss_level = torch.tensor(0.0, device=device)
        # reg_loss_level = torch.tensor(0.0, device=device)

    # 3. Нормализация и взвешивание потерь
    # Нормализуем на общее количество назначенных точек по всем уровням
    norm_factor = max(num_assigned_total.item(), 1.0)

    final_cls_loss = (total_cls_loss / norm_factor) * loss_weights.get('cls', 1.0)
    final_ctr_loss = (total_ctr_loss / norm_factor) * loss_weights.get('ctr', 1.0)
    final_reg_loss = (total_reg_loss / norm_factor) * loss_weights.get('reg', 1.0)  # DIoU

    total_loss = final_cls_loss + final_ctr_loss + final_reg_loss

    # Проверка градиентов (на всякий случай)
    grad_components = [final_cls_loss, final_ctr_loss, final_reg_loss]
    if not total_loss.requires_grad and any(
            isinstance(comp, torch.Tensor) and comp.requires_grad for comp in grad_components):
        total_loss = total_loss.clone().requires_grad_(True)

    if debug_loss:
        print(f"--- [DEBUG LOSS Final] ---")
        print(f"  Num Assigned Total: {num_assigned_total.item()}")
        print_tensor_stats(total_cls_loss, "Total Cls Loss (Sum)", detach=False)
        print_tensor_stats(total_ctr_loss, "Total Ctr Loss (Sum)", detach=False)
        print_tensor_stats(total_reg_loss, "Total Reg Loss (Sum)", detach=False)
        print(f"  Normalization Factor: {norm_factor:.1f}")
        print_tensor_stats(final_cls_loss, "Final Cls Loss (Weighted)", detach=False)
        print_tensor_stats(final_ctr_loss, "Final Ctr Loss (Weighted)", detach=False)
        print_tensor_stats(final_reg_loss, "Final Reg Loss (Weighted)", detach=False)
        print_tensor_stats(total_loss, "Total Loss", detach=False)

    return {
        'total_loss': total_loss,
        'cls_loss': final_cls_loss.detach(),
        'ctr_loss': final_ctr_loss.detach(),
        'reg_loss': final_reg_loss.detach(),  # Заменили center и size
        'num_assigned': num_assigned_total.detach()
    }
