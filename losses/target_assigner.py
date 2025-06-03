"""Логика назначения целей (target assignment) для FCAF3D."""

import torch
from typing import List, Tuple
from configs import fcaf3d_config as cfg


def assign_targets_fcaf3d(
        points_coords_list: List[torch.Tensor],  # Координаты точек для КАЖДОГО уровня FP [B, N_level, 3]
        gt_bboxes_norm: torch.Tensor,  # Нормализованные GT боксы [B, M, 7] (cx,cy,cz,w,h,l,cls)
        num_objects: torch.Tensor,  # Количество объектов в каждом элементе батча [B,]
        num_fg_classes: int = cfg.NUM_FG_CLASSES,
        radius_scale: float = cfg.ASSIGNMENT_RADIUS_SCALE,
        min_radius: float = cfg.ASSIGNMENT_MIN_RADIUS,
        topk: int = cfg.ASSIGNMENT_TOPK
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Назначает цели (классы, центрированность, регрессию) точкам на разных уровнях.

    Args:
        points_coords_list (List[torch.Tensor]): Список координат точек для каждого уровня FP
                                                (от грубого к детальному), [B, N_level, 3].
        gt_bboxes_norm (torch.Tensor): Нормализованные GT боксы [B, M_max, 7].
        num_objects (torch.Tensor): Количество реальных GT объектов в батче [B,].
        num_fg_classes (int): Количество классов переднего плана.
        radius_scale (float): Множитель для радиуса назначения.
        min_radius (float): Минимальный радиус назначения.
        topk (int): Количество ближайших точек для рассмотрения для каждого GT.

    Returns:
        Tuple[List[torch.Tensor], ...]: Кортеж из списков (длиной L = num_levels):
            - all_cls_targets: Целевые классы [B, N_level] (0=фон, 1...=FG).
            - all_ctr_targets: Целевая центрированность [B, N_level] ([0, 1] или -1).
            - all_reg_targets: Цели регрессии [B, N_level, 6] (offset_xyz, logsize_wlh).
            - all_assign_masks: Маска назначенных точек [B, N_level] (bool).
    """
    B = gt_bboxes_norm.shape[0]
    M_max = gt_bboxes_norm.shape[1]
    L = len(points_coords_list)  # Количество уровней FP
    device = gt_bboxes_norm.device

    all_cls_targets = []
    all_ctr_targets = []
    all_reg_targets = []
    all_assign_masks = []

    # Итерируемся по уровням FP (от грубого к детальному, как в points_coords_list)
    for level_idx in range(L):
        points_coords = points_coords_list[level_idx]  # [B, N_level, 3]
        N_level = points_coords.shape[1]

        # Инициализация таргетов для текущего уровня
        cls_targets = torch.zeros((B, N_level), dtype=torch.long, device=device)
        ctr_targets = torch.full((B, N_level), -1.0, dtype=torch.float32, device=device)
        reg_targets = torch.zeros((B, N_level, 6), dtype=torch.float32, device=device)
        assign_masks = torch.zeros((B, N_level), dtype=torch.bool, device=device)

        for b in range(B):  # Итерация по элементам батча
            n_gt = num_objects[b].item()
            if n_gt == 0:
                continue  # Нет объектов для назначения

            valid_gt = gt_bboxes_norm[b, :n_gt]  # [n_gt, 7]
            centers_gt = valid_gt[:, :3]  # [n_gt, 3]
            sizes_gt = valid_gt[:, 3:6]  # [n_gt, 3]
            classes_gt = valid_gt[:, 6].long()  # [n_gt], классы 0, 1...

            points_b = points_coords[b]  # [N_level, 3]

            # Радиусы назначения на основе размеров GT
            gt_half_diag = torch.sqrt(torch.sum(sizes_gt ** 2, dim=1)) / 2.0
            radii = torch.clamp(gt_half_diag * radius_scale, min=min_radius)
            radii_sq = radii ** 2  # [n_gt]

            # Расстояния от точек до центров GT
            # dists_sq: [N_level, n_gt]
            dists_sq = torch.sum((points_b.unsqueeze(1) - centers_gt.unsqueeze(0)) ** 2, dim=-1)

            # Маска кандидатов: точки внутри радиуса
            candidate_mask = dists_sq < radii_sq.unsqueeze(0)  # [N_level, n_gt]

            # Для каждой точки находим ближайший GT (если она кандидат)
            candidate_dists = dists_sq.clone()
            candidate_dists[~candidate_mask] = float('inf')  # Игнорируем точки вне радиуса

            # Используем topk для ограничения кандидатов (более стабильно)
            current_topk = min(topk, N_level)
            if current_topk <= 0: continue

            # Находим topk ближайших точек для КАЖДОГО GT бокса
            # topk_indices: [n_gt, current_topk] - индексы точек
            # topk_dists: [n_gt, current_topk] - квадраты расстояний
            topk_dists_sq, topk_indices = torch.topk(
                candidate_dists.transpose(0, 1),  # [n_gt, N_level]
                k=current_topk,
                dim=1,
                largest=False  # Берем наименьшие расстояния
            )
            # Маска валидных кандидатов (расстояние не inf)
            valid_topk_mask = topk_dists_sq < float('inf')  # [n_gt, current_topk]

            # Жадное назначение: точка назначается ближайшему GT из кандидатов
            point_assigned_gt_idx = torch.full((N_level,), -1, dtype=torch.long, device=device)
            point_min_dist_sq = torch.full((N_level,), float('inf'), device=device)

            for gt_idx in range(n_gt):
                # Индексы точек и расстояния для текущего GT
                current_point_indices = topk_indices[gt_idx][valid_topk_mask[gt_idx]]
                current_dists_sq = topk_dists_sq[gt_idx][valid_topk_mask[gt_idx]]

                if current_point_indices.numel() == 0: continue

                # Находим точки, для которых текущий GT является лучшим (ближайшим)
                better_assignment_mask = current_dists_sq < point_min_dist_sq[current_point_indices]
                points_to_update = current_point_indices[better_assignment_mask]
                dists_to_update = current_dists_sq[better_assignment_mask]

                # Обновляем назначение и минимальное расстояние
                point_assigned_gt_idx[points_to_update] = gt_idx
                point_min_dist_sq[points_to_update] = dists_to_update

            # --- Формируем таргеты для назначенных точек ---
            assigned_points_mask_b = point_assigned_gt_idx != -1  # Маска назначенных точек [N_level]
            assigned_points_indices_b = torch.where(assigned_points_mask_b)[0]
            assigned_gt_indices_b = point_assigned_gt_idx[assigned_points_mask_b]

            if assigned_points_indices_b.numel() > 0:
                # 1. Классы (GT класс + 1)
                assigned_gt_classes = classes_gt[assigned_gt_indices_b]
                cls_targets[b, assigned_points_indices_b] = assigned_gt_classes + 1

                # 2. Регрессия (смещение центра и логарифм размера)
                assigned_gt_centers = centers_gt[assigned_gt_indices_b]
                assigned_gt_sizes = sizes_gt[assigned_gt_indices_b]
                assigned_points_coords = points_b[assigned_points_indices_b]

                reg_offset = assigned_gt_centers - assigned_points_coords
                reg_size_log = torch.log(torch.clamp(assigned_gt_sizes, min=1e-6))  # log(size)
                reg_targets[b, assigned_points_indices_b, :3] = reg_offset
                reg_targets[b, assigned_points_indices_b, 3:] = reg_size_log

                # 3. Центрированность (1 - sqrt(dist / radius))
                assigned_radii_sq_b = radii_sq[assigned_gt_indices_b]
                assigned_dists_sq_b = point_min_dist_sq[assigned_points_mask_b]
                # Нормализуем расстояние на квадрат радиуса
                normalized_dist_sq = torch.clamp(
                    assigned_dists_sq_b / torch.clamp(assigned_radii_sq_b, min=1e-6),
                    min=0.0, max=1.0
                )
                ctr = 1.0 - torch.sqrt(normalized_dist_sq)
                ctr_targets[b, assigned_points_indices_b] = torch.clamp(ctr, min=0.0, max=1.0)

                # 4. Маска назначенных точек
                assign_masks[b, assigned_points_indices_b] = True

        # Добавляем результаты уровня в общие списки
        all_cls_targets.append(cls_targets)
        all_ctr_targets.append(ctr_targets)
        all_reg_targets.append(reg_targets)
        all_assign_masks.append(assign_masks)

    return all_cls_targets, all_ctr_targets, all_reg_targets, all_assign_masks
