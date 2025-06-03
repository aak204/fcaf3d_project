"""Логика оценки модели."""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List
from sklearn.metrics import classification_report, confusion_matrix
from configs import fcaf3d_config as cfg
from models.fcaf3d import FCAF3D
from losses.fcaf3d_loss import compute_fcaf3d_loss
from utils.metrics import compute_iou_3d, nms_3d
from data.preprocessing import denormalize_bbox


@torch.no_grad()  # Отключаем вычисление градиентов для оценки
def evaluate_fcaf3d_model(
        model: FCAF3D,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        loss_weights: Dict = cfg.LOSS_WEIGHTS,
        pc_range: List[float] = cfg.POINT_CLOUD_RANGE,
        iou_threshold_nms: float = cfg.NMS_IOU_THRESHOLD,
        max_objects: int = cfg.MAX_OBJECTS_PER_SCENE,
        score_threshold: float = cfg.SCORE_THRESHOLD,
        metrics_iou_threshold: float = cfg.METRICS_IOU_THRESHOLD,
        debug_eval: bool = cfg.DEBUG_EVALUATION
) -> Dict:
    """
    Оценивает модель FCAF3D на валидационном/тестовом датасете.

    Args:
        model (FCAF3D): Обученная модель.
        dataloader (DataLoader): Загрузчик данных для оценки.
        device (torch.device): Устройство (CPU/GPU).
        loss_weights (Dict): Веса лоссов (для информации).
        pc_range (List[float]): Диапазон облака точек.
        iou_threshold_nms (float): Порог IoU для NMS.
        max_objects (int): Максимальное количество объектов на сцену после NMS.
        score_threshold (float): Порог уверенности для предсказаний.
        metrics_iou_threshold (float): Порог IoU для сопоставления TP/FP/FN.
        debug_eval (bool): Флаг для вывода детальной информации.

    Returns:
        Dict: Словарь с вычисленными метриками.
    """
    model.eval()  # Переводим модель в режим оценки

    # Инициализация переменных для сбора статистики
    total_loss_sum = 0.0
    total_cls_loss_sum = 0.0
    total_ctr_loss_sum = 0.0
    total_reg_loss_sum = 0.0
    num_samples = 0

    all_ious_matched = []  # IoU для сопоставленных TP
    all_gt_classes_matched = []  # Классы GT для TP
    all_pred_classes_matched = []  # Классы Pred для TP
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print(f"\n--- Запуск оценки FCAF3D (IoU Thresh={metrics_iou_threshold}) ---")
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Оценка")):
        # Проверяем батч (collate_fn может вернуть пустой)
        if not batch: continue
        points = batch.get('points')
        gt_bboxes = batch.get('gt_bboxes')
        num_objects = batch.get('num_objects')

        if points is None or gt_bboxes is None or num_objects is None or points.numel() == 0:
            continue
        if gt_bboxes.shape[-1] != 7:
            print(f"Предупреждение: Пропуск батча {batch_idx}, неверная форма gt_bboxes: {gt_bboxes.shape}")
            continue

        points = points.to(device)
        gt_bboxes = gt_bboxes.to(device)  # Нормализованные [B, M_max, 7]
        num_objects = num_objects.to(device)  # [B,]
        B = points.shape[0]
        current_batch_samples = B
        num_samples += current_batch_samples

        # --- Прямой проход и расчет лосса (для информации) ---
        try:
            end_points = model(points)
            loss_dict = compute_fcaf3d_loss(
                end_points, gt_bboxes, num_objects, loss_weights, cfg.NUM_FG_CLASSES,
                debug_loss=False  # Не дебажим лосс на оценке
            )
            if isinstance(loss_dict.get('total_loss'), torch.Tensor):
                total_loss_sum += loss_dict['total_loss'].item() * current_batch_samples
                total_cls_loss_sum += loss_dict['cls_loss'].item() * current_batch_samples
                total_ctr_loss_sum += loss_dict['ctr_loss'].item() * current_batch_samples
                total_reg_loss_sum += loss_dict['reg_loss'].item() * current_batch_samples
            else:
                total_loss_sum += float('inf')  # Если лосс не посчитался
        except Exception as e:
            print(f"Ошибка расчета лосса на валидации (батч {batch_idx}): {e}")
            total_loss_sum += float('inf')
            continue  # Пропускаем батч, если лосс не считается

        # --- Декодирование предсказаний и NMS ---
        cls_preds_all = end_points['cls_preds']
        ctr_preds_all = end_points['ctr_preds']
        offset_preds_all = end_points['center_offset_preds']
        logsize_preds_all = end_points['size_log_preds']
        points_coords_fp_list = end_points['fp_xyz']  # [xyz3, xyz2, xyz1, xyz0]
        num_fp_levels = len(points_coords_fp_list)

        for b in range(B):  # Итерация по элементам батча
            preds_before_nms = []  # Список [score, class_id(1..), bbox_norm(6)]

            # Собираем предсказания со всех голов
            for head_idx, pred_level_idx in enumerate(cfg.PREDICTION_HEAD_LEVELS):
                cls_logits = cls_preds_all[head_idx][b]  # [C, N_level]
                ctr_logits = ctr_preds_all[head_idx][b]  # [1, N_level]
                offset_preds = offset_preds_all[head_idx][b]  # [3, N_level]
                logsize_preds = logsize_preds_all[head_idx][b]  # [3, N_level]

                # Координаты точек для этого уровня FP
                fp_level_map_idx = num_fp_levels - 1 - pred_level_idx
                points_coords = points_coords_fp_list[fp_level_map_idx][b]  # [N_level, 3]

                N_pred = cls_logits.shape[1]
                if N_pred == 0 or N_pred != points_coords.shape[0]: continue

                # Вычисляем скоры и классы
                cls_prob = torch.softmax(cls_logits, dim=0)  # [C, N_level]
                ctr_prob = torch.sigmoid(ctr_logits).squeeze(0)  # [N_level]
                # Макс. вероятность среди FG классов и их индексы (0, 1...)
                max_fg_cls_prob, pred_fg_cls_idx = torch.max(cls_prob[1:, :], dim=0)
                pred_cls = pred_fg_cls_idx + 1  # Классы 1, 2...
                score = max_fg_cls_prob * ctr_prob  # Комбинированный скор

                # Фильтруем по порогу уверенности
                valid_mask = score > score_threshold
                if not valid_mask.any(): continue

                score_f = score[valid_mask]
                pred_cls_f = pred_cls[valid_mask]
                offset_preds_f = offset_preds[:, valid_mask].T  # [N_valid, 3]
                logsize_preds_f = logsize_preds[:, valid_mask].T  # [N_valid, 3]
                points_coords_f = points_coords[valid_mask]  # [N_valid, 3]

                # Декодируем боксы (в нормализованных координатах)
                pred_centers_norm = points_coords_f + offset_preds_f
                pred_sizes_norm = torch.exp(logsize_preds_f)
                pred_bboxes_norm_level = torch.cat([pred_centers_norm, pred_sizes_norm], dim=1)  # [N_valid, 6]

                # Добавляем в список для NMS
                for i in range(pred_bboxes_norm_level.shape[0]):
                    preds_before_nms.append(
                        [score_f[i], pred_cls_f[i], pred_bboxes_norm_level[i]]
                    )

            # --- NMS ---
            final_bboxes_norm_b = torch.empty((0, 6), device=device)
            final_classes_b = torch.empty((0,), dtype=torch.long, device=device)  # Классы 1, 2...
            final_scores_b = torch.empty((0,), device=device)

            if preds_before_nms:
                scores_t = torch.stack([p[0] for p in preds_before_nms])
                classes_t = torch.stack([p[1] for p in preds_before_nms])
                bboxes_t = torch.stack([p[2] for p in preds_before_nms])  # [N_before, 6]

                keep_indices = nms_3d(bboxes_t, scores_t, iou_threshold_nms)
                if keep_indices.numel() > 0:
                    keep_indices = keep_indices[:max_objects]  # Ограничиваем макс. числом объектов
                    final_bboxes_norm_b = bboxes_t[keep_indices]
                    final_classes_b = classes_t[keep_indices]
                    final_scores_b = scores_t[keep_indices]

            # --- Сопоставление с GT ---
            n_gt = num_objects[b].item()
            n_pred = final_bboxes_norm_b.shape[0]
            gt_bboxes_b_norm = gt_bboxes[b, :n_gt, :6]  # Нормализованные GT [n_gt, 6]
            gt_classes_b = gt_bboxes[b, :n_gt, 6].long()  # Классы GT (0, 1...)

            # Предсказанные классы FG (0, 1...) для сопоставления
            pred_classes_b_original = final_classes_b - 1

            if n_gt == 0 and n_pred == 0: continue  # Ничего нет, идем дальше
            if n_gt == 0:  # Есть только предсказания -> все FP
                false_positives += n_pred
                continue
            if n_pred == 0:  # Есть только GT -> все FN
                false_negatives += n_gt
                continue

            # Денормализуем боксы для расчета IoU
            gt_bboxes_b_denorm = denormalize_bbox(gt_bboxes_b_norm, pc_range)
            pred_bboxes_b_denorm = denormalize_bbox(final_bboxes_norm_b, pc_range)

            # Матрица IoU [n_gt, n_pred]
            iou_matrix = compute_iou_3d(pred_bboxes_b_denorm, gt_bboxes_b_denorm).T  # Транспонируем

            gt_matched_mask = torch.zeros(n_gt, dtype=torch.bool, device=device)
            pred_matched_mask = torch.zeros(n_pred, dtype=torch.bool, device=device)

            # Жадное сопоставление по убыванию скора предсказаний
            sorted_pred_indices = torch.argsort(final_scores_b, descending=True)

            for pred_idx in sorted_pred_indices:
                pred_cls = pred_classes_b_original[pred_idx].item()  # Класс 0, 1...

                best_iou = -1.0
                best_match_gt_idx = -1

                # Ищем лучший НЕсопоставленный GT того же класса
                for gt_idx in range(n_gt):
                    if gt_matched_mask[gt_idx]: continue  # Пропускаем уже сопоставленные GT
                    if gt_classes_b[gt_idx].item() != pred_cls: continue  # Классы должны совпадать

                    current_iou = iou_matrix[gt_idx, pred_idx].item()
                    if current_iou >= metrics_iou_threshold and current_iou > best_iou:
                        best_iou = current_iou
                        best_match_gt_idx = gt_idx

                # Если нашли подходящий GT
                if best_match_gt_idx != -1:
                    # Помечаем оба как сопоставленные
                    gt_matched_mask[best_match_gt_idx] = True
                    pred_matched_mask[pred_idx] = True
                    # Сохраняем для статистики
                    true_positives += 1
                    all_ious_matched.append(best_iou)
                    all_gt_classes_matched.append(gt_classes_b[best_match_gt_idx].item())
                    all_pred_classes_matched.append(pred_cls)

            # Считаем FP и FN
            false_positives += n_pred - pred_matched_mask.sum().item()
            false_negatives += n_gt - gt_matched_mask.sum().item()

    # --- Финальные Метрики ---
    avg_loss = total_loss_sum / max(num_samples, 1)
    avg_cls_loss = total_cls_loss_sum / max(num_samples, 1)
    avg_ctr_loss = total_ctr_loss_sum / max(num_samples, 1)
    avg_reg_loss = total_reg_loss_sum / max(num_samples, 1)

    mean_iou_matched = np.mean(all_ious_matched) if all_ious_matched else 0.0
    precision = true_positives / max(true_positives + false_positives, 1e-6)
    recall = true_positives / max(true_positives + false_negatives, 1e-6)
    f1_detection = 2 * (precision * recall) / max(precision + recall, 1e-6)

    class_accuracy_tp = 0.0
    f1_macro_tp = 0.0
    f1_weighted_tp = 0.0
    report_str_tp = "Classification Report (TP vs Matched GT):\n N/A"
    cm_tp = np.zeros((cfg.NUM_FG_CLASSES, cfg.NUM_FG_CLASSES), dtype=int)

    if all_gt_classes_matched:  # Если были найдены TP
        class_accuracy_tp = np.mean(np.array(all_gt_classes_matched) == np.array(all_pred_classes_matched))
        try:
            cm_tp = confusion_matrix(
                all_gt_classes_matched, all_pred_classes_matched,
                labels=list(range(cfg.NUM_FG_CLASSES))
            )
            report_tp_dict = classification_report(
                all_gt_classes_matched, all_pred_classes_matched,
                labels=list(range(cfg.NUM_FG_CLASSES)), target_names=cfg.CLASS_NAMES,
                output_dict=True, zero_division=0
            )
            report_str_tp = classification_report(
                all_gt_classes_matched, all_pred_classes_matched,
                labels=list(range(cfg.NUM_FG_CLASSES)), target_names=cfg.CLASS_NAMES,
                zero_division=0
            )
            f1_macro_tp = report_tp_dict['macro avg']['f1-score']
            f1_weighted_tp = report_tp_dict['weighted avg']['f1-score']
        except Exception as e:
            print(f"Ошибка расчета classification report для TP пар: {e}")
            report_str_tp = "Classification Report (TP) Error"

    print(f"\n--- Результаты Оценки (IoU Thresh: {metrics_iou_threshold}) ---")
    print(f"Средний Loss: {avg_loss:.4f} (Cls:{avg_cls_loss:.4f}, Ctr:{avg_ctr_loss:.4f}, Reg:{avg_reg_loss:.4f})")
    print(f"IoU(TP): {mean_iou_matched:.4f} ({len(all_ious_matched)} пар), Acc(TP): {class_accuracy_tp:.4f}")
    print(
        f"Detection: P={precision:.4f}, R={recall:.4f}, F1={f1_detection:.4f} (TP={true_positives}, FP={false_positives}, FN={false_negatives})")
    print(f"Classification(TP): Macro-F1={f1_macro_tp:.4f}, Weighted-F1={f1_weighted_tp:.4f}")
    print(report_str_tp)

    return {
        'mean_iou': mean_iou_matched,
        'class_accuracy': class_accuracy_tp,
        'f1_score': f1_detection,  # F1 для детекции
        'precision': precision,
        'recall': recall,
        'confusion_matrix': cm_tp,
        'avg_loss': avg_loss,
        'avg_cls_loss': avg_cls_loss,
        'avg_ctr_loss': avg_ctr_loss,
        'avg_reg_loss': avg_reg_loss,
        'f1_macro': f1_macro_tp,  # F1 для классификации TP
        'f1_weighted': f1_weighted_tp,
        'cls_report': report_str_tp
    }
