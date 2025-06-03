"""Функции для расчета метрик оценки детекции и классификации."""

import numpy as np
import torch
from typing import Dict, Union, Optional
from configs import fcaf3d_config as cfg  # Для CLASS_NAMES
from .metrics import compute_iou_3d  # Используем функцию IoU из utils.metrics


def calculate_iou_based_metrics(
        gt_bboxes_denorm: np.ndarray,  # Денорм. GT боксы [N_gt, 6] (cx,cy,cz,w,h,l)
        gt_classes: np.ndarray,  # GT классы [N_gt] (0, 1...)
        pred_bboxes_denorm: np.ndarray,  # Денорм. предсказанные боксы [N_pred, 6]
        pred_classes: np.ndarray,  # Предсказанные классы [N_pred] (0, 1...) - УЖЕ КОНВЕРТИРОВАННЫЕ!
        pred_scores: np.ndarray,  # Скоры предсказаний [N_pred]
        device: torch.device,  # Устройство для расчета IoU
        iou_threshold: float = cfg.METRICS_IOU_THRESHOLD,
        num_classes: int = cfg.NUM_FG_CLASSES
) -> Dict[str, Union[float, str, np.ndarray]]:
    """
    Вычисляет метрики детекции на основе IoU (Precision, Recall, F1, Avg IoU).

    Args:
        gt_bboxes_denorm: Денормализованные GT боксы.
        gt_classes: GT классы (индексы 0, 1...).
        pred_bboxes_denorm: Денормализованные предсказанные боксы.
        pred_classes: Предсказанные классы (индексы 0, 1...).
        pred_scores: Скоры предсказаний.
        device: Устройство для вычислений.
        iou_threshold: Порог IoU для сопоставления TP.
        num_classes: Количество классов переднего плана.

    Returns:
        Словарь с метриками: 'f1_macro', 'avg_iou', 'precision_per_class',
                             'recall_per_class', 'f1_per_class', 'support_per_class',
                             'tp_counts', 'fp_counts', 'fn_counts', 'report'.
    """
    num_gt = gt_bboxes_denorm.shape[0]
    num_pred = pred_bboxes_denorm.shape[0]

    # Инициализация счетчиков
    tp_counts = np.zeros(num_classes, dtype=int)
    fp_counts = np.zeros(num_classes, dtype=int)
    fn_counts = np.zeros(num_classes, dtype=int)
    matched_ious = []  # Список IoU для TP

    # Маски для отслеживания сопоставленных GT и Pred
    gt_matched_mask = np.zeros(num_gt, dtype=bool)
    pred_matched_mask = np.zeros(num_pred, dtype=bool)

    # Выполняем сопоставление только если есть и GT, и предсказания
    if num_gt > 0 and num_pred > 0:
        # Переводим боксы в тензоры для расчета IoU
        gt_boxes_iou = torch.from_numpy(gt_bboxes_denorm).float().to(device)
        pred_boxes_iou = torch.from_numpy(pred_bboxes_denorm).float().to(device)

        # Вычисляем матрицу IoU [N_gt, N_pred]
        iou_matrix = np.zeros((num_gt, num_pred))
        # Эффективнее считать сразу матрицу, если позволяет память
        try:
            # compute_iou_3d(pred[N,6], gt[M,6]) -> [N, M] -> transpose -> [M, N]
            iou_matrix_torch = compute_iou_3d(pred_boxes_iou, gt_boxes_iou).T
            iou_matrix = iou_matrix_torch.cpu().numpy()
        except RuntimeError:  # Если памяти не хватает, считаем построчно
            print("Предупреждение: расчет матрицы IoU построчно (может быть медленно)")
            for i in range(num_gt):
                ious = compute_iou_3d(gt_boxes_iou[i].unsqueeze(0), pred_boxes_iou)
                iou_matrix[i, :] = ious.cpu().numpy().squeeze(0)

        # Сортируем предсказания по скору (убывание)
        sorted_pred_indices = np.argsort(-pred_scores)

        # Жадное сопоставление
        for pred_idx in sorted_pred_indices:
            pred_cls = pred_classes[pred_idx]  # Класс 0, 1...

            # Пропускаем, если класс некорректный (на всякий случай)
            if pred_cls >= num_classes or pred_cls < 0:
                continue

            best_iou = -1.0
            best_match_gt_idx = -1

            # Ищем лучший НЕсопоставленный GT того же класса
            for gt_idx in range(num_gt):
                # Пропускаем уже сопоставленные GT или GT другого класса
                if gt_matched_mask[gt_idx] or gt_classes[gt_idx] != pred_cls:
                    continue

                current_iou = iou_matrix[gt_idx, pred_idx]

                # Если IoU выше порога и лучше текущего лучшего для этого pred
                if current_iou >= iou_threshold and current_iou > best_iou:
                    best_iou = current_iou
                    best_match_gt_idx = gt_idx

            # Если нашли подходящий GT
            if best_match_gt_idx != -1:
                # Помечаем оба как сопоставленные
                gt_matched_mask[best_match_gt_idx] = True
                pred_matched_mask[pred_idx] = True
                # Увеличиваем счетчик TP для этого класса и сохраняем IoU
                tp_counts[pred_cls] += 1
                matched_ious.append(best_iou)

    # Считаем FN (несопоставленные GT)
    for i in range(num_gt):
        if not gt_matched_mask[i]:
            gt_cls = gt_classes[i]
            if 0 <= gt_cls < num_classes:
                fn_counts[gt_cls] += 1

    # Считаем FP (несопоставленные предсказания)
    for j in range(num_pred):
        if not pred_matched_mask[j]:
            pred_cls = pred_classes[j]
            if 0 <= pred_cls < num_classes:
                fp_counts[pred_cls] += 1

    # Рассчитываем Precision, Recall, F1 для каждого класса
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    support_per_class = tp_counts + fn_counts  # Количество GT объектов класса

    for c in range(num_classes):
        tp = tp_counts[c]
        fp = fp_counts[c]
        fn = fn_counts[c]
        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            precision_per_class[c] = tp / (tp + fp)
        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            recall_per_class[c] = tp / (tp + fn)
        # F1 = 2 * P * R / (P + R)
        if precision_per_class[c] + recall_per_class[c] > 0:
            f1_per_class[c] = (2 * precision_per_class[c] * recall_per_class[c] /
                               (precision_per_class[c] + recall_per_class[c]))

    # Рассчитываем Macro F1 (среднее F1 по классам, где есть GT объекты)
    valid_classes_mask = support_per_class > 0
    if np.any(valid_classes_mask):
        f1_macro = np.mean(f1_per_class[valid_classes_mask])
    elif num_gt == 0 and num_pred == 0:  # Если и GT, и Pred пусты, считаем идеальным
        f1_macro = 1.0
    else:  # Если один пуст, а другой нет, или есть только FP/FN
        f1_macro = 0.0

    # Средний IoU по всем True Positive совпадениям
    avg_iou = np.mean(matched_ious) if matched_ious else 0.0
    # Если нет ни GT, ни Pred, считаем IoU идеальным
    if num_gt == 0 and num_pred == 0:
        avg_iou = 1.0

    # Формируем текстовый отчет
    report_str = f"IoU-based Detection Report (IoU Threshold = {iou_threshold}):\n"
    header = "Class      \tPrec.\tRecall\tF1-Score\tSupport\tTP\tFP\tFN\n"
    report_str += header + "-" * len(header) + "\n"
    for c in range(num_classes):
        class_name = cfg.CLASS_NAMES[c] if c < len(cfg.CLASS_NAMES) else f"Class_{c}"
        report_str += (
            f"{class_name:<11s}\t{precision_per_class[c]:.3f}\t{recall_per_class[c]:.3f}\t"
            f"{f1_per_class[c]:.3f}\t{support_per_class[c]:<7d}\t{tp_counts[c]:<2d}\t"
            f"{fp_counts[c]:<2d}\t{fn_counts[c]:<2d}\n"
        )
    # Macro Average
    macro_prec = np.mean(precision_per_class[valid_classes_mask]) if np.any(valid_classes_mask) else 0.0
    macro_rec = np.mean(recall_per_class[valid_classes_mask]) if np.any(valid_classes_mask) else 0.0
    report_str += "-" * len(header) + "\n"
    report_str += (
        f"Macro Avg  \t{macro_prec:.3f}\t{macro_rec:.3f}\t{f1_macro:.4f}\t"
        f"{np.sum(support_per_class):<7d}\n"
    )
    report_str += f"\nAverage Matched IoU: {avg_iou:.4f} ({len(matched_ious)} matches)"

    return {
        'f1_macro': f1_macro,
        'avg_iou': avg_iou,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'tp_counts': tp_counts,
        'fp_counts': fp_counts,
        'fn_counts': fn_counts,
        'report': report_str
    }


def calculate_class_presence_metrics(
        gt_classes: Optional[np.ndarray],  # GT классы [N_gt] (0, 1...) или None
        pred_classes: Optional[np.ndarray],  # Предсказанные классы [N_pred] (0, 1...) или None
        num_classes: int = cfg.NUM_FG_CLASSES
) -> Dict[str, Union[float, str, np.ndarray]]:
    """
    Вычисляет метрики классификации (Precision, Recall, F1) на уровне сцены,
    игнорируя IoU и количество объектов, сравнивая только наборы присутствующих классов.

    Args:
        gt_classes: Массив GT классов (индексы 0, 1...).
        pred_classes: Массив предсказанных классов (индексы 0, 1...).
        num_classes: Количество классов переднего плана.

    Returns:
        Словарь с метриками: 'f1_macro_presence', 'precision_per_class',
                             'recall_per_class', 'f1_per_class', 'report_presence'.
    """
    if gt_classes is None or pred_classes is None:
        return {
            'f1_macro_presence': 0.0,
            'precision_per_class': np.zeros(num_classes),
            'recall_per_class': np.zeros(num_classes),
            'f1_per_class': np.zeros(num_classes),
            'report_presence': "Ошибка: Входные массивы классов None."
        }

    # Оставляем только уникальные классы FG в GT и Pred
    gt_present_classes = np.unique(gt_classes[(gt_classes >= 0) & (gt_classes < num_classes)])
    pred_present_classes = np.unique(pred_classes[(pred_classes >= 0) & (pred_classes < num_classes)])

    # Инициализация счетчиков TP, FP, FN на уровне присутствия класса
    tp_presence = np.zeros(num_classes, dtype=int)
    fp_presence = np.zeros(num_classes, dtype=int)
    fn_presence = np.zeros(num_classes, dtype=int)

    for c in range(num_classes):
        gt_present = c in gt_present_classes
        pred_present = c in pred_present_classes

        if gt_present and pred_present:
            tp_presence[c] = 1
        elif not gt_present and pred_present:
            fp_presence[c] = 1
        elif gt_present and not pred_present:
            fn_presence[c] = 1
        # else: TN (не считаем)

    # Рассчитываем Precision, Recall, F1 для каждого класса по присутствию
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)
    # Support здесь - был ли класс вообще в GT (1 или 0)
    support_per_class = np.isin(np.arange(num_classes), gt_present_classes).astype(int)

    for c in range(num_classes):
        tp = tp_presence[c]
        fp = fp_presence[c]
        fn = fn_presence[c]
        # Precision = TP / (TP + FP)
        if tp + fp > 0:
            precision_per_class[c] = tp / (tp + fp)
        # Recall = TP / (TP + FN)
        if tp + fn > 0:
            recall_per_class[c] = tp / (tp + fn)
        # F1 = 2 * P * R / (P + R)
        if precision_per_class[c] + recall_per_class[c] > 0:
            f1_per_class[c] = (2 * precision_per_class[c] * recall_per_class[c] /
                               (precision_per_class[c] + recall_per_class[c]))

    # Рассчитываем Macro F1 (среднее F1 по классам, которые были в GT)
    valid_classes_mask = support_per_class > 0
    if np.any(valid_classes_mask):
        f1_macro_presence = np.mean(f1_per_class[valid_classes_mask])
    # Если в GT не было ни одного FG класса, и в Pred тоже, считаем идеальным
    elif gt_present_classes.size == 0 and pred_present_classes.size == 0:
        f1_macro_presence = 1.0
    else:  # Иначе (например, в GT не было, а в Pred есть, или наоборот)
        f1_macro_presence = 0.0

    # Формируем текстовый отчет
    report_str = "Class Presence Report (Ignoring IoU & Counts):\n"
    header = "Class      \tPrec.\tRecall\tF1-Score\tSupport\tTP\tFP\tFN\n"
    report_str += header + "-" * len(header) + "\n"
    for c in range(num_classes):
        class_name = cfg.CLASS_NAMES[c] if c < len(cfg.CLASS_NAMES) else f"Class_{c}"
        report_str += (
            f"{class_name:<11s}\t{precision_per_class[c]:.3f}\t{recall_per_class[c]:.3f}\t"
            f"{f1_per_class[c]:.3f}\t{support_per_class[c]:<7d}\t{tp_presence[c]:<2d}\t"
            f"{fp_presence[c]:<2d}\t{fn_presence[c]:<2d}\n"
        )
    # Macro Average
    macro_prec = np.mean(precision_per_class[valid_classes_mask]) if np.any(valid_classes_mask) else 0.0
    macro_rec = np.mean(recall_per_class[valid_classes_mask]) if np.any(valid_classes_mask) else 0.0
    report_str += "-" * len(header) + "\n"
    report_str += (
        f"Macro Avg  \t{macro_prec:.3f}\t{macro_rec:.3f}\t{f1_macro_presence:.4f}\t"
        f"{np.sum(support_per_class):<7d}\n"
    )

    return {
        'f1_macro_presence': f1_macro_presence,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'report_presence': report_str
    }
