"""Скрипт для инференса и опциональной оценки модели FCAF3D на отдельных файлах."""

import os
import sys
import glob
import traceback
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
import json

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Импорты из проекта
from configs import fcaf3d_config as cfg  # Загружаем конфигурацию по умолчанию
from models.fcaf3d import FCAF3D
from data.preprocessing import (
    normalize_pc, normalize_bbox, clip_points_to_range,
    denormalize_bbox, denormalize_pc_coords
)
from data.utils import downsample_point_cloud
from utils.metrics import nms_3d
from utils.visualization import visualize_point_cloud_with_boxes, save_visualization
from utils.evaluation_metrics import calculate_iou_based_metrics, calculate_class_presence_metrics


def load_model(model_path: str, config, device: torch.device) -> FCAF3D:
    """Загружает обученную модель FCAF3D."""
    print(f"Загрузка модели из: {model_path}")
    print(f"  Параметры модели: INPUT_FEAT_DIM={config.INPUT_FEAT_DIM}, FP_DIM={config.FP_FEATURE_DIM}")
    model = FCAF3D(
        input_channels=config.INPUT_FEAT_DIM,
        num_fg_classes=config.NUM_FG_CLASSES,
        num_levels=config.NUM_LEVELS,
        fp_feature_dim=config.FP_FEATURE_DIM,
        pred_head_levels=config.PREDICTION_HEAD_LEVELS
    ).to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f"Веса модели успешно загружены (из эпохи {checkpoint.get('epoch', 'N/A')})")
        model.eval()
        return model
    except Exception as e:
        print(f"ОШИБКА: Не удалось загрузить веса модели: {e}")
        traceback.print_exc()
        sys.exit(1)


def preprocess_point_cloud(
        pcd_path: str,
        config
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Загружает и предобрабатывает одно облако точек для инференса."""
    expected_point_dim = 3 + config.INPUT_FEAT_DIM
    points_vis_np = None

    try:
        pcd = o3d.io.read_point_cloud(pcd_path)
        if not pcd.has_points():
            print(f"Предупреждение: Файл облака точек пуст: {pcd_path}")
            return None, None

        points_xyz_raw = np.asarray(pcd.points, dtype=np.float32)
        points_xyz_raw[:, 2] = -points_xyz_raw[:, 2]

        features_list = []
        if config.USE_NORMALS_AS_FEATURES:
            try:
                pcd_norm = o3d.geometry.PointCloud()
                pcd_norm.points = o3d.utility.Vector3dVector(points_xyz_raw)
                pcd_norm.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
                                          fast_normal_computation=True)
                pcd_norm.normalize_normals()
                normals = np.asarray(pcd_norm.normals, dtype=np.float32)
                if normals.shape[0] == points_xyz_raw.shape[0]:
                    features_list.append(normals)
                else:
                    features_list.append(np.zeros_like(points_xyz_raw))
            except Exception:
                features_list.append(np.zeros_like(points_xyz_raw))

        if config.USE_INTENSITY_AS_FEATURES:
            intensity = None
            if pcd.has_colors():
                intensity_raw = np.asarray(pcd.colors, dtype=np.float32)[:, 0:1]
                intensity = np.clip(intensity_raw, 0.0, 1.0)
            if intensity is not None and intensity.shape[0] == points_xyz_raw.shape[0]:
                features_list.append(intensity)
            else:
                features_list.append(np.zeros((points_xyz_raw.shape[0], 1), dtype=np.float32))

        if not features_list and config.INPUT_FEAT_DIM > 0:
            features_list.append(np.ones((points_xyz_raw.shape[0], config.INPUT_FEAT_DIM), dtype=np.float32))

        points_processed = np.hstack([points_xyz_raw] + features_list) if features_list else points_xyz_raw

        if points_processed.shape[1] != expected_point_dim:
            if points_processed.shape[1] > expected_point_dim:
                points_processed = points_processed[:, :expected_point_dim]
            else:
                padding = np.zeros((points_processed.shape[0], expected_point_dim - points_processed.shape[1]),
                                   dtype=points_processed.dtype);
                points_processed = np.hstack(
                    [points_processed, padding])

        points_processed = clip_points_to_range(points_processed, config.POINT_CLOUD_RANGE)
        if points_processed.shape[0] < 10:
            print(f"Предупреждение: Менее 10 точек после обрезки для {pcd_path}. Пропуск.")
            points_vis_np = points_processed
            return None, points_vis_np

        points_sampled = downsample_point_cloud(
            points_processed, method=config.DOWNSAMPLING_METHOD, max_points=config.MAX_POINTS
        )
        current_n_points = points_sampled.shape[0]
        if current_n_points == 0:
            print(f"Ошибка: 0 точек после даунсэмплинга для {pcd_path}.")
            return None, None

        if current_n_points < config.MAX_POINTS:
            repeat_indices = np.random.choice(current_n_points, config.MAX_POINTS - current_n_points, replace=True)
            points_final_np = np.vstack((points_sampled, points_sampled[repeat_indices]))
        else:
            points_final_np = points_sampled[:config.MAX_POINTS]

        points_vis_np = points_final_np.copy()
        points_norm_np = normalize_pc(points_final_np, config.POINT_CLOUD_RANGE)

        return points_norm_np, points_vis_np

    except Exception as e:
        print(f"Ошибка загрузки/предобработки {pcd_path}: {e}")
        traceback.print_exc()
        return None, None


def run_inference_single_file(
        model: FCAF3D,
        points_norm_np: np.ndarray,
        device: torch.device,
        config
) -> Optional[Dict[str, torch.Tensor]]:
    """Выполняет прямой проход модели и пост-обработку (NMS)."""
    points_tensor = torch.from_numpy(points_norm_np).float().unsqueeze(0).to(device)
    final_bboxes_norm_tensor = torch.empty((0, 6), device=device)
    final_classes_tensor = torch.empty((0,), dtype=torch.long, device=device)
    final_scores_tensor = torch.empty((0,), device=device)

    try:
        with torch.no_grad():
            end_points = model(points_tensor)

            cls_preds_all = end_points['cls_preds']
            ctr_preds_all = end_points['ctr_preds']
            offset_preds_all = end_points['center_offset_preds']
            logsize_preds_all = end_points['size_log_preds']
            points_coords_fp_list = end_points['fp_xyz']
            num_fp_levels = len(points_coords_fp_list)
            preds_before_nms = []

            for head_idx, pred_level_idx in enumerate(config.PREDICTION_HEAD_LEVELS):
                cls_logits = cls_preds_all[head_idx][0]
                ctr_logits = ctr_preds_all[head_idx][0]
                offset_preds = offset_preds_all[head_idx][0]
                logsize_preds = logsize_preds_all[head_idx][0]
                fp_level_map_idx = num_fp_levels - 1 - pred_level_idx
                points_coords = points_coords_fp_list[fp_level_map_idx][0]
                N_pred = cls_logits.shape[1]

                if N_pred == 0 or N_pred != points_coords.shape[0]: continue

                cls_prob = torch.softmax(cls_logits, dim=0)
                ctr_prob = torch.sigmoid(ctr_logits).squeeze(0)
                max_fg_cls_prob, pred_fg_cls_idx = torch.max(cls_prob[1:, :], dim=0)
                pred_cls = pred_fg_cls_idx + 1
                score = max_fg_cls_prob * ctr_prob
                valid_mask = score > config.SCORE_THRESHOLD

                if not valid_mask.any(): continue

                score_f = score[valid_mask]
                pred_cls_f = pred_cls[valid_mask]
                offset_preds_f = offset_preds[:, valid_mask].T
                logsize_preds_f = logsize_preds[:, valid_mask].T
                points_coords_f = points_coords[valid_mask]
                pred_centers_norm = points_coords_f + offset_preds_f
                pred_sizes_norm = torch.exp(logsize_preds_f)
                pred_bboxes_norm_level = torch.cat([pred_centers_norm, pred_sizes_norm], dim=1)

                for i in range(pred_bboxes_norm_level.shape[0]):
                    preds_before_nms.append([score_f[i], pred_cls_f[i], pred_bboxes_norm_level[i]])

            if preds_before_nms:
                scores_t = torch.stack([p[0] for p in preds_before_nms])
                classes_t = torch.stack([p[1] for p in preds_before_nms])
                bboxes_t = torch.stack([p[2] for p in preds_before_nms])

                keep_indices = nms_3d(bboxes_t, scores_t, config.NMS_IOU_THRESHOLD)
                if keep_indices.numel() > 0:
                    keep_indices = keep_indices[:config.MAX_OBJECTS_PER_SCENE]
                    final_bboxes_norm_tensor = bboxes_t[keep_indices]
                    final_classes_tensor = classes_t[keep_indices]
                    final_scores_tensor = scores_t[keep_indices]

        return {
            'pred_bboxes_norm': final_bboxes_norm_tensor,
            'pred_classes': final_classes_tensor,
            'pred_scores': final_scores_tensor
        }

    except Exception as e:
        print(f"Ошибка во время инференса модели: {e}")
        traceback.print_exc()
        return None


def load_gt_annotations(
        pcd_path: str,
        config
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Загружает GT аннотации для файла."""
    json_path = pcd_path + ".json"
    if not os.path.exists(json_path):
        return np.empty((0, 6)), np.empty((0,), dtype=int)

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        if 'objects' not in annotation or 'figures' not in annotation:
            return np.empty((0, 6)), np.empty((0,), dtype=int)

        objects_dict = {obj['key']: obj['classTitle'] for obj in annotation['objects']}
        gt_bboxes_raw = []

        for figure in annotation['figures']:
            if figure.get('geometryType') != 'cuboid_3d': continue
            object_key = figure.get('objectKey')
            class_title = objects_dict.get(object_key, 'unknown').lower()
            try:
                class_id = config.CLASS_NAMES.index(class_title)
            except ValueError:
                class_id = config.NUM_FG_CLASSES - 1

            try:
                geo = figure['geometry'];
                pos = geo['position'];
                dim = geo['dimensions']
                bbox = np.array([pos['x'], pos['y'], -pos['z'], dim['x'], dim['y'], dim['z'], class_id],
                                dtype=np.float32)
                if np.any(bbox[3:6] <= 1e-4): continue
                gt_bboxes_raw.append(bbox)
            except KeyError:
                continue

        if not gt_bboxes_raw:
            return np.empty((0, 6)), np.empty((0,), dtype=int)

        gt_bboxes_raw_np = np.array(gt_bboxes_raw)
        gt_bboxes_denorm = gt_bboxes_raw_np[:, :6]
        gt_classes = gt_bboxes_raw_np[:, 6].astype(int)
        return gt_bboxes_denorm, gt_classes

    except Exception as e:
        print(f"Ошибка загрузки/парсинга аннотации {json_path}: {e}")
        return None, None


# --- Убрали аргумент args ---
def main():
    """Основная функция запуска инференса и оценки."""
    # --- Используем константы, определенные ниже, и конфиг ---
    config = cfg  # Используем импортированный конфиг

    # --- Настройка устройства и загрузка модели ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    model = load_model(MODEL_PATH, config, device)  # Используем константу MODEL_PATH

    # --- Поиск файлов для обработки ---
    if os.path.isdir(PCD_PATH):  # Используем константу PCD_PATH
        pcd_files = sorted(glob.glob(os.path.join(PCD_PATH, '*.pcd')))
        print(f"Найдено {len(pcd_files)} PCD файлов в директории: {PCD_PATH}")
    elif os.path.isfile(PCD_PATH) and PCD_PATH.endswith('.pcd'):
        pcd_files = [PCD_PATH]
        print(f"Обработка одного файла: {PCD_PATH}")
    else:
        print(f"Ошибка: Путь не является файлом PCD или директорией: {PCD_PATH}")
        sys.exit(1)

    if not pcd_files:
        print("Не найдено PCD файлов для обработки.")
        sys.exit(0)

    # --- Создание директорий вывода ---
    vis_output_dir = os.path.join(OUTPUT_DIR, "visualizations")  # Используем константу OUTPUT_DIR
    results_output_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(vis_output_dir, exist_ok=True)
    os.makedirs(results_output_dir, exist_ok=True)

    # --- Инициализация для сбора общих метрик ---
    all_avg_ious = []
    all_f1_macros_presence = []
    processed_files_count = 0

    # --- Цикл обработки файлов ---
    for pcd_path in tqdm(pcd_files, desc="Обработка файлов"):
        base_name = os.path.basename(pcd_path)
        print(f"\n--- Обработка: {base_name} ---")

        # 1. Предобработка
        points_norm_np, points_vis_np = preprocess_point_cloud(pcd_path, config)
        if points_norm_np is None: continue

        # 2. Инференс
        inference_results = run_inference_single_file(model, points_norm_np, device, config)
        if inference_results is None: continue

        # 3. Денормализация предсказаний
        pred_bboxes_denorm_np = denormalize_bbox(
            inference_results['pred_bboxes_norm'], config.POINT_CLOUD_RANGE
        ).cpu().numpy()
        pred_classes_np = inference_results['pred_classes'].cpu().numpy()
        pred_scores_np = inference_results['pred_scores'].cpu().numpy()

        # 4. Загрузка GT
        gt_bboxes_denorm_np, gt_classes_np = load_gt_annotations(pcd_path, config)
        has_gt = gt_bboxes_denorm_np is not None and gt_classes_np is not None

        # 5. Визуализация (если включена)
        vis_file_path = None
        if SAVE_VISUALIZATIONS:  # Используем константу SAVE_VISUALIZATIONS
            vis_suffix = "_norm" if VISUALIZE_NORMALIZED_COORDS else "_orig"  # Используем константу
            vis_file_path = os.path.join(vis_output_dir, f"{os.path.splitext(base_name)[0]}{vis_suffix}.html")

            points_for_vis = None
            if VISUALIZE_NORMALIZED_COORDS:
                points_for_vis = points_norm_np
                pred_bboxes_vis = inference_results['pred_bboxes_norm'].cpu().numpy()
                gt_bboxes_vis = None
                if has_gt and gt_bboxes_denorm_np.shape[0] > 0:
                    gt_bboxes_vis_list = [normalize_bbox(b, config.POINT_CLOUD_RANGE) for b in gt_bboxes_denorm_np]
                    gt_bboxes_vis = np.array(gt_bboxes_vis_list) if gt_bboxes_vis_list else None
            else:
                if points_vis_np is not None:
                    points_vis_np[:, :3] = denormalize_pc_coords(points_vis_np[:, :3], config.POINT_CLOUD_RANGE)
                    points_for_vis = points_vis_np
                pred_bboxes_vis = pred_bboxes_denorm_np
                gt_bboxes_vis = gt_bboxes_denorm_np if has_gt else None

            if points_for_vis is not None and points_for_vis.shape[0] > 0:
                fig = visualize_point_cloud_with_boxes(
                    points=points_for_vis,
                    gt_bboxes=gt_bboxes_vis,
                    pred_bboxes=pred_bboxes_vis,
                    gt_classes=gt_classes_np if has_gt else None,
                    pred_classes=pred_classes_np,
                    title=f"Инференс: {base_name}",
                    coords_normalized=VISUALIZE_NORMALIZED_COORDS
                )
                save_visualization(fig, vis_file_path)
            else:
                print("Нет точек для визуализации.")

        # 6. Расчет и вывод метрик (если есть GT)
        if has_gt:
            pred_classes_original_for_metrics = pred_classes_np - 1

            metrics_iou = calculate_iou_based_metrics(
                gt_bboxes_denorm=gt_bboxes_denorm_np,
                gt_classes=gt_classes_np,
                pred_bboxes_denorm=pred_bboxes_denorm_np,
                pred_classes=pred_classes_original_for_metrics,
                pred_scores=pred_scores_np,
                device=device,
                iou_threshold=config.METRICS_IOU_THRESHOLD,  # Используем порог из конфига
                num_classes=config.NUM_FG_CLASSES
            )
            metrics_presence = calculate_class_presence_metrics(
                gt_classes=gt_classes_np,
                pred_classes=pred_classes_original_for_metrics,
                num_classes=config.NUM_FG_CLASSES
            )

            print(f"  Метрики для файла:")
            print(f"    Avg Matched IoU: {metrics_iou['avg_iou']:.4f}")
            print(f"    Macro F1 (Presence): {metrics_presence['f1_macro_presence']:.4f}")

            all_avg_ious.append(metrics_iou['avg_iou'])
            all_f1_macros_presence.append(metrics_presence['f1_macro_presence'])
        else:
            print("  GT аннотации не найдены, метрики не вычислены.")

        print(f"  Найдено предсказаний: {len(pred_bboxes_denorm_np)}")
        for i in range(len(pred_bboxes_denorm_np)):
            cls_id = int(pred_classes_np[i])
            score = pred_scores_np[i]
            class_name = config.CLASS_NAMES[cls_id - 1] if 1 <= cls_id <= len(
                config.CLASS_NAMES) else f"Unknown({cls_id})"
            print(f"    - Класс: {class_name}, Скор: {score:.3f}")

        processed_files_count += 1

    # --- Вывод итоговых средних метрик ---
    print("\n=========================================")
    print(f"Инференс завершен. Обработано файлов: {processed_files_count}")
    if processed_files_count > 0 and all_avg_ious:
        avg_iou_overall = np.mean(all_avg_ious)
        avg_f1_presence_overall = np.mean(all_f1_macros_presence)
        print(f"--- Итоговые средние метрики (по {len(all_avg_ious)} файлам с GT) ---")
        print(f"Средний Avg Matched IoU: {avg_iou_overall:.4f}")
        print(f"Средний Macro F1 (Presence):       {avg_f1_presence_overall:.4f}")
    else:
        print("Метрики не усреднены (не найдено файлов с GT или ни один файл не обработан).")
    print("=========================================")


if __name__ == "__main__":
    # --- Определяем константы здесь ---

    # --- Пути ---
    MODEL_RUN_NAME = 'fcaf3d_diou_feats3'
    # Имя файла модели
    MODEL_FILENAME = 'final_model.pth'  # Или 'best_model_iou.pth', 'best_model_f1.pth'

    # Строим АБСОЛЮТНЫЙ путь к модели, используя базовую директорию из конфига
    MODEL_PATH = os.path.join(
        cfg.OUTPUT_DIR_BASE,  # Базовая папка output/
        MODEL_RUN_NAME,  # Папка конкретного запуска
        'checkpoints',  # Папка чекпоинтов
        MODEL_FILENAME  # Имя файла
    )

    # Путь к данным для инференса
    PCD_PATH = cfg.DATA_DIR

    # Директория для сохранения результатов этого инференса
    # Создадим подпапку с именем модели, чтобы не путаться
    OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR_BASE, 'inference_results', MODEL_RUN_NAME)

    # --- Параметры инференса и оценки ---
    # Можно брать из конфига cfg или переопределить здесь
    SCORE_THRESHOLD = cfg.SCORE_THRESHOLD
    NMS_IOU_THRESHOLD = cfg.NMS_IOU_THRESHOLD
    METRICS_IOU_THRESHOLD = cfg.METRICS_IOU_THRESHOLD

    # --- Параметры визуализации ---
    SAVE_VISUALIZATIONS = True
    VISUALIZE_NORMALIZED_COORDS = True  # Визуализировать в оригинальных координатах
    # --- Конец определения констант ---

    # Проверка существования путей
    print(f"Проверка пути к модели: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"ОШИБКА: Файл модели не найден по пути: {MODEL_PATH}")
        print(f"Убедись, что имя запуска '{MODEL_RUN_NAME}' и имя файла '{MODEL_FILENAME}' верны.")
        sys.exit(1)
    if not os.path.exists(PCD_PATH):
        print(f"ОШИБКА: Путь к PCD не найден: {PCD_PATH}")
        sys.exit(1)

    # Запускаем основную функцию без аргументов
    main()  # main теперь использует глобальные константы или cfg

    print("\n--- Скрипт инференса завершен ---")
