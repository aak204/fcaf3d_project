import base64
from collections import OrderedDict
from io import BytesIO
import os
import sys
import traceback
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
import plotly.graph_objects as go
import torch
from torchvision import transforms
from torchvision.models import resnet18

# Добавляем корневую директорию проекта в sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from configs import fcaf3d_config as cfg
    from models.fcaf3d import FCAF3D
    from data.preprocessing import (
        normalize_pc, normalize_bbox, clip_points_to_range,
        denormalize_bbox, denormalize_pc_coords
    )
    from data.utils import downsample_point_cloud
    from utils.metrics import nms_3d

    FCAF3D_AVAILABLE = True
except ImportError as e:
    print(f"Ошибка импорта модулей проекта: {e}. Используются ЗАГЛУШКИ.")

# --- Глобальные переменные и константы ---
FIXED_BBOX_2D = (665, 7, 765, 244)
MARGIN = 0.02
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {DEVICE}")

FCAF3D_MODEL_PATH = "final_model.pth"
RESNET18_MODEL_PATH = "best_door_state_classifier.pth"

DOOR_STATE_TRANSLATIONS = {
    "UNKNOWN": "НЕИЗВЕСТНО", "SEMI": "ПРИОТКРЫТА",
    "CLOSED": "ЗАКРЫТА", "OPEN": "ОТКРЫТА"
}
OBJECT_CLASS_TRANSLATIONS = {
    name: name.capitalize() for name in getattr(cfg, 'CLASS_NAMES', [])
}


# --- Загрузка моделей ---
def load_fcaf3d_model_custom(
        model_path: str, config, device: torch.device
) -> FCAF3D:
    print(
        f"Загрузка FCAF3D: {model_path}, "
        f"INPUT_FEAT_DIM={config.INPUT_FEAT_DIM}, "
        f"FP_DIM={config.FP_FEATURE_DIM}"
    )
    model_params = {
        "input_channels": config.INPUT_FEAT_DIM,
        "num_fg_classes": config.NUM_FG_CLASSES,
        "num_levels": config.NUM_LEVELS,
        "fp_feature_dim": config.FP_FEATURE_DIM,
        "pred_head_levels": config.PREDICTION_HEAD_LEVELS
    }
    if not FCAF3D_AVAILABLE or not os.path.exists(model_path):
        print(
            f"ОШИБКА: FCAF3D файл не найден ({model_path}) или модуль не доступен. "
            "Используется заглушка FCAF3D."
        )
        return FCAF3D(**model_params).to(device).eval()
    model = FCAF3D(**model_params).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get(
            'model_state_dict', checkpoint.get('state_dict', checkpoint)
        )
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(
            f"Веса FCAF3D загружены (эпоха {checkpoint.get('epoch', 'N/A')})"
        )
        model.eval()
        return model
    except Exception as e:
        print(f"ОШИБКА загрузки FCAF3D: {e}. Используется заглушка.")
        traceback.print_exc()
        return FCAF3D(**model_params).to(device).eval()


def load_resnet18_model(model_path: str, device: torch.device):
    print(f"Загрузка ResNet18: {model_path}")
    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 4)
    if not os.path.exists(model_path):
        print(
            f"ОШИБКА: Файл ResNet18 не найден: {model_path}. "
            "Модель будет нетренированной."
        )
        model.to(device).eval()
        return model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        print("Модель ResNet18 успешно загружена.")
        return model
    except Exception as e:
        print(
            f"ОШИБКА загрузки ResNet18: {e}. Модель будет нетренированной."
        )
        model.to(device).eval()
        return model


# --- Предобработка облака точек ---
def preprocess_point_cloud_custom(pcd_file_obj, config):
    pcd_path = pcd_file_obj.name
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
                pcd_o3d_norm = o3d.geometry.PointCloud()
                pcd_o3d_norm.points = o3d.utility.Vector3dVector(points_xyz_raw)
                pcd_o3d_norm.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    ),
                    fast_normal_computation=True
                )
                pcd_o3d_norm.normalize_normals()
                normals = np.asarray(pcd_o3d_norm.normals, dtype=np.float32)
                if normals.shape[0] == points_xyz_raw.shape[0]:
                    features_list.append(normals)
                else:
                    features_list.append(np.zeros_like(points_xyz_raw))
            except Exception:
                features_list.append(np.zeros_like(points_xyz_raw))
        if config.USE_INTENSITY_AS_FEATURES:
            intensity = None
            if pcd.has_colors():
                intensity_raw = np.asarray(
                    pcd.colors, dtype=np.float32
                )[:, 0:1]
                intensity = np.clip(intensity_raw, 0.0, 1.0)
            if intensity is not None and \
                    intensity.shape[0] == points_xyz_raw.shape[0]:
                features_list.append(intensity)
            else:
                features_list.append(
                    np.zeros((points_xyz_raw.shape[0], 1), dtype=np.float32)
                )
        if not features_list and config.INPUT_FEAT_DIM > 0:
            features_list.append(
                np.ones(
                    (points_xyz_raw.shape[0], config.INPUT_FEAT_DIM),
                    dtype=np.float32
                )
            )
        if features_list:
            points_processed = np.hstack([points_xyz_raw] + features_list)
        else:
            points_processed = points_xyz_raw
        if points_processed.shape[1] != expected_point_dim:
            if points_processed.shape[1] > expected_point_dim:
                points_processed = points_processed[:, :expected_point_dim]
            else:
                padding_width = expected_point_dim - points_processed.shape[1]
                padding = np.zeros(
                    (points_processed.shape[0], padding_width),
                    dtype=points_processed.dtype
                )
                points_processed = np.hstack([points_processed, padding])
        points_processed = clip_points_to_range(
            points_processed, config.POINT_CLOUD_RANGE
        )
        if points_processed.shape[0] < 10:
            print(
                f"Предупреждение: Менее 10 точек после обрезки: {pcd_path}."
            )
            points_vis_np = points_processed
            return None, points_vis_np
        points_sampled = downsample_point_cloud(
            points_processed, config.DOWNSAMPLING_METHOD, config.MAX_POINTS
        )
        current_n_points = points_sampled.shape[0]
        if current_n_points == 0:
            print(f"Ошибка: 0 точек после даунсэмплинга: {pcd_path}.")
            return None, None
        if current_n_points < config.MAX_POINTS:
            num_to_add = config.MAX_POINTS - current_n_points
            repeat_indices = np.random.choice(
                current_n_points, num_to_add, replace=True
            )
            points_final_np = np.vstack(
                (points_sampled, points_sampled[repeat_indices])
            )
        else:
            points_final_np = points_sampled[:config.MAX_POINTS]
        points_vis_np = points_final_np.copy()
        points_norm_np = normalize_pc(
            points_final_np, config.POINT_CLOUD_RANGE
        )
        return points_norm_np, points_vis_np
    except Exception as e:
        print(f"Ошибка предобработки облака точек {pcd_path}: {e}")
        traceback.print_exc()
        return None, points_vis_np


# --- Инференс FCAF3D ---
def run_fcaf3d_inference_custom(
        model: FCAF3D, points_norm_np: np.ndarray, device: torch.device, config
):
    points_tensor = torch.from_numpy(points_norm_np).float().unsqueeze(0).to(device)
    final_bboxes_norm_tensor = torch.empty(
        (0, 6), device=device, dtype=torch.float
    )
    final_classes_tensor = torch.empty(
        (0,), device=device, dtype=torch.long
    )
    final_scores_tensor = torch.empty(
        (0,), device=device, dtype=torch.float
    )
    default_return = {
        'pred_bboxes_norm': final_bboxes_norm_tensor,
        'pred_classes': final_classes_tensor,
        'pred_scores': final_scores_tensor
    }
    if not FCAF3D_AVAILABLE:
        print("ЗАГЛУШКА: Пропуск инференса FCAF3D.")
        return default_return
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
                if N_pred == 0 or N_pred != points_coords.shape[0]:
                    continue
                cls_prob = torch.softmax(cls_logits, dim=0)
                ctr_prob = torch.sigmoid(ctr_logits).squeeze(0)
                max_fg_cls_prob, pred_fg_cls_idx = torch.max(cls_prob[1:, :], dim=0)
                pred_cls = pred_fg_cls_idx + 1
                score = max_fg_cls_prob * ctr_prob
                valid_mask = score > config.SCORE_THRESHOLD
                if not valid_mask.any():
                    continue
                score_f = score[valid_mask]
                pred_cls_f = pred_cls[valid_mask]
                offset_preds_f = offset_preds[:, valid_mask].T
                logsize_preds_f = logsize_preds[:, valid_mask].T
                points_coords_f = points_coords[valid_mask]
                pred_centers_norm = points_coords_f + offset_preds_f
                pred_sizes_norm = torch.exp(logsize_preds_f)
                pred_bboxes_norm_level = torch.cat(
                    [pred_centers_norm, pred_sizes_norm], dim=1
                )
                for i in range(pred_bboxes_norm_level.shape[0]):
                    preds_before_nms.append([
                        score_f[i], pred_cls_f[i], pred_bboxes_norm_level[i]
                    ])
            if preds_before_nms:
                scores_t = torch.stack([p[0] for p in preds_before_nms])
                classes_t = torch.stack([p[1] for p in preds_before_nms])
                bboxes_t = torch.stack([p[2] for p in preds_before_nms])
                keep_indices = nms_3d(
                    bboxes_t, scores_t, config.NMS_IOU_THRESHOLD
                )
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
        print(f"Ошибка инференса FCAF3D: {e}")
        traceback.print_exc()
        return default_return


# --- Функции для 2D анализа (состояние двери) ---
def crop_with_fixed_bbox(
        image: Image.Image, bbox: tuple = FIXED_BBOX_2D, margin: float = MARGIN
) -> Image.Image:
    x_min, y_min, x_max, y_max = bbox
    width, height = x_max - x_min, y_max - y_min
    delta_w, delta_h = int(width * margin), int(height * margin)
    img_width, img_height = image.size
    new_bbox = (
        max(0, x_min - delta_w),
        max(0, y_min - delta_h),
        min(img_width, x_max + delta_w),
        min(img_height, y_max + delta_h)
    )
    return image.crop(new_bbox)


def draw_bbox_on_image(
        image: Image.Image, bbox: tuple, color: str = "red", width: int = 3
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color, width=width)
    return image


def predict_door_state(
        model, image_input, transform, device: torch.device
):
    cti = {"UNKNOWN": 0, "SEMI": 1, "CLOSED": 2, "OPEN": 3}
    itc = {v: k for k, v in cti.items()}
    if isinstance(image_input, str):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    else:
        raise ValueError(
            f"Неподдерживаемый тип входного изображения: {type(image_input)}"
        )
    cropped_img = crop_with_fixed_bbox(img, FIXED_BBOX_2D, MARGIN)
    input_tensor = transform(cropped_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    predicted_label = itc[predicted_idx.item()]
    annotated_img = img.copy()
    if predicted_label != "UNKNOWN":
        annotated_img = draw_bbox_on_image(annotated_img, FIXED_BBOX_2D)
    return predicted_label, cropped_img, annotated_img


# --- Утилита IoU ---
def calculate_iou_3d(box1_norm: np.ndarray, box2_norm: np.ndarray) -> float:
    box1_min = box1_norm[:3] - box1_norm[3:] / 2
    box1_max = box1_norm[:3] + box1_norm[3:] / 2
    box2_min = box2_norm[:3] - box2_norm[3:] / 2
    box2_max = box2_norm[:3] + box2_norm[3:] / 2
    inter_min = np.maximum(box1_min, box2_min)
    inter_max = np.minimum(box1_max, box2_max)
    inter_size = np.maximum(0.0, inter_max - inter_min)
    intersection_vol = inter_size[0] * inter_size[1] * inter_size[2]
    vol1 = box1_norm[3] * box1_norm[4] * box1_norm[5]
    vol2 = box2_norm[3] * box2_norm[4] * box2_norm[5]
    iou = intersection_vol / (vol1 + vol2 - intersection_vol + 1e-6)
    return iou


# --- Проверка пересечения с порталом ---
def check_fcaf3d_portal_intersection(
        pred_bboxes_norm_np: np.ndarray,
        portal_bbox_norm: np.ndarray,
        iou_threshold: float = 0.05
):
    if pred_bboxes_norm_np is None or pred_bboxes_norm_np.shape[0] == 0:
        return False, 0, []
    intersecting_objects_details = []
    num_intersecting_objects = 0
    is_portal_intersected = False
    for i in range(pred_bboxes_norm_np.shape[0]):
        pred_bbox_norm = pred_bboxes_norm_np[i]
        iou_with_portal = calculate_iou_3d(pred_bbox_norm, portal_bbox_norm)
        if iou_with_portal > iou_threshold:
            is_portal_intersected = True
            num_intersecting_objects += 1
            details = f"Объект FCAF3D (IoU с порталом: {iou_with_portal:.2f})"
            intersecting_objects_details.append(details)
            print(f"Обнаружено пересечение FCAF3D объекта с порталом: {details}")
    return is_portal_intersected, num_intersecting_objects, intersecting_objects_details


# --- Визуализация Plotly ---
def create_plotly_visualization(
        points_norm_for_vis, pred_bboxes_norm_fcaf3d, pred_classes_fcaf3d,
        pred_scores_fcaf3d, portal_bbox_norm, is_portal_intersected_by_fcaf3d,
        config
):
    fig = go.Figure()
    if points_norm_for_vis is not None and points_norm_for_vis.shape[0] > 0:
        fig.add_trace(go.Scatter3d(
            x=points_norm_for_vis[:, 0], y=points_norm_for_vis[:, 1],
            z=points_norm_for_vis[:, 2],
            mode='markers',
            marker=dict(size=1.2, opacity=0.5, color='lightgrey'),
            name="Облако точек (норм.)"
        ))
    if pred_bboxes_norm_fcaf3d is not None and \
            pred_bboxes_norm_fcaf3d.shape[0] > 0:
        colors = ['magenta', 'cyan', 'yellow', 'lime', 'orange', 'blue']
        for i in range(pred_bboxes_norm_fcaf3d.shape[0]):
            bbox_norm = pred_bboxes_norm_fcaf3d[i]
            cls_id_1 = int(pred_classes_fcaf3d[i])
            score = pred_scores_fcaf3d[i]
            orig_cls_name = "Неизв."
            color_idx = 0
            if 1 <= cls_id_1 <= len(config.CLASS_NAMES):
                orig_cls_name = config.CLASS_NAMES[cls_id_1 - 1]
                color_idx = cls_id_1 - 1
            cls_name_disp = OBJECT_CLASS_TRANSLATIONS.get(
                orig_cls_name, orig_cls_name
            )
            color = colors[color_idx % len(colors)]
            center, size = bbox_norm[:3], bbox_norm[3:6]
            x_m, y_m, z_m = center - size / 2
            x_M, y_M, z_M = center + size / 2
            ex, ey, ez = [], [], []
            corners = np.array([
                [x_m, y_m, z_m], [x_M, y_m, z_m], [x_M, y_M, z_m], [x_m, y_M, z_m],
                [x_m, y_m, z_M], [x_M, y_m, z_M], [x_M, y_M, z_M], [x_m, y_M, z_M]
            ])
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)
            ]
            for p1_idx, p2_idx in edges:
                p1, p2 = corners[p1_idx], corners[p2_idx]
                ex.extend([p1[0], p2[0], None])
                ey.extend([p1[1], p2[1], None])
                ez.extend([p1[2], p2[2], None])
            fig.add_trace(go.Scatter3d(
                x=ex, y=ey, z=ez, mode='lines',
                line=dict(color=color, width=2.5),
                name=f"FCAF3D: {cls_name_disp} ({score:.2f})"
            ))
    if portal_bbox_norm is not None:
        center, size = portal_bbox_norm[:3], portal_bbox_norm[3:6]
        x_m, y_m, z_m = center - size / 2
        x_M, y_M, z_M = center + size / 2
        ex_p, ey_p, ez_p = [], [], []
        corners_p = np.array([
            [x_m, y_m, z_m], [x_M, y_m, z_m], [x_M, y_M, z_m], [x_m, y_M, z_m],
            [x_m, y_m, z_M], [x_M, y_m, z_M], [x_M, y_M, z_M], [x_m, y_M, z_M]
        ])
        for p1_idx, p2_idx in edges:  # Используем те же edges
            p1, p2 = corners_p[p1_idx], corners_p[p2_idx]
            ex_p.extend([p1[0], p2[0], None])
            ey_p.extend([p1[1], p2[1], None])
            ez_p.extend([p1[2], p2[2], None])
        portal_color = 'red' if is_portal_intersected_by_fcaf3d else 'green'
        fig.add_trace(go.Scatter3d(
            x=ex_p, y=ey_p, z=ez_p, mode='lines',
            line=dict(color=portal_color, width=3, dash='dash'),
            name="Портал (норм.)"
        ))
    title_text = "Анализ облака точек (нормализованные координаты)"
    title_font_color = 'black'
    if is_portal_intersected_by_fcaf3d:
        title_text += " | ⚠️ ОБЪЕКТ В ЗОНЕ ПОРТАЛА!"
        title_font_color = 'red'
    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(color=title_font_color, size=16)
        ),
        scene=dict(
            xaxis_title=dict(text="X (норм.)", font=dict(color="black")),
            yaxis_title=dict(text="Y (норм.)", font=dict(color="black")),
            zaxis_title=dict(text="Z (норм.)", font=dict(color="black")),
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.25, y=1.25, z=1.25)
            ),
            xaxis=dict(tickfont=dict(color="black")),
            yaxis=dict(tickfont=dict(color="black")),
            zaxis=dict(tickfont=dict(color="black")),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color="black")
        )
    )
    return fig


# --- Вспомогательная функция для рисования 3D bbox в Matplotlib ---
def _plot_3d_bbox_mpl(
        ax, center, size, color='blue', linestyle='-', linewidth=1.5, label=None
):
    cx, cy, cz = center
    sx, sy, sz = size
    _x = [cx - sx / 2, cx + sx / 2, cx + sx / 2, cx - sx / 2, cx - sx / 2, cx + sx / 2, cx + sx / 2, cx - sx / 2]
    _y = [cy - sy / 2, cy - sy / 2, cy + sy / 2, cy + sy / 2, cy - sy / 2, cy - sy / 2, cy + sy / 2, cy + sy / 2]
    _z = [cz - sz / 2, cz - sz / 2, cz - sz / 2, cz - sz / 2, cz + sz / 2, cz + sz / 2, cz + sz / 2, cz + sz / 2]
    corners_ordered = np.array([
        [_x[0], _y[0], _z[0]], [_x[1], _y[1], _z[1]], [_x[2], _y[2], _z[2]], [_x[3], _y[3], _z[3]],
        [_x[4], _y[4], _z[4]], [_x[5], _y[5], _z[5]], [_x[6], _y[6], _z[6]], [_x[7], _y[7], _z[7]]
    ])
    plot_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    plotted_with_label = False
    for i_start, i_end in plot_edges:
        p1 = corners_ordered[i_start]
        p2 = corners_ordered[i_end]
        current_label = None
        if label and not plotted_with_label:
            current_label = label
            plotted_with_label = True
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, linestyle=linestyle, linewidth=linewidth, label=current_label
        )


# --- Визуализация Matplotlib 3D ---
def create_matplotlib_visualization_3d(
        points_norm_for_vis, pred_bboxes_norm_fcaf3d,
        portal_bbox_norm, is_portal_intersected_by_fcaf3d,
):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(
        "3D Визуализация (нормализованные координаты)",
        color='black', fontsize=14, pad=20
    )
    ax.set_xlabel("X (норм.)", color='black')
    ax.set_ylabel("Y (норм.)", color='black')
    ax.set_zlabel("Z (норм.)", color='black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.tick_params(axis='z', colors='black')
    ax.grid(True, linestyle='--', alpha=0.6)
    if points_norm_for_vis is not None and points_norm_for_vis.shape[0] > 0:
        ax.scatter(
            points_norm_for_vis[:, 0], points_norm_for_vis[:, 1],
            points_norm_for_vis[:, 2],
            s=1.5, c='lightgrey', alpha=0.5, label="Облако точек"
        )
    if pred_bboxes_norm_fcaf3d is not None and \
            pred_bboxes_norm_fcaf3d.shape[0] > 0:
        for i in range(pred_bboxes_norm_fcaf3d.shape[0]):
            bbox_norm = pred_bboxes_norm_fcaf3d[i]
            _plot_3d_bbox_mpl(
                ax, bbox_norm[:3], bbox_norm[3:],
                color='magenta', linewidth=1.5,
                label="FCAF3D объект" if i == 0 else None
            )
    if portal_bbox_norm is not None:
        portal_color = 'red' if is_portal_intersected_by_fcaf3d else 'green'
        _plot_3d_bbox_mpl(
            ax, portal_bbox_norm[:3], portal_bbox_norm[3:],
            color=portal_color, linestyle='--', linewidth=2,
            label="Портал"
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)
    ax.view_init(elev=20., azim=-65)
    legend = ax.legend(loc='upper right')
    for text in legend.get_texts():
        text.set_color('black')
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_data_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_data_base64}"


# --- Основная функция анализа ---
@torch.no_grad()
def analyze_safety(image_pil: Image.Image, pcd_file_obj):
    """
    Анализирует безопасность двери на основе изображения и облака точек.
    Возвращает HTML-отчет и путь к файлу Plotly для Gradio.
    """
    plotly_file_path_for_gradio = None  # Инициализация
    try:
        # 0. Загрузка моделей (можно вынести для однократной загрузки)
        fcaf3d_model = load_fcaf3d_model_custom(FCAF3D_MODEL_PATH, cfg, DEVICE)
        resnet_model = load_resnet18_model(RESNET18_MODEL_PATH, DEVICE)
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])

        # 1. Анализ состояния двери
        door_state_internal, _, annotated_door_img = predict_door_state(
            resnet_model, image_pil, data_transform, DEVICE
        )
        door_state_display = DOOR_STATE_TRANSLATIONS.get(
            door_state_internal, door_state_internal
        )

        # 2. Предобработка облака точек
        points_norm_for_fcaf3d_and_vis, _ = \
            preprocess_point_cloud_custom(pcd_file_obj, cfg)

        if points_norm_for_fcaf3d_and_vis is None:
            error_html = (
                "<div style='color:red; padding:15px; border:1px solid red;'>"
                "<h2>Ошибка обработки облака точек</h2>"
                "<p>Не удалось получить нормализованные точки. "
                "Проверьте файл облака точек и конфигурацию.</p></div>"
            )
            return error_html, None  # Возвращаем HTML и None для gr.File

        # 3. Инференс FCAF3D
        inference_results_fcaf3d = run_fcaf3d_inference_custom(
            fcaf3d_model, points_norm_for_fcaf3d_and_vis, DEVICE, cfg
        )
        pred_bboxes_norm_fcaf3d = \
            inference_results_fcaf3d['pred_bboxes_norm'].cpu().numpy()
        pred_classes_fcaf3d = \
            inference_results_fcaf3d['pred_classes'].cpu().numpy()
        pred_scores_fcaf3d = \
            inference_results_fcaf3d['pred_scores'].cpu().numpy()

        # 4. Определение и НОРМАЛИЗАЦИЯ bbox'а портала
        bbox_corners_world_input_z_inv = np.array([
            [-0.081, -0.375, -3.0], [-1.02, -0.558, -3.0],
            [-0.407, -0.152, -1.4], [-0.709, -0.065, -1.4]
        ])
        bbox_min_portal_world_z_inv = np.min(
            bbox_corners_world_input_z_inv, axis=0
        )
        bbox_max_portal_world_z_inv = np.max(
            bbox_corners_world_input_z_inv, axis=0
        )
        portal_center_world_z_inv = \
            (bbox_min_portal_world_z_inv + bbox_max_portal_world_z_inv) / 2
        portal_size_world_z_inv = \
            bbox_max_portal_world_z_inv - bbox_min_portal_world_z_inv
        portal_bbox_world_for_norm_z_inv = np.concatenate([
            portal_center_world_z_inv, portal_size_world_z_inv
        ])
        portal_bbox_norm = normalize_bbox(
            portal_bbox_world_for_norm_z_inv, cfg.POINT_CLOUD_RANGE
        )

        # 5. Логика определения опасности
        is_overall_dangerous = False
        num_intersecting_fcaf3d_objects = 0
        intersecting_details_text = "Нет пересекающихся объектов FCAF3D."

        is_portal_intersected_by_fcaf3d, num_intersecting, intersecting_details = \
            check_fcaf3d_portal_intersection(
                pred_bboxes_norm_fcaf3d, portal_bbox_norm, iou_threshold=0.05
            )
        num_intersecting_fcaf3d_objects = num_intersecting
        if intersecting_details:
            intersecting_details_text = ", ".join(intersecting_details)

        if door_state_internal in ["CLOSED", "SEMI"]:
            if is_portal_intersected_by_fcaf3d:
                is_overall_dangerous = True

        # 6. Визуализация Plotly (сохранение в файл)
        plotly_fig = create_plotly_visualization(
            points_norm_for_fcaf3d_and_vis[:, :3],
            pred_bboxes_norm_fcaf3d,
            pred_classes_fcaf3d,
            pred_scores_fcaf3d,
            portal_bbox_norm,
            is_portal_intersected_by_fcaf3d,  # Для цвета портала в Plotly
            cfg
        )
        plotly_html_filename = "door_safety_plotly_norm.html"
        plotly_file_path_for_gradio = os.path.join(os.getcwd(), plotly_html_filename)

        try:
            plotly_fig.write_html(
                plotly_file_path_for_gradio, full_html=True, include_plotlyjs='cdn'
            )
            plotly_status_message = (
                "<p style='margin-top:10px; font-size:0.9em; color:#333;'>"
                "Интерактивный 3D график Plotly будет доступен как отдельный файл ниже."
                "</p>"
            )
        except Exception as e_save:
            print(f"Ошибка сохранения Plotly HTML: {e_save}")
            plotly_status_message = (
                f"<p style='color:orange; font-size:0.9em;'>"
                f"Не удалось сохранить Plotly HTML: {e_save}</p>"
            )
            plotly_file_path_for_gradio = None

        # 7. Визуализация Matplotlib 3D
        matplotlib_img_base64_src = create_matplotlib_visualization_3d(
            points_norm_for_fcaf3d_and_vis[:, :3],
            pred_bboxes_norm_fcaf3d,
            portal_bbox_norm,
            is_portal_intersected_by_fcaf3d
        )

        # 8. Подготовка изображения двери с аннотацией
        plt.figure(figsize=(6, 4.5))
        plt.imshow(annotated_door_img)
        title_matplotlib = f"Состояние двери: {door_state_display}"
        title_color = 'black'
        if is_overall_dangerous:
            title_matplotlib += "\n⚠️ ОБЪЕКТ В ЗОНЕ ПОРТАЛА!"
            title_color = 'red'
        elif door_state_internal == "UNKNOWN":
            title_color = 'darkorange'
        elif not is_portal_intersected_by_fcaf3d and \
                door_state_internal in ["CLOSED", "SEMI"]:
            title_color = 'green'
        plt.title(
            title_matplotlib, color=title_color, fontsize=13, pad=10
        )
        plt.axis("off")
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        annotated_img_base64 = base64.b64encode(
            buf.getvalue()
        ).decode('utf-8')

        # 9. Формирование HTML отчета
        safety_status_text = "✅ Ситуация безопасна"
        safety_status_color = 'green'
        if is_overall_dangerous:
            safety_status_text = "⚠️ ОПАСНАЯ СИТУАЦИЯ! ⚠️"
            safety_status_color = 'red'

        final_assessment_text = ""
        if is_overall_dangerous:
            final_assessment_text = (
                f"Дверь \"{door_state_display}\". "
                f"В зоне портала обнаружен(ы) объект(ы) FCAF3D "
                f"({num_intersecting_fcaf3d_objects} шт.). "
                f"Детали: {intersecting_details_text}."
            )
        elif door_state_internal in ["UNKNOWN", "OPEN"]:
            final_assessment_text = (
                f"Дверь \"{door_state_display}\". Ситуация считается безопасной. "
                f"(Объектов FCAF3D в/у портала: {num_intersecting_fcaf3d_objects}. "
                f"Детали: {intersecting_details_text})"
            )
        else:
            final_assessment_text = (
                f"Опасность не обнаружена. Дверь \"{door_state_display}\". "
                f"(Объектов FCAF3D в/у портала: {num_intersecting_fcaf3d_objects}. "
                f"Детали: {intersecting_details_text})"
            )

        result_html = f"""
        <div style="padding:15px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333;">
            <h2 style="color:{safety_status_color}; text-align:center; margin-bottom:20px; font-size:1.8em;">
                {safety_status_text}
            </h2>
            <div style="display:flex; flex-wrap:wrap; gap:20px; margin-bottom:20px; align-items: flex-start;">
                <div style="flex:1; min-width:300px; padding:15px; border:1px solid #e0e0e0; border-radius:8px; background-color:#f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin-top:0; font-size:1.3em; color:#1a1a1a; border-bottom: 1px solid #eee; padding-bottom:8px;">
                        Анализ состояния двери
                    </h3>
                    <p style="font-size:1.1em;"><b>Состояние:</b> <span style="font-weight:bold; color:{title_color};">{door_state_display}</span></p>
                    <img src="data:image/png;base64,{annotated_img_base64}" alt="Аннотированное изображение двери" style="max-width:100%; height:auto; border-radius:5px; border: 1px solid #ddd;">
                </div>
                <div style="flex:1.5; min-width:320px; padding:15px; border:1px solid #e0e0e0; border-radius:8px; background-color:#f9f9f9; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h3 style="margin-top:0; font-size:1.3em; color:#1a1a1a; border-bottom: 1px solid #eee; padding-bottom:8px;">
                        Анализ облака точек (3D, норм. коорд.)
                    </h3>
                    <p style="font-size:1.0em;"><b>Объектов FCAF3D в/у портала:</b> {num_intersecting_fcaf3d_objects}</p>
                    <p style="font-size:1.0em;"><b>Детали пересечений FCAF3D:</b> {intersecting_details_text}</p>
                    <img src="{matplotlib_img_base64_src}" alt="3D визуализация облака точек" style="max-width:100%; height:auto; border-radius:5px; border: 1px solid #ddd; margin-top:10px;">
                    {plotly_status_message}
                </div>
            </div>
            <div style="padding:15px; background-color:{'rgba(255,0,0,0.08)' if is_overall_dangerous else 'rgba(0,128,0,0.08)'}; border-radius:8px; border: 1px solid {'rgba(255,0,0,0.2)' if is_overall_dangerous else 'rgba(0,128,0,0.2)'};">
                <h3 style="margin-top:0; font-size:1.3em; color:#1a1a1a;">Итоговая оценка</h3>
                <p style="font-size:1.1em; font-weight:bold; color:{safety_status_color};">
                    {final_assessment_text}
                </p>
            </div>
        </div>
        """
        return result_html, plotly_file_path_for_gradio

    except FileNotFoundError as e:
        error_msg = (
            f"<h2>Ошибка: Файл модели не найден</h2><p>{e}.</p>"
            "<p>Убедитесь, что файлы моделей "
            f"'{FCAF3D_MODEL_PATH}' и '{RESNET18_MODEL_PATH}' "
            "находятся в корневой директории проекта.</p>"
        )
        error_html = (
            f"<div style='color:red; padding:15px; border:1px solid red; "
            f"background-color: #fff0f0;'>{error_msg}</div>"
        )
        return error_html, None
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Критическая ошибка в analyze_safety: {e}\n{error_trace}")
        error_msg = (
            f"<h2>Критическая ошибка анализа</h2><p>{e}</p>"
            f"<details><summary>Трассировка стека</summary>"
            f"<pre style='white-space: pre-wrap; word-wrap: break-word; "
            f"background-color:#f5f5f5; padding:10px; border-radius:4px; "
            f"border:1px solid #ccc;'>{error_trace}</pre></details>"
        )
        error_html = (
            f"<div style='color:red; padding:15px; border:1px solid red; "
            f"background-color: #fff0f0;'>{error_msg}</div>"
        )
        return error_html, None


# --- Интерфейс Gradio ---
def create_gradio_interface():
    """Создает и настраивает интерфейс Gradio."""
    with gr.Blocks(
            title="Система Анализа Безопасности Двери",
            theme=gr.themes.Soft(
                primary_hue="blue", secondary_hue="sky", neutral_hue="slate"
            )
    ) as interface:
        gr.Markdown(
            "<h1 style='text-align:center; color:#2c3e50;'>Система Анализа Безопасности Двери</h1>"
        )
        gr.Markdown(
            "<p style='text-align:center; color:#555; font-size:1.1em;'>"
            "Загрузите изображение двери и файл облака точек (.pcd) для "
            "комплексного анализа безопасности.</p>"
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("### 1. Входные данные")
                image_input = gr.Image(
                    label="Изображение двери",
                    type="pil",
                    height=300,
                    interactive=True
                )
                pcd_input = gr.File(
                    label="Облако точек (.pcd)",
                    file_types=['.pcd'],
                    interactive=True
                )
                analyze_btn = gr.Button(
                    "🚀 Анализировать безопасность",
                    variant="primary",
                    scale=2
                )
            with gr.Column(scale=2, min_width=600):
                gr.Markdown("### 2. Результаты анализа")
                output_html = gr.HTML(label="Отчет по безопасности")
                plotly_file_output = gr.File(
                    label="Интерактивный 3D график Plotly (HTML)",
                    interactive=False
                )

        analyze_btn.click(
            fn=analyze_safety,
            inputs=[image_input, pcd_input],
            outputs=[output_html, plotly_file_output]
        )

        with gr.Accordion("ℹ️ Дополнительная информация", open=False):
            gr.Markdown(
                "--- \n"
                "#### Как это работает:\n"
                "1.  **Анализ изображения двери**: Модель ResNet18 определяет "
                "состояние двери (Открыта, Закрыта, Приоткрыта, Неизвестно).\n"
                "2.  **Анализ облака точек**: Модель FCAF3D обнаруживает "
                "объекты в 3D-пространстве.\n"
                "3.  **Оценка безопасности**: Система проверяет, есть ли "
                "объекты FCAF3D в предопределенной зоне портала, если дверь "
                "закрыта или приоткрыта.\n"
                "\n"
                "#### Требования:\n"
                f"-   Модель детекции объектов: `{FCAF3D_MODEL_PATH}`\n"
                f"-   Модель классификации состояния двери: `{RESNET18_MODEL_PATH}`\n"
                "    (Убедитесь, что эти файлы находятся в корневой директории "
                "проекта или по указанным путям).\n"
                "-   Конфигурационные файлы и зависимости проекта должны быть "
                "доступны.\n"
                "\n"
                "#### Примечание:\n"
                "Точность анализа зависит от качества моделей и входных данных."
            )
    return interface


if __name__ == "__main__":
    for mp_check in [FCAF3D_MODEL_PATH, RESNET18_MODEL_PATH]:
        if not os.path.exists(mp_check):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Модель '{mp_check}' не найдена.")

    current_working_directory = os.getcwd()
    print(f"Текущая рабочая директория: {current_working_directory}")
    print(
        "Файлы, создаваемые в этой директории (например, Plotly HTML), "
        "будут доступны через компонент gr.File."
    )

    demo = create_gradio_interface()
    demo.launch(
        share=False,
        inbrowser=True,
        allowed_paths=[current_working_directory]
    )
