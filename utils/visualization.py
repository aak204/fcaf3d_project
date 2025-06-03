"""Функции для визуализации с использованием Plotly."""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple
from configs import fcaf3d_config as cfg  # Для CLASS_NAMES


def add_bbox_to_plotly(fig: go.Figure, bbox: np.ndarray, color: str = 'green', name: str = "Box"):
    """Добавляет один axis-aligned bbox на график Plotly."""
    center = bbox[:3]
    # Используем abs, т.к. размеры могут быть отрицательными из-за ошибок предсказания
    size = np.maximum(np.abs(bbox[3:6]), 1e-6)
    x_min, y_min, z_min = center - size / 2
    x_max, y_max, z_max = center + size / 2

    # Определяем 8 углов кубоида
    corners = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])

    # Определяем 12 ребер кубоида
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Нижняя грань
        (4, 5), (5, 6), (6, 7), (7, 4),  # Верхняя грань
        (0, 4), (1, 5), (2, 6), (3, 7)  # Соединяющие ребра
    ]

    edge_x, edge_y, edge_z = [], [], []
    for p1_idx, p2_idx in edges:
        p1 = corners[p1_idx]
        p2 = corners[p2_idx]
        edge_x.extend([p1[0], p2[0], None])  # None для разрыва линии
        edge_y.extend([p1[1], p2[1], None])
        edge_z.extend([p1[2], p2[2], None])

    fig.add_trace(go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color=color, width=3),
        name=name,
        showlegend=True
    ))


def visualize_point_cloud_with_boxes(
        points: np.ndarray,
        gt_bboxes: Optional[np.ndarray] = None,
        pred_bboxes: Optional[np.ndarray] = None,
        gt_classes: Optional[np.ndarray] = None,
        pred_classes: Optional[np.ndarray] = None,  # Ожидаются классы 1, 2...
        title: str = "Point Cloud Visualization",
        coords_normalized: bool = False  # Флаг для подписи осей
) -> go.Figure:
    """
    Создает интерактивную 3D визуализацию облака точек с GT и Pred боксами.

    Args:
        points (np.ndarray): Облако точек [N, D], где первые 3 столбца - XYZ.
        gt_bboxes (Optional[np.ndarray]): GT боксы [N_gt, 6] (cx, cy, cz, w, h, l).
        pred_bboxes (Optional[np.ndarray]): Предсказанные боксы [N_pred, 6].
        gt_classes (Optional[np.ndarray]): GT классы [N_gt] (индексы 0, 1...).
        pred_classes (Optional[np.ndarray]): Предсказанные классы [N_pred] (индексы 1, 2...).
        title (str): Заголовок графика.
        coords_normalized (bool): Указывает, нормализованы ли координаты.

    Returns:
        go.Figure: Объект фигуры Plotly.
    """
    fig = go.Figure()

    # --- Отображение точек ---
    marker_colors = 'lightblue'  # Цвет по умолчанию
    # Попробуем раскрасить по первой фиче (если есть)
    if points.shape[1] > 3 and cfg.INPUT_FEAT_DIM > 0:
        feature_vals = points[:, 3]
        # Нормализуем фичу для цвета (если нужно)
        # feature_vals = (feature_vals - np.min(feature_vals)) / (np.max(feature_vals) - np.min(feature_vals) + 1e-6)
        marker_colors = feature_vals

    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=marker_colors,
            colorscale='Viridis',  # Цветовая схема
            opacity=0.7
        ),
        name="Points"
    ))

    # --- Вспомогательная функция для имен и цветов классов ---
    def _get_class_name_color(cls_idx: Optional[int], is_gt: bool) -> Tuple[str, str]:
        colors_gt = ['darkgreen', 'limegreen', 'darkolivegreen', 'chartreuse']  # Добавь цвета, если классов > 2
        colors_pred = ['darkred', 'coral', 'firebrick', 'orangered']
        default_color = 'gray'
        class_names = cfg.CLASS_NAMES

        if is_gt:
            if cls_idx is not None and 0 <= cls_idx < len(class_names):
                name = class_names[cls_idx]
                color = colors_gt[cls_idx % len(colors_gt)]
            else:
                name = f'unknown_gt_{cls_idx}'
                color = default_color
        else:  # Предсказания (индексы 1, 2...)
            fg_cls_idx = int(cls_idx) - 1 if cls_idx is not None else -1
            if 0 <= fg_cls_idx < len(class_names):
                name = class_names[fg_cls_idx]
                color = colors_pred[fg_cls_idx % len(colors_pred)]
            else:
                name = f'unknown_pred_{cls_idx}'
                color = default_color
        return name, color

    # --- Отображение GT боксов ---
    if gt_bboxes is not None and gt_bboxes.shape[0] > 0:
        for i, bbox in enumerate(gt_bboxes):
            cls_idx_gt = int(gt_classes[i]) if gt_classes is not None and i < len(gt_classes) else None
            cls_name, color = _get_class_name_color(cls_idx_gt, is_gt=True)
            add_bbox_to_plotly(fig, bbox[:6], color, f"GT {i + 1} ({cls_name})")

    # --- Отображение Pred боксов ---
    if pred_bboxes is not None and pred_bboxes.shape[0] > 0:
        for i, bbox in enumerate(pred_bboxes):
            # Предсказанные классы идут как 1, 2...
            cls_idx_pred = int(pred_classes[i]) if pred_classes is not None and i < len(pred_classes) else None
            if cls_idx_pred == 0: continue  # Пропускаем фон, если он вдруг предсказан
            cls_name, color = _get_class_name_color(cls_idx_pred, is_gt=False)
            add_bbox_to_plotly(fig, bbox[:6], color, f"Pred {i + 1} ({cls_name})")

    # --- Настройка лейаута ---
    gt_count = gt_bboxes.shape[0] if gt_bboxes is not None else 0
    pred_count = pred_bboxes.shape[0] if pred_bboxes is not None else 0
    axis_suffix = " (Normalized)" if coords_normalized else ""
    xaxis_title = f"X{axis_suffix}"
    yaxis_title = f"Y{axis_suffix}"
    zaxis_title = f"Z{axis_suffix}"

    fig.update_layout(
        title=f"{title} - GT:{gt_count}, Pred:{pred_count}",
        scene=dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
            aspectmode='data'  # Сохраняет пропорции осей
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def save_visualization(fig: go.Figure, filename: str):
    """Сохраняет фигуру Plotly в HTML файл."""
    try:
        fig.write_html(filename)
        print(f"Визуализация сохранена в: {filename}")
    except Exception as e:
        print(f"Ошибка сохранения визуализации в {filename}: {e}")
