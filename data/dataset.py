"""Dataset class и collate_fn для загрузки данных."""

import os
import glob
import json
import traceback
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import Dataset
from tqdm import tqdm

from configs import fcaf3d_config as cfg
from .preprocessing import (
    normalize_pc, normalize_bbox, clip_points_to_range
)
from .utils import downsample_point_cloud
from .augmentations import (
    rotate_points_along_z, rotate_gt_bboxes_along_z, global_scale,
    random_flip_along_axis, point_cloud_noise, random_point_dropout
)


class PointCloudDatasetFCAF3D(Dataset):
    """Dataset для загрузки облаков точек и аннотаций FCAF3D."""

    def __init__(
            self,
            data_dir: str,
            pc_range: List[float],
            input_feat_dim: int = cfg.INPUT_FEAT_DIM,
            use_normals: bool = cfg.USE_NORMALS_AS_FEATURES,
            use_intensity: bool = cfg.USE_INTENSITY_AS_FEATURES,
            class_names: List[str] = cfg.CLASS_NAMES,
            transform: bool = True,
            test_mode: bool = False,
            fixed_points: bool = True,
            max_points: int = cfg.MAX_POINTS,
            downsampling_method: str = cfg.DOWNSAMPLING_METHOD,
            # Параметры аугментаций из конфига
            random_rotation: bool = True,
            rotation_range: Tuple[float, float] = cfg.AUG_ROTATION_RANGE,
            random_scaling: bool = True,
            scale_range: Tuple[float, float] = cfg.AUG_SCALE_RANGE,
            random_flip: bool = True,
            flip_prob: float = cfg.AUG_FLIP_PROB,
            add_noise: bool = True,
            noise_std: float = cfg.AUG_NOISE_STD,
            random_dropout: bool = True,
            dropout_ratio: float = cfg.AUG_DROPOUT_RATIO
    ):
        super().__init__()
        self.data_dir = data_dir
        self.pc_range = np.array(pc_range)  # Используем numpy array для удобства
        self.input_feat_dim = input_feat_dim
        self.use_normals = use_normals
        self.use_intensity = use_intensity
        self.class_names = class_names
        self.num_classes = len(class_names)
        # Ожидаемая размерность точки: 3 (XYZ) + input_feat_dim
        self.expected_point_dim = 3 + self.input_feat_dim
        self.transform = transform and not test_mode
        self.test_mode = test_mode
        self.fixed_points = fixed_points
        self.max_points = max_points
        self.downsampling_method = downsampling_method
        # Параметры аугментаций
        self.random_rotation = random_rotation
        self.rotation_range = rotation_range
        self.random_scaling = random_scaling
        self.scale_range = scale_range
        self.random_flip = random_flip
        self.flip_prob = flip_prob
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.random_dropout = random_dropout
        self.dropout_ratio = dropout_ratio

        self.file_list, self.bbox_list = self._load_annotations()

    def _load_annotations(self) -> Tuple[List[str], List[np.ndarray]]:
        """Загружает список файлов и аннотаций."""
        pcd_files = sorted(glob.glob(os.path.join(self.data_dir, "*.pcd")))
        if not pcd_files:
            raise ValueError(f"В директории {self.data_dir} не найдено PCD файлов")

        file_list = []
        bbox_list = []
        print(f"Найдено {len(pcd_files)} PCD файлов. Загрузка аннотаций...")
        skipped_files_count = 0
        files_with_valid_objects = 0
        total_valid_objects = 0
        class_counts = {i: 0 for i in range(self.num_classes)}

        for pcd_file in tqdm(pcd_files, desc="Загрузка аннотаций"):
            json_file = pcd_file + ".json"
            if not os.path.exists(json_file):
                skipped_files_count += 1
                continue

            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    annotation = json.load(f)
            except Exception as e:
                print(f"Ошибка чтения JSON {json_file}: {e}")
                skipped_files_count += 1
                continue

            # Проверяем наличие необходимых полей
            if 'objects' not in annotation or 'figures' not in annotation:
                # print(f"Пропуск {json_file}: отсутствуют 'objects' или 'figures'.")
                skipped_files_count += 1
                continue

            objects_dict = {obj['key']: obj['classTitle'] for obj in annotation['objects']}
            file_bboxes = []
            has_valid_object_in_file = False

            for figure in annotation['figures']:
                if figure.get('geometryType') != 'cuboid_3d':
                    continue
                object_key = figure.get('objectKey')
                if not object_key or object_key not in objects_dict:
                    continue

                class_title = objects_dict[object_key].lower()
                try:
                    # Классы 0, 1...
                    class_id = self.class_names.index(class_title)
                except ValueError:
                    # Если класс не найден, присваиваем последний класс (например, 'other')
                    class_id = self.num_classes - 1

                try:
                    geo = figure['geometry']
                    pos = geo['position']
                    dim = geo['dimensions']
                    # Создаем 7D бокс (cx, cy, cz, w, h, l, class_id)
                    # Инвертируем Z координату центра при загрузке
                    bbox = np.array([
                        pos['x'], pos['y'], -pos['z'],
                        dim['x'], dim['y'], dim['z'],
                        class_id
                    ], dtype=np.float32)

                    # Пропускаем вырожденные боксы
                    if np.any(bbox[3:6] <= 1e-4):
                        continue

                    file_bboxes.append(bbox)
                    has_valid_object_in_file = True
                    total_valid_objects += 1
                    if class_id in class_counts:
                        class_counts[class_id] += 1
                except KeyError as e:
                    print(f"Предупреждение: отсутствует ключ {e} в {pcd_file}")
                    continue

            # Добавляем файл в список, даже если в нем нет объектов (для обучения на фоне)
            # Но только если сам PCD файл существует и не пустой
            if os.path.exists(pcd_file):  # and os.path.getsize(pcd_file) > 1000:
                files_with_valid_objects += 1
                file_list.append(pcd_file)
                bbox_list.append(np.array(file_bboxes) if file_bboxes else np.empty((0, 7), dtype=np.float32))
            else:
                skipped_files_count += 1

        print(f"Загружено {total_valid_objects} валидных 3D объектов из {files_with_valid_objects} файлов.")
        num_skipped = len(pcd_files) - files_with_valid_objects
        print(f"Пропущено/пустых файлов: {num_skipped}")
        gt_class_str = ", ".join(
            [f"{self.class_names[i]}({i})={class_counts.get(i, 0)}" for i in range(self.num_classes)])
        print(f"Распределение GT по классам: {gt_class_str}")
        if files_with_valid_objects == 0:
            raise RuntimeError("Не найдено валидных файлов для обучения!")

        return file_list, bbox_list

    def __len__(self) -> int:
        return len(self.file_list)

    def _get_empty_item(self, pcd_file: str = "unknown") -> Dict:
        """Возвращает пустой элемент с правильной размерностью."""
        return {
            'points': torch.zeros((self.max_points, self.expected_point_dim), dtype=torch.float32),
            'gt_bboxes': torch.zeros((0, 7), dtype=torch.float32),
            'file_path': pcd_file,
            'num_objects': 0
        }

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """Загружает и обрабатывает один элемент датасета."""
        pcd_file = self.file_list[idx]
        bboxes_gt_raw = self.bbox_list[idx].copy()  # [N_gt, 7]
        num_objects = len(bboxes_gt_raw)

        try:
            pcd = o3d.io.read_point_cloud(pcd_file)
            if not pcd.has_points():
                # print(f"Предупреждение: Пустое облако точек {pcd_file}")
                # Возвращаем пустой элемент, если облако пустое, но аннотации могут быть (редко)
                return self._get_empty_item(pcd_file)

            points_xyz = np.asarray(pcd.points, dtype=np.float32)
            # Инвертируем Z координату точек
            points_xyz[:, 2] = -points_xyz[:, 2]

            features = []
            # 1. Добавляем нормали
            if self.use_normals:
                try:
                    # Используем копию для вычисления нормалей
                    pcd_o3d_norm = o3d.geometry.PointCloud()
                    pcd_o3d_norm.points = o3d.utility.Vector3dVector(points_xyz)
                    pcd_o3d_norm.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
                        fast_normal_computation=True
                    )
                    pcd_o3d_norm.normalize_normals()
                    normals = np.asarray(pcd_o3d_norm.normals, dtype=np.float32)
                    if normals.shape[0] == points_xyz.shape[0]:
                        features.append(normals)
                    else:
                        features.append(np.zeros_like(points_xyz))
                except Exception:
                    features.append(np.zeros_like(points_xyz))

            # 2. Добавляем Интенсивность (если нужно и есть)
            if self.use_intensity:
                intensity = None
                if pcd.has_colors():  # Пример: интенсивность из первого канала RGB
                    intensity_raw = np.asarray(pcd.colors, dtype=np.float32)[:, 0:1]
                    # Нормализация, если нужно (например, / 255.0)
                    intensity = np.clip(intensity_raw, 0.0, 1.0)

                if intensity is not None and intensity.shape[0] == points_xyz.shape[0]:
                    features.append(intensity)
                else:
                    features.append(np.zeros((points_xyz.shape[0], 1), dtype=np.float32))

            # 3. Если фичей нет, но input_feat_dim > 0, добавляем фиктивные
            if not features and self.input_feat_dim > 0:
                features.append(np.ones((points_xyz.shape[0], self.input_feat_dim), dtype=np.float32))

            # Собираем точки: XYZ + фичи
            points = np.hstack([points_xyz] + features) if features else points_xyz

            # Проверка размерности
            if points.shape[1] != self.expected_point_dim:
                if points.shape[1] > self.expected_point_dim:
                    points = points[:, :self.expected_point_dim]
                else:
                    padding = np.zeros((points.shape[0], self.expected_point_dim - points.shape[1]), dtype=points.dtype)
                    points = np.hstack([points, padding])

        except Exception as e:
            print(f"Ошибка обработки PCD {pcd_file}: {e}")
            traceback.print_exc()
            return None  # Возвращаем None, collate_fn отфильтрует

        # --- Применение Аугментаций (если режим transform) ---
        if self.transform:
            if self.random_rotation:
                angle = np.random.uniform(self.rotation_range[0], self.rotation_range[1])
                points = rotate_points_along_z(points, angle)
                bboxes_gt_raw = rotate_gt_bboxes_along_z(bboxes_gt_raw, angle)
            if self.random_scaling:
                points, bboxes_gt_raw = global_scale(points, bboxes_gt_raw, self.scale_range)
            # Отражение по X
            points, bboxes_gt_raw = random_flip_along_axis(
                points, bboxes_gt_raw, axis=0, flip_prob=self.flip_prob
            )

        # --- Обрезка по диапазону ПОСЛЕ аугментаций геометрии ---
        points = clip_points_to_range(points, self.pc_range)
        if points.shape[0] < 10:
            return None  # Слишком мало точек

        # --- Аугментации на точках (Шум, Дропаут) ПОСЛЕ обрезки ---
        if self.transform:
            if self.add_noise and self.noise_std > 0:
                # Шум добавляем к НЕнормализованным координатам
                points = point_cloud_noise(points, self.noise_std, self.pc_range)
                # Снова обрежем, если шум вывел точки за пределы
                points = clip_points_to_range(points, self.pc_range)
                if points.shape[0] < 10: return None

            if self.random_dropout and self.dropout_ratio > 0:
                points = random_point_dropout(points, self.dropout_ratio)
                if points.shape[0] < 10: return None

        # --- Даунсэмплинг ---
        points = downsample_point_cloud(
            points, method=self.downsampling_method, max_points=self.max_points
        )
        if points.shape[0] == 0: return None

        # --- Фиксация количества точек ---
        current_n_points = points.shape[0]
        if self.fixed_points:
            if current_n_points < self.max_points:
                repeat_indices = np.random.choice(current_n_points, self.max_points - current_n_points, replace=True)
                points = np.vstack((points, points[repeat_indices]))
            elif current_n_points > self.max_points:  # Не должно быть после downsample
                select_indices = np.random.choice(current_n_points, self.max_points, replace=False)
                points = points[select_indices]

        # --- Нормализация ---
        points_norm = normalize_pc(points, self.pc_range)

        # Нормализуем GT боксы
        bboxes_norm = np.zeros_like(bboxes_gt_raw)  # [N_gt, 7]
        valid_gt_mask = np.ones(num_objects, dtype=bool)
        if num_objects > 0:
            for i in range(num_objects):
                norm_geom = normalize_bbox(bboxes_gt_raw[i, :6], self.pc_range)
                if np.any(norm_geom[3:6] <= 1e-6):  # Проверка вырожденности ПОСЛЕ нормализации
                    valid_gt_mask[i] = False
                    continue
                bboxes_norm[i, :6] = norm_geom
            bboxes_norm[:, 6] = bboxes_gt_raw[:, 6]  # Копируем class_id

            bboxes_norm = bboxes_norm[valid_gt_mask]
            num_objects = bboxes_norm.shape[0]

        # Финальная проверка размерности
        if points_norm.shape[1] != self.expected_point_dim:
            print(
                f"КРИТИЧЕСКАЯ ОШИБКА: Несоответствие размерности точек в {pcd_file}. Ожидалось {self.expected_point_dim}, получено {points_norm.shape[1]}. Пропуск.")
            return None

        return {
            'points': torch.from_numpy(points_norm).float(),
            'gt_bboxes': torch.from_numpy(bboxes_norm).float(),
            'file_path': pcd_file,
            'num_objects': num_objects
        }


def custom_collate_fn(batch: List[Optional[Dict]]) -> Dict[str, Union[torch.Tensor, List]]:
    """
    Пользовательская функция для объединения элементов в батч.
    Фильтрует None элементы, возвращенные из Dataset при ошибках.
    """
    # Фильтруем None и элементы без точек
    batch = [item for item in batch if
             item is not None and item.get('points') is not None and item['points'].numel() > 0]
    if not batch:
        return {}  # Возвращаем пустой словарь, если батч пуст

    elem = batch[0]
    keys = elem.keys()
    batch_size = len(batch)
    result = {}
    expected_point_dim = elem['points'].shape[1]  # Ожидаемая размерность из первого элемента

    # --- Обработка 'points' ---
    if 'points' in keys:
        points_list = [item.get('points') for item in batch]
        standardized_points = []
        target_point_shape = (cfg.MAX_POINTS, expected_point_dim)
        default_point_tensor = torch.zeros(target_point_shape, dtype=torch.float32)

        for points in points_list:
            if not isinstance(points, torch.Tensor) or points.numel() == 0:
                standardized_points.append(default_point_tensor.clone())
                continue
            if points.shape != target_point_shape:
                print(
                    f"COLLATE WARNING: Неожиданная форма тензора points: {points.shape}, ожидалось {target_point_shape}. Используется default.")
                standardized_points.append(default_point_tensor.clone())
                continue
            standardized_points.append(points)

        try:
            result['points'] = torch.stack(standardized_points, dim=0)
        except Exception as e:
            print(f"Ошибка стекинга 'points': {e}. Формы: {[p.shape for p in standardized_points]}.")
            # Если стекинг не удался, возвращаем пустой батч
            return {}

    # --- Обработка 'gt_bboxes' и 'num_objects' ---
    if 'gt_bboxes' in keys and 'num_objects' in keys:
        max_objects = 0
        num_objects_list_raw = [item.get('num_objects', 0) for item in batch]
        gt_bboxes_list_raw = [item.get('gt_bboxes') for item in batch]

        if num_objects_list_raw:
            max_objects = max(num_objects_list_raw) if num_objects_list_raw else 0

        padded_bboxes = []
        num_objects_list_final = []
        bbox_dim = 7  # cx,cy,cz,w,h,l,cls
        default_bbox_tensor = torch.zeros((max_objects, bbox_dim), dtype=torch.float32)

        for i, item in enumerate(batch):
            gt_bboxes_item = gt_bboxes_list_raw[i]
            num_obj_item = num_objects_list_raw[i]

            if not isinstance(gt_bboxes_item, torch.Tensor) or num_obj_item == 0:
                padded_bboxes.append(default_bbox_tensor.clone())
                num_objects_list_final.append(0)
                continue

            if gt_bboxes_item.shape[0] != num_obj_item:
                num_obj_item = gt_bboxes_item.shape[0]  # Корректируем, если не совпадает
            if gt_bboxes_item.shape[1] != bbox_dim:
                print(f"COLLATE WARNING: Неверная размерность bbox ({gt_bboxes_item.shape[1]} != {bbox_dim}). Пропуск.")
                padded_bboxes.append(default_bbox_tensor.clone())
                num_objects_list_final.append(0)
                continue

            # Паддинг
            if num_obj_item < max_objects:
                padding_shape = (max_objects - num_obj_item, bbox_dim)
                padding = torch.zeros(padding_shape, dtype=gt_bboxes_item.dtype, device=gt_bboxes_item.device)
                padded_tensor = torch.cat([gt_bboxes_item, padding], dim=0)
            else:  # num_obj_item == max_objects (или > max_objects - обработано в Dataset)
                padded_tensor = gt_bboxes_item[:max_objects, :]
                num_obj_item = padded_tensor.shape[0]

            padded_bboxes.append(padded_tensor)
            num_objects_list_final.append(num_obj_item)

        # Стекинг
        if padded_bboxes:
            try:
                result['gt_bboxes'] = torch.stack(padded_bboxes, dim=0)
                result['num_objects'] = torch.tensor(num_objects_list_final, dtype=torch.long)
            except Exception as e:
                print(f"Ошибка стекинга 'gt_bboxes': {e}. Формы: {[p.shape for p in padded_bboxes]}.")
                return {}  # Возвращаем пустой батч
        else:
            result['gt_bboxes'] = torch.empty((batch_size, 0, bbox_dim), dtype=torch.float32)
            result['num_objects'] = torch.zeros(batch_size, dtype=torch.long)

        # Проверка согласованности размеров
        if 'points' in result and result['points'].shape[0] != result['gt_bboxes'].shape[0]:
            print(
                f"COLLATE FATAL ERROR: Несоответствие размера батча! Points: {result['points'].shape[0]}, BBoxes: {result['gt_bboxes'].shape[0]}.")
            return {}

    # --- Обработка остальных полей ---
    for key in keys:
        if key in result: continue
        values = [item.get(key) for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:  # Попробуем стекинг, если формы совпадают
                if all(isinstance(v, torch.Tensor) and v.shape == values[0].shape for v in values):
                    result[key] = torch.stack(values, dim=0)
                else:
                    result[key] = values
            except Exception:
                result[key] = values
        else:
            result[key] = values  # Собираем в список

    return result
