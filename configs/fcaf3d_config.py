"""Конфигурация модели FCAF3D и процесса обучения/оценки."""

import math
import os

# --- Параметры Данных ---
POINT_CLOUD_RANGE = [-2.1622, -1.4537, -7.5483, 2.2364, 1.4061, 0.1701]
MAX_POINTS = 3072
# FPS лучше сохраняет структуру для малых данных
DOWNSAMPLING_METHOD = 'random'  # random

# --- Параметры входных фичей ---
USE_NORMALS_AS_FEATURES = True
USE_INTENSITY_AS_FEATURES = False
_feat_dim = 0
if USE_NORMALS_AS_FEATURES:
    _feat_dim += 3  # nx, ny, nz
if USE_INTENSITY_AS_FEATURES:
    _feat_dim += 1  # intensity
# Минимум 1 фича (можно просто 1.0, если ничего нет)
INPUT_FEAT_DIM = _feat_dim if _feat_dim > 0 else 1

# --- Параметры модели FCAF3D ---
NUM_FG_CLASSES = 2  # human, other
CLASS_NAMES = ['human', 'other']
# 0: background, 1: human, 2: other
NUM_PRED_CLASSES = NUM_FG_CLASSES + 1
NUM_LEVELS = 4
FP_FEATURE_DIM = 256  # Оставляем увеличенным
PREDICTION_HEAD_LEVELS = [0, 1, 2]  # Уровни FP для предсказательных голов

# --- Параметры Target Assignment ---
ASSIGNMENT_RADIUS_SCALE = 1.2
ASSIGNMENT_MIN_RADIUS = 0.1
ASSIGNMENT_TOPK = 9

# --- Параметры обучения ---
BATCH_SIZE = 4
NUM_EPOCHS = 300
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = int(NUM_EPOCHS * 0.1)  # 10% эпох на разогрев
WARMUP_FACTOR = 1.0 / 1000

# --- Веса функции потерь FCAF3D (с DIoU) ---
LOSS_WEIGHTS = {
    'cls': 1.0,
    'ctr': 1.0,
    'reg': 2.5
}
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0
DIOU_LOSS_EPS = 1e-7  # Для стабильности DIoU

# --- Параметры аугментации ---
# Небольшой поворот может помочь
AUG_ROTATION_RANGE = (-math.pi / 6, math.pi / 6)
AUG_SCALE_RANGE = (0.9, 1.1)  # Увеличим диапазон масштабирования
AUG_FLIP_PROB = 0.5
# Стандартное отклонение шума (в масштабе нормализованных координат)
AUG_NOISE_STD = 0.005
AUG_DROPOUT_RATIO = 0.1  # Доля выбрасываемых точек

# --- Параметры оценки ---
SCORE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.25
MAX_OBJECTS_PER_SCENE = 10
METRICS_IOU_THRESHOLD = 0.5

# --- Пути ---
# Определяем базовую директорию проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'DF', 'point_clouds')
OUTPUT_DIR_BASE = os.path.join(BASE_DIR, 'output')

# --- Флаги дебага ---
DEBUG_LOSS_CALCULATION = False
DEBUG_EVALUATION = True
DEBUG_VISUALIZATION = True

# --- Вычисляемые параметры ---
RUN_NAME = f'fcaf3d_diou_feats{INPUT_FEAT_DIM}'
OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, RUN_NAME)
CHECKPOINTS_DIR = os.path.join(OUTPUT_DIR, f'checkpoints')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, f'visualizations')
