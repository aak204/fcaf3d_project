"""Основная архитектура модели FCAF3D."""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Tuple, Union
from configs import fcaf3d_config as cfg
from .pointnet_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class FCAF3DHead(nn.Module):
    """Предсказательная голова FCAF3D для одного уровня."""

    def __init__(self, in_channels: int, num_pred_classes: int):
        super().__init__()
        self.in_channels = in_channels
        self.num_pred_classes = num_pred_classes

        self.conv_cls = nn.Conv1d(in_channels, num_pred_classes, 1)
        self.conv_ctr = nn.Conv1d(in_channels, 1, 1)
        self.conv_center_offset = nn.Conv1d(in_channels, 3, 1)
        self.conv_size_log = nn.Conv1d(in_channels, 3, 1)

        self._initialize_biases()

    def _initialize_biases(self):
        """Инициализация смещений для лучшей сходимости."""
        # Инициализация смещения для классификации (фокус на фоне)
        if hasattr(self.conv_cls, 'bias') and self.conv_cls.bias is not None:
            try:
                # Prior probability of foreground (~0.01)
                bias_value = -math.log((1 - 0.01) / 0.01)
                nn.init.constant_(self.conv_cls.bias, bias_value)
            except Exception as e:
                print(f"Warning: Could not initialize cls bias: {e}")

        # Инициализация смещения для регрессии центра (в 0)
        if hasattr(self.conv_center_offset, 'bias') and self.conv_center_offset.bias is not None:
            try:
                nn.init.constant_(self.conv_center_offset.bias, 0.0)
            except Exception as e:
                print(f"Warning: Could not initialize center offset bias: {e}")

        # Инициализация смещения для регрессии размера (маленькие размеры)
        if hasattr(self.conv_size_log, 'bias') and self.conv_size_log.bias is not None:
            try:
                nn.init.constant_(self.conv_size_log.bias, -2.3)
            except Exception as e:
                print(f"Warning: Could not initialize size log bias: {e}")

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): Фичи уровня [B, C, N].

        Returns:
            Tuple[torch.Tensor, ...]: Кортеж с предсказаниями:
                - cls_preds: Логиты классов [B, num_pred_classes, N].
                - ctr_preds: Логиты центрированности [B, 1, N].
                - center_offset_preds: Предсказания смещения центра [B, 3, N].
                - size_log_preds: Предсказания логарифма размера [B, 3, N].
        """
        cls_preds = self.conv_cls(features)
        ctr_preds = self.conv_ctr(features)
        center_offset_preds = self.conv_center_offset(features)
        size_log_preds = self.conv_size_log(features)
        return cls_preds, ctr_preds, center_offset_preds, size_log_preds


class FCAF3D(nn.Module):
    """Архитектура FCAF3D."""

    def __init__(
            self,
            input_channels: int = cfg.INPUT_FEAT_DIM,
            num_fg_classes: int = cfg.NUM_FG_CLASSES,
            num_levels: int = cfg.NUM_LEVELS,
            fp_feature_dim: int = cfg.FP_FEATURE_DIM,
            pred_head_levels: List[int] = cfg.PREDICTION_HEAD_LEVELS
    ):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.num_pred_classes = num_fg_classes + 1  # Включая фон
        self.num_levels = num_levels
        self.pred_head_levels = pred_head_levels
        self.input_channels = input_channels  # Сохраняем для информации

        self.sa_layers = nn.ModuleList()
        self.fp_layers = nn.ModuleList()

        self._build_backbone(input_channels, num_levels, fp_feature_dim)
        self._build_prediction_heads(fp_feature_dim, self.num_pred_classes, pred_head_levels)

    def _build_backbone(self, input_channels: int, num_levels: int, fp_feature_dim: int):
        """Строит энкодер (SA) и декодер (FP) PointNet++."""
        sa_feature_channels = [input_channels]  # Фичи на исходном уровне
        # Параметры SA слоев (можно вынести в конфиг)
        npoints = [cfg.MAX_POINTS // (2 ** i) for i in range(1, num_levels + 1)]
        # npoints = [cfg.MAX_POINTS // 2, cfg.MAX_POINTS // 4, cfg.MAX_POINTS // 8, cfg.MAX_POINTS // 16]
        radii = [0.1, 0.2, 0.4, 0.8]
        nsamples = [32, 32, 32, 32]
        mlp_specs_sa = [[64, 64, 128], [128, 128, 256], [256, 256, 512], [512, 512, 1024]]

        last_sa_feat_ch = input_channels
        print(f"Building FCAF3D Backbone (Input Features: {input_channels})")
        # SA Layers
        for i in range(num_levels):
            current_in_channel_sa = 3 + last_sa_feat_ch  # 3 (XYZ) + фичи пред. уровня
            print(f"  SA Layer {i + 1}: npoint={npoints[i]}, radius={radii[i]}, nsample={nsamples[i]}, "
                  f"in_ch={current_in_channel_sa}, mlp_out={mlp_specs_sa[i][-1]}")
            self.sa_layers.append(
                PointNetSetAbstraction(
                    npoint=npoints[i], radius=radii[i], nsample=nsamples[i],
                    in_channel=current_in_channel_sa, mlp=mlp_specs_sa[i], group_all=False
                )
            )
            last_sa_feat_ch = mlp_specs_sa[i][-1]
            sa_feature_channels.append(last_sa_feat_ch)

        # FP Layers
        last_fp_feat_ch = sa_feature_channels[-1]  # Фичи с самого грубого SA
        self.fp_out_channels = []  # Сохраним выходные каналы FP для голов
        # Параметры FP слоев (можно вынести в конфиг)
        fp_mlp_specs = [
            [512, 256],  # SA4(1024)+SA3(512)=1536 -> 512 -> 256
            [256, fp_feature_dim],  # FP0(256)+SA2(256)=512 -> 256 -> 256
            [fp_feature_dim, fp_feature_dim],  # FP1(256)+SA1(128)=384 -> 256 -> 256
            [fp_feature_dim, fp_feature_dim]  # FP2(256)+SA0(input_channels) -> 256 -> 256
        ]
        # Корректируем входной канал последнего FP MLP spec
        fp_mlp_specs[-1][0] = fp_feature_dim + sa_feature_channels[0]

        print("\nBuilding FP Layers:")
        for i in range(num_levels):
            fp_level_idx = num_levels - 1 - i  # Индекс SA уровня для skip (3, 2, 1, 0)
            skip_conn_ch = sa_feature_channels[fp_level_idx]
            in_channel_fp = last_fp_feat_ch + skip_conn_ch
            mlp_fp = fp_mlp_specs[i]
            print(f"  FP Layer {i} (-> SA{fp_level_idx}): in_ch={in_channel_fp} "
                  f"(prev_fp={last_fp_feat_ch} + skip_sa={skip_conn_ch}), mlp_out={mlp_fp[-1]}")
            self.fp_layers.append(PointNetFeaturePropagation(in_channel_fp, mlp_fp))
            last_fp_feat_ch = mlp_fp[-1]
            self.fp_out_channels.append(last_fp_feat_ch)
        # Переворачиваем, чтобы индексы соответствовали уровням (0 - самый детальный)
        self.fp_out_channels.reverse()  # [ch_fp0, ch_fp1, ch_fp2, ch_fp3]

    def _build_prediction_heads(
            self, fp_feature_dim: int, num_pred_classes: int, pred_head_levels: List[int]
    ):
        """Строит предсказательные головы для указанных уровней FP."""
        self.prediction_heads = nn.ModuleList()
        print("\nBuilding Prediction Heads:")
        for level_idx in pred_head_levels:
            if level_idx < 0 or level_idx >= len(self.fp_out_channels):
                raise ValueError(
                    f"Неверный индекс уровня для головы: {level_idx}. Доступно: 0-{len(self.fp_out_channels) - 1}")
            head_input_channels = self.fp_out_channels[level_idx]
            print(f"  Head for Level {level_idx}: Input Channels = {head_input_channels}")
            self.prediction_heads.append(FCAF3DHead(head_input_channels, num_pred_classes))

    def forward(self, points: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Прямой проход модели.

        Args:
            points (torch.Tensor): Входные точки [B, N, 3 + input_feat_dim].

        Returns:
            Dict: Словарь с выходами модели ('fp_xyz', 'fp_features', 'cls_preds', ...).
        """
        B, N, D = points.shape
        if D != 3 + self.input_channels:
            raise ValueError(f"Input points dim mismatch. Expected {3 + self.input_channels}, got {D}")

        xyz = points[..., :3].contiguous()
        # Фичи: [B, N, input_feat_dim] -> permute -> [B, input_feat_dim, N]
        features = points[..., 3:].permute(0, 2, 1).contiguous() if self.input_channels > 0 else None

        # --- Энкодер (SA Layers) ---
        sa_xyz = [xyz]
        sa_features = [features]  # Уровень 0 (исходные фичи)
        for i, sa_layer in enumerate(self.sa_layers):
            # points для SA слоя - это фичи с предыдущего уровня [B, C_prev, N_prev]
            # Нужно передать как [B, N_prev, C_prev]
            sa_input_features = sa_features[-1].permute(0, 2, 1).contiguous() if sa_features[-1] is not None else None
            cur_xyz, cur_features = sa_layer(sa_xyz[-1], sa_input_features)
            sa_xyz.append(cur_xyz)
            sa_features.append(cur_features)  # Фичи уже в формате [B, C_new, N_new]

        # --- Декодер (FP Layers) ---
        # Начинаем с самого грубого уровня
        fp_xyz = [sa_xyz[-1]]
        fp_features = [sa_features[-1]]  # Фичи с последнего SA слоя [B, C_sa_last, N_sa_last]
        for i in range(self.num_levels):
            # Индекс SA уровня для skip connection (e.g., 3, 2, 1, 0)
            fp_target_idx = self.num_levels - 1 - i
            # points1: фичи с детального SA уровня [B, C_sa_skip, N_detailed]
            points1 = sa_features[fp_target_idx]
            # points2: фичи с предыдущего (грубого) FP уровня [B, C_fp_prev, N_coarse]
            cur_features = self.fp_layers[i](
                xyz1=sa_xyz[fp_target_idx],  # Координаты детального уровня
                xyz2=fp_xyz[-1],  # Координаты грубого уровня
                points1=points1,
                points2=fp_features[-1]
            )  # Выход: [B, C_fp_new, N_detailed]
            fp_xyz.append(sa_xyz[fp_target_idx])
            fp_features.append(cur_features)

        # Сохраняем результаты FP (пропуская самый грубый входной)
        # fp_xyz: [xyz_sa4, xyz_sa3, xyz_sa2, xyz_sa1, xyz_sa0]
        # fp_features: [feat_sa4, feat_fp3, feat_fp2, feat_fp1, feat_fp0]
        end_points = {
            'fp_xyz': fp_xyz[1:],  # [xyz_sa3, xyz_sa2, xyz_sa1, xyz_sa0]
            'fp_features': fp_features[1:]  # [feat_fp3, feat_fp2, feat_fp1, feat_fp0]
        }

        # --- Предсказательные Головы ---
        all_cls_preds, all_ctr_preds, all_offset_preds, all_logsize_preds = [], [], [], []
        # Фичи FP идут от грубого к детальному: [feat_fp3, feat_fp2, feat_fp1, feat_fp0]
        # Индексы соответствуют: 0 -> fp3, 1 -> fp2, 2 -> fp1, 3 -> fp0

        for i, level_idx in enumerate(self.pred_head_levels):  # 0, 1, 2
            # Получаем фичи нужного уровня (индекс в перевернутом списке)
            fp_level_features = end_points['fp_features'][::-1][level_idx]  # feat_fp0, feat_fp1, feat_fp2
            # Применяем соответствующую голову
            cls_preds, ctr_preds, offset_preds, logsize_preds = self.prediction_heads[i](fp_level_features)
            all_cls_preds.append(cls_preds)
            all_ctr_preds.append(ctr_preds)
            all_offset_preds.append(offset_preds)
            all_logsize_preds.append(logsize_preds)

        end_points['cls_preds'] = all_cls_preds
        end_points['ctr_preds'] = all_ctr_preds
        end_points['center_offset_preds'] = all_offset_preds
        end_points['size_log_preds'] = all_logsize_preds

        return end_points
