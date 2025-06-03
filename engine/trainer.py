"""Логика обучения модели."""

import os
import traceback
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data.dataset import PointCloudDatasetFCAF3D, custom_collate_fn
from models.fcaf3d import FCAF3D
from losses.fcaf3d_loss import compute_fcaf3d_loss
from engine.evaluator import evaluate_fcaf3d_model
from utils.visualization import visualize_point_cloud_with_boxes, save_visualization
from data.preprocessing import denormalize_bbox, denormalize_pc_coords


def train_fcaf3d_model(config):
    """
    Основная функция для обучения модели FCAF3D.

    Args:
        config: Модуль конфигурации (configs.fcaf3d_config).
    """
    # Создание директорий для вывода
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)

    # Настройка устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # --- Загрузка данных ---
    print("\n--- Загрузка данных ---")
    try:
        dataset = PointCloudDatasetFCAF3D(
            data_dir=config.DATA_DIR,
            pc_range=config.POINT_CLOUD_RANGE,
            input_feat_dim=config.INPUT_FEAT_DIM,
            use_normals=config.USE_NORMALS_AS_FEATURES,
            use_intensity=config.USE_INTENSITY_AS_FEATURES,
            class_names=config.CLASS_NAMES,
            transform=True,  # Включаем аугментации для трейна
            test_mode=False,
            fixed_points=True,
            max_points=config.MAX_POINTS,
            downsampling_method=config.DOWNSAMPLING_METHOD,
            # Передаем параметры аугментации из конфига
            rotation_range=config.AUG_ROTATION_RANGE,
            scale_range=config.AUG_SCALE_RANGE,
            flip_prob=config.AUG_FLIP_PROB,
            noise_std=config.AUG_NOISE_STD,
            dropout_ratio=config.AUG_DROPOUT_RATIO
        )
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при создании Dataset: {e}")
        traceback.print_exc()
        return None
    if len(dataset) == 0:
        print("КРИТИЧЕСКАЯ ОШИБКА: Dataset пуст!")
        return None

    # Разделение на train/val
    val_split = 0.2
    if len(dataset) < 10:
        train_size = len(dataset)
        val_size = 0
        print("Предупреждение: Очень маленький датасет, используется для обучения.")
    else:
        val_size = max(1, int(val_split * len(dataset)))
        train_size = len(dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    else:
        train_dataset = dataset
        val_dataset = None

    # Создание DataLoader'ов
    num_workers = min(4, os.cpu_count() or 1)
    print(f"Используется {num_workers} воркеров для DataLoader")
    persistent_workers = (num_workers > 0 and device.type == 'cuda')

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'),
        persistent_workers=persistent_workers,
        drop_last=True, collate_fn=custom_collate_fn,
        prefetch_factor=2 if num_workers > 0 else None
    )
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == 'cuda'),
            persistent_workers=persistent_workers,
            drop_last=False, collate_fn=custom_collate_fn,
            prefetch_factor=2 if num_workers > 0 else None
        )

    print(f"Обучающих сэмплов: {train_size}, Валидационных сэмплов: {val_size}")
    if len(train_dataloader) == 0:
        print("КРИТИЧЕСКАЯ ОШИБКА: Train DataLoader пуст!")
        return None

    # --- Инициализация модели, оптимизатора, планировщика ---
    print("\n--- Инициализация модели ---")
    model = FCAF3D(
        input_channels=config.INPUT_FEAT_DIM,
        num_fg_classes=config.NUM_FG_CLASSES,
        num_levels=config.NUM_LEVELS,
        fp_feature_dim=config.FP_FEATURE_DIM,
        pred_head_levels=config.PREDICTION_HEAD_LEVELS
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # LR Scheduler с Warmup
    main_scheduler = CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS - config.WARMUP_EPOCHS, eta_min=config.LEARNING_RATE * 0.01
    )
    warmup_iters = len(train_dataloader) * config.WARMUP_EPOCHS
    if warmup_iters == 0: warmup_iters = 1  # Избегаем деления на ноль
    lr_lambda = lambda current_step: config.WARMUP_FACTOR + (1.0 - config.WARMUP_FACTOR) * float(current_step) / float(
        warmup_iters)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"Используется LR Warmup для {config.WARMUP_EPOCHS} эпох ({warmup_iters} итераций)")

    # --- Визуализация начального сэмпла ---
    if config.DEBUG_VISUALIZATION:
        print("\n--- Визуализация начального сэмпла ---")
        try:
            sample = dataset[0]
            if sample:
                points_norm = sample['points'].numpy()
                gt_bboxes_norm = sample['gt_bboxes'].numpy()
                points_denorm_xyz = denormalize_pc_coords(points_norm[:, :3], config.POINT_CLOUD_RANGE)
                points_vis = np.hstack((points_denorm_xyz, points_norm[:, 3:])) if points_norm.shape[
                                                                                       1] > 3 else points_denorm_xyz
                gt_bboxes_denorm = None
                gt_classes_vis = None
                if gt_bboxes_norm.shape[0] > 0:
                    gt_bboxes_denorm = denormalize_bbox(gt_bboxes_norm[:, :6], config.POINT_CLOUD_RANGE)
                    gt_classes_vis = gt_bboxes_norm[:, 6].astype(int)
                fig = visualize_point_cloud_with_boxes(
                    points_vis, gt_bboxes=gt_bboxes_denorm, gt_classes=gt_classes_vis,
                    title=f"Initial Sample - {os.path.basename(sample['file_path'])}"
                )
                save_visualization(fig, os.path.join(config.VISUALIZATIONS_DIR, 'initial_sample_visualization.html'))
            else:
                print("Предупреждение: Не удалось получить начальный сэмпл.")
        except Exception as e:
            print(f"Предупреждение: Не удалось визуализировать начальный сэмпл: {e}")

    # --- Основной цикл обучения ---
    print("\n--- Начало обучения ---")
    best_val_f1 = -1.0
    best_val_iou = -1.0
    global_step = 0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        epoch_loss_sum = 0.0
        epoch_loss_comp = {'cls': 0.0, 'ctr': 0.0, 'reg': 0.0}
        epoch_assigned = 0
        processed_batches = 0

        progress_bar = tqdm(train_dataloader, desc=f"Эпоха {epoch + 1}/{config.NUM_EPOCHS}", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            if not batch: continue  # Пропускаем пустые батчи

            try:
                points = batch.get('points').to(device)
                gt_bboxes = batch.get('gt_bboxes').to(device)
                num_objects = batch.get('num_objects').to(device)

                optimizer.zero_grad()
                end_points = model(points)
                loss_dict = compute_fcaf3d_loss(
                    end_points, gt_bboxes, num_objects, config.LOSS_WEIGHTS,
                    debug_loss=(epoch == 0 and batch_idx == 0 and config.DEBUG_LOSS_CALCULATION)
                )
                loss = loss_dict['total_loss']

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(
                        f"\nПредупреждение: Обнаружен NaN/Inf loss в батче {batch_idx}, эпоха {epoch + 1}. Пропуск батча.")
                    print(
                        f"  Компоненты: cls={loss_dict['cls_loss']:.4f}, ctr={loss_dict['ctr_loss']:.4f}, reg={loss_dict['reg_loss']:.4f}")
                    optimizer.zero_grad()  # Очищаем градиенты перед пропуском
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Клиппинг градиентов
                optimizer.step()

                # Обновление LR
                if global_step < warmup_iters:
                    warmup_scheduler.step()
                global_step += 1

                # Сбор статистики
                bs = points.shape[0]
                epoch_loss_sum += loss.item() * bs
                epoch_loss_comp['cls'] += loss_dict['cls_loss'].item() * bs
                epoch_loss_comp['ctr'] += loss_dict['ctr_loss'].item() * bs
                epoch_loss_comp['reg'] += loss_dict['reg_loss'].item() * bs
                epoch_assigned += loss_dict['num_assigned'].item()
                processed_batches += 1

                # Обновление progress bar
                avg_loss_so_far = epoch_loss_sum / max(1, processed_batches * config.BATCH_SIZE)
                current_lr_print = optimizer.param_groups[0]['lr']
                postfix = {'loss': f"{avg_loss_so_far:.4f}", 'lr': f"{current_lr_print:.1e}",
                           'assign': f"{loss_dict['num_assigned']:.0f}"}
                progress_bar.set_postfix(postfix)

            except Exception as e:
                print(f"\nКРИТИЧЕСКАЯ ОШИБКА в батче {batch_idx}, эпоха {epoch + 1}: {e}")
                traceback.print_exc()
                # Попробуем продолжить со следующего батча
                # Очистка памяти может помочь
                del end_points, loss, loss_dict, points, gt_bboxes, num_objects
                torch.cuda.empty_cache()

        # Обновление основного LR шедулера после Warmup
        if epoch >= config.WARMUP_EPOCHS:
            main_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # --- Логирование и Валидация в конце эпохи ---
        if processed_batches == 0:
            print(f"Предупреждение: Ни один батч не обработан в эпохе {epoch + 1}.")
            continue

        num_train_samples_processed = processed_batches * config.BATCH_SIZE
        avg_total_loss = epoch_loss_sum / num_train_samples_processed
        avg_cls = epoch_loss_comp['cls'] / num_train_samples_processed
        avg_ctr = epoch_loss_comp['ctr'] / num_train_samples_processed
        avg_reg = epoch_loss_comp['reg'] / num_train_samples_processed
        avg_assign = epoch_assigned / processed_batches

        print(f"\n--- Итоги Эпохи {epoch + 1}/{config.NUM_EPOCHS} --- LR: {current_lr:.6f}")
        print(
            f"Train Loss: {avg_total_loss:.4f} (cls:{avg_cls:.4f}, ctr:{avg_ctr:.4f}, reg:{avg_reg:.4f}), Avg Assign: {avg_assign:.1f}")

        # Валидация каждые N эпох или в конце
        val_interval = 10
        if val_dataloader is not None and ((epoch + 1) % val_interval == 0 or epoch == config.NUM_EPOCHS - 1):
            val_metrics = evaluate_fcaf3d_model(
                model, val_dataloader, device, config.LOSS_WEIGHTS, config.POINT_CLOUD_RANGE,
                config.NMS_IOU_THRESHOLD, config.MAX_OBJECTS_PER_SCENE, config.SCORE_THRESHOLD,
                config.METRICS_IOU_THRESHOLD,
                debug_eval=(epoch == config.NUM_EPOCHS - 1 and config.DEBUG_EVALUATION)  # Дебаг только в конце
            )

            # Сохранение лучшей модели
            current_f1 = val_metrics.get('f1_score', -1.0)
            current_iou = val_metrics.get('mean_iou', -1.0)
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                save_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model_f1.pth')
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_f1': best_val_f1
                }, save_path)
                print(f"** Сохранена лучшая модель по F1 (F1_Det: {best_val_f1:.4f}) -> {save_path} **")
            if current_iou > best_val_iou:
                best_val_iou = current_iou
                save_path = os.path.join(config.CHECKPOINTS_DIR, 'best_model_iou.pth')
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'best_iou': best_val_iou
                }, save_path)
                print(f"** Сохранена лучшая модель по IoU (mIoU_TP: {best_val_iou:.4f}) -> {save_path} **")
        else:
            if (epoch + 1) % val_interval == 0:  # Печатаем, если должна была быть валидация
                print("Валидация пропущена (нет val_dataloader).")

        # Сохранение чекпоинта каждые N эпох
        save_interval = 20
        if (epoch + 1) % save_interval == 0:
            ckpt_path = os.path.join(config.CHECKPOINTS_DIR, f'model_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, ckpt_path)
            print(f"Чекпоинт сохранен: {ckpt_path}")

    # Сохранение финальной модели
    final_path = os.path.join(config.CHECKPOINTS_DIR, 'final_model.pth')
    torch.save({
        'epoch': config.NUM_EPOCHS - 1, 'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_f1': best_val_f1, 'best_iou': best_val_iou
    }, final_path)
    print(f"\n--- Обучение завершено ({config.RUN_NAME})! ---")
    print(f"Финальная модель сохранена: {final_path}")
    print(f"Лучший F1_Det: {best_val_f1:.4f}, Лучший mIoU(TP): {best_val_iou:.4f}")

    return model
