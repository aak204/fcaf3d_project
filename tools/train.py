"""Скрипт для запуска обучения модели FCAF3D."""

import os
import torch
from configs import fcaf3d_config as cfg  # Загружаем конфигурацию
from engine.trainer import train_fcaf3d_model


def main():
    print("--- Запуск обучения FCAF3D ---")
    print(f"Конфигурация загружена из: configs/fcaf3d_config.py")
    print(f"Результаты будут сохранены в: {cfg.OUTPUT_DIR}")

    # Проверка и создание директорий (на всякий случай, хотя trainer тоже создает)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(cfg.VISUALIZATIONS_DIR, exist_ok=True)

    # Установка seed для воспроизводимости (опционально)
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Запуск обучения
    trained_model = train_fcaf3d_model(cfg)

    if trained_model:
        print("\n--- Обучение успешно завершено ---")
    else:
        print("\n--- Обучение завершилось с ошибкой ---")


if __name__ == "__main__":
    main()
