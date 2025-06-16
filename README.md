# 🚆 Система безопасности посадки/высадки пассажиров ж/д транспорта

## 📜 Описание проекта

Данный проект представляет собой выпускную квалификационную работу (ВКР) на тему: **"Разработка системы оповещения, обеспечивающей безопасность посадки и высадки пассажиров железнодорожного транспорта"**.

Система предназначена для автоматического контроля дверных порталов вагонов с целью предотвращения зажатия пассажиров или их вещей автоматическими дверями. Она использует данные с RGB-камеры для определения состояния дверей и данные со стереокамеры (3D облака точек) для детекции людей и посторонних объектов в опасной зоне.

---

## 🎯 Цели и задачи

**Основная цель:** повышение уровня безопасности пассажиров железнодорожного транспорта за счет автоматизации контроля дверных проемов.

**Ключевые задачи:**
*   Разработка алгоритма определения состояния дверей (`OPEN`, `CLOSED`, `SEMI`, `UNKNOWN`) по RGB-изображениям с использованием нейронной сети ResNet18.
*   Разработка и сравнение двух подходов к детекции объектов (`HUMAN`, `OTHER`) в 3D облаках точек в зоне дверного портала:
    *   Нейросетевой подход на базе архитектуры FCAF3D.
    *   Аналитический подход на основе анализа плотности точек в заданной области.
*   Интеграция информации о состоянии дверей и наличии объектов для формирования итогового сигнала об опасности.
*   Реализация прототипа системы с пользовательским интерфейсом на Gradio для демонстрации работы.

---

## 🖼️ Концептуальная схема системы

![Концептуальная схема системы](https://www.plantuml.com/plantuml/png/fLPjJzjM5FxkNt6NILiQ5MXvwL25ea9UMgG15NJV9aLER8ABOqVspL0tJIgmqcqWfQqs3NKhNV-12Gsul2H_uVeVxPmJ9pSumGKjaGNttdEVyvpdl3WFNAuwl5OnbI_ucbxHsGz6qL4jhYXViv6H4IR-4fuu4Yrn1ay3dbCMR0OlH5ES1xjeeN23cpfBfSHREBq8Tk2e7Mm5st3lBDWJ5y8BwhqxmeD_XG_BhH02BhK9kXdy1biNS5XFfNhktmaFu9uB3opEEkUYEK31hZ9kiwBDkLrXck7e9MxOLW_a0270_YCy4eH8-BJnYS7P6VBS8IPROR0GE1EdeWMdFY3WfjUXI7xTTSEggevQOUXHjSeWn4yAWu-xfchsXc6LsQfgkZhjhN3TqMt-hAhdROjdRDDssCQQmOUThXW_wkoEydEVYwfQMbVBLpdejrGTnzu8xJfklfoY3-qyMJDAwvRkkkpkw1JS0v741qm7ITGYJTfTFKG3XCRmBuBTu0NkGlpX8YbEKC_WDQx5frl3v9YPcRUgDPvEXtJc88YoOzSiBIIMkvUVoDthdDcEfZjTmfcvwQaCxKkrYYPS_opbSoGSR8U9XSKXrSyX6PuAdoMn136DahGK62Wls5hDr6CvZ-Td1pbFZa_FPMQ7Snukpagj-3bdUVh4EBkvG6qgvNCuEGwsiTQ3bqFp2p81G4nkiofNOvXFuNUG-TmT_Fw_rC823O9VkoE3GLxfoWaTxy553n-M193d5DKnd3PYebdMtPh9O-B9vQTotmp6axypcvtCNH5FE8EWrvXuJKFqb8atLpSLZfLkfoVw3KLJ6i8YmQAj9KTGOgQOdlbgrJ3r4QC2G6cs_72EYNrnqck6uzuKJ3vUV7XhPgoOJZ3LPKQbN30GLi8F1I-PTarBiuai3jT3YCl3CSd2OcgAKy3DP2LkjQJ5u4P8bsmDE8Vlar-0vL6mZQo3dKwcsvXIJ2xwPSdbfNdfwAbhMmEUP1h6PHg4pyG_bqV5At51e3uevLrEXJGDzXWOzHiAVr3Pe4vqIF-2QgdcCzTm2nLgTYP7p63ke0eNTNvt6g_3lclfdFKF2epcchHQlSnuVPdfvCCsB4v9l14vgH0s9k85cwWaYdSUlZS_aWD2vKP9DGiLdQ_PMYI8270-CvkVoDBDrPAwEg4Dn1mcSE9o0jzVRvHXAll3-026csHU3zi0p1iqT7mYeTcsKt2vodNM7NEN0Dw5AhoaovJytr5Sh4e2i8jFmRnGXGpgLkaAbBUOdc0JlhkNCLgYqOzICbNNrLq9qbLZKghHqSA7WEkOcw43KlmR7P0HPIBrr8v5x0YbbmpNi2rPFPqn6ebqdu8wWeScCFq8X2KLKoVhXWMbuXfb-2D1PAIGiWOawmIZe7GZIoOEmrBmODD7Gx3lpLVyTPZkRt8_qYAA77SZO3cqMHKONT8cfd7GSoebze7fUn3wbNWZ_XP_GgKSt4-bhfvcCBqO_Y36gOQgKtwWaw0ewbGKyFWKdUomJhYAd7L8RR3b8WPXPm7FE4r0ik5jDVWEKcWy4NP_mznd2SPYM-nxYx5avjkbbLpsDbjwd5kypLPo2_EtsUZewAs4CX08UOXs5Bc9RKHo8_uGtXYOWNG0JH98FDbR8To1l1_iqlr7totCBeujFNcKMuRxeDqrnK4TRz9DYbGGwKKRIuc47sSJXZJK10Au2IkC8hPvbOW3SjhVp2HiF0Rf3DAtSHZ2LekDuLXhubjLRgBw3tHBgrNC_m00)

*Рисунок 1 – Общая архитектура системы оповещения.*

---

## 🛠️ Используемые технологии

*   **Язык программирования:** Python 3.8+
*   **Глубокое обучение:**
    *   PyTorch (для ResNet18 и FCAF3D)
    *   ResNet18 (для классификации состояния дверей)
    *   FCAF3D (для 3D детекции объектов)
*   **Обработка 3D-данных:**
    *   Open3D (для работы с облаками точек .pcd)
    *   NumPy
*   **Обработка изображений:**
    *   Pillow (PIL)
    *   OpenCV (опционально, для дополнительных манипуляций)
*   **Веб-интерфейс:**
    *   Gradio
*   **Визуализация:**
    *   Matplotlib
    *   Plotly
*   **Ускорение вычислений:**
    *   Numba (для FPS в одной из версий)

---

## 📂 Структура проекта

Проект организован следующим образом (основные компоненты):

```
.
├── data/                     # Папка для хранения датасетов (PNG, PCD, JSON аннотации)
│   ├── images/
│   └── point_clouds/
├── models_pytorch/           # Сохраненные веса моделей
│   ├── resnet18_door_state.pth
│   └── fcaf3d_final_model.pth
├── notebooks/                # Jupyter Notebooks для экспериментов и анализа данных
├── src/                      # Исходный код
│   ├── analytical_solution/  # Код аналитического подхода
│   │   └── door_cloud_analyzer.py
│   ├── fcaf3d_training/      # Код для обучения FCAF3D
│   │   ├── train_fcaf3d_model.py
│   │   ├── fcaf3d_model.py   # Определение архитектуры FCAF3D
│   │   └── dataset.py        # Класс Dataset для FCAF3D
│   ├── resnet_training/      # Код для обучения ResNet18 (если отдельный)
│   │   └── train_resnet_door.py
│   ├── inference/            # Скрипты для инференса и оценки
│   │   └── evaluate_fcaf3d.py
│   ├── utils/                # Вспомогательные функции
│   └── gradio_interface.py   # Код для веб-интерфейса Gradio (интегрированная система)
├── output/                   # Результаты работы, визуализации, логи
│   ├── inference_results/
│   └── training_runs/
├── README.md                 # Этот файл
└── requirements.txt          # Список зависимостей Python
```

**Ключевые файлы, доступные в репозитории:**
*   `src/resnet_training/train_resnet_door.py` (или аналогичный): Скрипт для обучения модели ResNet18 для определения состояния дверей.
*   `src/gradio_interface.py`: Файл, содержащий код для запуска веб-интерфейса Gradio, который интегрирует работу модулей ResNet18 и FCAF3D/аналитического метода для демонстрации системы.

---

## 🚀 Установка и запуск

### 1. Клонирование репозитория
```bash
git clone [ССЫЛКА_НА_ВАШ_РЕПОЗИТОРИЙ]
cd [НАЗВАНИЕ_ПАПКИ_ПРОЕКТА]
```

### 2. Создание виртуального окружения и установка зависимостей
Рекомендуется использовать виртуальное окружение:
```bash
python -m venv .venv
# Активация:
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```
*Примечание: Убедитесь, что у вас установлены PyTorch с поддержкой CUDA, если вы планируете использовать GPU.*

### 3. Подготовка данных
*   Разместите датасеты (RGB-изображения и .pcd файлы с аннотациями) в папке `data/` согласно структуре, ожидаемой скриптами.
*   Убедитесь, что пути к данным и моделям в конфигурационных файлах или скриптах указаны корректно.

### 4. Запуск обучения (пример для FCAF3D)
```bash
python src/fcaf3d_training/train_fcaf3d_model.py
```
*Параметры обучения (пути к данным, выходная директория, гиперпараметры) настраиваются внутри скрипта или через аргументы командной строки (если реализовано).*

### 5. Запуск Gradio-интерфейса
Для запуска демонстрационного интерфейса:
```bash
python src/gradio_interface.py
```
После запуска интерфейс будет доступен в веб-браузере по указанному адресу (обычно `http://127.0.0.1:7860`).

---

## 📊 Примеры работы и результаты

### Определение состояния двери (ResNet18)
Модель ResNet18 классифицирует загруженное RGB-изображение на одно из состояний: `OPEN`, `CLOSED`, `SEMI`, `UNKNOWN`.
*   **AUC ROC:** *[Вставьте ваше значение, например, 0.95]*

### Детекция объектов в облаке точек (FCAF3D)
Модель FCAF3D детектирует объекты классов `HUMAN` и `OTHER` в 3D облаке точек.
*   **Средний Avg Matched IoU:** 0.5841
*   **Средний Macro F1 (Presence):** 0.9775

*(Можно вставить скриншот из Gradio интерфейса или пример визуализации)*
![Пример работы интерфейса Gradio]([ССЫЛКА_НА_СКРИНШОТ_GRADIO_ИНТЕРФЕЙСА])
*Рисунок 2 – Демонстрация работы системы через интерфейс Gradio.*

---

## 🧑‍💻 Авторы и руководитель

*   **Студент:** [Ваше ФИО]
*   **Группа:** [Ваша группа]
*   **Научный руководитель:** [ФИО научного руководителя]
*   **Учебное заведение:** [Название вашего учебного заведения]
*   **Год:** 2024

---
```

**Что нужно будет сделать вам:**

1.  **Заменить плейсхолдеры:**
    *   `[ССЫЛКА_НА_ИЗОБРАЖЕНИЕ_СХЕМЫ_ИЗ_PLANTUML_ИЛИ_ДРУГОГО_ИСТОЧНИКА]`: Сгенерируйте PNG или SVG из вашей PlantUML диаграммы (например, `Рисунок 2.X – Блок-схема интегрированной логики...` или `Рисунок 1.4 – Концептуальная схема системы оповещения`), загрузите его в репозиторий (например, в папку `docs/images/`) и укажите относительный путь.
    *   `[ССЫЛКА_НА_ВАШ_РЕПОЗИТОРИЙ]`
    *   `[НАЗВАНИЕ_ПАПКИ_ПРОЕКТА]`
    *   `[Вставьте ваше значение, например, 0.95]` для AUC ROC.
    *   `[ССЫЛКА_НА_СКРИНШОТ_GRADIO_ИНТЕРФЕЙСА]`: Сделайте скриншот работающего интерфейса Gradio, добавьте его в репозиторий и укажите путь.
    *   Заполните информацию об авторах.
2.  **Проверить структуру проекта:** Убедитесь, что описанная структура (`📂 Структура проекта`) соответствует реальной структуре вашего репозитория. Скорректируйте при необходимости.
3.  **Проверить команды запуска:** Убедитесь, что команды для установки и запуска (`🚀 Установка и запуск`) корректны для вашего проекта.
4.  **Файл `requirements.txt`:** Не забудьте создать этот файл в корне проекта командой `pip freeze > requirements.txt` (после установки всех зависимостей в вашем виртуальном окружении).
5.  **Дополнительные изображения/GIF:** Если у вас есть другие наглядные материалы (например, GIF с демонстрацией работы, примеры удачных и неудачных детекций), их тоже можно вставить.

Этот README должен дать хорошее первое представление о вашем проекте. Можете его дорабатывать и улучшать по своему усмотрению.
