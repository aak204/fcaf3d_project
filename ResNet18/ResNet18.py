import os
import glob
import re
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import cycle

# Пути к данным
IMG_DIR_1 = "/content/data/point_cloud_gt/img_gt"
LABEL_FILE_1 = "/content/data/point_cloud_gt/door_state.txt"
IMG_DIR_2 = "/content/data/point _end/img"
LABEL_FILE_2 = "/content/data/point _end/door_state.txt"

# Фиксированный bounding box, полученный из статистики
FIXED_BBOX = (665, 7, 765, 244)
MARGIN = 0.02  # 2% расширение

def crop_with_fixed_bbox(image, bbox=FIXED_BBOX, margin=MARGIN):
    """
    Обрезает изображение по фиксированному bounding box с запасом.
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    delta_w = int(width * margin)
    delta_h = int(height * margin)
    new_bbox = (
        max(0, x_min - delta_w),
        max(0, y_min - delta_h),
        x_max + delta_w,
        y_max + delta_h
    )
    return image.crop(new_bbox)

def read_labels(label_file):
    """
    Читает файл с метками и возвращает словарь: {image_id: door_state}.
    """
    labels_dict = {}
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    image_id = parts[0]
                    state = parts[1].upper()
                    labels_dict[image_id] = state
    return labels_dict

def extract_image_id(filename):
    """
    Извлекает идентификатор изображения из имени файла.
    """
    match = re.search(r'raw_(\d+)\.png$', filename)
    if match:
        return match.group(1)
    return None

def get_data_from_folder(img_dir, label_file):
    """
    Возвращает список кортежей (img_path, label) для изображений.
    """
    labels_dict = read_labels(label_file)
    img_paths = glob.glob(os.path.join(img_dir, "*.png"))
    data = []
    for path in img_paths:
        img_id = extract_image_id(os.path.basename(path))
        if img_id is not None and img_id in labels_dict:
            label = labels_dict[img_id]
            data.append((path, label))
    return data

class DoorStateDataset(Dataset):
    def __init__(self, data_list, transform=None, crop_bbox=True):
        self.data = data_list
        self.transform = transform
        self.crop_bbox = crop_bbox
        self.class_to_idx = {"UNKNOWN": 0, "SEMI": 1, "CLOSED": 2, "OPEN": 3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.crop_bbox:
            image = crop_with_fixed_bbox(image)
        if self.transform:
            image = self.transform(image)
        label_idx = self.class_to_idx.get(label, 0)
        return image, label_idx, img_path

# Получение данных
data_list_1 = get_data_from_folder(IMG_DIR_1, LABEL_FILE_1)
data_list_2 = get_data_from_folder(IMG_DIR_2, LABEL_FILE_2)
combined_data_list = data_list_1 + data_list_2
print(f"Всего изображений: {len(combined_data_list)}")

# Аугментации
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Создание датасета
dataset = DoorStateDataset(combined_data_list, transform=data_transform)

# Параметры обучения
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Разделение на обучающую и валидационную выборки
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Распределение классов
from collections import Counter
all_labels = [label for _, label in combined_data_list]
class_counts = Counter(all_labels)
print("\nРаспределение по классам:")
class_order = ["UNKNOWN", "SEMI", "CLOSED", "OPEN"]
total = len(combined_data_list)
for class_name in class_order:
    count = class_counts.get(class_name, 0)
    percentage = (count / total) * 100 if total > 0 else 0
    print(f"{class_name}: {count} изображений ({percentage:.2f}%)")

# Модель
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

def evaluate_and_get_roc_data(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_labels_list, all_probs_list = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            all_labels_list.append(labels.cpu().numpy())
            all_probs_list.append(probs.cpu().numpy())
    all_labels_np = np.concatenate(all_labels_list)
    all_probs_np = np.concatenate(all_probs_list)
    roc_auc_ovr_macro = 0.0
    try:
        if len(np.unique(all_labels_np)) < 2:
            print("Предупреждение: менее двух классов в валидации.")
        else:
            roc_auc_ovr_macro = roc_auc_score(all_labels_np, all_probs_np, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"Ошибка при вычислении ROC AUC: {e}.")
    return running_loss / len(loader.dataset), roc_auc_ovr_macro, all_labels_np, all_probs_np

def plot_roc_curve(y_true, y_probs, num_classes, class_names):
    fpr, tpr, roc_auc = {}, {}, {}
    y_true_binarized = np.eye(num_classes)[y_true]
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
    for i, color in zip(range(num_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')
    plt.plot(fpr["macro"], tpr["macro"], label=f'Macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривые (Multi-class)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Обучение
CLASS_NAMES = ["UNKNOWN", "SEMI", "CLOSED", "OPEN"]
best_val_roc_auc = 0.0
best_model_path = "best_door_state_classifier_roc_auc.pth"
final_val_labels = None
final_val_probs = None

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_roc_auc, current_val_labels, current_val_probs = evaluate_and_get_roc_data(
        model, val_loader, criterion, device, num_classes=4
    )
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val ROC AUC: {val_roc_auc:.4f}")
    if val_roc_auc > best_val_roc_auc:
        best_val_roc_auc = val_roc_auc
        torch.save(model.state_dict(), best_model_path)
        print(f"Новая лучшая модель сохранена. ROC AUC: {val_roc_auc:.4f}")
        final_val_labels = current_val_labels
        final_val_probs = current_val_probs

final_model_path = "door_state_classifier_final_roc_auc.pth"
torch.save(model.state_dict(), final_model_path)
print(f"Финальная модель сохранена в {final_model_path}")
print(f"Лучшая модель сохранена в {best_model_path} с ROC AUC {best_val_roc_auc:.4f}")

if final_val_labels is not None and final_val_probs is not None:
    print("Построение ROC-кривой для лучшей модели...")
    plot_roc_curve(final_val_labels, final_val_probs, 4, CLASS_NAMES)
else:
    print("Нет данных для построения ROC-кривой.")
