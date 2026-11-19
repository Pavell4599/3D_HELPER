import os
import torch
import torch.nn as nn
import numpy as np
import trimesh
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- 1. Архитектура модели (должна совпадать с той, на которой обучались) ---
# ВАЖНО: Вставьте сюда класс вашей PointNet. Пример стандартной архитектуры:
class PointNet(nn.Module):
    def __init__(self, num_classes=40):
        super(PointNet, self).__init__()
        # Упрощенная версия для примера. Замените на код из вашего репозитория.
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# --- 2. Константы ---
MODEL_PATH = 'best_pointnet_modelnet40.pth'
SORTED_DIR = './sorted_models'
# Список классов ModelNet40 (в том же порядке, что и при обучении) [citation:3]
CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
           'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
           'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
           'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
           'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
           'wardrobe', 'xbox']

app = Flask(__name__)
CORS(app) # Разрешаем запросы с вашего сайта

# Загрузка модели при старте
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PointNet(num_classes=40).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Модель успешно загружена.")
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")

def process_mesh_to_points(file_path, num_points=1024):
    """Читает меш и сэмплирует облако точек."""
    mesh = trimesh.load(file_path, force='mesh')
    # Сэмплируем точки с поверхности
    points, _ = trimesh.sample.sample_surface(mesh, count=num_points) # [citation:5][citation:10]
    
    # Нормализация (центрирование и приведение к единичной сфере)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
        
    # Формат для PointNet: (Batch, Channels, Points) -> (1, 3, 1024)
    tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(2, 1).to(device)
    return tensor

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Имя файла пустое'}), 400

    # Сохраняем во временный файл
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)

    try:
        # 1. Подготовка данных
        points_tensor = process_mesh_to_points(temp_path)
        
        # 2. Инференс
        with torch.no_grad():
            outputs = model(points_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            class_name = CLASSES[class_idx]
            
            # Получаем уверенность (softmax)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs[0][class_idx].item()

        # 3. Сортировка по папкам
        target_folder = os.path.join(SORTED_DIR, class_name)
        os.makedirs(target_folder, exist_ok=True)
        
        # Перемещаем или копируем файл в нужную папку
        target_path = os.path.join(target_folder, file.filename)
        os.rename(temp_path, target_path)

        return jsonify({
            'class': class_name,
            'confidence': round(confidence * 100, 2),
            'saved_to': target_path
        })

    except Exception as e:
        # Удаляем временный файл в случае ошибки
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs(SORTED_DIR, exist_ok=True)
    app.run(port=5000, debug=True)