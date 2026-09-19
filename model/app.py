import os
import torch
import numpy as np
import trimesh


# Защита от конфликта библиотек OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Импортируем вашу модель
from model import PointNet 

CLASSES = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 
'bottle', 'bowl', 'car', 'chair', 'cone', 
'cup', 'curtain', 'desk', 'door', 'dresser', 
'flower', 'glass', 'guitar', 'keyboard', 'lamp', 
'laptop', 'mantel', 'monitor', 'night', 'person', 
'piano', 'plant', 'radio', 'range', 'sink', 
'sofa', 'stairs', 'stool', 'table', 'tent', 
'toilet', 'tv', 'vase', 'wardrobe', 'xbox']
WEIGHTS_PATH = "model/best_pointnet_modelnet40.pth" 

# =============================================================================
# ИНИЦИАЛИЗАЦИЯ МОДЕЛИ И ВЕСОВ
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(CLASSES)
model = PointNet(num_classes=num_classes)

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Файл весов '{WEIGHTS_PATH}' не найден!")

model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.eval()
model.to(device)
print(f"Модель успешно загружена на {device}. Родной парсер .off готов!")

# =============================================================================
# ВАША СТАНДАРТНАЯ ФУНКЦИЯ ЧТЕНИЯ .OFF ФАЙЛА (из скрипта обучения)
# =============================================================================
def read_off(file_path):
    with open(file_path, 'r') as f:
        # Проверяем заголовок файла
        header = f.readline().strip()
        if 'OFF' not in header:
            raise ValueError("Некорректный заголовок .OFF файла")
        
        # Если заголовок склеен со значениями (например, OFF 400 120 0)
        if len(header) > 3:
            line = header[3:].strip().split()
        else:
            line = f.readline().strip().split()
            
        if not line:
            line = f.readline().strip().split()
            
        n_verts = int(line[0])
        
        # Читаем координаты всех вершин
        verts = []
        for _ in range(n_verts):
            verts.append([float(x) for x in f.readline().strip().split()])
            
        return np.array(verts)
# =============================================================================
# ПРЕОБРАЗОВАТЕЛЬ ВСЕХ ФОРМАТОВ В .off
# =============================================================================
def convert_to_off(input_path: str, output_path: str) -> bool:
    """
    Принимает 3D-модель ЛЮБОГО популярного формата и конвертирует её в .off.
    
    Поддерживает: .obj, .stl, .fbx, .gltf, .glb, .ply, .dae, .3ds, .step и др.
    """
    if not os.path.exists(input_path):
        print(f"Ошибка: Входной файл не найден по пути: {input_path}")
        return False

    try:
        # 1. Загружаем сцену или сетку. Trimesh автоматически определяет формат по расширению.
        scene_or_mesh = trimesh.load(input_path)
        
        # 2. Обрабатываем случай, если файл сложный (состоит из группы объектов/сцены)
        if isinstance(scene_or_mesh, trimesh.Scene):
            print(f"Обнаружена 3D-сцена из {len(scene_or_mesh.geometry)} элементов. Объединяем геометрию...")
            # Объединяем все подобъекты сцены в один полигональный меш
            if len(scene_or_mesh.geometry) == 0:
                print("Ошибка: Сцена пуста, геометрия отсутствует.")
                return False
            mesh = trimesh.util.concatenate(
                [geom for geom in scene_or_mesh.geometry.values() if isinstance(geom, trimesh.Trimesh)]
            )
        else:
            # Если это уже одиночный меш (например, простой .stl или .obj)
            mesh = scene_or_mesh

        # 3. Экспортируем в формат .off
        # Важно: Формат .off хранит только геометрию (вершины и грани), без текстур.
        mesh.export(output_path, file_type='off')
        print(f"Успешно сконвертировано: {input_path} -> {output_path}")
        return True

    except Exception as e:
        print(f"Критическая ошибка при обработке файла {input_path}: {e}")
        return False

# =============================================================================
# ФУНКЦИЯ ОБРАБОТКИ И ИНФЕРЕНСА
# =============================================================================
def predict_only(file_obj):
    if file_obj is None:
        return {"Файл не выбран": 1.0}
    
    try:
        # 1. Получаем реальный путь к загруженному файлу из объекта Gradio
# Вместо input_path = file_obj.path используйте этот блок:

        if hasattr(file_obj, 'name'):
            input_path = file_obj.name
        elif isinstance(file_obj, dict) and 'name' in file_obj:
            input_path = file_obj['name']
        else:
            input_path = str(file_obj)

        
        # Создаем путь для сохранения временного .off файла
        output_off_path = "temp_converted_model.off"
        
        # 2. Конвертируем в .off с передачей ОБОИХ обязательных аргументов
        success = convert_to_off(input_path, output_off_path)
        if not success:
            return {"Ошибка конвертации файла в формат .off": 1.0}
            
        # 3. Читаем облако точек из созданного .off файла
        verts = read_off(output_off_path)
        
        # [Опционально] Удаляем временный .off файл после чтения, чтобы не засорять диск
        if os.path.exists(output_off_path):
            os.remove(output_off_path)
        
        # 4. Если точек больше или меньше 2048 — делаем сэмплирование/выборку
        num_points = 2048
        if len(verts) >= num_points:
            # Случайный выбор 2048 точек без повторений
            choice = np.random.choice(len(verts), num_points, replace=False)
        else:
            # Если точек не хватает, выбираем с повторениями
            choice = np.random.choice(len(verts), num_points, replace=True)
        points = verts[choice, :]
        
        # 5. Детерминированная нормализация (как при обучении)
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
            
        # Формируем тензор для вашей PointNet: [Batch=1, Num_Points=2048, Channels=3]
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 6. Прогон через нейросеть
        with torch.no_grad():
            outputs = model(points_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().squeeze(0).numpy()
            
        # Возвращаем результаты в формате {Класс: Вероятность}
        results = {CLASSES[i]: float(probabilities[i]) for i in range(num_classes)}
        return results
        
    except Exception as e:
        return {f"Ошибка при чтении или обработке файла: {str(e)}": 1.0}


if __name__ == '__main__':
    print('попа')


if __name__ == "__main__":
    # Скрипт «проглотит» любой из этих форматов:
    test_files = ["model.fbx", "part.stl", "avatar.gltf", "scan.ply", "building.obj"]
    
    # Симуляция работы функции
    input_file = "my_input_model.fbx"  # Подставьте имя вашего файла
    output_file = "converted_model.off"
    
    success = convert_to_off(input_file, output_file)
    if success:
        print("Файл готов к подаче на вход нейросети!")

