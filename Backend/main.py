import os
import json
import subprocess

# Константы путей под архитектуру вашего проекта 3D_HELPER
SLICER_EXE = r"C:\Program Files\Bambu Studio\bambu-studio.exe"
SETTINGS_DIR = r"Backend\static\settings"
FINAL_JSON_PATH = os.path.join(SETTINGS_DIR, "process.json")
OUTPUT_3MF = r"Backend\static\settings\TEST.3mf"

# Базовый вариативный шаблон настроек Bambu Studio
BASE_CONFIG_TEMPLATE = {
    "print": {
        "setting_id": "AI_CLASS_PROFILE",
        "name": "AI Generated Class Profile",
        "from": "User",
        "instantiation": "true",
        "inherits": "0.20mm Standard @Bambulab X1 Carbon",
        "layer_height": "0.2",
        "initial_layer_height": "0.2",
        "wall_loops": "2",
        "top_shell_layers": "4",
        "bottom_shell_layers": "3",
        "sparse_infill_density": "15%",
        "sparse_infill_pattern": "gyroid",
        "enable_support": "0",
        "support_type": "normal(auto)",
        "outer_wall_speed": "200",
        "inner_wall_speed": "300"
    },
    "filament": {
        "setting_id": "AI_FILAMENT_PROFILE",
        "name": "AI Generic PLA",
        "from": "User",
        "instantiation": "true",
        "inherits": "Generic PLA @Bambulab X1 Carbon",
        "filament_type": "PLA",
        "nozzle_temperature": "220",
        "hot_plate_temp": "55"
    },
    "printer": {
        "setting_id": "AI_PRINTER_PROFILE",
        "name": "Bambu Lab X1 Carbon 0.4 nozzle",
        "from": "User",
        "instantiation": "true",
        "inherits": "Bambu Lab X1 Carbon 0.4 nozzle",
        "nozzle_diameter": "0.4",
        "printable_area": "0x0,256x0,256x256,0x256",
        "printable_height": "256"
    }
}

def generate_json_and_convert(model_path: str, ai_class: str):
    """
    Подставляет нужные параметры под класс ИИ, создает JSON
    и упаковывает STL в 3MF со скидкой масштаба до 85%.
    """
    # 1. Глубокое копирование базового шаблона
    config = json.loads(json.dumps(BASE_CONFIG_TEMPLATE))
    
    # 2. Логика подстановки параметров под разные классы ИИ
    print(f" Настройка профиля для класса: {ai_class}")
    
    if ai_class == "miniature":
        # Высокая детализация, тонкие слои, обязательные поддержки
        config["print"]["layer_height"] = "0.12"
        config["print"]["initial_layer_height"] = "0.2"
        config["print"]["wall_loops"] = "3"
        config["print"]["sparse_infill_density"] = "20%"
        config["print"]["enable_support"] = "1"
        config["print"]["outer_wall_speed"] = "120"  # Медленнее для качества
        
    elif ai_class == "mechanical":
        # Высокая прочность, толстый слой, плотное заполнение
        config["print"]["layer_height"] = "0.24"
        config["print"]["wall_loops"] = "4"
        config["print"]["top_shell_layers"] = "5"
        config["print"]["sparse_infill_density"] = "40%"
        config["print"]["sparse_infill_pattern"] = "grid"
        config["print"]["enable_support"] = "0"
        
    elif ai_class == "draft":
        # Быстрая черновая печать
        config["print"]["layer_height"] = "0.28"
        config["print"]["sparse_infill_density"] = "10%"
        config["print"]["sparse_infill_pattern"] = "lightning"
        config["print"]["enable_support"] = "0"
        config["print"]["outer_wall_speed"] = "250"
        
    else:
        # Стандартные настройки для обычных или неопознанных классов
        print(f" Используются стандартные настройки для класса '{ai_class}'")

    # 3. Запись готового JSON файла на диск
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    with open(FINAL_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f" Конфигурационный JSON успешно обновлен: {FINAL_JSON_PATH}")

    # 4. Формирование CLI команды для сборки проекта в 3MF
    command = [
        SLICER_EXE,
        "--load-settings", FINAL_JSON_PATH,
        "--scale", "85",  # Уменьшаем модель до 85%
        "--export-3mf", OUTPUT_3MF,
        model_path
    ]

    # 5. Вызов Bambu Studio в фоновом режиме
    try:
        print(" Запуск фонового процесса слайсера...")
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True
        )
        print(f" Проект 3MF успешно собран и сохранен: {OUTPUT_3MF}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(" Критическая ошибка при сборке 3MF слайсером:")
        print(e.stderr.decode('utf-8', errors='ignore'))
        return False

if __name__ == "__main__":
    # Эмуляция работы бэкенда
    TEST_MODEL = r"C:\Users\Павел\Downloads\Jean_Bunny_Girl.stl"
    
    # Сюда будет приходить класс, который определила ваша ИИ модель (PointNet/др.)
    # Варианты для проверки: "miniature", "mechanical", "draft", "standard"
    PREDICTED_CLASS = "miniature" 
    
    generate_json_and_convert(TEST_MODEL, PREDICTED_CLASS)
