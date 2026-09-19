import subprocess

# 1. Укажите ваши пути
BAMBU_EXE = r"C:\Program Files\Bambu Studio\bambu-studio.exe"
MODEL_PATH = r"C:\path\to\your\model.stl"          # Путь к 3D-модели
CONFIG_PATH = r"C:\path\to\your\settings.json"    # Путь к вашему JSON-профилю

# 2. Запуск одной командой
subprocess.Popen([
    BAMBU_EXE, 
    MODEL_PATH, 
    "--load_settings", CONFIG_PATH
])
