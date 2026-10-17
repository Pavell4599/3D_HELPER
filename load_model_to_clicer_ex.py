import subprocess

# 1. Укажите ваши пути
BAMBU_EXE = r"C:\Program Files\Bambu Studio\bambu-studio.exe"
MODEL_PATH = r"C:\Users\Павел\Downloads\Jean_Bunny_Girl.stl"     # Путь к 3D-модели
CONFIG_PATH = r"settings\fdm_process_common.json"    # Путь к вашему JSON-профилю

# 2. Запуск одной командой
subprocess.Popen([
    BAMBU_EXE, 
    MODEL_PATH, 
    "--load-settings", CONFIG_PATH
])

#C:\Users\Павел\AppData\Roaming\BambuStudio\system\BBL\process - путь к файлам конфигурации бамбу лаб