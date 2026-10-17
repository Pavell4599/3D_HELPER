import os
import subprocess

# 1. Пути к файлам (замените на свои, если они отличаются)
SLICER_EXE = r"C:\Program Files\Bambu Studio\bambu-studio.exe"
MODEL_STL = r"C:\Users\Павел\Downloads\Jean_Bunny_Girl.stl"
SETTINGS_JSON = r"Data\settings\process_a1.json"
OUTPUT_3MF = r"Backend\static\settings\TEST.3mf"

def convert_stl_to_3mf():

    if not os.path.exists(MODEL_STL):
        print(f"Ошибка: Не найден STL файл по пути {MODEL_STL}")
        return
    if not os.path.exists(SETTINGS_JSON):
        print(f"Ошибка: Не найден JSON файл по пути {SETTINGS_JSON}")
        return


    command = [
        SLICER_EXE,
        "--load-settings", SETTINGS_JSON,
        "--scale", "85",
        "--export-3mf", OUTPUT_3MF,
        MODEL_STL
    ]
    print("Запуск конвертации.")
    
    try:
        # 3. Запуск слайсера в "тихом" CLI-режиме
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print(f"Успех! Файл сохранен в: {OUTPUT_3MF}")
        
    except subprocess.CalledProcessError as e:
        print("\nПроизошла ошибка при сборке 3MF:")
        print(e.stderr.decode('utf-8', errors='ignore'))

if __name__ == "__main__":
    convert_stl_to_3mf()
