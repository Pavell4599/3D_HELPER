# backend/main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import json
import asyncio

app = FastAPI(title="3D Helper Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Модели данных ---
class ScanRequest(BaseModel):
    path: str
    formats: list[str] = [".stl", ".obj", ".fbx"]

class SlicerSettings(BaseModel):
    slicer_name: str
    model_class: str

# --- 1. Логика Упорядочивания (Organizer) ---
@app.post("/organize")
async def organize_folder(request: ScanRequest):
    """
    Сканирует папку, классифицирует файлы и раскладывает их по папкам.
    """
    print(f"[Python] Сканирование папки: {request.path}")
    
    # ЗАГЛУШКА: Имитация работы ModelNet
    # В реальности здесь будет загрузка модели и предсказание класса
    categories = {"Мебель": [], "Детали": [], "Фигурки": []}
    
    # Имитация поиска файлов
    if os.path.exists(request.path):
        for file in os.listdir(request.path):
            if any(file.endswith(fmt) for fmt in request.formats):
                # Случайное распределение для демо
                import random
                cat = random.choice(list(categories.keys()))
                categories[cat].append(file)
                
                # Создание папки и перемещение (в демо режиме просто логируем)
                target_dir = os.path.join(request.path, cat)
                os.makedirs(target_dir, exist_ok=True)
                # shutil.move(os.path.join(request.path, file), os.path.join(target_dir, file)) 
                
    return {
        "status": "success",
        "message": f"Обработано файлов. Распределено по категориям: {', '.join(categories.keys())}",
        "details": categories
    }

# --- 2. Логика Подбора Настроек (Slicer Assistant) ---
@app.post("/analyze-model")
async def analyze_model(file: UploadFile = File(...)):
    """
    Анализирует 3D модель и возвращает настройки печати.
    """
    print(f"[Python] Анализ модели: {file.filename}")
    
    # Имитация анализа геометрии
    await asyncio.sleep(1.5)
    
    # Пример настроек для разных слайсеров
    settings = {
        "Cura": {
            "layer_height": 0.2,
            "infill_density": 20,
            "support_type": "Tree",
            "material": "PLA"
        },
        "PrusaSlicer": {
            "layer_height": 0.15,
            "infill_density": 15,
            "supports": "Organic",
            "material": "PETG"
        }
    }
    
    return {
        "status": "success",
        "model_name": file.filename,
        "detected_complexity": "Medium", # Результат работы ИИ
        "recommended_settings": settings
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)