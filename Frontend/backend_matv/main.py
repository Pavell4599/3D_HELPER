# backend/main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import json
import random
import asyncio

app = FastAPI(title="3D Helper Core")

# Разрешаем Tauri обращаться к нам
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
    formats: list[str] = [".stl", ".obj"]

# --- 1. Упорядочивание (Organizer) ---
@app.post("/organize")
async def organize_folder(request: ScanRequest):
    """
    Сканирует папку, имитирует классификацию ModelNet и раскладывает файлы.
    """
    print(f"[AI CORE] Сканирование: {request.path}")
    
    if not os.path.exists(request.path):
        return {"status": "error", "message": "Папка не найдена"}

    categories = {"Мебель": [], "Детали": [], "Фигурки": [], "Архитектура": []}
    processed_count = 0

    # Имитация работы нейросети ModelNet
    for file in os.listdir(request.path):
        ext = os.path.splitext(file)[1].lower()
        if ext in request.formats:
            # В реальности здесь будет предсказание модели
            category = random.choice(list(categories.keys()))
            categories[category].append(file)
            
            target_dir = os.path.join(request.path, category)
            os.makedirs(target_dir, exist_ok=True)
            
            src_file = os.path.join(request.path, file)
            dst_file = os.path.join(target_dir, file)
            
            # Реальное перемещение файла
            try:
                shutil.move(src_file, dst_file)
                processed_count += 1
            except Exception as e:
                print(f"Ошибка перемещения {file}: {e}")

    return {
        "status": "success",
        "processed": processed_count,
        "details": categories
    }

# --- 2. Подбор настроек (Slicer Assistant) ---
@app.post("/analyze-model")
async def analyze_model(file: UploadFile = File(...)):
    """
    Анализирует модель и выдает настройки для слайсера.
    """
    print(f"[AI CORE] Анализ модели: {file.filename}")
    await asyncio.sleep(1) # Имитация анализа геометрии

    settings = {
        "Cura": {"layer_height": 0.2, "infill": 20, "support": "Tree"},
        "PrusaSlicer": {"layer_height": 0.15, "infill": 15, "support": "Organic"},
        "OrcaSlicer": {"layer_height": 0.16, "infill": 18, "support": "Auto"}
    }

    return {
        "model_name": file.filename,
        "complexity": "Medium",
        "settings": settings
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)