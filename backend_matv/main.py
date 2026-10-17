# backend/main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

app = FastAPI(title="NeuroCore Backend")

# Разрешаем запросы от Tauri (localhost)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    text: str

@app.post("/generate")
async def generate_image(request: PromptRequest):
    """
    Заглушка для твоей нейросети.
    Здесь ты подключишь свою модель (Stable Diffusion, LLM и т.д.)
    """
    print(f"[Python] Получен запрос: {request.text}")
    
    # Имитация сложного вычисления
    await asyncio.sleep(2) 
    
    return {
        "status": "success",
        "message": f"Сгенерировано на основе: '{request.text}'",
        "mock_url": "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?w=800&q=80" # Пример картинки
    }

if __name__ == "__main__":
    # Запуск сервера на порту 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)