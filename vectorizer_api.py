import logging
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import os

# Настройка числа потоков(для сервера 4 потока, чтобы не падал прод)
# На локалке используйте os.cpu_count(), функция вернет доступное количество потоков
torch.set_num_threads(4)

# Настройка логирования
log_dir = "./logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("vectorizer_api")
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(os.path.join(log_dir, "app.log"), maxBytes=5*1024*1024, backupCount=3)
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

logger.info("FastAPI приложение запущено")

# Загрузка модели
model_name = "cointegrated/LaBSE-en-ru"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to('cpu')
    model.eval()
    logger.info("Модель загружена: %s", model_name)
except Exception as e:
    logger.exception("Ошибка при загрузке модели")
    raise  # Приложение не может работать без модели, пробрасываем исключение дальше

# Инициализация FastAPI
app = FastAPI()

# Модель данных для входящего запроса
class TextRequest(BaseModel):
    text: str

# Глобальные обработчики исключений
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Необработанное исключение")
    return JSONResponse(
        status_code=500,
        content={"detail": "Внутренняя ошибка сервера"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("Ошибка валидации запроса: %s", str(exc))
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.post("/vector")
async def get_embedding(request: TextRequest):
    input_text = request.text
    try:
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            model_output = model(**inputs)
        token_embeddings = model_output.last_hidden_state
        attention_mask = inputs['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        embedding = sum_embeddings / sum_mask
        embedding_list = embedding.squeeze().tolist()
    except Exception as e:
        logger.exception("Ошибка при обработке текста")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки текста: {str(e)}")

    return {"embedding": embedding_list}
