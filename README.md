### README для проекта `embedding`

---

## 🚀 Векторный поиск текстовых embeddings  
Микросервис для генерации векторных представлений текста с использованием модели `LaBSE-en-ru`.  
**Используется в проекте:** [gderabota.ru](https://gderabota.ru)

---

## ⚙️ Технические детали
- **Модель:** `cointegrated/LaBSE-en-ru` (поддержка рус/англ)
- **Фреймворк:** FastAPI
- **Логирование:** RotatingFileHandler (5 MB/файл, 3 бэкапа)
- **Потоки:** Фиксировано 4 потока (оптимизировано для прода)

---

## 🛠️ Установка и запуск

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/gderabota/embedding.git
cd embedding
```

### 2. Установите зависимости
```bash
pip install -r requirements.txt
```
*(Пример `requirements.txt`):*
```
fastapi
uvicorn[standard]
torch
transformers
logging
pydantic
```

### 3. Запустите сервер
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1
```
*Для продакшна используйте:*  
`--workers 4` + Nginx/Gunicorn (см. рекомендации ниже)

---

## 📡 Использование API

### Эндпоинт
`POST /embedding`

### Тело запроса (JSON)
```json
{
  "text": "Ваш текст для векторизации"
}
```

### Пример ответа (200 OK)
```json
{
  "embedding": [0.12, -0.05, ..., 0.87] // 768-мерный вектор
}
```

### Примеры запросов
#### cURL
```bash
curl -X POST "http://localhost:8000/embedding" \
-H "Content-Type: application/json" \
-d '{"text":"Привет, мир!"}'
```

#### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/embedding",
    json={"text": "Как работает векторный поиск?"}
)
print(response.json()["embedding"])
```

---

## ⚠️ Ошибки
| Код | Статус                  | Причина                     |
|-----|-------------------------|-----------------------------|
| 422 | Validation Error        | Неверный формат запроса     |
| 500 | Internal Server Error  | Ошибка обработки текста     |

---

## 🔧 Рекомендации для продакшна
1. **Докеризация**  
   Пример `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   COPY . /app
   WORKDIR /app
   RUN pip install -r requirements.txt
   CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
   ```

2. **Nginx конфиг**  
   Добавьте в `nginx.conf`:
   ```nginx
   location /embedding {
       proxy_pass http://localhost:8000;
       proxy_set_header Host $host;
   }
   ```

3. **Оптимизация GPU**  
   Замените `.to('cpu')` на `.to('cuda')` при наличии GPU.

---

## 📄 Логирование
Логи сохраняются в директорию `./logs`:
```
app.log       # Текущий лог
app.log.1     # Ротация (до 3 файлов)
```

---

## ⚖️ Лицензия
Проект использует [лицензию LaBSE](https://huggingface.co/cointegrated/LaBSE-en-ru) модели.  
**Коммерческое использование:** Проверьте совместимость с вашим проектом!

---

> **Важно!** Сервис требует ~1.5 ГБ RAM при запуске.  
> Для вопросов: ceo@gderabota.ru
