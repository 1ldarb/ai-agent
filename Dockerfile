# 1. Берем легкий образ Python
FROM python:3.11-slim

# 2. Создаем рабочую папку внутри контейнера
WORKDIR /app

# 3. Копируем файл зависимостей и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Копируем код сервера
COPY server.py .

# 5. Команда для запуска сервера (слушаем порт 8000)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

