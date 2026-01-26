# 1. Берем легкий образ Python
FROM python:3.11-slim

# 2. Создаем рабочую папку внутри контейнера
WORKDIR /app

# 3. Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. ИСПРАВЛЕНО: Копируем ВСЕ файлы (включая faq.txt и другие скрипты)
COPY . .

# 5. Запуск (слушаем порт 8000)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
