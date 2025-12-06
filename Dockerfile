# ----------------------
# Stage 1: Builder
# ----------------------
FROM python:3.12 AS builder
WORKDIR /install

# Устанавливаем необходимые инструменты для сборки
RUN apt-get update && apt-get install -y build-essential libffi-dev

# Копируем зависимости
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel --wheel-dir=/wheels -r requirements.txt

# ----------------------
# Stage 2: Final
# ----------------------
FROM python:3.12-slim
WORKDIR /app

# Копируем готовые колёса из билдера
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Копируем приложение
COPY app/ /app/

# Запуск приложения
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
