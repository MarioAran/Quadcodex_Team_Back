# Base Python
FROM python:3.10-slim

# Crear directorio de la app
WORKDIR /gym_app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements primero
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar aplicación
COPY app/ ./app/
COPY Data/ ./Data/

# Verificar estructura
RUN echo "=== Estructura de directorios ===" && \
    ls -la && \
    echo "=== Contenido de app/ ===" && \
    ls -la app/

# Exponer puerto
EXPOSE 5000

# Variables de entorno
ENV FLASK_APP=app/app.py
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Comando directo
CMD ["python", "app/api_gym.py"]


