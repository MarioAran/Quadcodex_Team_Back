FROM python:3.10-slim

# Crear directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Variables de entorno
ENV FLASK_APP=app/api_gym.py
ENV FLASK_RUN_HOST=0.0.0.0

# Exponer puerto
EXPOSE 5000

# Comando de arranque
CMD ["flask", "run"]