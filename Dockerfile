# Usamos Python 3.11 base
FROM python:3.11-slim

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos necesarios
COPY requirements.txt .
COPY app/ ./app

# Actualizamos pip y instalamos dependencias
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Exponemos el puerto que usará Render
EXPOSE 10000

# Comando para iniciar Gunicorn
CMD ["gunicorn", "app.api_gym:app", "--bind", "0.0.0.0:10000"]

