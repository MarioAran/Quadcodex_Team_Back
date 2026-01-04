# Base Python
FROM python:3.10-slim

# Crear directorio de la app
WORKDIR /gym_app

# Copiar dependencias y data
COPY requirements.txt .
COPY app/ app/
COPY Data/ Data/
COPY new-env.sh .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Dar permisos de ejecución al script (si lo necesita)
RUN chmod +x new-env.sh

# Exponer puerto Flask
EXPOSE 5000

# Comando por defecto
CMD ["./new-env.sh"]
