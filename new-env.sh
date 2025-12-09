#!/bin/bash
# new.sh: inicializa el entorno y arranca Flask
ENV_PATH="./app/.env"
if [ ! -d "$ENV_PATH" ]; then
    echo "Creando entorno virtual en $ENV_PATH..."
    python3.10 -m venv "$ENV_PATH"
fi
echo "Activando entorno virtual..."
# Detectar si es Windows o Linux/Mac
if [ -f "$ENV_PATH/Scripts/activate" ]; then
    source "$ENV_PATH/Scripts/activate"
else
    source "$ENV_PATH/bin/activate"
fi

echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

export FLASK_APP=./app/api_gym.py
export FLASK_ENV=development

echo "Iniciando Flask..."
flask run
