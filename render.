#!/bin/bash
# env_gunicorn.sh: inicializa dependencias y arranca Gunicorn

echo "Instalando dependencias..."
pip install --upgrade pip
pip install -r requirements.txt

export FLASK_APP=./app/api_gym.py
export FLASK_ENV=production

echo "Iniciando Gunicorn..."
gunicorn app.api_gym:app --bind 0.0.0.0:$PORT

