#!/bin/bash
# entrypoint.sh

# Utilisez le port fourni par Railway ou 10000 par défaut
PORT=${PORT:-10000}

echo "🚀 Démarrage de l'app sur le port $PORT"
exec uvicorn main:app --host 0.0.0.0 --port $PORT