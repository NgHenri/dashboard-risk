#!/bin/bash
# entrypoint.sh

echo "🚀 Démarrage de l'app sur le port $PORT"
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
