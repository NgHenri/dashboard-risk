# Étape 1 : image légère avec Python
FROM python:3.11-slim


# Étape 2 : Installer dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Étape 3 : création d’un répertoire de travail
WORKDIR /app

# Étape 4 : copie du fichier requirements
COPY backend/requirements.txt .

# Étape 5 : installation des dépendances Python 
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Étape 6 : copie du code fast api
COPY backend .

# Étape 7 : copie du csv
COPY backend/data ./data

# ✅ Étape 8 : copie du modèle
COPY backend/models ./models

# Étape 9 : définition du port et de la commande de démarrage
EXPOSE $PORT

# Copie du script d'entrée
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]