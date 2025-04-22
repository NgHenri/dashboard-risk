# -*- coding: utf-8 -*-

import os
from dotenv import load_dotenv, set_key
import secrets

# Chemin vers le .env du backend
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

# Charger le fichier .env s'il existe
load_dotenv(dotenv_path=ENV_PATH)

# Vérifier si la clé existe déjà
existing_key = os.getenv("API_KEY")

if existing_key:
    print(f"API_KEY déjà présente : {existing_key}")
else:
    # Générer une nouvelle clé
    new_key = secrets.token_urlsafe(32)
    # Créer le fichier .env s'il n'existe pas encore
    if not os.path.exists(ENV_PATH):
        open(ENV_PATH, "w").close()
    # Ajouter ou mettre à jour la clé dans le .env
    set_key(ENV_PATH, "API_KEY", new_key)
    print(f"Nouvelle API_KEY générée et enregistrée dans {ENV_PATH} :\n{new_key}")
