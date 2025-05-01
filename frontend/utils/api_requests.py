# frontend/utils/api_requests.py

import joblib
import shap
from urllib.parse import urlparse
from requests.exceptions import (
    ConnectionError,
    Timeout,
    RequestException,
    InvalidURL,
    MissingSchema,
)

import streamlit as st
import os
from dotenv import load_dotenv
import pandas as pd

# =================================================================================
# API_URL = "http://localhost:8000"
# API_KEY = "b678481b982dc71ab46e08255faefae5f73339c4f1339eec83edf10488502158"
# ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
# THRESHOLD = 0.0931515  # Seuil de risque
# TIMEOUT = 10  # seconds
# ==================================================================================

# Charger .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Charger les variables d'environnement

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")
ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
THRESHOLD = float(os.getenv("THRESHOLD"))  # THRESHOLD doit être casté en float
COST_FN = int(os.getenv("COST_FN"))  # idem
COST_FP = int(os.getenv("COST_FP"))
GLOBAL_DATA_PATH = os.getenv("GLOBAL_DATA_PATH")
TIMEOUT = 10  # Timeout pour les requêtes


# --- Fonctions ---
def is_valid_url(url: str) -> bool:
    """Vérifie si l'URL a un schéma et un netloc (domaine + port valides)"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False


def check_api_health(timeout: int = 10) -> str | bool:
    """Vérifie si l'API est disponible. Retourne True ou un message d'erreur."""
    if not is_valid_url(API_URL):
        return "🚫 URL invalide – vérifie le format (ex: http://localhost:8000)"

    try:
        response = requests.get(f"{API_URL}/health", timeout=timeout)
        if response.status_code != 200:
            return f"⚠️ L'API a répondu avec le code {response.status_code}"
        return True
    except (InvalidURL, MissingSchema):
        return "🚫 URL invalide – vérifie le format (ex: http://localhost:8000)"
    except (ConnectionError, Timeout):
        return "❌ Impossible de se connecter – serveur backend non démarré ?"
    except RequestException as e:
        return f"⚠️ Problème lors de la requête : {str(e)}"


@st.cache_data
def fetch_client_ids():
    try:
        response = requests.get(f"{API_URL}/client_ids", timeout=TIMEOUT)
        if response.status_code == 200:
            return response.json().get("client_ids", [])
        return []
    except Exception as e:
        st.error(f"API non disponible : {str(e)}")
        return []


def fetch_client_info(client_id):
    try:
        response = requests.get(f"{API_URL}/client_info/{client_id}", timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API : {str(e)}")
        return None


def fetch_prediction(API_URL, client_data):
    """Envoie la requête de prédiction à l'API et retourne le résultat"""
    try:
        response = requests.post(
            f"{API_URL}/predict", json={"data": client_data}, timeout=TIMEOUT
        )
        response.raise_for_status()  # Vérifie si la requête a échoué
        result = response.json()  # Récupère la réponse JSON
        return result
    except requests.exceptions.RequestException as e:
        # Gère les erreurs de requêtes (erreur réseau, réponse incorrecte, etc.)
        st.error(f"Erreur lors de la prédiction : {e}")
        return None


import requests


def fetch_population_stats(feature, filters, api_url, api_key, sample_size=1000):
    """
    Fonction pour appeler l'API et récupérer les statistiques de population.

    Args:
        feature (str): La feature à analyser (nom de la feature).
        filters (dict): Filtres à appliquer dans l'API.
        api_url (str): L'URL de l'API.
        api_key (str): La clé API.
        sample_size (int, optional): Le nombre d'échantillons à envoyer. Par défaut, 1000.

    Returns:
        dict: Les statistiques de population récupérées.

    Raises:
        ValueError: En cas d'erreur d'appel API ou de structure invalide.
    """
    try:
        # Construction de la requête
        response = requests.post(
            f"{api_url}/population_stats",
            json={
                "feature": feature,
                "filters": filters,
                "sample_size": sample_size,
            },
            headers={"x-api-key": api_key},
        )

        # Vérification de la réponse
        if response.status_code == 200:
            data = response.json()
            if "stats" in data:
                return data["stats"]
            else:
                raise ValueError(
                    "Structure de la réponse API invalide : clé 'stats' manquante."
                )
        else:
            raise ValueError(f"Erreur API : {response.status_code} - {response.text}")

    except Exception as e:
        raise ValueError(f"Erreur lors de l'appel API : {str(e)}")


@st.cache_data(ttl=3600)
def fetch_test_data():
    """Récupère les données de test depuis l'API"""
    try:
        response = requests.get(
            f"{API_URL}/get_test_data", headers={"x-api-key": API_KEY}, timeout=TIMEOUT
        )
        response.raise_for_status()
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Erreur API : {str(e)}")
        st.stop()
