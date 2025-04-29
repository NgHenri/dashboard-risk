# frontend/utils/shap_utils.py

import pandas as pd
import numpy as np
import joblib
import shap
import requests
import streamlit as st

# ========== Paramètres globaux ==========
API_URL = "http://localhost:8000"
API_KEY = "b678481b982dc71ab46e08255faefae5f73339c4f1339eec83edf10488502158"
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
THRESHOLD = 0.0931515  # Seuil de risque
TIMEOUT = 10  # seconds

# Charger .env
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

# Charger les variables d'environnement
# API_URL = os.getenv("API_URL")
# API_KEY = os.getenv("API_KEY")
# MODEL_PATH = os.getenv("MODEL_PATH")
# THRESHOLD = float(os.getenv("THRESHOLD"))  # Attention ici : THRESHOLD doit être casté en float
# COST_FN = int(os.getenv("COST_FN"))
# COST_FP = int(os.getenv("COST_FP"))
# GLOBAL_DATA_PATH = os.getenv("GLOBAL_DATA_PATH")

# ===== Chargement des données =====


# la fonction de chargement des données
@st.cache_data
def load_test_data_from_api():
    """Récupère les données de test via l'API"""
    try:
        headers = {"x-api-key": API_KEY, "Accept": "application/json"}

        # Test de connexion basique
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code != 200:
            st.error("L'API ne répond pas correctement")
            return pd.DataFrame()

        # Requête principale
        response = requests.get(
            f"{API_URL}/get_test_data", headers=headers, timeout=TIMEOUT
        )

        # Gestion spécifique des erreurs 403
        if response.status_code == 403:
            st.error("Permission refusée - Vérifiez la clé API")
            logger.error(f"Headers envoyés : {headers}")
            return pd.DataFrame()

        response.raise_for_status()

        return pd.DataFrame(response.json())

    except Exception as e:
        st.error(f"Erreur critique : {str(e)}")
        return pd.DataFrame()


# la fonction de chargement des artefacts
@st.cache_resource
def load_model_artifacts():
    artifacts = joblib.load(ARTIFACT_PATH)
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    features = artifacts["metadata"]["features"]
    explainer = shap.TreeExplainer(model)

    # Récupération des données via l'API pour SHAP global
    try:
        response = requests.get(
            f"{API_URL}/global_shap_sample",
            headers={"x-api-key": API_KEY},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        df_test_sample = pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"Erreur API : {str(e)}")
        st.stop()

    # Calcul des SHAP values
    df_test_sample_scaled = scaler.transform(df_test_sample[features])
    global_shap_values = explainer.shap_values(df_test_sample_scaled)

    return model, scaler, features, explainer, global_shap_values, df_test_sample


# ==================================================================================
def fetch_client_data_for_shap(client_id):
    """Récupère les données formatées pour le calcul SHAP"""
    try:
        response = requests.get(
            f"{API_URL}/client_shap_data/{client_id}",
            headers={"x-api-key": API_KEY},
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return pd.DataFrame([response.json()])
    except Exception as e:
        st.error(f"Erreur API : {str(e)}")
        return pd.DataFrame()


def fetch_client_shap_data(client_id):
    response = requests.get(f"{API_URL}/client_shap_data/{client_id}")
    return response.json()


@st.cache_data(ttl=600)
def fetch_global_shap_matrix(sample_size=1000):
    response = requests.get(
        f"{API_URL}/global_shap_matrix?sample_size={sample_size}",
        headers={"x-api-key": API_KEY},
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300)
def fetch_local_shap_explanation(client_id: int):
    response = requests.get(
        f"{API_URL}/shap/local/{client_id}", headers={"x-api-key": API_KEY}
    )
    response.raise_for_status()
    return response.json()
