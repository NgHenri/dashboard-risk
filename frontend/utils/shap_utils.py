# frontend/utils/shap_utils.py

import pandas as pd
import numpy as np
import joblib
import shap
import requests
import streamlit as st
import os

# from dotenv import load_dotenv
from config import API_URL, API_KEY, TIMEOUT


# ========== Paramètres globaux ==========

# Charger .env
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Charger les variables d'environnement
# API_URL = "https://dashboard-risk.onrender.com"

# API_KEY = os.getenv("API_KEY")
# MODEL_PATH = os.getenv("MODEL_PATH")
# THRESHOLD = float(os.getenv("THRESHOLD"))  # THRESHOLD doit être casté en float
# COST_FN = int(os.getenv("COST_FN"))
# COST_FP = int(os.getenv("COST_FP"))
# GLOBAL_DATA_PATH = os.getenv("GLOBAL_DATA_PATH")


# ==================================================================================
def fetch_client_shap_data(client_id: int, use_api_key: bool = True):
    """
    Appelle /client_shap_data/{client_id}.
    Si use_api_key=True, ajoute l’en-tête x-api-key.
    """
    url = f"{API_URL}/client_shap_data/{client_id}"
    headers = {"x-api-key": API_KEY} if use_api_key else {}
    try:
        response = requests.get(url, headers=headers, timeout=TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur API SHAP data pour client {client_id} : {e}")
        return {}


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
    try:
        response = requests.get(
            f"{API_URL}/shap/local/{client_id}", headers={"x-api-key": API_KEY}
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Erreur lors de la récupération des explications SHAP : {e}")
        return None


@st.cache_data(show_spinner="Chargement des prédictions batch...", ttl=600)
def fetch_batch_predictions(
    df_clients: pd.DataFrame, filter_decision: str = None
) -> pd.DataFrame:
    """
    Envoie un batch de clients à l’API FastAPI et récupère les prédictions.

    Args:
        df_clients (pd.DataFrame): Données des clients à prédire.
        filter_decision (str, optional): "✅ Accepté", "❌ Refusé" ou None.

    Returns:
        pd.DataFrame: Résultats avec ['SK_ID_CURR', 'probability', 'decision']
    """
    if df_clients.empty:
        return pd.DataFrame()

    try:
        batch_data = df_clients.to_dict(orient="records")
        response = requests.post(f"{API_URL}/predict_batch", json={"data": batch_data})
        response.raise_for_status()
        results = response.json()

        df_results = df_clients.copy()
        df_results["probability"] = [r["probability"] for r in results]
        df_results["decision"] = [r["decision"] for r in results]

        if filter_decision in ["✅ Accepté", "❌ Refusé"]:
            df_results = df_results[df_results["decision"] == filter_decision]

        return df_results

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API lors de la prédiction batch : {e}")
        return pd.DataFrame()
