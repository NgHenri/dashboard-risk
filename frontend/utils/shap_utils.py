import pandas as pd
import numpy as np
import joblib
import shap
import requests
import streamlit as st

# ========== Paramètres globaux ==========
API_URL = "http://localhost:8000"
API_KEY="b678481b982dc71ab46e08255faefae5f73339c4f1339eec83edf10488502158"
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
THRESHOLD = 0.0931515  # Seuil de risque
TIMEOUT = 10  # seconds

# ===== Chargement des données =====

# la fonction de chargement des données
@st.cache_data
def load_test_data_from_api():
    """Récupère les données de test via l'API"""
    try:
        headers = {
            "x-api-key": API_KEY,
            "Accept": "application/json"
        }
        
        # Test de connexion basique
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code != 200:
            st.error("L'API ne répond pas correctement")
            return pd.DataFrame()
        
        # Requête principale
        response = requests.get(
            f"{API_URL}/get_test_data",
            headers=headers,
            timeout=TIMEOUT
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
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['metadata']['features']
    explainer = shap.TreeExplainer(model)
    
    # Récupération des données via l'API pour SHAP global
    try:
        response = requests.get(
            f"{API_URL}/global_shap_sample",
            headers={"x-api-key": API_KEY},
            timeout=TIMEOUT
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

# ========== Chargement des données client ==========
def fetch_client_data_for_shap(client_id):
    try:
        response = requests.get(
            f"{API_URL}/client_shap_data/{client_id}",
            headers={"x-api-key": API_KEY},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return pd.DataFrame([response.json()])
    except Exception as e:
        st.error(f"Erreur API : {str(e)}")
        return pd.DataFrame()

# ========== Construction du DataFrame SHAP long ==========

# =================================================================================


#==================================================================================

def get_top_positive_negative_features(client_id: int, shap_df_long: pd.DataFrame, top_n: int = 5):
    """Retourne les top N features SHAP positives et négatives pour un individu"""
    client_shap = shap_df_long[shap_df_long["SK_ID_CURR"] == client_id]

    top_pos = (
        client_shap[client_shap["shap_value"] > 0]
        .nlargest(top_n, "shap_value")
        ["feature"]
        .tolist()
    )
    top_neg = (
        client_shap[client_shap["shap_value"] < 0]
        .nsmallest(top_n, "shap_value")
        ["feature"]
        .tolist()
    )

    return top_pos, top_neg


