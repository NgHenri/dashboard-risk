import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import warnings
import joblib

# Désactivation des avertissements
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# URL de l'API (changer l'URL selon ton serveur)
API_URL = "http://localhost:8000"  # Remplace par l'URL de ton serveur

# === Chargement des données test ===
df_test = pd.read_csv("../backend/data/test_1000_sample_for_api.csv")
client_ids = df_test["SK_ID_CURR"].unique()

st.title("🏦 Dashboard Crédit - Prédictions & Explicabilité")

# Désactivation des détails d'erreurs de Streamlit
st.set_option('client.showErrorDetails', False)

# === Choix d'un client ===
selected_id = st.selectbox("Sélectionner un client", client_ids)

# Initialisation de SHAP
# Charger les artefacts du modèle comme avant
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
artifacts = joblib.load(ARTIFACT_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['metadata']['features']
explainer = shap.Explainer(model)

# Bouton pour soumettre la prédiction et analyse SHAP
submitted = st.button("Submit")

if submitted:
    st.subheader(f"Client ID : {selected_id}")
    
    # Sélection des données du client et transformation en dict
    client_data = df_test[df_test["SK_ID_CURR"] == selected_id].drop(columns=["SK_ID_CURR"]).to_dict(orient="records")[0]

    # === Prédiction ===
    with st.spinner("Prédiction en cours..."):
        try:
            response = requests.post(f"{API_URL}/predict", json={"data": client_data})
            response.raise_for_status()  # Cela lance une erreur si le code de statut HTTP n'est pas 2xx
            result = response.json()
            st.write(f"📊 Probabilité de défaut : **{result['probability']}%**")
            st.write(f"💬 Décision : **{result['decision']}**")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la prédiction : {e}")

    # === Explication SHAP ===
    with st.spinner("Analyse SHAP en cours..."):
        try:
            # On transforme les données du client pour SHAP
            X = pd.DataFrame([client_data])[features]
            X_scaled = scaler.transform(X)
            shap_values = explainer(X_scaled)

            # Création d'un graphique SHAP avec summary_plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, X, plot_type="bar", show=False)  # Bar plot pour plus de clarté
            st.pyplot(fig)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de l'analyse SHAP : {e}")

