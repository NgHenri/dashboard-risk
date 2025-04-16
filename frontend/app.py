import streamlit as st
import pandas as pd
import requests

API_URL = "http://localhost:8000"  # ou l'URL du serveur

# === Chargement des données test ===
df_test = pd.read_csv("../backend/data/test_1000_sample_for_api.csv")
client_ids = df_test["SK_ID_CURR"].unique()

st.title("🏦 Dashboard Crédit - Prédictions & Explicabilité")

# === Choix du ou des clients ===
selected_ids = st.multiselect("Sélectionner un ou plusieurs clients", client_ids, max_selections=5)

for client_id in selected_ids:
    st.subheader(f"Client ID : {client_id}")
    client_data = df_test[df_test["SK_ID_CURR"] == client_id].drop(columns=["SK_ID_CURR"]).to_dict(orient="records")[0]

    # === Prédiction ===
    with st.spinner("Prédiction en cours..."):
        response = requests.post(f"{API_URL}/predict", json={"data": client_data})
        result = response.json()
        st.write(f"📊 Probabilité de défaut : **{result['probability']}%**")
        st.write(f"💬 Décision : **{result['decision']}**")

    # === Explication SHAP ===
    with st.spinner("Analyse SHAP en cours..."):
        response = requests.post(f"{API_URL}/explain", json={"data": client_data})
        explanation = response.json()
        st.write("🔍 Explication (Top 10 features):")
        for feat in explanation["explanation"]:
            for k, v in feat.items():
                st.write(f"• {k} ➜ {v}")
