import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import warnings
import joblib
import config
from risk_gauge import show_risk_gauge

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

API_URL = "http://localhost:8000"
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"

st.set_page_config(layout="wide")  # Pleine largeur
st.title("🏦 Dashboard Crédit - Prédictions & Explicabilité")

# ===== Chargement des données =====
@st.cache_data
def load_test_data():
    return pd.read_csv("../backend/data/test_2000_sample_for_api.csv")

@st.cache_resource
def load_model_artifacts():
    artifacts = joblib.load(ARTIFACT_PATH)
    return artifacts['model'], artifacts['scaler'], artifacts['metadata']['features'], shap.TreeExplainer(artifacts['model'])

df_test = load_test_data()
model, scaler, features, explainer = load_model_artifacts()

client_ids = df_test["SK_ID_CURR"].unique().astype(int)

# ===== Sidebar =====
st.sidebar.markdown("## 🔍 Analyse d'un client")
selected_id = st.sidebar.selectbox("Sélectionner un client", client_ids)
submitted = st.sidebar.button("Soumettre la prédiction")
show_shap = st.sidebar.checkbox("Afficher l'explication SHAP", value=False)

# === Initialisation de session state pour ne pas perdre les données ===
if "predicted" not in st.session_state:
    st.session_state.predicted = False
if "client_data" not in st.session_state:
    st.session_state.client_data = None
if "score_float" not in st.session_state:
    st.session_state.score_float = None
if "show_shap" not in st.session_state:
    st.session_state.show_shap = show_shap  # Initialiser l'état de la checkbox

# ===== Réinitialisation lors du changement d'ID =====
if "previous_id" in st.session_state and st.session_state.previous_id != selected_id:
    st.session_state.predicted = False
    st.session_state.client_data = None
    st.session_state.score_float = None
    st.session_state.previous_id = selected_id

if "previous_id" not in st.session_state:
    st.session_state.previous_id = selected_id

# ===== Soumission prédiction =====
if submitted:
    client_row = df_test[df_test["SK_ID_CURR"] == selected_id]
    client_data = client_row.drop(columns=["SK_ID_CURR"]).to_dict(orient="records")[0]

    try:
        response = requests.post(f"{API_URL}/predict", json={"data": client_data})
        response.raise_for_status()
        result = response.json()
        st.session_state.score_float = float(result['probability']) / 100
        st.session_state.client_data = client_data
        st.session_state.client_row = client_row
        st.session_state.predicted = True
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# ===== Affichage des résultats =====
if st.session_state.predicted and st.session_state.client_data is not None:
    client_data = st.session_state.client_data
    client_row = st.session_state.client_row

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📋 Infos Client")

        def format_currency(value):
            try:
                if pd.isna(value):
                    return "N/A"
                return f"{float(value):,.0f} €"
            except (ValueError, TypeError):
                return "N/A"

        def format_percentage(value):
            try:
                if pd.isna(value):
                    return "N/A"
                return f"{100 * float(value):.1f} %"
            except (ValueError, TypeError):
                return "N/A"

        def format_gender(gender):
            if gender == 1:
                return "Homme"
            elif gender == 0:
                return "Femme"
            return "Inconnu"

        def format_years_from_days(days):
            try:
                if pd.isna(days):
                    return "N/A"
                return f"{-int(days) // 365} ans"
            except (ValueError, TypeError):
                return "N/A"

        row = client_row.iloc[0]

        attributes = ["ID Client", "Âge", "Genre", "Charge du crédit", "Historique crédit"]
        values = [
            int(row["SK_ID_CURR"]),
            format_years_from_days(row.get("DAYS_BIRTH")),
            format_gender(row.get("CODE_GENDER")),
            format_percentage(row.get("INCOME_CREDIT_PERC")),
            format_years_from_days(row.get("BURO_DAYS_CREDIT_MEAN"))
        ]

        if "TARGET" in row:
            attributes.append("Risque (historique)")
            values.append("⚠️ À risque" if row["TARGET"] == 1 else "✅ Sain")

        df_infos = pd.DataFrame({
            "Attribut": attributes,
            "Valeur": values
        })

        st.table(df_infos)


    with col_right:
        show_risk_gauge(score=st.session_state.score_float, client_id=selected_id)

    # ===== Mettre à jour la session_state lorsque l'utilisateur coche/décoche la case SHAP =====
    st.session_state.show_shap = show_shap

    # ===== Explication SHAP conditionnelle =====
    if st.session_state.show_shap:
        st.markdown("---")
        col_shap, _ = st.columns([1, 1])
        with col_shap:
            st.subheader("🔍 Top 10 facteurs SHAP")
            with st.spinner("Analyse SHAP en cours..."):
                try:
                    X = pd.DataFrame([client_data])[features]
                    X_scaled = scaler.transform(X)
                    shap_values = explainer(X_scaled)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.bar(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Erreur SHAP : {e}")
