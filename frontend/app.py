import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import warnings
import joblib
import config
from risk_gauge import show_risk_gauge, display_risk_message, animate_risk_gauge

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

API_URL = "http://localhost:8000"
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
THRESHOLD = 0.0931515  # Seuil de risque

st.set_page_config(layout="wide")
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

# Gestion de la checkbox SHAP via session state
st.session_state.show_shap = st.sidebar.checkbox(
    "Afficher l'explication SHAP",
    value=st.session_state.get("show_shap", False)  # Utiliser .get() avec valeur par défaut
)

# === Initialisation de session state ===
required_states = {
    "predicted": False,
    "client_data": None,
    "score_float": None,
    "previous_id": None,
    "client_row": None,
    "show_shap": False  # <-- Ajouter cette ligne
}

for key, value in required_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===== Réinitialisation lors du changement d'ID =====
if st.session_state.previous_id != selected_id:
    for key in ["predicted", "client_data", "score_float", "show_shap"]:
        st.session_state[key] = required_states[key]
    # Réinitialiser l'état d'animation
    if 'current_animated_id' in st.session_state:
        del st.session_state.current_animated_id
    st.session_state.previous_id = selected_id


# ===== Soumission prédiction =====
if submitted:
    client_row = df_test[df_test["SK_ID_CURR"] == selected_id]
    
    if not client_row.empty:
        try:
            client_data = client_row.drop(columns=["SK_ID_CURR"]).to_dict(orient="records")[0]
            response = requests.post(f"{API_URL}/predict", json={"data": client_data})
            response.raise_for_status()
            result = response.json()
            
            # Mise à jour session state
            st.session_state.update({
                "score_float": float(result['probability']) / 100,
                "client_data": client_data,
                "client_row": client_row,
                "predicted": True
            })
            
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.session_state.predicted = False
    else:
        st.error("Client introuvable dans les données")
        st.session_state.predicted = False

# ===== Affichage des résultats =====
col_left, col_right = st.columns([1, 2])

# Colonne gauche - Toujours visible
with col_left:
    st.subheader("📋 Infos Client")
    
    # Fonctions de formatage
    # Fonctions de formatage
    def safe_get(row, col, default="N/A"):
        return row[col] if col in row and not pd.isna(row[col]) else default

    def format_currency(value):
        try:
            return f"{float(value):,.0f} €"
        except:
            return "N/A"

    def format_percentage(value):
        try:
            return f"{float(value)*100:.1f} %"
        except:
            return "N/A"

    def format_gender(value):
        return {1: "Homme", 0: "Femme"}.get(value, "Inconnu")

    def format_years(value):
        try:
            return f"{-int(value)//365} ans"
        except:
            return "N/A"

    # Sélection d'une ligne par ID
    row = df_test[df_test["SK_ID_CURR"] == selected_id].iloc[0]

    # Dictionnaire des infos formatées
    infos = {
        "ID Client": int(row["SK_ID_CURR"]),
        "Âge": format_years(safe_get(row, "DAYS_BIRTH")),
        "Genre": format_gender(safe_get(row, "CODE_GENDER")),
        "Charge crédit": format_percentage(safe_get(row, "INCOME_CREDIT_PERC")),
        "Historique crédit": format_years(safe_get(row, "BURO_DAYS_CREDIT_MEAN"))
    }

    # Construction du DataFrame pour affichage
    df_infos = pd.DataFrame(list(infos.items()), columns=["Libellé", "Valeur"])
    df_infos["Valeur"] = df_infos["Valeur"].astype(str)  # 🔥 force explicite en string
    st.dataframe(df_infos)

# Colonne droite - Résultats prédiction
with col_right:
    st.subheader("Analyse du risque client")

    if st.session_state.predicted:
        try:
            # --- 1. Gestion de l'animation ---
            # Réinitialiser l'état d'animation pour chaque nouvel ID
            if 'current_animated_id' not in st.session_state:
                st.session_state.current_animated_id = None
            
            if st.session_state.current_animated_id != selected_id:
                animate_risk_gauge(
                    score=st.session_state.score_float,
                    client_id=selected_id
                )
                st.session_state.current_animated_id = selected_id
            else:
                # Affichage statique si même client
                show_risk_gauge(
                    score=st.session_state.score_float, 
                    client_id=selected_id
                )

            # --- 2. Message d'alerte ---
            display_risk_message(
                score=st.session_state.score_float,
                threshold=THRESHOLD
            )

            # --- 3. Explications SHAP indépendantes ---
            if st.session_state.show_shap:
                st.markdown("---")
                with st.spinner("Génération des explications SHAP..."):
                    X = pd.DataFrame([st.session_state.client_data])[features]
                    X_scaled = pd.DataFrame(
                        scaler.transform(X),
                        columns=features,
                        index=X.index
                    )

                    shap_values = explainer(X_scaled)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.bar(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur d'affichage : {str(e)}")
    else:
        show_risk_gauge(None, client_id=selected_id)