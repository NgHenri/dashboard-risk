import streamlit as st
import pandas as pd
import requests
import shap
import matplotlib.pyplot as plt
import warnings
import joblib
import config
from risk_gauge import show_risk_gauge, display_risk_message, animate_risk_gauge
import numpy as np
from utils.formatters import (
    safe_get, format_currency, format_percentage,
    format_gender, format_years
)
from utils.styling import style_rules, build_dynamic_styling
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
from utils.visuals import plot_boxplot_comparison
from dotenv import load_dotenv
import os

#load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
#load_dotenv()

#API_URL = os.getenv("API_URL")
#API_KEY = os.getenv("API_KEY")

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

API_URL = "http://localhost:8000"
API_KEY="b678481b982dc71ab46e08255faefae5f73339c4f1339eec83edf10488502158"
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
THRESHOLD = 0.0931515  # Seuil de risque

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except:
        return False

if not check_api_health():
    st.error("L'API n'est pas disponible. Veuillez démarrer le backend.")
    st.stop()

st.set_page_config(layout="wide")
st.title("🏦 Dashboard Crédit - Prédictions & Explicabilité")

# ===== Chargement des données =====
@st.cache_data
def load_test_data():
    return pd.read_csv("../backend/data/test_2000_sample_for_api.csv")

@st.cache_resource
def load_model_artifacts():
    artifacts = joblib.load(ARTIFACT_PATH)
    model = artifacts['model']
    scaler = artifacts['scaler']
    features = artifacts['metadata']['features']
    explainer = shap.TreeExplainer(model)
    
    # Précalcul des SHAP values globales (échantillonné pour plus de rapidité)
    df_test_sample = df_test[features].sample(min(1000, len(df_test)), random_state=42)
    df_test_sample_scaled = scaler.transform(df_test_sample)
    global_shap_values = explainer.shap_values(df_test_sample_scaled)
    
    return model, scaler, features, explainer, global_shap_values, df_test_sample

df_test = load_test_data()
model, scaler, features, explainer, global_shap_values, df_test_sample = load_model_artifacts()

# ===== Fonctions de service =====
@st.cache_data
def load_test_data_from_api():
    if not API_KEY or not API_URL:
        st.error("Clé API ou URL manquante dans le fichier .env.")
        return pd.DataFrame()

    try:
        headers = {"x-api-key": API_KEY}
        response = requests.get(f"{API_URL}/get_test_data", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data)
        else:
            st.error(f"Erreur API : {response.status_code} - {response.text}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Exception lors de la requête API : {e}")
        return pd.DataFrame()


@st.cache_data
def fetch_client_ids():
    try:
        response = requests.get(f"{API_URL}/client_ids", timeout=0.05)
        if response.status_code == 200:
            return response.json().get("client_ids", [])
        return []
    except Exception as e:
        st.error(f"API non disponible : {str(e)}")
        return []

def fetch_client_info(client_id):
    try:
        response = requests.get(
            f"{API_URL}/client_info/{client_id}",
            timeout=5  # 5 secondes max
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur API : {str(e)}")
        return None

client_ids = fetch_client_ids()

# ===== Sidebar =====
st.sidebar.markdown("## 🔍 Analyse d'un client")
selected_id = st.sidebar.selectbox("Sélectionner un client", client_ids)
if not selected_id or not isinstance(selected_id, int):
    st.warning("Veuillez sélectionner un client valide")
    st.stop()
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

# ==== Visualisation =======    
def plot_shap_waterfall(shap_data):
    plt.figure(figsize=(10, 6))
    shap.plots._waterfall.waterfall_legacy(
        shap_data["base_value"],
        np.array(shap_data["values"]),
        feature_names=shap_data["features"]
    )
    st.pyplot(plt.gcf())
    plt.close()


# ===== Soumission prédiction =====
if submitted:
    #client_row = df_test[df_test["SK_ID_CURR"] == selected_id]
    client_info = fetch_client_info(selected_id)
    if not client_info:
        st.error("Données client non disponibles")
        st.stop()

    # Conversion en DataFrame
    client_row = pd.DataFrame([client_info])

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
col_left, col_right = st.columns([2, 3])

# Colonne gauche - Toujours visible
with col_left:
    st.subheader("📋 Infos Client")

    client_info = fetch_client_info(selected_id)

    if client_info is not None:
        row = pd.Series(client_info)
    else:
        st.stop()


    def get_employment_type(row):
        mapping = {
            "NAME_INCOME_TYPE_PENSIONER": "Retraité",
            "NAME_INCOME_TYPE_WORKING": "Salarié",
            "NAME_INCOME_TYPE_STUDENT": "Étudiant",
            "ORGANIZATION_TYPE_SELF_EMPLOYED": "Indépendant",
            "ORGANIZATION_TYPE_MILITARY": "Militaire"
        }
        for col, label in mapping.items():
            if safe_get(row, col) == 1:
                return label
        return "Autre"

    def format_days_delay(value):
        try:
            if pd.isna(value) or value <= 0:
                return "Aucun retard"
            return f"{int(value)} jours de retard"
        except:
            return "N/A"

    # Sélection d'une ligne par ID

    #row = df_test[df_test["SK_ID_CURR"] == selected_id].iloc[0]

    # Dictionnaire des infos formatées
    infos = {
    "ID Client": int(row["SK_ID_CURR"]),
    "Âge": format_years(safe_get(row, "DAYS_BIRTH")),
    "Genre": format_gender(safe_get(row, "CODE_GENDER")),
    "Type de logement": "Locataire" if safe_get(row, "NAME_HOUSING_TYPE_RENTED_APARTMENT") == 1 else "Autre",
    "Statut marital": "Marié(e)" if safe_get(row, "NAME_FAMILY_STATUS_MARRIED") == 1 else "Autre",
    "Type d'emploi": get_employment_type(row),
    "Stabilité professionnelle": format_percentage(safe_get(row, "DAYS_EMPLOYED_PERC")),
    "Revenu par personne": format_currency(safe_get(row, "INCOME_PER_PERSON")),
    "Montant du crédit demandé": format_currency(safe_get(row, "AMT_CREDIT")),
    "Charge crédit (revenu vs crédit)": format_percentage(safe_get(row, "INCOME_CREDIT_PERC")),
    "Poids des remboursements sur le revenu": format_percentage(safe_get(row, "ANNUITY_INCOME_PERC")),
    "Historique crédit (ancienneté moyenne)": format_years(safe_get(row, "BURO_DAYS_CREDIT_MEAN")),
    "Dernier retard de paiement": format_days_delay(safe_get(row, "INSTAL_DBD_MAX")),
}


    # Construction du DataFrame pour affichage
    df_infos = pd.DataFrame(list(infos.items()), columns=["Libellé", "Valeur"])
    df_infos["Valeur"] = df_infos["Valeur"].astype(str)  # 🔥 force explicite en string
    df_infos["Afficher"] = False  # Colonne pour checkboxes
    # Construction du GridOptions
    gb = GridOptionsBuilder.from_dataframe(df_infos)
    gb.configure_column("Afficher", editable=True)  # rendre la colonne interactive
    gb.configure_grid_options(domLayout='normal')

    grid_options = gb.build()

    # Affichage interactif avec retour des valeurs modifiées
    grid_response = AgGrid(
        df_infos,
        gridOptions=grid_options,
        height=500,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True  # évite certains bugs liés à la JS marshalling
    )

    # DataFrame mis à jour par les interactions utilisateur
    updated_df = grid_response['data']

    genre_client = None
    try:
        genre_client = updated_df.loc[updated_df["Libellé"] == "Genre", "Valeur"].values[0]
    except IndexError:
        pass  # Par sécurité si jamais non trouvé

    # Boucle sur les lignes cochées pour affichage de graphiques
    for _, row in updated_df.iterrows():
        if row["Afficher"]:
            label = row["Libellé"]
            valeur = row["Valeur"]

            # ============================
            # ÂGE
            # ============================
            if label == "Âge":
                # Détermine le filtre genre
                if genre_client == "Homme":
                    population_df = df_test[df_test["CODE_GENDER"] == 1]
                elif genre_client == "Femme":
                    population_df = df_test[df_test["CODE_GENDER"] == 0]
                else:
                    population_df = df_test

                client_age = float(valeur.split()[0])
                population = population_df["DAYS_BIRTH"]
                plot_boxplot_comparison(
                    population_series=population,
                    client_value=client_age,
                    title=f"Âge du client (Genre: {genre_client})",
                    xlabel="Âge (années)",
                    unit=" ans",
                    transform=lambda x: -x / 365
                )

            # ============================
            # REVENU PAR PERSONNE
            # ============================
            elif label == "Revenu par personne":
                client_val = float(valeur.replace("€", "").replace(" ", "").replace(",", "."))

                # Vérifie si "Genre" est aussi affiché
                genre_active = "Genre" in updated_df.loc[updated_df["Afficher"], "Libellé"].values

                if genre_active:
                    # Deux boxplots : homme vs femme
                    df_filtered = df_test[["INCOME_PER_PERSON", "CODE_GENDER"]].dropna().copy()
                    df_filtered["GENRE"] = df_filtered["CODE_GENDER"].map({1: "Homme", 0: "Femme"})

                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df_filtered, x="GENRE", y="INCOME_PER_PERSON", ax=ax)

                    if genre_client in ["Homme", "Femme"]:
                        pos_x = 0 if genre_client == "Homme" else 1
                        ax.scatter(pos_x, client_val, color='red', zorder=10, s=100, label="Client")
                        ax.legend()

                    ax.set_title("Revenu par personne – comparaison genre")
                    ax.set_xlabel("Genre")
                    ax.set_ylabel("Revenu (€)")
                    st.pyplot(fig)

                else:
                    # Filtré selon genre
                    if genre_client == "Homme":
                        population_df = df_test[df_test["CODE_GENDER"] == 1]
                    elif genre_client == "Femme":
                        population_df = df_test[df_test["CODE_GENDER"] == 0]
                    else:
                        population_df = df_test

                    population = population_df["INCOME_PER_PERSON"]
                    plot_boxplot_comparison(
                        population_series=population,
                        client_value=client_val,
                        title=f"Revenu par personne (Genre: {genre_client})",
                        xlabel="Revenu (€)",
                        unit="€"
                    )

            # ============================
            # STABILITÉ PROFESSIONNELLE
            # ============================
            elif label == "Stabilité professionnelle":
                client_val = float(valeur.replace("%", "").replace(",", ".")) / 100
                genre_active = "Genre" in updated_df.loc[updated_df["Afficher"], "Libellé"].values

                if genre_active:
                    df_filtered = df_test[["DAYS_EMPLOYED_PERC", "CODE_GENDER"]].dropna().copy()
                    df_filtered["GENRE"] = df_filtered["CODE_GENDER"].map({1: "Homme", 0: "Femme"})

                    fig, ax = plt.subplots(figsize=(8, 5))
                    sns.boxplot(data=df_filtered, x="GENRE", y="DAYS_EMPLOYED_PERC", ax=ax)

                    if genre_client in ["Homme", "Femme"]:
                        pos_x = 0 if genre_client == "Homme" else 1
                        ax.scatter(pos_x, client_val, color='red', zorder=10, s=100, label="Client")
                        ax.legend()

                    ax.set_title("Stabilité professionnelle – comparaison genre")
                    ax.set_xlabel("Genre")
                    ax.set_ylabel("Ratio emploi (%)")
                    ax.set_ylim(0, 1)
                    st.pyplot(fig)

                else:
                    if genre_client == "Homme":
                        population_df = df_test[df_test["CODE_GENDER"] == 1]
                    elif genre_client == "Femme":
                        population_df = df_test[df_test["CODE_GENDER"] == 0]
                    else:
                        population_df = df_test

                    population = population_df["DAYS_EMPLOYED_PERC"]
                    plot_boxplot_comparison(
                        population_series=population,
                        client_value=client_val,
                        title=f"Stabilité professionnelle (Genre: {genre_client})",
                        xlabel="Pourcentage",
                        unit="%",
                        transform=lambda x: x * 100
                    )



        # --- Analyse SHAP Globale ---
    if st.session_state.predicted and st.session_state.show_shap:
        st.markdown("---")
        st.subheader("Analyse Globale")
        with st.spinner("Calcul des tendances globales..."):
            try:
                # Récupération des données brutes
                response = requests.get(f"{API_URL}/global_shap_matrix?sample_size=1000")
                response.raise_for_status()
                data = response.json()
                
                # Conversion des données
                shap_values = np.array(data['shap_values'])
                feature_values = pd.DataFrame(data['feature_values'])
                features = data['features']
                base_value = data['base_value']
                
                # Création d'un array de base_values adapté
                n_samples = shap_values.shape[0]
                base_values = np.full(n_samples, base_value)  # <-- Correction clé
                
                # Création de l'objet Explanation
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=base_values,  # Maintenant un array
                    data=feature_values.values,
                    feature_names=features
                )
                
                # Génération du plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(
                    explanation,
                    plot_type="dot",
                    show=False
                )
                
                # Personnalisation
                plt.title("Impact Global des Variables", pad=20)
                st.pyplot(plt.gcf())
                plt.close()
                
            except Exception as e:
                st.error(f"Erreur : {str(e)}")
# Colonne droite - Résultats prédiction
with col_right:
    #st.subheader("Analyse du risque client")

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
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    st.pyplot(fig)

        except Exception as e:
            st.error(f"Erreur d'affichage : {str(e)}")
    else:
        show_risk_gauge(None, client_id=selected_id)