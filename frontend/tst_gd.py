import streamlit as st
import pandas as pd
import seaborn as sns
import requests
import shap
from matplotlib import pyplot as plt
import warnings
import joblib
import config
from risk_gauge import show_risk_gauge, display_risk_message, animate_risk_gauge
import numpy as np
from utils.formatters import (
    safe_get,
    format_currency,
    format_percentage,
    format_gender,
    format_years,
)
from utils.styling import style_rules, build_dynamic_styling

from utils.api_requests import (
    check_api_health,
    fetch_client_ids,
    fetch_client_info,
    fetch_prediction,
    fetch_population_stats,
)
from utils.user_interactions import (
    prepare_client_info,
    create_interactive_grid,
    process_user_interactions,
    build_feature_config,
    get_genre_client,
    get_updated_data,
    process_selection_and_display_plot,
)
from utils.shap_utils import (
    fetch_client_data_for_shap,
    fetch_client_shap_data,
    fetch_global_shap_matrix,
    fetch_local_shap_explanation,
)
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder, GridUpdateMode
from utils.visuals import (
    plot_boxplot_comparison,
    plot_waterfall_chart,
    plot_waterfall_chart_expandable,
    plot_summary_chart,
    get_title_font_size,
    plot_feature_distribution,
    plot_shap_by_decision,
    plot_shap_histogram,
)
from dotenv import load_dotenv
import os
import logging
import json


# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
# load_dotenv()

# API_URL = os.getenv("API_URL")
# API_KEY = os.getenv("API_KEY")


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

API_URL = "http://localhost:8000"
API_KEY = "b678481b982dc71ab46e08255faefae5f73339c4f1339eec83edf10488502158"
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
THRESHOLD = 0.0931515  # Seuil de risque
TIMEOUT = 10  # seconds


# 1. Fonction de vérification
if not check_api_health(TIMEOUT):
    st.error("Connexion API impossible - Vérifiez que le backend est démarré")
    st.stop()

st.set_page_config(layout="wide")
st.title("🏦 Dashboard Crédit - Prédictions & Explicabilité")


# ------------------ Main App UI ------------------ #

tab1, tab2, tab3, tab4 = st.tabs(["Prédiction", "Exploration", "Simulation", "Autre"])

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


# Chargement des données via l'API
df_test = load_test_data_from_api()

if df_test.empty:
    st.error("Erreur lors du chargement des données depuis l'API")
    st.stop()


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


# Chargement des données via l'API
model, scaler, features, explainer, global_shap_values, df_test_sample = (
    load_model_artifacts()
)

# ===== Fonctions de service =====


# Mettre à jour les appels dans les visualisations
def update_comparison_plots(updated_df, genre_client):
    for _, row in updated_df.iterrows():
        if row["Afficher"]:
            label = row["Libellé"]
            valeur = row["Valeur"]

            # Récupération des données de comparaison via API
            try:
                response = requests.post(
                    f"{API_URL}/population_stats",
                    json={
                        "feature": label,
                        "filters": {"genre": genre_client} if genre_client else {},
                    },
                    headers={"x-api-key": API_KEY},
                    timeout=TIMEOUT,
                )
                response.raise_for_status()
                stats = response.json()

                # Création du graphique avec les données de l'API
                plot_boxplot_comparison(
                    population_stats=stats,
                    client_value=parse_client_value(valeur, label),
                    title=f"Comparaison pour {label}",
                    unit=get_unit_for_label(label),
                )

            except Exception as e:
                st.error(f"Erreur API : {str(e)}")


client_ids = fetch_client_ids()
if not client_ids:
    st.error("Aucun client disponible - Vérifiez la connexion à l'API")
    st.stop()


# ======================================================================================


# ======================================================================================

# ===== Sidebar =====
st.sidebar.markdown("## 🔍 Analyse d'un client")
client_ids = sorted(client_ids)
selected_id = st.sidebar.selectbox("Sélectionner un client", client_ids)
if not selected_id or not isinstance(selected_id, int):
    st.warning("Veuillez sélectionner un client valide")
    st.stop()

submitted = st.sidebar.button("Soumettre la prédiction")

# Gestion de la checkbox SHAP via session state
st.session_state.show_shap = st.sidebar.checkbox(
    "Afficher l'explication SHAP",
    value=st.session_state.get(
        "show_shap", False
    ),  # Utiliser .get() avec valeur par défaut
)
st.markdown("---")
st.sidebar.markdown("## 🔧 Options")
compare_group = st.sidebar.radio(
    "Groupe de comparaison",
    ["Population totale", "Clients similaires"],
    help="Sélectionnez le groupe de référence pour les comparaisons",
)
# === Initialisation de session state ===
required_states = {
    "predicted": False,
    "client_data": None,
    "score_float": None,
    "previous_id": None,
    "client_row": None,
    "show_shap": False,  # <-- Ajouter cette ligne
}

for key, value in required_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===== Réinitialisation lors du changement d'ID =====
if st.session_state.previous_id != selected_id:
    for key in ["predicted", "client_data", "score_float", "show_shap"]:
        st.session_state[key] = required_states[key]
    # Réinitialiser l'état d'animation
    if "current_animated_id" in st.session_state:
        del st.session_state.current_animated_id
    st.session_state.previous_id = selected_id

# ===== Soumission prédiction =====
if submitted:
    client_info = fetch_client_info(selected_id)
    if not client_info:
        st.error("Données client non disponibles")
        st.stop()

    # Conversion en DataFrame
    client_row = pd.DataFrame([client_info])

    if not client_row.empty:
        try:
            client_data = client_row.drop(columns=["SK_ID_CURR"]).to_dict(
                orient="records"
            )[0]

            # Appel de la fonction fetch_predictionn
            result = fetch_prediction(API_URL, client_data)

            # Mise à jour session state
            st.session_state.update(
                {
                    "score_float": float(result["probability"]) / 100,
                    "client_data": client_data,
                    "client_row": client_row,
                    "predicted": True,
                }
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la prédiction : {e}")
            st.session_state.predicted = False
    else:
        st.error("Client introuvable dans les données")
        st.session_state.predicted = False

# ===== Affichage des résultats =====
col_left, col_right = st.columns([1, 1])


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
            "ORGANIZATION_TYPE_MILITARY": "Militaire",
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

    # Dictionnaire des infos formatées
    infos = {
        "ID Client": int(row["SK_ID_CURR"]),
        "Âge": format_years(safe_get(row, "DAYS_BIRTH")),
        "Genre": format_gender(safe_get(row, "CODE_GENDER")),
        "Type de logement": (
            "Locataire"
            if safe_get(row, "NAME_HOUSING_TYPE_RENTED_APARTMENT") == 1
            else "Autre"
        ),
        "Statut marital": (
            "Marié(e)" if safe_get(row, "NAME_FAMILY_STATUS_MARRIED") == 1 else "Autre"
        ),
        "Type d'emploi": get_employment_type(row),
        "Stabilité professionnelle": format_percentage(
            safe_get(row, "DAYS_EMPLOYED_PERC")
        ),
        "Revenu par personne": format_currency(safe_get(row, "INCOME_PER_PERSON")),
        "Montant du crédit demandé": format_currency(safe_get(row, "AMT_CREDIT")),
        "Charge crédit (revenu vs crédit)": format_percentage(
            safe_get(row, "INCOME_CREDIT_PERC")
        ),
        "Poids des remboursements sur le revenu": format_percentage(
            safe_get(row, "ANNUITY_INCOME_PERC")
        ),
        "Historique crédit (ancienneté moyenne)": format_years(
            safe_get(row, "BURO_DAYS_CREDIT_MEAN")
        ),
        "Dernier retard de paiement": format_days_delay(
            safe_get(row, "INSTAL_DBD_MAX")
        ),
    }

    # Construction du DataFrame pour affichage
    df_infos = pd.DataFrame(list(infos.items()), columns=["Libellé", "Valeur"])
    df_infos["Valeur"] = df_infos["Valeur"].astype(str)  # 🔥 force explicite en string
    df_infos["Afficher"] = False  # Colonne pour checkboxes

    # Construction du GridOptions
    # Configuration de la grille interactive
    gb = GridOptionsBuilder.from_dataframe(df_infos)
    gb.configure_column(
        "Afficher",
        editable=True,
        cellStyle={"color": "white", "background-color": "#4a6fa5"},
        headerClass="ag-header-cell-label",
    )
    gb.configure_grid_options(domLayout="normal")

    grid_response = AgGrid(
        df_infos,
        gridOptions=gb.build(),
        height=450,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
    )

    # DataFrame mis à jour par les interactions utilisateur
    updated_df = grid_response["data"]

    genre_client = None
    try:
        genre_client = updated_df.loc[
            updated_df["Libellé"] == "Genre", "Valeur"
        ].values[0]
    except IndexError:
        pass  # Par sécurité si jamais non trouvé

    for _, row in updated_df.iterrows():
        if row["Afficher"]:
            label = row["Libellé"]
            valeur = row["Valeur"]

            # Configuration dynamique par caractéristique
            feature_config = {
                "Âge": {
                    "api_feature": "DAYS_BIRTH",
                    "parse_func": lambda v: -float(v.split()[0]) * 365,
                    "transform_func": lambda x: -x / 365,
                    "unit": "ans",
                },
                "Revenu par personne": {
                    "api_feature": "INCOME_PER_PERSON",
                    "parse_func": lambda v: float(
                        v.replace("€", "").replace(" ", "").replace(",", "")
                    ),
                    "transform_func": None,
                    "unit": "€",
                },
                "Stabilité professionnelle": {
                    "api_feature": "DAYS_EMPLOYED_PERC",
                    "parse_func": lambda v: float(v.replace("%", "").replace(",", "."))
                    / 100,
                    "transform_func": lambda x: x * 100,
                    "unit": "%",
                },
            }

            # Récupération de la config
            config = feature_config.get(label, {})
            if not config:
                st.warning(f"Configuration manquante pour {label}")
                continue

            try:
                # Construction des filtres
                filters = {}
                if "Genre" in updated_df.loc[updated_df["Afficher"], "Libellé"].values:
                    filters["CODE_GENDER"] = 1 if genre_client == "Homme" else 0

                # Appel API générique
                response = requests.post(
                    f"{API_URL}/population_stats",
                    json={
                        "feature": config["api_feature"],
                        "filters": filters,
                        "sample_size": 1000,
                    },
                    headers={"x-api-key": API_KEY},
                )
                # Vérifier la structure de la réponse
                if response.status_code == 200:
                    data = response.json()
                    if "stats" not in data:
                        st.error("Structure de réponse API invalide")
                        continue

                    stats = data["stats"]
                else:
                    st.error(f"Erreur API : {response.status_code} - {response.text}")
                    continue

                response.raise_for_status()
                stats = response.json()["stats"]

                # Conversion de la valeur client
                client_value = config["parse_func"](valeur)

                # Génération du graphique
                plot_boxplot_comparison(
                    population_stats=stats,
                    client_value=client_value,
                    title=f"Position du client - {label}",
                    unit=config["unit"],
                    transform=config["transform_func"],
                )

            except Exception as e:
                st.error(f"Erreur lors de l'affichage de {label} : {str(e)}")

        # Expander pour afficher l'explication SHAP
    with st.expander("📖 Analyse Globale"):
        # --- Analyse SHAP Globale ---
        if st.session_state.predicted and st.session_state.show_shap:
            st.markdown("---")
            st.subheader("Analyse Globale")
            with st.spinner("Calcul des tendances globales..."):
                try:
                    # Récupération des données via le cache
                    data = fetch_global_shap_matrix(sample_size=1000)

                    # Conversion des données
                    shap_values = np.array(data["shap_values"])
                    feature_values = pd.DataFrame(data["feature_values"])
                    features = data["features"]  # Extraction depuis metadata
                    base_value = data["base_value"]

                    # Création d'un array de base_values adapté
                    n_samples = shap_values.shape[0]
                    base_values = np.full(n_samples, base_value)

                    # Création de la "structure" d'explication attendue par notre plot
                    explanation = {
                        "values": shap_values,
                        "data": feature_values.values,
                        "feature_names": features,
                    }

                    # Génération du summary plot avec Plotly
                    fig = plot_summary_chart(explanation, max_display=10)
                    st.plotly_chart(fig, use_container_width=True)

                    # Streamlit UI
                    selected_id = st.session_state.previous_id

                except Exception as e:
                    st.error(f"Erreur lors de l'analyse SHAP : {str(e)}")
                    st.stop()

# Colonne droite - Résultats prédiction
with col_right:
    # st.subheader("Analyse du risque client")

    if st.session_state.predicted:
        try:
            # --- 1. Gestion de l'animation ---
            # Réinitialiser l'état d'animation pour chaque nouvel ID
            if "current_animated_id" not in st.session_state:
                st.session_state.current_animated_id = None

            if st.session_state.current_animated_id != selected_id:
                animate_risk_gauge(
                    score=st.session_state.score_float, client_id=selected_id
                )
                st.session_state.current_animated_id = selected_id
            else:
                # Affichage statique si même client
                show_risk_gauge(
                    score=st.session_state.score_float, client_id=selected_id
                )

            # --- 2. Message d'alerte ---
            display_risk_message(
                score=st.session_state.score_float, threshold=THRESHOLD
            )
            # --- 3. Explications SHAP indépendantes ---
            if st.session_state.show_shap:
                st.markdown("---")
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)
                st.subheader("📖 Explication du score")
                with st.spinner("Génération des explications SHAP..."):
                    try:
                        explanation = fetch_local_shap_explanation(selected_id)

                        fig = plot_waterfall_chart_expandable(explanation)
                        # fig = plot_waterfall_chart(explanation)
                        # st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Erreur technique : {str(e)}")
                st.markdown("</div>", unsafe_allow_html=True)

                # Histogrammes SHAP comparatifs
                st.markdown('<div class="feature-card">', unsafe_allow_html=True)

                # 3. Nouvelle section pour les histogrammes (en réutilisant les données)
                st.markdown("### 🔍 Analyse de variables influentes")

                # Séparer les top 15 positives et négatives (SHAP values)
                shap_values_df = pd.DataFrame(
                    {
                        "feature": explanation["features"],
                        "shap_value": explanation["values"],
                        "feature_value": list(explanation["client_data"].values()),
                        "SK_ID_CURR": selected_id,
                    }
                ).sort_values(by="shap_value", ascending=False)

                top_15_positive = shap_values_df.head(15)
                top_15_negative = shap_values_df.tail(15).sort_values(by="shap_value")

                # Sélection des features
                selected_pos_feature = st.selectbox(
                    "📈 Variable augmentant le risque (Top 15)",
                    top_15_positive["feature"],
                    key="pos_feature",
                )

                selected_neg_feature = st.selectbox(
                    "📉 Variable réduisant le risque (Top 15)",
                    top_15_negative["feature"],
                    key="neg_feature",
                )

                # Préparation des données
                shap_df = pd.DataFrame(
                    {
                        "feature": explanation["features"],
                        "shap_value": explanation["values"],
                        "feature_value": list(explanation["client_data"].values()),
                        "SK_ID_CURR": selected_id,
                    }
                )

                # Sélection des features les plus importantes
                st.markdown("#### Distribution des variables influentes")
                # client_info = fetch_client_info(selected_id)
                dist1 = plot_feature_distribution(
                    feature_name=selected_pos_feature,
                    full_data=df_test,
                    client_data=client_info,
                    base_color="crimson",
                    client_bin_color="yellow",
                    title_prefix="📈 Risque ↑",
                )

                dist2 = plot_feature_distribution(
                    feature_name=selected_neg_feature,
                    full_data=df_test,
                    client_data=client_info,
                    base_color="seagreen",
                    client_bin_color="yellow",
                    title_prefix="📉 Risque ↓",
                )

                st.plotly_chart(dist1, use_container_width=True)
                st.plotly_chart(dist2, use_container_width=True)

        except Exception as e:
            st.error(f"Erreur d'affichage : {str(e)}")
    else:
        show_risk_gauge(None, client_id=selected_id)
