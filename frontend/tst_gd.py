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

# from utils.shap_utils
from utils.api_requests import (
    check_api_health,
    fetch_client_ids,
    fetch_client_info,
    fetch_prediction,
    fetch_population_stats,
)
from utils.shap_utils import (
    fetch_client_data_for_shap,
    fetch_client_shap_data,
    fetch_global_shap_matrix,
    fetch_local_shap_explanation,
)
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder, GridUpdateMode

from utils.user_interactions import (
    prepare_client_info,
    create_interactive_grid,
    build_feature_config,
    get_genre_client,
    get_updated_data,
    process_selection_and_display_plot,
)
from utils.visuals import (
    plot_boxplot_comparison,
    plot_waterfall_chart,
    plot_waterfall_chart_expandable,
    plot_summary_chart,
    get_title_font_size,
    plot_feature_distribution,
    plot_shap_by_decision,
    plot_shap_histogram,
    restore_discrete_types,
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
df_test_raw = load_test_data_from_api()
df_test = restore_discrete_types(df_test_raw, max_cardinality=15, verbose=True)

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


# ======================================================================================


# ======================================================================================

# Chargement des données via l'API
model, scaler, features, explainer, global_shap_values, df_test_sample = (
    load_model_artifacts()
)

client_ids = fetch_client_ids()
if not client_ids:
    st.error("Aucun client disponible - Vérifiez la connexion à l'API")
    st.stop()

# --- Définition des tabs ---
tab1, tab2, tab3 = st.tabs(["Prédiction", "Exploration", "Recomandation"])

# --- Sidebar ---
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

# === Initialisation de session state ===
for key, value in required_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ===== Réinitialisation lors du changement d'ID =====
if st.session_state.previous_id != selected_id:
    for key in ["predicted", "client_data", "score_float", "show_shap"]:
        st.session_state[key] = required_states[key]

    # ======= Réinitialiser l'état d'animation
    if "current_animated_id" in st.session_state:
        del st.session_state.current_animated_id
    st.session_state.previous_id = selected_id

# Charger automatiquement le premier client au démarrage
if st.session_state.client_row is None:
    # on prend le selected_id (qui vaut déjà le premier élément de la selectbox)
    client_info = fetch_client_info(selected_id)
    if client_info:
        client_row = pd.DataFrame([client_info])
        st.session_state.client_row = client_row
        st.session_state.client_data = client_row.drop(columns=["SK_ID_CURR"]).to_dict(
            orient="records"
        )[0]
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

# ===== Soumission prédiction =====

# --- Tab 1 - Prédiction ---
with tab1:
    st.header("Prédiction")

    # --- Container 1 - Grid et prédiction ---
    with st.container():
        col_left, col_right = st.columns(2)

        with col_left:
            # 1) Récupérer le client (déjà stocké dans st.session_state.client_row)
            client_info = fetch_client_info(selected_id)
            if client_info is None:
                st.error("Impossible de charger les infos client")
                st.stop()

            # 2) Préparer ou réutiliser la grille selon l'ID
            if st.session_state.get("grid_for_id") != selected_id:
                # Nouvel ID → (re)prépare la grille
                df_infos = prepare_client_info(st.session_state.client_row.iloc[0])
                st.session_state.df_infos = df_infos
                st.session_state.grid_for_id = selected_id
            else:
                # Même ID → réutilise l'ancienne grille
                df_infos = st.session_state.df_infos

            # 3) Afficher la grille (avec key unique)
            updated_df, grid_response = create_interactive_grid(
                df_infos,
                edit=False,
                context=f"prediction_{selected_id}",
                # key=f"grid_{selected_id}",
            )

            # 4) Construire la config des features
            feature_config = build_feature_config()

            # Traitement des sélections
            genre_client = get_genre_client(updated_df)
            updated_data = get_updated_data(updated_df, feature_config)

            # Pour chaque feature affichée par l'utilisateur
            process_selection_and_display_plot(
                grid_response, feature_config, genre_client, API_URL, API_KEY
            )

        with col_right:
            st.subheader("Affichage prédiction")
            if st.session_state.predicted:
                try:
                    if "current_animated_id" not in st.session_state:
                        st.session_state.current_animated_id = None

                    if st.session_state.current_animated_id != selected_id:
                        animate_risk_gauge(
                            score=st.session_state.score_float, client_id=selected_id
                        )
                        st.session_state.current_animated_id = selected_id
                    else:
                        show_risk_gauge(
                            score=st.session_state.score_float, client_id=selected_id
                        )

                    display_risk_message(
                        score=st.session_state.score_float, threshold=THRESHOLD
                    )

                except Exception as e:
                    st.error(f"Erreur affichage prédiction : {str(e)}")
            else:
                show_risk_gauge(None, client_id=selected_id)

    # --- Container 2 - Analyse globale et locale ---
    with st.container():
        col_global, col_local = st.columns(2)

        with col_global:
            with st.expander("📖 Analyse Globale"):
                # --- Analyse SHAP Globale ---
                if st.session_state.predicted and st.session_state.show_shap:
                    st.markdown("---")
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

                            st.markdown(
                                '<div class="feature-card">', unsafe_allow_html=True
                            )
                            st.subheader("📖 Explication du score")
                            with st.spinner("Génération des explications SHAP..."):
                                try:
                                    explanation = fetch_local_shap_explanation(
                                        selected_id
                                    )

                                    fig = plot_waterfall_chart_expandable(explanation)
                                except Exception as e:
                                    st.error(f"Erreur technique : {str(e)}")
                            st.markdown("</div>", unsafe_allow_html=True)
        # 2 ---- Analyse locale ---
        with col_local:
            st.subheader("Analyse locale")
            if st.session_state.show_shap:  # Vérifier uniquement show_shap
                if st.session_state.predicted:
                    with st.spinner("Génération des explications SHAP..."):
                        try:
                            explanation = fetch_local_shap_explanation(selected_id)
                            if explanation:  # Vérifier que les données sont valides
                                plot_waterfall_chart_expandable(explanation)
                            else:
                                st.warning("Données SHAP non disponibles")
                        except Exception as e:
                            st.error(f"Erreur technique : {str(e)}")
                else:
                    st.info("Soumettre la prédiction pour voir l'analyse")
            else:
                st.info(
                    "Cochez 'Afficher l'explication SHAP' pour voir l'analyse locale"
                )

    # --- Container 3 - Graphiques SHAP (distributions) ---
    with st.container():
        if st.session_state.predicted and st.session_state.show_shap:
            # if st.session_state.show_shap:
            st.markdown("---")

            # Déclaration obligatoire avant utilisation
            explanation = fetch_local_shap_explanation(selected_id)

            with st.container():
                col_viz1, col_viz2 = st.columns(2)

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

                selected_pos_feature = col_viz1.selectbox(
                    "📈 Variable augmentant le risque (Top 15)",
                    top_15_positive["feature"],
                    key="pos_feature",
                )

                selected_neg_feature = col_viz2.selectbox(
                    "📉 Variable réduisant le risque (Top 15)",
                    top_15_negative["feature"],
                    key="neg_feature",
                )

            with st.container():
                col_viz1, col_viz2 = st.columns(2)

            client_df = pd.DataFrame([client_info])
            client_df_visu = restore_discrete_types(client_df)
            client_visu = client_df_visu.iloc[0]
            print(client_df["CODE_GENDER"].unique())

            with col_viz1:
                st.subheader("Graphique SHAP 1")
                dist1 = plot_feature_distribution(
                    feature_name=selected_pos_feature,
                    full_data=df_test,
                    client_data=client_visu,
                    base_color="crimson",
                    client_bin_color="yellow",
                    title_prefix="📈 Risque ↑",
                )
                st.plotly_chart(dist1, use_container_width=True)

            with col_viz2:
                st.subheader("Graphique SHAP 2")
                dist2 = plot_feature_distribution(
                    feature_name=selected_neg_feature,
                    full_data=df_test,
                    client_data=client_visu,
                    base_color="seagreen",
                    client_bin_color="yellow",
                    title_prefix="📉 Risque ↓",
                )
                st.plotly_chart(dist2, use_container_width=True)

# --- Tab 2 - Exploration ---
with tab2:
    st.header("Exploration")
    feature_config = build_feature_config()

    # Container 1 - Grid éditable et boxplots
    with st.container():
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Grid éditable")
            if st.session_state.client_row is not None:
                df_infos = prepare_client_info(st.session_state.client_row.iloc[0])
                updated_df, grid_response = create_interactive_grid(
                    df_infos, edit=True, context=f"exploration_{selected_id}"
                )
            else:
                st.warning("Aucune donnée client disponible")
                st.stop()

            # Vérification supplémentaire
            if "updated_df" not in locals():
                st.error("Erreur de chargement des données client")
                st.stop()

            # Traitement des sélections
            genre_client = get_genre_client(updated_df) or 0  # Valeur par défaut
            updated_data = get_updated_data(updated_df, feature_config)

        with col_right:
            st.subheader("Boxplots")
            if genre_client is not None:
                process_selection_and_display_plot(
                    grid_response, feature_config, genre_client, API_URL, API_KEY
                )
            else:
                st.warning("Configuration du genre client non disponible")

# ===== TAB 3 : SIMULATION =====
with tab3:
    st.header("Recommendation")
    # --- Container 1 : Simulation ---
    with st.container():
        col_simul, col_results = st.columns([1, 2])

        with col_simul:
            st.subheader("⚙️ Paramètres de simulation")
            df_editable, _ = create_interactive_grid(df_infos, edit=True)

            if st.button("▶️ Exécuter la simulation"):
                results = run_simulation(df_editable)
                st.session_state.simul_results = results

        with col_results:
            st.subheader("📈 Résultats de simulation")
            if "simul_results" in st.session_state:
                plot_simulation_results(st.session_state.simul_results)
