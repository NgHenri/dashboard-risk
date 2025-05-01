# === Imports standards de Python ===
import os
import json
import logging
import warnings

# === Environnement ===
from dotenv import load_dotenv

# === Bibliothèques externes ===
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import requests
import joblib
from matplotlib import pyplot as plt

# === Modules internes ===
import config
from risk_gauge import show_risk_gauge, display_risk_message, animate_risk_gauge

# === Composants externes spécifiques ===
from st_aggrid import AgGrid, JsCode, GridOptionsBuilder, GridUpdateMode

# === Utils : formatage & style ===
from utils.formatters import (
    safe_get,
    format_currency,
    format_percentage,
    format_gender,
    format_years,
)
from utils.styling import style_rules, build_dynamic_styling

# === Utils : interactions avec l'API ===
from utils.api_requests import (
    check_api_health,
    fetch_client_ids,
    fetch_client_info,
    fetch_prediction,
    fetch_population_stats,
)

# === Utils : SHAP & interprétabilité ===
from utils.shap_utils import (
    fetch_client_data_for_shap,
    fetch_client_shap_data,
    fetch_global_shap_matrix,
    fetch_local_shap_explanation,
    fetch_batch_predictions,
)

# === Utils : interactions utilisateur ===
from utils.user_interactions import (
    prepare_client_info,
    create_interactive_grid,
    create_interactive_valeur_editor,
    build_feature_config,
    get_genre_client,
    get_updated_data,
    process_selection_and_display_plot,
)

# === Utils : visualisations ===
from utils.visuals import (
    plot_boxplot_comparison,
    plot_waterfall_chart_expandable,
    plot_summary_chart,
    get_title_font_size,
    plot_feature_distribution,
    restore_discrete_types,
    plot_feature_comparison,
    plot_client_position_in_group,
)

# ===== Paramètres de configuration =====

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
# load_dotenv()

API_URL = os.getenv("API_URL")
# print("🛠️ API_URL:", API_URL)
API_KEY = os.getenv("API_KEY")


# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=FutureWarning)

# API_URL = "http://localhost:8000"  # URL de l'API backend
# API_KEY = "b678481b982dc71ab46e08255faefae5f73339c4f1339eec83edf10488502158"
ARTIFACT_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"  # Chemin vers l'artefact du modèle
THRESHOLD = 0.0931515  # Seuil de risque
TIMEOUT = 10  # Timeout pour les requêtes API (en secondes)
BATCH_SIZE = 200  # Taille des lots pour les prédictions

# ===== Vérification de la connexion API =====
# Fonction de vérification
# ===== Vérification de la connexion API =====
check_result = check_api_health(TIMEOUT)
if check_result is not True:
    st.error(check_result)
    st.stop()

st.set_page_config(layout="wide")
st.title("🏦 Dashboard Crédit - Prédictions & Explicabilité")


# ===== Chargement des données =====
# Fonction pour récupérer les données de test depuis l'API
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


# Chargement des données depuis l'API
df_test_raw = load_test_data_from_api()
df_test = restore_discrete_types(df_test_raw, max_cardinality=15, verbose=True)

# Vérification si les données sont vides après chargement
if df_test.empty:
    st.error("Erreur lors du chargement des données depuis l'API")
    st.stop()


# ===== Chargement des artefacts du modèle =====
# Fonction pour charger le modèle et les artefacts associés (scaler, explainer, etc.)
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

    # Calcul des SHAP values pour l'échantillon de test
    df_test_sample_scaled = scaler.transform(df_test_sample[features])
    global_shap_values = explainer.shap_values(df_test_sample_scaled)

    return model, scaler, features, explainer, global_shap_values, df_test_sample


# ===== Chargement des artefacts du modèle =====
model, scaler, features, explainer, global_shap_values, df_test_sample = (
    load_model_artifacts()
)

# =============================================================================
# 📦 Initialisation et vérifications
# =============================================================================

# Récupération des IDs clients
client_ids = fetch_client_ids()
if not client_ids:
    st.error("Aucun client disponible - Vérifiez la connexion à l'API")
    st.stop()

# =============================================================================
# 📑 Interface principale (Tabs)
# =============================================================================
tab1, tab2, tab3 = st.tabs(["Prédiction", "Exploration", "Analyse & Recommandation"])

# =============================================================================
# 🧭 Barre latérale (sidebar)
# =============================================================================
st.sidebar.markdown("## 🔍 Analyse d'un client")

# Sélection du client
client_ids = sorted(client_ids)
selected_id = st.sidebar.selectbox("Sélectionner un client", client_ids)
if not selected_id or not isinstance(selected_id, int):
    st.warning("Veuillez sélectionner un client valide")
    st.stop()

# Bouton de soumission
submitted = st.sidebar.button("Soumettre la prédiction")

# Gestion de la checkbox SHAP via session state
st.session_state.show_shap = st.sidebar.checkbox(
    "Afficher l'explication SHAP",
    value=st.session_state.get(
        "show_shap", False
    ),  # Utiliser .get() avec valeur par défaut
)

# Options supplémentaires
st.markdown("---")
st.sidebar.markdown("## 🔧 Options")
compare_group = st.sidebar.radio(
    "Groupe de comparaison",
    ["Population totale", "Clients similaires"],
    help="Sélectionnez le groupe de référence pour les comparaisons",
)

# Initialisation de session state (valeurs par défaut)
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

# =============================================================================
# 🔁 Réinitialisation des données lors d’un changement de client
# =============================================================================
if st.session_state.previous_id != selected_id:
    for key in ["predicted", "client_data", "score_float", "show_shap"]:
        st.session_state[key] = required_states[key]

    # Réinitialiser l'état d'animation s'il existe
    if "current_animated_id" in st.session_state:
        del st.session_state.current_animated_id

    st.session_state.previous_id = selected_id
    # IMPORTANT : Réinitialiser la grid pour charger les nouvelles infos client
    st.session_state.client_row = None

# Charger automatiquement le premier client au démarrage ou en cas de changement
if st.session_state.client_row is None:
    client_info = fetch_client_info(selected_id)
    if client_info:
        client_row = pd.DataFrame([client_info])
        st.session_state.client_row = client_row
        st.session_state.client_data = client_row.drop(columns=["SK_ID_CURR"]).to_dict(
            orient="records"
        )[0]

# =============================================================================
# 🔮 Soumission de la prédiction
# =============================================================================

if submitted:
    client_info = fetch_client_info(selected_id)
    if not client_info:
        st.error("Données client non disponibles")
        st.stop()

    # Conversion en DataFrame puis mise à jour dans le session state
    client_row = pd.DataFrame([client_info])
    st.session_state.client_row = client_row
    st.session_state.client_data = client_row.drop(columns=["SK_ID_CURR"]).to_dict(
        orient="records"
    )[0]
    # ===
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

# with st.sidebar.expander("IDs des clientes", expanded=True):
#    female_ids = (
#        df_test[df_test["CODE_GENDER"] == 0]["SK_ID_CURR"].astype(int).sort_values()
#    )
#    st.write(female_ids.tolist())

# =============================================================================
# 📊 Onglet 1 : Prédiction
# =============================================================================
with tab1:
    st.header("Info Client")

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
            )

            # 4) Construire la config des features
            feature_config = build_feature_config()

            # Traitement des sélections
            genre_client = get_genre_client(updated_df)
            updated_data = get_updated_data(updated_df, feature_config)

            # Pour chaque feature affichée par l'utilisateur
            process_selection_and_display_plot(
                grid_response,
                feature_config,
                genre_client,
                compare_group,
                API_URL,
                API_KEY,
            )
        # ─────────────── 📈 Partie droite : Affichage du score + jauge ───────────────
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

    # ─────────────── 🔍 Partie inférieure : Analyse globale et locale ───────────────
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
        # ===========================
        # --- ANALYSE LOCALE (SHAP)
        # ===========================
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
    # ====================================================
    # --- ANALYSE SHAP - Visualisation des distributions
    # ====================================================
    # --- Container 3  ---
    with st.container():
        if st.session_state.predicted and st.session_state.show_shap:
            # if st.session_state.show_shap:
            st.markdown("---")

            # Récupération des valeurs SHAP
            explanation = fetch_local_shap_explanation(selected_id)

            with st.container():
                col_viz1, col_viz2 = st.columns(2)

                # Construction du DataFrame SHAP
                shap_values_df = pd.DataFrame(
                    {
                        "feature": explanation["features"],
                        "shap_value": explanation["values"],
                        "feature_value": list(explanation["client_data"].values()),
                        "SK_ID_CURR": selected_id,
                    }
                ).sort_values(by="shap_value", ascending=False)

                # Top 15 features contribuant positivement et négativement
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

            # Chargement des données client
            client_df = pd.DataFrame([client_info])
            client_df_visu = restore_discrete_types(client_df)
            client_visu = client_df_visu.iloc[0]

            # Affichage des graphiques de distribution SHAP
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

# ========================
# --- TAB 2 : EXPLORATION
# ========================
with tab2:
    st.header("Exploration")
    feature_config = build_feature_config()

    # Container 1 - Grid éditable et boxplots
    with st.container():
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Info Client")
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
            st.subheader("Illustration")
            if genre_client is not None:
                process_selection_and_display_plot(
                    grid_response,
                    feature_config,
                    genre_client,
                    compare_group,
                    API_URL,
                    API_KEY,
                    show_empty_info=True,
                )
            else:
                st.warning("Configuration du genre client non disponible")


# ===================================
# --- FUNCTION : BATCH SIMULATION
# ===================================
@st.cache_data(show_spinner="Chargement des prédictions batch...", ttl=600)
def simulate_batch(
    df_clients: pd.DataFrame, filter_decision: str = None
) -> pd.DataFrame:
    res = fetch_batch_predictions(df_clients, filter_decision)
    res["SK_ID_CURR"] = res["SK_ID_CURR"].astype(int)
    return res


# ========================
# --- TAB 3 : SIMULATION
# ========================
with tab3:
    st.header("📈 Analyse & Recommandation")
    # 1) Edit client dataset
    st.subheader("🔍 Éditer info Client")
    df = df_test.copy()
    tab_analyse, tab_recommandation = st.tabs(["Analyse", "Recommandation"])

    # --- ANALYSE GUIDÉE / LIBRE ---
    with tab_analyse:
        st.header("Analyse exploratoire")
        sub_guided, sub_custom = st.tabs(["1️⃣ Guidée", "2️⃣ Personnalisée"])
        with sub_guided:
            FEATURE_PAIRS = {
                "Stabilité vs Capacité remb.": (
                    "DAYS_EMPLOYED_PERC",
                    "INCOME_CREDIT_PERC",
                ),
                "Ancienneté vs Montant crédit": (
                    "DAYS_EMPLOYED_PERC",
                    "AMT_CREDIT",
                ),
                "Montant crédit vs Charge mens.": (
                    "AMT_CREDIT",
                    "ANNUITY_INCOME_PERC",
                ),
                "Montant crédit vs Capacité remb.": (
                    "AMT_CREDIT",
                    "INCOME_CREDIT_PERC",
                ),
                "Retards crédits passés vs Capacité remb.": (
                    "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN",
                    "INCOME_CREDIT_PERC",
                ),
                "Autres prêts vs Montant crédit": (
                    "BURO_CREDIT_TYPE_ANOTHER_TYPE_OF_LOAN_MEAN",
                    "AMT_CREDIT",
                ),
            }

            choice = st.selectbox("Paire de variables", list(FEATURE_PAIRS.keys()))
            fx, fy = FEATURE_PAIRS[choice]
            ptype = st.radio(
                "Type graphique",
                ["scatter", "histogram", "density", "histogram2d"],
                horizontal=True,
            )
            plot_feature_comparison(df, fx, fy, ptype)
        with sub_custom:
            numeric = df.select_dtypes("number").columns.tolist()
            cx = st.selectbox("Variable X", numeric)
            cy = st.selectbox("Variable Y", numeric)
            ctype = st.radio(
                "Type graphique", ["scatter", "histogram", "density"], horizontal=True
            )
            plot_feature_comparison(df, cx, cy, ctype)

    # --- RECOMMANDATION & SIMULATION BATCH ---
    with tab_recommandation:
        df_info = prepare_client_info(st.session_state.client_row.iloc[0])
        st.header("Recommandation et simulation batch")

        # Étape 1 : Édition des infos
        st.subheader("1️⃣ Éditer le dataset du client")
        df_editable, grid_opts = create_interactive_valeur_editor(
            df_info, edit=True, context=f"editor_{selected_id}"
        )

        # Étape 2 : Paramètres de simulation
        st.subheader("2️⃣ Paramètres de simulation")
        decision_filter = st.selectbox(
            "Filtrer par décision initiale:", ["Tous", "✅ Accepté", "❌ Refusé"]
        )
        filter_val = None if decision_filter == "Tous" else decision_filter

        if st.button("🔮 Lancer la simulation batch", key="run_tab3"):
            # -- Récupère la ligne du client sélectionné depuis le df d'origine
            selected_client_row = df[df["SK_ID_CURR"] == selected_id]

            # -- Échantillonne le reste du batch (hors client sélectionné)
            df_remaining = df[df["SK_ID_CURR"] != selected_id]
            sample_df = df_remaining.sample(BATCH_SIZE - 1, random_state=42)

            # -- Ajoute le client sélectionné en première ligne
            sample_df = pd.concat([selected_client_row, sample_df], ignore_index=True)

            # -- Simulation
            df_results = simulate_batch(sample_df, filter_val)

            # -- Enregistre résultats + ID sélectionné
            st.session_state["simul_results"] = df_results
            st.session_state["default_focus_id"] = selected_id
            st.success("Simulation terminée !")

        # Étape 3 : Affichage résultats
        st.subheader("3️⃣ Résultats de la simulation")
        if "simul_results" in st.session_state:
            res_df = st.session_state["simul_results"]
            default_id = st.session_state.get(
                "default_focus_id", res_df["SK_ID_CURR"].iloc[0]
            )
            client_choice = st.selectbox(
                "Client pour focus:",
                res_df["SK_ID_CURR"].tolist(),
                key="focus_tab3",
                index=res_df["SK_ID_CURR"].tolist().index(default_id),
            )

            st.dataframe(res_df[["SK_ID_CURR", "probability", "decision"]])
            plot_client_position_in_group(res_df, client_choice)
        else:
            st.info("Aucune simulation exécutée. Cliquez sur le bouton ci-dessus.")
