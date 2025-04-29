# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from utils.visuals import plot_boxplot_comparison, plot_multi_feature_shap_histogram

# from utils.visuals import  plot_shap_histogram_plotly
from utils.formatters import format_currency, format_percentage, format_years, safe_get
from utils.shap_utils import (
    load_model_artifacts,
    fetch_client_data_for_shap,
    load_test_data_from_api,
)

# get_top_positive_negative_features
# from utils.shap_utils import  get_shap_long_dataframe
from risk_gauge import show_risk_gauge, display_risk_message
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.graph_objects as go
import plotly.express as px

# Configuration du thème
st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personnalisé
st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #4a6fa5;
        color: white;
        border-radius: 8px;
    }
    .stSelectbox, .stRadio {
        padding: 8px;
        border-radius: 8px;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .feature-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_metadata():
    """Charge les métadonnées du modèle depuis l'API"""
    try:
        response = requests.get(f"{API_URL}/features", headers={"x-api-key": API_KEY})
        response.raise_for_status()
        return response.json()["features"]
    except Exception as e:
        st.error(f"Erreur de chargement des métadonnées : {str(e)}")
        st.stop()


# Chargement des données via l'API
df_test = load_test_data_from_api()

if df_test.empty:
    st.error("Erreur lors du chargement des données depuis l'API")
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


def display_client_info_grid(client: dict):
    """Affiche les informations client dans une carte interactive"""
    with st.container():
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)

        # Construction du tableau d'infos
        df_infos = pd.DataFrame(
            [
                {
                    "Libellé": "Âge",
                    "Valeur": format_years(client["DAYS_BIRTH"]),
                    "Afficher": False,
                },
                {
                    "Libellé": "Genre",
                    "Valeur": "Homme" if client["CODE_GENDER"] == 1 else "Femme",
                    "Afficher": True,
                },
                {
                    "Libellé": "Type de logement",
                    "Valeur": (
                        "Locataire"
                        if client["NAME_HOUSING_TYPE_RENTED_APARTMENT"] == 1
                        else "Autre"
                    ),
                    "Afficher": False,
                },
                {
                    "Libellé": "Statut marital",
                    "Valeur": (
                        "Marié(e)"
                        if client["NAME_FAMILY_STATUS_MARRIED"] == 1
                        else "Autre"
                    ),
                    "Afficher": False,
                },
                {
                    "Libellé": "Type d'emploi",
                    "Valeur": get_employment_type(client),
                    "Afficher": False,
                },
                {
                    "Libellé": "Stabilité professionnelle",
                    "Valeur": format_percentage(client["DAYS_EMPLOYED_PERC"]),
                    "Afficher": False,
                },
                {
                    "Libellé": "Revenu par personne",
                    "Valeur": format_currency(client["INCOME_PER_PERSON"]),
                    "Afficher": True,
                },
                {
                    "Libellé": "Montant crédit",
                    "Valeur": format_currency(client["AMT_CREDIT"]),
                    "Afficher": False,
                },
                {
                    "Libellé": "Charge crédit",
                    "Valeur": format_percentage(client["INCOME_CREDIT_PERC"]),
                    "Afficher": False,
                },
            ]
        )

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

        st.markdown("</div>", unsafe_allow_html=True)
        return grid_response["data"]


def plot_waterfall_chart(explanation):
    """Crée un waterfall plot avec Plotly"""
    features = explanation["features"]
    values = explanation["values"]
    base_value = explanation["base_value"]

    # Préparation des données
    df = pd.DataFrame({"Feature": features[:10], "Impact": values[:10]}).sort_values(
        "Impact", ascending=True
    )

    fig = go.Figure(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(df),
            x=df["Impact"],
            y=df["Feature"],
            base=base_value,
            decreasing={"marker": {"color": "#4a6fa5"}},
            increasing={"marker": {"color": "#ff7f0e"}},
            totals={"marker": {"color": "#d62728"}},
        )
    )

    fig.update_layout(
        title="Impact des caractéristiques sur le score",
        height=500,
        margin=dict(l=100, r=50, t=80, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# Configuration
API_URL = "http://localhost:8000"
API_KEY = "votre_clé_api"
THRESHOLD = 0.0931515

# Initialisation des features
# features = load_model_metadata()
model, scaler, features, explainer, global_shap_values, df_test_sample = (
    load_model_artifacts()
)

# --- Sidebar ---
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=LOGO", width=150)
    st.title("🔍 Navigation")

    try:
        response = requests.get(f"{API_URL}/client_ids")
        client_ids = response.json().get("client_ids", [])
        selected_id = st.selectbox("Sélectionnez un client", client_ids)
    except Exception as e:
        st.error("Erreur de connexion à l'API")
        st.stop()

    st.markdown("---")
    st.subheader("🔧 Options")
    compare_group = st.radio(
        "Groupe de comparaison",
        ["Population totale", "Clients similaires"],
        help="Sélectionnez le groupe de référence pour les comparaisons",
    )

    if st.button("Analyser", type="primary"):
        st.session_state.analyze_clicked = True

# --- Contenu principal ---
if "analyze_clicked" in st.session_state:
    try:
        # Chargement des données client
        response = requests.get(
            f"{API_URL}/client_info/{selected_id}", headers={"x-api-key": API_KEY}
        )
        client_data = response.json()

        # Calcul du score
        response = requests.post(
            f"{API_URL}/predict",
            json={"data": client_data},
            headers={"x-api-key": API_KEY},
        )
        prediction = response.json()

        client = {
            **client_data,
            "score": prediction["probability"] / 100,
            "decision": prediction["decision"],
        }

        # Layout en colonnes
        col1, col2 = st.columns([1, 2])

        with col1:
            # Carte de score
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.subheader("📊 Score de risque")
            display_risk_message(client["score"], THRESHOLD)
            show_risk_gauge(client["score"], THRESHOLD)

            st.markdown("</div>", unsafe_allow_html=True)

            # Infos client
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📌 Détails client")
            updated_df = display_client_info_grid(client)
            st.markdown("</div>", unsafe_allow_html=True)

            # Analyse comparative
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("� Analyse comparative")

            selected_features = st.multiselect(
                "Sélectionnez les caractéristiques à comparer",
                options=["Âge", "Charge crédit", "Montant crédit", "Charge crédit"],
                default=["Âge", "Charge crédit"],
            )

            for feature in selected_features:
                config = {
                    "Âge": {"api_feature": "DAYS_BIRTH", "unit": "ans"},
                    #'Revenu': {'api_feature': 'AMT_INCOME_TOTAL', 'unit': '€'},
                    "Montant crédit": {"api_feature": "AMT_CREDIT", "unit": "€"},
                    "Charge crédit": {"api_feature": "INCOME_CREDIT_PERC", "unit": "%"},
                }[feature]

                response = requests.post(
                    f"{API_URL}/population_stats",
                    json={
                        "feature": config["api_feature"],
                        "filters": (
                            {"CODE_GENDER": client["CODE_GENDER"]}
                            if compare_group == "Clients similaires"
                            else {}
                        ),
                        "sample_size": 1000,
                    },
                    headers={"x-api-key": API_KEY},
                )

                if response.status_code == 200:
                    stats = response.json()["stats"]
                    plot_boxplot_comparison(
                        population_stats=stats,
                        client_value=client[config["api_feature"]],
                        title=f"Comparaison {feature}",
                        unit=config["unit"],
                    )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Explication SHAP
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.subheader("📖 Explication du score")

            with st.spinner("Génération de l'analyse..."):
                try:
                    response = requests.get(
                        f"{API_URL}/shap/local/{selected_id}",
                        headers={"x-api-key": API_KEY},
                    )
                    explanation = response.json()
                    fig = plot_waterfall_chart(explanation)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Erreur technique : {str(e)}")

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

                def plot_feature_distribution(
                    dataframe, client_row, feature_name, color, title
                ):
                    fig = px.histogram(
                        dataframe[feature_name],
                        nbins=30,
                        title=title,
                        opacity=0.7,
                        color_discrete_sequence=[color],
                    )
                    client_value = client_row[feature_name]
                    fig.add_vline(
                        x=client_value,
                        line_width=3,
                        line_dash="dash",
                        line_color="yellow",
                        annotation_text="Client",
                        annotation_position="top",
                    )
                    fig.update_layout(height=300)
                    return fig

                dist1 = plot_feature_distribution(
                    df_test,
                    client_data,
                    selected_pos_feature,
                    "crimson",
                    f"Distribution de {selected_pos_feature}",
                )
                dist2 = plot_feature_distribution(
                    df_test,
                    client_data,
                    selected_neg_feature,
                    "seagreen",
                    f"Distribution de {selected_neg_feature}",
                )

                st.plotly_chart(dist1, use_container_width=True)
                st.plotly_chart(dist2, use_container_width=True)

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {str(e)}")
else:
    # Page d'accueil
    st.title("🏦 Dashboard Scoring Crédit")
    st.markdown(
        """
    <div style="background-color:#4a6fa5;padding:20px;border-radius:10px;color:white">
    <h2 style="color:white">Bienvenue sur le dashboard d'analyse de risque crédit</h2>
    <p>Sélectionnez un client dans la barre latérale pour commencer l'analyse</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ------------------ Main App UI ------------------ #
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Threat Model", "Attack Tree", "Mitigations", "DREAD", "Test Cases"]
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clients dans la base", len(client_ids))
    with col2:
        st.metric("Seuil de risque", f"{THRESHOLD*100:.1f}%")
    with col3:
        st.metric("Version API", "1.0.0")
