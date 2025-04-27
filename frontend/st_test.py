# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from utils.visuals import plot_boxplot_comparison
from utils.formatters import format_currency, format_percentage, format_years
from risk_gauge import show_risk_gauge, display_risk_message, animate_risk_gauge
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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
def display_client_info_grid(client: dict, api_url: str, api_key: str):
    # --- 1. Construction du tableau d'infos ---
    df_infos = pd.DataFrame([
        {"Libellé": "Âge", "Valeur": format_years(client['DAYS_BIRTH']), "Afficher": True},
        {"Libellé": "Charge crédit", "Valeur": format_percentage(client['INCOME_CREDIT_PERC']), "Afficher": True},
        {"Libellé": "Montant crédit", "Valeur": format_currency(client['AMT_CREDIT']), "Afficher": True},
        {"Libellé": "Revenu par personne", "Valeur": format_currency(client['INCOME_PER_PERSON']), "Afficher": False},
        {"Libellé": "Stabilité professionnelle", "Valeur": f"{100 * client['DAYS_EMPLOYED_PERC']:.1f}%", "Afficher": False},
        {"Libellé": "Genre", "Valeur": "Homme" if client["CODE_GENDER"] == 1 else "Femme", "Afficher": False}
    ])

    # --- 2. Affichage interactif ---
    st.subheader("📌 Caractéristiques clés")
    gb = GridOptionsBuilder.from_dataframe(df_infos)
    gb.configure_column("Afficher", editable=True)
    gb.configure_grid_options(domLayout='normal')
    grid_response = AgGrid(
        df_infos,
        gridOptions=gb.build(),
        height=300,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True
    )
    updated_df = grid_response['data']

    # --- 3. Config pour l'appel API ---
    feature_config = {
        "Âge": {
            "api_feature": "DAYS_BIRTH",
            "parse_func": lambda v: -float(v.split()[0]) * 365,
            "transform_func": lambda x: -x / 365,
            "unit": "ans"
        },
        "Revenu par personne": {
            "api_feature": "INCOME_PER_PERSON",
            "parse_func": lambda v: float(v.replace('€', '').replace(' ', '').replace(',', '')),
            "transform_func": None,
            "unit": "€"
        },
        "Stabilité professionnelle": {
            "api_feature": "DAYS_EMPLOYED_PERC",
            "parse_func": lambda v: float(v.replace('%', '').replace(',', '.')) / 100,
            "transform_func": lambda x: x * 100,
            "unit": "%"
        }
    }

    genre_client = updated_df.loc[updated_df["Libellé"] == "Genre", "Valeur"].values[0] if "Genre" in updated_df["Libellé"].values else None

    # --- 4. Affichage dynamique des stats ---
    for _, row in updated_df.iterrows():
        if row["Afficher"]:
            label = row["Libellé"]
            valeur = row["Valeur"]
            
            feature_config = {
                "Âge": {
                    "api_feature": "DAYS_BIRTH",
                    "parse_func": lambda v: -float(v.split()[0]) * 365,
                    "transform_func": lambda x: -x / 365,
                    "unit": "ans"
                },
                "Revenu par personne": {
                    "api_feature": "INCOME_PER_PERSON",
                    "parse_func": lambda v: float(v.replace('€', '').replace(' ', '').replace(',', '')),
                    "transform_func": None,
                    "unit": "€"
                },
                "Stabilité professionnelle": {
                    "api_feature": "DAYS_EMPLOYED_PERC",
                    "parse_func": lambda v: float(v.replace('%', '').replace(',', '.')) / 100,
                    "transform_func": lambda x: x * 100,
                    "unit": "%"
                }
            }

            config = feature_config.get(label, {})
            if not config:
                st.warning(f"⚠️ Configuration manquante pour {label}")
                continue

            # Filtres si nécessaire
            filters = {}
            if "Genre" in updated_df.loc[updated_df["Afficher"], "Libellé"].values:
                filters["CODE_GENDER"] = 1 if genre_client == "Homme" else 0

            try:
                response = requests.post(
                    f"{API_URL}/population_stats",
                    json={
                        "feature": config["api_feature"],
                        "filters": filters,
                        "sample_size": 1000
                    },
                    headers={"x-api-key": API_KEY}
                )

                if response.status_code == 200:
                    stats = response.json()
                    st.subheader(f"📈 {label} ({config['unit']})")

                    # Affichage valeur client formatée
                    valeur_affichee = valeur
                    if config.get("transform_func"):
                        try:
                            valeur_affichee = round(config["transform_func"](float(valeur)), 1)
                        except:
                            pass
                    st.write(f"Valeur client : **{valeur_affichee} {config['unit']}**")

                    # Affichage des stats
                    if "stats" in stats:
                        df_stats = pd.DataFrame(stats["stats"], index=["Valeur"]).T
                        df_stats.reset_index(inplace=True)
                        df_stats.columns = ["Statistique", "Valeur"]

                        if config["unit"] == "ans":
                            df_stats["Valeur"] = df_stats["Valeur"].apply(lambda x: round(-x / 365, 1))
                        elif config["unit"] == "%":
                            df_stats["Valeur"] = df_stats["Valeur"].apply(lambda x: round(x * 100, 2))
                        elif config["unit"] == "€":
                            df_stats["Valeur"] = df_stats["Valeur"].apply(lambda x: f"{x:,.0f} €".replace(",", " ").replace(".0", ""))

                        st.dataframe(df_stats, use_container_width=True)
                    else:
                        st.warning("⚠️ Les statistiques sont absentes de la réponse.")
                else:
                    st.warning(f"⚠️ Erreur API pour {label} : {response.status_code}")
            except Exception as e:
                st.error(f"Erreur API pour {label} : {e}")



# Configuration
API_URL = "http://localhost:8000"
API_KEY = "votre_clé_api"
THRESHOLD = 0.0931515


# Initialisation des features
features = load_model_metadata() 

# --- 1. Gestion de l'état de session ---
if 'selected_id' not in st.session_state:
    st.session_state.selected_id = None
if 'client_data' not in st.session_state:
    st.session_state.client_data = None

# --- 2. Sidebar pour la sélection client ---
with st.sidebar:
    st.header("🔍 Sélection du client")
    
    # Chargement des IDs clients
    try:
        response = requests.get(f"{API_URL}/client_ids")
        client_ids = response.json().get("client_ids", [])
    except Exception as e:
        st.error("Erreur de connexion à l'API")
        st.stop()
    
    selected_id = st.selectbox(
        "Choisir un client",
        client_ids,
        index=0,
        key="client_selector"
    )
    
    # Filtres avancés
    st.subheader("🎚️ Options de comparaison")
    compare_group = st.radio(
        "Groupe de comparaison",
        ["Population totale", "Clients similaires"],
        help="Comparez avec l'ensemble des clients ou un sous-groupe similaire"
    )
    
    submitted = st.button("Analyser ce client")

# --- 3. Vérification santé API ---
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

if not check_api_health():
    st.error("Service indisponible - Veuillez réessayer plus tard")
    st.stop()

# --- 4. Chargement données client ---
if submitted or st.session_state.selected_id != selected_id:
    try:
        # Chargement données brutes
        response = requests.get(
            f"{API_URL}/client_info/{selected_id}",
            headers={"x-api-key": API_KEY}
        )
        client_data = response.json()
        
        # Calcul du score
        response = requests.post(
            f"{API_URL}/predict",
            json={"data": client_data},
            headers={"x-api-key": API_KEY}
        )
        prediction = response.json()
        
        # Mise à jour session
        st.session_state.client_data = {
            **client_data,
            "score": prediction['probability'] / 100,
            "decision": prediction['decision']
        }
        st.session_state.selected_id = selected_id
        
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        st.stop()

# --- 5. Affichage principal ---
if st.session_state.client_data:
    client = st.session_state.client_data
    col1, col2 = st.columns([2, 3])

    with col1:
        # Score de risque
        st.subheader("📊 Score de risque")
        show_risk_gauge(client['score'], THRESHOLD)
        display_risk_message(client['score'], THRESHOLD)

        # Infos clés interactives
        display_client_info_grid(client, api_url=API_URL, api_key=API_KEY)

    with col2:
        # --- Section Analyse comparative ---
        st.subheader("🔎 Analyse détaillée")
        
        # Sélection caractéristiques
        features_to_compare = st.multiselect(
            "Choisir des caractéristiques à comparer",
            options=['Âge', 'Charge crédit', 'Montant crédit', 'Ancienneté emploi'],
            default=['Âge', 'Charge crédit'],
            help="Sélectionnez jusqu'à 4 caractéristiques pour la comparaison"
        )
        
        # Génération des graphiques
        for feature in features_to_compare:
            try:
                # Configuration dynamique
                config = {
                    'Âge': {'api_feature': 'DAYS_BIRTH', 'transform': lambda x: -x/365},
                    'Charge crédit': {'api_feature': 'INCOME_CREDIT_PERC', 'transform': None},
                    'Montant crédit': {'api_feature': 'AMT_CREDIT', 'transform': None},
                    'Ancienneté emploi': {'api_feature': 'DAYS_EMPLOYED', 'transform': lambda x: -x/365}
                }[feature]
                
                # Récupération données
                response = requests.post(
                    f"{API_URL}/population_stats",
                    json={
                        "feature": config['api_feature'],
                        "filters": {"CODE_GENDER": client['CODE_GENDER']} if compare_group == "Clients similaires" else {},
                        "sample_size": 1000
                    },
                    headers={"x-api-key": API_KEY}
                )
                stats = response.json()["stats"]
                
                # Création visualisation
                plot_boxplot_comparison(
                    population_stats=stats,
                    client_value=client[config['api_feature']],
                    title=f"Comparaison {feature}",
                    unit="ans" if feature == 'Âge' else '€',
                    transform=config['transform']
                )
                
            except Exception as e:
                st.error(f"Erreur d'affichage pour {feature} : {str(e)}")

        
        # --- Section Explication SHAP ---
        st.subheader("📖 Explication du score")
        with st.spinner("Génération de l'explication..."):
            try:
                response = requests.get(
                    f"{API_URL}/shap/local/{selected_id}",
                    headers={"x-api-key": API_KEY}
                )
                if response.status_code != 200:
                    st.error("Erreur lors du calcul des explications")
                    st.stop()

                explanation = response.json()

                # Conversion explicite en np.array
                values = np.array(explanation["values"])
                base_value = np.array([explanation["base_value"]])  # 👈 Wrap dans une liste
                
                # 🔐 Gestion robuste des features manquants
                data = np.array([explanation["client_data"].get(f, np.nan) for f in features])

                # Création de l'explication SHAP
                explainer = shap.Explanation(
                    values=values,
                    base_values=base_value,
                    data=data,
                    feature_names=features
                )

                # Affichage du waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(explainer, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)

            except requests.exceptions.RequestException:
                st.error("Erreur de connexion à l'API")
            except KeyError:
                st.error("Format de réponse API invalide")
            except Exception as e:
                st.error(f"Erreur technique : {str(e)}")
