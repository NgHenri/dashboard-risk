# frontend/utils/user_interactions.py
import pandas as pd
import streamlit as st
from utils.formatters import (
    safe_get,
    format_currency,
    format_percentage,
    format_gender,
    format_years,
    parse_currency,
)

# from dotenv import load_dotenv
import os

from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from config import API_URL, API_KEY, TIMEOUT


# ========== Paramètres globaux ==========

# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Charger les variables d'environnement

# API_URL = "https://dashboard-risk.onrender.com"

# API_KEY = os.getenv("API_KEY")
# ARTIFACT_PATH = os.getenv("ARTIFACT_PATH")
# THRESHOLD = float(os.getenv("THRESHOLD"))  # THRESHOLD doit être casté en float
# COST_FN = int(os.getenv("COST_FN"))  # idem
# COST_FP = int(os.getenv("COST_FP"))
# GLOBAL_DATA_PATH = os.getenv("GLOBAL_DATA_PATH")
# TIMEOUT = 10  # Timeout pour les requêtes


def prepare_client_info(row):
    """
    Prépare le DataFrame avec les informations formatées pour un client donné.
    """

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

    # Convertir en DataFrame
    df_infos = pd.DataFrame(list(infos.items()), columns=["Libellé", "Valeur"])
    df_infos["Valeur"] = df_infos["Valeur"].astype(str)  # 🔥 force explicite en string
    df_infos["Afficher"] = False  # Colonne pour checkboxes

    return df_infos


# Configuration de la grille avec AgGrid


def create_interactive_grid(df, edit=True, grid_height=450, context=""):
    """
    Crée une grille interactive avec cases à cocher désactivées pour certaines lignes
    """
    from st_aggrid import JsCode

    # Liste des libellés à désactiver
    # non_editable_labels = ["ID Client", "Type de logement", "Type d'emploi"]
    non_editable_labels = ["ID Client"]

    # Configuration conditionnelle des cellules
    checkbox_disabler = JsCode(
        f"""
    function(params) {{
        const libelle = params.data['Libellé'];
        const disabledLabels = {non_editable_labels};
        
        if (disabledLabels.includes(libelle)) {{
            return {{
                'disabled': true,
                'backgroundColor': '#7697ad',
                'color': '#d3d3d3',
                'cursor': 'not-allowed',
                'pointerEvents': 'none'
            }};
        }}
        return null;
    }}
    """
    )

    # Configuration du GridOptionsBuilder
    gb = GridOptionsBuilder.from_dataframe(df)

    # Configuration colonne "Afficher"
    gb.configure_column(
        "Afficher",
        editable=edit,
        cellStyle=checkbox_disabler,
        cellEditor="agCheckboxCellEditor",
        cellRenderer="agCheckboxCellRenderer",
    )

    # Configuration colonne "Libellé"
    gb.configure_column(
        "Libellé",
        cellStyle=JsCode(
            """
        function(params) {
            const disabledLabels = %s;
            if (disabledLabels.includes(params.value)) {
                return {'color': '#7697ad'};
            }
            return null;
        }
        """
            % non_editable_labels
        ),
        editable=False,
    )

    # Configuration globale
    gb.configure_grid_options(domLayout="normal", suppressRowClickSelection=True)

    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        height=grid_height,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        allow_unsafe_jscode=True,
        key=f"aggrid_{context}",
    )

    return grid_response["data"], grid_response


# ==============


def create_interactive_valeur_editor(df, edit=True, grid_height=450, context=""):
    """
    Crée une grille interactive permettant d'éditer la colonne 'Valeur',
    sauf pour certaines lignes définies via la colonne 'Libellé'.
    La colonne 'Afficher' est supprimée.
    """
    # Libellés pour lesquels l'édition est désactivée
    non_editable_labels = ["ID Client"]

    # Style pour désactiver visuellement la cellule
    valeur_cell_style = JsCode(
        f"""
        function(params) {{
            const libelle = params.data['Libellé'];
            const disabledLabels = {non_editable_labels};
            if (disabledLabels.includes(libelle)) {{
                return {{
                    'backgroundColor': '#7697ad',
                    'color': '#d3d3d3',
                    'cursor': 'not-allowed',
                    'pointerEvents': 'none'
                }};
            }}
            return null;
        }}
        """
    )

    # Rendre la cellule éditable ou non selon le libellé
    valeur_cell_editable = JsCode(
        f"""
        function(params) {{
            const libelle = params.data['Libellé'];
            const disabledLabels = {non_editable_labels};
            return !disabledLabels.includes(libelle);
        }}
        """
    )

    # Configuration AgGrid
    gb = GridOptionsBuilder.from_dataframe(df)

    # Colonne 'Valeur'
    gb.configure_column(
        "Valeur",
        editable=valeur_cell_editable if edit else False,
        cellStyle=valeur_cell_style,
    )

    # Colonne 'Libellé' non éditable
    gb.configure_column(
        "Libellé",
        editable=False,
        cellStyle=JsCode(
            f"""
            function(params) {{
                const disabledLabels = {non_editable_labels};
                if (disabledLabels.includes(params.value)) {{
                    return {{'color': '#7697ad'}};
                }}
                return null;
            }}
            """
        ),
    )

    # Options générales
    gb.configure_grid_options(domLayout="normal", suppressRowClickSelection=True)

    # Affichage de la grille
    grid_response = AgGrid(
        df,
        gridOptions=gb.build(),
        height=grid_height,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=True,
        theme="streamlit",
        allow_unsafe_jscode=True,
        key=f"aggrid_valeur_{context}",
    )

    return grid_response["data"], grid_response
    # ========================


def build_feature_config():
    """
    Construit la configuration des features utilisées dans l'application.
    """
    return {
        "Âge": {
            "api_feature": "DAYS_BIRTH",
            "parse_func": lambda v: -float(v.split()[0]) * 365,
            "transform_func": lambda x: -x / 365,
            "unit": "ans",
        },
        "Revenu par personne": {
            "api_feature": "INCOME_PER_PERSON",
            # "parse_func": lambda v: float(v.replace("€", "").replace(" ", "").replace(",", "")),
            "parse_func": lambda v: parse_currency(v),
            "transform_func": None,
            "unit": "€",
        },
        "Stabilité professionnelle": {
            "api_feature": "DAYS_EMPLOYED_PERC",
            "parse_func": lambda v: float(v.replace("%", "").replace(",", ".")) / 100,
            "transform_func": lambda x: x * 100,
            "unit": "%",
        },
        "Genre": {
            "api_feature": "CODE_GENDER",
            "parse_func": lambda v: 1 if v == "Homme" else 0,
            "transform_func": None,  # <-- Explicitement
            "unit": "",
        },
        "Montant du crédit demandé": {
            "api_feature": "AMT_CREDIT",
            "parse_func": lambda v: float(
                v.replace("€", "")
                .replace(" ", "")
                .replace(",", ".")
                .replace("\u202f", "")  # Gestion des espaces insécables
            ),
            "transform_func": None,
            "unit": "€",
        },
        "Charge crédit (revenu vs crédit)": {
            "api_feature": "INCOME_CREDIT_PERC",
            "parse_func": lambda v: float(v.replace("%", "").replace(",", ".").strip())
            / 100,
            "transform_func": lambda x: x * 100,
            "unit": "%",
        },
        "Poids des remboursements sur le revenu": {
            "api_feature": "ANNUITY_INCOME_PERC",
            "parse_func": lambda v: float(v.replace("%", "").replace(",", ".").strip())
            / 100,
            "transform_func": lambda x: x * 100,
            "unit": "%",
        },
        "Historique crédit (ancienneté moyenne)": {
            "api_feature": "BURO_DAYS_CREDIT_MEAN",
            "parse_func": lambda v: -int(v.split(" ")[0].strip())
            * 365,  # Convertit "5 ans" -> -1825
            "transform_func": lambda x: -x / 365,  # Convertit -1825 -> 5.0
            "unit": "ans",
        },
        "Dernier retard de paiement": {
            "api_feature": "INSTAL_DBD_MAX",
            "parse_func": lambda v: (int(v.split()[0]) if "jours" in v else 0),
            "transform_func": None,
            "unit": "jours",
        },
        "Type de logement": {
            "api_feature": "NAME_HOUSING_TYPE_RENTED_APARTMENT",
            "parse_func": lambda v: 1 if v == "Locataire" else 0,
            "transform_func": None,
            "unit": "",
        },
        "Statut marital": {
            "api_feature": "NAME_FAMILY_STATUS_MARRIED",
            "parse_func": lambda v: 1 if v == "Marié(e)" else 0,
            "transform_func": None,
            "unit": "",
        },
    }


# ==============================
def get_genre_client(updated_df):
    """
    Récupère le genre du client à partir du DataFrame mis à jour.
    """
    try:
        genre = updated_df.loc[updated_df["Libellé"] == "Genre", "Valeur"].values[0]
        return 1 if genre == "Homme" else 0  # Conversion en format attendu par l'API
    except IndexError:
        return None


# =============================
def get_updated_data(updated_df, feature_config):
    """
    Récupère les données mises à jour dans la grille interactive.
    """
    updated_data = {}
    for _, row in updated_df.iterrows():
        if row["Afficher"]:
            label = row["Libellé"]
            valeur = row["Valeur"]

            if label in feature_config:
                config = feature_config[label]
                parse_func = config["parse_func"]
                if parse_func:
                    updated_data[label] = parse_func(valeur)

    return updated_data


from utils.api_requests import fetch_population_stats
from utils.visuals import plot_boxplot_comparison


def process_selection_and_display_plot(
    grid_response,
    feature_config,
    genre_client,
    compare_group,
    api_url,
    api_key,
    cols_per_row=1,
    show_empty_info=False,
):
    """
    Traite la sélection de ligne, appelle l'API pour obtenir les statistiques de population et affiche les boxplots en grille.
    """
    updated_df = grid_response["data"]
    selected_rows = [row for _, row in updated_df.iterrows() if row["Afficher"]]

    if not selected_rows:
        if show_empty_info:
            st.info("Aucune variable sélectionnée pour affichage.")
        return

    for i in range(0, len(selected_rows), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, row in enumerate(selected_rows[i : i + cols_per_row]):
            with cols[j]:
                label = row["Libellé"]
                valeur = row["Valeur"]

                config = feature_config.get(label, {})
                if not config:
                    st.warning(f"Configuration manquante pour {label}")
                    continue

                try:
                    # Construction des filtres
                    if (
                        compare_group == "Clients similaires"
                        and genre_client is not None
                    ):
                        filters = {"CODE_GENDER": genre_client}

                        for col in [
                            "NAME_INCOME_TYPE_PENSIONER",
                            "NAME_INCOME_TYPE_WORKING",
                            "NAME_INCOME_TYPE_STUDENT",
                            "ORGANIZATION_TYPE_SELF_EMPLOYED",
                            "ORGANIZATION_TYPE_MILITARY",
                        ]:
                            if safe_get(row, col) == 1:
                                filters[col] = 1

                        if row.get("Type de logement") == "Locataire":
                            filters["NAME_HOUSING_TYPE_RENTED_APARTMENT"] = 1
                        if row.get("Statut marital") == "Marié(e)":
                            filters["NAME_FAMILY_STATUS_MARRIED"] = 1
                    else:
                        filters = {}

                    client_value = config["parse_func"](valeur)

                    stats = fetch_population_stats(
                        feature=config["api_feature"],
                        filters=filters,
                        api_url=api_url,
                        api_key=api_key,
                        sample_size=1000,
                    )

                    if stats:
                        plot_boxplot_comparison(
                            population_stats=stats,
                            client_value=client_value,
                            title=f"Position du client - {label}",
                            unit=config["unit"],
                            transform=config["transform_func"],
                            height=400,
                        )

                except Exception as e:
                    st.error(f"Erreur avec {label} : {str(e)}")
