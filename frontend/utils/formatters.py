# frontend/utils/formatters.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def safe_get(row, col, default="N/A"):
    return row[col] if col in row and not pd.isna(row[col]) else default


def format_currency(value):
    try:
        # Format "28 790,50 €"
        return f"{float(value):,.2f} €".replace(",", " ").replace(".", ",")
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


# fonction de formatage
def parse_client_value(value, label):
    """Convertit la valeur formatée en valeur numérique"""
    try:
        if label == "Âge":
            return float(value.split()[0])
        elif "€" in value:
            return float(value.replace("€", "").replace(" ", "").replace(",", "."))
        elif "%" in value:
            return float(value.replace("%", "").replace(",", ".")) / 100
        else:
            return float(value)
    except:
        return 0.0


def parse_currency(value):
    try:
        cleaned_value = (
            str(value)
            .replace("€", "")
            .translate(
                str.maketrans(
                    {
                        "\u202f": "",  # Espace insécable étroit
                        "\u00a0": "",  # Espace insécable standard
                        ",": ".",
                    }
                )
            )
            .strip()
        )
        return float(cleaned_value)
    except Exception as e:
        logger.error(f"Échec conversion devise : '{value}' → {str(e)}")
        return 0.0
