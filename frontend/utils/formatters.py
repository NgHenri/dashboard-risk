# frontend/utils/formatters.py

import pandas as pd

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