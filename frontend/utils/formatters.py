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
