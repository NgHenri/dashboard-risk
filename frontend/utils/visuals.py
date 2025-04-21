# frontend/utils/visuals.py

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_boxplot_comparison(
    population_series,
    client_value,
    title,
    xlabel="Valeur",
    unit="",
    transform=None
):
    """
    Affiche un boxplot de la distribution populationnelle avec la valeur du client.
    
    Args:
        population_series (pd.Series): données de référence.
        client_value (float): valeur du client à afficher.
        title (str): titre du graphe.
        xlabel (str): étiquette de l’axe x.
        unit (str): suffixe à afficher sur la valeur du client (%, €, etc.).
        transform (function, optional): fonction à appliquer à population_series.
    """
    if transform:
        population_series = population_series.dropna().apply(transform)

    fig, ax = plt.subplots(figsize=(8, 1.5))
    sns.boxplot(x=population_series, ax=ax, color="#a4c2f4")
    ax.axvline(client_value, color='red', linestyle='--', label=f"Client ({client_value:.2f}{unit})")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.get_yaxis().set_visible(False)
    ax.legend()
    st.pyplot(fig)
