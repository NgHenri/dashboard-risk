# frontend/utils/visuals.py

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots


def restore_discrete_types(df, max_cardinality=20, verbose=False):
    """
    Convertit certaines colonnes float en int ou category si elles semblent discrètes.

    Args:
        df (pd.DataFrame): Le DataFrame à traiter.
        max_cardinality (int): Seuil maximum de valeurs uniques pour considérer comme discret.
        verbose (bool): Affiche les colonnes converties.

    Returns:
        pd.DataFrame: Un nouveau DataFrame avec des types adaptés.
    """
    df_new = df.copy()
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            uniques = df[col].dropna().unique()
            if (
                all((uniques == uniques.astype(int)))
                and len(uniques) <= max_cardinality
            ):
                # Tous les floats sont des entiers et peu nombreux
                if verbose:
                    print(f"Colonne '{col}' convertie en int")
                df_new[col] = df[col].astype("Int64")  # Int64 = int nullable
            elif len(uniques) <= max_cardinality:
                # Peu de valeurs uniques => variable discrète => category
                if verbose:
                    print(f"Colonne '{col}' convertie en category")
                df_new[col] = df[col].astype("category")
    return df_new


def plot_boxplot_comparison(
    population_stats: dict,
    client_value: float,
    title: str,
    unit: str,
    transform=None,
    height=450,
):
    """
    Boxplot enrichi avec statistiques détaillées
    """
    # Transformation des valeurs
    stats = {k: transform(v) if transform else v for k, v in population_stats.items()}
    transformed_value = transform(client_value) if transform else client_value

    # Création de la figure
    fig = go.Figure()

    # Boxplot principal avec effet de profondeur
    fig.add_trace(
        go.Box(
            q1=[stats["25%"]],
            median=[stats["50%"]],
            q3=[stats["75%"]],
            lowerfence=[stats["min"]],
            upperfence=[stats["max"]],
            mean=[stats["mean"]],
            name="Population",
            marker_color="#636EFA",
            line_color="#2A3F5F",
            fillcolor="rgba(99, 110, 250, 0.3)",
            boxmean="sd",
            width=0.4,
            notchwidth=0.2,
            whiskerwidth=0.2,
            hoverinfo="none",
        )
    )

    # Point client stylisé
    fig.add_trace(
        go.Scatter(
            x=[0.15],
            y=[transformed_value],
            mode="markers+text",
            marker=dict(
                color="#FF6692",
                size=16,
                line=dict(color="black", width=1.5),
                symbol="diamond",
            ),
            text=[f"  Client : {transformed_value:.2f}{unit}"],
            textposition="middle right",
            textfont=dict(color="#FF6692", size=12),
            name="Client",
            hoverinfo="y+name",
        )
    )

    # Ligne de moyenne interactive
    fig.add_shape(
        type="line",
        x0=-0.4,
        y0=stats["mean"],
        x1=0.4,
        y1=stats["mean"],
        line=dict(color="#00CC96", width=2.5, dash="dot"),
        name="Moyenne",
        opacity=0.8,
    )

    # Annotation de la moyenne
    fig.add_annotation(
        x=0.5,
        y=stats["mean"],
        text=f"Moyenne : {stats['mean']:.2f}{unit}",
        showarrow=False,
        font=dict(color="#00CC96", size=11),
        xanchor="left",
    )

    # Zone de percentile
    fig.add_shape(
        type="rect",
        x0=-0.5,
        y0=stats["25%"],
        x1=0.5,
        y1=stats["75%"],
        fillcolor="rgba(99, 110, 250, 0.1)",
        line=dict(width=0),
        name="25-75%",
    )

    # Mise en forme avancée
    fig.update_layout(
        title={
            "text": f"<b>{title}</b><br><sub style='color:gray'>Comparaison avec la population</sub>",
            "x": 0.05,
            "xanchor": "left",
        },
        yaxis_title=f"Valeur ({unit})",
        xaxis={
            "showticklabels": False,
            "range": [-1, 1],
            "gridcolor": "rgba(0,0,0,0.05)",
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=80, l=40, r=20, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.8)",
        ),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        height=height,
        font_family="Arial",
        separators=",.",
    )

    # Ajout des éléments interactifs
    fig.update_traces(
        hoverlabel=dict(
            bgcolor="white", bordercolor="#2A3F5F", font=dict(color="#2A3F5F")
        ),
        selector=dict(type="box"),
    )

    # Affichage des statistiques au hover
    fig.data[0].hovertemplate = (
        "<b>Population</b><br>"
        "Min: %{lowerfence:.2f}<br>"
        "Q1: %{q1:.2f}<br>"
        "Médiane: %{median:.2f}<br>"
        "Q3: %{q3:.2f}<br>"
        "Max: %{upperfence:.2f}<br>"
        "Moyenne: %{mean:.2f}<br>"
        "Écart-type: %{sd:.2f}<extra></extra>"
    )

    return st.plotly_chart(fig, use_container_width=True)


# ======================================================================


def plot_summary_chart(explanation, max_display=10):
    """
    Crée un summary plot de style SHAP (dot plot) avec Plotly.

    Les éléments attendus dans "explanation" sont :
      - "values": un tableau numpy de forme (n_samples, n_features) contenant les valeurs SHAP
      - "data": un tableau numpy de forme (n_samples, n_features) avec les valeurs originales des features
      - "feature_names": une liste des noms de features

    Le plot affiche pour chaque feature (les `max_display` les plus importantes) la distribution
    des valeurs SHAP pour chaque observation, avec une coloration indiquant la valeur d'origine de la feature.
    """
    # Récupération des données
    values = np.array(explanation["values"])  # shape: (n_samples, n_features)
    data = np.array(explanation["data"])  # shape: (n_samples, n_features)
    feature_names = explanation["feature_names"]  # liste des noms

    # Calcul de l'importance moyenne (valeurs absolues)
    mean_abs_values = np.mean(np.abs(values), axis=0)
    # On trie les features par importance décroissante
    sorted_idx = np.argsort(mean_abs_values)[::-1]
    # On ne garde que les "max_display" premières en convertissant explicitement les indices en entiers Python
    selected_idx = [int(idx) for idx in sorted_idx[:max_display]]

    fig = go.Figure()

    # Pour chacune des features sélectionnées, tracer les points correspondants
    for i, idx in enumerate(selected_idx):
        feat_name = feature_names[idx]
        shap_vals = values[:, idx]
        feat_vals = data[:, idx]

        # Ajouter du "jitter" pour répartir les points sur l'axe y
        jitter = np.random.uniform(-0.2, 0.2, size=shap_vals.shape[0])
        y_coord = np.full_like(shap_vals, i, dtype=float) + jitter

        trace = go.Scatter(
            x=shap_vals,
            y=y_coord,
            mode="markers",
            marker=dict(
                color=feat_vals,
                colorscale="RdBu",
                reversescale=True,
                size=7,
                opacity=0.7,
                # Afficher la colorbar pour la première trace uniquement
                colorbar=dict(title=feat_name) if i == 0 else None,
            ),
            name=feat_name,
            text=[f"{feat_name}: {val}" for val in feat_vals],
            showlegend=False,
        )
        fig.add_trace(trace)

    # Définir les labels de l'axe y
    y_ticks = list(range(len(selected_idx)))
    y_labels = [feature_names[idx] for idx in selected_idx]

    fig.update_layout(
        title="Facteurs qui influencent les décisions du modèle",
        xaxis_title="Valeur SHAP",
        yaxis=dict(
            tickmode="array", tickvals=y_ticks, ticktext=y_labels, title="Features"
        ),
        height=600,
        margin=dict(l=150, r=50, t=80, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# =====================================================


def plot_waterfall_chart_expandable(explanation, initial_max_features=10):
    """Crée un waterfall plot interactif avec possibilité de changer le nombre de features affichées."""

    features = explanation["features"]
    values = explanation["values"]
    base_value = explanation["base_value"]

    with st.expander("Options d'affichage"):
        max_features = st.slider(
            "Nombre de features affichées",
            min_value=5,
            max_value=min(len(features), 50),
            value=initial_max_features,
            step=5,
            help="Sélectionnez combien de variables afficher.",
        )

        positive_color = st.color_picker(
            "Couleur des contributions positives", value="#ff7f0e"
        )
        negative_color = st.color_picker(
            "Couleur des contributions négatives", value="#4a6fa5"
        )

        show_base_value = st.checkbox(
            "Afficher la valeur de base (base value)", value=True
        )

    # --- Préparation des données ---
    # Affichage des n premières features sans filtrage des signes
    df = pd.DataFrame(
        {"Feature": features[:max_features], "Impact": values[:max_features]}
    )

    # Trier par impact croissant (en respectant les signes)
    df = df.sort_values("Impact", ascending=True)

    # --- Création du graphique ---
    fig = go.Figure()

    fig.add_trace(
        go.Waterfall(
            orientation="h",
            measure=["relative"] * len(df),
            x=df["Impact"],
            y=df["Feature"],
            base=base_value if show_base_value else 0,
            decreasing={"marker": {"color": negative_color}},
            increasing={"marker": {"color": positive_color}},
            totals={"marker": {"color": "#d62728"}},
        )
    )

    fig.update_layout(
        title="Impact des caractéristiques sur le score",
        height=600,
        margin=dict(l=100, r=50, t=80, b=50),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================================


def get_title_font_size(height):
    if height > 700:
        return 24
    elif height > 500:
        return 20
    else:
        return 16


def plot_feature_distribution(
    feature_name,
    full_data,
    client_data,
    base_color,
    client_bin_color="black",
    title_prefix="Distribution pour",
    height=450,
):
    # Récupération des données
    data = full_data[feature_name].dropna()
    client_value = (
        client_data[feature_name].values[0]
        if hasattr(client_data[feature_name], "values")
        else client_data[feature_name]
    )

    # Détection automatique du type de variable
    unique_vals = data.unique()
    is_discrete = False

    # 1. Détection des variables discrètes
    if len(unique_vals) <= 20 or all(np.equal(unique_vals, unique_vals.astype(int))):
        is_discrete = True
        int_vals = data.astype(int)
        min_val, max_val = int_vals.min(), int_vals.max()
        bins = np.arange(min_val - 0.5, max_val + 1.5)
        bin_labels = [str(i) for i in range(min_val, max_val + 1)]
    else:
        # 2. Cas continu avec règle de Sturges
        n_bins = min(30, int(np.log2(len(data)) + 1))
        counts, bins = np.histogram(data, bins=n_bins)
        bin_labels = None

    # Calcul des positions et largeurs des barres
    if is_discrete:
        bin_centers = bins[:-1] + 0.5
        bar_width = 0.8  # Largeur réduite pour espacement
    else:
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        bar_width = (bins[1] - bins[0]) * 0.9

    # Recréer l'histogramme avec les bons bins
    counts, _ = np.histogram(data, bins=bins)

    # Trouver l'index du bin du client (méthode robuste)
    if is_discrete:
        client_bin_index = np.searchsorted(bins, client_value) - 1
    else:
        client_bin_index = np.clip(
            np.digitize(client_value, bins) - 1, 0, len(counts) - 1
        )

    # Création de la figure
    fig = go.Figure()

    # Ajout des barres avec couleur spéciale pour le bin client
    colors = [
        client_bin_color if i == client_bin_index else base_color
        for i in range(len(counts))
    ]

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=counts,
            width=bar_width,
            marker_color=colors,
            hovertemplate="<b>Valeur</b>: %{x}<br><b>Clients</b>: %{y}<extra></extra>",
            name="Population",
        )
    )

    # Ajout ligne verticale pour le client
    fig.add_vline(
        x=client_value,
        line=dict(color=client_bin_color, width=2, dash="dot"),
        annotation_text=f"Client: {client_value:.2f}",
        annotation_position="top right",
    )

    # Mise en forme adaptative
    if is_discrete:
        fig.update_xaxes(tickvals=bin_centers, ticktext=bin_labels, type="category")
    else:
        fig.update_xaxes(tickformat=".1f")

    # if counts.max() / counts.min() > 100:
    # fig.update_yaxes(type="log", title="Nombre de clients (log)")

    # Protection contre divide by zero pour l'échelle log
    min_count = counts.min() if counts.size > 0 else 0
    max_count = counts.max() if counts.size > 0 else 0
    # On n’active le log que si tous les bins ont au moins 1 occurrence
    if min_count > 0 and (max_count / min_count) > 100:
        fig.update_yaxes(type="log", title="Nombre de clients (log)")

    # Paramètres graphiques
    fig.update_layout(
        title=f"{title_prefix} {feature_name}",
        bargap=0.05 if is_discrete else 0.01,
        height=height,
        xaxis_title=feature_name,
        yaxis_title="Nombre de clients",
        showlegend=False,
    )

    return fig


# =============================================================


def plot_feature_comparison(
    df, feature_x, feature_y, plot_type="scatter", client_data=None
):

    # Vérification des features
    missing = [f for f in [feature_x, feature_y] if f not in df.columns]
    if missing:
        st.error(f"Features manquantes : {', '.join(missing)}")
        return

    # Création des figures avec nouveau thème
    if plot_type == "scatter":
        fig = px.scatter(
            df,
            x=feature_x,
            y=feature_y,
            opacity=0.5,
            title=f"Relation {feature_x} vs {feature_y}",
            template="plotly_dark",
            color_discrete_sequence=["#2A9D8F"],
        )

    elif plot_type == "histogram":
        fig = px.histogram(
            df,
            x=feature_x,
            nbins=30,
            opacity=0.7,
            marginal="rug",
            color_discrete_sequence=["#264653"],
            template="plotly_dark",
        )
        fig.add_trace(
            px.histogram(
                df,
                x=feature_y,
                nbins=30,
                opacity=0.7,
                color_discrete_sequence=["#E9C46A"],
            ).data[0]
        )
        fig.update_layout(
            title=f"Distributions comparées de {feature_x} et {feature_y}"
        )

    elif plot_type == "density":
        fig = px.density_contour(
            df,
            x=feature_x,
            y=feature_y,
            title=f"Densité {feature_x} vs {feature_y}",
            template="plotly_dark",
            color_discrete_sequence=["#E9C46A"],  # Paramètre corrigé
        )

    elif plot_type == "histogram2d":
        fig = px.density_heatmap(
            df,
            x=feature_x,
            y=feature_y,
            nbinsx=30,
            nbinsy=30,
            title=f"Distribution conjointe {feature_x} vs {feature_y}",
            color_continuous_scale="Plasma",
            template="plotly_dark",
        )

    # Ajout du point client
    if client_data is not None and plot_type != "histogram":
        fig.add_trace(
            go.Scatter(
                x=[client_data[feature_x]],
                y=[client_data[feature_y]] if plot_type != "histogram" else [None],
                mode="markers",
                marker=dict(
                    color="#E76F51",
                    size=14,
                    line=dict(width=2, color="white"),
                    symbol="diamond",
                ),
                name="Position du client",
            )
        )

    # Mise à jour globale du style
    fig.update_layout(
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#404040", zerolinecolor="#404040", linecolor="#404040"),
        yaxis=dict(gridcolor="#404040", zerolinecolor="#404040", linecolor="#404040"),
        legend=dict(bgcolor="#333333", bordercolor="#404040"),
    )

    st.plotly_chart(fig, use_container_width=True)


# 3. Plotly for client position
def plot_client_position_in_group(df_results: pd.DataFrame, selected_client_id: int):
    if selected_client_id not in df_results["SK_ID_CURR"].values:
        st.warning("Client non trouvé dans le dataset de prédictions.")
        return
    client_row = df_results[df_results["SK_ID_CURR"] == selected_client_id].iloc[0]
    prob = client_row["probability"]
    decision = client_row["decision"]
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Tous les clients", f"Groupe {decision}")
    )
    fig.add_trace(
        go.Histogram(x=df_results["probability"], nbinsx=30, name="Tous", opacity=0.75),
        row=1,
        col=1,
    )
    fig.add_shape(
        dict(
            type="line",
            x0=prob,
            x1=prob,
            y0=0,
            y1=1,
            xref="x1",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
    )
    fig.add_annotation(
        x=prob,
        y=1.05,
        xref="x1",
        yref="paper",
        text=f"Client ({prob:.3f})",
        showarrow=False,
    )
    group_probs = df_results[df_results["decision"] == decision]["probability"]
    fig.add_trace(
        go.Histogram(x=group_probs, nbinsx=30, name=str(decision), opacity=0.75),
        row=1,
        col=2,
    )
    fig.add_shape(
        dict(
            type="line",
            x0=prob,
            x1=prob,
            y0=0,
            y1=1,
            xref="x2",
            yref="paper",
            line=dict(color="red", width=2, dash="dash"),
        )
    )
    fig.add_annotation(
        x=prob,
        y=1.05,
        xref="x2",
        yref="paper",
        text=f"Client ({prob:.3f})",
        showarrow=False,
    )
    fig.update_layout(
        template="plotly_dark",
        bargap=0.2,
        title_text="Position du client dans les distributions",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ===========


def plot_univariate_feature0(
    df: pd.DataFrame,
    feature: str,
    client_value: float = None,
    max_discrete_cardinality: int = 10,
) -> go.Figure:
    """
    Trace automatiquement un histogramme (continu) ou un camembert (discret)
    selon la nature et la cardinalité de la variable, et y marque la valeur du client.

    Args:
        df: DataFrame source.
        feature: nom de la colonne à afficher.
        client_value: valeur du client à repérer sur le graphique.
        max_discrete_cardinality: seuil pour passer en mode discret.
    """
    series = df[feature].dropna()
    n_unique = series.nunique()

    # --- Cas discret (camembert) ---
    if n_unique <= max_discrete_cardinality:
        counts = series.value_counts().reset_index()
        counts.columns = [feature, "count"]
        # on repère la part du client si elle existe
        if client_value is not None and client_value in counts[feature].values:
            counts["pull"] = counts[feature].apply(
                lambda x: 0.1 if x == client_value else 0
            )
        else:
            counts["pull"] = 0

        fig = px.pie(
            counts,
            names=feature,
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark",
        )
        fig.update_traces(
            pull=counts["pull"],
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#2F2F2F", width=1)),
        )
        fig.update_layout(
            title=f"Répartition de {feature} ({n_unique} modalités)",
            uniformtext_minsize=12,
            uniformtext_mode="hide",
        )
        return fig

    # --- Cas continu (histogramme + boxplot) ---
    fig = px.histogram(
        df,
        x=feature,
        nbins=min(30, int(np.log2(len(series))) + 1),
        marginal="box",
        opacity=0.85,
        template="plotly_dark",
    )
    # styling histogramme et boxplot (idem à ton code original)…
    fig.update_traces(
        selector=dict(type="histogram"),
        marker=dict(line=dict(color="#1D7870", width=1.5)),
    )
    if len(fig.data) > 1:
        fig.update_traces(
            selector=dict(type="box"),
            line=dict(color="#E76F51", width=2.5),
            fillcolor="rgba(231, 111, 81, 0.4)",
            marker=dict(color="#E76F51", size=3, opacity=0.6, symbol="diamond"),
        )

    # --- Tracer la ligne verticale du client ---
    if client_value is not None:
        fig.add_vline(
            x=client_value,
            line=dict(color="yellow", width=3, dash="dash"),
            annotation_text="Client",
            annotation_position="top right",
            annotation_font_color="yellow",
        )

    fig.update_layout(
        title=dict(text=f"Distribution de {feature}", font=dict(size=18)),
        hoverlabel=dict(bgcolor="#2F2F2F"),
        margin=dict(t=50, b=40),
        showlegend=False,
    )
    return fig


def plot_univariate_feature(
    df: pd.DataFrame,
    feature: str,
    client_value: float = None,
    max_discrete_cardinality: int = 10,
) -> go.Figure:
    """
    Trace automatiquement un histogramme (continu) ou un camembert (discret)
    selon la nature et la cardinalité de la variable, et y marque la valeur du client.

    Args:
        df: DataFrame source.
        feature: nom de la colonne à afficher.
        client_value: valeur du client à repérer sur le graphique.
        max_discrete_cardinality: seuil pour passer en mode discret.
    """
    series = df[feature].dropna()
    n_unique = series.nunique()

    # --- Cas discret (camembert) ---
    if n_unique <= max_discrete_cardinality:
        counts = series.value_counts().reset_index()
        counts.columns = [feature, "count"]
        # on repère la part du client si elle existe
        counts["pull"] = counts[feature].apply(
            lambda x: 0.1 if x == client_value else 0
        )

        fig = px.pie(
            counts,
            names=feature,
            values="count",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark",
        )
        fig.update_traces(
            pull=counts["pull"],
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="#2F2F2F", width=1)),
        )
        fig.update_layout(
            title=f"Répartition de {feature} ({n_unique} modalités)",
            uniformtext_minsize=12,
            uniformtext_mode="hide",
        )
        return fig

    # --- Cas continu (histogramme + boxplot) ---
    fig = px.histogram(
        df,
        x=feature,
        nbins=min(30, int(np.log2(len(series))) + 1),
        marginal="box",
        opacity=0.85,
        template="plotly_dark",
    )

    # Amélioration du style
    fig.update_traces(
        selector=dict(type="histogram"),
        marker=dict(line=dict(color="#1D7870", width=1.5)),
    )
    if len(fig.data) > 1:
        fig.update_traces(
            selector=dict(type="box"),
            line=dict(color="#E76F51", width=2.5),
            fillcolor="rgba(231, 111, 81, 0.4)",
            marker=dict(color="#E76F51", size=3, opacity=0.6, symbol="diamond"),
        )

    # --- Tracer la ligne verticale du client ---
    if client_value is not None:
        fig.add_vline(
            x=client_value,
            line=dict(color="yellow", width=3, dash="dash"),
            annotation_text="Client",
            annotation_position="top right",
            annotation_font_color="yellow",
        )

    fig.update_layout(
        title=dict(text=f"Distribution de {feature}", font=dict(size=18)),
        hoverlabel=dict(bgcolor="#2F2F2F"),
        margin=dict(t=50, b=40),
        showlegend=False,
    )
    return fig
