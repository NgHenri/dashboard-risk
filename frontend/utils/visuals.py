# frontend/utils/visuals.py

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_boxplot_comparison(population_stats: dict, client_value: float, 
                           title: str, unit: str, transform=None):
    """
    Affiche un boxplot de comparaison client/population avec Plotly
    Args:
        population_stats (dict): Statistiques de population de l'API
        client_value (float): Valeur brute du client
        title (str): Titre du graphique
        unit (str): Unité de mesure
        transform (function): Fonction de transformation des données si nécessaire
    """
    # Transformation de la valeur client si nécessaire
    transformed_value = transform(client_value) if transform else client_value
    
    # Création du boxplot avec Plotly
    fig = go.Figure()
    
    # Ajout du boxplot
    fig.add_trace(go.Box(
        q1=[population_stats.get('25%', 0)],
        median=[population_stats.get('50%', 0)],
        q3=[population_stats.get('75%', 0)],
        lowerfence=[population_stats.get('min', 0)],
        upperfence=[population_stats.get('max', 0)],
        mean=[population_stats.get('mean', 0)],
        name="Population",
        marker_color='#1f77b4',
        boxmean=True,
        width=0.5,
        showlegend=False
    ))
    
    # Ajout du point client
    fig.add_trace(go.Scatter(
        x=[0],
        y=[transformed_value],
        mode='markers',
        marker=dict(
            color='red',
            size=14,
            line=dict(
                color='black',
                width=2
            )
        ),
        name="Client"
    ))
    
    # Ajout de la ligne de moyenne
    fig.add_shape(
        type="line",
        x0=-0.4, y0=population_stats.get('mean', 0),
        x1=0.4, y1=population_stats.get('mean', 0),
        line=dict(
            color="green",
            width=2,
            dash="dash",
        ),
        name="Moyenne"
    )
    
    # Mise en forme du graphique (CORRIGÉ)
    fig.update_layout(
        title={
            'text': title,
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 18}
        },
        yaxis_title=f"Valeur ({unit})",
        showlegend=True,
        xaxis={
            'showticklabels': False,
            'range': [-1, 1]
        },
        plot_bgcolor='rgba(240,240,240,0.8)',
        paper_bgcolor='rgba(255,255,255,0.8)',
        margin=dict(l=40, r=40, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Affichage dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

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
    values = np.array(explanation['values'])   # shape: (n_samples, n_features)
    data = np.array(explanation['data'])         # shape: (n_samples, n_features)
    feature_names = explanation['feature_names'] # liste des noms

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
            showlegend=False
        )
        fig.add_trace(trace)

    # Définir les labels de l'axe y
    y_ticks = list(range(len(selected_idx)))
    y_labels = [feature_names[idx] for idx in selected_idx]

    fig.update_layout(
        title="Facteurs qui influencent les décisions du modèle",
        xaxis_title="Valeur SHAP",
        yaxis=dict(
            tickmode="array",
            tickvals=y_ticks,
            ticktext=y_labels,
            title="Features"
        ),
        height=600,
        margin=dict(l=150, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


# =======================================================================
def plot_shap_histogram(feature, client_id):
    # URL de l'endpoint. Ajuste le host et le port si nécessaire.
    url = "http://localhost:8000/shap/histogram"
    params = {"feature": feature, "client_id": client_id}
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("Erreur lors de la récupération des données:", response.json())
        return
    
    data = response.json()
    global_distribution = data["global_distribution"]
    client_value = data["client_value"]
    
    # Créer l'histogramme
    fig = px.histogram(
        global_distribution, 
        nbins=30, 
        labels={"value": "Valeur SHAP", "count": "Fréquence"},
        title=f"Distribution des valeurs SHAP pour la feature '{feature}'"
    )
    
    # Ajout d'une ligne verticale pour la valeur du client
    fig.add_vline(
        x=client_value, 
        line_color="red", 
        line_dash="dash", 
        annotation_text="Client",
        annotation_position="top right"
    )
    
    fig.show()
# ==========================================================

def plot_multi_feature_shap_histogram(shap_df_long, feature_list, client_id, title):
    """Affiche un histogramme Plotly de SHAP values pour plusieurs features"""
    import plotly.express as px
    import plotly.graph_objects as go

    df = shap_df_long[shap_df_long["feature"].isin(feature_list)]

    fig = px.box(
        df,
        x="shap_value",
        y="feature",
        orientation="h",
        color_discrete_sequence=["#8ecae6"],
        title=title
    )

    # Ajouter les valeurs du client sélectionné
    client_values = df[df["SK_ID_CURR"] == client_id]
    for _, row in client_values.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["shap_value"]],
            y=[row["feature"]],
            mode="markers+text",
            text=[f"{row['shap_value']:.3f}"],
            textposition="middle right",
            marker=dict(color="#fb8500", size=10),
            name=f"Client {client_id}",
            showlegend=False
        ))

    fig.update_layout(
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        margin=dict(l=100, r=40, t=50, b=40),
        height=400 + 30 * len(feature_list)
    )

    return fig

# =====================================================

def plot_waterfall_chart(explanation):
    """Crée un waterfall plot avec Plotly"""
    features = explanation['features']
    values = explanation['values']
    base_value = explanation['base_value']
    
    # Préparation des données
    df = pd.DataFrame({
        'Feature': features[:10],
        'Impact': values[:10]
    }).sort_values('Impact', ascending=True)
    
    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(df),
        x=df['Impact'],
        y=df['Feature'],
        base=base_value,
        decreasing={"marker":{"color":"#4a6fa5"}},
        increasing={"marker":{"color":"#ff7f0e"}},
        totals={"marker":{"color":"#d62728"}}
    ))
    
    fig.update_layout(
        title="Impact des caractéristiques sur le score",
        height=600,
        margin=dict(l=100, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

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
    height=400
):
    # Données
    hist_data = full_data[feature_name].dropna()
    client_feature_value = client_data[feature_name].values[0] if hasattr(client_data[feature_name], 'values') else client_data[feature_name]

    # Histogramme (données pour construire manuellement)
    counts, bins = np.histogram(hist_data, bins=30)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    # Identifier le bin du client
    client_bin_index = np.digitize(client_feature_value, bins) - 1
    client_bin_index = np.clip(client_bin_index, 0, len(counts)-1)

    # Couleurs : base + une différente pour le bin client
    bar_colors = [base_color] * len(counts)
    bar_colors[client_bin_index] = client_bin_color

    # Figure manuelle avec go.Bar
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=counts,
        marker_color=bar_colors,
        width=(bins[1] - bins[0]) * 0.9,
        hovertemplate='Valeur: %{x:.2f}<br>Nb clients: %{y}<extra></extra>'
    ))

    # Ligne verticale pour le client
    fig.add_vline(
        x=client_feature_value,
        line_width=3,
        line_dash="dash",
        line_color=client_bin_color,
        annotation_text="Client",
        annotation_position="top"
    )

    # Échelle log si besoin
    mean_val = np.mean(counts)
    std_val = np.std(counts)
    if std_val > 3 * mean_val:
        fig.update_layout(yaxis_type="log")

    # Mise en page
    title_font_size = get_title_font_size(height)
    fig.update_layout(
        title_text=f"{title_prefix} {feature_name}",
        title_font=dict(size=title_font_size),
        xaxis_title=feature_name,
        yaxis_title="Nombre de clients",
        title_x=0.3,
        height=height,
        margin=dict(t=50, l=20, r=20, b=40)
    )

    return fig    
# =====================================================================
def plot_comparative_shap(df_global_shap, df_client_shap, top_n=10):
    # Prendre les n principales features par impact global
    top_features = df_global_shap.groupby('feature')['shap_value'].apply(lambda x: x.abs().mean()).sort_values(ascending=False).head(top_n).index.tolist()
    
    # Extraire les données pour les features les plus importantes
    df_global_shap_top = df_global_shap[df_global_shap['feature'].isin(top_features)]
    df_client_shap_top = df_client_shap[df_client_shap['feature'].isin(top_features)]
    
    # Créer une figure
    fig = go.Figure()

    # Ajouter les valeurs SHAP globales
    fig.add_trace(go.Bar(
        x=df_global_shap_top['feature'],
        y=df_global_shap_top['shap_value'],
        name='SHAP Global',
        marker_color='rgba(255, 99, 132, 0.6)',
        orientation='v'
    ))

    # Ajouter les valeurs SHAP du client
    fig.add_trace(go.Bar(
        x=df_client_shap_top['feature'],
        y=df_client_shap_top['shap_value'],
        name='SHAP Client',
        marker_color='rgba(54, 162, 235, 0.6)',
        orientation='v'
    ))

    # Ajouter des détails de mise en page
    fig.update_layout(
        title=f"Comparaison des valeurs SHAP (global vs client)",
        xaxis_title="Features",
        yaxis_title="SHAP Values",
        barmode='group',  # Affichage groupé pour les deux sets de données
        showlegend=True,
        height=600
    )

    return fig
# =============================================================    
def plot_shap_by_decision(df_long, client_data, target_column='TARGET', top_n=10):
    # Séparer les décisions : octroi (1) ou refus (0)
    df_long['decision'] = client_data[target_column]
    
    # Calculer l'impact moyen des features SHAP par type de décision
    df_decision_shap = df_long.groupby(['feature', 'decision'])['shap_value'].mean().unstack().fillna(0)
    
    # Trier les features selon leur impact global
    top_features = df_decision_shap.abs().sum(axis=1).sort_values(ascending=False).head(top_n).index.tolist()
    
    # Sélectionner les top features
    df_decision_shap = df_decision_shap.loc[top_features]
    
    # Création du graphique
    fig = go.Figure()

    # Ajouter les valeurs SHAP pour l'octroi (1)
    fig.add_trace(go.Bar(
        x=df_decision_shap.index,
        y=df_decision_shap[1],
        name='Octroi (1)',
        marker_color='rgba(75, 192, 192, 0.6)',
        orientation='v'
    ))

    # Ajouter les valeurs SHAP pour le refus (0)
    fig.add_trace(go.Bar(
        x=df_decision_shap.index,
        y=df_decision_shap[0],
        name='Refus (0)',
        marker_color='rgba(153, 102, 255, 0.6)',
        orientation='v'
    ))

    # Détails de mise en page
    fig.update_layout(
        title="Comparaison des valeurs SHAP par type de décision (Octroi vs Refus)",
        xaxis_title="Features",
        yaxis_title="Moyenne des valeurs SHAP",
        barmode='group',  # Affichage groupé
        showlegend=True,
        height=600
    )

    return fig