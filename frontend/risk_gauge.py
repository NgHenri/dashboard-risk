THRESHOLD = 0.0931515
import plotly.graph_objects as go
import streamlit as st
import config  # ⬅️ Contient le seuil optimal (THRESHOLD)
import numpy as np

# 🔁 Fonction de recalibrage visuel
def visual_scale(score, threshold):
    """
    Recalibrage visuel : place le seuil à 0.5 sur la jauge.
    Étire la portion 0 → threshold sur 0 → 0.5 et le reste sur 0.5 → 1.
    """
    if score <= threshold:
        return 0.5 * score / threshold
    else:
        return 0.5 + 0.5 * (score - threshold) / (1 - threshold)

# 📊 Fonction principale
def show_risk_gauge(score, client_id, threshold=config.THRESHOLD):
    if score is None:
        # Jauge vide
        st.markdown("⏳ Aucune prédiction encore effectuée.")
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=0,
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "lightgray"}},
            title={'text': "Score de défaut (en attente)"},
        ))
        st.plotly_chart(fig, use_container_width=True)
        return

    # ⚙️ Sécurité typage
    score = float(score) if isinstance(score, (np.ndarray,)) else score
    threshold = float(threshold) if isinstance(threshold, (np.ndarray,)) else threshold

    # 🔁 Recalage visuel
    scaled_score = visual_scale(score, threshold)
    scaled_threshold = visual_scale(threshold, threshold)

    # 🔴🟢 Statut
    risk_status = "À risque" if score >= threshold else "Non à risque"
    risk_color = "red" if score >= threshold else "green"

     # 🟩🟥 Message de statut
    st.markdown(
        f"<div style='color:{risk_color}; font-size: 24px; font-weight:bold'>{risk_status}</div>",
        unsafe_allow_html=True
    )
    # 🔧 Jauge personnalisée
    risk_gauge = go.Indicator(
        mode="gauge+number",
        value=scaled_score,
        #title={'text': f"Client {client_id}", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "white"},
            'bgcolor': "#f0f0f0",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': "green"},
                {'range': [0.5, 0.75], 'color': "yellow"},
                {'range': [0.75, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': risk_color, 'width': 4},
                'thickness': 0.75,
                'value': scaled_threshold
            }
        },
        number={'valueformat': '.3f', 'font': {'size': 45}}
    )

    fig = go.Figure(risk_gauge)
    fig.update_layout(
    height=450,
    width=600,  # plus large
    margin=dict(l=0, r=0, t=0, b=0),  # supprime les marges hautes
    font={'color': 'white', 'family': 'Arial', 'size': 20}
)
    

    st.plotly_chart(fig, use_container_width=True)
   
