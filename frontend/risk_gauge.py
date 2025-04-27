import plotly.graph_objects as go
import streamlit as st
import config  # Contient le seuil optimal : THRESHOLD
import numpy as np
import time

# 🔁 Recalibrage visuel : place le seuil à 0.5 sur la jauge
def visual_scale(score, threshold):
    if score <= threshold:
        return 0.5 * score / threshold
    else:
        return 0.5 + 0.5 * (score - threshold) / (1 - threshold)

# 🟢🟡🔴 Fonction principale d'affichage
def show_risk_gauge(score, client_id, threshold=config.THRESHOLD):
    if score is None:
        #st.markdown("⏳ Aucune prédiction encore effectuée.")
        fig = go.Figure(go.Indicator(
            mode="gauge",
            value=0,
            gauge={'axis': {'range': [0, 1]}, 'bar': {'color': "lightgray"}},
            title={'text': "Score de défaut (en attente)"},
        ))
        st.plotly_chart(fig, use_container_width=True)
        return

    # Sécurité typage
    score = float(score) if isinstance(score, (np.ndarray,)) else score
    threshold = float(threshold) if isinstance(threshold, (np.ndarray,)) else threshold

    scaled_score = visual_scale(score, threshold)
    scaled_threshold = visual_scale(threshold, threshold)

    risk_status = "À risque" if score >= threshold else "Non à risque"
    risk_color = "#ff4d4d" if score >= threshold else "#2ecc71"  # rouge / vert doux

    # Titre
    #st.markdown(
    #    f"<div style='color:{risk_color}; font-size: 24px; font-weight:bold'>{risk_status}</div>",
    #    unsafe_allow_html=True
    #)

    # Contexte score et seuil
    #st.markdown(f"<p style='font-size:18px'>Score brut : <strong>{score:.4f}</strong> | Seuil : <strong>{threshold:.4f}</strong></p>", unsafe_allow_html=True)

    # Option : jauge de progression simple
    #st.progress(min(score, 1.0))

    # Option : metric
    #st.metric("Score de défaut", f"{score:.3f}")

    # Jauge visuelle
    risk_gauge = go.Indicator(
        mode="gauge+number",
        value=scaled_score,
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "white"},
            'bgcolor': "#f0f0f0",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': "#7fd47f"},
                {'range': [0.5, 0.75], 'color': "#ffe178"},
                {'range': [0.75, 1], 'color': "#ff6b6b"},
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
        width=600,
        margin=dict(l=0, r=0, t=0, b=0),
        font={'color': 'white', 'family': 'Arial', 'size': 20}
    )

    st.plotly_chart(fig, use_container_width=True)

# 🚨 Affichage du message de risque
def display_risk_message(score, threshold=config.THRESHOLD):
    if score is None:
        st.info("🔎 Veuillez lancer l’analyse pour voir le résultat.")
        return
    if score >= threshold:
        st.warning("🚨 Client à risque - Revue manuelle requise")
    else:
        st.success("✅ Client fiable - Prêt recommandé")

# ⚙️ Animation de la jauge (progressive)
def animate_risk_gauge(score: float, client_id: int, duration: float = 0.6, steps: int = 30):
    if score is None:
        show_risk_gauge(None, client_id)
        return

    scores = np.linspace(0, score, steps)
    delay = duration / steps
    placeholder = st.empty()

    for s in scores:
        with placeholder:
            show_risk_gauge(s, client_id)
        time.sleep(delay)
