from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === Charger les artefacts ===
ARTIFACT_PATH = "models/lightgbm_production_artifact_20250415_081218.pkl"
artifacts = joblib.load(ARTIFACT_PATH)
model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['metadata']['features']
threshold = artifacts['metadata']['optimal_threshold']

# === Initialisation de SHAP ===
explainer = shap.Explainer(model)

app = FastAPI()

class ClientData(BaseModel):
    data: dict  # exemple : {"EXT_SOURCE_1": 0.12, "AMT_CREDIT": 350000, ...}

@app.post("/predict")
def predict(client: ClientData):
    try:
        # Transformer les données d'entrée en DataFrame et appliquer le scaler
        X = pd.DataFrame([client.data])[features]
        X_scaled = scaler.transform(X)

        # Calculer la probabilité de défaut
        prob = model.predict_proba(X_scaled)[0, 1]
        
        # Définir la décision selon le seuil
        decision = "✅ Accepté" if prob < threshold else "❌ Refusé"
        
        return {"probability": round(prob * 100, 2), "decision": decision}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur dans la prédiction : {str(e)}")

@app.post("/explain")
def explain(client: ClientData):
    try:
        # Transformer les données d'entrée en DataFrame et appliquer le scaler
        X = pd.DataFrame([client.data])[features]
        X_scaled = scaler.transform(X)

        # Calculer les valeurs SHAP
        shap_values = explainer(X_scaled)
        
        # Extraire les 10 meilleures caractéristiques influençant la prédiction
        top_features = sorted(
            zip(features, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        
        # Formater l'explication de manière lisible
        explanation = [{"feature": f, "shap_value": round(val, 4)} for f, val in top_features]
        
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur dans l'explication : {str(e)}")
