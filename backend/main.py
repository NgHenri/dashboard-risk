from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import shap
import pandas as pd

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
        X = pd.DataFrame([client.data])[features]
        X_scaled = scaler.transform(X)
        prob = model.predict_proba(X_scaled)[0, 1]
        decision = "✅ Accepté" if prob < threshold else "❌ Refusé"
        return {"probability": round(prob * 100, 2), "decision": decision}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
def explain(client: ClientData):
    try:
        X = pd.DataFrame([client.data])[features]
        X_scaled = scaler.transform(X)
        shap_values = explainer(X_scaled)
        top_features = sorted(
            zip(features, shap_values.values[0]), key=lambda x: abs(x[1]), reverse=True
        )[:10]
        return {"explanation": [{f: round(val, 4)} for f, val in top_features]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
