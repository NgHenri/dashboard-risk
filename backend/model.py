import joblib
import pandas as pd
import numpy as np
import joblib

def load_model_and_pipeline(model_path):
    """Charge tous les artefacts nécessaires (modèle, scaler, métadonnées...)"""
    return joblib.load(model_path)  # Cela retourne un dict

def predict_score(artifacts, input_data, threshold=None):
    model = artifacts['model']
    scaler = artifacts['scaler']
    metadata = artifacts['metadata']
    
    # Vérifie si 'features' existe dans les métadonnées
    expected_features = metadata.get('features', None)
    
    if expected_features is None:
        raise ValueError("Les features attendues ne sont pas spécifiées dans les métadonnées.")
    
    # Vérification que input_data est bien un DataFrame
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Les données d'entrée doivent être un DataFrame.")
    
    # Vérification des colonnes manquantes
    missing_cols = set(expected_features) - set(input_data.columns)
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans les données d'entrée : {missing_cols}")
    
    # Filtrage des données d'entrée selon les features attendues
    input_data = input_data[expected_features]
    
    # Transformation des données d'entrée
    X_scaled = scaler.transform(input_data)
    
    # Prédiction
    proba = model.predict_proba(X_scaled)[:, 1]
    pred = (proba >= threshold).astype(int)
    
    return float(proba[0]), int(pred[0])





