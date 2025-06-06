import joblib
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Ajout d'un handler de sortie console si non déjà présent
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def load_model_and_pipeline(model_path):
    """Charge tous les artefacts nécessaires (modèle, scaler, métadonnées...)"""
    return joblib.load(model_path)  # Cela retourne un dict


def load_artifacts(artifact_path: str):
    """Charge les artefacts du modèle LightGBM (modèle, scaler, features, seuil)"""
    try:
        artifacts = joblib.load(artifact_path)
        model = artifacts["model"]
        scaler = artifacts["scaler"]
        features = artifacts["metadata"]["features"]
        threshold = artifacts["metadata"]["optimal_threshold"]
        logger.info("Artefacts chargés avec succès.")
        return model, scaler, features, threshold
    except Exception as e:
        logger.critical(f"Erreur de chargement des artefacts : {e}")
        raise RuntimeError("Impossible de charger les artefacts du modèle.")


def predict_score(artifacts, input_data, threshold=None):
    model = artifacts.get("model")
    scaler = artifacts.get("scaler")
    metadata = artifacts.get("metadata", {})

    if model is None or scaler is None:
        raise ValueError("Modèle ou scaler manquant dans les artefacts.")

    expected_features = metadata.get("features")
    if not expected_features:
        raise ValueError(
            "Les features attendues ne sont pas spécifiées dans les métadonnées."
        )

    # Vérification du type de données d'entrée
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Les données d'entrée doivent être un DataFrame.")

    # Vérification des colonnes manquantes
    missing_cols = set(expected_features) - set(input_data.columns)
    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans les données d'entrée : {missing_cols}"
        )

    # Filtrage et réordonnancement
    X = input_data[expected_features].copy()

    # Transformation des données
    try:
        X_scaled = scaler.transform(X)
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la transformation des données : {e}")

    # Calcul des probabilités
    try:
        proba = model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        raise RuntimeError(f"Erreur lors de la prédiction : {e}")

    # Utilisation du seuil par défaut si non spécifié
    if threshold is None:
        threshold = metadata.get("optimal_threshold", 0.5)

    pred = (proba >= threshold).astype(int)

    return float(proba[0]), int(pred[0])
