# === 1. Imports ===
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path
import joblib
import numpy as np
import shap
import pandas as pd
import warnings
import logging
import sys
import os
import asyncio

# Cache et Redis
from fastapi_cache.decorator import cache
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.backends.inmemory import InMemoryBackend
from functools import lru_cache
from upstash_redis.asyncio import Redis

# === 2. Configuration initiale ===
# Chargement des variables d'environnement
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# Désactivation des warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === 3. Configuration du logger ===
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# === 4. Initialisation FastAPI ===
app = FastAPI()

# === 5. Middleware CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "x-api-key"],
)

# === 6. Sécurité et clé API ===
api_key_header = APIKeyHeader(name="x-api-key")
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
redis_client = None


async def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide")
    return api_key


# === 7. Configuration Redis et Cache ===
@app.on_event("startup")
async def startup():
    global redis_client
    try:
        # Vérification des variables d'environnement
        redis_url = os.getenv("UPSTASH_REDIS_URL")
        redis_token = os.getenv("UPSTASH_REDIS_TOKEN")

        if not redis_url or not redis_token:
            raise ValueError("Configuration Redis manquante dans .env")

        # Conversion de l'URL
        redis_url = redis_url.replace("https://", "rediss://")

        # Initialisation Redis
        redis_client = Redis(url=redis_url, token=redis_token)

        # Test de connexion
        pong = await asyncio.wait_for(redis_client.ping(), timeout=3)
        if pong != "PONG":
            raise ConnectionError("Réponse Redis invalide")

        # Initialisation du cache
        FastAPICache.init(RedisBackend(redis_client), prefix="demo-cache")
        logger.info("✅ Redis configuré avec succès")

    except Exception as e:
        redis_client = None
        FastAPICache.init(InMemoryBackend())
        logger.error(f"⚠️ Cache mémoire activé - Erreur Redis : {str(e)}")


# === 8. Logger de cache (optionnel) ===
class CacheLogger:
    async def set(self, key: str, value: str, expire: int):
        logger.info(f"SET CACHE: {key} (TTL: {expire}s)")
        return await FastAPICache.backend.set(key, value, expire)


FastAPICache.backend = CacheLogger()

# === 9. Variables globales  ===
REDIS_MEMORY_WARNING = 200  # Mo

# === 5. Chargement des artefacts ===
ARTIFACT_PATH = "models/lightgbm_production_artifact_20250415_081218.pkl"
try:
    artifacts = joblib.load(ARTIFACT_PATH)
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    features = artifacts["metadata"]["features"]
    threshold = artifacts["metadata"]["optimal_threshold"]
    logger.info("Artefacts chargés avec succès.")
except Exception as e:
    logger.critical(f"Erreur de chargement des artefacts : {e}")
    raise RuntimeError("Impossible de charger les artefacts du modèle.")

# === 6. Initialisation SHAP ===
explainer = shap.Explainer(model)
logger.info(f"Type de expected_value: {type(explainer.expected_value)}")
logger.info(f"Valeur de expected_value: {explainer.expected_value}")

# === 7. Chargement des données globales ===
GLOBAL_DATA_PATH = "../backend/data/test_2000_sample_for_api.csv"  # Chemin corrigé
try:
    df_global = pd.read_csv(GLOBAL_DATA_PATH)[features]
    df_global_scaled = scaler.transform(df_global)  # Doit être un array numpy 2D
    assert df_global_scaled.shape[1] == len(features), "Incohérence features/scaler !"
    logger.info("Données globales chargées et prétraitées.")
except Exception as e:
    logger.critical(f"Erreur de chargement ou traitement des données globales : {e}")
    raise RuntimeError("Échec de la préparation des données globales.")

# === 8. Calcul SHAP global dès le démarrage ===
try:
    global_shap_values = explainer.shap_values(df_global_scaled)
    if isinstance(global_shap_values, list):  # Cas classification binaire
        global_shap_values = global_shap_values[
            1
        ]  # Garder seulement la classe positive

    # === VÉRIFICATION PRIMAIRE ===
    assert (
        len(df_global) == global_shap_values.shape[0]
    ), f"Données/SHAP incohérents ({len(df_global)} vs {global_shap_values.shape[0]})"

    global_shap_mean = global_shap_values.mean(axis=0)
except Exception as e:
    logger.critical(f"Erreur calcul SHAP global : {str(e)}")
    raise RuntimeError("Impossible de calculer les SHAP globaux")

# === VÉRIFICATION REDONDANTE POUR SÉCURITÉ ===
global_shap_matrix = global_shap_values
assert len(df_global) == len(
    global_shap_matrix
), f"Données/SHAP incohérents ({len(df_global)} vs {len(global_shap_matrix)})"

# Après le précalcul
# print(f"Type SHAP global : {type(global_shap_values)}")
# print(f"Shape SHAP global : {global_shap_values.shape}")

# === 9. Liste des clients ===
try:
    full_df = pd.read_csv(GLOBAL_DATA_PATH)
    client_ids = full_df["SK_ID_CURR"].dropna().astype(int).unique().tolist()
    logger.info(f"Chargement de {len(client_ids)} clients réussis")
except Exception as e:
    logger.critical(f"Échec du chargement des données : {str(e)}")
    raise RuntimeError("Impossible de démarrer l'API - données corrompues")


# ==== All data ====


@app.get("/get_test_data")
@cache(expire=3600)
async def get_test_data(_: str = Depends(validate_api_key)):
    """Renvoie les données de test"""
    try:
        if full_df.empty:
            raise HTTPException(status_code=404, detail="Aucune donnée disponible")

        return full_df.to_dict(orient="records")

    except Exception as e:
        logger.error(f"Erreur get_test_data : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur de chargement des données")


# === stat de polulation


@app.post("/population_stats")
def get_population_stats(feature_request: dict):
    """
    Calcule les statistiques de population pour une feature donnée
    Format attendu :
    {
        "feature": "DAYS_BIRTH",
        "filters": {"CODE_GENDER": 1},
        "sample_size": 1000
    }
    """
    try:
        # Validation des entrées
        feature = feature_request.get("feature")
        if feature not in full_df.columns:
            raise HTTPException(status_code=400, detail="Feature invalide")

        # Application des filtres
        filtered_df = full_df.copy()
        for f, v in feature_request.get("filters", {}).items():
            if f in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[f] == v]

        # Échantillonnage
        sample_size = min(feature_request.get("sample_size", 1000), len(filtered_df))
        sample_df = (
            filtered_df.sample(sample_size, random_state=42)
            if not filtered_df.empty
            else pd.DataFrame()
        )

        # Calcul des statistiques
        stats = {
            "mean": float(sample_df[feature].mean()),
            "std": float(sample_df[feature].std()),
            "min": float(sample_df[feature].min()),
            "25%": float(sample_df[feature].quantile(0.25)),
            "50%": float(sample_df[feature].quantile(0.5)),
            "75%": float(sample_df[feature].quantile(0.75)),
            "max": float(sample_df[feature].max()),
        }

        return {"feature": feature, "stats": stats}

    except Exception as e:
        logger.error(f"Erreur population_stats : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ IDS Client =====================


@app.get("/client_ids")
def get_client_ids(limit: int = 2000):
    """Renvoie la liste des IDs clients"""
    try:
        if not client_ids:
            logger.warning("Aucun client trouvé dans les données")
            raise HTTPException(status_code=404, detail="Aucun client disponible")

        return {"client_ids": client_ids[:limit]}

    except Exception as e:
        logger.error(f"Erreur /client_ids : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")


# ============= info Client =============================


@app.get("/client_info/{client_id}", response_model=dict)
def get_client_info(client_id: int):
    """
    Retourne les informations détaillées d'un client
    - **client_id** : Identifiant unique du client (ex: 100001)
    """
    try:
        client_data = full_df.loc[full_df["SK_ID_CURR"] == client_id].copy()

        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client introuvable")

        return client_data.iloc[0].to_dict()

    except Exception as e:
        logger.error(f"Erreur client {client_id} : {str(e)}")
        raise HTTPException(
            status_code=500, detail="Erreur technique - contactez l'administrateur"
        )


# ===========


class ClientData(BaseModel):
    data: dict  # exemple : {"EXT_SOURCE_1": 0.12, "AMT_CREDIT": 350000, ...}


class BatchPredictionRequest(BaseModel):
    data: List[dict]


# ====== probabilité =====


@app.post("/predict")
def predict(client: ClientData):
    """
    Prédit la probabilité de défaut d'un client et donne une décision (Accepté/Refusé).

    Les données client sont transformées, scalées et passées au modèle pour obtenir la probabilité.
    Une décision est ensuite prise selon un seuil optimisé (threshold).

    Args:
        client (ClientData): dictionnaire contenant les valeurs des features du client.

    Returns:
        dict: {
            "probability": probabilité de défaut (en pourcentage),
            "decision": "✅ Accepté" ou "❌ Refusé"
        }
    """
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
        raise HTTPException(
            status_code=400, detail=f"Erreur dans la prédiction : {str(e)}"
        )


# ==== batch de clients ====


@app.post("/predict_batch")
def predict_batch(batch: BatchPredictionRequest):
    try:
        # Transformer en DataFrame
        df = pd.DataFrame(batch.data)

        # Sélection des colonnes dans le bon ordre
        X = df[features]

        # Scaling
        X_scaled = scaler.transform(X)

        # Prédictions
        probs = model.predict_proba(X_scaled)[:, 1]
        decisions = ["✅ Accepté" if p < threshold else "❌ Refusé" for p in probs]

        results = [
            {
                "id": row.get("SK_ID_CURR", None),
                "probability": round(p * 100, 2),
                "decision": d,
            }
            for row, p, d in zip(batch.data, probs, decisions)
        ]

        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur batch : {str(e)}")


# ======== Calcule les 10 SHAP les plus influents pour un client donné via une API POST.


@app.post("/explain")
def explain(client: ClientData):
    """
    Fournit une explication locale des prédictions du modèle pour un client donné.

    Utilise SHAP pour identifier les 10 features les plus influentes dans la décision du modèle,
    en indiquant leur valeur d’impact (positive ou négative).

    Args:
        client (ClientData): dictionnaire contenant les valeurs des features du client.

    Returns:
        dict: {
            "explanation": [
                {
                    "feature": nom de la feature,
                    "shap_value": valeur SHAP (positive ou négative)
                },
                ...
            ]
        }
    """

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
        explanation = [
            {"feature": f, "shap_value": round(val, 4)} for f, val in top_features
        ]

        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Erreur dans l'explication : {str(e)}"
        )


# ===== shap client


@app.get("/client_shap_data/{client_id}")
def get_client_shap_data(client_id: int):
    """Renvoie les données brutes (non-scalées) nécessaires au calcul SHAP pour un client donné"""
    try:
        client_data = full_df[full_df["SK_ID_CURR"] == client_id][features]
        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client introuvable")

        return client_data.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Erreur client_shap_data : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur technique")


# =======
@app.get("/global_shap_sample")
@cache(expire=3600)
async def get_global_shap_sample(sample_size: int = 1000):
    """Renvoie un échantillon brut des données globales (non-scalées) pour usage
    frontend ou calculs.filtré avec random_state et paramètre sample_size.
    """
    try:
        sample = df_global.sample(min(sample_size, len(df_global)), random_state=42)
        return sample.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============


@app.get("/shap/local/{client_id}")
async def get_local_shap(client_id: int):
    """Calcule les valeurs SHAP locales pour un client spécifique basé sur l'ID"""
    try:
        # Récupération des données client
        client_data = full_df[full_df["SK_ID_CURR"] == client_id][features]
        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client introuvable")

        # Prétraitement des données
        X_scaled = scaler.transform(client_data.values.reshape(1, -1))

        # Calcul SHAP
        shap_values = explainer.shap_values(X_scaled)
        base_value = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, list)
            else explainer.expected_value
        )

        return {
            "values": shap_values[0].tolist(),
            "base_value": float(base_value),
            "features": features,
            "client_data": client_data.iloc[0].to_dict(),
            "feature_names": list(features),
        }

    except Exception as e:
        logger.error(f"Erreur SHAP locale : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================


@app.post("/shap/local")
def get_local_shap(client: ClientData):
    """Calcule les valeurs SHAP locales
    Même but que /explain, mais retourne aussi base_value, values et features.
    """
    try:
        # Validation des données d'entrée
        missing_features = [feat for feat in features if feat not in client.data]
        if missing_features:
            raise ValueError(f"Features manquantes : {missing_features}")

        # Transformation des données
        X = pd.DataFrame([client.data])[features]
        X_scaled = scaler.transform(X)

        # Calcul SHAP avec gestion multi-classe
        shap_values = explainer.shap_values(X_scaled)
        if isinstance(explainer.expected_value, list):
            base_value = float(explainer.expected_value[1])  # Classification binaire
        else:
            base_value = float(explainer.expected_value)  # Régression

        return {
            "base_value": base_value,
            "values": shap_values[0].tolist(),
            "features": features,
            "explanation": [
                {"feature": f, "shap_value": round(float(v), 4)}
                for f, v in sorted(
                    zip(features, shap_values[0]), key=lambda x: abs(x[1]), reverse=True
                )[:10]
            ],
        }

    except Exception as e:
        logger.error(f"Erreur SHAP locale : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur de calcul SHAP")

    except ValueError as e:
        logger.warning(f"Données invalides : {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur SHAP locale : {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur de calcul SHAP")


# ====================
@cache(expire=3600)  # Cache 1 heure
@app.get("/global_features")
def get_global_features(top_n: int = 10):
    """Renvoie les features ayant le plus grand impact moyen (directionnel) sur les prédictions du modèle.

    Le score moyen est calculé à partir des valeurs SHAP sur l'ensemble global des données.
    Chaque feature est associée à :
    - son impact moyen (positif ou négatif)
    - sa direction d'influence (positive = favorise l'acceptation, négative = favorise le refus)

    Args:
        top_n (int): nombre de features à retourner (classées par importance décroissante)

    Returns:
        dict: {
            "global_importances": [
                {
                    "feature": nom de la feature,
                    "impact": valeur moyenne (signée) de SHAP,
                    "direction": "positive" ou "negative"
                },
                ...
            ]
        }"""
    try:
        features_sorted = sorted(
            zip(features, global_shap_mean), key=lambda x: abs(x[1]), reverse=True
        )[:top_n]

        return {
            "global_importances": [
                {
                    "feature": feat,
                    "impact": round(float(impact), 4),
                    "direction": "positive" if impact > 0 else "negative",
                }
                for feat, impact in features_sorted
            ]
        }
    except Exception as e:
        logger.error(f"Erreur /global_features : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne lors du calcul des SHAP globaux : {str(e)}",
        )


# ============Au démarrage ==================
global_shap_matrix = global_shap_values  # Déjà calculé


@app.get("/global_shap_matrix")
@cache(expire=3600)
def get_global_shap_matrix(
    sample_size: int = 1000,
    random_state: int = 42,
    long_format: Optional[bool] = Query(
        False, description="Retourne un format long si True"
    ),
):
    """SHAP + features en format large (via long_format=False)
    ou en format long (via une fonction get_shap_long_dataframe)
    visualisations SHAP globales type summary plot, bar, beeswarm.
    """
    try:
        sample_size = min(sample_size, len(global_shap_matrix))
        rng = np.random.default_rng(seed=random_state)
        sample_idx = rng.choice(len(global_shap_matrix), sample_size, replace=False)

        shap_sample = global_shap_matrix[sample_idx]
        feature_sample = df_global.iloc[sample_idx]

        if long_format:
            df_long = get_shap_long_dataframe(shap_sample, feature_sample[features])
            return df_long.to_dict(orient="records")

        return {
            "shap_values": shap_sample.tolist(),
            "feature_values": feature_sample.to_dict(orient="records"),
            "features": features,
            "base_value": float(
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value
            ),
        }
    except Exception as e:
        logger.error(f"Erreur globale_shap_matrix : {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Erreur de traitement des données globales"
        )


# ===== control =======
@cache(expire=30)  # Cache 30 secondes
async def get_redis_memory_usage(redis_client: Redis) -> str:
    """Retourne l'utilisation mémoire de Redis avec gestion d'erreur"""
    try:
        info = await redis_client.info("memory")
        return f"{int(info['used_memory'])/1e6:.2f} MB"
    except Exception as e:
        logger.error(f"Erreur mémoire Redis : {str(e)}", exc_info=True)
        return "N/A"


# === health ==========
#
#       control et memory
#
# ===== control =======
@cache(expire=30)
async def get_redis_memory_usage(redis_client):
    """Récupère l'utilisation mémoire de Redis"""
    try:
        info = await redis_client.info("memory")
        return f"{int(info['used_memory']) / 1e6:.2f} MB"
    except Exception as e:
        logger.error(f"Erreur mémoire Redis : {str(e)}")
        return "N/A"


# ===== health check endpoint =====
@app.get("/health")
async def health_check():
    """Endpoint de santé complet avec monitoring Redis"""
    checks = {
        "status": "API opérationnelle 🚀",
        "redis": {
            "status": "active" if redis_client else "inactive",
            "cache_type": "redis" if redis_client else "memory",
            "memory_used": "N/A",
        },
        "model": {
            "loaded": model is not None,
            "type": type(model).__name__ if model else None,
        },
        "scaler": {
            "loaded": scaler is not None,
            "type": type(scaler).__name__ if scaler else None,
        },
        "features": {
            "loaded": isinstance(features, list) and len(features) > 0,
            "count": len(features) if features else 0,
        },
        "threshold": threshold if isinstance(threshold, float) else None,
        "shap": {
            "values_ready": isinstance(global_shap_matrix, np.ndarray),
            "mean_ready": isinstance(global_shap_mean, np.ndarray),
        },
        "data": {
            "global_ready": isinstance(df_global, pd.DataFrame) and not df_global.empty,
            "clients_loaded": len(client_ids),
            "sample_size": (
                df_global.shape[0] if isinstance(df_global, pd.DataFrame) else 0
            ),
        },
    }

    try:
        # Vérification Redis seulement si le client est initialisé
        if redis_client:
            checks["redis"]["status"] = "active"
            try:
                checks["redis"]["memory_used"] = await get_redis_memory_usage(
                    redis_client
                )
                if await redis_client.ping() != "PONG":
                    checks["redis"]["status"] = "unstable"
            except Exception as e:
                checks["redis"]["status"] = f"error: {str(e)}"
                logger.error(f"Erreur Redis: {str(e)}")

    except Exception as e:
        logger.error(f"Erreur globale: {str(e)}")
        checks["redis"]["status"] = "error"

    # Détermination du statut global
    critical_services = [
        checks["model"]["loaded"],
        checks["scaler"]["loaded"],
        checks["features"]["loaded"],
        checks["data"]["global_ready"],
        checks["redis"]["status"]
        in ["active", "unstable"],  # Considère Redis comme non critique
    ]

    if not all(critical_services):
        checks["status"] = "🔴 API non opérationnelle"
    elif not all(
        [
            checks["shap"]["values_ready"],
            checks["shap"]["mean_ready"],
            checks["data"]["clients_loaded"] > 0,
        ]
    ):
        checks["status"] = "🟡 API partiellement opérationnelle"

    return checks
