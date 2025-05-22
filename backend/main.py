# === 0. Imports ===
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Query, Depends, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import time
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
from contextlib import asynccontextmanager
from redis.exceptions import TimeoutError, ConnectionError, RedisError
from redis.backoff import ExponentialBackoff

# Cache et Redis
from fastapi_cache.decorator import cache
from redis.asyncio import Redis
from redis.asyncio.retry import Retry
from fastapi_cache import FastAPICache, coder as _fastapi_coder
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.backends.inmemory import InMemoryBackend
from functools import lru_cache

# from cachetools import TTLCache, cached
from collections.abc import AsyncIterator

import re
from redis import asyncio as aioredis
import json

# === 1. Chargement des variables d'environnement ===
# load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
env_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
load_dotenv(dotenv_path=env_path)

# === 2. Désactivation des warnings inutiles ===
warnings.filterwarnings("ignore", category=UserWarning)


# === 3. Logger global ===


# — filtre global pour redaction —
class RedisCredentialFilter(logging.Filter):
    def filter(self, record):
        record.msg = re.sub(
            r"rediss://default:[^@]+@", "rediss://default:****@", str(record.msg)
        )
        return True


logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
logger.addFilter(RedisCredentialFilter())
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)


async def warmup_redis(client: Redis, count: int = 5):
    async with client.pipeline(transaction=True) as pipe:
        for i in range(count):
            pipe.ping()
        await pipe.execute()
        logger.info(f"✅ Pool Redis préchauffé avec {count} connexions")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === Startup ===
    # 1) Initialise Redis
    client = await init_redis_client()
    await warmup_redis(client, count=2)
    app.state.redis_client = client
    if client:
        FastAPICache.init(RedisBackend(client), prefix="demo-cache")
        logger.info("Cache logging activé")
    else:
        FastAPICache.init(InMemoryBackend(), prefix="memory-only")
        logger.warning("🛑 Redis non disponible, utilisant un cache en mémoire.")

    # 2) Génère et injecte l'exemple aléatoire pour /predict
    try:
        example_id, example_data = random_client_example()  # renvoie juste le dict data
        ClientData.model_config["json_schema_extra"]["example"] = {"data": example_data}
        logger.info("✅ Exemple dynamique injecté dans ClientData")
        app.state.example_id = example_id

    except Exception as e:
        logger.warning(f"⚠️ Impossible d'injecter l'exemple dynamique : {e}")

    yield  # Fin du startup, début du runtime
    # === Shutdown ===
    if client:
        try:
            await client.close()
            logger.info("🧹 Connexion Redis fermée proprement")
        except Exception as e:
            logger.warning(f"⚠️ Erreur à la fermeture Redis : {e}")
    else:
        logger.info("ℹ️ Aucun client Redis à fermer")


# === 4. FastAPI app ===
app = FastAPI(lifespan=lifespan)

# === 5. Middleware CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "x-api-key"],
)

# === 6. Clé API ===
api_key_header = APIKeyHeader(name="x-api-key")
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")

# === 7. global ===
LATENCY_THRESHOLD_MS = 200  # seuil acceptable
REDIS_MEMORY_WARNING = 200  # Mo


# === 8. Validation de la clé API ===
async def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Clé API invalide")
    return api_key


# === 9. Initialisation dynamique de Redis ===


async def init_redis_client() -> Redis | None:
    logger.info("🧪 Appel de init_redis_client()")
    try:
        upstash_url = os.getenv("UPSTASH_REDIS_URL", "")
        upstash_token = os.getenv("UPSTASH_REDIS_TOKEN", "")

        if upstash_url and upstash_token:
            # Nettoyer et normaliser le host:port
            host = upstash_url.replace("https://", "").replace("http://", "")
            if ":" not in host:
                host = f"{host}:6379"

            redis_url = f"rediss://default:{upstash_token}@{host}"
            logger.info(f"Upstash host: {host}")  # 👈 safe
            logger.info(f"→ Upstash URI: {redis_url}")  # 🔒 sera automatiquement filtré

            client = Redis.from_url(
                redis_url,
                decode_responses=False,
                max_connections=20,
                health_check_interval=30,
                retry=Retry(ExponentialBackoff(), 3),
                retry_on_timeout=True,
                socket_connect_timeout=3,  # Timeout de connexion réduit
                socket_timeout=10,
                socket_keepalive=True,
            )
            logger.info("🔌 Connexion à Redis via Upstash")
        else:
            client = Redis.from_url("redis://localhost:6379", decode_responses=False)
            logger.info("🖥️ Connexion à Redis local")

        pong = await client.ping()
        if pong is not True:
            raise ConnectionError("Réponse Redis invalide")

        logger.info("✅ Redis configuré avec succès")
        return client

    except Exception as e:
        logger.error(f"⚠️ Cache mémoire activé - Erreur Redis : {e}")
        FastAPICache.init(InMemoryBackend())
        return None


# === 10 Configuration Redis et Cache ===


async def get_healthy_client(request: Request) -> Redis | None:
    client = request.app.state.redis_client
    if not client:
        return None

    try:
        await client.ping()
        return client
    except (TimeoutError, ConnectionError, RedisError) as e:
        logger.warning(f"Redis error: {e} → Reconnexion...")
        try:
            await client.aclose()  # Fermeture asynchrone
        except Exception as e:
            logger.error(f"Fermeture client échouée: {e}")

        try:
            new_client = await init_redis_client()  # Nouvelle tentative
            request.app.state.redis_client = new_client
            return new_client
        except Exception as e:
            logger.critical(f"Échec reconnexion Redis: {e}")
            return None


# === 11. Logger de cache (optionnel) ===
async def get_redis(request: Request):
    client = request.app.state.redis_client
    if not client:
        raise HTTPException(503, "Redis non initialisé")
    return client


# === 12. LoggingRedisBackend ===
class LoggingRedisBackend(RedisBackend):
    async def set(self, key: str, value: str, expire: int):
        logger.info(
            f"SET CACHE: {key} (TTL: {expire}s)"
        )  # Log lors de la mise en cache
        return await super().set(key, value, expire)


# === 13. Script de rotation (à exécuter périodiquement)  ===

import requests


def rotate_redis_token():
    headers = {"Authorization": f"Bearer {CURRENT_TOKEN}"}
    response = requests.post(
        "https://api.upstash.com/v2/tokens/rotate", headers=headers
    )
    new_token = response.json()["new_token"]
    update_env_file("UPSTASH_REDIS_TOKEN", new_token)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === 14. Chargement des artefacts ===
# ARTIFACT_PATH = "models/lightgbm_production_artifact_20250415_081218.pkl"
ARTIFACT_PATH = os.path.join(
    BASE_DIR, "models", "lightgbm_production_artifact_20250415_081218.pkl"
)

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

# === 15. Initialisation SHAP ===
explainer = shap.Explainer(model)
logger.info(f"Type de expected_value: {type(explainer.expected_value)}")
logger.info(f"Valeur de expected_value: {explainer.expected_value}")

# === 16. Chargement des données globales ===
# GLOBAL_DATA_PATH = "data/test_2000_sample_for_api.csv"
GLOBAL_DATA_PATH = os.path.join(BASE_DIR, "data", "test_2000_sample_for_api.csv")

try:
    df_global = pd.read_csv(GLOBAL_DATA_PATH)[features]
    df_global_scaled = scaler.transform(df_global)  # Doit être un array numpy 2D
    assert df_global_scaled.shape[1] == len(features), "Incohérence features/scaler !"
    logger.info("Données globales chargées et prétraitées.")
except Exception as e:
    logger.critical(f"Erreur de chargement ou traitement des données globales : {e}")
    raise RuntimeError("Échec de la préparation des données globales.")

# === 17. Calcul SHAP global dès le démarrage ===
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


# === 18. Liste des clients ===
try:
    full_df = pd.read_csv(GLOBAL_DATA_PATH)
    client_ids = full_df["SK_ID_CURR"].dropna().astype(int).unique().tolist()
    logger.info(f"Chargement de {len(client_ids)} clients réussis")
except Exception as e:
    logger.critical(f"Échec du chargement des données : {str(e)}")
    raise RuntimeError("Impossible de démarrer l'API - données corrompues")


# ==== test ==== ===============================================================


@app.get("/", include_in_schema=False)
async def read_root():
    return {"message": "Bienvenue sur l'API !"}


@app.get("/cache-example", include_in_schema=False)
@cache(expire=30)
async def cache_example():
    logger.info("Cache utilisé")  # Log lors de l'appel de la fonction
    return {"message": "Ce message est mis en cache pendant 60 secondes"}


@app.get("/test-redis", include_in_schema=False)
async def test_redis():
    redis_client = await app.state.redis_client
    if redis_client is None:
        return {"status": "❌ Redis non initialisé"}
    try:
        pong = await redis_client.ping()
        return {"status": f"✅ Redis OK: {pong}"}
    except Exception as e:
        return {"status": f"❌ Erreur Redis: {str(e)}"}


# ========== all data ===============================


@app.get("/get_test_data", include_in_schema=False)
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


# ============ IDS Client ================================


@app.get("/client_ids", include_in_schema=False)
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


# ====== basemodel =======================================


class ClientData(BaseModel):
    data: dict

    model_config = {
        "json_schema_extra": {"example": {}}  # on va remplacer ça au startup
    }


# fonction “random” (pas de route, juste un utilitaire)
def random_client_example() -> dict:
    if full_df.empty:
        raise RuntimeError("full_df vide")

    row = full_df.sample(1).iloc[0].to_dict()
    client_id = row.pop("SK_ID_CURR", None)
    if client_id is None:
        raise RuntimeError("SK_ID_CURR manquant")

    # On retourne juste le champ `data` car on ne veut préremplir que `data`
    example_id = int(client_id)
    return example_id, row


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


@app.get("/client_shap_data/{client_id}", include_in_schema=False)
def get_client_shap_data(client_id: int):
    # 1) on réutilise get_client_info pour lever 404 si besoin
    info = get_client_info(client_id)
    # 2) on filtre seulement les colonnes du modèle
    try:
        return {feat: info[feat] for feat in features}
    except KeyError as e:
        # au cas où un feature manquerait (devrait rarement arriver)
        raise HTTPException(
            status_code=500, detail=f"Problème d’accès à la feature {e.args[0]}"
        )


# === stat de polulation =====================


@app.post("/population_stats")
def get_population_stats(feature_request: dict):
    """
    Calcule les statistiques de population pour une feature donnée
    Format attendu :
    {
    "feature": "AMT_CREDIT",
    "filters": {
        "CODE_GENDER": 0,
        "FLAG_OWN_CAR": 1
    },
    "sample_size": 1000
    }

    - Combinaison de filtres (femmes propriétaires de voiture)

    - Taille d'échantillon max
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


class BatchPredictionRequest(BaseModel):
    data: List[dict]


@app.post("/predict_batch", include_in_schema=False)
def predict_batch(batch: BatchPredictionRequest):
    """
    Traite un lot de données clients et renvoie les probabilités de défaut
    ainsi que la décision pour chaque client.

    Cette route reçoit en entrée une liste de dictionnaires représentant
    plusieurs clients (`batch.data`), applique le scaler et le modèle pour
    prédire la probabilité de défaut, et formate une décision binaire
    selon le seuil optimisé.

    Args:
        batch (BatchPredictionRequest): Objet Pydantic contenant :
            - data (List[Dict]): liste des enregistrements clients bruts,
            chacun devant contenir la clé "SK_ID_CURR" et les features
            nécessaires au modèle.

    Returns:
        List[Dict]: Pour chaque client fourni :
            - "id" (int | None): l’identifiant client (ou None s’il est absent)
            - "probability" (float): probabilité de défaut en pourcentage
            - "decision" (str): "✅ Accepté" si probabilité < threshold,
            sinon "❌ Refusé"

    Raises:
        HTTPException(400): Si une erreur se produit pendant la transformation
                            ou la prédiction (format des données, dimensions, etc.).
    """
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


# ======= échantillon de données ==============


@app.get("/global_shap_sample", include_in_schema=False)
@cache(expire=3600)
async def get_global_shap_sample(sample_size: int = 1000):
    """
    Récupère un échantillon des données globales non-scalées.

    Cette route renvoie un sous-ensemble aléatoire des enregistrements
    du jeu de données global (avant mise à l’échelle), destiné à être
    utilisé côté frontend pour générer des plots ou faire des calculs.

    Args:
        sample_size (int, optional): Nombre maximal de lignes à retourner.
                                    Si la population contient moins de lignes,
                                    on renvoie tout le DataFrame. (par défaut 1000)

    Returns:
        List[dict]: Liste de dictionnaires, un par enregistrement,
                    conforme à `df_global.to_dict(orient="records")`.

    Raises:
        HTTPException(500): En cas d’erreur lors de l’échantillonnage.
    """
    try:
        sample = df_global.sample(min(sample_size, len(df_global)), random_state=42)
        return sample.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========== les valeurs SHAP pour un jeu de données client avec ID


@app.get("/shap/local/{client_id}")
async def get_local_shap(client_id: int):
    """
    Calcule et renvoie les valeurs SHAP pour un client existant.

    Args:
        client_id (int): Identifiant du client.

    Returns:
        dict: {
            "values": List[float],        # Valeurs SHAP pour chaque feature
            "base_value": float,         # Valeur de base (expected_value)
            "features": List[str],       # Liste des noms de features
            "feature_names": List[str],  # (même que `features`)
            "client_data": dict          # Données brutes du client
        }

    Raises:
        HTTPException(404): Si l’ID n’existe pas.
        HTTPException(500): En cas d’erreur interne.
    """
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


# ==========les valeurs SHAP pour un jeu de données client arbitraire


@app.post("/shap/local")
def get_local_shap_by_payload(client: ClientData):
    """
    Calcule les valeurs SHAP pour un jeu de données client arbitraire
    et renvoie un résumé des 10 features les plus influentes.

    Args:
        client (ClientData): Pydantic model contenant `"data": {feature: valeur,…}`.

    Returns:
        dict: {
            "base_value": float,      # Valeur de base (expected_value)
            "values": List[float],    # Valeurs SHAP brutes
            "features": List[str],    # Liste des noms de features
            "explanation": List[{     # Top-10 features classées par contribution
                "feature": str,
                "shap_value": float
            }]
        }

    Raises:
        HTTPException(400): Si des features manquent dans l’entrée.
        HTTPException(500): En cas d’erreur interne.
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
@app.get("/global_features", include_in_schema=False)
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


@app.get("/global_shap_matrix", include_in_schema=False)
@cache(expire=3600)
def get_global_shap_matrix(
    sample_size: int = 1000,
    random_state: int = 42,
    long_format: Optional[bool] = Query(
        False, description="Retourne un format long si True"
    ),
):
    """
    Renvoie un échantillon des valeurs SHAP globales et des données associées.

    Cette route permet d’extraire, pour visualisation (summary_plot, bar chart, beeswarm, etc.),
    un sous-ensemble aléatoire des contributions SHAP calculées sur l’ensemble global des données.

    Args:
        sample_size (int, optional): Nombre maximal de lignes à renvoyer (par défaut 1000).
        random_state (int, optional): Graine pour la reproductibilité de l’échantillonnage.
        long_format (bool, optional):
            - False (par défaut) : format “large” renvoyant un JSON
            contenant :
                - shap_values: List[List[float]]
                - feature_values: List[dict]
                - features: List[str]
                - base_value: float
            - True : format “long” renvoyé sous forme de liste d’enregistrements
            (un dict par valeur SHAP), pour faciliter certains traitements.

    Returns:
        dict | List[dict]:
        - Si `long_format=False` : un dict regroupant :
            • "shap_values",
            • "feature_values",
            • "features",
            • "base_value".
        - Si `long_format=True` : une liste de dicts (format long).

    Raises:
        HTTPException(500): En cas d’erreur interne lors de l’échantillonnage ou de la mise en forme.
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


# ==== Monkey-patche ===================================================
from fastapi.openapi.utils import get_openapi


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    # génère le schéma de base
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    # récupère l’ID d’exemple stocké
    example_id = getattr(app.state, "example_id", None)
    if example_id is not None:
        # liste des routes à mettre à jour
        for route in [
            "/client_info/{client_id}",
            "/shap/local/{client_id}",
            "/client_shap_data/{client_id}",
        ]:
            # récupère le bloc "get" pour cette route
            if route in schema["paths"]:
                op = schema["paths"][route].get("get")
                if op and "parameters" in op:
                    for param in op["parameters"]:
                        if param.get("name") == "client_id":
                            param["example"] = example_id
                            break
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi

# === health ==========
#
#       control et memory
#
# ===== control =======


# ===== control =======
# un cache TTL en mémoire : max 1 entrée, expire au bout de 30 s
# En haut de ton module
async def _raw_redis_memory_usage(redis_client: Redis) -> float:
    try:
        info = await redis_client.info("memory")
        used = info.get("used_memory", 0)

        # Gestion unifiée des types
        if isinstance(used, bytes):
            used = used.decode("utf-8")
        used = int(used)

        return used / 1e6
    except Exception as e:
        logger.error(f"Error in memory check: {str(e)}")
        return -1.0


@cache(expire=30)
async def get_redis_memory_usage(redis_client: Redis) -> float:
    """Retourne l'utilisation mémoire en MB (ou -1.0 si erreur)."""
    return await _raw_redis_memory_usage(redis_client)


# ===== health check endpoint =====


@app.get("/health", include_in_schema=False)
async def health_check(
    redis_client=Depends(get_redis), response: Response = Response()
):
    """Endpoint de santé complet avec monitoring Redis"""
    response.headers["Cache-Control"] = "no-store, max-age=0"
    checks = {
        "status": "API opérationnelle 🚀",
        "redis": {
            "status": "inactive",
            "cache_type": "redis" if redis_client else "memory",
            "memory_used": "N/A",
            "latency_ms": None,
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

    # Vérification Redis avec ping et latence
    if redis_client:
        try:
            t0 = time.perf_counter()
            pong = await redis_client.ping()
            latency = (time.perf_counter() - t0) * 1000
            latency_ms = round(latency, 2)

            # Déterminer le statut
            if pong and latency_ms < LATENCY_THRESHOLD_MS:
                checks["redis"]["status"] = "active"
            elif pong:
                checks["redis"]["status"] = "unstable"
                logger.warning(f"⚠️ Latence Redis élevée : {latency_ms:.2f} ms")
            else:
                checks["redis"]["status"] = "unstable"
                logger.warning("⚠️ Redis a répondu mais sans 'pong' explicite")

            checks["redis"]["latency_ms"] = latency_ms
            # checks["redis"]["memory_used"] = await get_redis_memory_usage(redis_client)
            used_mb = await get_redis_memory_usage(redis_client)
            checks["redis"]["memory_used"] = (
                f"{used_mb:.2f} MB" if used_mb >= 0 else "N/A"
            )

        except Exception as e:
            checks["redis"]["status"] = f"error: {str(e)}"
            logger.error(f"❌ Erreur Redis : {str(e)}")

    # Détermination du statut global
    critical_services = [
        checks["model"]["loaded"],
        checks["scaler"]["loaded"],
        checks["features"]["loaded"],
        checks["data"]["global_ready"],
        checks["redis"]["status"] in ["active", "unstable"],  # Redis non critique
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

    code = 200 if checks["redis"]["status"] in ("active", "unstable") else 503
    return JSONResponse(checks, status_code=code)
