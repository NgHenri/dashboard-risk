# =======================================================================
#   test de déclaration globale explicite :
#   Dans la fonction startup(), utilisez global redis_client avant de
#   lui assigner une valeur.
#   Ceci est indispensable pour modifier la variable globale la fonction.
#   La fonction init_redis_client() retourne correctement None en cas d'échec,
#   et startup() gère ce cas en initialisant un cache en mémoire.
#   Redis local

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.requests import Request
from starlette.responses import Response


from redis import asyncio as aioredis

# ======================================================================
import os
import sys
import time, logging
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from fastapi import HTTPException
from redis.asyncio import Redis

# from upstash_redis.asyncio import Redis
from pydantic import BaseModel
from redis import asyncio as aioredis
from dotenv import load_dotenv


env_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
load_dotenv(dotenv_path=env_path)
# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

redis_client: Redis | None = None


async def init_redis_client() -> "Redis | None":
    logger.info("🧪 Appel de init_redis_client()")
    try:
        # Test des variables d'environnement pour Upstash
        upstash_url = ""  # os.getenv("UPSTASH_REDIS_URL")
        upstash_token = ""  # os.getenv("UPSTASH_REDIS_TOKEN")

        if upstash_url and upstash_token:
            # 🔐 Construction de l’URL Upstash au format URI sécurisé
            redis_url = f"rediss://default:{upstash_token}@{upstash_url}"
            client = Redis.from_url(redis_url)  # Utilise 'from_url' pour initialiser
            logger.info("🔌 Connexion à Redis via Upstash")
        else:
            from redis.asyncio import Redis as LocalRedis

            client = LocalRedis.from_url("redis://localhost:6379")  # Connexion locale
            logger.info("🖥️ Connexion à Redis local")

        # Test de connexion
        pong = await client.ping()
        if pong != True:
            raise ConnectionError("Réponse Redis invalide")

        logger.info("✅ Redis configuré avec succès")
        return client

    except Exception as e:
        logger.error(f"⚠️ Cache mémoire activé - Erreur Redis : {e}")
        FastAPICache.init(InMemoryBackend())
        return None


class LoggingRedisBackend(RedisBackend):
    async def set(self, key: str, value: str, expire: int):
        logger.info(
            f"SET CACHE: {key} (TTL: {expire}s)"
        )  # Log lors de la mise en cache
        return await super().set(key, value, expire)


# Appel de la fonction pour tester
@app.on_event("startup")
async def startup():
    global redis_client  # Déclare l'utilisation de la variable globale
    client = await init_redis_client()  # Récupère le client de la fonction
    redis_client = client  # Assigne à la variable globale
    if redis_client:
        # Initialisation du cache avec LoggingRedisBackend
        FastAPICache.init(LoggingRedisBackend(redis_client), prefix="demo-cache")
        logger.info("Cache logging activé")
    else:
        # Fallback en cas d'erreur de connexion
        logger.warning("🛑 Redis non disponible, utilisant un cache en mémoire.")
        FastAPICache.init(InMemoryBackend(), prefix="demo-cache")


@app.get("/cache-test")
@cache(expire=60)
async def cache_test():
    # retourne un horodatage pour voir si le cache fonctionne
    return {"time": time.time()}


@app.get("/items/{item_id}")
@cache(expire=60)
async def read_item(item_id: int):
    global redis_client
    if redis_client is None:
        logger.error("❌ Redis non initialisé")
        raise HTTPException(status_code=500, detail="Redis indisponible")

    try:
        # Génération des données si non en cache
        item_data = f"Item data for {item_id}"
        return {"item_id": item_id, "data": item_data}

    except Exception as e:
        logger.error(f"Erreur lors de l'accès à Redis : {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur de serveur")
