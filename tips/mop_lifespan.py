from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.requests import Request
from starlette.responses import Response


from redis import asyncio as aioredis

# ======================================================================
import socket

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
from redis.asyncio.retry import Retry

# from upstash_redis.asyncio import Redis
from pydantic import BaseModel
from redis import asyncio as aioredis
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from redis.exceptions import TimeoutError, ConnectionError, RedisError
from redis.backoff import ExponentialBackoff

env_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
load_dotenv(dotenv_path=env_path)
# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = await init_redis_client()
    app.state.redis_client = client
    if client:
        FastAPICache.init(RedisBackend(client), prefix="demo-cache")
    else:
        FastAPICache.init(InMemoryBackend(), prefix="demo-cache")
    yield
    # Shutdown
    if client:
        try:
            await client.close()
            logger.info("🧹 Connexion Redis fermée proprement")
        except Exception as e:
            logger.warning(f"⚠️ Erreur à la fermeture Redis : {e}")
    else:
        logger.info("ℹ️ Aucun client Redis à fermer")


app = FastAPI(lifespan=lifespan)


async def test_redis_reconnection():
    client = await init_redis_client()
    await asyncio.sleep(600)  # 10min d'inactivité
    await client.ping()  # Doit reconnecter automatiquement


async def init_redis_client() -> Redis | None:
    upstash = os.getenv("UPSTASH_REDIS_URL", "")
    token = os.getenv("UPSTASH_REDIS_TOKEN", "")
    if upstash and token:
        # on nettoie et ajoute le port si besoin
        upstash = upstash.replace("https://", "").replace("http://", "")
        if ":" not in upstash:
            upstash = f"{upstash}:6379"
        redis_url = f"rediss://default:{token}@{upstash}"
        logger.info(f"→ Upstash URI: {redis_url}")
        client = Redis.from_url(
            redis_url,
            decode_responses=True,
            max_connections=20,  # Taille du pool
            health_check_interval=30,  # Vérifie la santé des connexions
            retry=Retry(ExponentialBackoff(), 3),
            retry_on_timeout=True,
            socket_timeout=10,
            socket_keepalive=True,  # Options TCP keep-alive
        )
    else:
        client = Redis.from_url("redis://localhost:6379", decode_responses=True)
    try:
        pong = await client.ping()
        logger.info(f"PONG: {pong}")
        return client
    except Exception as e:
        logger.error(f"Redis ping failed: {e}")
        return None


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


@app.get("/items/{item_id}")
async def read_item(item_id: int, request: Request):
    redis_client = await get_healthy_client(request)

    if not redis_client:
        # Vérifie si le backend actuel est déjà InMemory
        if not isinstance(FastAPICache.get_backend(), InMemoryBackend):
            FastAPICache.init(InMemoryBackend(), prefix="demo-cache")
            logger.warning("Fallback vers cache mémoire")

        backend = FastAPICache.get_backend()
        cached = await backend.get(f"item_{item_id}")
        if cached:
            return {"item_id": item_id, "cached": True, "data": cached}

        data = f"Item data for {item_id}"
        await backend.set(f"item_{item_id}", data, expire=3600)  # expire en secondes
        return {"item_id": item_id, "cached": False, "data": data}

    # Logique Redis normale
    cached = await redis_client.get(f"item_{item_id}")
    if cached:
        return {"item_id": item_id, "cached": True, "data": cached}

    data = f"Item data for {item_id}"
    await redis_client.setex(f"item_{item_id}", 3600, data)
    return {"item_id": item_id, "cached": False, "data": data}


@app.get("/cache-test")
@cache(expire=30)
async def cache_test():
    # retourne un horodatage pour voir si le cache fonctionne
    return {"time": time.time()}


async def kill_redis_connection():
    client = request.app.state.redis_client
    await client.aclose()  # Fermeture manuelle
