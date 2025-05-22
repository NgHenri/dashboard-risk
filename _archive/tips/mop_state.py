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


@app.on_event("startup")
async def startup():
    client = await init_redis_client()
    if client:
        app.state.redis_client = client
        FastAPICache.init(RedisBackend(client), prefix="demo-cache")
        logger.info("Cache Redis activé")
    else:
        app.state.redis_client = None
        FastAPICache.init(InMemoryBackend(), prefix="demo-cache")
        logger.warning("Fallback cache mémoire")


@app.get("/items/{item_id}")
async def read_item(item_id: int, request: Request):
    redis_client = request.app.state.redis_client
    if not redis_client:
        raise HTTPException(500, "Redis non initialisé")
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


@app.on_event("shutdown")
async def shutdown():
    redis_client = app.state.redis_client
    if redis_client:
        try:
            await redis_client.close()  # ou await redis_client.close()
            logger.info("🧹 Connexion Redis fermée proprement")
        except Exception as e:
            logger.warning(f"⚠️ Erreur à la fermeture Redis : {e}")
    else:
        logger.info("ℹ️ Aucun client Redis à fermer")
