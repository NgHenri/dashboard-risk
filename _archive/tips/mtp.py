import os
import sys
import time, logging

from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.backends.inmemory import InMemoryBackend
from redis.asyncio import Redis
from fastapi_cache.decorator import cache
from fastapi.middleware.cors import CORSMiddleware

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.requests import Request
from starlette.responses import Response


from redis import asyncio as aioredis
from dotenv import load_dotenv


env_path = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
load_dotenv(dotenv_path=env_path)

# ─── Logger ─────────────────────────────────────────────────────
logger = logging.getLogger("main")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

# ─── FastAPI ────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "x-api-key"],
)

redis_client = None


@app.on_event("startup")
async def on_startup():
    global redis_client
    # Si on a l’URL Upstash, on l’utilise, sinon on bascule sur local
    upstash_url = os.getenv("UPSTASH_REDIS_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_TOKEN")

    if upstash_url and upstash_token:
        # Connexion Upstash (redis-py)
        redis_client = Redis.from_url(f"rediss://default:{upstash_token}@{upstash_url}")
        logger.info("🔌 Connexion à Redis Upstash")
        backend = RedisBackend(redis_client)
    else:
        # Connexion Redis local
        redis_client = Redis.from_url("redis://localhost:6379")
        logger.info("🖥️ Connexion à Redis local")
        backend = InMemoryBackend()  # fallback mémoire

    # Initialisation du cache
    FastAPICache.init(backend, prefix="demo-cache")
    logger.info("✅ FastAPI-Cache initialisé")


@app.get("/cache-test")
@cache(expire=30)
async def cache_test():
    return {"time": time.time()}


@app.get("/test-set")
async def test_set():
    await FastAPICache.set("clef", "valeur", expire=30)
    v = await FastAPICache.get("clef")
    return {"valeur": v}


@app.get("/test-cache-direct")
async def test_cache_direct():
    await FastAPICache.set("key1", "value1", expire=10)
    v = await FastAPICache.get("key1")
    return {"cached_value": v}
