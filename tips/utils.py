from fastapi import FastAPI, Request, HTTPException
from datetime import datetime, timedelta
import logging
from redis.asyncio import Redis  # Client Redis asynchrone
import os
from typing import Dict, Any

app = FastAPI()

# Configuration Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = Redis.from_url(REDIS_URL, decode_responses=True)

# Configuration du logging
logger = logging.getLogger("api_tracker")
logger.setLevel(logging.INFO)


# Fonction pour obtenir l'IP réelle (gère les proxies)
async def get_real_ip(request: Request) -> str:
    if x_forwarded_for := request.headers.get("X-Forwarded-For"):
        return x_forwarded_for.split(",")[0]
    return request.client.host if request.client else "unknown"


# Middleware de tracking
@app.middleware("http")
async def track_visits(request: Request, call_next):
    try:
        ip = await get_real_ip(request)

        # Clé Redis pour le compteur de visites
        count_key = f"visits:{ip}:count"
        timestamp_key = f"visits:{ip}:timestamps"

        # Incrémente le compteur
        visit_count = await redis_client.incr(count_key)

        # Enregistre le timestamp
        await redis_client.rpush(timestamp_key, datetime.now().isoformat())

        # Expire après 24h
        await redis_client.expire(count_key, 86400)
        await redis_client.expire(timestamp_key, 86400)

        logger.info(f"Visite de {ip} - Total: {visit_count}")

    except Exception as e:
        logger.error(f"Erreur de tracking: {str(e)}")

    return await call_next(request)


# Endpoint pour récupérer les stats
@app.get("/tracking/{ip}")
async def get_tracking_data(ip: str) -> Dict[str, Any]:
    try:
        count = await redis_client.get(f"visits:{ip}:count") or 0
        timestamps = await redis_client.lrange(f"visits:{ip}:timestamps", 0, -1)

        return {
            "ip": ip,
            "total_visits": int(count),
            "first_visit": timestamps[0] if timestamps else None,
            "last_visit": timestamps[-1] if timestamps else None,
            "timestamps": timestamps,
        }
    except Exception as e:
        logger.error(f"Erreur de récupération: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur interne")


@app.get("/")
async def root():
    return {"message": "Bienvenue sur l'API de tracking !"}


# Endpoint de statistiques globales
@app.get("/tracking")
async def get_all_tracking():
    try:
        keys = await redis_client.keys("visits:*:count")
        results = {}

        for key in keys:
            ip = key.split(":")[1]
            count = await redis_client.get(key)
            results[ip] = int(count)

        return {"tracking_data": results}
    except Exception as e:
        logger.error(f"Erreur global tracking: {str(e)}")
        raise HTTPException(status_code=500, detail="Erreur de traitement")
