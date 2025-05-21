import asyncio
import os
from dotenv import load_dotenv

# Chargement des variables d'environnement depuis .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

try:
    from upstash_redis.asyncio import Redis
except ImportError:
    print(
        "❌ Le module 'upstash_redis' n'est pas installé. Exécutez : pip install upstash-redis"
    )
    exit(1)


async def test_upstash_connection():
    url = os.getenv("UPSTASH_REDIS_URL")
    token = os.getenv("UPSTASH_REDIS_TOKEN")

    if not url or not token:
        print("❌ UPSTASH_REDIS_URL ou UPSTASH_REDIS_TOKEN non défini dans .env")
        return

    try:
        redis = Redis(url, token=token)
        await redis.set("connectivity-test", "ok", ex=10)
        value = await redis.get("connectivity-test")
        if value == "ok":
            print("✅ Connexion Upstash Redis réussie 🎉")
        else:
            print(f"⚠️ Réponse inattendue : {value}")
    except Exception as e:
        print(f"❌ Échec de la connexion à Upstash : {e}")


if __name__ == "__main__":
    asyncio.run(test_upstash_connection())
