# frontend/config.pyimport os

import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime


def load_env(env_file: str = ".env"):
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_file)

    # Vérifie ce qui est chargé
    api_url = os.getenv("API_URL")

    print(f"🔧 API_URL chargée : {api_url}")

    return {
        "API_URL": api_url,
        "API_KEY": api_key,
        "ARTIFACT_PATH": os.getenv("ARTIFACT_PATH"),
    }


# Charger le .env depuis la racine du projet
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))


def _get_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        raise ValueError(f"❌ La variable '{key}' doit être un float valide.")


def _get_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        raise ValueError(f"❌ La variable '{key}' doit être un int valide.")


# Chemin du fichier courant
current_path = Path(__file__).resolve()

# Si on est en local (structure projet avec dossier parent), sinon Render (copié à la racine /app)
if (current_path.parent / "models").exists():
    ROOT_DIR = current_path.parent
else:
    ROOT_DIR = current_path.parent.parent

artifact_path_str = os.getenv("ARTIFACT_PATH")

if not artifact_path_str:
    raise ValueError("❌ La variable d'environnement 'ARTIFACT_PATH' est manquante.")

ARTIFACT_PATH = ROOT_DIR / artifact_path_str

if not ARTIFACT_PATH.exists():
    raise FileNotFoundError(f"❌ Modèle introuvable à l'emplacement : {ARTIFACT_PATH}")


# Configuration centrale
API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

UPSTASH_REDIS_URL = os.getenv(
    "UPSTASH_REDIS_URL", "settling-parakeet-22795.upstash.io:6379"
)
UPSTASH_REDIS_TOKEN = os.getenv("UPSTASH_REDIS_TOKEN", "")

GLOBAL_DATA_PATH = os.getenv("GLOBAL_DATA_PATH")
COST_FN = _get_int("COST_FN", 10)
COST_FP = _get_int("COST_FP", 1)

LOGO_PATH = os.getenv("LOGO_PATH", "")

THRESHOLD = _get_float("THRESHOLD", 0.0931515)
TIMEOUT = _get_int("TIMEOUT", 10)
TIMEOUT_GLOBAL = _get_int("TIMEOUT_GLOBAL", 180)
RETRY_EVERY = _get_int("RETRY_EVERY", 5)
BATCH_SIZE = _get_int("BATCH_SIZE", 200)
DEBUG = os.getenv("DEBUG", "False").lower() in ("1", "true", "yes")

# Pour debug visuel au démarrage
print(
    f"🔧 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Variables chargées depuis config.py"
)
print(
    f"🔧 API_URL utilisée : {API_URL} (source = {'Render/env' if 'API_URL' in os.environ else '.env'})"
)
print(f"   • THRESHOLD = {THRESHOLD}")
print(f"   • TIMEOUT   = {TIMEOUT}")
print(f"   • DEBUG     = {DEBUG}")
