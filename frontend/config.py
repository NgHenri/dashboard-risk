# Chemins relatifs
MODEL_PATH = "../backend/models/lightgbm_production_artifact_20250415_081218.pkl"
THRESHOLD = 0.0931515

# Données
TEST_SAMPLE_WITH_TARGET = "../backend/data/test_1000_sample_with_target.csv"
TEST_SAMPLE_FOR_API = "../backend/data/test_1000_sample_for_api.csv"

import os
from dotenv import load_dotenv


def load_env(env_file: str = ".env"):
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=env_file)

    # Vérifie ce qui est chargé
    api_url = os.getenv("API_URL")
    print(f"Chargé API_URL: {api_url}")

    return {
        "API_URL": api_url,
        "MODEL_PATH": os.getenv("MODEL_PATH"),
    }
