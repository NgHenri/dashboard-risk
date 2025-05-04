import os
from dotenv import load_dotenv

# Charge le fichier .env.test depuis le bon chemin
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env.test")
load_dotenv(dotenv_path)


def test_env_vars():
    assert os.getenv("API_URL") == "http://localhost:8000"
    assert os.getenv("MODEL_PATH") == "models/test_model.joblib"
