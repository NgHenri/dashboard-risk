import sys
import os

# Ajouter frontend au PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../frontend"))
)

# Importer config depuis frontend
from config import load_env


def test_load_env_test(monkeypatch):
    # Construire le chemin complet vers .env.test dans frontend
    frontend_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../frontend")
    )
    env_test_path = os.path.join(frontend_dir, ".env.test")

    # Charger manuellement le fichier .env.test
    config = load_env(env_test_path)

    # Tests des valeurs chargées depuis .env.test
    assert config["API_URL"] == "http://localhost:8000"
    assert config["MODEL_PATH"] == "models/test_model.joblib"
