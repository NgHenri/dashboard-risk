# 📊 Dashboard de scoring de risque de crédit - Projet "Prêt à Dépenser"

## 🌟 Objectif du projet
Ce projet vise à créer un outil d'aide à la décision pour les conseillers clientèle de l'organisme de prêt "Prêt à dépenser". L'objectif est d'évaluer le risque qu'un client présente un défaut de paiement afin d'aider à l'acceptation ou au refus d'une demande de crédit. Le modèle tient compte du fait qu'un faux négatif (accorder un crédit à un mauvais payeur) est beaucoup plus coûteux qu'un faux positif (refuser un bon payeur).

---

## 🧠 Approche Machine Learning

### 🔄 Données
- Plus de **300 000 clients** avec 122 variables.
- Environ **8 %** de défauts de paiement.

### 🧬 Prétraitement et Feature Engineering
- Nettoyage des données (valeurs aberrantes, outliers)
- LabelEncoding / OneHotEncoding des variables catégorielles
- Agrégations et création de nouvelles variables (ratios, différences, etc.)
- Normalisation et gestion des valeurs manquantes

### 📊 Modèles testés
- DummyClassifier (baseline)
- Logistic Regression
- RandomForest
- Gradient Boosting
- XGBoost
- **LightGBM** (meilleur modèle)

### ⚖️ Optimisation
- **Métriques classiques** : AUC ROC, Accuracy, Recall, F1
- **Score métier personnalisé** : `10 x FN + 1 x FP`
- Optimisation du **seuil de décision** pour minimiser ce score
- Validation croisée stratifieée avec pipelines intégrés (SMOTE + StandardScaler + Modèle)

---

## 📈 Tracking & Expérimentation avec MLflow
- Enregistrement de **tous les modèles**, hyperparamètres, métriques, temps d'exécution
- Fonction générique `run_single_model()` pour automatiser le pipeline et le logging
- Meilleur modèle enregistré comme **Registered Model** dans MLflow

---

## 🔍 Explicabilité
- **SHAP Values** pour explicabilité globale et locale
- Visualisation dynamique dans l'interface utilisateur (Streamlit)
- Présentation des variables les plus influentes pour chaque client

---

## 🤖 Analyse de la robustesse et du Drift
- Utilisation de **Evidently AI** pour la détection de **data drift** entre jeu d'entraînement et données en production
- Plus de 35 % des colonnes impactées, notamment celles liées à l'âge
- Recommandation : surveillance périodique continue

---

## 🚀 Déploiement

### 🚧 Backend (API FastAPI)
- Prédiction à partir des données client
- Retour du score et des valeurs SHAP
- Modèles enregistrés sous Joblib ou Pickle

### 👁️ Frontend (Streamlit)
- Application web utilisable par les conseillers
- Interface graphique intuitive
- Requêtes vers l'API via `api_requests.py`

---

## 📂 Structure du projet
```
dashboard-risk/
├── backend/           # API FastAPI, modèles et outils
├── frontend/          # Interface utilisateur Streamlit
├── assets/            # CSS et ressources
├── notebooks/         # Exploration et notebooks de travail
├── tests/             # Tests unitaires
├── requirements.txt   # Dépendances backend
├── environment.yml    # Environnement complet
└── README.md
```

---

## 🔧 Installation rapide
```bash
# 1. Créer un environnement conda
conda env create -f environment.yml
conda activate dashboard-risk

# 2. Lancer l'API
cd backend
uvicorn main:app --reload

# 3. Lancer l'app Streamlit
cd ../frontend
streamlit run app.py
```

---

## 🎉 Résultats
- Modèle final : **LightGBM avec class_weight + seuil optimisé**
- AUC ROC : 0.76+, Score métier : ~0.51 (sur jeu test)
- Interprétabilité intégrée + pipeline complet enregistré

---

## 🚀 Prochaines étapes possibles
- Déploiement sur AWS ou Render
- Monitoring en production (Evidently ou Prometheus)
- Intégration continue avec GitHub Actions
- Authentification utilisateurs et gestion de sessions (API key)

---

Pour toute question ou amélioration, n'hésitez pas à me contacter !

[![Model](https://img.shields.io/badge/LightGBM-0.789_AUC-blue)](https://lightgbm.readthedocs.io/)

