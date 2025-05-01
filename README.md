## 📝 **Description du projet — Scoring client pour "Prêt à dépenser"**

L'organisme de prêt *"Prêt à dépenser"* souhaite fournir à ses **conseillers clientèle** un outil d'aide à la décision leur permettant d'évaluer le risque de défaut d’un client avant l’octroi d’un prêt.

### 🎯 Objectif
Développer un **modèle de scoring fiable, explicable et intégré dans une application web** (via une API) permettant d’estimer, à partir des informations d’un client, s’il existe un **risque suffisant d’insolvabilité** pour refuser un prêt.

Une **contrainte métier forte** est imposée :  
> Le **coût d’un prêt accordé à un mauvais payeur est 10 fois (voire 100 fois) plus élevé** que celui d’un refus injustifié.

---

## 📊 Données

- **Jeu d’entraînement** : 307 511 clients, 122 variables
- **Variable cible** : défaut de paiement (environ 8% de défauts → données déséquilibrées)
- **Jeu de production** : 50 000 nouveaux clients, avec données préremplies

---

## 🔧 Feature engineering

Les transformations sont inspirées des meilleures pratiques du domaine (notamment les notebooks de référence sur Kaggle) :

- Suppression d'outliers
- Regroupement et transformation de jeux de données secondaires (bureau, previous_application, POS, etc.)
- Encodage : Label Encoding / One-hot Encoding
- Création de nouvelles variables (ratios, différences de jours, scores normalisés)
- Traitement du déséquilibre : **SMOTE**, **class_weight**, ou **undersampling**
- Nettoyage des colonnes et traitement des valeurs aberrantes

---

## 🤖 Modélisation & suivi MLflow

### 📚 Modèles testés :

- `DummyClassifier` (baseline)
- `LogisticRegression`
- `RandomForestClassifier`
- `XGBoostClassifier`
- `GradientBoostingClassifier`
- **`LightGBMClassifier` (modèle final)**

Chaque modèle a été :

- Entraîné avec **validation croisée**
- Suivi avec **MLflow** (enregistrement des hyperparamètres, métriques, durée d’entraînement, artefacts)
- Évalué avec des métriques classiques (**AUC, Accuracy, Recall, Confusion matrix**) mais aussi :
  - Un **score métier personnalisé**, prenant en compte le **coût différentiel entre faux positifs et faux négatifs**
  - Une **optimisation du seuil de décision** (proba) pour **minimiser ce score métier**

💡 Le **modèle LightGBM**, avec `class_weight` et seuil optimisé, a offert **le meilleur compromis performance/coût métier/rapidité**.

---

## 📦 Packaging du pipeline

Un pipeline complet a été mis en place, incluant :

- Imputation des valeurs manquantes
- Normalisation / standardisation si nécessaire
- Modèle LightGBM final
- Enregistrement via `joblib` et MLflow
- Chargement dynamique dans l’API FastAPI

---

## 🔍 Explicabilité

- **Globale** : SHAP importance plot
- **Locale** : SHAP values pour un client donné, affichées dans le dashboard
- Ajout d’un score lisible pour le conseiller : risque faible / moyen / élevé

---

## 📈 Analyse du Data Drift

- Analyse menée avec **Evidently**
- Comparaison entre jeux `train_samp` et `test_samp`
- 35% des variables présentent un drift (ex. : `DAYS_BIRTH`, `EMPLOYED_TO_AGE_RATIO`, etc.)
- Ces dérives sont à surveiller régulièrement en production pour maintenir la robustesse du modèle

---

## 🌐 Architecture logicielle

- **Backend** (FastAPI) :
  - Endpoint `/predict` : score + SHAP
  - Endpoint `/health` : monitoring complet du système
- **Frontend** (Streamlit) :
  - Dashboard dynamique interrogeant uniquement l’API
  - Aucun modèle local ni base de données stockée
- **Déploiement** :
  - Prêt pour un environnement **CI/CD**
  - Intégration continue facilitée avec versionnement du modèle via MLflow

---

## ✅ Résultat

Le système final permet aux conseillers d’obtenir :

- Un **score de risque** clair
- Une **explication de la décision**
- Un **processus fiable et automatisé**, intégrable dans leur environnement métier


## 📂 Structure du projet
dashboard-risk 
.
├── README.md
├── assets
│   └── style.css
├── backend
│   ├── data
│   │   ├── test_1000_sample_for_api.csv
│   │   ├── test_1000_sample_with_target.csv
│   │   ├── test_2000_sample_for_api.csv
│   │   ├── test_2000_sample_with_target.csv
│   │   ├── test_500_sample_for_api.csv
│   │   └── test_500_sample_with_target.csv
│   ├── generate_api_key.py
│   ├── main.py
│   ├── model.py
│   ├── models
│   │   ├── inference_pipeline_20250413_17.joblib
│   │   └── lightgbm_production_artifact_20250415_081218.pkl
│   ├── requirements.txt
│   ├── tests
│   │   ├── test_config.py
│   │   └── test_sanity.py
│   └── utils.py
├── environment.yml
├── frontend
│   ├── app.py
│   ├── assets
│   ├── config.py
│   ├── requirements.txt
│   ├── risk_gauge.py
│   ├── static
│   ├── tests│   
│   │   └── test_project
│   └── utils
│       ├── api_requests.py
│       ├── formatters.py
│       ├── shap_utils.py
│       ├── styling.py
│       ├── user_interactions.py
│       └── visuals.py
├── git_push.sh
├── htmlcov
│   
├── install_rclone.sh
├── notebooks
│   └── tmp.ipynb
├── pytest.ini
├── report.html
├── requirements.txt
├── setup.cfg
└── sync_to_drive.sh
