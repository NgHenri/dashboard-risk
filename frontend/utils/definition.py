import streamlit as st
import re


# 1) Labels explicites pour certaines features
_FIELD_LABELS = {
    "amt_payment": "montant du paiement",
    "amt_annuity": "montant de l'annuité",
    "rate_down_payment": "taux d'acompte",
    "hour_appr_process_start": "heure de début du traitement de l'approbation",
    "days_credit_enddate": "jours restants avant la fin du crédit",
    "days_instalment": "nombre de jours avant le paiement",
    # … compléter au besoin …
}

_FIELD_DICT = {
    "DAYS_CREDIT_ENDDATE": "le nombre de jours restants avant la fin du crédit",
    "HOUR_APPR_PROCESS_START": "l'heure de début du traitement d'approbation",
    "DAYS_OVERDUE": "le nombre de jours de retard",
    "AMT_ANNUITY": "le montant de l'annuité",
    "CNT_INSTALMENT": "le nombre d'échéances",
}

# 2) Dictionnaires pour génération automatique
_SOURCE_DICT = {
    "APPROVED": "les demandes approuvées",
    "ACTIVE": "les crédits actifs",
    "CREDIT": "les crédits",
    "PREV": "les demandes précédentes",
    "BUREAU": "les crédits du bureau de crédit",
    "POS": "les points de vente",
    "INSTAL": "les paiements échelonnés",
    "CASH": "les avances de trésorerie",
}


_AGG_DICT = {
    "MEAN": "moyenne de",
    "MAX": "valeur maximale de",
    "MIN": "valeur minimale de",
    "SUM": "somme de",
    "STD": "écart-type de",
    "MEDI": "médiane de",
    "COUNT": "nombre de fois que",
    "RATIO": "ratio de",
}


def generate_feature_definition(feature_name: str) -> str:
    """
    Génère une définition pour une feature en combinant plusieurs dictionnaires :
    - _FIELD_LABELS pour les bases connues
    - _FIELD_DICT pour des libellés plus complets
    - _SOURCE_DICT pour le préfixe
    - _AGG_DICT pour l'agrégat
    """
    name = feature_name.upper()
    parts = name.split("_")
    prefix = parts[0]  # ex: INSTAL
    agg = parts[-1]  # ex: MIN
    core = "_".join(parts[1:-1])  # ex: AMT_PAYMENT

    # 1) Si la partie centrale correspond à _FIELD_LABELS + agg connu
    if core.lower() in _FIELD_LABELS and agg in _AGG_DICT:
        base_label = _FIELD_LABELS[core.lower()]
        agg_label = _AGG_DICT[agg]
        # ajouter le préfixe discret si besoin
        source_lbl = _SOURCE_DICT.get(prefix)
        if source_lbl:
            return f"{agg_label} {base_label} pour {source_lbl}."
        else:
            return f"{agg_label} {base_label}."

    # 2) Sinon si on a un libellé complet dans _FIELD_DICT
    if core in _FIELD_DICT and agg in _AGG_DICT:
        field_label = _FIELD_DICT[core]
        agg_label = _AGG_DICT[agg]
        source_lbl = _SOURCE_DICT.get(prefix)
        if source_lbl:
            return f"{agg_label} {field_label} pour {source_lbl}."
        else:
            return f"{agg_label} {field_label}."

    # 3) Sinon, fallback générique sur préfixe + agg
    source_lbl = _SOURCE_DICT.get(prefix)
    agg_label = _AGG_DICT.get(agg)
    if source_lbl and agg_label:
        # description plus brute du core si rien d'autre
        core_fallback = core.replace("_", " ").lower()
        return f"{agg_label} {core_fallback} pour {source_lbl}."

    # 4) Cas par défaut:
    return None


def display_feature_definition(feature_name: str, definitions_dict: dict):
    """
    Affiche la définition d'une variable depuis un dictionnaire ou en la générant automatiquement.
    Gère aussi le cas où aucune définition ne peut être produite.

    :param feature_name: Nom de la variable sélectionnée.
    :param definitions_dict: Dictionnaire contenant les définitions des variables.
    """
    definition = definitions_dict.get(feature_name)

    if definition:
        st.markdown(f"ℹ️ **Définition de `{feature_name}`** : {definition}")
    else:
        # Tentative de génération
        generated_def = generate_feature_definition(feature_name)
        if generated_def:
            st.markdown(f"🧠 *Définition de `{feature_name}`* : {generated_def}")
        else:
            st.markdown(f"ℹ️ *Aucune définition disponible pour `{feature_name}`.*")


DEFINITIONS_VARIABLES = {
    "DAYS_EMPLOYED_PERC": "Ratio entre les jours d'emploi et l'âge du client.",
    "INCOME_CREDIT_PERC": "Ratio entre le revenu total et le montant du crédit demandé.",
    "INCOME_PER_PERSON": "Revenu moyen par personne dans le foyer, calculé comme le revenu total du client divisé par le nombre de membres du foyer.",
    "AMT_CREDIT": "Montant total du crédit demandé par le client.",
    "ANNUITY_INCOME_PERC": "Ratio entre l'annuité du crédit et le revenu total.",
    "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": "Montant moyen maximal en retard sur les crédits précédents.",
    "BURO_CREDIT_TYPE_ANOTHER_TYPE_OF_LOAN_MEAN": "Proportion des autres types de prêts dans l'historique du client.",
    "EXT_SOURCE_1": "Score externe d’évaluation de la solvabilité, provenant d’une source externe",
    "EXT_SOURCE_2": "Deuxième score externe, souvent complémentaire à EXT_SOURCE_1, également indicateur de risque.",
    "EXT_SOURCE_3": "Troisième score externe de solvabilité, mesurant la stabilité financière d’un client.",
    "DAYS_BIRTH": "Âge du client exprimé en jours",
    "BURO_DAYS_CREDIT_MEAN": "Moyenne des jours écoulés depuis l’octroi des anciens crédits",
    "BURO_CREDIT_ACTIVE_CLOSED_MEAN": "Proportion moyenne des crédits passés (fermé = remboursé) dans l’historique du client.",
    "NAME_INCOME_TYPE_WORKING": "Variable binaire indiquant si le client est employé.",
    "CODE_GENDER": "Sexe du client (1 = homme, 0 = femme dans les versions encodées).",
    "BURO_DAYS_CREDIT_UPDATE_MEAN": "Moyenne des jours depuis la dernière mise à jour du crédit (indicateur d’activité récente).",
    "REGION_RATING_CLIENT_W_CITY": "Évaluation de la région de résidence du client, prenant en compte la ville. Plus la note est élevée, plus la région est jugée' à risque'.",
    "NAME_INCOME_TYPE_PENSIONER": "Variable binaire indiquant si le client est retraité.",
    "OCCUPATION_TYPE_LABORERS": "Variable binaire : le client est un ouvrier manuel.",
    "APPROVED_DAYS_DECISION_MIN": "Délai minimum de décision (en jours) pour les crédits approuvés dans le passé (previous_application.csv).",
    "NAME_EDUCATION_TYPE_SECONDARY__SECONDARY_SPECIAL": "Niveau d'études secondaires ou spéciales.",
    "DAYS_LAST_PHONE_CHANGE": "Jours écoulés depuis le dernier changement de téléphone (indicateur de stabilité).",
    "REG_CITY_NOT_LIVE_CITY": "Indique si le client n'habite pas dans la même ville que celle où il est enregistré (peut signaler une instabilité).",
    "FLAG_DOCUMENT_3": "Variable binaire indiquant si le client a fourni le document 3 (ex. : carte d’identité).",
    "BURO_CREDIT_TYPE_MICROLOAN_MEAN": "Moyenne des crédits de type microprêt dans le fichier bureau.csv.",
    "DAYS_ID_PUBLISH": "Jours depuis la délivrance du document d'identité actuel.",
    "BURO_DAYS_CREDIT_MAX": "Nombre maximum de jours depuis la prise d’un crédit dans le passé. Indique l’ancienneté du plus vieux crédit.",
    "FLAG_EMP_PHONE": "Le client a-t-il fourni un numéro de téléphone professionnel ? (1 = oui, 0 = non).",
    "DAYS_REGISTRATION": "Jours écoulés depuis l'enregistrement de la résidence actuelle du client.",
    "BURO_DAYS_CREDIT_ENDDATE_MEAN": "Moyenne des jours restants jusqu'à l'échéance des crédits dans bureau.csv.",
    "APPROVED_AMT_ANNUITY_MEAN": "Montant moyen des annuités pour les crédits approuvés dans le passé.",
    "LIVE_CITY_NOT_WORK_CITY": "Variable binaire indiquant si le client travaille dans une ville différente de celle où il vit.",
    "ORGANIZATION_TYPE_SELF_EMPLOYED": "Type d'organisation dans laquelle le client travaille : ici, auto-entrepreneur.",
    "OCCUPATION_TYPE_LOW_SKILL_LABORERS": "Le client est-il classé comme ouvrier non qualifié ?",
    "OCCUPATION_TYPE_DRIVERS": "Le client travaille-t-il comme conducteur (chauffeur, livreur...) ?",
    "FLAG_DOCUMENT_6": "Le client a-t-il fourni le document numéro 6 ? (type de justificatif administratif).",
    "FLAG_WORK_PHONE": "Le client a-t-il fourni un numéro de téléphone professionnel ? (souvent identique à FLAG_EMP_PHONE).",
    "ORGANIZATION_TYPE_BUSINESS_ENTITY_TYPE_3": "	Type d'organisation employeur appartenant à la catégorie 'Business Entity Type 3'.",
    "OCCUPATION_TYPE_ACCOUNTANTS": "Le client travaille-t-il comme comptable ?",
    "NAME_HOUSING_TYPE_RENTED_APARTMENT": "Type de logement : appartement loué.",
    "NAME_FAMILY_STATUS_MARRIED": "Statut familial : marié(e).",
    "PAYMENT_RATE": "Ratio entre le montant d’annuité et le montant du crédit (AMT_ANNUITY / AMT_CREDIT).",
    "FLAG_OWN_CAR": "Le client possède-t-il une voiture ?",
    "FLOORSMAX_MODE": "Nombre maximum d'étages dans le bâtiment (mode le plus fréquent dans les données de logement).",
    "LIVINGAREA_MODE": "Surface habitable du logement (normalisée, version mode).",
    "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN": "Nombre de jours en moyenne qu’il reste avant la fin des crédits encore en cours pour un client.",
    "APPROVED_HOUR_APPR_PROCESS_START_MEAN": "Heure moyenne de la journée à laquelle le client commence ses demandes de crédit ayant été approuvées (sur 24h, entre 0 et 23h).",
}

PAIR_DEFINITIONS = {
    "Stabilité vs Capacité remb.": (
        "Compare la stabilité professionnelle (ancienneté / âge) "
        "à la capacité de remboursement (revenu / crédit)."
    ),
    "Ancienneté vs Montant crédit": (
        "Étudie l’impact de l’ancienneté dans l’emploi sur le montant du crédit demandé."
    ),
    "Montant crédit vs Charge mens.": (
        "Mets en relation le montant du crédit et la charge mensuelle "
        "(annuité / revenu) pour voir l’effort financier mensuel."
    ),
    "Montant crédit vs Capacité remb.": (
        "Compare le montant du crédit demandé à la capacité de remboursement "
        "(revenu / crédit)."
    ),
    "Retards crédits passés vs Capacité remb.": (
        "Analyse comment les montants en retard (max) affectent la capacité de remboursement."
    ),
    "Autres prêts vs Montant crédit": (
        "Examine la part des autres types de prêts dans l’historique du client "
        "et son influence sur le montant du crédit demandé."
    ),
}
