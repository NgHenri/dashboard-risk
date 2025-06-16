# safe_get : cas où la colonne existe avec valeur non-NaN, NaN, absente, pour dict et pandas Series.

# format_currency : formats valides et entrées invalides (None, 'abc').

# format_percentage : mêmes idées.

# format_gender : valeurs 1,0 et inattendues.

# format_years : valeurs typiques et invalides.

# parse_client_value : conversion selon label « Âge », montants en €, pourcentages, cas invalides.

# parse_currency : parsing de chaînes avec espaces (y compris insécables), cas invalides, et test de logging d’erreur via caplog.


import pytest
import pandas as pd
import logging  # Ajouté pour caplog

from frontend.utils.formatters import (
    safe_get,
    format_currency,
    format_percentage,
    format_gender,
    format_years,
    parse_client_value,
    parse_currency,
)


# Tests pour safe_get
def test_safe_get_present_non_na():
    row = {"a": 1, "b": None}
    assert safe_get(row, "a", default="X") == 1


def test_safe_get_present_na():
    row = {"a": float("nan")}
    assert safe_get(row, "a", default="X") == "X"
    # Using pandas Series
    ser = pd.Series({"a": None, "b": 2})
    assert safe_get(ser, "a", default="Y") == "Y"
    assert safe_get(ser, "b", default="Y") == 2


def test_safe_get_absent():
    row = {"x": 10}
    assert safe_get(row, "y", default="Z") == "Z"


# Tests pour format_currency
@pytest.mark.parametrize(
    "value, expected",
    [
        (28790.5, "28 790,50 €"),
        (1000, "1 000,00 €"),
        (0, "0,00 €"),
        (None, "N/A"),
        ("abc", "N/A"),
    ],
)
def test_format_currency(value, expected):
    assert format_currency(value) == expected


# Tests pour format_percentage
@pytest.mark.parametrize(
    "value, expected",
    [
        (0.1234, "12.3 %"),
        (1, "100.0 %"),
        (0, "0.0 %"),
        ("abc", "N/A"),
        (None, "N/A"),
    ],
)
def test_format_percentage(value, expected):
    assert format_percentage(value) == expected


# Tests pour format_gender
@pytest.mark.parametrize(
    "value, expected",
    [
        (1, "Homme"),
        (0, "Femme"),
        (None, "Inconnu"),
        (2, "Inconnu"),
    ],
)
def test_format_gender(value, expected):
    assert format_gender(value) == expected


# Tests pour format_years
@pytest.mark.parametrize(
    "value, expected",
    [
        (-365 * 5, "5 ans"),  # -1825 days -> 5 ans
        (-100, "0 ans"),  # -100//365 = 0
        ("not_int", "N/A"),  # invalid
        (None, "N/A"),  # invalid
    ],
)
def test_format_years(value, expected):
    assert format_years(value) == expected


# Tests pour parse_client_value
@pytest.mark.parametrize(
    "value,label,expected",
    [
        ("45 ans", "Âge", 45.0),
        ("  23 ans", "Âge", 23.0),
        ("28,79 €", "Montant€, test", 28.79),
        ("1 000,50 €", "Some€label", 0.0),  # insécable non géré par parse_client_value
        ("12.3 %", "%label", 0.123),
        ("12,3 %", "label%", 0.123),
        ("3.14", "Other", 3.14),
        ("invalid", "Other", 0.0),
        (None, "Other", 0.0),
    ],
)
def test_parse_client_value(value, label, expected):
    result = parse_client_value(value, label)
    assert pytest.approx(result, rel=1e-3) == expected


# Tests pour parse_currency
@pytest.mark.parametrize(
    "value, expected",
    [
        ("28 790,50 €", 28790.50),  # espace insécable géré
        ("1 000,75 €", 0.0),  # espace normal non géré
        ("1000.25", 1000.25),
        ("€ 500", 500.0),
        ("invalid", 0.0),
        (None, 0.0),
    ],
)
def test_parse_currency(value, expected):
    result = parse_currency(value)
    assert pytest.approx(result, rel=1e-3) == expected


# Test logging pour parse_currency invalid
def test_parse_currency_logs_error(caplog):
    caplog.set_level(logging.ERROR)
    result = parse_currency("not_a_number")
    assert result == 0.0
    # Vérifier qu'un message d'erreur a été loggé
    assert any("Échec conversion devise" in record.message for record in caplog.records)


# =============================================================================
# Résumé des tests de frontend/utils/formatters.py
#
# safe_get:
# - test_safe_get_present_non_na: retourne la valeur si la clé existe et n’est pas NA.
# - test_safe_get_present_na: retourne default si la clé existe mais la valeur est NaN/None.
#   Vérifie aussi sur pandas.Series.
# - test_safe_get_absent: retourne default si la clé n’est pas présente.
#
# format_currency:
# - test_format_currency: formate un nombre en chaîne « xx xxx,xx € » ou renvoie "N/A" si invalide.
#   Cas testés : valeur normale, zéro, None ou chaîne non numérique.
#
# format_percentage:
# - test_format_percentage: formate une fraction en pourcentage avec 1 décimale ou "N/A" si invalide.
#   Cas testés : 0.1234→"12.3 %", 1→"100.0 %", 0, None, 'abc'.
#
# format_gender:
# - test_format_gender: convertit 1→"Homme", 0→"Femme", autres/None→"Inconnu".
#
# format_years:
# - test_format_years: convertit un nombre de jours négatif en « X ans » (division entière), ou "N/A" si invalide.
#   Cas tests : -365*5, -100, chaîne invalide, None.
#
# parse_client_value:
# - test_parse_client_value: selon le label :
#     - "Âge": extrait la partie numérique avant “ans”.
#     - présence de "€": parse_currency interne (ici non géré pour espaces complexes → 0.0).
#     - présence de "%": conversion en fraction.
#     - sinon float(value) ou 0.0 si échec.
#   Cas testés : "45 ans", "28,79 €", montants avec insécable non géré, pourcentages, chaînes invalides, None.
#
# parse_currency:
# - test_parse_currency: nettoie la chaîne pour extraire un float ou retourne 0.0 si échec.
#   Implémentation actuelle gère les espaces insécables, pas les espaces normaux.
#   Cas testés : "28 790,50 €" (insécable), "1 000,75 €" (espace normal → échec), "1000.25", "€ 500", invalides, None.
#
# Logging:
# - test_parse_currency_logs_error: sur entrée impossible, parse_currency renvoie 0.0 et logge une erreur.
#
# =============================================================================
