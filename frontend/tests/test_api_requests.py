import unittest
from frontend.utils.api_requests import (
    fetch_client_ids,
    fetch_client_info,
    fetch_prediction,
)
from frontend.utils.shap_utils import fetch_local_shap_explanation
from unittest.mock import patch, MagicMock
import requests

# Données JSON à tester
mock_client_info = {
    "EXT_SOURCE_1": 0.5074440332442849,
    "EXT_SOURCE_2": 0.5445642510623193,
    "EXT_SOURCE_3": 0.2636468134452008,
    "AMT_CREDIT": 343800,
    "ANNUITY_INCOME_PERC": 0.2202941176470588,
    "BURO_DAYS_CREDIT_ENDDATE_MEAN": 4430,
    "BURO_DAYS_CREDIT_UPDATE_MEAN": -877,
    "DAYS_BIRTH": -19457,
    "DAYS_ID_PUBLISH": -2976,
    "FLAG_EMP_PHONE": 1,
    "NAME_INCOME_TYPE_PENSIONER": 0,
    "BURO_AMT_CREDIT_SUM_MAX": 19224,
    "BURO_CREDIT_ACTIVE_CLOSED_MEAN": 0.5,
    "BURO_CREDIT_TYPE_CAR_LOAN_MEAN": 0,
    "BURO_CREDIT_TYPE_CREDIT_CARD_MEAN": 0.5,
    "BURO_CREDIT_TYPE_MICROLOAN_MEAN": 0,
    "BURO_CREDIT_TYPE_MORTGAGE_MEAN": 0,
    "BURO_DAYS_CREDIT_ENDDATE_MAX": 8763,
    "BURO_DAYS_CREDIT_MEAN": -1171.5,
    "CODE_GENDER": 1,
    "DAYS_EMPLOYED_PERC": 0.0199928046461427,
    "DAYS_LAST_PHONE_CHANGE": -206,
    "DAYS_REGISTRATION": -5927,
    "DEF_60_CNT_SOCIAL_CIRCLE": 0,
    "FLAG_DOCUMENT_3": 1,
    "FLAG_DOCUMENT_6": 0,
    "NAME_HOUSING_TYPE_RENTED_APARTMENT": 0,
    "OCCUPATION_TYPE_ACCOUNTANTS": 0,
    "ORGANIZATION_TYPE_BUSINESS_ENTITY_TYPE_3": 0,
    "ORGANIZATION_TYPE_MILITARY": 0,
    "ORGANIZATION_TYPE_SELF_EMPLOYED": 0,
    "PAYMENT_RATE": 0.0490183246073298,
    "REGION_POPULATION_RELATIVE": 0.010032,
    "REGION_RATING_CLIENT_W_CITY": 2,
    "REG_CITY_NOT_LIVE_CITY": 0,
    "ACTIVE_DAYS_CREDIT_ENDDATE_MEAN": 97,
    "AMT_REQ_CREDIT_BUREAU_QRT": 0,
    "APPROVED_AMT_ANNUITY_MEAN": 23236.185,
    "APPROVED_DAYS_DECISION_MIN": -2591,
    "APPROVED_HOUR_APPR_PROCESS_START_MEAN": 4.333333333333333,
    "BURO_AMT_CREDIT_MAX_OVERDUE_MEAN": 562.5,
    "BURO_AMT_CREDIT_SUM_DEBT_SUM": 9904.5,
    "BURO_CREDIT_TYPE_ANOTHER_TYPE_OF_LOAN_MEAN": 0,
    "BURO_CREDIT_TYPE_LOAN_FOR_BUSINESS_DEVELOPMENT_MEAN": 0,
    "BURO_CREDIT_TYPE_LOAN_FOR_THE_PURCHASE_OF_EQUIPMENT_MEAN": 0,
    "BURO_DAYS_CREDIT_MAX": -147,
    "BURO_DAYS_CREDIT_VAR": 2099200.5,
    "CLOSED_AMT_CREDIT_SUM_SUM": 0,
    "ELEVATORS_AVG": 0,
    "FLAG_DOCUMENT_13": 0,
    "FLAG_DOCUMENT_15": 0,
    "FLAG_DOCUMENT_16": 0,
    "FLAG_DOCUMENT_17": 0,
    "FLAG_DOCUMENT_20": 0,
    "FLAG_DOCUMENT_21": 0,
    "FLAG_OWN_CAR": 0,
    "FLAG_WORK_PHONE": 0,
    "FLOORSMAX_MODE": 0.2083,
    "INCOME_CREDIT_PERC": 0.2225130890052356,
    "INCOME_PER_PERSON": 76500,
    "INSTAL_AMT_PAYMENT_MAX": 651607.29,
    "INSTAL_AMT_PAYMENT_MEAN": 29686.25951612904,
    "INSTAL_AMT_PAYMENT_MIN": 337.815,
    "INSTAL_AMT_PAYMENT_SUM": 2760822.135,
    "INSTAL_DAYS_ENTRY_PAYMENT_MAX": -174,
    "INSTAL_DAYS_ENTRY_PAYMENT_SUM": -116122,
    "INSTAL_DBD_MAX": 47,
    "INSTAL_DBD_MEAN": 6.655913978494624,
    "INSTAL_DBD_SUM": 619,
    "INSTAL_DPD_MEAN": 2.3440860215053765,
    "INSTAL_PAYMENT_DIFF_MEAN": 3211.712903225807,
    "INSTAL_PAYMENT_DIFF_SUM": 298689.30000000005,
    "LIVE_CITY_NOT_WORK_CITY": 0,
    "LIVINGAREA_MODE": 0.1355,
    "NAME_EDUCATION_TYPE_ACADEMIC_DEGREE": 0,
    "NAME_EDUCATION_TYPE_SECONDARY__SECONDARY_SPECIAL": 1,
    "NAME_FAMILY_STATUS_MARRIED": 0,
    "NAME_HOUSING_TYPE_CO_OP_APARTMENT": 0,
    "NAME_HOUSING_TYPE_MUNICIPAL_APARTMENT": 0,
    "NAME_INCOME_TYPE_STUDENT": 0,
    "NAME_INCOME_TYPE_WORKING": 1,
    "NAME_TYPE_SUITE_OTHER_B": 0,
    "OCCUPATION_TYPE_DRIVERS": 0,
    "OCCUPATION_TYPE_LABORERS": 0,
    "OCCUPATION_TYPE_LOW_SKILL_LABORERS": 0,
    "SK_ID_CURR": 400464,
}


class TestApiRequests(unittest.TestCase):
    @patch("frontend.tests.test_api_requests.fetch_client_info")
    def test_fetch_client_info(self, mock_fetch):
        # Données simulées
        mock_client_info = {
            "SK_ID_CURR": 400464,
            "AMT_CREDIT": 343800,
            "PAYMENT_RATE": 0.0490183246073298,
            "NAME_INCOME_TYPE_WORKING": 1,
        }

        # Mock de la fonction
        mock_fetch.return_value = mock_client_info

        # Appel réel
        client_info = fetch_client_info(400464)

        print(
            f"Client info: {client_info}"
        )  # Ajoutez ceci pour voir ce qui est retourné
        # Assertions
        self.assertEqual(client_info["SK_ID_CURR"], 400464)
        self.assertEqual(client_info["AMT_CREDIT"], 343800)
        self.assertAlmostEqual(client_info["PAYMENT_RATE"], 0.0490183246073298)
        self.assertEqual(client_info["NAME_INCOME_TYPE_WORKING"], 1)

    @patch("frontend.tests.test_api_requests.fetch_client_info")
    def test_fetch_client_info_not_found(self, mock_fetch):
        # Simule un retour None ou erreur gérée
        mock_fetch.return_value = None
        client_info = fetch_client_info(999)
        self.assertIsNone(client_info)


# Test fonctionnel en-dehors de classe
@patch("frontend.utils.shap_utils.requests.get")
def test_fetch_local_shap_explanation(mock_get):
    # Simule la réponse JSON de l'API
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "values": [
            0.01305076156604987,
            0.07097157387902969,
            0.4857280408875004,
            0.020167035319394597,
        ],
        "base_value": 0.19183999932179563,
        "features": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_CREDIT"],
    }

    mock_get.return_value = mock_response

    result = fetch_local_shap_explanation(400464)

    assert result["values"][:4] == mock_response.json.return_value["values"]
    assert result["base_value"] == mock_response.json.return_value["base_value"]
    assert result["features"] == mock_response.json.return_value["features"]
    assert mock_get.call_count == 1


@patch("frontend.utils.shap_utils.requests.get")
def test_fetch_local_shap_explanation_not_found(mock_get):
    # Simuler une réponse avec erreur HTTP (404 ou 500)
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Not Found"
    )
    mock_get.return_value = mock_response

    result = fetch_local_shap_explanation(999999)

    assert result is None
    mock_get.assert_called_once()
