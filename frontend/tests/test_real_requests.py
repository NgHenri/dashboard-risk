# Tester fetch_client_info en simulant requests.get.
# Tester fetch_local_shap_explanation en simulant requests.get.

import pytest
from unittest.mock import patch, MagicMock
import requests

import frontend.utils.api_requests as api_requests
import frontend.utils.shap_utils as shap_utils

import streamlit as st


# Tests pour fetch_client_info
@patch("frontend.utils.api_requests.requests.get")
def test_fetch_client_info_success(mock_get):
    # Vider le cache Streamlit pour forcer l'appel HTTP
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # Simuler une réponse HTTP valide avec JSON
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "SK_ID_CURR": 400464,
        "AMT_CREDIT": 343800,
        "PAYMENT_RATE": 0.0490183246073298,
        "NAME_INCOME_TYPE_WORKING": 1,
    }
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    client_id = 400464
    client_info = api_requests.fetch_client_info(client_id)

    # Vérifier l'appel HTTP : URL et timeout
    expected_url = f"{api_requests.API_URL}/client_info/{client_id}"
    mock_get.assert_called_once_with(expected_url, timeout=api_requests.TIMEOUT)

    # Vérifier le contenu retourné
    assert isinstance(client_info, dict)
    assert client_info["SK_ID_CURR"] == 400464
    assert client_info["AMT_CREDIT"] == 343800


@patch("frontend.utils.api_requests.requests.get")
def test_fetch_client_info_http_error(mock_get):
    try:
        st.cache_data.clear()
    except Exception:
        pass
    # Simuler une erreur HTTP dans raise_for_status
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "404 Not Found"
    )
    mock_get.return_value = mock_resp

    client_id = 999999
    result = api_requests.fetch_client_info(client_id)
    # L'implémentation attrape l'exception et retourne None
    assert result is None


# Tests pour fetch_local_shap_explanation
@patch("frontend.utils.shap_utils.requests.get")
def test_fetch_local_shap_explanation_success(mock_get):
    # Simuler une réponse HTTP valide avec JSON
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "values": [0.01, 0.02, 0.03],
        "base_value": 0.1,
        "features": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    }
    mock_get.return_value = mock_resp

    client_id = 400464
    result = shap_utils.fetch_local_shap_explanation(client_id)

    # Vérifier qu'une requête a été effectuée
    assert mock_get.call_count == 1
    called_url = mock_get.call_args[0][0]
    assert called_url.startswith(api_requests.API_URL)
    assert str(client_id) in called_url

    # Vérifier le contenu retourné
    assert isinstance(result, dict)
    assert result["values"] == mock_resp.json.return_value["values"]
    assert result["base_value"] == mock_resp.json.return_value["base_value"]
    assert result["features"] == mock_resp.json.return_value["features"]


@patch("frontend.utils.shap_utils.requests.get")
def test_fetch_local_shap_explanation_http_error(mock_get):
    # Simuler une erreur HTTP dans raise_for_status
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Internal Server Error"
    )
    mock_get.return_value = mock_resp

    client_id = 999999
    result = shap_utils.fetch_local_shap_explanation(client_id)

    # Selon implémentation, on s'attend à None en cas d'erreur
    assert result is None


# Tests pour fetch_prediction
@patch("frontend.utils.api_requests.requests.post")
@patch("frontend.utils.api_requests.st.error")
def test_fetch_prediction_success(mock_st_error, mock_post):
    # Simuler une réponse HTTP valide avec JSON
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"prediction": 0.85, "status": "ok"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    client_data = {"SK_ID_CURR": 400464, "AMT_CREDIT": 343800}
    # Appeler la fonction en passant API_URL paramètre
    result = api_requests.fetch_prediction(api_requests.API_URL, client_data)

    # Vérifier qu'aucune erreur n'a été signalée
    mock_st_error.assert_not_called()

    # Vérifier appel HTTP : URL, json et timeout
    expected_url = f"{api_requests.API_URL}/predict"
    mock_post.assert_called_once_with(
        expected_url, json={"data": client_data}, timeout=api_requests.TIMEOUT
    )

    # Vérifier le contenu retourné
    assert isinstance(result, dict)
    assert result.get("prediction") == 0.85
    assert result.get("status") == "ok"


@patch("frontend.utils.api_requests.requests.post")
@patch("frontend.utils.api_requests.st.error")
def test_fetch_prediction_http_error(mock_st_error, mock_post):
    # Simuler une erreur HTTP dans raise_for_status
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError(
        "500 Internal Server Error"
    )
    mock_post.return_value = mock_resp

    client_data = {"SK_ID_CURR": 999999}
    result = api_requests.fetch_prediction(api_requests.API_URL, client_data)

    # L'implémentation attrape l'exception, signale l'erreur et retourne None
    mock_st_error.assert_called_once()
    assert result is None
