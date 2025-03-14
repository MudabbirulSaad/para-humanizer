"""
Unit tests for the Para-Humanizer API endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import json
import time

from api.app import app
from api.models import ParaphraseRequest
from api.config import settings


# Create test client
client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "version" in data


def test_paraphrase_endpoint():
    """Test the paraphrase endpoint with valid data."""
    # Create a valid paraphrase request
    request_data = {
        "text": "This is a test sentence for paraphrasing.",
        "rule_based_rate": 0.5,
        "transformer_rate": 0.0,
        "humanize": True,
        "humanize_intensity": 0.5,
        "typo_rate": 0.0,
        "no_parallel": False
    }
    
    response = client.post("/v1/paraphrase", json=request_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "text" in data["choices"][0]
    assert data["choices"][0]["text"] != request_data["text"]
    assert "usage" in data
    assert "prompt_tokens" in data["usage"]
    assert "completion_tokens" in data["usage"]
    assert "total_tokens" in data["usage"]


def test_paraphrase_empty_text():
    """Test the paraphrase endpoint with empty text."""
    request_data = {
        "text": "",
        "rule_based_rate": 0.5,
        "transformer_rate": 0.0,
        "humanize": True,
        "humanize_intensity": 0.5,
        "typo_rate": 0.0
    }
    
    response = client.post("/v1/paraphrase", json=request_data)
    assert response.status_code == 422  # Validation error


def test_paraphrase_invalid_rates():
    """Test the paraphrase endpoint with invalid rates."""
    request_data = {
        "text": "This is a test sentence.",
        "rule_based_rate": 0.0,  # Invalid: both rates are 0
        "transformer_rate": 0.0,
        "humanize": True,
        "humanize_intensity": 0.5,
        "typo_rate": 0.0
    }
    
    response = client.post("/v1/paraphrase", json=request_data)
    assert response.status_code == 422  # Validation error
    
    # Test with rates out of range
    request_data = {
        "text": "This is a test sentence.",
        "rule_based_rate": 1.5,  # Invalid: > 1.0
        "transformer_rate": 0.0,
        "humanize": True,
        "humanize_intensity": 0.5,
        "typo_rate": 0.0
    }
    
    response = client.post("/v1/paraphrase", json=request_data)
    assert response.status_code == 422  # Validation error


def test_intelligent_parameter_selection():
    """Test that intelligent parameter selection is triggered with default values."""
    request_data = {
        "text": "This is a test sentence for intelligent parameter selection.",
        # Using default values to trigger intelligent selection
        "rule_based_rate": 0.4,
        "transformer_rate": 0.0,
        "humanize_intensity": 0.5,
        "typo_rate": 0.0
    }
    
    response = client.post("/v1/paraphrase", json=request_data)
    assert response.status_code == 200
    
    # The intelligent selection is internal, so we can only check that the response is valid
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "text" in data["choices"][0]
    assert data["choices"][0]["text"] != request_data["text"]


def test_api_key_authentication():
    """Test API key authentication if enabled."""
    # Skip test if authentication is not enabled
    if not settings.security.api_keys_enabled:
        pytest.skip("API key authentication is not enabled")
    
    request_data = {
        "text": "This is a test sentence.",
        "rule_based_rate": 0.5,
        "transformer_rate": 0.0
    }
    
    # Test without API key
    response = client.post("/v1/paraphrase", json=request_data)
    assert response.status_code == 401
    
    # Test with invalid API key
    headers = {"X-API-Key": "invalid_key"}
    response = client.post("/v1/paraphrase", json=request_data, headers=headers)
    assert response.status_code == 401
    
    # Test with valid API key
    if settings.security.api_keys:
        headers = {"X-API-Key": settings.security.api_keys[0]}
        response = client.post("/v1/paraphrase", json=request_data, headers=headers)
        assert response.status_code == 200
