import pytest
from fastapi.testclient import TestClient
from main import app
import os

client = TestClient(app)

# Note: These tests assume GeminiClassifier is mockable or skip if no API KEY
@pytest.fixture
def api_key_header():
    key = os.getenv("HEALTH_CLAIM_API_KEY", "test_key")
    return {"X-API-Key": key}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]

def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200

def test_classify_unauthorized():
    response = client.post("/v1/classify")
    assert response.status_code in (401, 403) # Depends on if a key is configured

def test_extract_unauthorized():
    response = client.post("/v1/extract")
    assert response.status_code in (401, 403)
