from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_prediction():
    payload = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 29,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 211.3375,
        "Embarked": "C"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "survived" in data
    assert "survival_probability" in data

    assert data["survived"] in [0, 1]
    assert 0 <= data["survival_probability"] <= 1