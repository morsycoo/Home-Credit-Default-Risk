from conftest import client

def test_health_check():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Credit Risk Intelligence API is running."

def test_prediction():

    payload = {
        "features": {
            "EXT_SOURCE_3": 0.7463,
            "EXT_SOURCE_2": 0.6222,
            "EXT_SOURCE_1": 0.0830,
            "PAYMENT_RATE": 0.045,
            "CREDIT_TO_ANNUITY_RATIO": 22.0,
            "BUREAU_DAYS_CREDIT_MAX": -500,
            "POS_FUTURE_INSTALLMENTS_MEAN": 2.3,
            "ANNUITY_INCOME_RATIO": 0.18,
            "AMT_CREDIT": 406597.5,
            "AMT_ANNUITY": 24700.5,
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "risk_score" in data
    assert "threshold" in data

    assert data["prediction"] in [0, 1]

def test_invalid_request():

    response = client.post("/predict", json={})

    assert response.status_code == 422