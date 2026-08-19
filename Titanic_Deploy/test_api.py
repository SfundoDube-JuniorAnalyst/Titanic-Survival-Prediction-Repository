"""
test_api.py — Verifies the Titanic Survival Prediction API works correctly.

Run the API first in a separate terminal:
    uvicorn app:app --reload --port 8000

Then run this test script:
    python test_api.py
"""
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_health_check():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    print("[PASS] Health check endpoint working")


def test_predict_first_class_woman():
    """A 1st-class woman traveling with family should have a high survival probability."""
    payload = {
        "Pclass": 1, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0,
        "Fare": 71.28, "Title": "Mrs", "CabinKnown": True, "TicketFreq": 2
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "prediction" in data and "survival_probability" in data
    assert data["prediction"] == 1
    assert data["survival_probability"] > 0.5
    print(f"[PASS] 1st-class woman -> {data['prediction_label']} "
          f"(probability={data['survival_probability']})")


def test_predict_third_class_man():
    """A 3rd-class man traveling alone should have a low survival probability."""
    payload = {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 0, "Parch": 0,
        "Fare": 7.25, "Title": "Mr", "CabinKnown": False, "TicketFreq": 1
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 0
    assert data["survival_probability"] < 0.5
    print(f"[PASS] 3rd-class man -> {data['prediction_label']} "
          f"(probability={data['survival_probability']})")


def test_invalid_input_rejected():
    """Pclass=5 is invalid (only 1/2/3 allowed) and should return HTTP 422."""
    payload = {
        "Pclass": 5, "Sex": "female", "Age": 38, "SibSp": 1, "Parch": 0,
        "Fare": 71.28, "Title": "Mrs", "CabinKnown": True
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 422
    print("[PASS] Invalid Pclass correctly rejected with HTTP 422")


def test_missing_required_field():
    """Missing a required field (Age) should return HTTP 422."""
    payload = {
        "Pclass": 1, "Sex": "female", "SibSp": 1, "Parch": 0,
        "Fare": 71.28, "Title": "Mrs", "CabinKnown": True
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 422
    print("[PASS] Missing required field correctly rejected with HTTP 422")


if __name__ == "__main__":
    print("Running Titanic API tests against", BASE_URL)
    print("-" * 50)
    test_health_check()
    test_predict_first_class_woman()
    test_predict_third_class_man()
    test_invalid_input_rejected()
    test_missing_required_field()
    print("-" * 50)
    print("All tests passed.")