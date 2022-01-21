import requests
import logging
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_hello():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_predict_empty():
    r = client.post("/predict/")
    assert r.status_code == 422


def test_predict_salary_low():
    data_low = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 83311,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 13,
        "native_country": "United-States"
    }
    r = client.post("/predict/", json=data_low)
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_predict_salary_high():
    data_high = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 45,
        "native_country": "United-States"
    }
    r = client.post("/predict/", json=data_high)
    assert r.status_code == 200
    assert r.json() == ">50K"
