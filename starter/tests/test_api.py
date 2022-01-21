import imp
import requests
from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)


def test_get_hello():
    r = requests.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Hello World!"}


def test_predict_empty():
    r = requests.post("/predict")
    assert r.status_code == 422


def test_predict_salary_low():
    data_low = {
        "age": 50,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 83311,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 13,
        "native-country": "United-States"
    }
    r = requests.post("/predict", json=data_low)
    assert r.status_code == 200
    assert r.json() == "<=50K"


def test_predict_salary_high():
    data_high = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 45,
        "native-country": "United-States"
    }
    r = requests.post("/predict", json=data_high)
    assert r.status_code == 200
    assert r.json() == ">50K"
