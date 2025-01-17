import requests
import json

data = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 287927,
    "education": "HS-grad",
    "education_num": 9,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Wife",
    "race": "White",
    "sex": "Female",
    "capital_gain": 15024,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}
response = requests.post(
    'https://census-salary.herokuapp.com/predict/', data=json.dumps(data))

print(
    f"Live request - status code: {response.status_code}, result: {response.json()}")
