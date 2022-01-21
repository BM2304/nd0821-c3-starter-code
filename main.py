import os
import pickle
from pydantic import BaseModel
from fastapi import Body, FastAPI
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference

# Instatiate the app
app = FastAPI()


if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# load model, encoder and binarizer
pkl_model = './model/trained_model.pkl'
pkl_encoder = './model/binarizer.pkl'
pkl_lb = './model/lb.pkl'

with open(pkl_model, 'rb') as f:
    model = pickle.load(f)
with open(pkl_encoder, 'rb') as f:
    encoder = pickle.load(f)
with open(pkl_lb, 'rb') as f:
    lb = pickle.load(f)


class DataItem(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/predict/")
async def predict_salary_level(data: DataItem):
    data_dict = {
        "age": [data.age],
        "workclass": [data.workclass],
        "fnlgt": [data.fnlgt],
        "education": [data.education],
        "education-num": [data.education_num],
        "marital-status": [data.marital_status],
        "occupation": [data.occupation],
        "relationship": [data.relationship],
        "race": [data.race],
        "sex": [data.sex],
        "capital-gain": [data.capital_gain],
        "capital-loss": [data.capital_loss],
        "hours-per-week": [data.hours_per_week],
        "native-country": [data.native_country]
    }

    data = pd.DataFrame.from_dict(data_dict)

    X, _, _, _ = process_data(
        data, categorical_features=cat_features, training=False, encoder=encoder, lb=lb
    )

    preds = inference(model, X)
    preds = lb.inverse_transform(preds)
    return preds[0]
